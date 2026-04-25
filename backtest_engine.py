"""
backtest_engine.py — Backtest engine cho thị trường chứng khoán Việt Nam.

Flow:
  1. Load OHLCV từ vn_loader (Entrade/Fireant/TCBS)
  2. Đọc signal_engine.py từ run_dir/code/signal_engine.py
  3. Chạy signal_engine.generate_signals(df) → Series[+1/0/-1]
  4. Tính P&L, metrics (Sharpe, Max DD, Win Rate, ...)
  5. Vẽ equity curve + subplot signals → lưu chart.png

Compatible với backtest_tool.py (BacktestTool) — trả về cùng JSON format.

Usage:
    result = run_vn_backtest(run_dir="/path/to/run")
    # result: {"status": "ok", "metrics": {...}, "chart_path": "...", ...}
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import importlib.util
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Optional matplotlib ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # Headless backend cho server
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    logger.warning("matplotlib không có — chart sẽ bị bỏ qua")


# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_CAPITAL   = 100_000_000   # 100 triệu VND
DEFAULT_COMM_PCT  = 0.0015        # 0.15% phí giao dịch (HoSE standard)
DEFAULT_SLIPPAGE  = 0.001         # 0.1% slippage
RISK_FREE_RATE    = 0.045         # 4.5% lãi suất phi rủi ro (VN 2024)
TRADING_DAYS_YR   = 252


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Config reader
# ═══════════════════════════════════════════════════════════════════════════════

def _read_config(run_path: Path) -> dict:
    """Đọc và validate config.json."""
    config_path = run_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("config.json not found")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Defaults cho VN market
    config.setdefault("source",        "entrade")
    config.setdefault("days",          365)
    config.setdefault("initial_capital", DEFAULT_CAPITAL)
    config.setdefault("commission_pct",  DEFAULT_COMM_PCT)
    config.setdefault("slippage_pct",    DEFAULT_SLIPPAGE)
    config.setdefault("position_size",   1.0)   # 1.0 = 100% vốn mỗi lệnh
    config.setdefault("allow_short",     False)  # HOSE không short dễ dàng

    # Symbol bắt buộc
    if not config.get("symbol"):
        raise ValueError("config.json thiếu 'symbol'")

    return config


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Signal engine loader
# ═══════════════════════════════════════════════════════════════════════════════

def _load_signal_engine(run_path: Path):
    """
    Import signal_engine.py từ run_path/code/signal_engine.py.

    Module phải expose hàm:
        generate_signals(df: pd.DataFrame) -> pd.Series
        df columns: [date, open, high, low, close, volume]
        returns: Series index=df.index, values in {+1, 0, -1}
    """
    signal_path = run_path / "code" / "signal_engine.py"
    if not signal_path.exists():
        raise FileNotFoundError(f"code/signal_engine.py not found in {run_path}")

    spec   = importlib.util.spec_from_file_location("signal_engine", signal_path)
    module = importlib.util.module_from_spec(spec)
    # Thêm thư mục code vào sys.path để signal_engine có thể import helper
    code_dir = str(signal_path.parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec.loader.exec_module(module)

    if not hasattr(module, "generate_signals"):
        raise AttributeError(
            "signal_engine.py phải định nghĩa hàm: "
            "generate_signals(df: pd.DataFrame) -> pd.Series"
        )
    return module


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Backtest core
# ═══════════════════════════════════════════════════════════════════════════════

def _run_backtest_core(
    df: pd.DataFrame,
    signals: pd.Series,
    config: dict,
) -> dict:
    """
    Vectorised backtest đơn giản:
    - Signal +1 → MUA (next open)
    - Signal -1 → BÁN/đóng vị thế (next open)
    - Signal  0 → giữ nguyên

    Returns dict với equity_series, trades, metrics.
    """
    capital     = float(config["initial_capital"])
    comm_pct    = float(config["commission_pct"])
    slip_pct    = float(config["slippage_pct"])
    pos_size    = float(config.get("position_size", 1.0))
    allow_short = bool(config.get("allow_short", False))

    n       = len(df)
    equity  = np.zeros(n)
    equity[0] = capital

    cash       = capital
    position   = 0.0   # số tiền đang đầu tư (dương = long)
    shares     = 0
    entry_price = 0.0
    in_position = False
    trades     = []

    for i in range(1, n):
        sig    = int(signals.iloc[i - 1])  # tín hiệu từ ngày i-1
        o      = float(df["open"].iloc[i])  # thực hiện tại open ngày i
        exec_p = o * (1 + slip_pct * (1 if sig == 1 else -1))  # slippage

        # MUA
        if sig == 1 and not in_position:
            invest   = cash * pos_size
            cost_all = exec_p * (1 + comm_pct)
            shares   = invest / cost_all
            cash    -= shares * cost_all
            entry_price = exec_p
            in_position = True
            trades.append({
                "type":        "BUY",
                "date":        df["date"].iloc[i].strftime("%Y-%m-%d"),
                "price":       round(exec_p, 0),
                "shares":      round(shares, 4),
                "cash_before": round(cash + shares * cost_all, 0),
            })

        # BÁN
        elif sig == -1 and in_position:
            proceeds  = shares * exec_p * (1 - comm_pct)
            pnl       = proceeds - shares * entry_price * (1 + comm_pct)
            pnl_pct   = pnl / (shares * entry_price * (1 + comm_pct)) * 100
            cash     += proceeds
            trades.append({
                "type":      "SELL",
                "date":      df["date"].iloc[i].strftime("%Y-%m-%d"),
                "price":     round(exec_p, 0),
                "shares":    round(shares, 4),
                "pnl_vnd":   round(pnl, 0),
                "pnl_pct":   round(pnl_pct, 2),
            })
            in_position = False
            shares      = 0
            entry_price = 0.0

        # Short (nếu được phép)
        elif sig == -1 and allow_short and not in_position:
            pass  # TODO: short logic nếu cần

        # Mark-to-market equity
        mtm = shares * float(df["close"].iloc[i]) if in_position else 0
        equity[i] = cash + mtm

    # Đóng vị thế cuối nếu còn
    if in_position and shares > 0:
        last_close = float(df["close"].iloc[-1])
        proceeds   = shares * last_close * (1 - comm_pct)
        cash      += proceeds
        equity[-1] = cash

    return {
        "equity":  equity,
        "trades":  trades,
        "final_equity": equity[-1],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _calc_metrics(equity: np.ndarray, trades: list, config: dict) -> dict:
    """Tính các chỉ số hiệu suất chuẩn."""
    capital    = float(config["initial_capital"])
    final      = float(equity[-1])
    total_ret  = (final - capital) / capital * 100

    # Daily returns
    returns    = np.diff(equity) / equity[:-1]
    returns    = returns[np.isfinite(returns)]

    # Annualized return (CAGR)
    n_days     = len(equity)
    years      = n_days / TRADING_DAYS_YR
    cagr       = ((final / capital) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0

    # Sharpe ratio
    if len(returns) > 1 and np.std(returns) > 0:
        rf_daily   = RISK_FREE_RATE / TRADING_DAYS_YR
        sharpe     = (np.mean(returns) - rf_daily) / np.std(returns) * np.sqrt(TRADING_DAYS_YR)
    else:
        sharpe     = 0.0

    # Max Drawdown
    peak       = np.maximum.accumulate(equity)
    drawdown   = (equity - peak) / peak * 100
    max_dd     = float(np.min(drawdown))

    # Calmar ratio
    calmar     = cagr / abs(max_dd) if max_dd != 0 else 0

    # Volatility (annualized)
    vol_ann    = float(np.std(returns) * np.sqrt(TRADING_DAYS_YR) * 100) if len(returns) > 1 else 0

    # Win rate từ trades
    sells = [t for t in trades if t.get("type") == "SELL"]
    n_trades = len(sells)
    wins  = [t for t in sells if t.get("pnl_vnd", 0) > 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win  = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    loses    = [t for t in sells if t.get("pnl_vnd", 0) <= 0]
    avg_loss = np.mean([t["pnl_pct"] for t in loses]) if loses else 0
    profit_factor = (
        sum(t["pnl_vnd"] for t in wins) / abs(sum(t["pnl_vnd"] for t in loses))
        if loses and sum(t["pnl_vnd"] for t in loses) != 0 else 0
    )

    return {
        "initial_capital_vnd":  int(capital),
        "final_equity_vnd":     int(final),
        "total_return_pct":     round(total_ret, 2),
        "cagr_pct":             round(cagr, 2),
        "sharpe_ratio":         round(sharpe, 3),
        "max_drawdown_pct":     round(max_dd, 2),
        "calmar_ratio":         round(calmar, 3),
        "volatility_ann_pct":   round(vol_ann, 2),
        "n_trades":             n_trades,
        "win_rate_pct":         round(win_rate, 2),
        "avg_win_pct":          round(avg_win, 2),
        "avg_loss_pct":         round(avg_loss, 2),
        "profit_factor":        round(profit_factor, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Chart
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_chart(
    df: pd.DataFrame,
    signals: pd.Series,
    equity: np.ndarray,
    metrics: dict,
    config: dict,
    out_path: str,
) -> str:
    """
    Vẽ 3-panel chart:
      Panel 1: Candlestick price với buy/sell markers
      Panel 2: Volume bars
      Panel 3: Equity curve so với Buy & Hold

    Lưu ra PNG. Trả về đường dẫn.
    """
    if not _HAS_MPL:
        logger.warning("matplotlib không có — bỏ qua chart")
        return ""

    symbol  = config.get("symbol", "N/A")
    capital = float(config["initial_capital"])
    dates   = pd.to_datetime(df["date"])

    # ── Tính Buy & Hold ──────────────────────────────────────────────────────
    bh_equity = capital * df["close"] / df["close"].iloc[0]

    # ── Buy/Sell markers ─────────────────────────────────────────────────────
    buy_idx  = [i + 1 for i in range(len(signals) - 1) if signals.iloc[i] == 1]
    sell_idx = [i + 1 for i in range(len(signals) - 1) if signals.iloc[i] == -1]
    buy_idx  = [i for i in buy_idx if i < len(df)]
    sell_idx = [i for i in sell_idx if i < len(df)]

    # ── Layout ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1, 2]},
        facecolor="#0d1117",
    )
    fig.suptitle(
        f"Backtest: {symbol}  |  "
        f"Return: {metrics['total_return_pct']:+.1f}%  |  "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
        f"MaxDD: {metrics['max_drawdown_pct']:.1f}%  |  "
        f"Win: {metrics['win_rate_pct']:.0f}%",
        color="white", fontsize=12, fontweight="bold", y=0.98,
    )

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("#30363d")

    # ── Panel 1: Price ────────────────────────────────────────────────────────
    ax1 = axes[0]
    close = df["close"].values

    # Đường giá
    ax1.plot(dates, close, color="#58a6ff", linewidth=1.2, label="Close")

    # Markers mua/bán
    if buy_idx:
        ax1.scatter(dates.iloc[buy_idx], df["close"].iloc[buy_idx],
                    marker="^", color="#3fb950", s=80, zorder=5, label="MUA")
    if sell_idx:
        ax1.scatter(dates.iloc[sell_idx], df["close"].iloc[sell_idx],
                    marker="v", color="#f85149", s=80, zorder=5, label="BÁN")

    ax1.set_ylabel("Giá (VND)", color="gray", fontsize=9)
    ax1.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
               labelcolor="white", fontsize=8)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    # ── Panel 2: Volume ───────────────────────────────────────────────────────
    ax2 = axes[1]
    colors_vol = ["#3fb950" if df["close"].iloc[i] >= df["open"].iloc[i]
                  else "#f85149" for i in range(len(df))]
    ax2.bar(dates, df["volume"], color=colors_vol, alpha=0.7, width=0.8)
    ax2.set_ylabel("Volume", color="gray", fontsize=9)
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
    )

    # ── Panel 3: Equity Curve ─────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(dates, equity,    color="#58a6ff", linewidth=1.5,  label="Strategy")
    ax3.plot(dates, bh_equity, color="#8b949e", linewidth=1.0,
             linestyle="--", label="Buy & Hold")

    # Shade profit/loss vs B&H
    ax3.fill_between(dates, equity, bh_equity,
                     where=(equity >= bh_equity),
                     alpha=0.15, color="#3fb950", interpolate=True)
    ax3.fill_between(dates, equity, bh_equity,
                     where=(equity < bh_equity),
                     alpha=0.15, color="#f85149", interpolate=True)

    ax3.set_ylabel("Vốn (VND)", color="gray", fontsize=9)
    ax3.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d",
               labelcolor="white", fontsize=8)
    ax3.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M")
    )

    # Định dạng trục X chỉ panel cuối
    for ax in axes[:2]:
        ax.set_xticklabels([])
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.get_xticklabels(), rotation=30, ha="right", color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    logger.info(f"Chart saved: {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_vn_backtest(run_dir: str) -> dict:
    """
    Chạy toàn bộ pipeline backtest cho VN market.

    Args:
        run_dir: Thư mục chứa config.json và code/signal_engine.py

    Returns:
        dict tương thích với BacktestTool:
        {
            "status":     "ok" | "error",
            "metrics":    {...},
            "trades":     [...],
            "chart_path": "...",
            "artifacts":  {"chart": "...", "report": "..."},
            "run_dir":    "...",
            "stdout":     "...",
            "stderr":     "",
        }
    """
    t0       = time.time()
    run_path = Path(run_dir)
    stdout_lines = []

    def log(msg: str):
        stdout_lines.append(msg)
        logger.info(msg)

    try:
        # ── 1. Config ─────────────────────────────────────────────────────────
        config = _read_config(run_path)
        symbol = config["symbol"].upper()
        days   = int(config.get("days", 365))
        log(f"Config OK: symbol={symbol}, days={days}, capital={config['initial_capital']:,}")

        # ── 2. Load data ──────────────────────────────────────────────────────
        log(f"Loading OHLCV for {symbol}...")
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=days)
        log(f"Data OK: {len(df)} bars ({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")

        # ── 3. Load & run signal engine ───────────────────────────────────────
        log("Loading signal_engine.py...")
        engine  = _load_signal_engine(run_path)
        log("Generating signals...")
        signals = engine.generate_signals(df.copy())

        if not isinstance(signals, pd.Series):
            raise TypeError("generate_signals() phải trả về pd.Series")
        signals = signals.reindex(df.index).fillna(0).astype(int)
        n_buys  = (signals == 1).sum()
        n_sells = (signals == -1).sum()
        log(f"Signals OK: {n_buys} BUY, {n_sells} SELL")

        # ── 4. Backtest ───────────────────────────────────────────────────────
        log("Running backtest...")
        bt = _run_backtest_core(df, signals, config)
        equity = bt["equity"]
        trades = bt["trades"]
        log(f"Backtest OK: {len(trades)//2} trades, final={bt['final_equity']:,.0f} VND")

        # ── 5. Metrics ────────────────────────────────────────────────────────
        metrics = _calc_metrics(equity, trades, config)
        log(f"Metrics: Return={metrics['total_return_pct']:+.2f}% | "
            f"Sharpe={metrics['sharpe_ratio']:.2f} | "
            f"MaxDD={metrics['max_drawdown_pct']:.2f}% | "
            f"WinRate={metrics['win_rate_pct']:.1f}%")

        # ── 6. Chart ──────────────────────────────────────────────────────────
        artifacts_dir = run_path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        chart_path = str(artifacts_dir / "equity_curve.png")

        try:
            _draw_chart(df, signals, equity, metrics, config, chart_path)
            log(f"Chart saved: {chart_path}")
        except Exception as ce:
            log(f"Chart warning: {ce}")
            chart_path = ""

        # ── 7. JSON report ────────────────────────────────────────────────────
        report = {
            "symbol":     symbol,
            "period":     f"{df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}",
            "bars":       len(df),
            "config":     config,
            "metrics":    metrics,
            "trades":     trades[-20:],      # 20 trades gần nhất
            "n_trades_total": len([t for t in trades if t["type"] == "SELL"]),
        }
        report_path = str(artifacts_dir / "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        elapsed = round(time.time() - t0, 1)
        log(f"Done in {elapsed}s")

        return {
            "status":     "ok",
            "exit_code":  0,
            "symbol":     symbol,
            "metrics":    metrics,
            "trades":     trades[-20:],
            "n_trades":   len([t for t in trades if t["type"] == "SELL"]),
            "chart_path": chart_path,
            "artifacts":  {
                "chart":  chart_path,
                "report": report_path,
            },
            "run_dir":   run_dir,
            "stdout":    "\n".join(stdout_lines),
            "stderr":    "",
            "elapsed_s": elapsed,
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"run_vn_backtest error: {e}\n{tb}")
        return {
            "status":    "error",
            "exit_code": 1,
            "error":     str(e),
            "stdout":    "\n".join(stdout_lines),
            "stderr":    tb[-1000:],
            "run_dir":   run_dir,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python backtest_engine.py <run_dir>")
        _sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    result = run_vn_backtest(_sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2))
