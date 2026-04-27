"""
backtest_rule_cmd.py — /backtest_rule command handler cho bot.py.

Import vào bot.py:
    from backtest_rule_cmd import backtest_rule_cmd, BACKTEST_RULE_COOLDOWN, _last_backtest_rule

Đăng ký trong main():
    app.add_handler(CommandHandler("backtest_rule", backtest_rule_cmd))
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import shutil
import tempfile
import time
import json as _json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BACKTEST_RULE_COOLDOWN = 120   # 2 phút per user
_last_backtest_rule: dict[str, float] = {}

# ── Helpers tái sử dụng từ bot.py (import lúc runtime để tránh circular) ──────
def _plain(text: str) -> str:
    import re
    text = re.sub(r"[*`]", "", text)
    return text.replace("\\_", "_")


# ══════════════════════════════════════════════════════════════════════════════
# PARSE COMMAND ARGS
# ══════════════════════════════════════════════════════════════════════════════

_HELP_TEXT = """\
BACKTEST_RULE — Kiem tra rule tu hoi dong phan tich
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cu phap:
  /backtest_rule <MA> "<entry_rule>" "<exit_rule>" [ngay]

INDICATORS:
  close, open, high, low, volume
  sma20, sma50, ema12, ema26      (so bat ki: sma(N))
  rsi, rsi(N)                     (mac dinh N=14)
  macd, macd_signal, macd_hist
  bb_upper, bb_lower, bb_mid
  atr, stoch_k, stoch_d
  high20, low20                   (rolling N bar: highN, lowN)
  volume_sma20                    (volume MA)

FUNCTIONS:
  crossover(a, b)   — a cat len b
  crossunder(a, b)  — a cat xuong b
  rising(ind, N)    — tang N bar lien tiep
  falling(ind, N)   — giam N bar lien tiep
  breakout(high,N)  — close > high N bar truoc
  prev(ind, N)      — gia tri N bar truoc

EXIT ONLY:
  trailing_stop(5%) — trailing stop 5%
  take_profit(10%)  — chot loi 10%
  stop_loss(5%)     — cat lo 5%
  hold(20)          — giu toi da 20 bar

PHEP TINH: >, <, >=, <=, ==, and, or, not, +, -, *, /

VI DU:
  /backtest_rule VCB "rsi < 30" "rsi > 70"
  /backtest_rule HPG "rsi < 30 and close > sma20" "rsi > 70 or trailing_stop(5%)"
  /backtest_rule STB "crossover(ema12, ema26)" "crossunder(ema12, ema26)"
  /backtest_rule FPT "close > high20 and volume > volume_sma20 * 1.5" "take_profit(15%) or stop_loss(7%)"
  /backtest_rule VCB "close < 62100 and rsi < 30" "rsi > 70 or close > 68000" 500
"""

_EXAMPLES_SHORT = (
    "Vi du:\n"
    "  /backtest_rule VCB \"rsi < 30\" \"rsi > 70\"\n"
    "  /backtest_rule HPG \"rsi < 30 and close > sma20\" \"rsi > 70 or trailing_stop(5%)\"\n"
    "  /backtest_rule STB \"crossover(ema12, ema26)\" \"crossunder(ema12, ema26)\"\n"
    "  /backtest_rule FPT \"close < 62100 and rsi < 30\" \"close > 68000\""
)


def _parse_args(args: list[str]) -> tuple[str, str, str, int] | str:
    """
    Parse context.args → (symbol, entry_rule, exit_rule, days) hoặc error string.

    Hỗ trợ 2 cách nhập:
      1. Đã có quotes từ Telegram: args = ["VCB", "rsi < 30", "rsi > 70", "365"]
      2. Không có quotes (user gõ cả chuỗi): ghép lại và split bằng dấu "
    """
    if not args:
        return "error:empty"

    symbol = args[0].upper().strip()
    if not symbol.isalnum() or len(symbol) > 10:
        return f"error:symbol:{symbol}"

    if len(args) < 3:
        return "error:notenough"

    # Telegram thường tách theo khoảng trắng, quotes nằm trong một arg
    # → ghép args[1:] lại và tìm các chuỗi trong ""
    joined = " ".join(args[1:])

    # Tìm các chuỗi trong dấu nháy kép
    import re
    quoted = re.findall(r'"([^"]*)"', joined)

    if len(quoted) >= 2:
        entry_rule = quoted[0].strip()
        exit_rule  = quoted[1].strip()
        # days là phần còn lại ngoài quotes
        remainder = re.sub(r'"[^"]*"', '', joined).strip()
        try:
            days = max(100, min(int(remainder), 1500)) if remainder else 365
        except ValueError:
            days = 365
    else:
        # Không có quotes — args[1] = entry, args[2] = exit, args[3] = days
        entry_rule = args[1].strip()
        exit_rule  = args[2].strip()
        try:
            days = max(100, min(int(args[3]), 1500)) if len(args) > 3 else 365
        except (ValueError, IndexError):
            days = 365

    if not entry_rule:
        return "error:empty_entry"
    if not exit_rule:
        return "error:empty_exit"

    return symbol, entry_rule, exit_rule, days


# ══════════════════════════════════════════════════════════════════════════════
# RESULT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════


def _calc_trade_analytics(trades: list, df) -> dict:
    """
    Hold Duration Stats và MAE/MFE từ danh sách trades + OHLCV df.
    Hold = số bars giữa BUY→SELL.
    MAE  = min(low trong window) / entry - 1
    MFE  = max(high trong window) / entry - 1
    MFE capture rate = actual_return / MFE × 100
    """
    import numpy as np
    sells = [t for t in trades if t.get("type") == "SELL"]
    buys  = [t for t in trades if t.get("type") == "BUY"]
    if not sells or df is None or len(df) < 2:
        return {}
    try:
        df_r = df.reset_index(drop=True)
        date_col = "date" if "date" in df_r.columns else df_r.columns[0]
        d2i = {str(row[date_col])[:10]: i for i, row in df_r.iterrows()}
    except Exception:
        return {}

    holds, maes, mfes, caps = [], [], [], []
    buy_idx = 0
    for sell in sells:
        sell_date = str(sell.get("date", ""))[:10]
        entry = None
        for b in buys[buy_idx:]:
            if str(b.get("date", ""))[:10] < sell_date:
                entry = b; buy_idx += 1
            else:
                break
        if entry is None:
            continue
        ep = float(entry.get("price", 0))
        xp = float(sell.get("price", 0))
        if ep <= 0:
            continue
        ed = str(entry.get("date", ""))[:10]
        i0 = d2i.get(ed); i1 = d2i.get(sell_date)
        if i0 is None or i1 is None or i1 <= i0:
            continue
        holds.append(i1 - i0)
        win = df_r.iloc[i0:i1+1]
        if "low" in win.columns and "high" in win.columns:
            lows  = win["low"].values.astype(float)
            highs = win["high"].values.astype(float)
            mae = (min(lows)  - ep) / ep * 100
            mfe = (max(highs) - ep) / ep * 100
            maes.append(round(mae, 2))
            mfes.append(round(mfe, 2))
            if mfe > 0:
                caps.append(round((xp - ep) / ep * 100 / mfe * 100, 1))

    result = {}
    if holds:
        result.update({"hold_avg": round(float(np.mean(holds)), 1),
                       "hold_median": round(float(np.median(holds)), 1),
                       "hold_max": int(max(holds)), "hold_min": int(min(holds))})
    if maes:
        result.update({"mae_avg": round(float(np.mean(maes)), 2),
                       "mae_median": round(float(np.median(maes)), 2),
                       "mae_worst": round(float(min(maes)), 2)})
    if mfes:
        result.update({"mfe_avg": round(float(np.mean(mfes)), 2),
                       "mfe_median": round(float(np.median(mfes)), 2),
                       "mfe_best": round(float(max(mfes)), 2)})
    if caps:
        result["mfe_capture_rate"] = round(float(np.mean(caps)), 1)
    return result

def _format_rule_result(
    metrics: dict,
    symbol: str,
    entry_rule: str,
    exit_rule: str,
    days: int,
    n_trades: int,
    exit_ec_entry,
    exit_ec_exit,
    trades_sample: list,
    n_buy_signals: int,
    n_sell_signals: int,
    bars: int,
    trade_analytics: dict = None,
) -> str:
    """Tạo text output đầy đủ cho /backtest_rule."""
    ret   = metrics.get("total_return_pct", 0)
    cagr  = metrics.get("cagr_pct", 0)
    sh    = metrics.get("sharpe_ratio", 0)
    mdd   = metrics.get("max_drawdown_pct", 0)
    wr    = metrics.get("win_rate_pct", 0)
    pf    = metrics.get("profit_factor", 0)
    vol   = metrics.get("volatility_ann_pct", 0)
    cal   = metrics.get("calmar_ratio", 0)
    cap   = metrics.get("initial_capital_vnd", 100_000_000)
    fin   = metrics.get("final_equity_vnd", cap)
    aw    = metrics.get("avg_win_pct", 0)
    al    = metrics.get("avg_loss_pct", 0)
    pnl   = fin - cap

    grade = "A" if sh > 1.5 and mdd > -20 else \
            "B" if sh > 0.8 else \
            "C" if sh > 0 else "D"

    em_r  = "🟢" if ret > 0 else "🔴"

    # Dynamic exit summary
    dyn_parts = []
    _ts_pct = getattr(exit_ec_exit, "trailing_stop_pct", None)
    _tp_pct = getattr(exit_ec_exit, "take_profit_pct",  None)
    _sl_pct = getattr(exit_ec_exit, "stop_loss_pct",    None)
    _hb     = getattr(exit_ec_exit, "hold_bars",        None)
    if _ts_pct: dyn_parts.append(f"TrailingStop {_ts_pct:.1f}%")
    if _tp_pct: dyn_parts.append(f"TakeProfit {_tp_pct:.1f}%")
    if _sl_pct: dyn_parts.append(f"StopLoss {_sl_pct:.1f}%")
    if _hb:     dyn_parts.append(f"MaxHold {_hb}bar")
    dyn_str = " + ".join(dyn_parts) if dyn_parts else "Khong"

    lines = [
        f"BACKTEST RULE: {symbol} | {days}D ({bars} bar)",
        "═" * 40,
        f"ENTRY: {entry_rule[:60]}",
        f"EXIT : {exit_rule[:60]}",
        f"Dynamic exit: {dyn_str}",
        "─" * 40,
        f"Xep hang    : [{grade}]",
        "",
        "HIEU SUAT:",
        f"  Loi nhuan  : {em_r} {ret:+.2f}%  ({pnl:+,.0f} VND)",
        f"  CAGR/nam   : {cagr:+.2f}%",
        f"  Sharpe     : {sh:.3f}",
        f"  Calmar     : {cal:.3f}",
        f"  Vol/nam    : {vol:.1f}%",
        "",
        "RUI RO:",
        f"  Max DD     : {mdd:.2f}%",
        "",
        "LENH GIAO DICH:",
        f"  Tin hieu   : {n_buy_signals} MUA / {n_sell_signals} BAN",
        f"  Thuc hien  : {n_trades} lenh",
        f"  Win Rate   : {wr:.1f}%",
        f"  TB thang   : {aw:+.2f}%",
        f"  TB thua    : {al:+.2f}%",
        f"  ProfitFac  : {pf:.2f}",
        "",
        f"Von ban dau : {cap:,.0f} VND",
        f"Von cuoi    : {fin:,.0f} VND",
        "═" * 40,
    ]

    # Exit reasons breakdown (nếu có dynamic exit)
    if dyn_parts and trades_sample:
        sells = [t for t in trades_sample if t.get("type") == "SELL"]
        reasons: dict[str, int] = {}
        for t in sells:
            r = t.get("exit_by", "Signal")
            reasons[r] = reasons.get(r, 0) + 1
        if reasons:
            lines.append("LY DO THOAT LENH:")
            for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {r}: {cnt} lan")
            lines.append("─" * 40)

    # ── Hold Duration & MAE/MFE ─────────────────────────────────
    if trade_analytics:
        ta = trade_analytics
        lines.append("")
        lines.append("THOI GIAN NAM GIU (bars):")
        if "hold_avg" in ta:
            lines.append(
                f"  TB:{ta['hold_avg']:.1f} | Median:{ta['hold_median']:.1f}"
                f" | Min:{ta['hold_min']} | Max:{ta['hold_max']}"
            )
        if "mae_avg" in ta:
            lines.append("")
            lines.append("MAE / MFE (excursion toi da):")
            lines.append(f"  MAE TB: {ta['mae_avg']:+.2f}%"
                         f"  (worst: {ta['mae_worst']:+.2f}%)")
            lines.append(f"  MFE TB: {ta['mfe_avg']:+.2f}%"
                         f"  (best:  {ta['mfe_best']:+.2f}%)")
        if "mfe_capture_rate" in ta:
            cr = ta["mfe_capture_rate"]
            note = " ⚠️ Exit qua som" if cr < 40 else " ✅ Hieu qua" if cr > 80 else ""
            lines.append(f"  MFE thu duoc: {cr:.1f}%{note}")
        mae_a = ta.get("mae_avg", 0); mfe_a = ta.get("mfe_avg", 0)
        if mae_a < 0 and mfe_a > 0:
            lines.append(f"  MFE/MAE ratio: {abs(mfe_a/mae_a):.2f}x (ly tuong > 2.0x)")
        lines.append("─" * 40)

    lines.append("Bieu do equity curve da duoc gui kem.")
    return "\n".join(lines)


def _format_trades_msg(trades: list) -> str:
    """Format 5 lệnh gần nhất."""
    sells = [t for t in trades if t.get("type") == "SELL"][-5:]
    if not sells:
        return ""
    lines = ["5 LENH BAN GAN NHAT:"]
    for t in sells:
        em  = "✅" if t.get("pnl_vnd", 0) > 0 else "❌"
        by  = f" [{t['exit_by']}]" if t.get("exit_by") else ""
        lines.append(
            f"{em} {t['date']} | Gia: {t['price']:,.0f}{by}\n"
            f"   PnL: {t.get('pnl_pct', 0):+.2f}%"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CORE RUNNER (sync, chạy trong thread)
# ══════════════════════════════════════════════════════════════════════════════

def _run_rule_backtest_sync(
    symbol: str,
    entry_rule: str,
    exit_rule: str,
    days: int,
) -> dict:
    """
    Pipeline hoàn chỉnh: load data → parse rules → run backtest → chart.
    Chạy trong asyncio.to_thread.
    """
    try:
        from vn_loader import load_vn_ohlcv
        from rule_engine import (
            generate_rule_signals, apply_dynamic_exit, parse_rule,
            format_rule_explanation,
        )
        from backtest_engine import _run_backtest_core, _calc_metrics, _draw_chart
    except ImportError as _ie:
        return {"status": "error", "error": f"Thieu module: {_ie}"}

    config = {
        "symbol":           symbol,
        "initial_capital":  100_000_000,
        "commission_pct":   0.0015,
        "slippage_pct":     0.001,
        "position_size":    1.0,
        "allow_short":      False,
        "days":             days,
    }

    # 1. Load data — thử vn_loader trước, fallback sang analyzer.get_price_data
    df = load_vn_ohlcv(symbol, days=days)
    if df is None or (hasattr(df, '__len__') and len(df) < 30):
        try:
            from analyzer import get_price_data
            result = get_price_data(symbol, days=min(days, 500))
            if isinstance(result, dict) and result.get("success"):
                df = result["df"]
                logger.info(f"backtest: fallback sang analyzer.get_price_data cho {symbol}")
        except Exception as _fe:
            logger.warning(f"backtest fallback get_price_data fail: {_fe}")
    if df is None or len(df) < 30:
        return {
            "status": "error",
            "error":  f"Khong du du lieu cho {symbol} (can toi thieu 30 bars). "
                      f"Thu /debug {symbol} de kiem tra data sources."
        }

    # 2. Parse & compile rules → signals
    signals, entry_ec, exit_ec, ctx = generate_rule_signals(df, entry_rule, exit_rule)

    n_buy  = int((signals == 1).sum())
    n_sell = int((signals == -1).sum())

    if n_buy == 0:
        return {
            "status": "warn_no_signals",
            "message": (
                f"Rule ENTRY '{entry_rule}' khong tao duoc tin hieu nao "
                f"trong {days} ngay du lieu.\n"
                f"Thu no long rules (vd: rsi < 35 thay vi rsi < 30), "
                f"hoac tang so ngay."
            ),
        }

    # 3. Backtest — dùng dynamic exit nếu có trailing/tp/sl/hold
    if exit_ec.has_dynamic():
        bt = apply_dynamic_exit(df, signals, exit_ec, config)
    else:
        bt = _run_backtest_core(df, signals, config)

    equity = bt["equity"]
    trades = bt["trades"]

    # 4. Metrics
    metrics = _calc_metrics(equity, trades, config)

    # 5. Chart
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        chart_tmp = tf.name

    try:
        _draw_chart(df, signals, equity, metrics, config, chart_tmp)
        chart_ok = pathlib.Path(chart_tmp).exists()
    except Exception as ce:
        logger.warning(f"Chart warning: {ce}")
        chart_ok = False
        chart_tmp = ""

    return {
        "status":    "ok",
        "metrics":   metrics,
        "trades":    trades[-30:],
        "n_trades":  len([t for t in trades if t.get("type") == "SELL"]),
        "n_buy":     n_buy,
        "n_sell":    n_sell,
        "bars":      len(df),
        "chart_path": chart_tmp if chart_ok else "",
        "entry_ec":   entry_ec,
        "exit_ec":    exit_ec,
        "df":         df,
    }



# ══════════════════════════════════════════════════════════════════════════════
# AUTO CONTEXT HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def _handle_auto_context(update, context, symbol: str, plain_fn):
    """
    /backtest_rule VCB (không có rule):
      1. Load context từ DB → ưu tiên vibe (trade_plan) trong 7 ngày.
      2a. Có trade_plan → convert rule → backtest.
      2b. Chỉ có state_vector → analog search.
      3. Không có gì → hướng dẫn.
    """
    try:
        from auto_context import (
            load_auto_context, trade_plan_to_rules,
            format_trade_plan_summary, format_no_context_msg,
        )
        from historical_analog import (
            cache_exists, find_similar, format_analog_report, build_vector_cache,
        )
    except ImportError as _ie:
        await update.message.reply_text(
            f"Thieu module: {_ie}\n"
            f"De dung rule thu cong: /backtest_rule {symbol} \"rsi < 30\" \"rsi > 70\""
        )
        return

    NL      = "\n"
    chat_id = update.effective_chat.id
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"

    # Rate limit
    since = time.time() - _last_backtest_rule.get(user_id, 0)
    if since < BACKTEST_RULE_COOLDOWN:
        wait = int(BACKTEST_RULE_COOLDOWN - since)
        await update.message.reply_text(
            f"Vui long cho {wait}s truoc khi /backtest_rule tiep."
        )
        return
    _last_backtest_rule[user_id] = time.time()

    msg = await update.message.reply_text(
        f"Auto Context: dang tim boi canh phan tich cho {symbol}..."
    )

    try:
        # ── Bước 1: Load context từ DB ────────────────────────────────────
        ctx = await asyncio.to_thread(load_auto_context, symbol)

        if not ctx["found"]:
            await msg.edit_text(plain_fn(format_no_context_msg(symbol))[:4096])
            return

        source     = ctx["source"]
        trade_plan = ctx["trade_plan"]
        state_vec  = ctx["state_vector"]
        verdict    = ctx["verdict"] or "N/A"
        created    = ctx["created_at"]
        age_str    = ""
        if created:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_hours = (now - created).total_seconds() / 3600
            age_str   = f"{age_hours:.0f} gio truoc"

        # ── Bước 2a: Có trade_plan → backtest ────────────────────────────
        if trade_plan and trade_plan.get("entry"):
            entry_rule, exit_rule = trade_plan_to_rules(trade_plan)
            plan_summary          = format_trade_plan_summary(trade_plan)

            loading_lines = [
                f"Tim thay Trade Plan tu /vibe ({age_str}).",
                f"Verdict: {verdict}",
                "",
                plan_summary,
                "",
                "Dang chuyen thanh rule va chay backtest 365 ngay...",
            ]
            await msg.edit_text(plain_fn(NL.join(loading_lines))[:4096])

            result = await asyncio.to_thread(
                _run_rule_backtest_sync, symbol, entry_rule, exit_rule, 365
            )

            header_lines = [
                f"AUTO BACKTEST: {symbol}",
                f"Nguon: /vibe ({age_str}) | Verdict: {verdict}",
                plan_summary,
                f"Rule entry : {entry_rule[:70]}",
                f"Rule exit  : {exit_rule[:70]}",
                "=" * 40,
            ]
            header = NL.join(header_lines)

            if result["status"] == "warn_no_signals":
                warn_lines = [
                    header,
                    "",
                    "Canh bao: Rule khong tao duoc tin hieu nao trong 365 ngay.",
                    "Co the gia hien tai da vuot entry price.",
                    f"Thu /backtest_rule {symbol} voi rule thu cong.",
                ]
                await msg.edit_text(plain_fn(NL.join(warn_lines))[:4096])
                return

            if result["status"] == "error":
                err_lines = [header, "", f"Loi: {result.get('error','?')[:200]}"]
                await msg.edit_text(plain_fn(NL.join(err_lines))[:4096])
                return

            # Format kết quả
            metrics  = result["metrics"]
            n_trades = result["n_trades"]
            _ta_auto = _calc_trade_analytics(result["trades"], result.get("df"))
            bt_summary = _format_rule_result(
                metrics=metrics, symbol=symbol,
                entry_rule=entry_rule, exit_rule=exit_rule,
                days=365, n_trades=n_trades,
                exit_ec_entry=result["entry_ec"], exit_ec_exit=result["exit_ec"],
                trades_sample=result["trades"],
                n_buy_signals=result["n_buy"], n_sell_signals=result["n_sell"],
                bars=result["bars"],
                trade_analytics=_ta_auto,
            )
            full_text = plain_fn(header + NL + bt_summary)
            await msg.edit_text(full_text[:4096])

            # Chart
            chart_path = result.get("chart_path", "")
            if chart_path and pathlib.Path(chart_path).exists():
                cap_lines = [
                    f"Auto Backtest: {symbol} | Vibe Trade Plan",
                    f"Return: {metrics.get('total_return_pct',0):+.1f}% | Sharpe: {metrics.get('sharpe_ratio',0):.2f}",
                ]
                try:
                    with open(chart_path, "rb") as f:
                        await context.bot.send_photo(
                            chat_id=chat_id, photo=f,
                            caption=plain_fn(NL.join(cap_lines))[:1024],
                        )
                finally:
                    pathlib.Path(chart_path).unlink(missing_ok=True)

            # Trades
            trade_msg = _format_trades_msg(result["trades"])
            if trade_msg:
                await context.bot.send_message(
                    chat_id=chat_id, text=plain_fn(trade_msg)[:4096]
                )

        # ── Bước 2b: Chỉ có state_vector → analog search ─────────────────
        elif state_vec:
            loading_lines2 = [
                f"Tim thay boi canh tu /check ({age_str}).",
                f"Verdict: {verdict}",
                "Dang tim ngay lich su tuong dong...",
            ]
            await msg.edit_text(plain_fn(NL.join(loading_lines2))[:4096])

            cache_ok = await asyncio.to_thread(cache_exists, symbol)
            if not cache_ok:
                await msg.edit_text(
                    plain_fn(
                        f"Dang tao cache vector cho {symbol}..."
                        + NL +
                        "Vui long doi ~20-30 giay (chi can lam 1 lan)."
                    )[:4096]
                )
                ok, build_msg = await asyncio.to_thread(build_vector_cache, symbol)
                if not ok:
                    fail_lines = [
                        f"Khong tao duoc cache: {build_msg[:200]}",
                        f"Thu /backtest_rule {symbol} voi rule thu cong.",
                    ]
                    await msg.edit_text(plain_fn(NL.join(fail_lines))[:4096])
                    return

            analogs = await asyncio.to_thread(
                find_similar, symbol, state_vec, top_n=3, years=5
            )

            if not analogs:
                no_analog_lines = [
                    f"Khong tim thay ngay tuong dong cho {symbol}.",
                    "Co the cache chua du du lieu (can >= 6 thang).",
                    f"Thu /backtest_rule {symbol} voi rule thu cong.",
                ]
                await msg.edit_text(plain_fn(NL.join(no_analog_lines))[:4096])
                return

            from historical_analog import format_analog_report
            # Lay gia hien tai thuc te — KHONG dung analog["close"] vi do la gia lich su
            _current_price = 0.0
            try:
                from vn_loader import load_vn_ohlcv
                _df_price = load_vn_ohlcv(symbol, days=5, min_bars=1)
                if _df_price is not None and len(_df_price) > 0:
                    # close tu vn_loader don vi NGHIN DONG → nhan 1000 ra dong
                    _current_price = float(_df_price["close"].iloc[-1]) * 1000
            except Exception as _pe:
                logger.warning(f"backtest_rule: lay gia hien tai {symbol} fail: {_pe}")
            report       = format_analog_report(symbol, analogs, state_vec,
                                                current_price=_current_price)
            header_lines = [
                f"PHAN TICH TUONG DONG: {symbol}",
                f"Nguon: /check ({age_str}) | Verdict: {verdict}",
                "=" * 40,
            ]
            full_report = NL.join(header_lines) + NL + report
            await msg.edit_text(plain_fn(full_report)[:4096])

        else:
            no_info_lines = [
                f"Boi canh {symbol} ({age_str}) khong co du thong tin.",
                f"Hay chay lai /check {symbol} hoac /vibe {symbol}.",
            ]
            await msg.edit_text(plain_fn(NL.join(no_info_lines))[:4096])

    except Exception as e:
        import traceback
        logger.error(f"_handle_auto_context error: {e}\n{traceback.format_exc()}")
        try:
            await msg.edit_text(
                plain_fn(f"Loi Auto Context {symbol}: {str(e)[:200]}")[:4096]
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def backtest_rule_cmd(update, context):
    """
    /backtest_rule <MA> ["<entry_rule>" "<exit_rule>"] [ngay]

    CHE DO AUTO CONTEXT (khong co rule):
      /backtest_rule VCB  → tu dong lay context tu /vibe hoac /check gan nhat

    CHE DO MANUAL (co rule):
      /backtest_rule VCB "rsi < 30" "rsi > 70"
      /backtest_rule HPG "rsi < 30 and close > sma20" "rsi > 70 or trailing_stop(5%)"
    """
    from telegram import Update
    from telegram.ext import ContextTypes

    try:
        from bot import is_allowed, _deny
        plain = _plain   # dùng local _plain thay vì import từ bot (tránh circular)
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass
        plain = _plain

    if not is_allowed(update):
        await _deny(update); return

    args = context.args or []

    if not args:
        await update.message.reply_text(_HELP_TEXT[:4096])
        return

    if args[0].lower() in ("help", "--help", "-h", "?"):
        await update.message.reply_text(_HELP_TEXT[:4096])
        return

    # Validate symbol
    symbol_raw = args[0].upper().strip()
    if not symbol_raw.isalnum() or len(symbol_raw) > 10:
        await update.message.reply_text(f"Ma '{symbol_raw}' khong hop le (2-10 chu/so)."); return
    symbol = symbol_raw

    # AUTO CONTEXT khi chỉ có 1 arg (symbol)
    if len(args) == 1:
        await _handle_auto_context(update, context, symbol, plain)
        return

    # MANUAL: parse đầy đủ
    parsed = _parse_args(args)
    if isinstance(parsed, str):
        if parsed == "error:empty":
            await update.message.reply_text(_HELP_TEXT[:4096]); return
        if parsed.startswith("error:symbol:"):
            sym = parsed.split(":")[-1]
            await update.message.reply_text(f"Ma '{sym}' khong hop le (2-10 chu/so)."); return
        if parsed == "error:notenough":
            await update.message.reply_text(
                "Thieu tham so. Can ca entry_rule va exit_rule.\n\n" + _EXAMPLES_SHORT
            ); return
        if parsed == "error:empty_entry":
            await update.message.reply_text("Entry rule khong duoc de trong."); return
        if parsed == "error:empty_exit":
            await update.message.reply_text("Exit rule khong duoc de trong."); return
        await update.message.reply_text(f"Loi: {parsed}"); return

    symbol, entry_rule, exit_rule, days = parsed

    # Validate rules trước khi bắt đầu (fast fail, không tốn thời gian load data)
    try:
        from rule_engine import parse_rule
        parse_rule(entry_rule)
        parse_rule(exit_rule)
    except Exception as pe:
        await update.message.reply_text(
            f"LOI PARSE RULE:\n{str(pe)}\n\n" + _EXAMPLES_SHORT
        )
        return

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_backtest_rule.get(user_id, 0)
    if since < BACKTEST_RULE_COOLDOWN:
        wait = int(BACKTEST_RULE_COOLDOWN - since)
        await update.message.reply_text(f"Vui long cho {wait}s truoc khi /backtest_rule tiep."); return
    _last_backtest_rule[user_id] = time.time()

    chat_id = update.effective_chat.id

    # Gửi loading message
    msg = await update.message.reply_text(
        f"Dang chay rule backtest: {symbol} ({days} ngay)...\n"
        f"ENTRY: {entry_rule[:80]}\n"
        f"EXIT : {exit_rule[:80]}\n"
        f"Co the mat 30-90 giay."
    )

    # Chạy trong background task
    async def _bg():
        try:
            result = await asyncio.to_thread(
                _run_rule_backtest_sync, symbol, entry_rule, exit_rule, days
            )

            if result["status"] == "warn_no_signals":
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=plain(result["message"])[:4096],
                )
                return

            if result["status"] == "error":
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=f"Loi backtest: {result.get('error','?')[:300]}",
                )
                return

            # Build summary text
            _ta = _calc_trade_analytics(result["trades"], result.get("df"))
            summary = _format_rule_result(
                metrics          = result["metrics"],
                symbol           = symbol,
                entry_rule       = entry_rule,
                exit_rule        = exit_rule,
                days             = days,
                n_trades         = result["n_trades"],
                exit_ec_entry    = result["entry_ec"],
                exit_ec_exit     = result["exit_ec"],
                trades_sample    = result["trades"],
                n_buy_signals    = result["n_buy"],
                n_sell_signals   = result["n_sell"],
                bars             = result["bars"],
                trade_analytics  = _ta,
            )

            # Gửi metrics text
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=plain(summary)[:4096],
            )

            # Gửi chart
            chart_path = result.get("chart_path", "")
            if chart_path and pathlib.Path(chart_path).exists():
                m = result["metrics"]
                caption = (
                    f"Equity Curve: {symbol} | Rule Backtest | {days}D\n"
                    f"ENTRY: {entry_rule[:50]}\n"
                    f"EXIT : {exit_rule[:50]}\n"
                    f"Return: {m.get('total_return_pct',0):+.1f}% | "
                    f"Sharpe: {m.get('sharpe_ratio',0):.2f} | "
                    f"MaxDD: {m.get('max_drawdown_pct',0):.1f}%"
                )
                try:
                    with open(chart_path, "rb") as f:
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=f,
                            caption=plain(caption)[:1024],
                        )
                except Exception as pe:
                    logger.warning(f"Khong gui duoc chart: {pe}")
                finally:
                    pathlib.Path(chart_path).unlink(missing_ok=True)

            # Gửi 5 lệnh gần nhất
            trade_msg = _format_trades_msg(result["trades"])
            if trade_msg:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=plain(trade_msg)[:4096],
                )

        except Exception as e:
            import traceback
            logger.error(f"backtest_rule_cmd bg error: {e}\n{traceback.format_exc()}")
            err_text = f"Loi xu ly /backtest_rule: {str(e)[:300]}"
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=err_text,
                )
            except Exception:
                # edit fail (msg đã bị xoá hoặc timeout) → gửi message mới
                try:
                    await context.bot.send_message(chat_id=chat_id, text=err_text)
                except Exception:
                    pass

    await _bg()
