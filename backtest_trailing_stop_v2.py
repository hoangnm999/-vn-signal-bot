"""
backtest_trailing_stop_v2.py — Per-symbol ATR-based Trailing Stop Discovery
VN Trader Bot V6 — Session 31

Pipeline:
  Bước 1: Backtest training data → tìm ATR multiplier tối ưu per symbol
  Bước 2: Walk Forward validation → confirm không overfit
  Bước 3: Output config per symbol (TRAIL hoặc NO_TRAIL)

Chạy: python backtest_trailing_stop_v2.py
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Chỉ backtest 18 mã MỚI từ S31 expand (20 mã cũ đã có kết quả rồi)
MR_SYMBOLS  = ["NLG", "HDB", "BMP"]
MOM_SYMBOLS = [
    "SSI", "VND", "VIX", "CTS", "VCI", "HAG", "BCM",
    "ORS", "BSR", "VSC", "DIG", "LPB", "FTS", "APG", "VDS",
]

FWD_DAYS = {"Mean Reversion": 20, "Momentum": 10}

SIGNAL_CONFIG = {
    "Mean Reversion": {
        "regime_indicator":   "price_vs_sma50",
        "regime_condition":   "low",
        "trigger_indicators": ["stoch_k", "volume_spike", "momentum_5d"],
        "trigger_direction":  {"stoch_k": "low", "volume_spike": "high",
                               "momentum_5d": "high"},
    },
    "Momentum": {
        "regime_indicator":   "ema_cross",
        "regime_condition":   "high",
        "trigger_indicators": ["momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction":  {"momentum_5d": "high", "volume_spike": "high",
                               "candle_body": "high"},
    },
}

MIN_TRIGGERS = 2
TRIGGER_PCT  = 70

# ATR multipliers để thử
ATR_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]

# Activation: khi pnl >= ACTIVATION_ATR_MULT × ATR_pct thì bắt đầu trail
# Lock level: SL = peak_price - TRAIL_ATR_MULT × ATR
ACTIVATION_ATR_MULT = 1.5   # kích hoạt trailing khi đạt +1.5×ATR

# Walk Forward config
WF_TRAIN_MONTHS = 18
WF_TEST_MONTHS  = 6
WF_START        = date(2022, 1, 1)
WF_MIN_FOLDS    = 3          # cần ít nhất 3 folds để kết luận
WF_MIN_WFE      = 0.3        # WFE tối thiểu để chấp nhận
WF_MAX_NEG_FOLDS = 2         # tối đa 2 folds OOS âm

# Training cutoff (Bước 1)
TRAIN_END = date(2023, 12, 31)
MIN_TRADES_FOR_CONCLUSION = 10


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ema(c, span):
    return pd.Series(c.astype(float)).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c.astype(float)).rolling(p, min_periods=p).mean().values

def _compute_atr_series(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    return _sma(tr, 14)

def _compute_indicators_at(df: pd.DataFrame, i: int,
                            atr_series: np.ndarray) -> Optional[dict]:
    if i < 60:
        return None
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    vol   = df["volume"].values.astype(float)
    opn   = df["open"].values.astype(float)

    ema12  = _ema(close[:i+1], 12)
    ema26  = _ema(close[:i+1], 26)
    sma20  = _sma(close[:i+1], 20)
    sma50  = _sma(close[:i+1], 50)
    vsma20 = _sma(vol[:i+1],   20)

    lo14 = pd.Series(low[:i+1]).rolling(14).min().values
    hi14 = pd.Series(high[:i+1]).rolling(14).max().values
    denom = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch = 100 * (close[:i+1] - lo14) / denom

    px    = close[i]
    atr_v = atr_series[i] if np.isfinite(atr_series[i]) else px * 0.02
    s50   = sma50[-1]  if np.isfinite(sma50[-1])  else px
    vs20v = vsma20[-1] if np.isfinite(vsma20[-1]) else vol[i]
    c5    = close[max(i-5, 0)]

    return {
        "close":          px,
        "atr":            atr_v,
        "atr_pct":        atr_v / px * 100,
        "price_vs_sma50": float((px - s50) / (px + 1e-9) * 100),
        "ema_cross":      float((ema12[-1] - ema26[-1]) / (px + 1e-9) * 100),
        "momentum_5d":    float((px / (c5 + 1e-9) - 1.0) * 100),
        "volume_spike":   float((vol[i] / (vs20v + 1e-9)) - 1.0),
        "stoch_k":        float(stoch[-1]),
        "candle_body":    float(np.clip(abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
    }

def _compute_thresholds(df: pd.DataFrame, cluster: str,
                        train_end_idx: int) -> Optional[dict]:
    cfg      = SIGNAL_CONFIG[cluster]
    reg_ind  = cfg["regime_indicator"]
    trig_ind = cfg["trigger_indicators"]
    atr_ser  = _compute_atr_series(df)

    reg_vals  = []
    trig_vals = {t: [] for t in trig_ind}

    for k in range(60, train_end_idx):
        ind = _compute_indicators_at(df, k, atr_ser)
        if ind is None:
            continue
        v = ind.get(reg_ind)
        if v is not None and np.isfinite(v):
            reg_vals.append(v)
        for t in trig_ind:
            tv = ind.get(t)
            if tv is not None and np.isfinite(tv):
                trig_vals[t].append(tv)

    if len(reg_vals) < 20:
        return None

    reg_cond  = cfg["regime_condition"]
    reg_thresh = np.percentile(reg_vals,
                               TRIGGER_PCT if reg_cond == "low" else 100 - TRIGGER_PCT)

    trig_thresh = {}
    for t in trig_ind:
        if len(trig_vals[t]) < 20:
            return None
        direction = cfg["trigger_direction"].get(t, "high")
        trig_thresh[t] = np.percentile(
            trig_vals[t],
            TRIGGER_PCT if direction == "high" else 100 - TRIGGER_PCT
        )

    return {"reg_thresh": reg_thresh, "trig_thresh": trig_thresh}

def _has_signal(ind: dict, thresh: dict, cluster: str) -> bool:
    cfg      = SIGNAL_CONFIG[cluster]
    reg_ind  = cfg["regime_indicator"]
    reg_cond = cfg["regime_condition"]

    val = ind.get(reg_ind, float("nan"))
    if not np.isfinite(val):
        return False
    in_regime = (val <= thresh["reg_thresh"]) if reg_cond == "low" \
                else (val > thresh["reg_thresh"])
    if not in_regime:
        return False

    triggered = 0
    for t in cfg["trigger_indicators"]:
        v  = ind.get(t, float("nan"))
        th = thresh["trig_thresh"].get(t, float("nan"))
        if not (np.isfinite(v) and np.isfinite(th)):
            continue
        direction = cfg["trigger_direction"].get(t, "high")
        if (direction == "low" and v <= th) or (direction == "high" and v >= th):
            triggered += 1

    return triggered >= MIN_TRIGGERS


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATE SINGLE TRADE với exit strategy
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_date:  date
    exit_date:   date
    entry_price: float
    exit_price:  float
    pnl_pct:     float
    hold_days:   int
    exit_reason: str
    atr_pct:     float     # ATR/price tại entry (%) — đặc trưng của mã


def simulate_trade(entry_price: float, atr_at_entry: float,
                   future_closes: np.ndarray, future_dates: list[date],
                   fwd: int, trail_mult: Optional[float],
                   entry_date: date) -> Trade:
    """
    Simulate 1 trade với:
    - trail_mult=None  → Time Stop thuần
    - trail_mult=X     → ATR-based trailing: SL trail = peak - X×ATR
                         Kích hoạt khi pnl >= ACTIVATION_ATR_MULT × ATR_pct
    """
    atr_pct = atr_at_entry / entry_price * 100
    activation_threshold_pct = ACTIVATION_ATR_MULT * atr_pct  # % để kích hoạt

    peak_price  = entry_price
    trail_sl    = -np.inf
    trailing_on = False

    n = min(fwd, len(future_closes))

    for day in range(n):
        c = future_closes[day]

        if trail_mult is not None:
            pnl_pct = (c - entry_price) / entry_price * 100

            # Cập nhật peak
            if c > peak_price:
                peak_price = c

            # Kích hoạt trailing khi đạt threshold
            if pnl_pct >= activation_threshold_pct:
                trailing_on = True

            # Cập nhật SL trailing (chỉ kéo lên, không kéo xuống)
            if trailing_on:
                new_sl = peak_price - trail_mult * atr_at_entry
                trail_sl = max(trail_sl, new_sl)

                # Check stop
                if c <= trail_sl:
                    exit_price = c
                    hold_days  = day + 1
                    exit_date  = future_dates[day] if day < len(future_dates) \
                                 else entry_date + timedelta(days=hold_days)
                    return Trade(
                        entry_date=entry_date, exit_date=exit_date,
                        entry_price=entry_price, exit_price=exit_price,
                        pnl_pct=(exit_price - entry_price) / entry_price * 100,
                        hold_days=hold_days, exit_reason="TRAIL_STOP",
                        atr_pct=atr_pct,
                    )

        # Time stop
        if day == n - 1:
            exit_price = future_closes[day]
            exit_date  = future_dates[day] if day < len(future_dates) \
                         else entry_date + timedelta(days=n)
            return Trade(
                entry_date=entry_date, exit_date=exit_date,
                entry_price=entry_price, exit_price=exit_price,
                pnl_pct=(exit_price - entry_price) / entry_price * 100,
                hold_days=n, exit_reason="TIME_STOP",
                atr_pct=atr_pct,
            )

    # Fallback
    return Trade(
        entry_date=entry_date, exit_date=entry_date + timedelta(days=fwd),
        entry_price=entry_price, exit_price=entry_price,
        pnl_pct=0.0, hold_days=fwd, exit_reason="NO_DATA",
        atr_pct=atr_pct,
    )


# ══════════════════════════════════════════════════════════════════════════════
# COLLECT TRADES trong 1 window
# ══════════════════════════════════════════════════════════════════════════════

def collect_trades(df: pd.DataFrame, cluster: str,
                   window_start: date, window_end: date,
                   train_end: date,
                   trail_mult: Optional[float]) -> list[Trade]:
    """
    Collect tất cả trades trong [window_start, window_end].
    Threshold tính từ data trước train_end.
    """
    df = df.reset_index(drop=True)
    dates_arr = pd.to_datetime(df["date"]).dt.date.values
    closes    = df["close"].values.astype(float)
    atr_ser   = _compute_atr_series(df)
    fwd       = FWD_DAYS[cluster]

    # Train end index
    train_end_idx = next(
        (i for i, d in enumerate(dates_arr) if d > train_end),
        len(dates_arr)
    )
    thresh = _compute_thresholds(df, cluster, train_end_idx)
    if thresh is None:
        return []

    trades = []
    n = len(df)

    for i in range(200, n - fwd):
        d = dates_arr[i]
        if d < window_start or d > window_end:
            continue

        ind = _compute_indicators_at(df, i, atr_ser)
        if ind is None:
            continue
        if not _has_signal(ind, thresh, cluster):
            continue

        entry_price   = closes[i]
        atr_at_entry  = atr_ser[i] if np.isfinite(atr_ser[i]) else entry_price * 0.02
        future_closes = closes[i+1 : i+1+fwd]
        future_dates  = list(dates_arr[i+1 : i+1+fwd])

        if len(future_closes) < fwd:
            continue

        trade = simulate_trade(
            entry_price, atr_at_entry,
            future_closes, future_dates,
            fwd, trail_mult, d,
        )
        trades.append(trade)

    return trades


def metrics_from_trades(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0, "avg_pnl": 0.0, "avg_hold": 0.0,
                "pnl_per_day": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    pnls   = [t.pnl_pct for t in trades]
    holds  = [t.hold_days for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p <= 0]
    gw     = sum(wins)   if wins   else 0.0
    gl     = sum(losses) if losses else 1e-9
    avg_h  = float(np.mean(holds))
    avg_p  = float(np.mean(pnls))
    return {
        "n":             len(trades),
        "avg_pnl":       round(avg_p, 3),
        "avg_hold":      round(avg_h, 1),
        "pnl_per_day":   round(avg_p / avg_h, 4) if avg_h > 0 else 0.0,
        "win_rate":      round(len(wins) / len(trades) * 100, 1),
        "profit_factor": round(gw / gl, 2),
        "trail_pct":     round(sum(1 for t in trades
                                   if t.exit_reason == "TRAIL_STOP")
                               / len(trades) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: TÌM ATR MULTIPLIER TỐI ƯU (training data)
# ══════════════════════════════════════════════════════════════════════════════

def step1_find_best_multiplier(symbol: str, cluster: str,
                               df: pd.DataFrame) -> dict:
    """
    Backtest trên training data, thử các ATR multipliers.
    Trả về best_mult (None nếu không có cái nào beat baseline).
    """
    fwd = FWD_DAYS[cluster]

    # Baseline
    baseline_trades = collect_trades(
        df, cluster,
        window_start=date(2019, 1, 1),
        window_end=TRAIN_END,
        train_end=TRAIN_END,
        trail_mult=None,
    )
    if len(baseline_trades) < MIN_TRADES_FOR_CONCLUSION:
        return {"symbol": symbol, "status": "INSUFFICIENT_DATA",
                "best_mult": None, "baseline": {}, "candidates": {}}

    baseline_m = metrics_from_trades(baseline_trades)
    baseline_ppd = baseline_m["pnl_per_day"]

    candidates = {}
    for mult in ATR_MULTIPLIERS:
        trades = collect_trades(
            df, cluster,
            window_start=date(2019, 1, 1),
            window_end=TRAIN_END,
            train_end=TRAIN_END,
            trail_mult=mult,
        )
        if len(trades) < MIN_TRADES_FOR_CONCLUSION:
            continue
        m = metrics_from_trades(trades)
        candidates[mult] = m

    # Tìm multiplier tốt nhất theo PnL/Day
    best_mult = None
    best_ppd  = baseline_ppd  # phải beat baseline mới tính

    for mult, m in candidates.items():
        if m["pnl_per_day"] > best_ppd:
            best_ppd  = m["pnl_per_day"]
            best_mult = mult

    status = "TRAIL_CANDIDATE" if best_mult is not None else "NO_TRAIL"

    return {
        "symbol":     symbol,
        "cluster":    cluster,
        "status":     status,
        "best_mult":  best_mult,
        "baseline":   baseline_m,
        "candidates": candidates,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: WALK FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def step2_walk_forward(symbol: str, cluster: str, df: pd.DataFrame,
                       best_mult: float) -> dict:
    """
    Walk Forward với expanding window.
    Mỗi fold: train 18 tháng, test 6 tháng.
    So sánh trail vs baseline trên OOS.
    """
    folds = []
    fold_start = WF_START

    while True:
        train_end_wf = fold_start + timedelta(days=WF_TRAIN_MONTHS * 30)
        test_end     = train_end_wf + timedelta(days=WF_TEST_MONTHS * 30)

        # Check đủ data không
        dates_arr = pd.to_datetime(df["date"]).dt.date.values
        if test_end > dates_arr[-1]:
            break

        # OOS baseline
        base_trades = collect_trades(
            df, cluster,
            window_start=train_end_wf,
            window_end=test_end,
            train_end=train_end_wf,
            trail_mult=None,
        )

        # OOS trail
        trail_trades = collect_trades(
            df, cluster,
            window_start=train_end_wf,
            window_end=test_end,
            train_end=train_end_wf,
            trail_mult=best_mult,
        )

        # IS trail (để tính WFE)
        is_trail_trades = collect_trades(
            df, cluster,
            window_start=fold_start,
            window_end=train_end_wf,
            train_end=train_end_wf,
            trail_mult=best_mult,
        )

        base_m     = metrics_from_trades(base_trades)
        trail_m    = metrics_from_trades(trail_trades)
        is_trail_m = metrics_from_trades(is_trail_trades)

        wfe = (trail_m["pnl_per_day"] / is_trail_m["pnl_per_day"]
               if is_trail_m["pnl_per_day"] > 0 else 0.0)

        improvement = trail_m["pnl_per_day"] - base_m["pnl_per_day"]

        folds.append({
            "fold":       f"{train_end_wf.strftime('%Y-%m')}→{test_end.strftime('%Y-%m')}",
            "n_base":     base_m["n"],
            "n_trail":    trail_m["n"],
            "base_ppd":   base_m["pnl_per_day"],
            "trail_ppd":  trail_m["pnl_per_day"],
            "is_ppd":     is_trail_m["pnl_per_day"],
            "wfe":        round(wfe, 2),
            "improvement": round(improvement, 4),
            "trail_pct":  trail_m["trail_pct"],
        })

        # Expanding: chỉ tăng fold_start 6 tháng (test window slide)
        fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)

    if len(folds) < WF_MIN_FOLDS:
        return {"wf_status": "INSUFFICIENT_FOLDS", "folds": folds}

    # Đánh giá
    oos_improvements = [f["improvement"] for f in folds]
    wfes             = [f["wfe"] for f in folds]
    neg_folds        = sum(1 for x in oos_improvements if x < 0)
    avg_wfe          = float(np.mean([w for w in wfes if w > 0])) if wfes else 0
    avg_improvement  = float(np.mean(oos_improvements))
    consistency      = round((len(folds) - neg_folds) / len(folds) * 100, 1)

    if neg_folds > WF_MAX_NEG_FOLDS or avg_wfe < WF_MIN_WFE:
        wf_status = "OVERFIT"
    elif avg_improvement <= 0:
        wf_status = "NO_IMPROVEMENT"
    else:
        wf_status = "VALIDATED"

    return {
        "wf_status":       wf_status,
        "avg_wfe":         round(avg_wfe, 2),
        "avg_improvement": round(avg_improvement, 4),
        "consistency":     consistency,
        "neg_folds":       neg_folds,
        "total_folds":     len(folds),
        "folds":           folds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: GENERATE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def step3_generate_config(step1: dict, step2: Optional[dict],
                          df: pd.DataFrame) -> dict:
    """Tổng hợp kết quả → config per symbol."""
    symbol  = step1["symbol"]
    cluster = step1.get("cluster", "")

    # Tính ATR trung bình của mã (để hiển thị ngưỡng thực tế)
    atr_ser = _compute_atr_series(df)
    closes  = df["close"].values.astype(float)
    valid   = [(atr_ser[i] / closes[i] * 100)
               for i in range(len(atr_ser))
               if np.isfinite(atr_ser[i]) and closes[i] > 0]
    avg_atr_pct = round(float(np.median(valid[-60:])), 2) if valid else 2.0

    if step1["status"] == "NO_TRAIL":
        return {
            "symbol":       symbol,
            "cluster":      cluster,
            "trail":        False,
            "reason":       "Trailing stop không cải thiện PnL/Day trong training",
            "avg_atr_pct":  avg_atr_pct,
        }

    if step1["status"] == "INSUFFICIENT_DATA":
        return {
            "symbol":       symbol,
            "cluster":      cluster,
            "trail":        False,
            "reason":       "Không đủ trades để kết luận",
            "avg_atr_pct":  avg_atr_pct,
        }

    if step2 is None or step2["wf_status"] in ("OVERFIT", "NO_IMPROVEMENT",
                                                "INSUFFICIENT_FOLDS"):
        return {
            "symbol":         symbol,
            "cluster":        cluster,
            "trail":          False,
            "reason":         f"Walk Forward: {step2['wf_status'] if step2 else 'N/A'}",
            "best_mult_is":   step1["best_mult"],
            "avg_atr_pct":    avg_atr_pct,
        }

    mult = step1["best_mult"]
    activation_pct = round(ACTIVATION_ATR_MULT * avg_atr_pct, 2)
    trail_sl_pct   = round(mult * avg_atr_pct, 2)

    return {
        "symbol":           symbol,
        "cluster":          cluster,
        "trail":            True,
        "trail_mult":       mult,
        "activation_pct":   activation_pct,    # % gain để kích hoạt
        "trail_sl_pct":     trail_sl_pct,       # khoảng trail từ peak
        "avg_atr_pct":      avg_atr_pct,
        "wf_status":        step2["wf_status"],
        "wf_avg_wfe":       step2["avg_wfe"],
        "wf_consistency":   step2["consistency"],
        "wf_improvement":   step2["avg_improvement"],
        "reason":           (f"Validated: mult={mult}x, "
                             f"WFE={step2['avg_wfe']}, "
                             f"consistency={step2['consistency']}%"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    try:
        from vn_loader import load_vn_ohlcv
    except ImportError:
        print("ERROR: Khong import duoc vn_loader. Chay trong project dir.")
        sys.exit(1)

    all_configs = []

    for cluster, symbols in [("Mean Reversion", MR_SYMBOLS),
                              ("Momentum", MOM_SYMBOLS)]:
        print(f"\n{'='*65}")
        print(f"CLUSTER: {cluster}")
        print(f"{'='*65}")

        for symbol in symbols:
            print(f"\n  [{symbol}]")
            try:
                df = load_vn_ohlcv(symbol, days=2500, min_bars=400)
                if df is None or len(df) < 400:
                    print(f"    ⚠️  Không đủ data, bỏ qua")
                    continue
            except Exception as e:
                print(f"    ❌ Load error: {e}")
                continue

            # Bước 1
            print(f"    Bước 1: Tìm ATR multiplier tối ưu...", end=" ")
            s1 = step1_find_best_multiplier(symbol, cluster, df)
            base_n   = s1["baseline"].get("n", 0)
            base_ppd = s1["baseline"].get("pnl_per_day", 0)
            print(f"baseline: {base_n} trades, PPD={base_ppd:+.4f}%")

            # In chi tiết từng multiplier
            if s1["candidates"]:
                for mult, m in sorted(s1["candidates"].items()):
                    beat = "✅" if m["pnl_per_day"] > base_ppd else "❌"
                    print(f"      mult={mult}x: PPD={m['pnl_per_day']:+.4f}% "
                          f"WR={m['win_rate']}% trail%={m['trail_pct']}% {beat}")

            if s1["status"] == "NO_TRAIL":
                print(f"    → NO_TRAIL (không có multiplier nào beat baseline)")
                cfg = step3_generate_config(s1, None, df)
                all_configs.append(cfg)
                continue

            print(f"    → Best mult: {s1['best_mult']}x")

            # Bước 2
            print(f"    Bước 2: Walk Forward validation...", end=" ")
            s2 = step2_walk_forward(symbol, cluster, df, s1["best_mult"])
            print(f"{s2['wf_status']} "
                  f"(WFE={s2.get('avg_wfe','N/A')}, "
                  f"consistency={s2.get('consistency','N/A')}%, "
                  f"improvement={s2.get('avg_improvement','N/A')})")

            # In fold details
            for fold in s2.get("folds", []):
                ok = "✅" if fold["improvement"] > 0 else "❌"
                print(f"      {fold['fold']}: "
                      f"base={fold['base_ppd']:+.4f}% → "
                      f"trail={fold['trail_ppd']:+.4f}% "
                      f"(Δ={fold['improvement']:+.4f}%, "
                      f"WFE={fold['wfe']}, trail%={fold['trail_pct']}%) {ok}")

            # Bước 3
            cfg = step3_generate_config(s1, s2, df)
            all_configs.append(cfg)
            status_icon = "✅ TRAIL" if cfg["trail"] else "❌ NO_TRAIL"
            print(f"    → {status_icon}: {cfg['reason']}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("FINAL CONFIG — Copy vào cluster_scanner.py")
    print(f"{'='*65}")

    trail_yes = [c for c in all_configs if c["trail"]]
    trail_no  = [c for c in all_configs if not c["trail"]]

    print(f"\n✅ SYMBOLS CÓ TRAILING STOP ({len(trail_yes)}):")
    print(f"{'Symbol':<8} {'Cluster':<18} {'Mult':>5} {'Act%':>7} "
          f"{'Trail%':>7} {'ATR%':>6} {'WFE':>6} {'Consist':>8}")
    print("-" * 65)
    for c in trail_yes:
        print(f"{c['symbol']:<8} {c['cluster']:<18} "
              f"{c['trail_mult']:>5.1f}x "
              f"{c['activation_pct']:>6.2f}% "
              f"{c['trail_sl_pct']:>6.2f}% "
              f"{c['avg_atr_pct']:>5.2f}% "
              f"{c['wf_avg_wfe']:>6.2f} "
              f"{c['wf_consistency']:>7.1f}%")

    print(f"\n❌ SYMBOLS KHÔNG DÙNG TRAILING STOP ({len(trail_no)}):")
    for c in trail_no:
        print(f"  {c['symbol']:<8} {c['cluster']:<18} — {c['reason']}")

    # Python dict để dùng trong code
    print(f"\n\n# TRAIL_CONFIG — paste vào cluster_scanner.py")
    print("TRAIL_CONFIG = {")
    for c in all_configs:
        if c["trail"]:
            print(f"    '{c['symbol']}': {{"
                  f"'mult': {c['trail_mult']}, "
                  f"'activation_pct': {c['activation_pct']}, "
                  f"}},")
        else:
            print(f"    # '{c['symbol']}': NO_TRAIL")
    print("}")


if __name__ == "__main__":
    run()
