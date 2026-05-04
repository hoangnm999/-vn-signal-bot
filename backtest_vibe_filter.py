"""
backtest_vibe_filter.py — Vibe Agents Confirmation Filter Backtest
VN Trader Bot V6 — Session 32

Pipeline 4 bước (theo SESSION_PROMPT_S32_VIBE.md):

  Bước 1: Tính predictive power (Cohen's d) từng engine trên historical data
          → Loại engine |d| < COHEN_D_MIN (gần random)

  Bước 2: Backtest filter combinations
          Baseline vs Filtered (cluster signal + engine = +1)
          So sánh Exp, WR, PF, n_trades

  Bước 3: Walk Forward validation (expanding 18M train / 6M test)
          WFE = OOS_exp / IS_exp
          Accept: WFE >= 0.3, consistency >= 60%

  Bước 4: Output kết quả → gợi ý integrate vào cluster_scanner.py
          (score bonus +20% nếu đồng thuận, -20% nếu ngược chiều)

Chạy: python backtest_vibe_filter.py [--cluster MR|MOM|BO|ALL] [--step 1|2|3|ALL]

Yêu cầu:
  pip install scipy
  File vibe_skills.py trong cùng thư mục hoặc PYTHONPATH
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# --- Watchlist (đồng bộ cluster_scanner.py S31) ---
MR_SYMBOLS = [
    "DCM", "NKG", "DPM", "HAH", "HCM", "HSG", "DGC", "GAS",
    "NLG", "HDB", "BMP",
]
MOM_SYMBOLS = [
    "VCB", "BID", "MBB", "MWG", "CTG", "FRT", "REE", "FPT",
    "GMD", "STB", "PNJ", "TCB",
    "SSI", "VND", "VIX", "CTS", "VCI", "HAG", "BCM",
    "ORS", "BSR", "VSC", "DIG", "LPB", "FTS", "APG", "VDS",
]
BO_SYMBOLS = [
    "VIB", "KBC", "KSB", "HT1", "SHB", "GVR", "SIP",
    "DGC", "DCM", "NKG", "HSG",          # overlap MR+BO
    "VIX", "FPT", "TCB", "BSR", "VND", "GMD", "LPB",  # overlap MOM+BO
]

FWD_DAYS = {"Mean Reversion": 20, "Momentum": 10, "Breakout": 15}
FEE_PCT  = 0.15   # tổng phí mua + bán (0.15%)

# --- Cluster signal config (đồng bộ cluster_scanner.py) ---
SIGNAL_CONFIG = {
    "Mean Reversion": {
        "regime_indicator":  "price_vs_sma50",
        "regime_condition":  "low",
        "trigger_indicators":["stoch_k", "volume_spike", "momentum_5d"],
        "trigger_direction": {"stoch_k": "low", "volume_spike": "high",
                              "momentum_5d": "high"},
    },
    "Momentum": {
        "regime_indicator":  "ema_cross",
        "regime_condition":  "high",
        "trigger_indicators":["momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction": {"momentum_5d": "high", "volume_spike": "high",
                              "candle_body": "high"},
    },
    "Breakout": {
        "regime_indicator":  "bb_squeeze",
        "regime_condition":  "high",
        "trigger_indicators":["consolidation", "vol_dry_up"],
        "trigger_direction": {"consolidation": "low", "vol_dry_up": "high"},
    },
}

TRIGGER_PCT  = 70
MIN_TRIGGERS = 2

# --- Backtest window ---
DATA_START  = date(2019, 1, 1)
TRAIN_END   = date(2023, 12, 31)
DATA_END    = date(2024, 12, 31)

# --- WF config ---
WF_TRAIN_MONTHS = 18
WF_TEST_MONTHS  = 6
WF_START        = date(2022, 1, 1)
WF_MIN_FOLDS    = 3
WF_MIN_WFE      = 0.30
WF_MAX_NEG_FOLDS= 2

# --- Bộ lọc Cohen's d ---
COHEN_D_MIN     = 0.20   # loại engine có |d| < 0.2

# --- Vibe agent score bonus (Bước 4) ---
VIBE_AGREE_BONUS    = 0.20   # +20% score khi đồng thuận
VIBE_DISAGREE_BONUS = -0.20  # -20% score khi ngược chiều

MIN_TRADES_REPORT = 5  # tối thiểu số trades để báo kết quả

# --- Engines bị skip trong backtest ---
# MLStrategy      : RandomForest fit lại trên mỗi df_slice -> rat cham (~10 phut/symbol).
#                   Refit tren same data cung la leakage risk trong backtest context.
# FundamentalFilter: HTTP call (Fireant/KBS) moi lan goi, khong co cache per-date
#                   -> treo pipeline. Fundamental la static, khong thay doi theo
#                   signal date trong lich su -> khong phu hop lam time-series filter.
# Seasonal        : Chi lookup month/weekday co dinh, khong dung df -> predictive
#                   power rat thap, bo de pipeline gon.
SKIP_ENGINES: set = {
    "MLStrategy",
    "FundamentalFilter",
    "Seasonal",
}


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS (replicate cluster_scanner.py)
# ══════════════════════════════════════════════════════════════════════════════

def _ema(c: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(c.astype(float)).ewm(span=span, adjust=False).mean().values

def _sma(c: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(c.astype(float)).rolling(p, min_periods=p).mean().values

def _compute_atr(df: pd.DataFrame) -> np.ndarray:
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float)
    lw = df["low"].values.astype(float)
    h_prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - lw, np.maximum(np.abs(h - h_prev), np.abs(lw - h_prev)))
    return _sma(tr, 14)


def _indicators_at(df: pd.DataFrame, i: int,
                   atr_ser: np.ndarray) -> Optional[dict]:
    """Tính indicators tại row i — dùng slice [:i+1] tránh lookahead."""
    if i < 60:
        return None

    c   = df["close"].values.astype(float)
    h   = df["high"].values.astype(float)
    lw  = df["low"].values.astype(float)
    v   = df["volume"].values.astype(float)
    opn = df["open"].values.astype(float)

    # Slice lịch sử đến i (inclusive) → không lookahead
    ci   = c[:i+1]
    vi   = v[:i+1]
    lwi  = lw[:i+1]
    hi   = h[:i+1]

    ema12  = _ema(ci, 12)[-1]
    ema26  = _ema(ci, 26)[-1]
    sma20  = _sma(ci, 20)[-1]
    sma50  = _sma(ci, 50)[-1]
    vsma20 = _sma(vi, 20)[-1]
    vsma60 = _sma(vi, 60)[-1]

    lo14 = pd.Series(lwi).rolling(14).min().values[-1]
    hi14 = pd.Series(hi).rolling(14).max().values[-1]
    denom = hi14 - lo14 if hi14 != lo14 else 1e-9
    stoch = 100 * (ci[-1] - lo14) / denom

    px    = ci[-1]
    atr_v = atr_ser[i] if np.isfinite(atr_ser[i]) else px * 0.02
    s20   = sma20 if np.isfinite(sma20) else px
    s50   = sma50 if np.isfinite(sma50) else px
    vs20v = vsma20 if np.isfinite(vsma20) else vi[-1]
    vs60v = vsma60 if np.isfinite(vsma60) else vs20v
    c5    = ci[max(len(ci)-6, 0)]

    # BB width (breakout indicator)
    bb_std = float(pd.Series(ci).rolling(20).std().iloc[-1]) if len(ci) >= 20 else atr_v
    bb_width = float(4 * bb_std / (s20 + 1e-9) * 100)

    # Consolidation (breakout indicator)
    window_c = ci[max(0, i-14):i+1]
    consol   = float(np.sum(np.abs(window_c - px) / (px + 1e-9) < 0.03)) / max(len(window_c), 1)

    return {
        "price_vs_sma50": float((px - s50) / (px + 1e-9) * 100),
        "ema_cross":      float((ema12 - ema26) / (px + 1e-9) * 100),
        "momentum_5d":    float((px / (c5 + 1e-9) - 1.0) * 100),
        "volume_spike":   float((vi[-1] / (vs20v + 1e-9)) - 1.0),
        "stoch_k":        float(stoch),
        "candle_body":    float(np.clip(abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
        "bb_squeeze":     bb_width,
        "consolidation":  consol,
        "vol_dry_up":     float((vs20v / (vs60v + 1e-9)) - 1.0),
        "close":          px,
        "atr":            atr_v,
    }


def _compute_thresholds(df: pd.DataFrame, cluster: str,
                        train_end_idx: int) -> Optional[dict]:
    """Tính thresholds từ training data (rows 0..train_end_idx-1)."""
    if train_end_idx < 200:
        return None
    cfg     = SIGNAL_CONFIG[cluster]
    atr_ser = _compute_atr(df)
    rows    = []
    for i in range(60, train_end_idx):
        ind = _indicators_at(df, i, atr_ser)
        if ind:
            rows.append(ind)
    if len(rows) < 20:
        return None

    reg_ind  = cfg["regime_indicator"]
    trig_ind = cfg["trigger_indicators"]
    trig_dir = cfg["trigger_direction"]

    reg_vals  = [r[reg_ind] for r in rows if np.isfinite(r.get(reg_ind, float("nan")))]
    reg_cond  = cfg["regime_condition"]
    reg_pct   = TRIGGER_PCT if reg_cond == "low" else 100 - TRIGGER_PCT
    reg_thresh = float(np.percentile(reg_vals, reg_pct)) if reg_vals else 0.0

    trig_thresh = {}
    for t in trig_ind:
        vals = [r[t] for r in rows if np.isfinite(r.get(t, float("nan")))]
        if len(vals) < 20:
            continue
        d = "low" if trig_dir.get(t, "high") == "low" else "high"
        trig_thresh[t] = float(np.percentile(vals,
                                100 - TRIGGER_PCT if d == "low" else TRIGGER_PCT))
    return {"reg_thresh": reg_thresh, "trig_thresh": trig_thresh}


def _has_cluster_signal(ind: dict, thresh: dict, cluster: str) -> bool:
    """Kiểm tra cluster signal tại row ind với thresh đã tính."""
    cfg     = SIGNAL_CONFIG[cluster]
    reg_ind = cfg["regime_indicator"]
    reg_cond= cfg["regime_condition"]
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
        d  = cfg["trigger_direction"].get(t, "high")
        if (d == "low" and v <= th) or (d == "high" and v >= th):
            triggered += 1
    return triggered >= MIN_TRIGGERS


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_symbol(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load OHLCV cho symbol từ vn_loader (KBS source).
    Trả về DataFrame với columns: date, open, high, low, close, volume
    """
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2000, min_bars=200)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except ImportError:
        logger.error("Không tìm thấy vn_loader — đặt vn_loader.py cùng thư mục")
        return None
    except Exception as e:
        logger.warning(f"[Load] {symbol}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# VIBE AGENT RUNNER (wrapper gọi vibe_skills.py)
# ══════════════════════════════════════════════════════════════════════════════

# Timeout (giây) cho mỗi engine call — tránh treo vô hạn khi engine fetch HTTP
ENGINE_TIMEOUT_SEC = 8

def _run_engine_safe(name: str, engine, data_map: dict,
                     symbol: str, timeout: int = ENGINE_TIMEOUT_SEC) -> int:
    """
    Chay 1 engine với hard timeout dùng signal.alarm (Linux only).
    Trả về signal int (+1/0/-1), hoặc 0 nếu timeout/exception.
    """
    import signal as _signal

    class _Timeout(Exception):
        pass

    def _handler(signum, frame):
        raise _Timeout()

    old_handler = _signal.signal(_signal.SIGALRM, _handler)
    _signal.alarm(timeout)
    try:
        sigs, _ = engine.generate(data_map)
        return int(sigs.get(symbol, 0))
    except _Timeout:
        logger.warning(f"[Vibe] {symbol}/{name}: TIMEOUT >{timeout}s — skip")
        return 0
    except Exception as e:
        logger.debug(f"[Vibe] {symbol}/{name}: {e}")
        return 0
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, old_handler)


def run_vibe_on_slice(symbol: str, df_slice: pd.DataFrame) -> dict:
    """
    Chay cac vibe engines tren df_slice (data den ngay i, khong lookahead).
    - Skip engines trong SKIP_ENGINES
    - Hard timeout ENGINE_TIMEOUT_SEC giay moi engine
    Tra ve {engine_name: int} -- +1 bull / -1 bear / 0 neutral
    """
    try:
        from vibe_skills import _ENGINES, _prep
        df_prep = _prep(df_slice)
        if len(df_prep) < 20:
            return {}

        data_map = {symbol: df_prep}
        signals: dict = {}

        for name, engine in _ENGINES.items():
            if name in SKIP_ENGINES:
                continue
            signals[name] = _run_engine_safe(name, engine, data_map, symbol)

        return signals
    except ImportError:
        logger.error("Khong tim thay vibe_skills.py -- dat vao cung thu muc")
        return {}
    except Exception as e:
        logger.debug(f"[Vibe] {symbol}: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class Trade:
    """Trade record với vibe signals tại ngày entry."""
    __slots__ = (
        "symbol", "cluster", "entry_date", "exit_date",
        "entry_price", "exit_price", "pnl_pct", "hold_days", "vibe_signals",
    )

    def __init__(
        self,
        symbol:       str,
        cluster:      str,
        entry_date:   date,
        exit_date:    date,
        entry_price:  float,
        exit_price:   float,
        pnl_pct:      float,
        hold_days:    int,
        vibe_signals: dict | None = None,
    ):
        self.symbol       = symbol
        self.cluster      = cluster
        self.entry_date   = entry_date
        self.exit_date    = exit_date
        self.entry_price  = entry_price
        self.exit_price   = exit_price
        self.pnl_pct      = pnl_pct
        self.hold_days    = hold_days
        self.vibe_signals = vibe_signals if vibe_signals is not None else {}


def _metrics(trades: list[Trade]) -> dict:
    if not trades:
        return {"n": 0, "exp": 0.0, "wr": 0.0, "pf": 0.0, "avg_hold": 0.0}
    pnls  = [t.pnl_pct for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses= [abs(p) for p in pnls if p <= 0]
    gw    = sum(wins)   if wins   else 0.0
    gl    = sum(losses) if losses else 1e-9
    return {
        "n":        len(trades),
        "exp":      round(float(np.mean(pnls)), 3),
        "wr":       round(len(wins) / len(trades) * 100, 1),
        "pf":       round(gw / gl, 2),
        "avg_hold": round(float(np.mean([t.hold_days for t in trades])), 1),
    }


def collect_trades_for_symbol(
    symbol: str,
    cluster: str,
    df: pd.DataFrame,
    window_start: date,
    window_end: date,
    train_end_date: date,
    with_vibe: bool = False,
) -> list[Trade]:
    """
    Collect trades trong cửa sổ [window_start, window_end].
    Threshold tính từ data <= train_end_date (expanding window, không lookahead).

    with_vibe=True → ghi thêm vibe signals vào mỗi trade (chậm hơn ~10x).
    """
    dates_arr = pd.to_datetime(df["date"]).dt.date.values
    closes    = df["close"].values.astype(float)
    atr_ser   = _compute_atr(df)
    fwd       = FWD_DAYS[cluster]

    # Train end index
    train_end_idx = next(
        (i for i, d in enumerate(dates_arr) if d > train_end_date),
        len(dates_arr)
    )
    thresh = _compute_thresholds(df, cluster, train_end_idx)
    if thresh is None:
        return []

    trades: list[Trade] = []
    n = len(df)

    for i in range(200, n - fwd):
        d = dates_arr[i]
        if d < window_start or d > window_end:
            continue

        ind = _indicators_at(df, i, atr_ser)
        if ind is None:
            continue
        if not _has_cluster_signal(ind, thresh, cluster):
            continue

        # Entry T+1 (mua ngày hôm sau)
        if i + 1 >= n:
            continue
        entry_price = closes[i + 1]
        entry_date  = dates_arr[i + 1]

        # Exit: time stop tại T+fwd từ entry
        exit_idx  = i + 1 + fwd
        if exit_idx >= n:
            continue
        exit_price = closes[exit_idx]
        exit_date  = dates_arr[exit_idx]

        pnl_pct    = (exit_price / entry_price - 1) * 100 - FEE_PCT
        hold_days  = fwd

        # Vibe signals tại ngày signal (df slice đến i, không lookahead)
        vibe_sigs: dict = {}
        if with_vibe:
            df_slice = df.iloc[: i + 1].copy()
            vibe_sigs = run_vibe_on_slice(symbol, df_slice)

        trades.append(Trade(
            symbol       = symbol,
            cluster      = cluster,
            entry_date   = entry_date,
            exit_date    = exit_date,
            entry_price  = entry_price,
            exit_price   = exit_price,
            pnl_pct      = pnl_pct,
            hold_days    = hold_days,
            vibe_signals = vibe_sigs,
        ))

    return trades


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1 — COHEN'S D: Predictive power từng engine
# ══════════════════════════════════════════════════════════════════════════════

def step1_cohen_d(cluster: str, symbols: list[str]) -> pd.DataFrame:
    """
    Với mỗi engine, tính Cohen's d của forward return khi engine=+1 vs engine=0/-1.

    d = (mean_bull - mean_other) / pooled_std

    Giải thích:
      d > 0.2  : engine có predictive power → giữ lại cho bước 2
      |d| < 0.2: engine gần như random → loại
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BƯỚC 1: Cohen's d — cluster={cluster}, {len(symbols)} mã")
    logger.info(f"{'='*60}")

    # Thu thập tất cả (vibe_signals, fwd_return) trên training window
    all_records: list[dict] = []   # {engine: signal, fwd_return: float}

    for sym in symbols:
        logger.info(f"  [{sym}] Loading...")
        df = load_symbol(sym)
        if df is None or len(df) < 300:
            logger.warning(f"  [{sym}] Skip — không đủ data")
            continue

        dates_arr  = pd.to_datetime(df["date"]).dt.date.values
        closes     = df["close"].values.astype(float)
        atr_ser    = _compute_atr(df)
        fwd        = FWD_DAYS[cluster]

        # Train end index
        train_end_idx = next(
            (i for i, d in enumerate(dates_arr) if d > TRAIN_END),
            len(dates_arr)
        )
        thresh = _compute_thresholds(df, cluster, train_end_idx)
        if thresh is None:
            continue

        # Pre-scan: dem so cluster signals truoc (khong goi vibe)
        signal_rows = []
        for i in range(200, train_end_idx - fwd):
            d = dates_arr[i]
            if d < DATA_START:
                continue
            ind = _indicators_at(df, i, atr_ser)
            if ind is None:
                continue
            if not _has_cluster_signal(ind, thresh, cluster):
                continue
            if i + 1 + fwd >= len(closes):
                continue
            entry  = closes[i + 1]
            exit_  = closes[i + 1 + fwd]
            fwd_ret = (exit_ / entry - 1) * 100 - FEE_PCT
            signal_rows.append((i, d, fwd_ret))

        logger.info(f"  [{sym}] {len(signal_rows)} cluster signals -> chay vibe engines "
                    f"(skip: {SKIP_ENGINES})")

        for idx, (i, d, fwd_ret) in enumerate(signal_rows):
            if idx % 20 == 0 and idx > 0:
                logger.info(f"  [{sym}] vibe progress: {idx}/{len(signal_rows)}")

            # Vibe signals (khong lookahead)
            df_slice = df.iloc[: i + 1].copy()
            vibe_sigs = run_vibe_on_slice(sym, df_slice)
            if not vibe_sigs:
                continue

            rec = {"symbol": sym, "date": d, "fwd_return": fwd_ret}
            rec.update(vibe_sigs)
            all_records.append(rec)

    if not all_records:
        logger.warning("  Không có records — kiểm tra vibe_skills.py và data loader")
        return pd.DataFrame()

    df_rec = pd.DataFrame(all_records)
    logger.info(f"  Tổng records (cluster signals): {len(df_rec)}")

    # Tính Cohen's d từng engine
    engine_cols = [c for c in df_rec.columns
                   if c not in ("symbol", "date", "fwd_return")]

    results = []
    for eng in engine_cols:
        bull_mask  = df_rec[eng] == 1
        other_mask = df_rec[eng] != 1
        bull_ret   = df_rec.loc[bull_mask,  "fwd_return"].values
        other_ret  = df_rec.loc[other_mask, "fwd_return"].values

        if len(bull_ret) < 10 or len(other_ret) < 10:
            results.append({
                "engine": eng, "d": 0.0, "n_bull": len(bull_ret),
                "n_other": len(other_ret),
                "mean_bull": 0.0, "mean_other": 0.0,
                "status": "INSUFFICIENT",
            })
            continue

        # Cohen's d (pooled std)
        pooled_std = np.sqrt(
            ((len(bull_ret) - 1) * np.var(bull_ret, ddof=1) +
             (len(other_ret) - 1) * np.var(other_ret, ddof=1)) /
            (len(bull_ret) + len(other_ret) - 2)
        )
        d = (np.mean(bull_ret) - np.mean(other_ret)) / (pooled_std + 1e-9)

        # t-test p-value
        _, p_val = scipy_stats.ttest_ind(bull_ret, other_ret, equal_var=False)

        results.append({
            "engine":      eng,
            "d":           round(float(d), 3),
            "n_bull":      len(bull_ret),
            "n_other":     len(other_ret),
            "mean_bull":   round(float(np.mean(bull_ret)), 3),
            "mean_other":  round(float(np.mean(other_ret)), 3),
            "p_value":     round(float(p_val), 4),
            "status":      "KEEP" if abs(d) >= COHEN_D_MIN else "DROP",
        })

    df_result = pd.DataFrame(results).sort_values("d", ascending=False)

    # Print summary
    logger.info(f"\n  {'Engine':<20} {'d':>6} {'n_bull':>7} {'mean_bull':>10} {'mean_other':>11} {'p':>7} {'Status'}")
    logger.info(f"  {'-'*75}")
    for _, row in df_result.iterrows():
        keep_marker = "✓" if row["status"] == "KEEP" else "✗"
        logger.info(
            f"  {keep_marker} {row['engine']:<18} {row['d']:>6.3f} "
            f"{row['n_bull']:>7} {row['mean_bull']:>10.3f} "
            f"{row['mean_other']:>11.3f} {row.get('p_value', 0):>7.4f}  {row['status']}"
        )

    keep_engines = df_result[df_result["status"] == "KEEP"]["engine"].tolist()
    logger.info(f"\n  → Giữ lại {len(keep_engines)}/{len(engine_cols)} engines: {keep_engines}")

    return df_result


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2 — BACKTEST FILTER COMBINATIONS
# ══════════════════════════════════════════════════════════════════════════════

def step2_backtest_filters(
    cluster: str,
    symbols: list[str],
    keep_engines: list[str],
) -> pd.DataFrame:
    """
    Với mỗi engine trong keep_engines:
      Baseline: cluster signal thuần (tất cả trades)
      Filtered: cluster signal + engine == +1

    So sánh Exp, WR, PF, n_trades trên training window.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BƯỚC 2: Backtest Filter — cluster={cluster}")
    logger.info(f"{'='*60}")

    if not keep_engines:
        logger.warning("  Không có engine nào pass bước 1")
        return pd.DataFrame()

    # Thu thập tất cả trades với vibe signals (chậm — chỉ chạy 1 lần)
    all_trades: list[Trade] = []

    for sym in symbols:
        logger.info(f"  [{sym}] Collecting trades with vibe signals...")
        df = load_symbol(sym)
        if df is None or len(df) < 300:
            continue

        trades = collect_trades_for_symbol(
            sym, cluster, df,
            window_start   = DATA_START,
            window_end     = TRAIN_END,
            train_end_date = TRAIN_END,
            with_vibe      = True,
        )
        all_trades.extend(trades)

    if not all_trades:
        logger.warning("  Không có trades")
        return pd.DataFrame()

    logger.info(f"  Tổng trades baseline: {len(all_trades)}")
    baseline = _metrics(all_trades)
    logger.info(f"  Baseline: Exp={baseline['exp']:.3f}% | WR={baseline['wr']:.1f}% "
                f"| PF={baseline['pf']:.2f} | n={baseline['n']}")

    results = []

    for eng in keep_engines:
        # AND filter: cluster signal + engine == +1
        filtered_and = [t for t in all_trades if t.vibe_signals.get(eng) == 1]
        # NOT filter: cluster signal + engine != +1 (ngược chiều để so sánh)
        filtered_excl = [t for t in all_trades if t.vibe_signals.get(eng) == -1]

        m_and  = _metrics(filtered_and)
        m_excl = _metrics(filtered_excl)

        # Delta vs baseline
        exp_delta = round(m_and["exp"] - baseline["exp"], 3)
        wr_delta  = round(m_and["wr"]  - baseline["wr"],  1)
        pf_delta  = round(m_and["pf"]  - baseline["pf"],  2)
        trade_pct = round(m_and["n"] / len(all_trades) * 100, 1) if all_trades else 0

        verdict = "USEFUL" if (exp_delta > 0 and m_and["n"] >= MIN_TRADES_REPORT) else "WEAK"

        results.append({
            "engine":        eng,
            "n_base":        baseline["n"],
            "n_filtered":    m_and["n"],
            "trade_pct":     trade_pct,       # % trades còn lại sau filter
            "exp_base":      baseline["exp"],
            "exp_filtered":  m_and["exp"],
            "exp_delta":     exp_delta,
            "wr_base":       baseline["wr"],
            "wr_filtered":   m_and["wr"],
            "wr_delta":      wr_delta,
            "pf_base":       baseline["pf"],
            "pf_filtered":   m_and["pf"],
            "pf_delta":      pf_delta,
            # Khi ngược chiều (vibe=-1) thì tệ hơn hay không?
            "n_excl":        m_excl["n"],
            "exp_excl":      m_excl["exp"],
            "verdict":       verdict,
        })

    df_result = pd.DataFrame(results).sort_values("exp_delta", ascending=False)

    logger.info(f"\n  {'Engine':<20} {'n_filt':>7} {'trade%':>7} {'Δexp':>7} {'Δwr':>6} {'Δpf':>6} {'Verdict'}")
    logger.info(f"  {'-'*65}")
    for _, row in df_result.iterrows():
        marker = "✓" if row["verdict"] == "USEFUL" else "✗"
        logger.info(
            f"  {marker} {row['engine']:<18} {row['n_filtered']:>7} "
            f"{row['trade_pct']:>6.0f}% {row['exp_delta']:>+7.3f} "
            f"{row['wr_delta']:>+6.1f} {row['pf_delta']:>+6.2f}  {row['verdict']}"
        )

    useful = df_result[df_result["verdict"] == "USEFUL"]["engine"].tolist()
    logger.info(f"\n  → {len(useful)} engines USEFUL cho bước 3: {useful}")

    return df_result


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3 — WALK FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def step3_walk_forward(
    cluster: str,
    symbols: list[str],
    useful_engines: list[str],
) -> pd.DataFrame:
    """
    Walk Forward (expanding window) cho từng engine trong useful_engines.

    Mỗi fold:
      IS  = [WF_START .. train_end_wf]
      OOS = [train_end_wf .. test_end]

    WFE = OOS_exp_filtered / IS_exp_filtered
    Consistency = % folds OOS_exp_filtered > 0
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BƯỚC 3: Walk Forward — cluster={cluster}")
    logger.info(f"{'='*60}")

    if not useful_engines:
        logger.warning("  Không có engine nào pass bước 2")
        return pd.DataFrame()

    results = []

    for eng in useful_engines:
        logger.info(f"\n  Engine: {eng}")
        folds = []
        fold_start = WF_START

        while True:
            train_end_wf = fold_start + timedelta(days=WF_TRAIN_MONTHS * 30)
            test_end     = train_end_wf + timedelta(days=WF_TEST_MONTHS * 30)

            # Kiểm tra còn đủ data
            any_data = False
            for sym in symbols:
                df = load_symbol(sym)
                if df is not None:
                    last_date = pd.to_datetime(df["date"]).dt.date.values[-1]
                    if last_date >= test_end:
                        any_data = True
                        break
            if not any_data:
                break

            # Thu thập trades IS và OOS
            is_trades_base: list[Trade] = []
            is_trades_filt: list[Trade] = []
            oos_trades_base: list[Trade] = []
            oos_trades_filt: list[Trade] = []

            for sym in symbols:
                df = load_symbol(sym)
                if df is None:
                    continue

                # IS trades
                is_all = collect_trades_for_symbol(
                    sym, cluster, df,
                    window_start   = fold_start,
                    window_end     = train_end_wf,
                    train_end_date = train_end_wf,
                    with_vibe      = True,
                )
                is_trades_base.extend(is_all)
                is_trades_filt.extend(
                    [t for t in is_all if t.vibe_signals.get(eng) == 1])

                # OOS trades
                oos_all = collect_trades_for_symbol(
                    sym, cluster, df,
                    window_start   = train_end_wf,
                    window_end     = test_end,
                    train_end_date = train_end_wf,
                    with_vibe      = True,
                )
                oos_trades_base.extend(oos_all)
                oos_trades_filt.extend(
                    [t for t in oos_all if t.vibe_signals.get(eng) == 1])

            m_is_base  = _metrics(is_trades_base)
            m_is_filt  = _metrics(is_trades_filt)
            m_oos_base = _metrics(oos_trades_base)
            m_oos_filt = _metrics(oos_trades_filt)

            # WFE = OOS_exp_filtered / IS_exp_filtered
            wfe = (m_oos_filt["exp"] / m_is_filt["exp"]
                   if m_is_filt["exp"] > 0 else 0.0)

            fold_label = (f"{train_end_wf.strftime('%Y-%m')}"
                          f"→{test_end.strftime('%Y-%m')}")

            folds.append({
                "fold":          fold_label,
                "n_is_base":     m_is_base["n"],
                "n_is_filt":     m_is_filt["n"],
                "n_oos_base":    m_oos_base["n"],
                "n_oos_filt":    m_oos_filt["n"],
                "is_exp_filt":   m_is_filt["exp"],
                "oos_exp_base":  m_oos_base["exp"],
                "oos_exp_filt":  m_oos_filt["exp"],
                "oos_wr_filt":   m_oos_filt["wr"],
                "oos_pf_filt":   m_oos_filt["pf"],
                "wfe":           round(wfe, 3),
                "oos_positive":  m_oos_filt["exp"] > 0,
            })

            logger.info(
                f"    Fold {fold_label}: "
                f"IS_exp={m_is_filt['exp']:.3f}% | "
                f"OOS_exp(filtered)={m_oos_filt['exp']:.3f}% | "
                f"WFE={wfe:.2f} | n_oos={m_oos_filt['n']}"
            )

            fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)

        if len(folds) < WF_MIN_FOLDS:
            logger.warning(f"    Không đủ folds ({len(folds)} < {WF_MIN_FOLDS})")
            results.append({
                "engine": eng, "wf_status": "INSUFFICIENT_FOLDS",
                "avg_wfe": 0, "consistency": 0, "total_folds": len(folds),
                "avg_oos_exp": 0, "avg_oos_wr": 0, "folds": folds,
            })
            continue

        wfes         = [f["wfe"] for f in folds if f["wfe"] > 0]
        pos_folds    = sum(1 for f in folds if f["oos_positive"])
        neg_folds    = len(folds) - pos_folds
        avg_wfe      = float(np.mean(wfes)) if wfes else 0.0
        consistency  = round(pos_folds / len(folds) * 100, 1)
        avg_oos_exp  = round(float(np.mean([f["oos_exp_filt"] for f in folds])), 3)
        avg_oos_wr   = round(float(np.mean([f["oos_wr_filt"] for f in folds
                                            if f["n_oos_filt"] > 0])), 1)

        # Verdict
        if neg_folds > WF_MAX_NEG_FOLDS or avg_wfe < WF_MIN_WFE:
            wf_status = "OVERFIT"
        elif avg_oos_exp <= 0:
            wf_status = "NO_IMPROVEMENT"
        elif consistency >= 60:
            wf_status = "VALIDATED"
        else:
            wf_status = "MARGINAL"

        logger.info(
            f"    → WF_status={wf_status} | avg_WFE={avg_wfe:.2f} | "
            f"consistency={consistency}% | avg_OOS_exp={avg_oos_exp}%"
        )

        results.append({
            "engine":       eng,
            "wf_status":    wf_status,
            "avg_wfe":      round(avg_wfe, 3),
            "consistency":  consistency,
            "total_folds":  len(folds),
            "neg_folds":    neg_folds,
            "avg_oos_exp":  avg_oos_exp,
            "avg_oos_wr":   avg_oos_wr,
            "folds":        folds,
        })

    df_result = pd.DataFrame([
        {k: v for k, v in r.items() if k != "folds"}
        for r in results
    ]).sort_values("avg_oos_exp", ascending=False)

    logger.info(f"\n  {'Engine':<20} {'WF_Status':<16} {'WFE':>6} {'Cons%':>7} "
                f"{'OOS_exp':>8} {'OOS_wr':>7}")
    logger.info(f"  {'-'*70}")
    for _, row in df_result.iterrows():
        marker = "✓" if row["wf_status"] == "VALIDATED" else \
                 "~" if row["wf_status"] == "MARGINAL" else "✗"
        logger.info(
            f"  {marker} {row['engine']:<18} {row['wf_status']:<16} "
            f"{row['avg_wfe']:>6.2f} {row['consistency']:>6.0f}% "
            f"{row['avg_oos_exp']:>+8.3f}% {row['avg_oos_wr']:>6.1f}%"
        )

    return df_result, results


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4 — GENERATE INTEGRATION CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def step4_generate_config(
    cluster: str,
    step1_df: pd.DataFrame,
    step2_df: pd.DataFrame,
    step3_df: pd.DataFrame,
) -> dict:
    """
    Tổng hợp kết quả 3 bước → output config để integrate vào cluster_scanner.py.

    Config format:
      VIBE_FILTER_CONFIG = {
          "Mean Reversion": {
              "engines": ["SMC", "TechnicalBasic"],   # validated engines
              "agree_bonus":    0.20,   # +20% score nếu đồng thuận
              "disagree_bonus": -0.20,  # -20% score nếu ngược chiều
          },
          ...
      }
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BƯỚC 4: Generate Config — cluster={cluster}")
    logger.info(f"{'='*60}")

    if step3_df is None or len(step3_df) == 0:
        logger.warning("  Không có kết quả WF")
        return {}

    validated = step3_df[step3_df["wf_status"] == "VALIDATED"]["engine"].tolist()
    marginal  = step3_df[step3_df["wf_status"] == "MARGINAL"]["engine"].tolist()

    logger.info(f"  VALIDATED: {validated}")
    logger.info(f"  MARGINAL:  {marginal}")
    logger.info(f"  OVERFIT/NO_IMPROVEMENT: "
                f"{step3_df[~step3_df['wf_status'].isin(['VALIDATED','MARGINAL'])]['engine'].tolist()}")

    config = {
        "cluster":        cluster,
        "validated":      validated,
        "marginal":       marginal,
        "agree_bonus":    VIBE_AGREE_BONUS,
        "disagree_bonus": VIBE_DISAGREE_BONUS,
    }

    if validated:
        logger.info(f"\n  ✅ Gợi ý thêm vào cluster_scanner.py:")
        logger.info(f"""
  VIBE_FILTER_CONFIG["{cluster}"] = {{
      "engines":        {validated},
      "agree_bonus":    {VIBE_AGREE_BONUS},   # +20% score nếu vibe đồng thuận
      "disagree_bonus": {VIBE_DISAGREE_BONUS},  # -20% score nếu vibe ngược chiều
  }}

  # Trong _calc_score hoặc _format_signal:
  def _apply_vibe_bonus(score: float, cluster: str, vibe_result: dict) -> float:
      cfg = VIBE_FILTER_CONFIG.get(cluster, {{}})
      engines = cfg.get("engines", [])
      if not engines:
          return score
      votes = [vibe_result.get("signals", {{}}).get(e, 0) for e in engines]
      bull_votes = sum(1 for v in votes if v > 0)
      bear_votes = sum(1 for v in votes if v < 0)
      if bull_votes > len(engines) * 0.5:
          score *= (1 + cfg["agree_bonus"])     # đồng thuận
      elif bear_votes > len(engines) * 0.5:
          score *= (1 + cfg["disagree_bonus"])  # ngược chiều
      return score
        """)
    else:
        logger.info(f"  ⚠️ Không có engine nào VALIDATED cho {cluster}")
        logger.info(f"     Không nên thêm vibe filter cho cluster này")

    return config


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

CLUSTER_MAP = {
    "MR":  ("Mean Reversion", MR_SYMBOLS),
    "MOM": ("Momentum",       MOM_SYMBOLS),
    "BO":  ("Breakout",       BO_SYMBOLS),
}


def run_pipeline(cluster_key: str = "ALL", step_key: str = "ALL",
                 output_json: str = "vibe_filter_results.json") -> None:
    """
    Chạy pipeline đầy đủ hoặc từng bước.

    cluster_key: "MR" | "MOM" | "BO" | "ALL"
    step_key:    "1" | "2" | "3" | "ALL"
    """
    targets = (
        list(CLUSTER_MAP.keys()) if cluster_key == "ALL"
        else [cluster_key.upper()]
    )

    all_results = {}

    for ck in targets:
        if ck not in CLUSTER_MAP:
            logger.error(f"Unknown cluster key: {ck}. Dùng MR/MOM/BO/ALL")
            continue

        cluster, symbols = CLUSTER_MAP[ck]
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Cluster: {cluster} ({len(symbols)} mã)")
        logger.info(f"{'#'*60}")

        step1_result = pd.DataFrame()
        step2_result = pd.DataFrame()
        step3_result = pd.DataFrame()
        step3_raw    = []
        keep_engines  = []
        useful_engines= []

        # ── Bước 1
        if step_key in ("1", "ALL"):
            step1_result = step1_cohen_d(cluster, symbols)
            if not step1_result.empty:
                keep_engines = step1_result[
                    step1_result["status"] == "KEEP"]["engine"].tolist()

        # ── Bước 2
        if step_key in ("2", "ALL"):
            if not keep_engines and step_key == "ALL":
                logger.warning("  Bỏ qua bước 2 — không có engine nào pass bước 1")
            else:
                if step_key == "2":
                    # Nếu chạy bước 2 riêng, dùng tất cả engines
                    try:
                        from vibe_skills import _ENGINES
                        keep_engines = list(_ENGINES.keys())
                    except Exception:
                        keep_engines = []
                step2_result = step2_backtest_filters(cluster, symbols, keep_engines)
                if not step2_result.empty:
                    useful_engines = step2_result[
                        step2_result["verdict"] == "USEFUL"]["engine"].tolist()

        # ── Bước 3
        if step_key in ("3", "ALL"):
            if not useful_engines and step_key == "ALL":
                logger.warning("  Bỏ qua bước 3 — không có engine nào pass bước 2")
            else:
                if step_key == "3":
                    try:
                        from vibe_skills import _ENGINES
                        useful_engines = list(_ENGINES.keys())
                    except Exception:
                        useful_engines = []
                step3_result, step3_raw = step3_walk_forward(
                    cluster, symbols, useful_engines)

        # ── Bước 4 (chỉ khi chạy ALL hoặc step 3)
        if step_key in ("3", "ALL") and not step3_result.empty:
            config = step4_generate_config(
                cluster, step1_result, step2_result, step3_result)
        else:
            config = {}

        all_results[ck] = {
            "cluster":      cluster,
            "step1":        step1_result.to_dict("records") if not step1_result.empty else [],
            "step2":        step2_result.to_dict("records") if not step2_result.empty else [],
            "step3":        step3_raw,
            "config":       config,
        }

    # Save JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"\n✅ Kết quả lưu tại: {output_json}")

    # Print summary table
    logger.info(f"\n{'='*60}")
    logger.info("TỔNG KẾT")
    logger.info(f"{'='*60}")
    for ck, res in all_results.items():
        cfg = res.get("config", {})
        val = cfg.get("validated", [])
        mar = cfg.get("marginal", [])
        logger.info(f"  {res['cluster']:<20}: "
                    f"VALIDATED={val} | MARGINAL={mar}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest Vibe Agents làm confirmation filter cho MR/MOM/BO clusters"
    )
    parser.add_argument(
        "--cluster",
        default="ALL",
        choices=["MR", "MOM", "BO", "ALL"],
        help="Cluster cần test (default: ALL)",
    )
    parser.add_argument(
        "--step",
        default="ALL",
        choices=["1", "2", "3", "ALL"],
        help="Chạy từ bước nào (default: ALL)",
    )
    parser.add_argument(
        "--output",
        default="vibe_filter_results.json",
        help="File JSON output (default: vibe_filter_results.json)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run_pipeline(
        cluster_key=args.cluster,
        step_key=args.step,
        output_json=args.output,
    )
