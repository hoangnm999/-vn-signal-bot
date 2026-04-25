"""
state_vector.py — Vector trạng thái 15 chiều từ OHLCV.

Tất cả dimensions đã chuẩn hóa về [-1,1] hoặc [0,1]
để cosine similarity có ý nghĩa thực tế.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# Tên 15 dimensions — thứ tự CỐ ĐỊNH không đổi
VECTOR_DIMS = [
    "rsi_norm", "macd_hist_norm", "bb_position", "volume_spike",
    "trend_slope", "price_vs_sma20", "price_vs_sma50", "atr_ratio",
    "stoch_k_norm", "ema_cross", "momentum_5d", "momentum_20d",
    "high_low_pos", "vol_trend", "candle_body",
]
# Alias cho historical_analog.py
VECTOR_KEYS = VECTOR_DIMS
VECTOR_DIM  = len(VECTOR_DIMS)  # 15


def _safe(v, default=0.0):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _ema_arr(c: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(c).ewm(span=span, adjust=False).mean().values


def _sma_arr(c: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(c).rolling(p, min_periods=p).mean().values


def compute_state_vector_from_df(df: pd.DataFrame) -> Optional[dict]:
    """
    Tính vector từ DataFrame OHLCV (cần >= 60 bars).
    Trả về dict {dim: float, "_array": np.ndarray[15]}.
    """
    if df is None or len(df) < 60:
        return None

    df    = df.reset_index(drop=True)
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    vol   = df["volume"].values.astype(float)
    opn   = df["open"].values.astype(float)
    n     = len(df)
    i     = n - 1   # index ngày cuối

    # ── Tính indicators ───────────────────────────────────────────────────────
    # RSI(14) với EWM
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = _ema_arr(gain, 14)
    avg_l = _ema_arr(loss, 14)
    avg_l = np.where(avg_l == 0, 1e-9, avg_l)
    rsi_arr = 100 - 100 / (1 + avg_g / avg_l)

    # MACD histogram
    ema12   = _ema_arr(close, 12)
    ema26   = _ema_arr(close, 26)
    macd    = ema12 - ema26
    signal  = _ema_arr(macd, 9)
    hist    = macd - signal

    # SMA
    sma20 = _sma_arr(close, 20)
    sma50 = _sma_arr(close, 50)

    # ATR
    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low, np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr    = _sma_arr(tr, 14)

    # Bollinger Bands
    std20    = pd.Series(close).rolling(20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    # Stochastic K(14)
    lo14   = pd.Series(low).rolling(14).min().values
    hi14   = pd.Series(high).rolling(14).max().values
    denom  = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch  = 100 * (close - lo14) / denom

    # Volume MA
    vsma5  = _sma_arr(vol, 5)
    vsma20 = _sma_arr(vol, 20)

    # ── Lấy giá trị tại index i ───────────────────────────────────────────────
    px       = _safe(close[i], 1.0)
    rsi_v    = _safe(rsi_arr[i], 50.0)
    hist_v   = _safe(hist[i], 0.0)
    atr_v    = _safe(atr[i], px * 0.02)
    bb_u     = _safe(bb_upper[i], px * 1.04)
    bb_l     = _safe(bb_lower[i], px * 0.96)
    s20      = _safe(sma20[i], px)
    s50      = _safe(sma50[i], px)
    stk_v    = _safe(stoch[i], 50.0)
    e12      = _safe(ema12[i], px)
    e26      = _safe(ema26[i], px)
    vol_v    = _safe(vol[i], 1.0)
    vs5      = _safe(vsma5[i], vol_v)
    vs20     = _safe(vsma20[i], vol_v)
    op_v     = _safe(opn[i], px)
    close_5d = _safe(close[max(i - 5,  0)], px)
    close_20d= _safe(close[max(i - 20, 0)], px)
    hi20     = float(np.nanmax(high[max(i - 19, 0):i + 1]))
    lo20     = float(np.nanmin(low[max(i - 19,  0):i + 1]))

    # ── Chuẩn hóa ─────────────────────────────────────────────────────────────
    def clip11(v, scale=1.0): return max(-1.0, min(1.0, v / scale))
    def clip01(v, lo, hi):
        if hi == lo: return 0.5
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))

    rsi_norm       = clip11((rsi_v - 50.0) / 50.0)
    macd_hist_norm = clip11(_safe(hist_v / (atr_v + 1e-9)), 2.0)
    bb_range       = bb_u - bb_l
    bb_position    = clip01(px, bb_l, bb_u) if bb_range > 0 else 0.5
    volume_spike   = clip11(_safe((vol_v / (vs20 + 1e-9)) - 1.0), 2.0)
    trend_slope    = clip11(_safe((s20 - s50) / (px + 1e-9)), 0.10)
    price_vs_sma20 = clip11(_safe((px - s20) / (px + 1e-9)), 0.15)
    price_vs_sma50 = clip11(_safe((px - s50) / (px + 1e-9)), 0.20)
    atr_ratio      = clip01(_safe(atr_v / (px + 1e-9)), 0.0, 0.05)
    stoch_k_norm   = clip11((stk_v - 50.0) / 50.0)
    ema_cross      = clip11(_safe((e12 - e26) / (px + 1e-9)), 0.05)
    momentum_5d    = clip11(_safe((px / (close_5d + 1e-9)) - 1.0), 0.10)
    momentum_20d   = clip11(_safe((px / (close_20d + 1e-9)) - 1.0), 0.30)
    high_low_pos   = clip01(px, lo20, hi20) if hi20 > lo20 else 0.5
    vol_trend      = clip11(_safe((vs5 / (vs20 + 1e-9)) - 1.0), 0.50)
    candle_body    = clip01(_safe(abs(px - op_v) / (atr_v + 1e-9)), 0.0, 1.5)

    values = [
        rsi_norm, macd_hist_norm, bb_position, volume_spike,
        trend_slope, price_vs_sma20, price_vs_sma50, atr_ratio,
        stoch_k_norm, ema_cross, momentum_5d, momentum_20d,
        high_low_pos, vol_trend, candle_body,
    ]
    vec = {dim: float(v) for dim, v in zip(VECTOR_DIMS, values)}
    vec["_array"] = np.array(values, dtype=np.float32)
    return vec


def compute_state_vector_for_date(df: pd.DataFrame, idx: int, **_kwargs) -> Optional[dict]:
    """
    Tính vector cho ngày tại index idx trong df.
    Dùng subset df[:idx+1] để simulate "chỉ biết đến ngày idx".
    """
    if idx < 59:
        return None
    sub = df.iloc[max(0, idx - 299):idx + 1].copy()
    return compute_state_vector_from_df(sub)


def compute_state_vector(symbol: str, days: int = 100) -> Optional[dict]:
    """Tính vector cho mã symbol, tự load data từ vn_loader."""
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=max(days, 80), min_bars=60)
        return compute_state_vector_from_df(df)
    except Exception as e:
        logger.warning(f"compute_state_vector({symbol}): {e}")
        return None


def vector_to_list(vec: dict) -> list:
    return [float(vec.get(d, 0.0)) for d in VECTOR_DIMS]


def list_to_dict(values: list) -> dict:
    d = {dim: float(v) for dim, v in zip(VECTOR_DIMS, values)}
    d["_array"] = np.array(values, dtype=np.float32)
    return d


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
