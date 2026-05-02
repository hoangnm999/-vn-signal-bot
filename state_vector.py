"""
state_vector.py V2 — Vector trạng thái 11 chiều từ OHLCV.

Nguyên tắc thiết kế V2: mỗi dim đo 1 khía cạnh ĐỘC LẬP của market,
không overlap — để cosine similarity thực sự phân biệt được các ngày.

5 nhóm:
  MOMENTUM      : momentum_5d, momentum_20d, macd_hist_norm
  MEAN REVERSION: bb_position, rsi_slope
  TREND         : trend_slope, trend_consistency
  VOLATILITY    : atr_ratio, atr_trend
  VOLUME        : volume_spike, vol_trend, volume_price_confirm
  STRUCTURE     : range_position_60d

Tất cả normalize về [-1,1] hoặc [0,1].
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# Tên 11 dimensions — thứ tự CỐ ĐỊNH không đổi
VECTOR_DIMS = [
    # MOMENTUM (3)
    "momentum_5d",
    "momentum_20d",
    "macd_hist_norm",
    # MEAN REVERSION (2)
    "bb_position",
    "rsi_slope",
    # TREND (2)
    "trend_slope",
    "trend_consistency",
    # VOLATILITY (2)
    "atr_ratio",
    "atr_trend",
    # VOLUME (3)
    "volume_spike",
    "vol_trend",
    "volume_price_confirm",
    # STRUCTURE (1) — đo vị trí giá trong range thực tế 60 ngày
    "range_position_60d",
]

# Tổng 13 dims (3+2+2+2+3+1)
VECTOR_KEYS = VECTOR_DIMS
VECTOR_DIM  = len(VECTOR_DIMS)   # 13


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
    Trả về dict {dim: float, "_array": np.ndarray[13]}.
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
    i     = n - 1

    # ── Tính indicators ───────────────────────────────────────────────────────

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = _ema_arr(gain, 14)
    avg_l = _ema_arr(loss, 14)
    avg_l = np.where(avg_l == 0, 1e-9, avg_l)
    rsi_arr = 100 - 100 / (1 + avg_g / avg_l)

    # MACD histogram
    ema12  = _ema_arr(close, 12)
    ema26  = _ema_arr(close, 26)
    macd   = ema12 - ema26
    signal = _ema_arr(macd, 9)
    hist   = macd - signal

    # SMA
    sma20 = _sma_arr(close, 20)
    sma50 = _sma_arr(close, 50)

    # ATR(14)
    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low, np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr14  = _sma_arr(tr, 14)

    # Bollinger Bands(20)
    std20    = pd.Series(close).rolling(20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    # Volume MA
    vsma5  = _sma_arr(vol, 5)
    vsma20 = _sma_arr(vol, 20)

    # ── Lấy giá trị tại index i ───────────────────────────────────────────────
    px        = _safe(close[i], 1.0)
    atr_v     = _safe(atr14[i], px * 0.02)
    s20       = _safe(sma20[i], px)
    s50       = _safe(sma50[i], px)
    bb_u      = _safe(bb_upper[i], px * 1.04)
    bb_l      = _safe(bb_lower[i], px * 0.96)
    vol_v     = _safe(vol[i], 1.0)
    vs5       = _safe(vsma5[i], vol_v)
    vs20      = _safe(vsma20[i], vol_v)
    op_v      = _safe(opn[i], px)
    hist_v    = _safe(hist[i], 0.0)

    close_5d  = _safe(close[max(i - 5,  0)], px)
    close_20d = _safe(close[max(i - 20, 0)], px)
    close_60d = _safe(close[max(i - 60, 0)], px)

    # RSI hiện tại và 5 bars trước
    rsi_now  = _safe(rsi_arr[i], 50.0)
    rsi_5d   = _safe(rsi_arr[max(i - 5, 0)], 50.0)

    # ATR 10 bars trước (để đo ATR đang tăng hay giảm)
    atr_10d  = _safe(atr14[max(i - 10, 0)], atr_v)

    # High/Low 60 ngày thực tế
    hi60 = float(np.nanmax(high[max(i - 59, 0):i + 1]))
    lo60 = float(np.nanmin(low[max(i - 59,  0):i + 1]))

    # trend_consistency: % ngày close > sma20 trong 20 bars gần nhất
    # (cần sma20 hợp lệ — không NaN)
    if i >= 19:
        window_close = close[i - 19:i + 1]        # 20 bars
        window_sma20 = sma20[i - 19:i + 1]
        valid_mask   = np.isfinite(window_sma20)
        if valid_mask.sum() > 0:
            above = np.sum((window_close > window_sma20) & valid_mask)
            trend_cons_raw = above / valid_mask.sum()
        else:
            trend_cons_raw = 0.5
    else:
        trend_cons_raw = 0.5

    # ── Normalize ─────────────────────────────────────────────────────────────
    def clip11(v, scale=1.0):
        """Normalize về [-1, 1]."""
        return float(max(-1.0, min(1.0, v / scale)))

    def clip01(v, lo, hi):
        """Normalize về [0, 1]."""
        if hi == lo:
            return 0.5
        return float(max(0.0, min(1.0, (v - lo) / (hi - lo))))

    # --- MOMENTUM ---
    momentum_5d        = clip11(_safe((px / (close_5d + 1e-9)) - 1.0), 0.10)
    momentum_20d       = clip11(_safe((px / (close_20d + 1e-9)) - 1.0), 0.30)
    macd_hist_norm     = clip11(_safe(hist_v / (atr_v + 1e-9)), 2.0)

    # --- MEAN REVERSION ---
    bb_range           = bb_u - bb_l
    bb_position        = clip01(px, bb_l, bb_u) if bb_range > 0 else 0.5
    # rsi_slope: RSI đang phục hồi (+) hay tiếp tục giảm (-)
    # Scale: ±20 RSI points → [-1, 1]
    rsi_slope          = clip11(rsi_now - rsi_5d, 20.0)

    # --- TREND ---
    trend_slope        = clip11(_safe((s20 - s50) / (px + 1e-9)), 0.10)
    # trend_consistency: [0,1], 0.5 = sideways, 1.0 = strong uptrend, 0.0 = downtrend
    trend_consistency  = float(trend_cons_raw)

    # --- VOLATILITY ---
    atr_ratio          = clip01(_safe(atr_v / (px + 1e-9)), 0.0, 0.05)
    # atr_trend: ATR đang tăng (>1) hay giảm (<1) so với 10 bars trước
    # Scale: [0.5, 2.0] → normalize về [0, 1]
    atr_ratio_now      = _safe(atr_v / (atr_10d + 1e-9), 1.0)
    atr_trend          = clip01(atr_ratio_now, 0.5, 2.0)

    # --- VOLUME ---
    volume_spike       = clip11(_safe((vol_v / (vs20 + 1e-9)) - 1.0), 2.0)
    vol_trend          = clip11(_safe((vs5 / (vs20 + 1e-9)) - 1.0), 0.50)
    # volume_price_confirm: volume lớn + nến xanh = +, volume lớn + nến đỏ = -
    candle_dir         = 1.0 if px > op_v else (-1.0 if px < op_v else 0.0)
    raw_vpc            = candle_dir * _safe((vol_v / (vs20 + 1e-9)) - 1.0, 0.0)
    volume_price_confirm = clip11(raw_vpc, 2.0)

    # --- STRUCTURE ---
    range_position_60d = clip01(px, lo60, hi60) if hi60 > lo60 else 0.5

    values = [
        momentum_5d, momentum_20d, macd_hist_norm,
        bb_position, rsi_slope,
        trend_slope, trend_consistency,
        atr_ratio, atr_trend,
        volume_spike, vol_trend, volume_price_confirm,
        range_position_60d,
    ]

    vec = {dim: float(v) for dim, v in zip(VECTOR_DIMS, values)}
    vec["_array"] = np.array(values, dtype=np.float32)
    return vec


def compute_state_vector_for_date(df: pd.DataFrame, idx: int, **_kwargs) -> Optional[dict]:
    """
    Tính vector cho ngày tại index idx trong df.
    Dùng subset df[:idx+1] để simulate 'chỉ biết đến ngày idx'.
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
