"""
wave_pattern.py — Phân tích Sóng Lịch Sử (Wave Pattern Analysis).

Triết lý:
  Thay vì tìm "ngày giống hôm nay" (analog), module này quét TOÀN BỘ
  lịch sử giá, xác định tất cả các con sóng tăng/giảm lớn, lấy dấu
  vân tay (state vector 15D) tại điểm BẮT ĐẦU mỗi sóng, rồi so sánh
  trạng thái hiện tại với 2 nhóm đó.

  Câu hỏi trả lời: "Hiện tại giống bối cảnh trước sóng tăng hay sóng giảm?"

Pipeline:
  1. ZigZag(min_pct=10%) → danh sách đỉnh/đáy cục bộ
  2. Lọc sóng đủ biên độ (top 30% động)
  3. Tính vector tại điểm bắt đầu mỗi sóng (không lookahead bias)
  4. Tính phân phối (median, P25, P75) của 15 dimensions cho 2 nhóm
  5. Z-score membership: hiện tại nằm ở đâu trong phân phối mỗi nhóm
  6. Cache kết quả vào data/{SYMBOL}_waves.json

Cache:
  data/{SYMBOL}_waves.json — lưu wave_up_vectors + wave_down_vectors
  Rebuild khi: file không tồn tại, hoặc data cũ hơn 7 ngày.

Command:
  /wave <MÃ> [--rebuild]
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from state_vector import (
    VECTOR_KEYS, VECTOR_DIM,
    compute_state_vector_for_date,
    compute_state_vector_from_df,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR          = pathlib.Path("data")
WAVE_CACHE_SUFFIX = "_waves.json"
CACHE_MAX_AGE_DAYS = 7        # rebuild nếu cache > 7 ngày
ZIGZAG_MIN_PCT    = 10.0      # % — bước tối thiểu để tính đỉnh/đáy ZigZag
WAVE_TOP_PCT      = 30        # lấy top 30% sóng lớn nhất mỗi chiều
MIN_WAVES         = 5         # tối thiểu sóng mỗi nhóm để phân tích
MIN_BARS_REQUIRED = 200       # tối thiểu bars lịch sử
LOAD_DAYS         = 1500      # ~6 năm

# Labels thân thiện cho 15 dimensions
DIM_LABEL = {
    "rsi_norm":        "RSI",
    "macd_hist_norm":  "MACD Hist",
    "bb_position":     "BB Position",
    "volume_spike":    "Volume Spike",
    "trend_slope":     "Trend (SMA20/50)",
    "price_vs_sma20":  "Gia vs SMA20",
    "price_vs_sma50":  "Gia vs SMA50",
    "atr_ratio":       "ATR Ratio",
    "stoch_k_norm":    "Stoch K",
    "ema_cross":       "EMA Cross",
    "momentum_5d":     "Momentum 5D",
    "momentum_20d":    "Momentum 20D",
    "high_low_pos":    "Vi tri 20D H/L",
    "vol_trend":       "Volume Trend",
    "candle_body":     "Than nen",
}

# Mô tả ngắn khi dimension cao/thấp — dùng trong summary
DIM_HIGH = {
    "rsi_norm":        "RSI trung tinh/thap",
    "macd_hist_norm":  "MACD Hist duong",
    "bb_position":     "gia giua BB",
    "volume_spike":    "volume tang dot bien",
    "trend_slope":     "SMA20 cat len SMA50",
    "price_vs_sma20":  "gia tren SMA20",
    "price_vs_sma50":  "gia tren SMA50",
    "atr_ratio":       "bien dong cao",
    "stoch_k_norm":    "Stoch K tang",
    "ema_cross":       "EMA12 vuot EMA26",
    "momentum_5d":     "tang 5 phien gan",
    "momentum_20d":    "xu huong tang 20D",
    "high_low_pos":    "gia phan hoi tu day",
    "vol_trend":       "vol ngan han tang",
    "candle_body":     "than nen day",
}
DIM_LOW = {
    "rsi_norm":        "RSI cao (overbought)",
    "macd_hist_norm":  "MACD Hist am/cham day",
    "bb_position":     "gia sat BB upper",
    "volume_spike":    "volume suy yeu",
    "trend_slope":     "SMA20 duoi SMA50",
    "price_vs_sma20":  "gia duoi SMA20",
    "price_vs_sma50":  "gia duoi SMA50",
    "atr_ratio":       "bien dong thap",
    "stoch_k_norm":    "Stoch K giam",
    "ema_cross":       "EMA12 duoi EMA26",
    "momentum_5d":     "giam 5 phien gan",
    "momentum_20d":    "xu huong giam 20D",
    "high_low_pos":    "gia o vung giua",
    "vol_trend":       "vol on dinh/giam",
    "candle_body":     "than nen nho (do du)",
}


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1 — ZIGZAG ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def _zigzag(close: np.ndarray, min_pct: float = ZIGZAG_MIN_PCT) -> list[tuple[int, float, str]]:
    """
    ZigZag algorithm — xác định đỉnh/đáy cục bộ có biên độ >= min_pct%.

    Thuật toán:
      - Bắt đầu từ điểm đầu tiên, theo dõi hướng hiện tại (UP/DOWN).
      - Khi giá di chuyển ngược hướng >= min_pct% so với đỉnh/đáy gần nhất
        → xác nhận đỉnh/đáy, đổi hướng.

    Returns:
        list of (index, price, type) — type là "TOP" hoặc "BOT" (bottom).
        Luôn bắt đầu bằng điểm đầu tiên, kết thúc bằng điểm cuối.
    """
    n = len(close)
    if n < 10:
        return []

    pivots: list[tuple[int, float, str]] = []

    # Xác định hướng ban đầu
    direction = "UP"   # giả sử bắt đầu đi lên
    last_pivot_idx   = 0
    last_pivot_price = close[0]
    pivots.append((0, close[0], "BOT"))

    for i in range(1, n):
        p = close[i]

        if direction == "UP":
            if p > last_pivot_price:
                # Tiếp tục tăng → cập nhật đỉnh hiện tại
                last_pivot_idx   = i
                last_pivot_price = p
            elif last_pivot_price > 0 and (last_pivot_price - p) / last_pivot_price * 100 >= min_pct:
                # Giảm đủ min_pct% → xác nhận đỉnh, đổi hướng
                # Cập nhật pivot cuối thành đỉnh thực sự
                if pivots and pivots[-1][2] == "BOT":
                    pivots.append((last_pivot_idx, last_pivot_price, "TOP"))
                else:
                    pivots[-1] = (last_pivot_idx, last_pivot_price, "TOP")
                direction        = "DOWN"
                last_pivot_idx   = i
                last_pivot_price = p

        else:  # direction == "DOWN"
            if p < last_pivot_price:
                # Tiếp tục giảm → cập nhật đáy hiện tại
                last_pivot_idx   = i
                last_pivot_price = p
            elif last_pivot_price > 0 and (p - last_pivot_price) / last_pivot_price * 100 >= min_pct:
                # Tăng đủ min_pct% → xác nhận đáy, đổi hướng
                if pivots and pivots[-1][2] == "TOP":
                    pivots.append((last_pivot_idx, last_pivot_price, "BOT"))
                else:
                    pivots[-1] = (last_pivot_idx, last_pivot_price, "BOT")
                direction        = "UP"
                last_pivot_idx   = i
                last_pivot_price = p

    # Thêm điểm cuối nếu chưa có
    if not pivots or pivots[-1][0] != n - 1:
        last_type = "TOP" if direction == "UP" else "BOT"
        pivots.append((n - 1, close[n - 1], last_type))

    return pivots


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2 — EXTRACT WAVES
# ══════════════════════════════════════════════════════════════════════════════

def _extract_waves(
    pivots: list[tuple[int, float, str]],
    wave_top_pct: int = WAVE_TOP_PCT,
) -> tuple[list[dict], list[dict]]:
    """
    Từ danh sách pivots (đỉnh/đáy ZigZag), trích xuất các con sóng.

    Sóng tăng: BOT → TOP (amplitude = (top - bot) / bot * 100)
    Sóng giảm: TOP → BOT (amplitude = (top - bot) / top * 100)

    Lọc top wave_top_pct% biên độ lớn nhất cho mỗi chiều.

    Returns:
        (waves_up, waves_down) — mỗi phần tử là dict:
        {
            "start_idx":  int,    # index trong df gốc của điểm bắt đầu
            "end_idx":    int,
            "start_price": float,
            "end_price":   float,
            "amplitude":  float,  # % tăng/giảm
            "duration":   int,    # số ngày
            "type":       "UP" | "DOWN",
        }
    """
    waves_up:   list[dict] = []
    waves_down: list[dict] = []

    for i in range(len(pivots) - 1):
        p1_idx, p1_price, p1_type = pivots[i]
        p2_idx, p2_price, p2_type = pivots[i + 1]

        if p1_price <= 0:
            continue

        if p1_type == "BOT" and p2_type == "TOP":
            amp = (p2_price - p1_price) / p1_price * 100
            waves_up.append({
                "start_idx":   p1_idx,
                "end_idx":     p2_idx,
                "start_price": p1_price,
                "end_price":   p2_price,
                "amplitude":   round(amp, 2),
                "duration":    p2_idx - p1_idx,
                "type":        "UP",
            })
        elif p1_type == "TOP" and p2_type == "BOT":
            amp = (p1_price - p2_price) / p1_price * 100
            waves_down.append({
                "start_idx":   p1_idx,
                "end_idx":     p2_idx,
                "start_price": p1_price,
                "end_price":   p2_price,
                "amplitude":   round(amp, 2),
                "duration":    p2_idx - p1_idx,
                "type":        "DOWN",
            })

    # Lọc top wave_top_pct% theo biên độ — ngưỡng động
    def _filter_top(waves: list[dict]) -> list[dict]:
        if not waves:
            return []
        amps = [w["amplitude"] for w in waves]
        thresh = float(np.percentile(amps, 100 - wave_top_pct))
        return [w for w in waves if w["amplitude"] >= thresh]

    return _filter_top(waves_up), _filter_top(waves_down)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3 — TÍNH VECTOR TẠI ĐIỂM BẮT ĐẦU SÓNG
# ══════════════════════════════════════════════════════════════════════════════

def _compute_wave_vectors(
    df:    pd.DataFrame,
    waves: list[dict],
) -> list[np.ndarray]:
    """
    Tính state vector 15D tại start_idx của mỗi sóng.
    Dùng compute_state_vector_for_date → không lookahead bias.

    Returns:
        list of np.ndarray shape (15,) — chỉ những sóng tính được vector.
    """
    vectors = []
    for w in waves:
        idx = w["start_idx"]
        if idx < 59:   # cần ít nhất 60 bars trước
            continue
        vec = compute_state_vector_for_date(df, idx)
        if vec is None:
            continue
        arr = vec.get("_array")
        if arr is not None and len(arr) == VECTOR_DIM:
            vectors.append(arr.astype(np.float32))
    return vectors


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4 — PHÂN PHỐI VÀ Z-SCORE MEMBERSHIP
# ══════════════════════════════════════════════════════════════════════════════

def _group_distribution(vectors: list[np.ndarray]) -> dict:
    """
    Tính phân phối của 15 dimensions cho một nhóm vectors.

    Returns dict:
    {
        dim_name: {
            "median": float, "mean": float,
            "p25": float, "p75": float,
            "min": float, "max": float, "std": float,
        },
        ...
    }
    """
    if not vectors:
        return {}

    mat = np.stack(vectors, axis=0)   # shape (N, 15)
    dist = {}
    for j, dim in enumerate(VECTOR_KEYS):
        col = mat[:, j]
        dist[dim] = {
            "median": float(np.median(col)),
            "mean":   float(np.mean(col)),
            "p25":    float(np.percentile(col, 25)),
            "p75":    float(np.percentile(col, 75)),
            "min":    float(np.min(col)),
            "max":    float(np.max(col)),
            "std":    float(np.std(col, ddof=1)) if len(col) > 1 else 0.0,
        }
    return dist


def _zscore_membership(
    current_vec: np.ndarray,
    dist:        dict,
) -> tuple[float, list[tuple[str, float]]]:
    """
    Tính mức độ "thuộc về" nhóm của vector hiện tại dựa trên z-score.

    Logic:
    - Với mỗi dimension, tính z-score của current_vec so với phân phối nhóm.
    - |z| <= 1.0 → nằm trong 1 std → "điển hình" → đóng góp +1 điểm
    - |z| <= 0.5 → rất điển hình → đóng góp +1.5 điểm
    - |z| > 2.0 → ngoại lệ → đóng góp -0.5 điểm
    - Tổng điểm / (15 * 1.5) → normalize về [0, 1]

    Returns:
        (score_0_to_1, [(dim, z_score), ...] sorted by |z| asc — các dim điển hình nhất đầu)
    """
    if not dist:
        return 0.0, []

    total_score = 0.0
    dim_zscores = []

    for j, dim in enumerate(VECTOR_KEYS):
        if dim not in dist:
            continue
        d     = dist[dim]
        std   = d["std"] if d["std"] > 1e-6 else 0.01
        z     = (float(current_vec[j]) - d["mean"]) / std
        dim_zscores.append((dim, round(z, 2)))

        if abs(z) <= 0.5:
            total_score += 1.5
        elif abs(z) <= 1.0:
            total_score += 1.0
        elif abs(z) <= 1.5:
            total_score += 0.5
        elif abs(z) > 2.0:
            total_score -= 0.5

    max_score = len(VECTOR_KEYS) * 1.5
    score = max(0.0, min(1.0, total_score / max_score))

    # Sort: |z| nhỏ nhất (điển hình nhất) lên đầu
    dim_zscores.sort(key=lambda x: abs(x[1]))

    return round(score, 3), dim_zscores


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 5 — TOP DISCRIMINANT DIMENSIONS
# ══════════════════════════════════════════════════════════════════════════════

def _top_discriminant_dims(
    dist_up:   dict,
    dist_down: dict,
    top_n:     int = 5,
) -> list[tuple[str, float, float, float]]:
    """
    Tìm top_n dimensions phân biệt rõ nhất giữa sóng tăng và sóng giảm.

    Tiêu chí: |median_up - median_down| / (std_up + std_down + 1e-6)
    → Effect size — lớn nghĩa là 2 nhóm tách biệt rõ.

    Returns:
        list of (dim, median_up, median_down, effect_size) sorted desc.
    """
    scores = []
    for dim in VECTOR_KEYS:
        if dim not in dist_up or dim not in dist_down:
            continue
        mu = dist_up[dim]["median"]
        md = dist_down[dim]["median"]
        su = dist_up[dim]["std"]
        sd = dist_down[dim]["std"]
        effect = abs(mu - md) / (su + sd + 1e-6)
        scores.append((dim, mu, md, round(effect, 3)))

    scores.sort(key=lambda x: -x[3])
    return scores[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(symbol: str) -> pathlib.Path:
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / f"{symbol.upper()}{WAVE_CACHE_SUFFIX}"


def _load_cache(symbol: str) -> Optional[dict]:
    """Load cache nếu tồn tại và còn mới (< CACHE_MAX_AGE_DAYS ngày)."""
    p = _cache_path(symbol)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Kiểm tra tuổi cache
        built = datetime.fromisoformat(data.get("built_at", "2000-01-01"))
        age   = (datetime.now() - built).total_seconds() / 86400
        if age > CACHE_MAX_AGE_DAYS:
            logger.info(f"wave cache {symbol}: {age:.1f} ngay → rebuild")
            return None
        return data
    except Exception as e:
        logger.warning(f"wave cache load fail {symbol}: {e}")
        return None


def _save_cache(symbol: str, data: dict):
    """Lưu cache."""
    try:
        p = _cache_path(symbol)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"wave cache saved: {p}")
    except Exception as e:
        logger.warning(f"wave cache save fail {symbol}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_wave_cache(symbol: str, force: bool = False) -> tuple[bool, str]:
    """
    Build toàn bộ wave analysis cho symbol.

    Returns (success, message).
    """
    symbol = symbol.upper()
    t0     = time.time()

    if not force:
        cached = _load_cache(symbol)
        if cached:
            return True, f"Cache hit: {symbol} ({cached.get('n_up',0)} up, {cached.get('n_down',0)} down waves)"

    # Load OHLCV
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=LOAD_DAYS, min_bars=MIN_BARS_REQUIRED)
    except Exception as e:
        return False, f"Khong load duoc du lieu {symbol}: {e}"

    n_bars = len(df)
    close  = df["close"].values.astype(float)

    # ZigZag
    pivots = _zigzag(close, min_pct=ZIGZAG_MIN_PCT)
    if len(pivots) < 6:
        return False, f"Khong du pivots ({len(pivots)}) — lich su qua ngan"

    # Extract waves
    waves_up, waves_down = _extract_waves(pivots, wave_top_pct=WAVE_TOP_PCT)

    if len(waves_up) < MIN_WAVES:
        return False, f"Chi co {len(waves_up)} song tang (can >= {MIN_WAVES})"
    if len(waves_down) < MIN_WAVES:
        return False, f"Chi co {len(waves_down)} song giam (can >= {MIN_WAVES})"

    # Compute vectors tại điểm bắt đầu
    vecs_up   = _compute_wave_vectors(df, waves_up)
    vecs_down = _compute_wave_vectors(df, waves_down)

    if len(vecs_up) < MIN_WAVES:
        return False, f"Chi tinh duoc {len(vecs_up)} vector song tang"
    if len(vecs_down) < MIN_WAVES:
        return False, f"Chi tinh duoc {len(vecs_down)} vector song giam"

    # Tính phân phối
    dist_up   = _group_distribution(vecs_up)
    dist_down = _group_distribution(vecs_down)

    # Tính stats cho waves
    amp_up_mean   = float(np.mean([w["amplitude"] for w in waves_up]))
    amp_down_mean = float(np.mean([w["amplitude"] for w in waves_down]))
    dur_up_mean   = float(np.mean([w["duration"]  for w in waves_up]))
    dur_down_mean = float(np.mean([w["duration"]  for w in waves_down]))

    cache_data = {
        "symbol":      symbol,
        "built_at":    datetime.now().isoformat(),
        "n_bars":      n_bars,
        "n_up":        len(vecs_up),
        "n_down":      len(vecs_down),
        "n_up_total":  len(waves_up),
        "n_down_total":len(waves_down),
        "amp_up_mean":  round(amp_up_mean, 1),
        "amp_down_mean":round(amp_down_mean, 1),
        "dur_up_mean":  round(dur_up_mean, 1),
        "dur_down_mean":round(dur_down_mean, 1),
        "zigzag_pct":  ZIGZAG_MIN_PCT,
        "wave_top_pct":WAVE_TOP_PCT,
        "dist_up":     dist_up,
        "dist_down":   dist_down,
        # Lưu vectors để so sánh sau nếu cần
        "vecs_up":     [v.tolist() for v in vecs_up],
        "vecs_down":   [v.tolist() for v in vecs_down],
    }
    _save_cache(symbol, cache_data)

    elapsed = round(time.time() - t0, 1)
    return True, (
        f"OK: {symbol} | {n_bars} bars | "
        f"Song tang: {len(vecs_up)} ({amp_up_mean:+.1f}%) | "
        f"Song giam: {len(vecs_down)} (-{amp_down_mean:.1f}%) | "
        f"{elapsed}s"
    )


def analyze_wave(symbol: str, force_rebuild: bool = False) -> dict:
    """
    Phân tích sóng cho symbol và so sánh với vector hiện tại.

    Returns dict:
    {
        "ok":          bool,
        "error":       str | None,
        "symbol":      str,
        "n_bars":      int,
        "n_up":        int,    # số sóng tăng có vector
        "n_down":      int,
        "amp_up_mean":  float,
        "amp_down_mean":float,
        "dur_up_mean":  float,
        "dur_down_mean":float,
        "score_up":    float,  # 0-1: mức độ giống bối cảnh sóng tăng
        "score_down":  float,
        "verdict":     str,    # "SONG TANG" | "SONG GIAM" | "KHONG RO"
        "confidence":  float,  # |score_up - score_down|
        "base_rate_up":   float,  # % ngày lịch sử có sóng tăng bắt đầu
        "base_rate_down": float,
        "avg_interval_up":   float,  # TB cứ bao nhiêu ngày/1 sóng tăng
        "avg_interval_down": float,
        "top_dims":    list,   # top 5 dimensions phân biệt
        "dist_up":     dict,
        "dist_down":   dict,
        "current_vec": list,   # vector hiện tại
        "dim_z_up":    list,   # [(dim, z)] sorted by |z| asc
        "dim_z_down":  list,
    }
    """
    symbol = symbol.upper()
    result_base = {
        "ok": False, "error": None, "symbol": symbol,
        "n_bars": 0, "n_up": 0, "n_down": 0,
        "amp_up_mean": 0.0, "amp_down_mean": 0.0,
        "dur_up_mean": 0.0, "dur_down_mean": 0.0,
        "score_up": 0.0, "score_down": 0.0,
        "verdict": "KHONG RO", "confidence": 0.0,
        "base_rate_up": 0.0, "base_rate_down": 0.0,
        "avg_interval_up": 0.0, "avg_interval_down": 0.0,
        "top_dims": [], "dist_up": {}, "dist_down": {},
        "current_vec": [], "dim_z_up": [], "dim_z_down": [],
    }

    # Build/load cache
    ok, msg = build_wave_cache(symbol, force=force_rebuild)
    if not ok:
        result_base["error"] = msg
        return result_base

    cache = _load_cache(symbol)
    if not cache:
        result_base["error"] = "Khong load duoc cache sau khi build"
        return result_base

    dist_up   = cache["dist_up"]
    dist_down = cache["dist_down"]

    # Vector hiện tại
    try:
        from vn_loader import load_vn_ohlcv
        df_now = load_vn_ohlcv(symbol, days=120, min_bars=60)
        cur_vec_dict = compute_state_vector_from_df(df_now)
        if cur_vec_dict is None:
            result_base["error"] = "Khong tinh duoc vector hien tai"
            return result_base
        cur_arr = cur_vec_dict["_array"]
    except Exception as e:
        result_base["error"] = f"Load du lieu hien tai that bai: {e}"
        return result_base

    # Z-score membership
    score_up,   dim_z_up   = _zscore_membership(cur_arr, dist_up)
    score_down, dim_z_down = _zscore_membership(cur_arr, dist_down)

    # Verdict
    diff = score_up - score_down
    if diff >= 0.08:
        verdict = "SONG TANG"
    elif diff <= -0.08:
        verdict = "SONG GIAM"
    else:
        verdict = "KHONG RO"

    # Top discriminant dims
    top_dims = _top_discriminant_dims(dist_up, dist_down)

    # Base rate — tỷ lệ thực tế trong lịch sử, không liên quan đến score
    # n_up_total / n_bars = bao nhiêu % ngày có sóng tăng bắt đầu
    # avg_interval = trung bình cứ bao nhiêu ngày có 1 sóng
    n_bars     = cache["n_bars"]
    n_up_total = cache.get("n_up_total", cache["n_up"])
    n_dn_total = cache.get("n_down_total", cache["n_down"])
    base_rate_up   = round(n_up_total / n_bars * 100, 2) if n_bars > 0 else 0.0
    base_rate_down = round(n_dn_total / n_bars * 100, 2) if n_bars > 0 else 0.0
    avg_interval_up   = round(n_bars / n_up_total,  1) if n_up_total  > 0 else 0.0
    avg_interval_down = round(n_bars / n_dn_total, 1) if n_dn_total > 0 else 0.0

    result_base.update({
        "ok":               True,
        "n_bars":           cache["n_bars"],
        "n_up":             cache["n_up"],
        "n_down":           cache["n_down"],
        "n_up_total":       n_up_total,
        "n_down_total":     n_dn_total,
        "amp_up_mean":      cache["amp_up_mean"],
        "amp_down_mean":    cache["amp_down_mean"],
        "dur_up_mean":      cache["dur_up_mean"],
        "dur_down_mean":    cache["dur_down_mean"],
        "zigzag_pct":       cache.get("zigzag_pct", ZIGZAG_MIN_PCT),
        "score_up":         score_up,
        "score_down":       score_down,
        "verdict":          verdict,
        "confidence":       round(abs(diff), 3),
        "top_dims":         top_dims,
        "dist_up":          dist_up,
        "dist_down":        dist_down,
        "current_vec":      cur_arr.tolist(),
        "dim_z_up":         dim_z_up,
        "dim_z_down":       dim_z_down,
        "built_at":         cache.get("built_at", "?"),
        # Base rate — dùng trong format để calibrate kỳ vọng user
        "base_rate_up":       base_rate_up,
        "base_rate_down":     base_rate_down,
        "avg_interval_up":    avg_interval_up,
        "avg_interval_down":  avg_interval_down,
    })
    return result_base


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def format_wave_report(result: dict) -> str:
    """
    Format kết quả analyze_wave() thành text Telegram-friendly.
    Tối đa ~4000 ký tự.
    """
    if not result.get("ok"):
        return f"❌ Loi phan tich song {result['symbol']}: {result.get('error','?')}"

    sym        = result["symbol"]
    n_bars     = result["n_bars"]
    n_up       = result["n_up"]
    n_down     = result["n_down"]
    n_up_tot   = result.get("n_up_total", n_up)
    n_down_tot = result.get("n_down_total", n_down)
    amp_u      = result["amp_up_mean"]
    amp_d      = result["amp_down_mean"]
    dur_u      = result["dur_up_mean"]
    dur_d      = result["dur_down_mean"]
    zz_pct     = result.get("zigzag_pct", ZIGZAG_MIN_PCT)
    score_up   = result["score_up"]
    score_down = result["score_down"]
    verdict    = result["verdict"]
    confidence = result["confidence"]
    top_dims   = result["top_dims"]
    dist_up    = result["dist_up"]
    dist_down  = result["dist_down"]
    cur_vec    = result["current_vec"]
    dim_z_up   = result["dim_z_up"]
    dim_z_down = result["dim_z_down"]
    built      = result.get("built_at", "?")[:10]

    lines = []

    # ── Header ────────────────────────────────────────────────────────────
    lines += [
        f"PHAN TICH SONG LICH SU: {sym}",
        "═" * 40,
        f"Lich su: {n_bars} ngay | ZigZag: {zz_pct:.0f}% | Cache: {built}",
        f"Song tang: {n_up} mau vec ({n_up_tot} song, TB +{amp_u:.1f}%, ~{dur_u:.0f} ngay)",
        f"Song giam: {n_down} mau vec ({n_down_tot} song, TB -{amp_d:.1f}%, ~{dur_d:.0f} ngay)",
        "",
    ]

    # ── Base rate — đặt ngay sau header để user thấy trước khi đọc score ──
    base_rate_up   = result.get("base_rate_up",   0.0)
    base_rate_down = result.get("base_rate_down",  0.0)
    avg_int_up     = result.get("avg_interval_up",  0.0)
    avg_int_down   = result.get("avg_interval_down", 0.0)
    lines += [
        "TAN SUAT LICH SU (BASE RATE):",
        "─" * 40,
        f"  Song tang: {n_up_tot} lan / {n_bars} ngay = {base_rate_up:.1f}%"
        + (f"  (TB cu {avg_int_up:.0f} ngay/lan)" if avg_int_up > 0 else ""),
        f"  Song giam: {n_down_tot} lan / {n_bars} ngay = {base_rate_down:.1f}%"
        + (f"  (TB cu {avg_int_down:.0f} ngay/lan)" if avg_int_down > 0 else ""),
        "  * Day la xac suat nen (base rate) — khong lien quan den score ben duoi.",
        "",
    ]

    # ── Đặc điểm sóng tăng ────────────────────────────────────────────────
    lines += [
        f"DAU HIEN CHUNG CUA SONG TANG ({n_up} lan):",
        "─" * 40,
        f"  {'Dimension':<18} {'Median':>7}  {'P25':>7}  {'P75':>7}",
        "  " + "-" * 38,
    ]
    # Hiển thị top 5 discriminant dims cho sóng tăng
    disc_dims = [d[0] for d in top_dims]
    for dim in disc_dims[:5]:
        if dim not in dist_up:
            continue
        d   = dist_up[dim]
        lbl = DIM_LABEL.get(dim, dim)
        lines.append(
            f"  {lbl:<18} {d['median']:>+7.2f}  {d['p25']:>+7.2f}  {d['p75']:>+7.2f}"
        )
    # Mô tả ngắn
    up_chars = []
    for dim, mu, md, _ in top_dims[:3]:
        up_chars.append(DIM_HIGH.get(dim, dim) if mu > md else DIM_LOW.get(dim, dim))
    if up_chars:
        lines.append(f"  => {', '.join(up_chars)}.")
    lines.append("")

    # ── Đặc điểm sóng giảm ────────────────────────────────────────────────
    lines += [
        f"DAU HIEN CHUNG CUA SONG GIAM ({n_down} lan):",
        "─" * 40,
        f"  {'Dimension':<18} {'Median':>7}  {'P25':>7}  {'P75':>7}",
        "  " + "-" * 38,
    ]
    for dim in disc_dims[:5]:
        if dim not in dist_down:
            continue
        d   = dist_down[dim]
        lbl = DIM_LABEL.get(dim, dim)
        lines.append(
            f"  {lbl:<18} {d['median']:>+7.2f}  {d['p25']:>+7.2f}  {d['p75']:>+7.2f}"
        )
    down_chars = []
    for dim, mu, md, _ in top_dims[:3]:
        down_chars.append(DIM_LOW.get(dim, dim) if mu > md else DIM_HIGH.get(dim, dim))
    if down_chars:
        lines.append(f"  => {', '.join(down_chars)}.")
    lines.append("")

    # ── So sánh với hiện tại ──────────────────────────────────────────────
    bar_up   = _score_bar(score_up)
    bar_down = _score_bar(score_down)
    verdict_em = (
        "🟢 GIONG BOI CANH TRUOC SONG TANG"  if verdict == "SONG TANG"  else
        "🔴 GIONG BOI CANH TRUOC SONG GIAM"  if verdict == "SONG GIAM"  else
        "🟡 KHONG RO RANG"
    )

    lines += [
        "SO SANH VOI HIEN TAI:",
        "─" * 40,
        f"  Song tang : {bar_up} {score_up:.1%}",
        f"  Song giam : {bar_down} {score_down:.1%}",
        f"  Chenh lech: {confidence:.1%}",
        "",
        f"=> {verdict_em}",
        f"   (Do tin cay: {'Cao' if confidence >= 0.15 else 'Trung binh' if confidence >= 0.08 else 'Thap'})",
        "",
    ]

    # ── Top 5 dimensions điển hình nhất với hiện tại ──────────────────────
    # Hiển thị dims mà current_vec khớp nhất với sóng tăng (|z| nhỏ nhất)
    if dim_z_up and verdict in ("SONG TANG", "KHONG RO"):
        typical_up = [(dim, z) for dim, z in dim_z_up[:5]]
        lines.append("DIMENSIONS DIEN HINH VOI SONG TANG (hien tai):")
        for dim, z in typical_up:
            lbl = DIM_LABEL.get(dim, dim)
            j   = VECTOR_KEYS.index(dim) if dim in VECTOR_KEYS else -1
            cur = f"{cur_vec[j]:+.2f}" if j >= 0 else "?"
            lines.append(f"  {lbl:<18} hien tai={cur}  z={z:+.2f}")
        lines.append("")

    if dim_z_down and verdict in ("SONG GIAM", "KHONG RO"):
        typical_down = [(dim, z) for dim, z in dim_z_down[:5]]
        lines.append("DIMENSIONS DIEN HINH VOI SONG GIAM (hien tai):")
        for dim, z in typical_down:
            lbl = DIM_LABEL.get(dim, dim)
            j   = VECTOR_KEYS.index(dim) if dim in VECTOR_KEYS else -1
            cur = f"{cur_vec[j]:+.2f}" if j >= 0 else "?"
            lines.append(f"  {lbl:<18} hien tai={cur}  z={z:+.2f}")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────
    lines += [
        "─" * 40,
        "GIAI THICH SCORE:",
        f"  Score {score_up:.0%} (song tang) co nghia: hien tai nam TRONG",
        "  phan phoi dien hinh cua song tang — KHONG phai xac suat tang",
        f"  la {score_up:.0%}. Xac suat nen thuc te chi la {base_rate_up:.1f}%.",
        "─" * 40,
        "⚠️  CANH BAO: Phan tich chi mang tinh tham khao.",
        "   Qua khu khong dam bao tuong lai.",
        "   Khong su dung de ra quyet dinh giao dich.",
    ]

    return "\n".join(lines)


def _score_bar(score: float, width: int = 10) -> str:
    """Thanh tiến trình ASCII cho score 0-1."""
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def wave_cmd(update, context):
    """
    /wave <MÃ> [--rebuild]

    Phân tích sóng lịch sử cho mã cổ phiếu.
    --rebuild: bỏ qua cache, build lại từ đầu.
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Cu phap: /wave <MA> [--rebuild]\n"
            "Vi du:  /wave STB\n"
            "        /wave HAH --rebuild\n\n"
            "Phan tich song lich su: tim dau hieu chung truoc song tang/giam\n"
            "va so sanh voi trang thai hien tai."
        )
        return

    import re as _re
    symbol_raw = args[0].upper().strip()
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol_raw):
        await update.message.reply_text(f"Ma '{symbol_raw}' khong hop le.")
        return

    symbol        = symbol_raw
    force_rebuild = "--rebuild" in args

    msg = await update.message.reply_text(
        f"🔍 Dang phan tich song lich su: {symbol}...\n"
        f"{'(Rebuild cache) ' if force_rebuild else ''}"
        f"Co the mat 20-60 giay lan dau."
    )

    import asyncio
    try:
        result = await asyncio.to_thread(analyze_wave, symbol, force_rebuild)
    except Exception as e:
        import traceback
        logger.error(f"wave_cmd {symbol}: {e}\n{traceback.format_exc()}")
        await msg.edit_text(f"❌ Loi xu ly /wave {symbol}: {str(e)[:200]}")
        return

    report = format_wave_report(result)

    # Split nếu > 4096
    if len(report) <= 4096:
        try:
            await msg.edit_text(report)
        except Exception:
            await update.message.reply_text(report[:4096])
    else:
        # Part 1: edit
        split_at = report.rfind("\n\n", 0, 4000)
        if split_at < 0:
            split_at = 4000
        part1 = report[:split_at].strip()
        part2 = report[split_at:].strip()
        try:
            await msg.edit_text(part1)
        except Exception:
            await update.message.reply_text(part1)
        if part2:
            try:
                await update.message.reply_text(part2[:4096])
            except Exception:
                pass
