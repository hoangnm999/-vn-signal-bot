"""
wave_pattern.py — Phân tích Sóng Lịch Sử (Wave Pattern Analysis).

Triết lý:
  Thay vì tìm "ngày giống hôm nay" (analog), module này quét TOÀN BỘ
  lịch sử giá, xác định tất cả các con sóng tăng/giảm lớn, lấy dấu
  vân tay (state vector 15D) tại điểm BẮT ĐẦU mỗi sóng, rồi so sánh
  trạng thái hiện tại với 2 nhóm đó.

  Câu hỏi trả lời: "Hiện tại giống bối cảnh trước sóng tăng hay sóng giảm?"

Pipeline:
  1. ZigZag(min_pct=auto-tune 3-15%) → danh sách đỉnh/đáy cục bộ
  2. Lọc sóng đủ biên độ (top 30% động, safety floor = MIN_WAVES*2)
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
ZIGZAG_MIN_PCT    = 5.0       # fallback nếu auto-tune thất bại
MAX_ZIGZAG_PCT    = 10.0      # giới hạn trên — sóng > 10% là super-cycle
TARGET_WAVES_MIN  = 15        # tối thiểu sóng mong muốn sau auto-tune
TARGET_WAVES_MAX  = 80        # tối đa — tránh noise
MIN_WAVES         = 5         # tối thiểu sóng mỗi nhóm để phân tích
WARN_WAVES        = 15        # cảnh báo mẫu thấp
MIN_MEANINGFUL_SCORE = 0.25   # threshold weak signal note
MIN_BARS_REQUIRED = 200       # tối thiểu bars lịch sử
LOAD_DAYS         = 2000      # ~8 năm

# Negative sampling — phân biệt đáy/đỉnh thực vs oversold/overbought thường
OVERSOLD_THRESH    = -0.60    # stoch_k_norm < -0.60 để qualify false bottom (G2)
OVERBOUGHT_THRESH  = +0.60    # stoch_k_norm > +0.60 để qualify false top (G4)
PIVOT_EXCL_WINDOW  = 15       # exclude ±15 ngày quanh pivot thực (BOT/TOP)
NEG_SAMPLE_RATIO   = 3        # lấy tối đa NEG_SAMPLE_RATIO × n_positive negatives

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
    # Giá trị dương (+1) → ý nghĩa kỹ thuật
    "rsi_norm":        "RSI cao (overbought ~70+)",
    "macd_hist_norm":  "MACD Hist duong manh",
    "bb_position":     "gia sat BB upper",
    "volume_spike":    "volume tang dot bien",
    "trend_slope":     "SMA20 tren SMA50 (uptrend)",
    "price_vs_sma20":  "gia tren SMA20",
    "price_vs_sma50":  "gia tren SMA50",
    "atr_ratio":       "bien dong cao (ATR lon)",
    "stoch_k_norm":    "Stoch K cao (overbought)",
    "ema_cross":       "EMA12 tren EMA26",
    "momentum_5d":     "tang manh 5 phien gan",
    "momentum_20d":    "xu huong tang 20D",
    "high_low_pos":    "gia gan dinh 20 phien",
    "vol_trend":       "vol ngan han tang",
    "candle_body":     "than nen day (quyet doan)",
}
DIM_LOW = {
    # Giá trị âm (-1) → ý nghĩa kỹ thuật
    "rsi_norm":        "RSI thap (oversold ~30-)",
    "macd_hist_norm":  "MACD Hist am (cham day)",
    "bb_position":     "gia sat BB lower",
    "volume_spike":    "volume suy yeu",
    "trend_slope":     "SMA20 duoi SMA50 (downtrend)",
    "price_vs_sma20":  "gia duoi SMA20",
    "price_vs_sma50":  "gia duoi SMA50",
    "atr_ratio":       "bien dong thap (tich luy)",
    "stoch_k_norm":    "Stoch K thap (oversold)",
    "ema_cross":       "EMA12 duoi EMA26",
    "momentum_5d":     "giam manh 5 phien gan",
    "momentum_20d":    "xu huong giam 20D",
    "high_low_pos":    "gia gan day 20 phien",
    "vol_trend":       "vol on dinh/giam",
    "candle_body":     "than nen nho (do du)",
}


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 0 — AUTO-TUNE ZIGZAG THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════

def _find_optimal_zigzag_pct(
    close: np.ndarray,
    candidates: tuple = (3.0, 5.0, 7.0, 10.0, 12.0, 15.0),
) -> float:
    """
    Tự động tìm ngưỡng ZigZag phù hợp nhất cho mã.

    Logic:
    - Thử các ngưỡng từ LỚN → NHỎ (ưu tiên ngưỡng lớn = sóng có ý nghĩa hơn)
    - Chọn ngưỡng LỚN NHẤT mà vẫn cho >= TARGET_WAVES_MIN sóng mỗi chiều
    - Không bao giờ vượt TARGET_WAVES_MAX (quá nhiều = noise)
    - Fallback về ngưỡng nhỏ nhất trong candidates nếu không tìm được ngưỡng tốt

    Ví dụ STB: 15%→3 sóng, 12%→4 sóng, 10%→6 sóng, 7%→22 sóng (OK → chọn 7%),
               5%→38 sóng (OK), 3%→80 sóng (quá nhiều)
    """
    # Đếm sóng cho từng ngưỡng — từ lớn → nhỏ
    candidates_desc = sorted(candidates, reverse=True)
    results = []
    for pct in candidates_desc:
        pivots = _zigzag(close, min_pct=pct)
        n_up = sum(
            1 for i in range(len(pivots) - 1)
            if pivots[i][2] == "BOT" and pivots[i+1][2] == "TOP"
        )
        n_down = sum(
            1 for i in range(len(pivots) - 1)
            if pivots[i][2] == "TOP" and pivots[i+1][2] == "BOT"
        )
        n_min = min(n_up, n_down)
        results.append((pct, n_min))
        logger.debug(f"ZigZag tune {pct:.0f}%: {n_up}↑ {n_down}↓ → min={n_min}")

    # Chọn ngưỡng lớn nhất trong vùng [TARGET_WAVES_MIN, TARGET_WAVES_MAX]
    # VÀ không vượt MAX_ZIGZAG_PCT (sóng > MAX% là super-cycle, không phải trading wave)
    for pct, n_min in results:  # đã sort desc → gặp cái đầu tiên đạt yêu cầu là tốt nhất
        if pct > MAX_ZIGZAG_PCT:
            continue  # bỏ qua ngưỡng quá lớn dù có đủ sóng
        if TARGET_WAVES_MIN <= n_min <= TARGET_WAVES_MAX:
            logger.info(f"ZigZag auto-tune: chon {pct:.0f}% → {n_min} waves/side")
            return pct

    # Không tìm được trong vùng lý tưởng dưới MAX_ZIGZAG_PCT
    # → Chọn ngưỡng nhỏ nhất có sóng > TARGET_WAVES_MAX (vẫn dưới MAX_ZIGZAG_PCT)
    for pct, n_min in reversed(results):  # từ nhỏ → lớn
        if pct > MAX_ZIGZAG_PCT:
            continue
        if n_min > TARGET_WAVES_MAX:
            logger.info(f"ZigZag auto-tune (fallback many): {pct:.0f}% → {n_min} waves/side")
            return pct

    # Fallback cuối: ngưỡng nhỏ nhất (nhiều sóng nhất), dù dưới TARGET_WAVES_MIN
    best_pct = min(candidates)
    best_n   = next((n for p, n in results if p == best_pct), 0)
    logger.warning(
        f"ZigZag auto-tune: khong tim duoc nguong ly tuong, "
        f"fallback {best_pct:.0f}% → {best_n} waves/side"
    )
    return best_pct


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
) -> tuple[list[dict], list[dict]]:
    """
    Từ danh sách pivots (đỉnh/đáy ZigZag), trích xuất TẤT CẢ các con sóng.

    Sóng tăng: BOT → TOP (amplitude = (top - bot) / bot * 100)
    Sóng giảm: TOP → BOT (amplitude = (top - bot) / top * 100)

    Không filter thêm — ZigZag đã là bộ lọc (chỉ xác nhận pivot >= min_pct%).
    Tất cả sóng ZigZag đều đủ biên độ có ý nghĩa.

    Sub-type classification (context layer, không tách phân phối):
    ─────────────────────────────────────────────────────────────
    Sóng tăng (UP): BOT[i] → TOP[i]
      Nhìn pivot tiếp theo BOT[i+1]:
      - after_lower_low = True  → BOT[i+1] < BOT[i]  → sóng tăng là relief rally
      - after_lower_low = False → BOT[i+1] >= BOT[i] → sóng tăng là đáy thực (bottom)
      sub_type: "BOTTOM_REAL" | "RELIEF_RALLY" | "UNKNOWN" (sóng cuối)

    Sóng giảm (DOWN): TOP[i] → BOT[i]
      Nhìn pivot tiếp theo TOP[i+1]:
      - after_higher_high = True  → TOP[i+1] > TOP[i]  → sóng giảm là correction
      - after_higher_high = False → TOP[i+1] <= TOP[i] → sóng giảm là peak thực
      sub_type: "PEAK_REAL" | "CORRECTION" | "UNKNOWN" (sóng cuối)

    Returns:
        (waves_up, waves_down) — mỗi phần tử là dict:
        {
            "start_idx":   int,
            "end_idx":     int,
            "start_price": float,
            "end_price":   float,
            "amplitude":   float,
            "duration":    int,
            "type":        "UP" | "DOWN",
            "sub_type":    str,   # xem mô tả trên
            "next_pivot_price": float | None,  # giá pivot tiếp theo (để debug)
        }
    """
    waves_up:   list[dict] = []
    waves_down: list[dict] = []

    n_pivots = len(pivots)

    for i in range(n_pivots - 1):
        p1_idx, p1_price, p1_type = pivots[i]
        p2_idx, p2_price, p2_type = pivots[i + 1]

        if p1_price <= 0:
            continue

        # Pivot tiếp theo sau sóng (i+2) để xác định sub_type
        # Đây là đỉnh/đáy sau khi sóng kết thúc — nhìn về tương lai của sóng
        has_next   = (i + 2) < n_pivots
        next_price = pivots[i + 2][1] if has_next else None

        if p1_type == "BOT" and p2_type == "TOP":
            # Sóng tăng: BOT[i] → TOP[i]
            # Pivot tiếp theo (i+2) là BOT[i+1]
            amp = (p2_price - p1_price) / p1_price * 100
            if next_price is not None:
                # BOT[i+1] < BOT[i] → lower low → đây là relief rally
                after_lower_low = next_price < p1_price
                sub_type = "RELIEF_RALLY" if after_lower_low else "BOTTOM_REAL"
            else:
                sub_type   = "UNKNOWN"
                next_price = None

            waves_up.append({
                "start_idx":       p1_idx,
                "end_idx":         p2_idx,
                "start_price":     p1_price,
                "end_price":       p2_price,
                "amplitude":       round(amp, 2),
                "duration":        p2_idx - p1_idx,
                "type":            "UP",
                "sub_type":        sub_type,
                "next_pivot_price": next_price,
            })

        elif p1_type == "TOP" and p2_type == "BOT":
            # Sóng giảm: TOP[i] → BOT[i]
            # Pivot tiếp theo (i+2) là TOP[i+1]
            amp = (p1_price - p2_price) / p1_price * 100
            if next_price is not None:
                # TOP[i+1] > TOP[i] → higher high → đây là correction
                after_higher_high = next_price > p1_price
                sub_type = "CORRECTION" if after_higher_high else "PEAK_REAL"
            else:
                sub_type   = "UNKNOWN"
                next_price = None

            waves_down.append({
                "start_idx":       p1_idx,
                "end_idx":         p2_idx,
                "start_price":     p1_price,
                "end_price":       p2_price,
                "amplitude":       round(amp, 2),
                "duration":        p2_idx - p1_idx,
                "type":            "DOWN",
                "sub_type":        sub_type,
                "next_pivot_price": next_price,
            })

    return waves_up, waves_down


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2B — SUBTYPE STATS (context layer, không tách phân phối)
# ══════════════════════════════════════════════════════════════════════════════

def _subtype_stats(waves: list[dict]) -> dict:
    """
    Tính stats theo sub_type cho một chiều sóng (up hoặc down).

    Không tách phân phối — chỉ tính median amplitude/duration theo subgroup
    để dùng làm context layer trong report.

    Returns dict:
    {
        "counts":  { sub_type: int },           # số sóng mỗi loại
        "amp":     { sub_type: float },          # median amplitude
        "dur":     { sub_type: float },          # median duration
        "pct":     { sub_type: float },          # tỷ lệ %
        "key_dims":{ sub_type: {dim: median} },  # median của key dims (trend_slope, momentum_20d)
        "total_known": int,                      # tổng REAL + CORRECTION/RALLY (bỏ UNKNOWN)
    }

    Key dims được dùng để so sánh với vector hiện tại:
    - trend_slope: SMA20 vs SMA50 — uptrend hay downtrend
    - momentum_20d: xu hướng 20 ngày
    - price_vs_sma50: khoảng cách giá vs MA50 — overextension
    """
    KEY_DIMS = ["trend_slope", "momentum_20d", "price_vs_sma50"]

    counts:   dict[str, int]         = {}
    amp_vals: dict[str, list[float]] = {}
    dur_vals: dict[str, list[float]] = {}

    for w in waves:
        st = w.get("sub_type", "UNKNOWN")
        counts[st]   = counts.get(st, 0) + 1
        amp_vals.setdefault(st, []).append(w["amplitude"])
        dur_vals.setdefault(st, []).append(float(w["duration"]))

    amp_med = {st: float(np.median(v)) for st, v in amp_vals.items()}
    dur_med = {st: float(np.median(v)) for st, v in dur_vals.items()}

    total_known = sum(v for k, v in counts.items() if k != "UNKNOWN")
    pct = {
        st: round(n / total_known * 100, 1) if total_known > 0 else 0.0
        for st, n in counts.items()
        if st != "UNKNOWN"
    }

    return {
        "counts":      counts,
        "amp":         {k: round(v, 1) for k, v in amp_med.items()},
        "dur":         {k: round(v, 1) for k, v in dur_med.items()},
        "pct":         pct,
        "total_known": total_known,
    }


def _compare_current_to_subtypes(
    cur_vec:     np.ndarray,
    waves:       list[dict],
    df:          pd.DataFrame,
    wave_type:   str,   # "UP" hoặc "DOWN"
) -> dict:
    """
    So sánh vector hiện tại với median của từng subgroup theo KEY_DIMS.

    Không tính score mới — chỉ tính khoảng cách (absolute diff) giữa
    giá trị hiện tại và median của mỗi subgroup trên 3 key dims.

    Kết quả: subgroup nào "gần" hơn → hiện tại giống loại đó hơn.

    Returns:
    {
        "key_dims":  [str],             # 3 dims dùng để so sánh
        "subtypes":  {
            sub_type: {
                "n":        int,
                "medians":  {dim: float},   # median của subgroup
                "cur_vals": {dim: float},   # giá trị hiện tại
                "diffs":    {dim: float},   # |cur - median|
                "total_dist": float,        # sum of |diffs| → nhỏ hơn = gần hơn
            }
        },
        "closest":   str,   # subtype gần nhất
        "margin":    float, # khoảng cách giữa closest và second closest
    }
    """
    KEY_DIMS = ["trend_slope", "momentum_20d", "price_vs_sma50"]

    # Map dim name → index trong VECTOR_KEYS
    dim_indices = {d: VECTOR_KEYS.index(d) for d in KEY_DIMS if d in VECTOR_KEYS}
    if not dim_indices:
        return {}

    # Tập hợp vectors theo subtype (chỉ dùng waves có sub_type != UNKNOWN)
    sub_vecs: dict[str, list[np.ndarray]] = {}
    for w in waves:
        st = w.get("sub_type", "UNKNOWN")
        if st == "UNKNOWN":
            continue
        idx = w["start_idx"]
        if idx < 59:
            continue
        vec = compute_state_vector_for_date(df, idx)
        if vec is None:
            continue
        arr = vec.get("_array")
        if arr is not None and len(arr) == VECTOR_DIM:
            sub_vecs.setdefault(st, []).append(arr.astype(np.float32))

    if not sub_vecs:
        return {}

    # Tính median và khoảng cách với cur_vec
    result_subtypes = {}
    for st, vecs in sub_vecs.items():
        if len(vecs) < 2:
            continue
        mat      = np.stack(vecs, axis=0)
        medians  = {}
        cur_vals = {}
        diffs    = {}
        for dim, j in dim_indices.items():
            med        = float(np.median(mat[:, j]))
            cur        = float(cur_vec[j])
            medians[dim]  = round(med, 3)
            cur_vals[dim] = round(cur, 3)
            diffs[dim]    = round(abs(cur - med), 3)

        total_dist = sum(diffs.values())
        result_subtypes[st] = {
            "n":          len(vecs),
            "medians":    medians,
            "cur_vals":   cur_vals,
            "diffs":      diffs,
            "total_dist": round(total_dist, 3),
        }

    if not result_subtypes:
        return {}

    # Tìm subtype gần nhất
    sorted_st = sorted(result_subtypes.items(), key=lambda x: x[1]["total_dist"])
    closest   = sorted_st[0][0]
    margin    = (
        round(sorted_st[1][1]["total_dist"] - sorted_st[0][1]["total_dist"], 3)
        if len(sorted_st) >= 2 else 0.0
    )

    return {
        "key_dims":  list(dim_indices.keys()),
        "subtypes":  result_subtypes,
        "closest":   closest,
        "margin":    margin,
    }


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
# BƯỚC 3B — NEGATIVE SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def _build_negative_samples(
    df:           pd.DataFrame,
    pivot_indices: list[int],
    mode:          str,            # "bottom" hoặc "top"
    n_positive:    int,
    rng_seed:      int = 42,
) -> list[np.ndarray]:
    """
    Xây dựng G2 (false bottoms) hoặc G4 (false tops) — negative samples.

    Logic:
    - G2 (false bottom): ngày stoch_k_norm < OVERSOLD_THRESH
      NHƯNG không phải pivot thực, VÀ không trong ±PIVOT_EXCL_WINDOW ngày của pivot thực
    - G4 (false top): tương tự với overbought

    Tại sao cần exclude window:
    - Ngày gần đáy thực thường cũng rất oversold → nếu để vào G2 sẽ làm G1 vs G2
      gần giống nhau → không phân biệt được gì mới

    Returns:
        list of np.ndarray — tối đa NEG_SAMPLE_RATIO × n_positive vectors
        (đã random sample nếu quá nhiều)
    """
    if "stoch_k_norm" not in VECTOR_KEYS:
        return []

    stoch_idx = VECTOR_KEYS.index("stoch_k_norm")
    n_bars    = len(df)
    thresh    = OVERSOLD_THRESH if mode == "bottom" else OVERBOUGHT_THRESH

    # Tập hợp index cần exclude (±window quanh mỗi pivot thực)
    excluded: set[int] = set()
    for pi in pivot_indices:
        for d in range(-PIVOT_EXCL_WINDOW, PIVOT_EXCL_WINDOW + 1):
            excluded.add(pi + d)

    # Tìm tất cả ngày thỏa điều kiện oversold/overbought và không bị exclude
    # Tính vector 1 lần duy nhất, lưu cả (idx, arr) để dùng lại
    candidates: list[tuple[int, np.ndarray]] = []
    for i in range(60, n_bars):
        if i in excluded:
            continue
        vec = compute_state_vector_for_date(df, i)
        if vec is None:
            continue
        arr = vec.get("_array")
        if arr is None or len(arr) != VECTOR_DIM:
            continue
        sk = float(arr[stoch_idx])
        qualify = (sk < thresh) if mode == "bottom" else (sk > thresh)
        if qualify:
            candidates.append((i, arr.astype(np.float32)))

    if not candidates:
        logger.warning(f"Negative sampling ({mode}): 0 candidates found")
        return []

    # Random sample tối đa NEG_SAMPLE_RATIO × n_positive
    max_neg = n_positive * NEG_SAMPLE_RATIO
    rng = np.random.default_rng(rng_seed)
    if len(candidates) > max_neg:
        sampled_idx = rng.choice(len(candidates), size=max_neg, replace=False)
        candidates  = [candidates[i] for i in sampled_idx]

    vectors: list[np.ndarray] = [arr for _, arr in candidates]

    logger.info(
        f"Negative sampling ({mode}): {len(candidates)} sampled "
        f"(max={max_neg}, thresh={thresh:+.2f})"
    )
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

    total_score  = 0.0
    max_possible = 0.0   # chỉ tính dims hợp lệ (không degenerate)
    dim_zscores  = []

    for j, dim in enumerate(VECTOR_KEYS):
        if dim not in dist:
            continue
        d   = dist[dim]
        std = d["std"]

        # Degenerate dimension: std≈0 nghĩa là tất cả mẫu bị clip tại 1 giá trị
        # (thường gặp với momentum dims khi toàn bộ sóng bắt đầu từ vùng crash)
        # → dimension này không có discriminating power → bỏ qua hoàn toàn
        if std < 0.02:
            dim_zscores.append((dim, float("nan")))
            continue

        z = (float(current_vec[j]) - d["mean"]) / std
        dim_zscores.append((dim, round(z, 2)))

        max_possible += 1.5  # chỉ cộng max nếu dim hợp lệ
        if abs(z) <= 0.5:
            total_score += 1.5
        elif abs(z) <= 1.0:
            total_score += 1.0
        elif abs(z) <= 1.5:
            total_score += 0.5
        elif abs(z) > 2.0:
            total_score -= 0.5

    if max_possible < 1e-6:
        return 0.0, []
    score = max(0.0, min(1.0, total_score / max_possible))

    # Sort: nan (degenerate) ra cuối, |z| nhỏ nhất lên đầu
    dim_zscores.sort(key=lambda x: float("inf") if x[1] != x[1] else abs(x[1]))

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

    # Auto-tune ZigZag threshold — tìm ngưỡng cho 15-60 sóng mỗi chiều
    optimal_pct = _find_optimal_zigzag_pct(close)
    pivots = _zigzag(close, min_pct=optimal_pct)
    if len(pivots) < 6:
        return False, f"Khong du pivots ({len(pivots)}) — lich su qua ngan"

    # Extract waves — dùng tất cả, không filter top%
    waves_up, waves_down = _extract_waves(pivots)

    if len(waves_up) < MIN_WAVES:
        return False, f"Chi co {len(waves_up)} song tang (can >= {MIN_WAVES})"
    if len(waves_down) < MIN_WAVES:
        return False, f"Chi co {len(waves_down)} song giam (can >= {MIN_WAVES})"

    # Compute vectors tại điểm bắt đầu (G1 và G3 — positive samples)
    vecs_up   = _compute_wave_vectors(df, waves_up)
    vecs_down = _compute_wave_vectors(df, waves_down)

    if len(vecs_up) < MIN_WAVES:
        return False, f"Chi tinh duoc {len(vecs_up)} vector song tang"
    if len(vecs_down) < MIN_WAVES:
        return False, f"Chi tinh duoc {len(vecs_down)} vector song giam"

    # Negative sampling (G2 và G4) — cần toàn bộ df để scan
    bot_indices = [w["start_idx"] for w in waves_up]
    top_indices = [w["start_idx"] for w in waves_down]

    vecs_false_bottom = _build_negative_samples(
        df, pivot_indices=bot_indices, mode="bottom", n_positive=len(vecs_up)
    )
    vecs_false_top = _build_negative_samples(
        df, pivot_indices=top_indices, mode="top",    n_positive=len(vecs_down)
    )

    logger.info(
        f"{symbol}: G1={len(vecs_up)} true_bot, G2={len(vecs_false_bottom)} false_bot | "
        f"G3={len(vecs_down)} true_top, G4={len(vecs_false_top)} false_top"
    )

    # Tính phân phối cho cả 4 nhóm
    dist_up           = _group_distribution(vecs_up)
    dist_down         = _group_distribution(vecs_down)
    dist_false_bottom = _group_distribution(vecs_false_bottom) if vecs_false_bottom else {}
    dist_false_top    = _group_distribution(vecs_false_top)    if vecs_false_top    else {}

    # Tính stats cho waves
    amp_up_mean   = float(np.mean([w["amplitude"] for w in waves_up]))
    amp_down_mean = float(np.mean([w["amplitude"] for w in waves_down]))
    dur_up_mean   = float(np.mean([w["duration"]  for w in waves_up]))
    dur_down_mean = float(np.mean([w["duration"]  for w in waves_down]))

    # Subtype stats — context layer (không tách phân phối)
    subtype_stats_up   = _subtype_stats(waves_up)
    subtype_stats_down = _subtype_stats(waves_down)

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
        "zigzag_pct":  optimal_pct,
        "dist_up":     dist_up,
        "dist_down":   dist_down,
        "dist_false_bottom": dist_false_bottom,   # G2: oversold thường (không tạo đáy)
        "dist_false_top":    dist_false_top,       # G4: overbought thường (không tạo đỉnh)
        "n_false_bottom": len(vecs_false_bottom),
        "n_false_top":    len(vecs_false_top),
        # Subtype stats — context layer
        "subtype_stats_up":   subtype_stats_up,
        "subtype_stats_down": subtype_stats_down,
        # Lưu waves (có sub_type) để _compare_current_to_subtypes() dùng lại
        "waves_up":   waves_up,
        "waves_down": waves_down,
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


def _bootstrap_score_ci(
    current_vec:   np.ndarray,
    vecs_up:       list[np.ndarray],
    vecs_down:     list[np.ndarray],
    n_boot:        int = 500,
    ci:            float = 0.90,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Bootstrap confidence interval cho score_up và score_down.

    Với n nhỏ (8-15 mẫu), score điểm đơn rất không ổn định.
    Bootstrap resample 500 lần, lấy percentile 5%-95% → CI 90%.

    Returns:
        (ci_up, ci_down) — mỗi cái là (lo, hi) float tuple.
        Trả về (0.0, 0.0) nếu không đủ dữ liệu.
    """
    rng = np.random.default_rng(seed=42)
    alpha = (1 - ci) / 2  # 0.05 cho CI 90%

    def _boot_one(vecs: list[np.ndarray]) -> tuple[float, float]:
        n = len(vecs)
        if n < 3:
            return (0.0, 0.0)
        scores = []
        mat = np.stack(vecs, axis=0)
        for _ in range(n_boot):
            idx    = rng.integers(0, n, size=n)
            sample = mat[idx]
            # Tính dist từ bootstrap sample
            dist_b: dict = {}
            for j, dim in enumerate(VECTOR_KEYS):
                col = sample[:, j]
                dist_b[dim] = {
                    "mean": float(np.mean(col)),
                    "std":  float(np.std(col, ddof=1)) if n > 1 else 0.01,
                }
            s, _ = _zscore_membership(current_vec, dist_b)
            scores.append(s)
        lo = float(np.percentile(scores, alpha * 100))
        hi = float(np.percentile(scores, (1 - alpha) * 100))
        return (round(lo, 3), round(hi, 3))

    ci_up   = _boot_one(vecs_up)
    ci_down = _boot_one(vecs_down)
    return ci_up, ci_down


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
        "score_discrim_bottom": 0.0,   # G1 vs G2: đáy thực vs oversold thường
        "score_discrim_top":    0.0,   # G3 vs G4: đỉnh thực vs overbought thường
        "n_false_bottom": 0,
        "n_false_top":    0,
        # Subtype context layer
        "subtype_stats_up":    {},
        "subtype_stats_down":  {},
        "subtype_compare_up":  {},
        "subtype_compare_down":{},
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
    dist_false_bottom = cache.get("dist_false_bottom", {})
    dist_false_top    = cache.get("dist_false_top",    {})

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

    # Z-score membership — G1 và G3 (positive, như cũ)
    score_up,   dim_z_up   = _zscore_membership(cur_arr, dist_up)
    score_down, dim_z_down = _zscore_membership(cur_arr, dist_down)

    # Discriminative score — G1 vs G2, G3 vs G4
    # score_discrim_bottom > 0: hôm nay giống đáy thực hơn oversold thường
    # score_discrim_top    > 0: hôm nay giống đỉnh thực hơn overbought thường
    score_false_bottom, _ = _zscore_membership(cur_arr, dist_false_bottom) if dist_false_bottom else (0.0, [])
    score_false_top,    _ = _zscore_membership(cur_arr, dist_false_top)    if dist_false_top    else (0.0, [])
    score_discrim_bottom = round(score_up   - score_false_bottom, 3)
    score_discrim_top    = round(score_down - score_false_top,    3)

    # Subtype context layer — so sánh hiện tại với median của từng subgroup
    # Dùng waves đã lưu trong cache (có sub_type field)
    waves_up_cached   = cache.get("waves_up",   [])
    waves_down_cached = cache.get("waves_down", [])
    subtype_stats_up   = cache.get("subtype_stats_up",   {})
    subtype_stats_down = cache.get("subtype_stats_down", {})

    # Load df một lần để dùng trong _compare_current_to_subtypes
    # Ưu tiên dùng df_now nếu đủ bars, fallback load full
    try:
        from vn_loader import load_vn_ohlcv as _load_vn
        df_full = _load_vn(symbol, days=LOAD_DAYS, min_bars=MIN_BARS_REQUIRED)
    except Exception:
        df_full = df_now  # fallback

    subtype_compare_up   = _compare_current_to_subtypes(cur_arr, waves_up_cached,   df_full, "UP")
    subtype_compare_down = _compare_current_to_subtypes(cur_arr, waves_down_cached, df_full, "DOWN")

    # Verdict
    diff = score_up - score_down
    # Khi cả 2 scores thấp → trung lập, không nên kết luận chiều
    # Tránh contradiction: vừa SONG GIAM vừa nói "trung lập"
    if max(score_up, score_down) < MIN_MEANINGFUL_SCORE:
        verdict = "KHONG RO"
    elif diff >= 0.08:
        verdict = "SONG TANG"
    elif diff <= -0.08:
        verdict = "SONG GIAM"
    else:
        verdict = "KHONG RO"

    # Regime weight — điều chỉnh ĐỊNH HƯỚNG theo Market Regime VNINDEX
    # Bull regime → khuếch đại score_up, giảm score_down
    # Bear regime → khuếch đại score_down, giảm score_up
    # Ví dụ STB ở R3 Bear Quiet: score_up ×0.8=10.7%, score_down ×1.2=61.3%
    regime_data    = None
    score_up_adj   = score_up
    score_down_adj = score_down
    regime_note_up = ""
    regime_note_dn = ""
    try:
        from market_regime import get_market_regime, apply_regime_weight
        regime_data = get_market_regime()
        score_up_adj, score_down_adj, regime_note_up, regime_note_dn = (
            apply_regime_weight(score_up, score_down, regime_data)
        )
    except Exception as _re:
        logger.debug(f"regime weight skip: {_re}")

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
        "score_discrim_bottom": score_discrim_bottom,
        "score_discrim_top":    score_discrim_top,
        "score_false_bottom_raw": score_false_bottom,  # G2 score raw
        "score_false_top_raw":    score_false_top,      # G4 score raw
        "n_false_bottom":   cache.get("n_false_bottom", 0),
        "n_false_top":      cache.get("n_false_top",    0),
        "verdict":          verdict,
        "confidence":       round(abs(diff), 3),
        "top_dims":         top_dims,
        "dist_up":          dist_up,
        "dist_down":        dist_down,
        "current_vec":      cur_arr.tolist(),
        "dim_z_up":         dim_z_up,
        "dim_z_down":       dim_z_down,
        "built_at":         cache.get("built_at", "?"),
        # Regime-adjusted scores
        "score_up_adj":       score_up_adj,
        "score_down_adj":     score_down_adj,
        "regime_note_up":     regime_note_up,
        "regime_note_dn":     regime_note_dn,
        "regime_data":        regime_data,
        # Base rate — dùng trong format để calibrate kỳ vọng user
        "base_rate_up":       base_rate_up,
        "base_rate_down":     base_rate_down,
        "avg_interval_up":    avg_interval_up,
        "avg_interval_down":  avg_interval_down,
        # Subtype context layer
        "subtype_stats_up":    subtype_stats_up,
        "subtype_stats_down":  subtype_stats_down,
        "subtype_compare_up":  subtype_compare_up,
        "subtype_compare_down":subtype_compare_down,
    })
    return result_base


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_subtype_section(lines: list[str], result: dict, verdict: str) -> None:
    """
    Thêm section PHAN LOAI SONG vào lines (in-place).

    Hiển thị:
    - Tỷ lệ phân loại sóng (Loại A / Loại B) cho cả 2 chiều
    - So sánh vector hiện tại với median của từng subgroup (key dims)
    - Kết luận: hiện tại giống loại nào hơn
    - Safety: chỉ hiển thị nếu total_known >= MIN_SUBTYPE_KNOWN

    Hoàn toàn là context layer — không thay đổi verdict chính.
    """
    MIN_SUBTYPE_KNOWN = 6   # cần ít nhất 6 sóng có sub_type xác định

    # Labels thân thiện
    SUBTYPE_LABEL = {
        # Sóng tăng
        "BOTTOM_REAL":  "Day that     (Loai B)",
        "RELIEF_RALLY": "Relief Rally (Loai A)",
        # Sóng giảm
        "PEAK_REAL":    "Peak that    (Loai B)",
        "CORRECTION":   "Correction   (Loai A)",
        "UNKNOWN":      "Chua xac dinh",
    }
    SUBTYPE_EMOJI = {
        "BOTTOM_REAL":  "🟢",
        "RELIEF_RALLY": "🟡",
        "PEAK_REAL":    "🔴",
        "CORRECTION":   "🟡",
        "UNKNOWN":      "⚪",
    }
    # Mô tả ngắn khi "closest"
    CLOSEST_DESC = {
        "BOTTOM_REAL":  "Day that → co the tang ben vung, it co lower low",
        "RELIEF_RALLY": "Relief Rally → tang ngan han trong downtrend, chu y lower low",
        "PEAK_REAL":    "Peak that → co the giam sau va dai, it co higher high",
        "CORRECTION":   "Correction → giam ngan han trong uptrend, chu y higher high",
    }

    KEY_DIM_LABEL = {
        "trend_slope":   "Trend SMA20/50",
        "momentum_20d":  "Momentum 20D",
        "price_vs_sma50":"Gia vs SMA50",
    }

    def _fmt_one_side(
        label_header: str,
        stats: dict,
        compare: dict,
        subtype_a: str,   # e.g. "CORRECTION" / "RELIEF_RALLY"
        subtype_b: str,   # e.g. "PEAK_REAL"  / "BOTTOM_REAL"
    ) -> list[str]:
        out = []
        if not stats:
            return out

        counts      = stats.get("counts", {})
        amp         = stats.get("amp",    {})
        dur         = stats.get("dur",    {})
        pct         = stats.get("pct",    {})
        total_known = stats.get("total_known", 0)

        if total_known < MIN_SUBTYPE_KNOWN:
            return out   # không đủ mẫu → bỏ qua

        out.append(label_header)
        out.append("─" * 40)

        for st in [subtype_a, subtype_b, "UNKNOWN"]:
            n   = counts.get(st, 0)
            if n == 0:
                continue
            lbl = SUBTYPE_LABEL.get(st, st)
            em  = SUBTYPE_EMOJI.get(st, "")
            p   = pct.get(st, 0.0)
            a   = amp.get(st, 0.0)
            d   = dur.get(st, 0.0)
            if st == "UNKNOWN":
                out.append(f"  {em} {lbl}: {n} song (cuoi chuoi, chua biet ket qua)")
            else:
                sign = "+" if st in ("BOTTOM_REAL", "RELIEF_RALLY") else "-"
                out.append(
                    f"  {em} {lbl}: {n} song ({p:.0f}%)  "
                    f"TB {sign}{a:.0f}%  ~{d:.0f} ngay"
                )

        # So sánh hiện tại với subgroup
        if compare and compare.get("subtypes"):
            subtypes_data = compare["subtypes"]
            closest       = compare.get("closest", "")
            margin        = compare.get("margin",  0.0)
            key_dims      = compare.get("key_dims", [])

            out.append("")
            out.append("  So sanh hien tai vs cac loai (3 key dims):")

            # Header dims
            dim_hdrs = "  ".join(f"{KEY_DIM_LABEL.get(d,d):>14}" for d in key_dims)
            out.append(f"  {'':20}  {dim_hdrs}")

            for st in [subtype_a, subtype_b]:
                if st not in subtypes_data:
                    continue
                sd   = subtypes_data[st]
                lbl  = SUBTYPE_LABEL.get(st, st)[:18]
                meds = "  ".join(
                    f"{sd['medians'].get(d,0):>+14.2f}" for d in key_dims
                )
                out.append(f"  {'['+lbl+']':<20}  {meds}")

            # Hàng "Hien tai"
            if subtypes_data:
                first_st    = next(iter(subtypes_data))
                cur_vals_ex = subtypes_data[first_st]["cur_vals"]
                cur_row     = "  ".join(
                    f"{cur_vals_ex.get(d,0):>+14.2f}" for d in key_dims
                )
                out.append(f"  {'[Hien tai]':<20}  {cur_row}")

            # Kết luận — luôn hiển thị nếu có ít nhất 1 subtype
            # closest có thể không nằm trong subtypes_data nếu chỉ có 1 bên đủ mẫu
            # → fallback: lấy subtype có total_dist nhỏ nhất trong subtypes_data
            closest_display = closest if closest in subtypes_data else (
                min(subtypes_data, key=lambda s: subtypes_data[s]["total_dist"])
                if subtypes_data else None
            )
            if closest_display:
                em_c   = SUBTYPE_EMOJI.get(closest_display, "")
                lbl_c  = SUBTYPE_LABEL.get(closest_display, closest_display)
                desc   = CLOSEST_DESC.get(closest_display, "")
                # margin < 0.15: 2 loại khá gần nhau → note thêm
                margin_str = (
                    f" (margin={margin:.3f} — rat gan)"
                    if 0 < margin < 0.15 else ""
                )
                out.append("")
                out.append(f"  → Hien tai gan nhat voi: {em_c} {lbl_c}{margin_str}")
                if desc:
                    out.append(f"     {desc}")
                if 0 < margin < 0.15:
                    out.append("     ⚠️  Hai loai kha gan nhau — ket luan chi mang tinh tham khao.")

        out.append("")
        return out

    # ── Sóng tăng ──────────────────────────────────────────────────────────
    st_up  = result.get("subtype_stats_up",    {})
    cmp_up = result.get("subtype_compare_up",  {})
    lines += _fmt_one_side(
        label_header = "PHAN LOAI SONG TANG (context layer):",
        stats        = st_up,
        compare      = cmp_up,
        subtype_a    = "RELIEF_RALLY",
        subtype_b    = "BOTTOM_REAL",
    )

    # ── Sóng giảm ──────────────────────────────────────────────────────────
    st_dn  = result.get("subtype_stats_down",   {})
    cmp_dn = result.get("subtype_compare_down", {})
    lines += _fmt_one_side(
        label_header = "PHAN LOAI SONG GIAM (context layer):",
        stats        = st_dn,
        compare      = cmp_dn,
        subtype_a    = "CORRECTION",
        subtype_b    = "PEAK_REAL",
    )


def format_wave_report(result: dict) -> str:
    """
    Format compact — 5-8 dòng, chỉ kết luận + key evidence.

    Cấu trúc:
      Header:  /wave STB ════
      Verdict: emoji + SONG TANG/GIAM/KHONG RO + reliability
      Score:   goc → adj theo regime (1 dòng)
      Regime:  label + weights (1 dòng)
      ApproachA: 1 dòng nếu có signal rõ
      Subtype: loại A/B của chiều phù hợp verdict (1-2 dòng)
      Kỳ vọng: TB ±X% trong ~Y ngày
    """
    if not result.get("ok"):
        return f"❌ /wave {result['symbol']}: {result.get('error','?')}"

    sym        = result["symbol"]
    n_up       = result["n_up"]
    n_down     = result["n_down"]
    score_up   = result["score_up"]
    score_down = result["score_down"]
    verdict    = result["verdict"]
    confidence = result["confidence"]
    amp_u      = result["amp_up_mean"]
    amp_d      = result["amp_down_mean"]
    dur_u      = result["dur_up_mean"]
    dur_d      = result["dur_down_mean"]
    n_up_tot   = result.get("n_up_total",   n_up)
    n_down_tot = result.get("n_down_total", n_down)
    zz_pct     = result.get("zigzag_pct",   ZIGZAG_MIN_PCT)
    built      = result.get("built_at", "?")[:10]

    # Regime
    score_up_adj   = result.get("score_up_adj",   score_up)
    score_down_adj = result.get("score_down_adj", score_down)
    regime_note_up = result.get("regime_note_up", "")
    regime_note_dn = result.get("regime_note_dn", "")
    regime_data    = result.get("regime_data")
    has_regime     = bool(regime_note_up)

    # Reliability
    n_min = min(n_up, n_down)
    def _rel(n, conf):
        if n >= 20 and conf >= 0.15: return "★★★"
        if n >= 15 and conf >= 0.10: return "★★☆"
        if n >= 10 and conf >= 0.08: return "★★☆"
        if n >= 7  and conf >= 0.08: return "★☆☆"
        return "☆☆☆"
    reliability = _rel(n_min, confidence)

    # Verdict emoji + label
    if verdict == "SONG TANG":
        v_em  = "🟢"
        v_lbl = "SONG TANG"
    elif verdict == "SONG GIAM":
        v_em  = "🔴"
        v_lbl = "SONG GIAM"
    else:
        v_em  = "🟡"
        v_lbl = "KHONG RO"

    # Approach A — chỉ lấy signal phù hợp với verdict
    score_discrim_top    = result.get("score_discrim_top",    0.0)
    score_discrim_bottom = result.get("score_discrim_bottom", 0.0)
    n_fb = result.get("n_false_bottom", 0)
    n_ft = result.get("n_false_top",    0)

    def _approach_a_line(verdict):
        """1 dòng Approach A cho chiều phù hợp verdict."""
        if verdict == "SONG GIAM" and n_ft > 0:
            v = score_discrim_top
            if v >= 0.15:  return f"✅ Dac trung dinh that ro rang ({v:+.1%} vs overbought thuong)"
            if v >= 0.05:  return f"🟡 Hoi giong dinh that ({v:+.1%})"
            if v >= -0.05: return f"⚪ Khong phan biet duoc dinh that vs overbought thuong"
            return             f"🔴 Giong overbought thuong ({v:+.1%}) — thieu exhaustion"
        if verdict == "SONG TANG" and n_fb > 0:
            v = score_discrim_bottom
            if v >= 0.15:  return f"✅ Dac trung day that ro rang ({v:+.1%} vs oversold thuong)"
            if v >= 0.05:  return f"🟡 Hoi giong day that ({v:+.1%})"
            if v >= -0.05: return f"⚪ Khong phan biet duoc day that vs oversold thuong"
            return             f"🔴 Giong oversold thuong ({v:+.1%}) — thieu capitulation"
        return ""

    approach_a = _approach_a_line(verdict)

    # Subtype — chỉ chiều phù hợp verdict
    def _subtype_line(verdict):
        """1-2 dòng subtype cho chiều phù hợp verdict."""
        lines_st = []
        MIN_KNOWN = 6
        LABEL = {
            "PEAK_REAL":    "Peak that (Loai B)",
            "CORRECTION":   "Correction (Loai A)",
            "BOTTOM_REAL":  "Day that (Loai B)",
            "RELIEF_RALLY": "Relief Rally (Loai A)",
        }
        EM = {
            "PEAK_REAL": "🔴", "CORRECTION": "🟡",
            "BOTTOM_REAL": "🟢", "RELIEF_RALLY": "🟡",
        }
        WARN = {
            "PEAK_REAL":    "giam sau, it higher high",
            "CORRECTION":   "giam ngan, se co higher high",
            "BOTTOM_REAL":  "tang ben vung, it lower low",
            "RELIEF_RALLY": "tang ngan trong downtrend",
        }

        if verdict == "SONG GIAM":
            stats   = result.get("subtype_stats_down", {})
            compare = result.get("subtype_compare_down", {})
            subtypes_pair = ("CORRECTION", "PEAK_REAL")
            amp_ref = amp_d
            dur_ref = dur_d
        elif verdict == "SONG TANG":
            stats   = result.get("subtype_stats_up", {})
            compare = result.get("subtype_compare_up", {})
            subtypes_pair = ("RELIEF_RALLY", "BOTTOM_REAL")
            amp_ref = amp_u
            dur_ref = dur_u
        else:
            return []

        if not stats or stats.get("total_known", 0) < MIN_KNOWN:
            return []

        counts = stats.get("counts", {})
        pct    = stats.get("pct",    {})
        amp    = stats.get("amp",    {})
        dur    = stats.get("dur",    {})

        # Tỷ lệ 2 loại
        st_a, st_b = subtypes_pair
        n_a   = counts.get(st_a, 0)
        n_b   = counts.get(st_b, 0)
        pct_a = pct.get(st_a, 0.0)
        pct_b = pct.get(st_b, 0.0)
        amp_a = amp.get(st_a, 0.0)
        amp_b = amp.get(st_b, 0.0)
        dur_a = dur.get(st_a, 0.0)
        dur_b = dur.get(st_b, 0.0)

        sign = "-" if verdict == "SONG GIAM" else "+"
        lines_st.append(
            f"  {EM[st_a]} {LABEL[st_a]}: {pct_a:.0f}% ({n_a} song, TB {sign}{amp_a:.0f}%, ~{dur_a:.0f}n)"
        )
        lines_st.append(
            f"  {EM[st_b]} {LABEL[st_b]}: {pct_b:.0f}% ({n_b} song, TB {sign}{amp_b:.0f}%, ~{dur_b:.0f}n)"
        )

        # Closest subtype
        closest = compare.get("closest", "")
        if not closest and compare.get("subtypes"):
            closest = min(compare["subtypes"], key=lambda s: compare["subtypes"][s]["total_dist"])

        if closest and closest in LABEL:
            margin  = compare.get("margin", 0.0)
            warn    = WARN.get(closest, "")
            margin_note = " (gan nhau)" if 0 < margin < 0.15 else ""
            lines_st.append(
                f"  → {EM[closest]} Gan nhat: {LABEL[closest]}{margin_note} — {warn}"
            )

        return lines_st

    subtype_lines = _subtype_line(verdict)

    # Kỳ vọng biên độ
    if verdict == "SONG TANG":
        expectation = f"   Neu xay ra: TB +{amp_u:.0f}% trong ~{dur_u:.0f} ngay ({n_up_tot} song)"
    elif verdict == "SONG GIAM":
        expectation = f"   Neu xay ra: TB -{amp_d:.0f}% trong ~{dur_d:.0f} ngay ({n_down_tot} song)"
    else:
        expectation = ""

    # Cảnh báo mẫu thấp
    low_warn = ""
    if n_min < WARN_WAVES:
        low_warn = f"⚠️ Mau thap (n={n_min}) — tham khao yeu"

    # ── Assemble ──────────────────────────────────────────────────────────
    sep = "═" * 32
    lines = [f"/wave {sym}  [{built} | ZigZag {zz_pct:.0f}% | n={n_up}/{n_down}]",
             sep]

    # Dòng 1 — Verdict
    lines.append(f"{v_em} {v_lbl}  {reliability}  (chenh {confidence:.0%})")

    # Dòng 2 — Score gốc → adj
    if has_regime:
        regime_lbl = ""
        if regime_data and regime_data.get("ok"):
            r = regime_data.get("regime", "?")
            rl = regime_data.get("label", "").split("—")[1].strip() if "—" in regime_data.get("label","") else ""
            weights = regime_data.get("weights", {})
            regime_lbl = f"R{r} {rl} (↑×{weights.get('bull',1):.2f} ↓×{weights.get('bear',1):.2f})"
        lines.append(
            f"   Score: giam {score_down:.0%}→{score_down_adj:.0%} | "
            f"tang {score_up:.0%}→{score_up_adj:.0%}"
        )
        if regime_lbl:
            lines.append(f"   Regime: {regime_lbl}")
    else:
        lines.append(f"   Score: giam {score_down:.0%} | tang {score_up:.0%}  (n={n_up}/{n_down})")

    # Dòng 3 — Approach A
    if approach_a:
        lines.append(f"   {approach_a}")

    # Separator nhỏ
    lines.append("─" * 32)

    # Dòng 4-6 — Subtype
    if subtype_lines:
        lines += subtype_lines

    # Dòng 7 — Kỳ vọng
    if expectation:
        lines.append(expectation)

    # Cảnh báo mẫu
    if low_warn:
        lines.append(low_warn)

    # Trung lập note
    if verdict == "KHONG RO":
        lines.append("   Thi truong chua chon huong — cho tin hieu xac nhan.")

    lines.append(sep)

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
