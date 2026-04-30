"""
historical_analog.py - Phân tích tương đồng lịch sử (Historical Analog).

FLOW:
  1. Lần đầu: build_vector_cache(symbol) -> tính vector cho từng ngày lịch sử,
     lưu vào data/{symbol}_vectors.csv.
  2. Hàng ngày (cron 18:00): append_today_vector(symbol) -> nối dòng mới.
  3. Khi cần tìm: find_similar(symbol, target_vector, top_n=3, years=5)
     -> load CSV, cosine similarity, trả về top N ngày kèm forward returns.

CSV FORMAT (data/VCB_vectors.csv):
  date, rsi_norm, macd_sign, ..., [14 more cols], close
  (close lưu thêm để tính forward return mà không cần load lại price data)

TARGET: response < 5s (CSV scan ~1500 dòng = rất nhanh với numpy).
"""

from __future__ import annotations

import os
import logging
import time
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from state_vector import (
    VECTOR_KEYS, VECTOR_DIM,
    compute_state_vector,
    compute_state_vector_from_df,
    compute_state_vector_for_date,
    vector_to_list,
    cosine_similarity,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR         = pathlib.Path("data")
CACHE_SUFFIX     = "_vectors.csv"
MIN_HISTORY_BARS = 200    # tối thiểu để build cache có ý nghĩa
BUILD_TIMEOUT    = 60     # giây - tối đa chờ build cache lần đầu
FORWARD_DAYS             = [30, 60, 90]
SIMILARITY_THRESHOLDS    = [0.80, 0.75, 0.70]
MIN_RESULTS              = 3
MIN_SAMPLE_WARNING       = 5
MIN_SAMPLE_DISTANCE_DAYS = 30
MIN_SAMPLE_DISTANCE_FB   = 20

# ── Regime Filter Constants ───────────────────────────────────────────────────
# Soft weight - không loại bỏ sample, chỉ giảm "trọng số" khi tính WR/Expectancy
# R1↔R2 và R3↔R4 là "gần nhau" (cùng bull/bear, khác volatility)
# R1↔R3, R1↔R4, R2↔R3, R2↔R4 là "xa nhau" (khác bull/bear)
REGIME_WEIGHT_SAME    = 1.00   # cùng regime
REGIME_WEIGHT_CLOSE   = 0.50   # regime gần (cùng bull/bear, khác volatility)
REGIME_WEIGHT_FAR     = 0.15   # regime xa (bull vs bear)
REGIME_MIN_WEIGHTED   = 5.0    # weighted sample count tối thiểu (tránh too few)
REGIME_AUTO_EXPAND    = 8      # nếu weighted_n < này -> tự mở rộng years lên 8

# Map: (current_regime, sample_regime) -> weight
_REGIME_WEIGHT_MAP: dict[tuple[int, int], float] = {}
for _r in range(1, 5):
    _REGIME_WEIGHT_MAP[(_r, _r)] = REGIME_WEIGHT_SAME
# R1↔R2 gần (đều Bull)
_REGIME_WEIGHT_MAP[(1, 2)] = _REGIME_WEIGHT_MAP[(2, 1)] = REGIME_WEIGHT_CLOSE
# R3↔R4 gần (đều Bear)
_REGIME_WEIGHT_MAP[(3, 4)] = _REGIME_WEIGHT_MAP[(4, 3)] = REGIME_WEIGHT_CLOSE
# Còn lại là xa
for _a in range(1, 5):
    for _b in range(1, 5):
        if (_a, _b) not in _REGIME_WEIGHT_MAP:
            _REGIME_WEIGHT_MAP[(_a, _b)] = REGIME_WEIGHT_FAR


def _get_regime_weight(current_regime: int, sample_regime: int) -> float:
    """Trả về weight cho 1 sample dựa trên regime match."""
    if current_regime <= 0 or sample_regime <= 0:
        return 1.0   # không có regime info -> không filter
    return _REGIME_WEIGHT_MAP.get((current_regime, sample_regime), REGIME_WEIGHT_FAR)


def _get_current_regime() -> int:
    """
    Lấy market regime hiện tại (1-4).
    Returns 0 nếu không xác định được.
    """
    try:
        from market_regime import get_market_regime
        mr = get_market_regime()
        return int(mr.get("regime", 0)) if mr else 0
    except Exception as e:
        logger.debug(f"_get_current_regime fail: {e}")
        return 0


def _classify_date_regime(date_str: str, vnindex_regime_cache: dict) -> int:
    """
    Lookup regime của một ngày lịch sử từ cache VNINDEX regime history.

    vnindex_regime_cache: dict {date_str: regime_int} được build 1 lần
    Returns 0 nếu không tìm thấy.
    """
    if not vnindex_regime_cache:
        return 0
    return vnindex_regime_cache.get(date_str, 0)


def _build_vnindex_regime_cache(years: int = 8) -> dict[str, int]:
    """
    Build cache {date_str: regime_int} cho VNINDEX trong N năm qua.
    Dùng rolling computation từ market_regime module.
    Gọi 1 lần trong find_similar, kết quả dùng cho tất cả samples.

    Returns {} nếu không build được (graceful fallback).
    """
    try:
        from vn_loader import load_vn_ohlcv
        from market_regime import compute_regime
        days = years * 365 + 90
        df   = load_vn_ohlcv("VNINDEX", days=days, min_bars=60)
        if df is None or len(df) < 60:
            return {}

        # compute_regime trả về history_90d - nhưng chỉ 90 ngày
        # Ta cần full history -> dùng rolling window tự tính
        result   = compute_regime(df)
        history  = result.get("history_90d", [])

        # Với history_90d chỉ có 90 ngày gần nhất - không đủ cho 5-8 năm
        # Giải pháp: dùng history_90d làm base, sau đó extend bằng cách
        # recompute toàn bộ rolling (market_regime._compute_regime_history)
        try:
            from market_regime import _compute_regime_history
            import numpy as np
            close  = df["close"].values.astype(float)
            dates  = df["date"] if "date" in df.columns else df.index
            full_h = _compute_regime_history(close, dates, days=years * 252)
            cache  = {}
            for entry in full_h:
                d = entry.get("date", "")
                r = entry.get("regime", 0)
                if d:
                    cache[str(d)[:10]] = int(r)
            logger.info(f"[RegimeCache] Built {len(cache)} date->regime entries")
            return cache
        except Exception as e:
            logger.debug(f"_build_vnindex_regime_cache full history fail: {e}")

        # Fallback: chỉ dùng 90 ngày từ history_90d
        cache = {}
        for entry in history:
            d = entry.get("date", "")
            r = entry.get("regime", 0)
            if d:
                cache[str(d)[:10]] = int(r)
        logger.info(f"[RegimeCache] Fallback: {len(cache)} entries (90D only)")
        return cache

    except Exception as e:
        logger.warning(f"_build_vnindex_regime_cache error: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# CACHE PATH
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(symbol: str) -> pathlib.Path:
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / f"{symbol.upper()}{CACHE_SUFFIX}"


def cache_exists(symbol: str) -> bool:
    p = _cache_path(symbol)
    return p.exists() and p.stat().st_size > 500


# ══════════════════════════════════════════════════════════════════════════════
# BUILD / UPDATE CACHE
# ══════════════════════════════════════════════════════════════════════════════

def build_vector_cache(
    symbol: str,
    days:   int = 1500,
    progress_cb=None,
) -> tuple[bool, str]:
    """
    Build toàn bộ cache từ đầu.
    Tính vector cho từng ngày lịch sử, lưu CSV.

    Args:
        symbol:       Mã CK
        days:         Số ngày lịch sử load (default 1500 ~ 6 năm)
        progress_cb:  Callback(pct: float, msg: str) để báo tiến độ

    Returns:
        (success, message)
    """
    symbol = symbol.upper()
    t0     = time.time()

    # Load OHLCV
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=days, min_bars=MIN_HISTORY_BARS)
    except Exception as e:
        return False, f"Khong the load du lieu {symbol}: {e}"

    n = len(df)
    logger.info(f"build_vector_cache({symbol}): {n} bars, computing vectors...")

    # Tính vector cho từng ngày (bắt đầu từ bar thứ 59 để có đủ 60 bars)
    rows = []
    for i in range(59, n):
        if progress_cb and (i % 50 == 0):
            pct = (i - 59) / max(n - 59, 1) * 100
            progress_cb(pct, f"Tinh vector: {i-59}/{n-59} bars...")

        vec = compute_state_vector_for_date(df, i, market_regime=0.0)
        if vec is None:
            continue

        date_val  = df["date"].iloc[i]
        close_val = float(df["close"].iloc[i])
        row = {"date": date_val.strftime("%Y-%m-%d"), "close": close_val}
        row.update(vec)
        rows.append(row)

    if not rows:
        return False, f"Khong tinh duoc vector nao cho {symbol}"

    # Lưu CSV
    cols = ["date"] + VECTOR_KEYS + ["close"]
    cache_df = pd.DataFrame(rows, columns=cols)
    path = _cache_path(symbol)
    cache_df.to_csv(path, index=False, float_format="%.6f")

    elapsed = round(time.time() - t0, 1)
    msg = f"Cache OK: {symbol} | {len(rows)} vectors | {elapsed}s | {path}"
    logger.info(msg)
    return True, msg


def append_today_vector(symbol: str, df: pd.DataFrame = None) -> bool:
    """
    Nối vector của ngày hôm nay vào cache (gọi từ cron 18:00).
    Nếu cache chưa tồn tại -> build toàn bộ.

    Args:
        symbol:  Mã CK
        df:      DataFrame OHLCV (nếu None sẽ tự load)

    Returns:
        True nếu thành công.
    """
    symbol = symbol.upper()

    if not cache_exists(symbol):
        logger.info(f"append_today_vector({symbol}): cache chưa có -> build full")
        ok, msg = build_vector_cache(symbol)
        logger.info(msg)
        return ok

    # Load df nếu chưa có
    if df is None:
        try:
            from vn_loader import load_vn_ohlcv
            df = load_vn_ohlcv(symbol, days=300, min_bars=60)
        except Exception as e:
            logger.warning(f"append_today_vector({symbol}): load fail: {e}")
            return False

    if len(df) < 60:
        return False

    vec = compute_state_vector_from_df(df)
    if vec is None:
        return False

    today     = df["date"].iloc[-1]
    today_str = today.strftime("%Y-%m-%d") if hasattr(today, "strftime") else str(today)[:10]

    # Đọc cache hiện tại để check duplicate
    try:
        existing = pd.read_csv(_cache_path(symbol), usecols=["date"])
        if today_str in existing["date"].values:
            logger.debug(f"append_today_vector({symbol}): {today_str} đã có -> skip")
            return True
    except Exception:
        pass

    # Ghi thêm dòng mới
    close_val = float(df["close"].iloc[-1])
    row = {"date": today_str, "close": close_val}
    row.update(vec)
    cols = ["date"] + VECTOR_KEYS + ["close"]
    new_row = pd.DataFrame([row], columns=cols)

    path = _cache_path(symbol)
    # Append mode (không header)
    new_row.to_csv(path, mode="a", header=False, index=False, float_format="%.6f")
    logger.info(f"append_today_vector({symbol}): added {today_str}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def find_similar(
    symbol:        str,
    target_vector: dict,
    top_n:         int  = 3,
    years:         int  = 5,
    exclude_days:  int  = 90,
    min_results:   int  = MIN_RESULTS,
    current_regime: int = -1,   # -1 = auto-detect, 0 = disable filter
) -> list | None:
    """
    Tìm mẫu tương đồng với:
    1. Bậc thang ngưỡng: 80% -> 75% -> 70% cho đến khi đủ min_results
    2. Minimum Distance Sampling 30D (fallback 20D) -> mẫu độc lập
    3. Regime Soft Filter: weight samples theo regime match (1.0/0.5/0.15)
       - Cùng regime          -> weight 1.0 (full)
       - Regime gần (Bull/Bear group) -> weight 0.5
       - Regime khác xa       -> weight 0.15
       - current_regime=-1    -> tự động detect từ market_regime
       - current_regime=0     -> tắt filter (backward compatible)
    4. Auto-expand years 5->8 nếu weighted_n < REGIME_AUTO_EXPAND
    5. _calc_price_journey đầy đủ cho mỗi mẫu

    _meta: total_matches, independent_n, search_bars,
           avg_similarity, threshold_used, min_distance_used,
           current_regime, weighted_n, regime_filter_active
    """
    symbol = symbol.upper()
    if not cache_exists(symbol):
        return None
    try:
        cache_df = pd.read_csv(_cache_path(symbol))
    except Exception as e:
        logger.warning(f"find_similar({symbol}): {e}"); return None
    if len(cache_df) < exclude_days + 90 + 10:
        return None

    # ── Regime setup ──────────────────────────────────────────────────────────
    if current_regime == -1:
        current_regime = _get_current_regime()

    regime_filter_active = (current_regime > 0)

    # Build VNINDEX regime cache 1 lần (dùng cho tất cả samples)
    vnindex_regime_cache: dict[str, int] = {}
    if regime_filter_active:
        vnindex_regime_cache = _build_vnindex_regime_cache(years=max(years, 8))

    # ── Auto-expand years nếu cần ─────────────────────────────────────────────
    # Thử years gốc trước, nếu weighted_n quá thấp thì expand
    years_to_try = [years, 8] if years < 8 else [years]

    for years_attempt in years_to_try:
        result = _find_similar_inner(
            symbol               = symbol,
            cache_df             = cache_df,
            target_vector        = target_vector,
            top_n                = top_n,
            years                = years_attempt,
            exclude_days         = exclude_days,
            min_results          = min_results,
            current_regime       = current_regime,
            regime_filter_active = regime_filter_active,
            vnindex_regime_cache = vnindex_regime_cache,
        )
        if result is None:
            continue

        # Kiểm tra weighted_n đủ chưa
        weighted_n = result[0].get("_meta", {}).get("weighted_n", 999) if result else 999
        if weighted_n >= REGIME_AUTO_EXPAND or years_attempt == years_to_try[-1]:
            if years_attempt > years and result:
                # Gắn flag đã auto-expand
                for r in result:
                    r["_meta"]["years_expanded"] = True
                    r["_meta"]["years_used"]      = years_attempt
                logger.info(f"find_similar({symbol}): auto-expanded to {years_attempt}Y "
                            f"(weighted_n={weighted_n:.1f})")
            return result

    return None


def _find_similar_inner(
    symbol:               str,
    cache_df:             "pd.DataFrame",
    target_vector:        dict,
    top_n:                int,
    years:                int,
    exclude_days:         int,
    min_results:          int,
    current_regime:       int,
    regime_filter_active: bool,
    vnindex_regime_cache: dict,
) -> list | None:
    """
    Core search logic - tách ra để find_similar có thể retry với years khác nhau.
    """

    cutoff_start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    cutoff_end   = (datetime.now() - timedelta(days=exclude_days)).strftime("%Y-%m-%d")
    search_df    = cache_df[
        (cache_df["date"] >= cutoff_start) &
        (cache_df["date"] <= cutoff_end)
    ].copy().reset_index(drop=True)
    if len(search_df) < 3:
        return None

    avail = [c for c in VECTOR_KEYS if c in search_df.columns]
    if len(avail) < VECTOR_DIM - 2:
        return None

    # Chỉ dùng dimensions có mặt trong cả target vector và cache
    # fillna(0) gây underestimate similarity khi dimension bị thiếu
    # → loại dimension thiếu ra khỏi cả 2 vectors trước khi tính cosine
    tarr_avail = np.array([vector_to_list(target_vector)[VECTOR_KEYS.index(k)]
                           for k in avail], dtype=float)
    tnorm = np.linalg.norm(tarr_avail)
    if tnorm == 0:
        return None

    mat   = search_df[avail].fillna(0.0).values.astype(float)
    norms = np.linalg.norm(mat, axis=1)
    norms = np.where(norms == 0, 1e-9, norms)
    sims  = (mat @ tarr_avail) / (norms * tnorm)

    # ── Bậc thang ngưỡng ─────────────────────────────────────────────────────
    # Primary: chỉ lấy mẫu sim >= 0.80 (chất lượng cao)
    # Fallback: 0.75 → 0.70 chỉ khi không đủ min_results, kèm flag cảnh báo
    thresh_used   = SIMILARITY_THRESHOLDS[0]
    below_80_warn = False   # flag: phải dùng ngưỡng thấp hơn 0.80
    raw_idx       = np.array([], dtype=int)
    for thresh in SIMILARITY_THRESHOLDS:
        cands = np.where(sims >= thresh)[0]
        if len(cands) >= min_results:
            raw_idx     = cands
            thresh_used = thresh
            if thresh < 0.80:
                below_80_warn = True
            break
        raw_idx     = cands
        thresh_used = thresh
    if len(raw_idx) == 0:
        raw_idx       = np.argsort(-sims)[:max(top_n, 3)]
        thresh_used   = float(sims[raw_idx[-1]]) if len(raw_idx) else 0.0
        below_80_warn = True

    total_matches = len(raw_idx)
    search_bars   = len(search_df)

    # ── Gắn regime weight cho từng sample ────────────────────────────────────
    # regime_weights[i] = weight của candidate i trong raw_idx
    regime_weights: dict[int, float] = {}
    sample_regimes: dict[int, int]   = {}

    if regime_filter_active and vnindex_regime_cache:
        for idx in raw_idx:
            date_str = str(search_df.iloc[idx]["date"])[:10]
            sample_r = _classify_date_regime(date_str, vnindex_regime_cache)
            w        = _get_regime_weight(current_regime, sample_r)
            regime_weights[idx] = w
            sample_regimes[idx] = sample_r
    else:
        for idx in raw_idx:
            regime_weights[idx] = 1.0
            sample_regimes[idx] = 0

    # Sort theo thời gian -> Minimum Distance Sampling
    # Sort by sim DESC trước để _mds giữ sample có sim cao nhất trong mỗi cluster
    pairs_sim_desc = sorted(
        [(str(search_df.iloc[i]["date"]), i) for i in raw_idx],
        key=lambda x: -sims[x[1]]   # sim cao nhất trước
    )

    def _mds(pairs_by_sim, md):
        """
        Minimum Distance Sampling — giữ mẫu có similarity CAO NHẤT trong mỗi cluster.
        pairs_by_sim: đã sorted by sim DESC → mẫu đầu tiên không bị loại là mẫu tốt nhất.
        """
        kept      = []
        kept_dates = []
        for d, i in pairs_by_sim:
            try:
                dt_cur = datetime.strptime(d, "%Y-%m-%d")
                too_close = any(
                    abs((dt_cur - datetime.strptime(ld, "%Y-%m-%d")).days) < md
                    for ld in kept_dates
                )
                if not too_close:
                    kept.append((d, i))
                    kept_dates.append(d)
            except Exception:
                continue
        return kept

    min_dist = MIN_SAMPLE_DISTANCE_DAYS
    kept     = _mds(pairs_sim_desc, min_dist)
    if len(kept) < min_results:
        kept_fb = _mds(pairs_sim_desc, MIN_SAMPLE_DISTANCE_FB)
        if len(kept_fb) >= min_results:
            kept = kept_fb; min_dist = MIN_SAMPLE_DISTANCE_FB
        else:
            kept = kept_fb; min_dist = MIN_SAMPLE_DISTANCE_FB

    kept_s  = sorted(kept, key=lambda x: -sims[x[1]])
    ind_n   = len(kept_s)
    avg_sim = float(np.mean([sims[i] for _, i in kept_s]))

    # Tính weighted_n - effective sample count sau regime weighting
    weighted_n = sum(regime_weights.get(i, 1.0) for _, i in kept_s)

    # Regime breakdown — đếm số mẫu theo từng nhóm để hiển thị rõ
    regime_breakdown = {"same": 0, "close": 0, "far": 0, "unknown": 0}
    if regime_filter_active:
        for _, idx in kept_s:
            w = regime_weights.get(idx, 1.0)
            if w >= 1.0:
                regime_breakdown["same"]  += 1
            elif w >= 0.5:
                regime_breakdown["close"] += 1
            elif w > 0:
                regime_breakdown["far"]   += 1
            else:
                regime_breakdown["unknown"] += 1

    results = []
    for date_str, idx in kept_s:
        row        = search_df.iloc[idx]
        close_val  = float(row.get("close", 0))
        j          = _calc_price_journey(cache_df, date_str, close_val)
        vec_vals   = {k: float(row[k]) for k in VECTOR_KEYS if k in row and pd.notna(row[k])}
        r_weight   = regime_weights.get(idx, 1.0)
        r_regime   = sample_regimes.get(idx, 0)

        results.append({
            "date":             date_str,
            "similarity":       round(float(sims[idx]), 4),
            "close":            close_val,
            "fwd_30":           j["fwd_30"],
            "fwd_60":           j["fwd_60"],
            "fwd_90":           j["fwd_90"],
            "max_gain":         j["max_gain"],
            "max_gain_day":     j["max_gain_day"],
            "max_drawdown":     j["max_drawdown"],
            "max_dd_day":       j["max_dd_day"],
            "max_drawdown_30d": j.get("max_drawdown_30d"),
            "daily_volatility": j["daily_volatility"],
            "conclusion":       j["conclusion"],
            "recovery_days":    j.get("recovery_days"),
            "outcome":          _classify_outcome(j["fwd_30"]),
            "regime_weight":    round(r_weight, 2),   # field mới
            "sample_regime":    r_regime,              # field mới
            **vec_vals,
            "_meta": {
                "total_matches":        total_matches,
                "independent_n":        ind_n,
                "search_bars":          search_bars,
                "avg_similarity":       round(avg_sim, 4),
                "threshold_used":       thresh_used,
                "below_80_warn":        below_80_warn,
                "min_distance_used":    min_dist,
                "current_regime":       current_regime,
                "weighted_n":           round(weighted_n, 2),
                "regime_filter_active": regime_filter_active,
                "regime_breakdown":     regime_breakdown,
                "years_expanded":       False,
                "years_used":           years,
            },
        })
    return results if results else None


def _apply_min_distance_filter(sorted_dates: list, min_days: int) -> list:
    """Minimum Distance Sampling - chỉ giữ ngày cách nhau >= min_days."""
    if not sorted_dates:
        return []
    kept = [sorted_dates[0]]; last = sorted_dates[0]
    for d in sorted_dates[1:]:
        try:
            if (datetime.strptime(d, "%Y-%m-%d") -
                    datetime.strptime(last, "%Y-%m-%d")).days >= min_days:
                kept.append(d); last = d
        except Exception:
            continue
    return kept


def _calc_price_journey(cache_df, from_date: str, from_close: float,
                        horizon: int = 90) -> dict:
    """
    Phân tích hành trình giá trong horizon ngày tiếp theo.
    Trả về: fwd_30/60/90, max_gain, max_drawdown, daily_volatility, conclusion.
    """
    result = {
        "fwd_30": None, "fwd_60": None, "fwd_90": None,
        "max_gain": None, "max_gain_day": None,
        "max_drawdown": None, "max_dd_day": None,
        "max_drawdown_30d": None,   # MAE trong 30D đầu - nhất quán với TP1/TP2
        "daily_volatility": None, "conclusion": "CHUA RO",
    }
    if from_close <= 0:
        return result
    future = cache_df[cache_df["date"] > from_date].reset_index(drop=True)
    if len(future) < 5:
        return result

    # future.iloc[0] = T+1, future.iloc[29] = T+30 → dùng iloc[days-1] để lấy đúng T+days
    for days, key in [(30, "fwd_30"), (60, "fwd_60"), (90, "fwd_90")]:
        if len(future) >= days:
            fc = float(future.iloc[days - 1]["close"])
            result[key] = round((fc - from_close) / from_close * 100, 2)

    closes = future.head(horizon)["close"].values.astype(float)
    pct    = (closes - from_close) / from_close * 100

    mg_idx = int(np.argmax(pct))
    result["max_gain"]     = round(float(pct[mg_idx]), 2)
    result["max_gain_day"] = int(mg_idx + 1)

    peak   = np.maximum.accumulate(pct)
    dd     = pct - peak
    md_idx = int(np.argmin(dd))
    result["max_drawdown"] = round(float(dd[md_idx]), 2)
    result["max_dd_day"]   = int(md_idx + 1)

    # MAE trong 30D đầu - dùng cho SL (nhất quán khung thời gian với TP1/TP2)
    # Tính từ entry price (min của pct trong 30D), không dùng running peak
    # Đây là MAE chuẩn: mức xuống tệ nhất so với điểm vào lệnh
    pct_30 = pct[:30] if len(pct) >= 30 else pct
    if len(pct_30) > 0:
        result["max_drawdown_30d"] = round(float(pct_30.min()), 2)

    if len(closes) > 1:
        result["daily_volatility"] = round(
            float(np.mean(np.abs(np.diff(closes) / closes[:-1] * 100))), 2)

    # Tính số ngày hồi phục về hoà vốn sau MAE
    # max_dd_day là 1-indexed (ngày 1 = closes[0]) → convert sang 0-indexed
    result["recovery_days"] = None
    if result["max_drawdown"] is not None and result["max_drawdown"] < 0 and result["max_dd_day"] is not None:
        dd_idx_0 = result["max_dd_day"] - 1   # convert 1-indexed → 0-indexed
        post_dd  = closes[dd_idx_0 + 1:]       # slice từ ngày SAU điểm drawdown
        for k, c in enumerate(post_dd):
            if from_close > 0 and c >= from_close:
                result["recovery_days"] = k + 1
                break

    f30 = result["fwd_30"]; mdd = result["max_drawdown"] or 0
    if f30 is None:             result["conclusion"] = "CHUA RO"
    elif f30 >= 8 and mdd > -8: result["conclusion"] = "TANG MANH, it rung lac"
    elif f30 >= 5:              result["conclusion"] = "TANG MANH"
    elif f30 >= 2:              result["conclusion"] = "TANG NHE"
    elif f30 <= -8:             result["conclusion"] = "GIAM MANH"
    elif f30 <= -3:             result["conclusion"] = "GIAM"
    else:                       result["conclusion"] = "DI NGANG"
    return result


def _calc_forward_returns(
    cache_df:  pd.DataFrame,
    from_date: str,
    from_close: float,
) -> dict:
    """Tính % return sau 30/60/90 ngày từ cache."""
    result = {}
    future = cache_df[cache_df["date"] > from_date].reset_index(drop=True)

    for days in FORWARD_DAYS:
        # future.iloc[0] = T+1, future.iloc[days-1] = T+days → đúng convention
        if len(future) >= days:
            future_close = float(future.iloc[days - 1]["close"])
            if from_close > 0:
                result[days] = round((future_close - from_close) / from_close * 100, 2)
            else:
                result[days] = None
        else:
            result[days] = None  # Chưa đủ dữ liệu tương lai

    return result


def _classify_outcome(fwd_30: Optional[float]) -> str:
    """Phân loại kết quả sau 30 ngày."""
    if fwd_30 is None:
        return "CHUA RO"
    if fwd_30 >= 5:
        return "TANG MANH"
    if fwd_30 >= 2:
        return "TANG"
    if fwd_30 <= -5:
        return "GIAM MANH"
    if fwd_30 <= -2:
        return "GIAM"
    return "DI NGANG"


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _best_holding(analogs: list) -> str:
    """
    CHANGE 1: Dung median(max_gain_day) lam thoi gian nam giu khuyen nghi.
    Round ve boi so 5 gan nhat. Thay WR cao nhat de tranh bull bias dai han.
    """
    gain_days = [a["max_gain_day"] for a in analogs if a.get("max_gain_day") is not None]
    if not gain_days:
        return "N/A"
    median_hold = float(np.median(gain_days))
    recommended = max(5, round(median_hold / 5) * 5)
    vals_90 = [a["fwd_90"] for a in analogs if a.get("fwd_90") is not None]
    wr_90_str = ""
    if vals_90:
        wr_90 = sum(1 for v in vals_90 if v > 0) / len(vals_90)
        wr_90_str = f" | WR 90D={wr_90:.0%}"
    return f"{recommended}D (median dinh: {median_hold:.0f}D{wr_90_str})"


def _format_big_wave(analogs: list) -> list[str]:
    """
    Big Wave Analysis - Đặc điểm Sóng Lớn (top 30% fwd_90).

    Thiết kế:
    - Lấy tất cả analogs có fwd_90 không None -> tính ngưỡng p70 (top 30%)
    - Nhóm "Sóng Lớn" = analogs có fwd_90 >= p70
    - So sánh phân phối vector của Sóng Lớn vs phần còn lại
    - Highlight top 3-5 dimension phân kỳ rõ nhất
    - Hiển thị range [min, max] thay vì con số dứt khoát
    - Tối thiểu 3 mẫu Sóng Lớn mới hiển thị
    """
    MIN_BIG_WAVE = 3   # tối thiểu mẫu Sóng Lớn
    DIVERGE_THRESH = 0.18   # thấp hơn 0.20 vì so sánh với "phần còn lại" rộng hơn

    # ── Bước 1: Lọc analogs có fwd_90 và vector ──────────────────────────
    pool = []
    for a in analogs:
        fwd90 = a.get("fwd_90")
        if fwd90 is None:
            continue
        vec = {k: a.get(k) for k in VECTOR_KEYS if a.get(k) is not None}
        if len(vec) < 10:
            continue
        pool.append({"fwd_90": fwd90, "vec": vec, "date": a.get("date", "?")})

    if len(pool) < MIN_BIG_WAVE + 2:   # cần đủ cả 2 nhóm
        return []

    # ── Bước 2: Tính ngưỡng p70 -> top 30% = Sóng Lớn ────────────────────
    fwd90_vals = np.array([e["fwd_90"] for e in pool])
    p70 = float(np.percentile(fwd90_vals, 70))

    big_wave = [e for e in pool if e["fwd_90"] >= p70]
    rest     = [e for e in pool if e["fwd_90"] <  p70]

    nb, nr = len(big_wave), len(rest)
    if nb < MIN_BIG_WAVE:
        return []

    # ── Bước 3: Tính stats cho 2 nhóm ────────────────────────────────────
    def _stats(entries: list) -> dict:
        out = {}
        for dim in VECTOR_KEYS:
            vals = [e["vec"][dim] for e in entries if dim in e["vec"]]
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            out[dim] = {
                "mean": float(np.mean(arr)),
                "min":  float(np.min(arr)),
                "max":  float(np.max(arr)),
                "std":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            }
        return out

    bw_stats   = _stats(big_wave)
    rest_stats = _stats(rest)

    # ── Bước 4: Tìm dimension phân kỳ ────────────────────────────────────
    divergent = []
    for dim in VECTOR_KEYS:
        if dim not in bw_stats or dim not in rest_stats:
            continue
        mb   = bw_stats[dim]["mean"]
        mr   = rest_stats[dim]["mean"]
        diff = mb - mr
        noise = 0.5 * (bw_stats[dim]["std"] + rest_stats[dim]["std"])
        if abs(diff) >= DIVERGE_THRESH and abs(diff) > noise:
            divergent.append((
                dim, diff, mb, mr,
                bw_stats[dim]["min"],  bw_stats[dim]["max"],
                rest_stats[dim]["min"], rest_stats[dim]["max"],
            ))

    divergent.sort(key=lambda x: -abs(x[1]))
    top_div = divergent[:5]

    # ── Bước 5: Labels ───────────────────────────────────────────────────
    _DIM_LABEL = {
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
    _HIGH_MEANING = {
        "rsi_norm":        "RSI trung tinh-thap",
        "macd_hist_norm":  "MACD Hist duong manh",
        "bb_position":     "gia giua BB",
        "volume_spike":    "volume dot bien tang",
        "trend_slope":     "SMA20 cat len SMA50",
        "price_vs_sma20":  "gia tren SMA20",
        "price_vs_sma50":  "gia tren SMA50",
        "atr_ratio":       "bien dong cao (ATR lon)",
        "stoch_k_norm":    "Stoch K tang",
        "ema_cross":       "EMA12 vuot EMA26",
        "momentum_5d":     "tang gia 5 phien gan",
        "momentum_20d":    "xu huong tang 20 phien",
        "high_low_pos":    "gia phan hoi tu day",
        "vol_trend":       "vol ngan han tang manh",
        "candle_body":     "than nen day (quyet doan)",
    }
    _LOW_MEANING = {
        "rsi_norm":        "RSI cao (overbought)",
        "macd_hist_norm":  "MACD Hist am/yeu",
        "bb_position":     "gia sat BB upper/lower",
        "volume_spike":    "volume binh thuong",
        "trend_slope":     "SMA20 duoi SMA50",
        "price_vs_sma20":  "gia duoi SMA20",
        "price_vs_sma50":  "gia duoi SMA50",
        "atr_ratio":       "bien dong thap (tich luy)",
        "stoch_k_norm":    "Stoch K giam",
        "ema_cross":       "EMA12 duoi EMA26",
        "momentum_5d":     "gia phang/giam ngan han",
        "momentum_20d":    "xu huong giam 20 phien",
        "high_low_pos":    "gia o vung giua",
        "vol_trend":       "vol on dinh/giam",
        "candle_body":     "than nen nho (do du)",
    }

    # ── Bước 6: Format ────────────────────────────────────────────────────
    bw_fwd90_mean = float(np.mean([e["fwd_90"] for e in big_wave]))
    rt_fwd90_mean = float(np.mean([e["fwd_90"] for e in rest]))

    lines = [
        "",
        "═" * 38,
        f"SONG LON: TOP 30% FWD90 ({nb} mau, nguong >= {p70:+.1f}%)",
        "═" * 38,
        f"⚠️  Chi {nb} mau Song Lon - mau rat nho, chi mang tinh tham khao.",
        "    Khong su dung de ra quyet dinh giao dich.",
        "─" * 38,
        f"Song Lon ({nb} mau): fwd90 TB = {bw_fwd90_mean:+.1f}%",
        f"Phan con lai ({nr} mau): fwd90 TB = {rt_fwd90_mean:+.1f}%",
        "",
    ]

    if not top_div:
        lines.append("Khong co dimension nao phan ky ro rang (< 0.18 delta).")
        lines.append("=> Song Lon va phan con lai co boi canh ky thuat tuong tu.")
    else:
        lines.append(f"TOP {len(top_div)} DIMENSION KHAC BIET (|delta| >= 0.18):")
        lines.append(
            f"  {'Dimension':<18} {'Song Lon':>14}  {'Con lai':>14}  {'Delta':>7}"
        )
        lines.append("  " + "-" * 57)
        for dim, diff, mb, mr, bmin, bmax, rmin, rmax in top_div:
            label = _DIM_LABEL.get(dim, dim)
            arrow = "▲ " if diff > 0 else "▼ "
            lines.append(
                f"  {label:<18} "
                f"{mb:>+6.2f}[{bmin:+.2f},{bmax:+.2f}]  "
                f"{mr:>+6.2f}[{rmin:+.2f},{rmax:+.2f}]  "
                f"{arrow}{abs(diff):.2f}"
            )

        # Mô tả 1 dòng tóm tắt
        bw_chars = []
        for dim, diff, *_ in top_div[:3]:
            bw_chars.append(
                _HIGH_MEANING.get(dim, f"{dim} cao") if diff > 0
                else _LOW_MEANING.get(dim, f"{dim} thap")
            )
        lines.append("")
        lines.append(
            f"  Song Lon thuong gap: {', '.join(bw_chars)}."
        )
        lines.append(
            "  Luu y: Day la mo ta thong ke, khong phai dieu kien du/can."
        )

    lines.append("─" * 38)
    return lines


def format_analog_report(
    symbol:        str,
    analogs:       list,
    current_vec:   dict,
    max_chars:     int   = 4000,
    current_price: float = 0.0,   # gia hien tai thuc te - PHAI TRUYEN TU CALLER
) -> str:
    """
    Bao cao 4 phan - KHONG liet ke tung ngay.
    1. Tom tat nhanh (mau doc lap, MDS, nguong)
    2. Tom tat hanh trinh gia (MFE/MAE, capture rate, hold)
    3. Thong ke day du dong bo scan_watchlist
    4. Canh bao rui ro
    current_price: bat buoc truyen gia hien tai thuc te tu caller (vd: df["close"].iloc[-1])
                   de hien Ke hoach hanh dong dung. Neu = 0 -> section bi an.
    """
    if not analogs:
        return f"Khong tim thay ngay tuong dong cho {symbol}."

    meta     = analogs[0].get("_meta", {})
    ind_n    = meta.get("independent_n",    len(analogs))
    srch_bars= meta.get("search_bars",       0)
    avg_sim  = meta.get("avg_similarity",    0.0)
    thresh   = meta.get("threshold_used",    0.80)
    min_dist = meta.get("min_distance_used", MIN_SAMPLE_DISTANCE_DAYS)
    total_m  = meta.get("total_matches",     len(analogs))

    def _vals(key): return [a[key] for a in analogs if a.get(key) is not None]
    vf30  = _vals("fwd_30"); vf60 = _vals("fwd_60"); vf90 = _vals("fwd_90")
    vmg   = _vals("max_gain"); vmdd = _vals("max_drawdown")
    vmdd30= _vals("max_drawdown_30d"); vmgd = _vals("max_gain_day")
    n     = len(vf30)

    # ── Stats ─────────────────────────────────────────────────────────
    _WIN_THRESH = 1.0   # +1% tối thiểu để tính là "thắng" sau phí
    wr      = len([x for x in vf30 if x >= _WIN_THRESH]) / len(vf30) if vf30 else 0.0
    med30   = float(np.median(vf30))      if vf30  else 0.0
    med60   = float(np.median(vf60))      if vf60  else 0.0
    med90   = float(np.median(vf90))      if vf90  else 0.0
    p75_30  = float(np.percentile(vf30, 75)) if n >= 4 else med30
    exp30   = float(np.mean(vf30))        if vf30  else 0.0
    pos_s   = sum(x for x in vf30 if x >= _WIN_THRESH)
    neg_s   = abs(sum(x for x in vf30 if x < _WIN_THRESH))
    pf      = round(pos_s / neg_s, 2)     if neg_s > 0 else 99.0
    mae_30  = float(np.median(vmdd30))    if vmdd30 else (float(np.mean(vmdd)) if vmdd else 0.0)
    mfe_avg = float(np.mean(vmg))         if vmg   else 0.0
    hold_avg= float(np.mean(vmgd))        if vmgd  else 0.0
    be_rr   = round((1 - wr) / wr, 1)    if wr > 0 else 99.0

    # Capture rate
    caps = [a["fwd_30"] / a["max_gain"] * 100
            for a in analogs
            if a.get("fwd_30") is not None and a.get("max_gain") and a["max_gain"] > 0]
    cap_rate = float(np.mean(caps)) if caps else 0.0

    # Entry/SL/TP
    entry_price = float(current_price) if current_price and current_price > 0 else 0.0
    sl_pct   = mae_30 - 2.0
    # Sanity check: SL phải luôn dưới entry (sl_pct âm). Nếu MAE = 0, dùng -2% mặc định.
    if sl_pct >= 0:
        sl_pct = -2.0
    sl_price = entry_price * (1 + sl_pct / 100)  if entry_price else 0.0
    tp1_price= entry_price * (1 + med30 / 100)   if entry_price else 0.0
    tp2_price= entry_price * (1 + p75_30 / 100)  if entry_price else 0.0
    risk_abs = abs(entry_price - sl_price)        if entry_price else 0.0
    rr1 = round(abs(tp1_price - entry_price) / risk_abs, 1) if risk_abs > 0 else 0.0
    rr2 = round(abs(tp2_price - entry_price) / risk_abs, 1) if risk_abs > 0 else 0.0
    rr_best = rr2 if rr2 > 0 else rr1

    # Verdict
    if wr >= 0.65 and rr_best >= be_rr:
        em_v = "✅"; verdict_txt = f"Co hoi tot: WR {wr:.0%}, R:R 1:{rr_best}, PF {pf:.2f}"
    elif wr < 0.50 or (rr_best > 0 and rr_best < be_rr):
        em_v = "🔴"; verdict_txt = f"Can than: WR {wr:.0%} hoac R:R 1:{rr_best} < break-even 1:{be_rr}"
    else:
        em_v = "🟡"; verdict_txt = f"Theo doi them: WR {wr:.0%}, R:R 1:{rr_best}"

    # Warnings inline
    warn_parts = []
    if avg_sim < 0.85: warn_parts.append(f"Do TD {avg_sim:.0%}<85%")
    if ind_n < 5:      warn_parts.append(f"n={ind_n} mau nho")
    if thresh < 0.80:  warn_parts.append(f"nguong ha {thresh:.0%}")
    if mae_30 < -15:   warn_parts.append(f"MAE cao {mae_30:.1f}%")
    cap_note = ""
    if cap_rate > 0:
        cap_note = " ⚠️ Exit som" if cap_rate < 40 else " ✅ Hieu qua" if cap_rate > 80 else ""

    # Wave summary - load từ wave_pattern nếu có
    wave_lines = []
    try:
        from wave_pattern import analyze_wave
        _wr = analyze_wave(symbol, force_rebuild=False)
        if _wr and _wr.get("ok"):
            wd   = _wr.get("verdict", "KHONG RO")
            ws_up  = _wr.get("score_up_adj",   _wr.get("score_up",   0))
            ws_dn  = _wr.get("score_down_adj", _wr.get("score_down", 0))
            wconf  = _wr.get("confidence", 0)
            wn     = min(_wr.get("n_up", 0), _wr.get("n_down", 0))
            def _wrel(n, c):
                if n >= 20 and c >= 0.15: return "★★★"
                if n >= 15 and c >= 0.10: return "★★☆"
                return "★☆☆"
            wrel = _wrel(wn, wconf)
            w_em = "🟢" if wd == "SONG TANG" else "🔴" if wd == "SONG GIAM" else "🟡"
            wave_lines.append(
                f"🌊 Wave: {w_em} {wd} {wrel} "
                f"({ws_dn:.0%} giam vs {ws_up:.0%} tang)"
            )
            # Subtype
            if wd == "SONG GIAM":
                st  = _wr.get("subtype_stats_down", {})
                cmp = _wr.get("subtype_compare_down", {})
            elif wd == "SONG TANG":
                st  = _wr.get("subtype_stats_up", {})
                cmp = _wr.get("subtype_compare_up", {})
            else:
                st = {}; cmp = {}
            if st and st.get("total_known", 0) >= 6:
                pct = st.get("pct", {})
                closest = cmp.get("closest", "")
                if wd == "SONG GIAM":
                    cl_lbl = "Peak that (giam sau)" if closest == "PEAK_REAL" else "Correction (giam roi len)"
                    cl_em  = "🔴" if closest == "PEAK_REAL" else "🟡"
                    wave_lines.append(
                        f"   Loai: Peak that {pct.get('PEAK_REAL',0):.0f}% | "
                        f"Correction {pct.get('CORRECTION',0):.0f}% | "
                        f"{cl_em} Gan nhat: {cl_lbl}"
                    )
                elif wd == "SONG TANG":
                    cl_lbl = "Day that (tang ben vung)" if closest == "BOTTOM_REAL" else "Relief Rally (tang ngan)"
                    cl_em  = "🟢" if closest == "BOTTOM_REAL" else "🟡"
                    wave_lines.append(
                        f"   Loai: Day that {pct.get('BOTTOM_REAL',0):.0f}% | "
                        f"Relief Rally {pct.get('RELIEF_RALLY',0):.0f}% | "
                        f"{cl_em} Gan nhat: {cl_lbl}"
                    )
    except Exception:
        pass  # wave không bắt buộc

    # ── Assemble ──────────────────────────────────────────────────────
    sep = "═" * 32
    lines = [
        f"/backtest {symbol}  [{srch_bars}n | {ind_n} mau | Do TD: {avg_sim:.0%}]",
        sep,
        f"📊 WR {wr:.0%}  |  PF {pf:.2f}  |  Expectancy {exp30:+.2f}%",
        f"   LN: 30D {med30:+.1f}% | 60D {med60:+.1f}% | 90D {med90:+.1f}%",
        f"   MAE {mae_30:+.1f}%  |  MFE {mfe_avg:+.1f}%  |  Hold ~{hold_avg:.0f}D"
        + (f"  |  Capture {cap_rate:.0f}%{cap_note}" if cap_rate else ""),
    ]

    # Entry/SL/TP
    if entry_price and sl_price and tp1_price:
        lines += [
            "─" * 32,
            f"📌 Entry: {entry_price:,.0f}  |  SL: {sl_price:,.0f} ({sl_pct:.1f}%)",
            f"   TP1: {tp1_price:,.0f} ({med30:+.1f}%)  R:R 1:{rr1}"
            + (f"  |  TP2: {tp2_price:,.0f} ({p75_30:+.1f}%)  R:R 1:{rr2}" if rr2 != rr1 else ""),
        ]

    # Wave
    if wave_lines:
        lines.append("─" * 32)
        lines += wave_lines

    # Stock Regime (Phase 2 GMM)
    try:
        from stock_regime import get_stock_regime, format_stock_regime_for_backtest
        _sr = get_stock_regime(symbol)
        _sr_block = format_stock_regime_for_backtest(_sr)
        if _sr_block:
            lines.append("─" * 32)
            lines += _sr_block.splitlines()
    except Exception:
        pass

    # Verdict + warnings
    lines.append("─" * 32)
    lines.append(f"{em_v} {verdict_txt}")
    if warn_parts:
        lines.append(f"   ⚠️ {' | '.join(warn_parts)}")
    lines.append(sep)

    return "\n".join(lines)



# ══════════════════════════════════════════════════════════════════════════════
# CRON UPDATE (gọi từ _start_cron trong bot.py)
# ══════════════════════════════════════════════════════════════════════════════

def update_all_caches(watchlist: list) -> str:
    """
    Cập nhật cache cho tất cả mã trong watchlist.
    Gọi từ cron 18:00 mỗi ngày.
    Returns: summary string.
    """
    ok_list, fail_list = [], []
    for symbol in watchlist:
        try:
            ok = append_today_vector(symbol)
            if ok:
                ok_list.append(symbol)
            else:
                fail_list.append(symbol)
        except Exception as e:
            fail_list.append(f"{symbol}({e})")

    return (
        f"Vector cache update: {len(ok_list)} OK, {len(fail_list)} fail. "
        f"{'Fail: ' + ', '.join(fail_list) if fail_list else ''}"
    )
