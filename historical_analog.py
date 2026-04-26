"""
historical_analog.py — Phân tích tương đồng lịch sử (Historical Analog).

FLOW:
  1. Lần đầu: build_vector_cache(symbol) → tính vector cho từng ngày lịch sử,
     lưu vào data/{symbol}_vectors.csv.
  2. Hàng ngày (cron 18:00): append_today_vector(symbol) → nối dòng mới.
  3. Khi cần tìm: find_similar(symbol, target_vector, top_n=3, years=5)
     → load CSV, cosine similarity, trả về top N ngày kèm forward returns.

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
BUILD_TIMEOUT    = 60     # giây — tối đa chờ build cache lần đầu
FORWARD_DAYS            = [30, 60, 90]   # ngày kiểm tra forward performance
SIMILARITY_THRESHOLDS   = [0.80, 0.75, 0.70]  # bậc thang ngưỡng
MIN_RESULTS             = 3      # tối thiểu trước khi hạ ngưỡng
MAX_DISPLAY             = 10     # tối đa hiển thị chi tiết
MIN_SAMPLE_WARNING      = 5      # dưới này cảnh báo mẫu nhỏ
MIN_SAMPLE_DISTANCE_DAYS = 30    # khoảng cách tối thiểu giữa 2 mẫu độc lập
MIN_SAMPLE_DISTANCE_FALLBACK = 20  # giảm xuống nếu mẫu sau lọc < MIN_RESULTS


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
    Nếu cache chưa tồn tại → build toàn bộ.

    Args:
        symbol:  Mã CK
        df:      DataFrame OHLCV (nếu None sẽ tự load)

    Returns:
        True nếu thành công.
    """
    symbol = symbol.upper()

    if not cache_exists(symbol):
        logger.info(f"append_today_vector({symbol}): cache chưa có → build full")
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
            logger.debug(f"append_today_vector({symbol}): {today_str} đã có → skip")
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
    top_n:         int = 3,       # giữ để backward-compat
    years:         int = 5,
    exclude_days:  int = 90,
    sim_threshold: float = SIMILARITY_THRESHOLDS[0],
    min_results:   int  = MIN_RESULTS,
) -> Optional[list]:
    """
    Tìm TẤT CẢ ngày lịch sử có similarity >= sim_threshold, sau đó:
      1. Bậc thang ngưỡng: 80% → 75% → 70% nếu < min_results kết quả
      2. Minimum Distance Sampling (>=30 ngày): loại bỏ các ngày trong
         cùng một "sự kiện thị trường" — đảm bảo mẫu độc lập về mặt thống kê
      3. Nếu sau lọc < MIN_RESULTS → thử giảm khoảng cách xuống 20 ngày
      4. Tính _calc_price_journey đầy đủ cho từng mẫu

    _meta trong mỗi kết quả chứa:
        total_matches     : số ngày thô vượt ngưỡng (trước lọc khoảng cách)
        independent_n     : số mẫu độc lập (sau lọc khoảng cách) — dùng để tính stats
        search_bars       : số ngày đã tìm kiếm
        avg_similarity    : similarity TB của mẫu độc lập
        threshold_used    : ngưỡng similarity thực sự dùng
        min_distance_used : khoảng cách tối thiểu đã dùng (30 hoặc 20)
    """
    symbol = symbol.upper()
    if not cache_exists(symbol):
        return None

    try:
        t0       = time.time()
        cache_df = pd.read_csv(_cache_path(symbol))
        logger.debug(f"find_similar({symbol}): {len(cache_df)} rows in {time.time()-t0:.2f}s")
    except Exception as e:
        logger.warning(f"find_similar({symbol}): load fail: {e}")
        return None

    if len(cache_df) < exclude_days + 90 + 10:
        return None

    cutoff_start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    cutoff_end   = (datetime.now() - timedelta(days=exclude_days)).strftime("%Y-%m-%d")
    search_df    = cache_df[
        (cache_df["date"] >= cutoff_start) &
        (cache_df["date"] <= cutoff_end)
    ].copy().reset_index(drop=True)

    if len(search_df) < 3:
        return None

    # Vectorized cosine similarity
    target_arr = np.array(vector_to_list(target_vector), dtype=float)
    t_norm     = np.linalg.norm(target_arr)
    if t_norm == 0:
        return None

    available_cols = [c for c in VECTOR_KEYS if c in search_df.columns]
    if len(available_cols) < VECTOR_DIM - 2:
        return None

    mat          = search_df[available_cols].fillna(0.0).values.astype(float)
    similarities = (mat @ target_arr) / (
        np.where(np.linalg.norm(mat, axis=1) == 0, 1e-9, np.linalg.norm(mat, axis=1)) * t_norm
    )

    # ── Bậc thang ngưỡng ─────────────────────────────────────────────
    threshold_used = SIMILARITY_THRESHOLDS[0]
    raw_indices    = np.array([], dtype=int)
    for thresh in SIMILARITY_THRESHOLDS:
        cands = np.where(similarities >= thresh)[0]
        if len(cands) >= min_results:
            raw_indices    = cands
            threshold_used = thresh
            break
        raw_indices    = cands
        threshold_used = thresh   # giữ ngưỡng thấp nhất đã thử

    if len(raw_indices) == 0:
        # Fallback tuyệt đối: top_n cao nhất
        raw_indices    = np.argsort(-similarities)[:max(top_n, 3)]
        threshold_used = float(similarities[raw_indices[-1]]) if len(raw_indices) else 0.0

    # Sort raw_indices theo ngày (thời gian tăng dần) để Minimum Distance Sampling
    raw_dates_sorted = sorted(
        [(str(search_df.iloc[i]["date"]), i) for i in raw_indices],
        key=lambda x: x[0]
    )

    total_matches = len(raw_indices)
    search_bars   = len(search_df)

    # ── Minimum Distance Sampling ─────────────────────────────────────
    def _filter_with_distance(pairs, min_d):
        kept = [pairs[0]]
        last_date = pairs[0][0]
        for d, i in pairs[1:]:
            try:
                delta = (datetime.strptime(d, "%Y-%m-%d") -
                         datetime.strptime(last_date, "%Y-%m-%d")).days
                if delta >= min_d:
                    kept.append((d, i))
                    last_date = d
            except Exception:
                continue
        return kept

    min_dist_used = MIN_SAMPLE_DISTANCE_DAYS
    kept_pairs    = _filter_with_distance(raw_dates_sorted, min_dist_used)

    # Nếu sau lọc < MIN_RESULTS → thử khoảng cách fallback 20 ngày
    if len(kept_pairs) < MIN_RESULTS:
        min_dist_fallback = MIN_SAMPLE_DISTANCE_FALLBACK
        kept_pairs_fb     = _filter_with_distance(raw_dates_sorted, min_dist_fallback)
        if len(kept_pairs_fb) >= MIN_RESULTS:
            kept_pairs    = kept_pairs_fb
            min_dist_used = min_dist_fallback
            logger.info(
                f"find_similar({symbol}): sau lọc {MIN_SAMPLE_DISTANCE_DAYS}D chỉ có "
                f"{len(kept_pairs)} mẫu → giảm xuống {min_dist_fallback}D "
                f"({len(kept_pairs_fb)} mẫu)"
            )
        else:
            kept_pairs    = kept_pairs_fb   # dù ít vẫn dùng, caller sẽ cảnh báo
            min_dist_used = min_dist_fallback

    # Sort kết quả cuối theo similarity giảm dần
    kept_pairs_by_sim = sorted(kept_pairs, key=lambda x: -similarities[x[1]])
    independent_n     = len(kept_pairs_by_sim)
    avg_sim           = float(np.mean([similarities[i] for _, i in kept_pairs_by_sim]))

    # ── Tính price journey ────────────────────────────────────────────
    results = []
    for date_str, idx in kept_pairs_by_sim:
        row       = search_df.iloc[idx]
        sim       = float(similarities[idx])
        close_val = float(row.get("close", 0))
        journey   = _calc_price_journey(cache_df, date_str, close_val, horizon=90)

        results.append({
            "date":             date_str,
            "similarity":       round(sim, 4),
            "close":            close_val,
            "fwd_30":           journey["fwd_30"],
            "fwd_60":           journey["fwd_60"],
            "fwd_90":           journey["fwd_90"],
            "max_gain":         journey["max_gain"],
            "max_gain_day":     journey["max_gain_day"],
            "max_drawdown":     journey["max_drawdown"],
            "max_dd_day":       journey["max_dd_day"],
            "daily_volatility": journey["daily_volatility"],
            "conclusion":       journey["conclusion"],
            "outcome":          _classify_outcome(journey["fwd_30"]),
            "_meta": {
                "total_matches":      total_matches,
                "independent_n":      independent_n,
                "search_bars":        search_bars,
                "avg_similarity":     round(avg_sim, 4),
                "threshold_used":     threshold_used,
                "min_distance_used":  min_dist_used,
            },
        })

    return results if results else None




def _apply_min_distance_filter(
    sorted_dates: list[str],
    min_days:     int,
) -> list[str]:
    """
    Minimum Distance Sampling: chỉ giữ ngày cách ngày được chọn trước ít nhất min_days.
    Input đã sort theo thời gian (tăng dần).
    Chiến lược greedy: duyệt từ đầu, chọn ngày đầu tiên hợp lệ mỗi cụm.
    """
    if not sorted_dates:
        return []
    kept     = [sorted_dates[0]]
    last_str = sorted_dates[0]
    for d in sorted_dates[1:]:
        try:
            delta = (
                datetime.strptime(d, "%Y-%m-%d") -
                datetime.strptime(last_str, "%Y-%m-%d")
            ).days
            if delta >= min_days:
                kept.append(d)
                last_str = d
        except Exception:
            continue
    return kept


def _calc_price_journey(
    cache_df:   pd.DataFrame,
    from_date:  str,
    from_close: float,
    horizon:    int = 90,
) -> dict:
    """
    Phân tích hành trình giá trong `horizon` ngày tiếp theo.
    Returns: fwd_30/60/90, max_gain, max_drawdown, daily_volatility, conclusion.
    """
    result = {
        "fwd_30": None, "fwd_60": None, "fwd_90": None,
        "max_gain": None, "max_gain_day": None,
        "max_drawdown": None, "max_dd_day": None,
        "daily_volatility": None, "conclusion": "CHUA RO",
    }
    if from_close <= 0:
        return result
    future = cache_df[cache_df["date"] > from_date].reset_index(drop=True)
    if len(future) < 5:
        return result

    window = future.head(horizon)
    closes = window["close"].values.astype(float)

    for days, key in [(30, "fwd_30"), (60, "fwd_60"), (90, "fwd_90")]:
        if len(future) > days:
            fc = float(future.iloc[days]["close"])
            result[key] = round((fc - from_close) / from_close * 100, 2)

    pct    = (closes - from_close) / from_close * 100
    mg_idx = int(np.argmax(pct))
    result["max_gain"]     = round(float(pct[mg_idx]), 2)
    result["max_gain_day"] = int(mg_idx + 1)

    peak   = np.maximum.accumulate(pct)
    dd     = pct - peak
    md_idx = int(np.argmin(dd))
    result["max_drawdown"] = round(float(dd[md_idx]), 2)
    result["max_dd_day"]   = int(md_idx + 1)

    if len(closes) > 1:
        result["daily_volatility"] = round(
            float(np.mean(np.abs(np.diff(closes) / closes[:-1] * 100))), 2
        )

    f30 = result["fwd_30"]; mdd = result["max_drawdown"] or 0
    if f30 is None:           result["conclusion"] = "CHUA RO"
    elif f30 >= 8 and mdd > -8: result["conclusion"] = "TANG MANH, it rung lac"
    elif f30 >= 5:            result["conclusion"] = "TANG MANH"
    elif f30 >= 2:            result["conclusion"] = "TANG NHE"
    elif f30 <= -8:           result["conclusion"] = "GIAM MANH"
    elif f30 <= -3:           result["conclusion"] = "GIAM"
    else:                     result["conclusion"] = "DI NGANG"
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
        if len(future) > days:
            future_close = float(future.iloc[days]["close"])
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

def _best_holding_period(analogs: list) -> str:
    """Tìm thời gian tối ưu theo tỷ lệ thắng (không bị bias return TB)."""
    best_period, best_wr = 30, -1.0
    for days, key in [(30, "fwd_30"), (60, "fwd_60"), (90, "fwd_90")]:
        vals = [a[key] for a in analogs if a.get(key) is not None]
        if not vals: continue
        wr = sum(1 for v in vals if v > 0) / len(vals)
        avg = float(np.mean(vals))
        if wr > best_wr:
            best_wr, best_period = wr, days
    return f"{best_period}D (WR={best_wr:.0%})"


def format_analog_report(
    symbol:      str,
    analogs:     list,
    current_vec: dict,
    max_chars:   int = 4000,
) -> str:
    """
    Báo cáo 4 phần với Minimum Distance Sampling metadata.
    """
    if not analogs:
        return f"Khong tim thay ngay tuong dong cho {symbol}."

    meta          = analogs[0].get("_meta", {})
    total_matches = meta.get("total_matches",     len(analogs))
    independent_n = meta.get("independent_n",     len(analogs))
    search_bars   = meta.get("search_bars",        0)
    avg_sim       = meta.get("avg_similarity",     0.0)
    thresh        = meta.get("threshold_used",     0.80)
    min_dist      = meta.get("min_distance_used",  MIN_SAMPLE_DISTANCE_DAYS)

    # ── Phần 1: Tóm tắt ──────────────────────────────────────────────
    p1 = [
        f"PHAN TICH TUONG DONG: {symbol}",
        "═" * 38,
        f"Nguong      : {thresh:.0%}" + (" (ha tu 80%)" if thresh < 0.80 else ""),
        f"So mau doc lap: {independent_n} (tu {total_matches} ngay thô, loc {min_dist}D)",
        f"Tu          : {search_bars:,} ngay lich su",
        f"Do TD TB    : {avg_sim:.1%}",
    ]
    if min_dist < MIN_SAMPLE_DISTANCE_DAYS:
        p1.append(
            f"  * Giam khoang cach loc tu {MIN_SAMPLE_DISTANCE_DAYS}D "
            f"xuong {min_dist}D de du mau toi thieu."
        )
    if independent_n < MIN_SAMPLE_WARNING:
        p1.append(f"  ⚠️ Mau nho ({independent_n} mau doc lap), ket qua co do tin cay thap.")
    p1.append("")

    # ── Phần 2: Chi tiết ─────────────────────────────────────────────
    def _fmt_details(items):
        out = [f"CHI TIET {len(items)} NGAY TIEU BIEU:", "─" * 38]
        for i, a in enumerate(items, 1):
            mg=a.get("max_gain"); mgd=a.get("max_gain_day")
            mdd=a.get("max_drawdown"); mddd=a.get("max_dd_day")
            dv=a.get("daily_volatility"); conc=a.get("conclusion","?")
            em="🟢" if "TANG" in conc else "🔴" if "GIAM" in conc else "🟡"
            f30,f60,f90=a.get("fwd_30"),a.get("fwd_60"),a.get("fwd_90")
            out.append(f"#{i} — {a['date']} | Gia:{a['close']:,.0f} | TD:{a['similarity']:.1%}")
            pk=f"+{mg:.1f}%(N{mgd})" if mg is not None else "N/A"
            dd=f"{mdd:.1f}%(N{mddd})" if mdd is not None else "N/A"
            out.append(f"  ⏱️  Dinh:{pk} | Day:{dd}")
            fwd=" | ".join(f"{d}D:{v:+.1f}%" for d,v in [(30,f30),(60,f60),(90,f90)] if v is not None)
            if fwd: out.append(f"  📈 {fwd}")
            if dv is not None: out.append(f"  📊 Bien dong:{dv:.1f}%/ngay")
            out.append(f"  {em} {conc}")
            out.append("")
        return out

    # ── Phần 3: Thống kê (trên mẫu độc lập) ─────────────────────────
    def _fmt_stats(items):
        vf30=[a["fwd_30"] for a in items if a.get("fwd_30") is not None]
        vf60=[a["fwd_60"] for a in items if a.get("fwd_60") is not None]
        vf90=[a["fwd_90"] for a in items if a.get("fwd_90") is not None]
        vmdd=[a["max_drawdown"] for a in items if a.get("max_drawdown") is not None]
        vdv =[a["daily_volatility"] for a in items if a.get("daily_volatility") is not None]
        n=len(items)
        lines=[f"THONG KE ({n} MAU DOC LAP, loc {min_dist}D):", "─" * 38]
        if vf30:
            wins=[x for x in vf30 if x>0]; loss=[x for x in vf30 if x<=0]
            wr=len(wins)/len(vf30)
            exp=round(float(np.mean(vf30)),2)
            ps=round(sum(x for x in vf30 if x>0),2)
            ns=abs(round(sum(x for x in vf30 if x<0),2))
            pf=round(ps/ns,2) if ns>0 else 99.0
            pf_str="99.00" if pf>=99 else f"{pf:.2f}"
            lines+=[
                f"  Ty le tang 30D : {len(wins)}/{len(vf30)} ({wr:.0%})",
                f"  Expectancy     : {exp:+.2f}%",
                f"  Profit Factor  : {pf_str}",
                f"  LN TB sau 30D  : {float(np.mean(vf30)):+.2f}%",
            ]
        if vf60: lines.append(f"  LN TB sau 60D  : {float(np.mean(vf60)):+.2f}%")
        if vf90: lines.append(f"  LN TB sau 90D  : {float(np.mean(vf90)):+.2f}%")
        if vmdd: lines.append(f"  Max DD TB      : {float(np.mean(vmdd)):.2f}%")
        if vdv:  lines.append(f"  Bien dong TB   : {float(np.mean(vdv)):.1f}%/ngay")
        lines.append(f"  Thoi gian TU   : {_best_holding_period(items)}")
        return lines

    # ── Phần 4: Cảnh báo ─────────────────────────────────────────────
    def _fmt_warnings(items):
        vmdd=[a["max_drawdown"] for a in items if a.get("max_drawdown") is not None]
        warns=[]
        if vmdd and float(np.mean(vmdd)) < -5:
            warns.append(f"⚠️ MDD TB {float(np.mean(vmdd)):.1f}%: Rủi ro giảm trước khi tăng.")
        if avg_sim < 0.85:
            warns.append("⚠️ Độ TD TB < 85%, kết quả chỉ mang tính tham khảo.")
        if independent_n < 5:
            warns.append(f"⚠️ {independent_n} mẫu độc lập, chưa đủ ý nghĩa thống kê.")
        if thresh < 0.80:
            warns.append(f"⚠️ Đã hạ ngưỡng xuống {thresh:.0%} để đủ mẫu.")
        if min_dist < MIN_SAMPLE_DISTANCE_DAYS:
            warns.append(
                f"⚠️ Khoảng cách lọc giảm xuống {min_dist}D "
                f"(mặc định {MIN_SAMPLE_DISTANCE_DAYS}D). "
                f"Một số mẫu có thể không hoàn toàn độc lập."
            )
        return (["", "CANH BAO:"] + warns) if warns else []

    footer = ["", "Luu y: Phan tich chi mang tinh tham khao. QK khong dam bao TL."]
    p3 = _fmt_stats(analogs)
    p4 = _fmt_warnings(analogs)

    for n_show in [MAX_DISPLAY, 5]:
        p2   = _fmt_details(analogs[:n_show])
        full = "\n".join(p1 + p2 + p3 + p4 + footer)
        if len(full) <= max_chars:
            return full

    return ("\n".join(p1 + p3 + p4 + footer))[:max_chars]


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
