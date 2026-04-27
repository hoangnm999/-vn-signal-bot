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
FORWARD_DAYS             = [30, 60, 90]
SIMILARITY_THRESHOLDS    = [0.80, 0.75, 0.70]
MIN_RESULTS              = 3
MIN_SAMPLE_WARNING       = 5
MIN_SAMPLE_DISTANCE_DAYS = 30
MIN_SAMPLE_DISTANCE_FB   = 20


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
    top_n:         int  = 3,
    years:         int  = 5,
    exclude_days:  int  = 90,
    min_results:   int  = MIN_RESULTS,
) -> list | None:
    """
    Tìm mẫu tương đồng với:
    1. Bậc thang ngưỡng: 80% → 75% → 70% cho đến khi đủ min_results
    2. Minimum Distance Sampling 30D (fallback 20D) → mẫu độc lập
    3. _calc_price_journey đầy đủ cho mỗi mẫu
    _meta: total_matches, independent_n, search_bars,
           avg_similarity, threshold_used, min_distance_used
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

    cutoff_start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    cutoff_end   = (datetime.now() - timedelta(days=exclude_days)).strftime("%Y-%m-%d")
    search_df    = cache_df[
        (cache_df["date"] >= cutoff_start) &
        (cache_df["date"] <= cutoff_end)
    ].copy().reset_index(drop=True)
    if len(search_df) < 3:
        return None

    tarr  = np.array(vector_to_list(target_vector), dtype=float)
    tnorm = np.linalg.norm(tarr)
    if tnorm == 0:
        return None

    avail = [c for c in VECTOR_KEYS if c in search_df.columns]
    if len(avail) < VECTOR_DIM - 2:
        return None
    mat   = search_df[avail].fillna(0.0).values.astype(float)
    norms = np.linalg.norm(mat, axis=1)
    norms = np.where(norms == 0, 1e-9, norms)
    sims  = (mat @ tarr) / (norms * tnorm)

    # Bậc thang ngưỡng
    thresh_used = SIMILARITY_THRESHOLDS[0]
    raw_idx     = np.array([], dtype=int)
    for thresh in SIMILARITY_THRESHOLDS:
        cands = np.where(sims >= thresh)[0]
        if len(cands) >= min_results:
            raw_idx = cands; thresh_used = thresh; break
        raw_idx = cands; thresh_used = thresh
    if len(raw_idx) == 0:
        raw_idx     = np.argsort(-sims)[:max(top_n, 3)]
        thresh_used = float(sims[raw_idx[-1]]) if len(raw_idx) else 0.0

    total_matches = len(raw_idx)
    search_bars   = len(search_df)

    # Sort theo thời gian → Minimum Distance Sampling
    pairs_time = sorted(
        [(str(search_df.iloc[i]["date"]), i) for i in raw_idx],
        key=lambda x: x[0]
    )

    def _mds(pairs, md):
        kept = [pairs[0]]; last = pairs[0][0]
        for d, i in pairs[1:]:
            try:
                if (datetime.strptime(d, "%Y-%m-%d") -
                        datetime.strptime(last, "%Y-%m-%d")).days >= md:
                    kept.append((d, i)); last = d
            except Exception:
                continue
        return kept

    min_dist = MIN_SAMPLE_DISTANCE_DAYS
    kept     = _mds(pairs_time, min_dist)
    if len(kept) < min_results:
        kept_fb = _mds(pairs_time, MIN_SAMPLE_DISTANCE_FB)
        if len(kept_fb) >= min_results:
            kept = kept_fb; min_dist = MIN_SAMPLE_DISTANCE_FB
        else:
            kept = kept_fb; min_dist = MIN_SAMPLE_DISTANCE_FB

    kept_s  = sorted(kept, key=lambda x: -sims[x[1]])
    ind_n   = len(kept_s)
    avg_sim = float(np.mean([sims[i] for _, i in kept_s]))

    results = []
    for date_str, idx in kept_s:
        row       = search_df.iloc[idx]
        close_val = float(row.get("close", 0))
        j         = _calc_price_journey(cache_df, date_str, close_val)
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
            "daily_volatility": j["daily_volatility"],
            "conclusion":       j["conclusion"],
            "recovery_days":    j.get("recovery_days"),
            "outcome":          _classify_outcome(j["fwd_30"]),
            "_meta": {
                "total_matches":     total_matches,
                "independent_n":     ind_n,
                "search_bars":       search_bars,
                "avg_similarity":    round(avg_sim, 4),
                "threshold_used":    thresh_used,
                "min_distance_used": min_dist,
            },
        })
    return results if results else None


def _apply_min_distance_filter(sorted_dates: list, min_days: int) -> list:
    """Minimum Distance Sampling — chỉ giữ ngày cách nhau >= min_days."""
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
        "daily_volatility": None, "conclusion": "CHUA RO",
    }
    if from_close <= 0:
        return result
    future = cache_df[cache_df["date"] > from_date].reset_index(drop=True)
    if len(future) < 5:
        return result

    for days, key in [(30, "fwd_30"), (60, "fwd_60"), (90, "fwd_90")]:
        if len(future) > days:
            fc = float(future.iloc[days]["close"])
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

    if len(closes) > 1:
        result["daily_volatility"] = round(
            float(np.mean(np.abs(np.diff(closes) / closes[:-1] * 100))), 2)

    # CHANGE 4: tinh so ngay hoi phuc ve hoa von sau MAE
    result["recovery_days"] = None
    if result["max_drawdown"] is not None and result["max_drawdown"] < 0 and result["max_dd_day"] is not None:
        dd_idx = result["max_dd_day"]  # 1-indexed, so slice from dd_idx onward
        post_dd = closes[dd_idx:]
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


def format_analog_report(
    symbol:        str,
    analogs:       list,
    current_vec:   dict,
    max_chars:     int   = 4000,
    current_price: float = 0.0,   # gia hien tai thuc te — PHAI TRUYEN TU CALLER
) -> str:
    """
    Bao cao 4 phan — KHONG liet ke tung ngay.
    1. Tom tat nhanh (mau doc lap, MDS, nguong)
    2. Tom tat hanh trinh gia (MFE/MAE, capture rate, hold)
    3. Thong ke day du dong bo scan_watchlist
    4. Canh bao rui ro
    current_price: bat buoc truyen gia hien tai thuc te tu caller (vd: df["close"].iloc[-1])
                   de hien Ke hoach hanh dong dung. Neu = 0 -> section bi an.
    """
    if not analogs:
        return f"Khong tim thay ngay tuong dong cho {symbol}."

    meta      = analogs[0].get("_meta", {})
    total_m   = meta.get("total_matches",     len(analogs))
    ind_n     = meta.get("independent_n",     len(analogs))
    srch_bars = meta.get("search_bars",        0)
    avg_sim   = meta.get("avg_similarity",     0.0)
    thresh    = meta.get("threshold_used",     0.80)
    min_dist  = meta.get("min_distance_used",  MIN_SAMPLE_DISTANCE_DAYS)

    def _vals(key): return [a[key] for a in analogs if a.get(key) is not None]
    vf30 = _vals("fwd_30"); vf60 = _vals("fwd_60"); vf90 = _vals("fwd_90")
    vmg  = _vals("max_gain"); vmdd = _vals("max_drawdown")
    vmgd = _vals("max_gain_day"); vdv = _vals("daily_volatility")
    n    = len(vf30)

    # ── Phần 1: Tóm tắt ──────────────────────────────────────────────
    p1 = [
        f"PHAN TICH TUONG DONG: {symbol}",
        "═" * 38,
        f"Nguong     : {thresh:.0%}" + (" (da ha)" if thresh < 0.80 else ""),
        f"Mau doc lap: {ind_n}/{total_m} ngay (loc {min_dist}D)",
        f"Lich su    : {srch_bars:,} ngay | Do TD TB: {avg_sim:.1%}",
    ]
    if min_dist < MIN_SAMPLE_DISTANCE_DAYS:
        p1.append(f"  * Giam khoang cach loc: {MIN_SAMPLE_DISTANCE_DAYS}D->{min_dist}D")
    if ind_n < MIN_SAMPLE_WARNING:
        p1.append(f"  ⚠️ Mau nho ({ind_n} doc lap), do tin cay thap.")
    p1.append("")

    # ── Phần 2: Tóm tắt hành trình giá ──────────────────────────────
    p2 = ["HANH TRINH GIA (90D tiep theo):", "─" * 38]
    if vmg and vmdd:
        avg_mg = float(np.mean(vmg)); avg_mdd = float(np.mean(vmdd))
        p2.append(f"  MFE (dinh cao nhat): TB {avg_mg:+.1f}% | Max {max(vmg):+.1f}%")
        p2.append(f"  MAE (day sau nhat) : TB {avg_mdd:+.1f}% | Worst {min(vmdd):+.1f}%")
        if avg_mdd != 0:
            p2.append(f"  MFE/MAE ratio      : {abs(avg_mg/avg_mdd):.2f}x (ly tuong>2.0x)")
        caps = [a["fwd_30"] / a["max_gain"] * 100
                for a in analogs
                if a.get("fwd_30") is not None and a.get("max_gain") and a["max_gain"] > 0]
        if caps:
            cr   = float(np.mean(caps))
            if cr < 40:
                note = f"⚠️ exit som (Chi giu duoc {cr:.0f}% tiem nang – nen nam giu qua 30D)"
            elif cr > 80:
                note = f"✅ hieu qua (Giu duoc {cr:.0f}% muc loi nhuan tiem nang trong 30D)"
            else:
                note = f"(Giu duoc {cr:.0f}% muc loi nhuan tiem nang neu thoat sau 30D)"
            p2.append(f"  MFE thu duoc (30D) : {cr:.0f}%  {note}")
    if vmgd:
        p2.append(f"  Hold den dinh TB   : {float(np.mean(vmgd)):.0f}D "
                  f"(median {float(np.median(vmgd)):.0f}D, max {int(max(vmgd))}D)")
    if vf30:
        p2.append(f"  Ket qua 30D: tot {max(vf30):+.1f}% | xau {min(vf30):+.1f}%")
    p2.append("")

    # ── Phần 3: Thống kê đầy đủ ──────────────────────────────────────
    p3 = [f"THONG KE ({ind_n} MAU DOC LAP — loc {min_dist}D):", "─" * 38]
    if vf30:
        wins  = [x for x in vf30 if x > 0]; loss = [x for x in vf30 if x <= 0]
        wr    = len(wins) / len(vf30)
        med30 = float(np.median(vf30))
        exp   = round(float(np.mean(vf30)), 2)
        p25   = float(np.percentile(vf30, 25)) if n >= 4 else None
        p75   = float(np.percentile(vf30, 75)) if n >= 4 else None
        pos_s = sum(x for x in vf30 if x > 0)
        neg_s = abs(sum(x for x in vf30 if x < 0))
        pf    = round(pos_s / neg_s, 2) if neg_s > 0 else 99.0
        std30 = float(np.std(vf30, ddof=1)) if n > 1 else 0.0
        rvr   = round(med30 / std30, 2) if std30 > 0 else 0.0
        ci    = f" [P25:{p25:+.1f}% P75:{p75:+.1f}%]" if p25 is not None else ""
        # CHANGE 4: 3 chi so quan trong nhat — hien thi noi bat truoc
        _mae_med_highlight = float(np.median(vmdd)) if vmdd else 0.0
        _be_rr_highlight   = round((1 - wr) / wr, 1) if wr > 0 else 99.0
        p3.append(
            f"  >>> WR 30D: {wr:.0%}  |  MAE median: {_mae_med_highlight:+.1f}%"
            f"  |  Break-even R:R: 1:{_be_rr_highlight}"
        )
        p3.append("  " + "-" * 36)
        p3 += [
            f"  WR 30D        : {len(wins)}/{len(vf30)} ({wr:.0%}) | Thua: {1-wr:.0%}",
            f"  Median LN 30D : {med30:+.2f}%{ci}",
            f"  Expectancy    : {exp:+.2f}%",
            f"  Profit Factor : {'99.00' if pf >= 99 else f'{pf:.2f}'}",
            f"  Return/Vol 30D: {rvr:.2f}" + (
                " (Thap do bien dong manh dau ky – phu hop nguoi chiu rung lac)"
                if rvr < 0.5 else ""
            ),
        ]
    if vf60: p3.append(f"  Median LN 60D : {float(np.median(vf60)):+.2f}%")
    if vf90: p3.append(f"  Median LN 90D : {float(np.median(vf90)):+.2f}%")
    if vmdd:
        mae_med = float(np.median(vmdd))
        p3.append(f"  MAE TB (MDD)  : {float(np.mean(vmdd)):+.2f}% | Median: {mae_med:+.2f}%")
    if vmg:  p3.append(f"  MFE TB (Peak) : {float(np.mean(vmg)):+.2f}%")
    if vdv:  p3.append(f"  Bien dong TB  : {float(np.mean(vdv)):.1f}%/ngay")
    p3.append(f"  Thoi gian TU  : {_best_holding(analogs)}")
    p3.append("")

    # ── CHANGE 4: Phục hồi sau sụt giảm ─────────────────────────────
    vrec = [a.get("recovery_days") for a in analogs if a.get("recovery_days") is not None]
    vdd_any = [a for a in analogs if a.get("max_drawdown") is not None and a["max_drawdown"] < -1.0]
    not_healed = len(vdd_any) - len(vrec)
    p3b = ["PHUC HOI SAU SUT GIAM:", "─" * 38]
    if vrec:
        healed_n    = len(vrec)
        not_heal_n  = not_healed
        p3b.append(
            f"  Phuc hoi TB sau sut giam : {float(np.mean(vrec)):.0f} ngay "
            f"(tren {healed_n} mau co phuc hoi; {not_heal_n} mau khong phuc hoi trong 90D)"
        )
    else:
        p3b.append("  Khong du lieu phuc hoi (tat ca chua ve hoa von trong 90D)")
    p3b.append("")

    # ── CHANGE 5: MAE Survival Rate ──────────────────────────────────
    had_dd = [a for a in analogs if a.get("max_drawdown") is not None and a["max_drawdown"] < -1.0]
    survived = [a for a in had_dd if a.get("fwd_30") is not None and a["fwd_30"] > 0]
    p3c = ["MAE SURVIVAL RATE:", "─" * 38]
    if had_dd:
        sr = len(survived) / len(had_dd)
        p3c.append(f"  Ty le vuot qua sut giam : {len(survived)}/{len(had_dd)} ({sr:.0%})")
        if sr >= 0.70:
            p3c.append("  -> Phuc hoi TOT: phan lon truong hop sut giam roi phuc hoi")
        elif sr >= 0.50:
            p3c.append("  -> Phuc hoi TRUNG BINH: can quan sat them")
        else:
            p3c.append("  -> Phuc hoi THAP: rui ro khong phuc hoi cao")
    else:
        p3c.append("  Khong du lieu (khong co mau nao co drawdown >1%)")
    p3c.append("")

    # ── CHANGE 3: Ke hoach hanh dong — MAE median SL + R:R ──────────
    p3d = []
    if vf30 and vmdd and vmg:
        med30_ap  = float(np.median(vf30))
        p75_ap    = float(np.percentile(vf30, 75)) if n >= 4 else None
        mae_med_ap  = float(np.median(vmdd))           # MAE median dung cho SL
        mae_avg_ap  = float(np.mean(vmdd))
        # entry_price = gia hien tai THUC TE truyen vao tu caller
        # KHONG dung analog["close"] vi do la gia lich su (co the sai hoan toan)
        entry_price = float(current_price) if current_price and current_price > 0 else 0.0
        p3d = ["KE HOACH HANH DONG (tu dong):", "─" * 38]
        if entry_price <= 0:
            p3d.append("  (Can truyen current_price vao format_analog_report de hien ke hoach)")
            p3d.append("")
        if entry_price > 0:
            # SL = MAE median + buffer -2%
            sl_pct    = mae_med_ap - 2.0           # e.g. -14% - 2% = -16%
            sl_price  = entry_price * (1 + sl_pct / 100)
            tp1_price = entry_price * (1 + med30_ap / 100)

            # CHANGE 1: R:R dung gia tuyet doi
            # Risk  = |Entry - SL|  (so tien co the mat)
            # Reward = |TP - Entry| (so tien co the loi)
            # Format "1:X" voi X = Reward/Risk (X > 1 la tot)
            risk_abs  = abs(entry_price - sl_price)  # duong
            rw1_abs   = abs(tp1_price - entry_price)
            rr1 = round(rw1_abs / risk_abs, 1) if risk_abs > 0 else 0

            p3d.append(f"  Entry  : {entry_price:,.1f} (gia hien tai)")
            p3d.append(f"  SL     : {sl_price:,.1f} (MAE median {mae_med_ap:+.1f}% + buffer -2%) [{sl_pct:+.1f}%]")
            p3d.append(f"  TP1    : {tp1_price:,.1f} ({med30_ap:+.1f}% - P50) | Risk:Reward = 1:{rr1}")
            if p75_ap is not None:
                tp2_price = entry_price * (1 + p75_ap / 100)
                rw2_abs   = abs(tp2_price - entry_price)
                rr2 = round(rw2_abs / risk_abs, 1) if risk_abs > 0 else 0
                p3d.append(f"  TP2    : {tp2_price:,.1f} ({p75_ap:+.1f}% - P75) | Risk:Reward = 1:{rr2}")

            # CHANGE 3: R:R khuyen nghi toi thieu, co dieu chinh theo WR
            # WR cao -> nguong R:R co the thap hon (expectancy van duong)
            # Cong thuc: min_rr = (1 - WR) / WR  (break-even R:R)
            wr_ap = len([x for x in vf30 if x > 0]) / len(vf30) if vf30 else 0.5
            breakeven_rr = round((1 - wr_ap) / wr_ap, 1) if wr_ap > 0 else 2.0
            standard_rr  = 2.0   # rule of thumb pho bien
            rec_rr = max(breakeven_rr, 1.0)  # toi thieu bang break-even, va >= 1.0

            rr_best = rr2 if p75_ap is not None else rr1
            if rr_best >= standard_rr:
                rr_verdict = f"✅ DAT chuan 1:2.0"
            elif rr_best >= breakeven_rr:
                # >= break-even: expectancy duong, chap nhan duoc voi WR nay
                rr_verdict = f"⚠️ CHAP NHAN DUOC (break-even R:R={breakeven_rr} voi WR {wr_ap:.0%})"
            else:
                rr_verdict = f"❌ THAP (< break-even {breakeven_rr} voi WR {wr_ap:.0%}) — can xem lai SL/TP"

            p3d.append(f"  R:R khuyen nghi (1:2.0 chuan) | Thuc te tot nhat: 1:{rr_best} → {rr_verdict}")
        else:
            # Fallback: chi hien % neu khong co gia
            sl_pct = mae_med_ap - 2.0
            risk_pct = abs(sl_pct)
            rr1_pct = round(med30_ap / risk_pct, 1) if risk_pct > 0 else 0
            p3d.append(f"  SL     : {sl_pct:+.1f}% tu entry (MAE median + buffer -2%)")
            p3d.append(f"  TP1    : {med30_ap:+.1f}% tu entry (P50) | R:R = 1:{rr1_pct}")
            if p75_ap is not None:
                rr2_pct = round(p75_ap / risk_pct, 1) if risk_pct > 0 else 0
                p3d.append(f"  TP2    : {p75_ap:+.1f}% tu entry (P75) | R:R = 1:{rr2_pct}")
        if mae_avg_ap < -10:
            p3d.append(f"  Rui ro: CAO (xem Canh bao ben duoi)")
        p3d.append("")

    # ── CHANGE 2: Kết luận hành động — tổng kết 3 yếu tố ───────────
    p3e = []
    if vf30 and vmdd:
        _wr_kl    = len([x for x in vf30 if x > 0]) / len(vf30)
        _mae_kl   = float(np.mean(vmdd))
        _be_kl    = round((1 - _wr_kl) / _wr_kl, 1) if _wr_kl > 0 else 99.0

        # Xac suat thang
        if _wr_kl >= 0.65:
            _wr_txt = f"xac suat thang cao ({_wr_kl:.0%})"
        elif _wr_kl >= 0.50:
            _wr_txt = f"xac suat thang trung binh ({_wr_kl:.0%})"
        else:
            _wr_txt = f"xac suat thang thap ({_wr_kl:.0%})"

        # R:R (dung gia tot nhat = TP2 neu co)
        _rr_best = 0.0
        if vf30 and vmdd and current_price > 0:
            _sl_pct_kl = float(np.median(vmdd)) - 2.0
            _sl_abs_kl = abs(current_price * _sl_pct_kl / 100)
            _p75_kl    = float(np.percentile(vf30, 75)) if n >= 4 else float(np.median(vf30))
            _tp_abs_kl = abs(current_price * _p75_kl / 100)
            _rr_best   = round(_tp_abs_kl / _sl_abs_kl, 1) if _sl_abs_kl > 0 else 0.0

        if _rr_best > 0:
            if _rr_best >= 2.0:
                _rr_txt = f"R:R tot (1:{_rr_best})"
            elif _rr_best >= _be_kl:
                _rr_txt = f"R:R chap nhan duoc (1:{_rr_best}, break-even 1:{_be_kl})"
            else:
                _rr_txt = f"R:R thap (1:{_rr_best} < break-even 1:{_be_kl})"
        else:
            _rr_txt = f"break-even R:R = 1:{_be_kl}"

        # Rui ro sut giam
        if _mae_kl < -20:
            _mae_txt = f"rui ro sut giam rat manh ({_mae_kl:.1f}%)"
        elif _mae_kl < -10:
            _mae_txt = f"rui ro sut giam manh ({_mae_kl:.1f}%)"
        else:
            _mae_txt = f"rui ro sut giam vua ({_mae_kl:.1f}%)"

        # Verdict tong hop
        if _wr_kl >= 0.65 and _rr_best >= _be_kl:
            _overall = "Co the xem xet vao lenh neu chap nhan duoc rui ro."
        elif _wr_kl < 0.50 or (_rr_best > 0 and _rr_best < _be_kl):
            _overall = "Can than — xac suat hoac R:R chua du dieu kien."
        else:
            _overall = "Theo doi them truoc khi quyet dinh."

        p3e = [
            "KET LUAN HANH DONG:",
            "─" * 38,
            f"  Co hoi {_wr_txt}, {_rr_txt}, {_mae_txt}.",
            f"  => {_overall}",
            "",
        ]

    # ── Phần 4: Cảnh báo ─────────────────────────────────────────────
    warns = []
    if vmdd and float(np.mean(vmdd)) < -5:
        warns.append(f"⚠️ MAE TB {float(np.mean(vmdd)):.1f}%: nhip giam manh truoc khi tang.")
    if avg_sim < 0.85:
        warns.append("⚠️ Do TD TB < 85%, ket qua chi mang tinh tham khao.")
    if ind_n < 5:
        warns.append(f"⚠️ {ind_n} mau doc lap, chua du y nghia thong ke.")
    if thresh < 0.80:
        warns.append(f"⚠️ Da ha nguong xuong {thresh:.0%} de du mau.")
    if min_dist < MIN_SAMPLE_DISTANCE_DAYS:
        warns.append(f"⚠️ Khoang cach loc giam {min_dist}D.")
    p4 = (["CANH BAO:", "─" * 38] + warns + [""]) if warns else []

    footer = ["Luu y: Phan tich chi mang tinh tham khao. QK khong dam bao TL."]
    return ("\n".join(p1 + p2 + p3 + p3b + p3c + p3d + p3e + p4 + footer))[:max_chars]


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
