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
FORWARD_DAYS       = [30, 60, 90]   # ngày kiểm tra forward performance
SIMILARITY_THRESHOLDS = [0.80, 0.75, 0.70]  # thử lần lượt nếu < min_results
MIN_RESULTS        = 3               # tối thiểu kết quả trước khi hạ ngưỡng
MAX_DISPLAY        = 10              # tối đa hiển thị chi tiết
MIN_SAMPLE_WARNING = 3               # dưới này cảnh báo mẫu nhỏ


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
    top_n:         int = 3,
    years:         int = 5,
    exclude_days:  int = 90,
    min_results:   int = MIN_RESULTS,
) -> Optional[list]:
    """
    Tìm TẤT CẢ ngày lịch sử có similarity >= ngưỡng (mặc định 80%).

    Điểm 1 — Bậc thang ngưỡng:
      Thử 80% → 75% → 70% cho đến khi có >= min_results kết quả.
      Output luôn ghi rõ ngưỡng thực sự đã dùng.

    Trả về list sắp xếp theo similarity giảm dần, mỗi phần tử có:
      date, similarity, close, fwd_30/60/90,
      max_gain, max_gain_day, max_drawdown, max_dd_day,
      daily_volatility, conclusion, outcome,
      _meta: {total_matches, search_bars, avg_similarity, threshold_used}
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
        logger.warning(f"find_similar({symbol}): cache quá ngắn ({len(cache_df)} rows)")
        return None

    cutoff_start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    cutoff_end   = (datetime.now() - timedelta(days=exclude_days)).strftime("%Y-%m-%d")
    search_df    = cache_df[
        (cache_df["date"] >= cutoff_start) &
        (cache_df["date"] <= cutoff_end)
    ].copy().reset_index(drop=True)

    if len(search_df) < 3:
        return None

    target_list = vector_to_list(target_vector)
    target_arr  = np.array(target_list, dtype=float)
    t_norm      = np.linalg.norm(target_arr)
    if t_norm == 0:
        return None

    available_cols = [c for c in VECTOR_KEYS if c in search_df.columns]
    if len(available_cols) < VECTOR_DIM - 2:
        logger.warning(f"find_similar: cache thiếu columns")
        return None

    mat          = search_df[available_cols].fillna(0.0).values.astype(float)
    dot_prods    = mat @ target_arr
    mat_norms    = np.linalg.norm(mat, axis=1)
    mat_norms    = np.where(mat_norms == 0, 1e-9, mat_norms)
    similarities = dot_prods / (mat_norms * t_norm)

    # ── Điểm 1: Bậc thang ngưỡng 80% → 75% → 70% ──────────────────────
    threshold_used = SIMILARITY_THRESHOLDS[0]
    matched_idx    = np.array([], dtype=int)
    for thresh in SIMILARITY_THRESHOLDS:
        candidates = np.where(similarities >= thresh)[0]
        if len(candidates) >= min_results:
            matched_idx    = candidates
            threshold_used = thresh
            break
        logger.info(
            f"find_similar({symbol}): {len(candidates)} kq tai {thresh:.0%}, "
            f"ha nguong xuong {SIMILARITY_THRESHOLDS[SIMILARITY_THRESHOLDS.index(thresh)+1]:.0%}"
            if thresh != SIMILARITY_THRESHOLDS[-1] else
            f"find_similar({symbol}): {len(candidates)} kq tai {thresh:.0%} (nguong thap nhat)"
        )
        matched_idx    = candidates
        threshold_used = thresh

    if len(matched_idx) == 0:
        # Fallback tuyệt đối: top_n cao nhất
        matched_idx    = np.argsort(-similarities)[:max(top_n, 3)]
        threshold_used = float(similarities[matched_idx[-1]]) if len(matched_idx) > 0 else 0.0
        logger.info(f"find_similar({symbol}): fallback top {len(matched_idx)}, sim={threshold_used:.2%}")

    sorted_idx   = matched_idx[np.argsort(-similarities[matched_idx])]
    total_match  = len(sorted_idx)
    avg_sim      = float(np.mean(similarities[sorted_idx]))
    search_bars  = len(search_df)

    results = []
    for idx in sorted_idx:
        sim       = float(similarities[idx])
        row       = search_df.iloc[idx]
        date_str  = str(row["date"])
        close_val = float(row.get("close", 0))

        journey = _calc_price_journey(cache_df, date_str, close_val, horizon=90)

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
                "total_matches":   total_match,
                "search_bars":     search_bars,
                "avg_similarity":  round(avg_sim, 4),
                "threshold_used":  threshold_used,
            },
        })

    return results if results else None


def _calc_price_journey(
    cache_df:   pd.DataFrame,
    from_date:  str,
    from_close: float,
    horizon:    int = 90,
) -> dict:
    """
    Phân tích hành trình giá trong `horizon` ngày tiếp theo.
    Trả về: fwd_30/60/90, max_gain, max_drawdown, daily_volatility, conclusion.
    """
    result = {
        "fwd_30": None, "fwd_60": None, "fwd_90": None,
        "max_gain": None, "max_gain_day": None,
        "max_drawdown": None, "max_dd_day": None,
        "daily_volatility": None,
        "conclusion": "CHUA RO",
    }
    if from_close <= 0:
        return result

    future = cache_df[cache_df["date"] > from_date].reset_index(drop=True)
    if len(future) < 5:
        return result

    window = future.head(horizon)
    closes = window["close"].values.astype(float)

    # Forward returns tại mốc cố định
    for days, key in [(30, "fwd_30"), (60, "fwd_60"), (90, "fwd_90")]:
        if len(future) > days:
            fc = float(future.iloc[days]["close"])
            result[key] = round((fc - from_close) / from_close * 100, 2)

    # Max gain
    pct = (closes - from_close) / from_close * 100
    mg_idx = int(np.argmax(pct))
    result["max_gain"]     = round(float(pct[mg_idx]), 2)
    result["max_gain_day"] = int(mg_idx + 1)

    # Max drawdown từ đỉnh (running peak)
    peak = np.maximum.accumulate(pct)
    dd   = pct - peak
    md_idx = int(np.argmin(dd))
    result["max_drawdown"] = round(float(dd[md_idx]), 2)
    result["max_dd_day"]   = int(md_idx + 1)

    # Daily volatility
    if len(closes) > 1:
        result["daily_volatility"] = round(
            float(np.mean(np.abs(np.diff(closes) / closes[:-1] * 100))), 2
        )

    # Conclusion
    f30 = result["fwd_30"]
    mdd = result["max_drawdown"] or 0
    if f30 is None:
        result["conclusion"] = "CHUA RO"
    elif f30 >= 8 and mdd > -8:
        result["conclusion"] = "TANG MANH, it rung lac"
    elif f30 >= 5:
        result["conclusion"] = "TANG MANH, rung lac dau ky" if mdd < -5 else "TANG MANH"
    elif f30 >= 2:
        result["conclusion"] = "TANG NHE"
    elif f30 <= -8:
        result["conclusion"] = "GIAM MANH"
    elif f30 <= -3:
        result["conclusion"] = "GIAM"
    else:
        result["conclusion"] = "DI NGANG"

    return result


def _calc_forward_returns(
    cache_df:  pd.DataFrame,
    from_date: str,
    from_close: float,
) -> dict:
    """Wrapper backward-compat → gọi _calc_price_journey."""
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

def _best_holding_period(analogs: list) -> tuple[int, str]:
    """
    Điểm 2: Tìm thời gian nắm giữ tối ưu dựa trên tỷ lệ thắng cao nhất
    (không dùng return TB vì bị bias dài hạn tăng của TTCK VN).
    Tie-break: nếu tỷ lệ thắng bằng nhau → chọn mốc ngắn hơn (exit sớm).
    """
    best_period = 30
    best_wr     = -1.0
    best_label  = ""
    for days, key in [(30, "fwd_30"), (60, "fwd_60"), (90, "fwd_90")]:
        vals = [a[key] for a in analogs if a.get(key) is not None]
        if not vals:
            continue
        wr  = sum(1 for v in vals if v > 0) / len(vals)
        avg = float(np.mean(vals))
        if wr > best_wr:               # tỷ lệ thắng cao hơn → chọn
            best_wr     = wr
            best_period = days
            best_label  = f"{days}D (WR={wr:.0%}, TB {avg:+.1f}%)"
        # Tie-break: tỷ lệ thắng bằng → mốc ngắn hơn đã được chọn sẵn
    return best_period, best_label


def format_analog_report(
    symbol:      str,
    analogs:     list,
    current_vec: dict,
    max_chars:   int = 4000,
) -> str:
    """
    Báo cáo 4 phần:
      1. Tóm tắt nhanh (ngưỡng thực dùng, tổng khớp)
      2. Chi tiết tối đa 10 ngày tiêu biểu (tự giảm xuống 5 nếu quá dài)
      3. Thống kê tổng hợp toàn bộ mẫu
      4. Cảnh báo rủi ro tự động
    """
    if not analogs:
        return f"Khong tim thay ngay tuong dong cho {symbol}."

    meta         = analogs[0].get("_meta", {})
    total_match  = meta.get("total_matches",  len(analogs))
    search_bars  = meta.get("search_bars",    0)
    avg_sim_all  = meta.get("avg_similarity", 0.0)
    thresh_used  = meta.get("threshold_used", 0.80)

    # ── Phần 1: Tóm tắt ──────────────────────────────────────────────
    p1 = [
        f"PHAN TICH TUONG DONG: {symbol}",
        "═" * 38,
        f"Nguong dung : {thresh_used:.0%}" +
            (" (ha tu 80%)" if thresh_used < 0.80 else ""),
        f"So khop     : {total_match} ngay",
        f"Tu          : {search_bars:,} ngay lich su",
        f"Do TD TB    : {avg_sim_all:.1%} (toan bo mau)",
    ]
    if total_match < MIN_SAMPLE_WARNING:
        p1.append("⚠️ Mau nho, ket qua chi mang tinh tham khao.")
    p1.append("")

    # ── Phần 2: Chi tiết ngày tiêu biểu ──────────────────────────────
    def _fmt_details(items: list) -> list:
        out = [f"CHI TIET {len(items)} NGAY TIEU BIEU:", "─" * 38]
        for i, a in enumerate(items, 1):
            mg   = a.get("max_gain");       mgd  = a.get("max_gain_day")
            mdd  = a.get("max_drawdown");   mddd = a.get("max_dd_day")
            dv   = a.get("daily_volatility")
            conc = a.get("conclusion", "?")
            em   = "🟢" if "TANG" in conc else "🔴" if "GIAM" in conc else "🟡"
            f30, f60, f90 = a.get("fwd_30"), a.get("fwd_60"), a.get("fwd_90")

            out.append(
                f"#{i} — {a['date']} | Gia: {a['close']:,.0f} | TD: {a['similarity']:.1%}"
            )
            peak_s = f"+{mg:.1f}% (N{mgd})" if mg is not None else "N/A"
            dd_s   = f"{mdd:.1f}% (N{mddd})" if mdd is not None else "N/A"
            out.append(f"  ⏱️ 90D: Dinh: {peak_s} | Day: {dd_s}")
            fwd = " | ".join(f"{d}D:{v:+.1f}%" for d, v in
                             [(30,f30),(60,f60),(90,f90)] if v is not None)
            if fwd: out.append(f"  📈 Fwd: {fwd}")
            if dv is not None: out.append(f"  📊 Bien dong: {dv:.1f}%/ngay")
            out.append(f"  {em} {conc}")
            out.append("")
        return out

    # ── Phần 3: Thống kê tổng hợp ────────────────────────────────────
    def _fmt_stats(items: list) -> list:
        vf = {d: [a[k] for a in items if a.get(k) is not None]
              for d, k in [(30,"fwd_30"),(60,"fwd_60"),(90,"fwd_90")]}
        vmdd = [a["max_drawdown"]     for a in items if a.get("max_drawdown") is not None]
        vdv  = [a["daily_volatility"] for a in items if a.get("daily_volatility") is not None]
        n    = len(items)

        lines = [f"THONG KE TONG HOP ({n} ngay, nguong {thresh_used:.0%}):", "─" * 38]

        if vf[30]:
            wins = [x for x in vf[30] if x > 0]
            loss = [x for x in vf[30] if x <= 0]
            wr   = len(wins) / len(vf[30])
            lines += [
                f"  Ty le tang 30D : {len(wins)}/{len(vf[30])} ({wr:.0%})",
                f"  LN TB khi tang : {float(np.mean(wins)):+.2f}%"   if wins else "  LN TB khi tang : N/A",
                f"  TL TB khi giam : {float(np.mean(loss)):+.2f}%"   if loss else "  TL TB khi giam : N/A",
                f"  LN TB sau 30D  : {float(np.mean(vf[30])):+.2f}%",
            ]
        if vf[60]:
            lines.append(f"  LN TB sau 60D  : {float(np.mean(vf[60])):+.2f}%")
        if vf[90]:
            lines.append(f"  LN TB sau 90D  : {float(np.mean(vf[90])):+.2f}%")
        if vmdd:
            lines.append(f"  Max DD TB      : {float(np.mean(vmdd)):.2f}%")
        if vdv:
            lines.append(f"  Bien dong TB   : {float(np.mean(vdv)):.1f}%/ngay")

        # Điểm 2: tối ưu theo tỷ lệ thắng, không phải return TB
        _, best_lbl = _best_holding_period(items)
        if best_lbl:
            lines.append(f"  Thoi gian TU   : {best_lbl}")

        return lines

    # ── Phần 4: Cảnh báo ─────────────────────────────────────────────
    def _fmt_warnings(items: list) -> list:
        wmdd = [a["max_drawdown"] for a in items if a.get("max_drawdown") is not None]
        warns = []
        if wmdd and float(np.mean(wmdd)) < -5:
            warns.append(
                f"⚠️ MDD TB {float(np.mean(wmdd)):.1f}%: Kich ban tuong tu thuong "
                f"co nhip giam manh. Can chuan bi tam ly va von."
            )
        if avg_sim_all < 0.85:
            warns.append("⚠️ Do TD TB < 85%, ket qua chi mang tinh tham khao.")
        if total_match < 5:
            warns.append(f"⚠️ {total_match} mau, chua du y nghia thong ke.")
        if thresh_used < 0.80:
            warns.append(
                f"⚠️ Da ha nguong xuong {thresh_used:.0%} de du mau. "
                f"Ket qua it chinh xac hon."
            )
        return (["", "CANH BAO RUI RO:"] + warns) if warns else []

    footer = ["", "Luu y: Phan tich chi mang tinh tham khao. QK khong dam bao TL."]

    p3 = _fmt_stats(analogs)
    p4 = _fmt_warnings(analogs)

    # Thử full 10 ngày → 5 ngày → chỉ stats
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
