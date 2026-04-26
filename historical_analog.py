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
FORWARD_DAYS     = [30, 60, 90]   # ngày kiểm tra forward performance


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

# Ngưỡng tương đồng tối thiểu để lọc kết quả
SIMILARITY_THRESHOLD = 0.80   # 80%
MAX_DISPLAY          = 10     # tối đa hiển thị chi tiết
MIN_DISPLAY_WARNING  = 3      # dưới ngưỡng này thêm cảnh báo mẫu nhỏ


def find_similar(
    symbol:        str,
    target_vector: dict,
    top_n:         int = 3,       # giữ param để backward-compat
    years:         int = 5,
    exclude_days:  int = 90,
    sim_threshold: float = SIMILARITY_THRESHOLD,
) -> Optional[list]:
    """
    Tìm TẤT CẢ ngày lịch sử có similarity >= sim_threshold (mặc định 80%).

    Trả về list đã sắp xếp theo similarity giảm dần, mỗi phần tử:
    {
        "date", "similarity", "close",
        "fwd_30", "fwd_60", "fwd_90",   # % return tại mốc cố định
        "max_gain", "max_gain_day",      # đỉnh tăng cao nhất (%)
        "max_drawdown", "max_dd_day",    # sụt giảm lớn nhất từ đỉnh (%)
        "daily_volatility",              # biến động TB ngày (%)
        "conclusion",                    # nhận định nhanh
        "outcome",                       # TANG/GIAM/DI NGANG (backward-compat)
    }

    Field bổ sung trong list metadata (truy cập qua results[0]["_meta"]):
        "total_matches"   : tổng ngày >= threshold
        "search_bars"     : số ngày tìm kiếm
        "avg_similarity"  : similarity trung bình của tất cả match
    """
    symbol = symbol.upper()

    if not cache_exists(symbol):
        return None

    try:
        t0       = time.time()
        cache_df = pd.read_csv(_cache_path(symbol))
        logger.debug(f"find_similar({symbol}): loaded {len(cache_df)} rows in {time.time()-t0:.2f}s")
    except Exception as e:
        logger.warning(f"find_similar({symbol}): load cache fail: {e}")
        return None

    if len(cache_df) < exclude_days + 90 + 10:
        logger.warning(f"find_similar({symbol}): cache quá ngắn ({len(cache_df)} rows)")
        return None

    # Giới hạn theo years và bỏ exclude_days cuối
    cutoff_start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    cutoff_end   = (datetime.now() - timedelta(days=exclude_days)).strftime("%Y-%m-%d")
    search_df    = cache_df[
        (cache_df["date"] >= cutoff_start) &
        (cache_df["date"] <= cutoff_end)
    ].copy().reset_index(drop=True)

    if len(search_df) < 5:
        return None

    # Vectorized cosine similarity
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

    # Lọc TẤT CẢ ngày >= threshold, sắp xếp giảm dần
    above_thresh = np.where(similarities >= sim_threshold)[0]
    if len(above_thresh) == 0:
        # Fallback: lấy top_n nếu không có ngày nào >= threshold
        above_thresh = np.argsort(-similarities)[:max(top_n, 3)]
        logger.info(f"find_similar({symbol}): không có ngày >= {sim_threshold:.0%}, fallback top {top_n}")

    sorted_idx  = above_thresh[np.argsort(-similarities[above_thresh])]
    total_match = len(sorted_idx)
    avg_sim     = float(np.mean(similarities[sorted_idx]))
    search_bars = len(search_df)

    # Tính đầy đủ price journey cho từng ngày match
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
                "total_matches": total_match,
                "search_bars":   search_bars,
                "avg_similarity": round(avg_sim, 4),
            },
        })

    return results if results else None


def _calc_forward_returns(
    cache_df:  pd.DataFrame,
    from_date: str,
    from_close: float,
) -> dict:
    """Wrapper backward-compat → gọi _calc_price_journey."""
    j = _calc_price_journey(cache_df, from_date, from_close)
    return {30: j["fwd_30"], 60: j["fwd_60"], 90: j["fwd_90"]}


def _calc_price_journey(
    cache_df:   pd.DataFrame,
    from_date:  str,
    from_close: float,
    horizon:    int = 90,
) -> dict:
    """
    Phân tích toàn bộ hành trình giá trong `horizon` ngày tiếp theo.

    Returns dict:
        fwd_30, fwd_60, fwd_90      : % return tại mốc cố định
        max_gain, max_gain_day      : đỉnh tăng cao nhất (%), ngày đạt đỉnh
        max_drawdown, max_dd_day    : sụt giảm lớn nhất từ đỉnh (%), ngày chạm đáy
        daily_volatility            : biến động TB hàng ngày (%)
        conclusion                  : str nhận định nhanh
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

    # Hành trình giá — pct change so với from_close
    pct_changes = (closes - from_close) / from_close * 100

    # Max gain
    max_gain_idx = int(np.argmax(pct_changes))
    result["max_gain"]     = round(float(pct_changes[max_gain_idx]), 2)
    result["max_gain_day"] = int(max_gain_idx + 1)

    # Max drawdown từ đỉnh (running peak)
    running_peak = np.maximum.accumulate(pct_changes)
    drawdowns    = pct_changes - running_peak   # luôn <= 0
    max_dd_idx   = int(np.argmin(drawdowns))
    result["max_drawdown"] = round(float(drawdowns[max_dd_idx]), 2)
    result["max_dd_day"]   = int(max_dd_idx + 1)

    # Daily volatility (mean |daily pct change|)
    if len(closes) > 1:
        daily_ret = np.abs(np.diff(closes) / closes[:-1] * 100)
        result["daily_volatility"] = round(float(np.mean(daily_ret)), 2)

    # Conclusion
    f30 = result["fwd_30"]
    mdd = result["max_drawdown"] or 0
    mg  = result["max_gain"] or 0
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

def format_analog_report(
    symbol:      str,
    analogs:     list,
    current_vec: dict,
    max_chars:   int = 4000,
) -> str:
    """
    Tạo báo cáo phân tích tương đồng 4 phần:
      1. Tóm tắt nhanh
      2. Chi tiết ngày tiêu biểu (tối đa MAX_DISPLAY=10, tự giảm nếu quá dài)
      3. Thống kê tổng hợp (tính trên TOÀN BỘ ngày >= 80%)
      4. Cảnh báo rủi ro tự động
    """
    if not analogs:
        return f"Khong tim thay ngay tuong dong cho {symbol}."

    # ── Meta từ kết quả find_similar ──────────────────────────────────
    meta         = analogs[0].get("_meta", {})
    total_match  = meta.get("total_matches", len(analogs))
    search_bars  = meta.get("search_bars",  0)
    avg_sim_all  = meta.get("avg_similarity", 0)

    # ── PHẦN 1: Tóm tắt nhanh ─────────────────────────────────────────
    p1 = [
        f"PHAN TICH TUONG DONG: {symbol}",
        "═" * 38,
        f"So khop : {total_match} ngay tuong dong >= 80%",
        f"Tu      : {search_bars:,} ngay lich su",
        f"Do TD TB: {avg_sim_all:.1%} (toan bo mau khop)",
    ]
    if total_match < MIN_DISPLAY_WARNING:
        p1.append("⚠️ Mau du lieu nho, ket qua chi mang tinh tham khao.")
    p1.append("")

    # ── PHẦN 2: Chi tiết ngày tiêu biểu ──────────────────────────────
    display_list = analogs[:MAX_DISPLAY]   # top similarity đã sort sẵn

    def _fmt_details(items: list) -> list:
        lines = [f"CHI TIET {len(items)} NGAY TIEU BIEU:", "─" * 38]
        for i, a in enumerate(items, 1):
            sim  = a["similarity"]
            date = a["date"]
            cl   = a["close"]
            f30  = a["fwd_30"]
            f60  = a["fwd_60"]
            f90  = a["fwd_90"]
            mg   = a.get("max_gain")
            mgd  = a.get("max_gain_day")
            mdd  = a.get("max_drawdown")
            mddd = a.get("max_dd_day")
            dv   = a.get("daily_volatility")
            conc = a.get("conclusion", "?")
            em   = "🟢" if "TANG" in conc else "🔴" if "GIAM" in conc else "🟡"

            lines.append(f"#{i} — {date} | Gia: {cl:,.0f} | TD: {sim:.1%}")
            # Hành trình 90 ngày
            peak_str = f"+{mg:.1f}% (N{mgd})" if mg is not None else "N/A"
            dd_str   = f"{mdd:.1f}% (N{mddd})" if mdd is not None else "N/A"
            lines.append(f"  ⏱️ 90D: Dinh: {peak_str} | Day: {dd_str}")
            # Forward returns
            fwd_parts = []
            if f30 is not None: fwd_parts.append(f"30D:{f30:+.1f}%")
            if f60 is not None: fwd_parts.append(f"60D:{f60:+.1f}%")
            if f90 is not None: fwd_parts.append(f"90D:{f90:+.1f}%")
            if fwd_parts:
                lines.append(f"  📈 Fwd: {' | '.join(fwd_parts)}")
            if dv is not None:
                lines.append(f"  📊 Bien dong TB: {dv:.1f}%/ngay")
            lines.append(f"  {em} {conc}")
            lines.append("")
        return lines

    p2 = _fmt_details(display_list)

    # ── PHẦN 3: Thống kê tổng hợp (toàn bộ analogs, không chỉ display) ─
    def _build_stats(items: list) -> list:
        vf30 = [a["fwd_30"] for a in items if a.get("fwd_30") is not None]
        vf60 = [a["fwd_60"] for a in items if a.get("fwd_60") is not None]
        vf90 = [a["fwd_90"] for a in items if a.get("fwd_90") is not None]
        vmdd = [a["max_drawdown"] for a in items if a.get("max_drawdown") is not None]
        vdv  = [a["daily_volatility"] for a in items if a.get("daily_volatility") is not None]
        n    = len(items)

        lines = [
            f"THONG KE TONG HOP ({n} ngay, do TD >= 80%):",
            "─" * 38,
        ]
        if vf30:
            wins   = [x for x in vf30 if x > 0]
            losses = [x for x in vf30 if x <= 0]
            wr     = len(wins) / len(vf30)
            avg30  = float(np.mean(vf30))
            aw30   = float(np.mean(wins))   if wins   else 0
            al30   = float(np.mean(losses)) if losses else 0
            lines += [
                f"  Ty le tang sau 30D : {len(wins)}/{len(vf30)} ({wr:.0%})",
                f"  LN TB khi tang     : {aw30:+.2f}%",
                f"  TL TB khi giam     : {al30:+.2f}%",
                f"  LN TB sau 30D      : {avg30:+.2f}%",
            ]
        if vf60:
            lines.append(f"  LN TB sau 60D      : {float(np.mean(vf60)):+.2f}%")
        if vf90:
            lines.append(f"  LN TB sau 90D      : {float(np.mean(vf90)):+.2f}%")
        if vmdd:
            avg_mdd = float(np.mean(vmdd))
            lines.append(f"  Max DD TB          : {avg_mdd:.2f}%")
        if vdv:
            lines.append(f"  Bien dong TB ngay  : {float(np.mean(vdv)):.1f}%")

        # Thời gian nắm giữ tối ưu
        avgs = {}
        for k, v in [(30, vf30), (60, vf60), (90, vf90)]:
            if v: avgs[k] = float(np.mean(v))
        if avgs:
            best_k = max(avgs, key=avgs.get)
            lines.append(f"  Thoi gian toi uu   : {best_k} ngay (TB {avgs[best_k]:+.1f}%)")

        return lines

    p3 = _build_stats(analogs)

    # ── PHẦN 4: Cảnh báo rủi ro ─────────────────────────────────────
    warnings = []
    vmdd_all  = [a["max_drawdown"] for a in analogs if a.get("max_drawdown") is not None]
    vf30_all  = [a["fwd_30"]       for a in analogs if a.get("fwd_30") is not None]

    avg_mdd_val = float(np.mean(vmdd_all)) if vmdd_all else 0
    if avg_mdd_val < -5:
        warnings.append(
            f"⚠️ MDD TB = {avg_mdd_val:.1f}%: Cac kich ban qua khu "
            f"thuong co nhip giam manh truoc khi tang. "
            f"Can chuan bi tam ly va von."
        )
    if avg_sim_all < 0.85 and total_match > 0:
        warnings.append(
            "⚠️ Do tuong dong TB < 85%, ket qua chi mang tinh tham khao."
        )
    if total_match < 5:
        warnings.append(
            f"⚠️ Chi co {total_match} mau, chua du y nghia thong ke."
        )

    p4 = []
    if warnings:
        p4 = ["", "CANH BAO RUI RO:"] + warnings

    # ── Assemble với giới hạn ký tự ─────────────────────────────────
    footer = ["", "Luu y: Phan tich chi mang tinh tham khao. QK khong dam bao TL."]
    base   = p1 + p3 + p4 + footer
    base_txt = "\n".join(base)

    # Thử full (10 ngày chi tiết)
    full = "\n".join(p1 + p2 + p3 + p4 + footer)
    if len(full) <= max_chars:
        return full

    # Giảm xuống 5 ngày
    p2_short = _fmt_details(display_list[:5])
    short    = "\n".join(p1 + p2_short + p3 + p4 + footer)
    if len(short) <= max_chars:
        return short

    # Cuối cùng: chỉ stats + warnings, không chi tiết
    return base_txt[:max_chars]


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
