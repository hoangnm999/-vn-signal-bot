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

def find_similar(
    symbol:        str,
    target_vector: dict,
    top_n:         int = 3,
    years:         int = 5,
    exclude_days:  int = 90,
) -> Optional[list]:
    """
    Tìm top_n ngày lịch sử có vector tương đồng nhất với target_vector.

    Args:
        symbol:        Mã CK
        target_vector: dict từ compute_state_vector()
        top_n:         Số kết quả trả về
        years:         Giới hạn tìm kiếm trong N năm gần nhất
        exclude_days:  Bỏ qua N ngày gần nhất (tránh match với chính nó)

    Returns:
        list of dict, mỗi phần tử:
        {
            "date":      str,
            "similarity": float,   # cosine similarity [0, 1]
            "close":     float,    # giá tại ngày đó
            "fwd_30":    float,    # % return sau 30 ngày
            "fwd_60":    float,    # % return sau 60 ngày
            "fwd_90":    float,    # % return sau 90 ngày
            "outcome":   str,      # "TANG", "GIAM", "DI NGANG"
        }
        None nếu cache chưa có hoặc lỗi.
    """
    symbol = symbol.upper()

    if not cache_exists(symbol):
        return None

    # Load cache
    try:
        t0       = time.time()
        cache_df = pd.read_csv(_cache_path(symbol))
        logger.debug(f"find_similar({symbol}): loaded {len(cache_df)} rows in {time.time()-t0:.2f}s")
    except Exception as e:
        logger.warning(f"find_similar({symbol}): load cache fail: {e}")
        return None

    if len(cache_df) < top_n + exclude_days + 90:
        logger.warning(f"find_similar({symbol}): cache quá ngắn ({len(cache_df)} rows)")
        return None

    # Giới hạn theo years và bỏ exclude_days cuối
    cutoff_start = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    cutoff_end   = (datetime.now() - timedelta(days=exclude_days)).strftime("%Y-%m-%d")
    search_df    = cache_df[
        (cache_df["date"] >= cutoff_start) &
        (cache_df["date"] <= cutoff_end)
    ].copy().reset_index(drop=True)

    if len(search_df) < top_n:
        return None

    # Tính cosine similarity với target_vector
    target_list = vector_to_list(target_vector)
    target_arr  = np.array(target_list, dtype=float)
    t_norm      = np.linalg.norm(target_arr)

    # Lấy ma trận vector (bỏ cột date, close)
    vec_cols  = VECTOR_KEYS
    available_cols = [c for c in vec_cols if c in search_df.columns]
    if len(available_cols) < VECTOR_DIM - 2:
        logger.warning(f"find_similar: cache thiếu columns: {set(vec_cols) - set(available_cols)}")
        return None

    mat = search_df[available_cols].fillna(0.0).values.astype(float)

    # Vectorized cosine similarity
    if t_norm == 0:
        return None
    dot_prods   = mat @ target_arr
    mat_norms   = np.linalg.norm(mat, axis=1)
    mat_norms   = np.where(mat_norms == 0, 1e-9, mat_norms)
    similarities = dot_prods / (mat_norms * t_norm)

    # Top N indices
    top_indices = np.argsort(-similarities)[:top_n * 3]  # lấy dư để filter

    results = []
    for idx in top_indices:
        if len(results) >= top_n:
            break

        sim       = float(similarities[idx])
        row       = search_df.iloc[idx]
        date_str  = str(row["date"])
        close_val = float(row.get("close", 0))

        # Tính forward returns từ cache (dùng close của các ngày sau)
        fwd = _calc_forward_returns(cache_df, date_str, close_val)

        results.append({
            "date":       date_str,
            "similarity": round(sim, 4),
            "close":      close_val,
            "fwd_30":     fwd.get(30),
            "fwd_60":     fwd.get(60),
            "fwd_90":     fwd.get(90),
            "outcome":    _classify_outcome(fwd.get(30)),
        })

    return results if results else None


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

def format_analog_report(
    symbol:   str,
    analogs:  list,
    current_vec: dict,
) -> str:
    """Tạo text báo cáo phân tích tương đồng."""
    if not analogs:
        return f"Khong tim thay ngay tuong dong cho {symbol}."

    lines = [
        f"PHAN TICH TUONG DONG: {symbol}",
        "═" * 40,
        "Tim kiem ngay lich su co trang thai ky thuat",
        "tuong dong nhat voi hien tai (cosine similarity).",
        "",
        f"TOP {len(analogs)} NGAY TUONG DONG:",
        "─" * 40,
    ]

    for i, a in enumerate(analogs, 1):
        sim   = a["similarity"]
        date  = a["date"]
        close = a["close"]
        f30   = a["fwd_30"]
        f60   = a["fwd_60"]
        f90   = a["fwd_90"]
        out   = a["outcome"]

        em = "🟢" if "TANG" in out else "🔴" if "GIAM" in out else "🟡"

        lines.append(f"#{i} — {date} | Tuong dong: {sim:.1%}")
        lines.append(f"    Gia luc do: {close:,.0f} | Ket qua: {em} {out}")

        fwd_parts = []
        if f30 is not None: fwd_parts.append(f"30D:{f30:+.1f}%")
        if f60 is not None: fwd_parts.append(f"60D:{f60:+.1f}%")
        if f90 is not None: fwd_parts.append(f"90D:{f90:+.1f}%")
        if fwd_parts:
            lines.append(f"    Forward returns: {' | '.join(fwd_parts)}")
        lines.append("")

    # Thống kê tổng hợp
    valid_f30 = [a["fwd_30"] for a in analogs if a["fwd_30"] is not None]
    if valid_f30:
        avg30  = np.mean(valid_f30)
        pos30  = sum(1 for x in valid_f30 if x > 0)
        n30    = len(valid_f30)
        lines += [
            "─" * 40,
            "THONG KE (sau 30 ngay):",
            f"  TB return : {avg30:+.2f}%",
            f"  Tang/Giam : {pos30}/{n30} ({pos30/n30:.0%} tang)",
            "─" * 40,
        ]

    lines.append("Luu y: Phan tich tuong dong chi mang tinh tham khao.")
    lines.append("Qua khu khong dam bao tuong lai.")
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
