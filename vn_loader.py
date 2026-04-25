"""
vn_loader.py — DataLoader cho thị trường chứng khoán Việt Nam.

NGUỒN DỮ LIỆU (waterfall):
  1. analyzer.get_price_data()   — TÁI SỬ DỤNG logic đang hoạt động trong /check
  2. Entrade/DNSE API            — chart-api v2 (với User-Agent header)
  3. DNSE API v3                 — endpoint mới hơn
  4. Fireant API                 — cần FIREANT_TOKEN env var
  5. TCBS API                    — v2 endpoint (v1 đã 404)
  6. vnstock                     — nhiều sources: TCBS/VCI/MSN

Tất cả sources đều được log ở WARNING level khi fail để dễ debug.
"""

from __future__ import annotations

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_DAYS     = 365
REQUEST_TIMEOUT  = 20   # tăng lên 20s cho Railway

# Headers giả browser để bypass một số CDN block
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Referer": "https://banggia.vn/",
    "Origin": "https://banggia.vn",
}


def _to_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 0: analyzer.py get_price_data() — PRIMARY (đang hoạt động)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_from_analyzer(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """
    Tái sử dụng get_price_data() từ analyzer.py — đây là source đang work
    vì /check VCB hoạt động bình thường.

    analyzer.get_price_data() trả về dict:
        {"success": True/False, "df": pd.DataFrame, "error": str}
    df có columns: open, high, low, close, volume (index = date hoặc có cột date)
    """
    try:
        from analyzer import get_price_data  # type: ignore
        result = get_price_data(symbol, days + 60)   # buffer thêm 60 ngày

        if not result or not result.get("success"):
            err = result.get("error", "unknown") if result else "no result"
            logger.warning(f"analyzer.get_price_data fail for {symbol}: {err}")
            return None

        df = result.get("df")
        if df is None or df.empty:
            logger.warning(f"analyzer.get_price_data: empty df for {symbol}")
            return None

        df = df.copy()
        # Lowercase column names
        df.columns = [str(c).lower() for c in df.columns]

        # Đảm bảo có cột 'date':
        # Case 1: đã có cột 'date'
        if "date" not in df.columns:
            # Case 2: index là DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                df.index.name = "date"
                df = df.reset_index()
            # Case 3: index có tên 'date' hoặc 'time'
            elif df.index.name and df.index.name.lower() in ("date", "time", "tradingdate"):
                df.index.name = "date"
                df = df.reset_index()
            # Case 4: không có thông tin date nào → tạo từ số ngày
            else:
                logger.warning(f"analyzer df không có cột date, cols={list(df.columns)}")
                return None

        # Map tên cột khác về chuẩn
        rename_map = {}
        for c in df.columns:
            if c in ("o", "open_price") and "open" not in df.columns:
                rename_map[c] = "open"
            elif c in ("h", "high_price") and "high" not in df.columns:
                rename_map[c] = "high"
            elif c in ("l", "low_price") and "low" not in df.columns:
                rename_map[c] = "low"
            elif c in ("c", "close_price", "adjclose") and "close" not in df.columns:
                rename_map[c] = "close"
            elif c in ("vol", "v", "volume_match") and "volume" not in df.columns:
                rename_map[c] = "volume"
        if rename_map:
            df = df.rename(columns=rename_map)

        needed = ["date", "open", "high", "low", "close", "volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            logger.warning(
                f"analyzer df thiếu columns: {missing} | có: {list(df.columns)[:8]}"
            )
            return None

        df = df[needed].copy()
        df["date"]   = pd.to_datetime(df["date"], errors="coerce")
        df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df = df.dropna(subset=["date", "close"])
        df = df[df["close"] > 0]
        df = df.sort_values("date").tail(days).reset_index(drop=True)

        logger.info(f"analyzer.get_price_data OK: {symbol} {len(df)} bars")
        return df

    except ImportError:
        logger.debug("analyzer.py không có — bỏ qua source này")
        return None
    except Exception as e:
        logger.warning(f"analyzer.get_price_data exception for {symbol}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: Entrade/DNSE chart-api v2
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_entrade(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """Entrade chart-api v2 — cần User-Agent header để bypass CDN."""
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 60)

    params = {
        "resolution": "D",
        "symbol":     symbol,
        "from":       int(start_dt.timestamp()),
        "to":         int(end_dt.timestamp()),
    }
    try:
        r = requests.get(
            "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock",
            params=params,
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )

        if r.status_code == 403:
            logger.warning(f"Entrade 403 for {symbol}: Host not in allowlist (server-side block)")
            return None

        r.raise_for_status()
        data = r.json()

        if not data.get("t"):
            logger.warning(f"Entrade empty response for {symbol}: keys={list(data.keys())}")
            return None

        df = pd.DataFrame({
            "date":   pd.to_datetime(data["t"], unit="s"),
            "open":   data["o"],
            "high":   data["h"],
            "low":    data["l"],
            "close":  data["c"],
            "volume": data["v"],
        })
        df = df.sort_values("date").reset_index(drop=True)
        df = df.tail(days).reset_index(drop=True)
        logger.info(f"Entrade OK: {symbol} {len(df)} bars")
        return df

    except Exception as e:
        logger.warning(f"Entrade fail for {symbol}: {type(e).__name__}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: DNSE API v3 (endpoint mới)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_dnse_v3(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """DNSE/Entrade API v3 — thử các endpoint mới hơn."""
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 60)

    endpoints = [
        # Endpoint mới của Entrade
        (
            "https://services.entrade.com.vn/chart-api/v3/ohlcs/stock",
            {"resolution": "D", "symbol": symbol,
             "from": int(start_dt.timestamp()), "to": int(end_dt.timestamp())},
        ),
        # DNSE market-data endpoint
        (
            f"https://marketdata.dnse.com.vn/api/v2/trade-insights/stocks/{symbol}/historical-price",
            {"resolution": "D", "fromDate": _to_date_str(start_dt),
             "toDate": _to_date_str(end_dt)},
        ),
        # SSI API
        (
            "https://fc-data.ssi.com.vn/api/v2/GetOhlcFromHistory",
            {"Symbol": symbol, "StartDate": _to_date_str(start_dt),
             "EndDate": _to_date_str(end_dt), "Market": "HOSE", "Frequency": "D",
             "PageIndex": 1, "PageSize": days + 60},
        ),
    ]

    for url, params in endpoints:
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                logger.debug(f"DNSE v3 {url[:50]}: HTTP {r.status_code}")
                continue

            data = r.json()

            # Thử parse format Entrade-like (t/o/h/l/c/v)
            if data.get("t"):
                df = pd.DataFrame({
                    "date":   pd.to_datetime(data["t"], unit="s"),
                    "open":   data["o"], "high": data["h"],
                    "low":    data["l"], "close": data["c"],
                    "volume": data["v"],
                })
                df = df.sort_values("date").tail(days).reset_index(drop=True)
                logger.info(f"DNSE v3 OK: {symbol} {len(df)} bars from {url[:50]}")
                return df

            # Thử format records
            records_key = next((k for k in ("data", "items", "Data", "result") if data.get(k)), None)
            if records_key:
                raw_records = data[records_key]
                if not raw_records:
                    continue
                first = raw_records[0] if isinstance(raw_records, list) else {}
                # Detect date key
                date_key = next((k for k in ("TradingDate","tradingDate","Date","date","t") if k in first), None)
                if not date_key:
                    continue
                records = []
                for b in raw_records:
                    ts = b.get(date_key)
                    try:
                        dt = pd.to_datetime(ts, unit="s") if isinstance(ts, (int, float)) else pd.to_datetime(ts)
                    except Exception:
                        continue
                    records.append({
                        "date":   dt,
                        "open":   float(b.get("Open", b.get("open", b.get("o", 0)))),
                        "high":   float(b.get("High", b.get("high", b.get("h", 0)))),
                        "low":    float(b.get("Low",  b.get("low",  b.get("l", 0)))),
                        "close":  float(b.get("Close",b.get("close",b.get("c", 0)))),
                        "volume": float(b.get("Volume",b.get("volume",b.get("v",0)))),
                    })
                if len(records) >= 60:
                    df = pd.DataFrame(records).sort_values("date").tail(days).reset_index(drop=True)
                    logger.info(f"DNSE v3 records OK: {symbol} {len(df)} bars")
                    return df

        except Exception as e:
            logger.debug(f"DNSE v3 endpoint {url[:50]} fail: {e}")
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: Fireant API
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_fireant(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """Fireant historical-quotes — cần FIREANT_TOKEN env var."""
    token = os.environ.get("FIREANT_TOKEN", "").strip()
    if not token:
        logger.debug("FIREANT_TOKEN chưa set — bỏ qua Fireant")
        return None

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 60)

    headers = dict(_HEADERS)
    headers["Authorization"] = f"Bearer {token}"

    params = {
        "startDate": _to_date_str(start_dt),
        "endDate":   _to_date_str(end_dt),
        "offset":    0,
        "limit":     days + 80,
    }
    try:
        url = f"https://restv2.fireant.vn/symbols/{symbol}/historical-quotes"
        r   = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)

        if r.status_code == 403:
            logger.warning(f"Fireant 403 for {symbol}: token hết hạn hoặc bị block")
            return None
        if r.status_code == 401:
            logger.warning(f"Fireant 401 for {symbol}: FIREANT_TOKEN không hợp lệ")
            return None

        r.raise_for_status()
        raw = r.json()

        if not raw:
            logger.warning(f"Fireant empty for {symbol}")
            return None

        records = []
        for item in raw:
            # Fireant trả giá chia 1000 hoặc đầy đủ — detect tự động
            close_raw = float(item.get("priceClose", item.get("close", 0)))
            # Nếu giá < 1000 thì nhân 1000 (đơn vị ngàn đồng)
            multiplier = 1000 if close_raw < 1000 else 1
            records.append({
                "date":   pd.to_datetime(item.get("date", item.get("tradingDate", ""))),
                "open":   float(item.get("priceOpen",  item.get("open",  0))) * multiplier,
                "high":   float(item.get("priceHigh",  item.get("high",  0))) * multiplier,
                "low":    float(item.get("priceLow",   item.get("low",   0))) * multiplier,
                "close":  close_raw * multiplier,
                "volume": float(item.get("totalVolume", item.get("volume", 0))),
            })

        df = pd.DataFrame(records).sort_values("date").tail(days).reset_index(drop=True)
        logger.info(f"Fireant OK: {symbol} {len(df)} bars")
        return df

    except Exception as e:
        logger.warning(f"Fireant fail for {symbol}: {type(e).__name__}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 4: TCBS API v2 (v1 đã 404)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_tcbs(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """TCBS API — thử v2 endpoint (v1 đã trả 404)."""
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 60)

    # Thử các endpoint TCBS theo thứ tự ưu tiên
    endpoints = [
        # v2 endpoint mới
        f"https://apipubaws.tcbs.com.vn/stock-insight/v2/stock/{symbol}/bars-long-term",
        # Endpoint search (dùng date string thay timestamp)
        f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/{symbol}/bars",
    ]

    params_list = [
        {"resolution": "D", "from": int(start_dt.timestamp()), "to": int(end_dt.timestamp()), "type": "stock"},
        {"resolution": "D", "from": int(start_dt.timestamp()), "to": int(end_dt.timestamp())},
    ]

    for url, params in zip(endpoints, params_list):
        try:
            r = requests.get(url, params=params, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
            if r.status_code not in (200,):
                logger.debug(f"TCBS {url[-30:]}: HTTP {r.status_code}")
                continue

            data = r.json()
            bars = data.get("data", data.get("bars", []))
            if not bars:
                logger.debug(f"TCBS {url[-30:]}: empty bars")
                continue

            records = []
            for b in bars:
                # TCBS format: TradingDate (string) hoặc t (timestamp)
                ts = b.get("TradingDate", b.get("tradingDate", b.get("t", None)))
                try:
                    if isinstance(ts, (int, float)):
                        dt = pd.to_datetime(ts, unit="s")
                    else:
                        dt = pd.to_datetime(ts)
                except Exception:
                    continue
                records.append({
                    "date":   dt,
                    "open":   float(b.get("Open",   b.get("open",   b.get("o",  0)))),
                    "high":   float(b.get("High",   b.get("high",   b.get("h",  0)))),
                    "low":    float(b.get("Low",    b.get("low",    b.get("l",  0)))),
                    "close":  float(b.get("Close",  b.get("close",  b.get("c",  0)))),
                    "volume": float(b.get("Volume", b.get("volume", b.get("v",  0)))),
                })

            if len(records) >= 60:
                df = pd.DataFrame(records).sort_values("date").tail(days).reset_index(drop=True)
                logger.info(f"TCBS OK: {symbol} {len(df)} bars from {url[-30:]}")
                return df

        except Exception as e:
            logger.debug(f"TCBS {url[-30:]} fail: {e}")
            continue

    logger.warning(f"TCBS: tất cả endpoints fail cho {symbol}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 5: vnstock (nhiều sources)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_vnstock(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """vnstock với nhiều sources: TCBS, VCI, MSN — xử lý column name thay đổi."""
    try:
        from vnstock import Vnstock  # type: ignore
    except ImportError:
        logger.debug("vnstock chưa cài")
        return None

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 60)
    start_s  = _to_date_str(start_dt)
    end_s    = _to_date_str(end_dt)

    for src in ("TCBS", "VCI", "MSN"):
        try:
            stock  = Vnstock().stock(symbol=symbol, source=src)
            df_raw = stock.quote.history(start=start_s, end=end_s, interval="1D")

            if df_raw is None or df_raw.empty:
                logger.debug(f"vnstock {src}: empty for {symbol}")
                continue

            df = df_raw.copy()
            # Reset MultiIndex nếu có
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            elif not isinstance(df.index, pd.RangeIndex):
                df = df.reset_index()

            # Lowercase tất cả columns
            df.columns = [str(c).lower().strip() for c in df.columns]

            # Mapping column names vnstock 3.x
            rename_map = {}
            for c in df.columns:
                if c in ("time", "tradingdate", "trading_date", "datetime", "index"):
                    rename_map[c] = "date"
                elif c in ("o", "open_price", "open"):
                    if c != "open": rename_map[c] = "open"
                elif c in ("h", "high_price", "high"):
                    if c != "high": rename_map[c] = "high"
                elif c in ("l", "low_price", "low"):
                    if c != "low": rename_map[c] = "low"
                elif c in ("c", "close_price", "close"):
                    if c != "close": rename_map[c] = "close"
                elif c in ("vol", "v", "volume_match", "match_vol", "volume"):
                    if c != "volume": rename_map[c] = "volume"
            df = df.rename(columns=rename_map)

            needed = ["date", "open", "high", "low", "close", "volume"]
            missing = [c for c in needed if c not in df.columns]
            if missing:
                logger.debug(f"vnstock {src}: missing cols {missing}, have={list(df.columns)[:8]}")
                continue

            df = df[needed].copy()
            df["date"]   = pd.to_datetime(df["date"], errors="coerce")
            df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

            # Giá vnstock: nếu < 1000 thì đang ở đơn vị ngàn đồng
            if df["close"].median() < 1000:
                for col in ("open", "high", "low", "close"):
                    df[col] = pd.to_numeric(df[col], errors="coerce") * 1000

            df = df.dropna(subset=["date", "close"])
            df = df[df["close"] > 0]
            df = df.sort_values("date").tail(days).reset_index(drop=True)

            if len(df) >= 60:
                logger.info(f"vnstock {src} OK: {symbol} {len(df)} bars")
                return df
            else:
                logger.debug(f"vnstock {src}: chỉ {len(df)} bars < 60")

        except Exception as e:
            logger.warning(f"vnstock {src} fail for {symbol}: {type(e).__name__}: {e}")
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def load_vn_ohlcv(
    symbol: str,
    days: int = DEFAULT_DAYS,
    min_bars: int = 60,
) -> pd.DataFrame:
    """
    Load OHLCV cho mã chứng khoán Việt Nam.

    Waterfall (theo thứ tự ưu tiên):
      0. analyzer.get_price_data()  — tái dụng logic đang work trong /check
      1. Entrade/DNSE chart-api v2
      2. DNSE v3 + SSI endpoints
      3. Fireant (cần FIREANT_TOKEN)
      4. TCBS v2
      5. vnstock (TCBS/VCI/MSN sources)

    Args:
        symbol:   Mã CK (VD: "VCB", "HPG")
        days:     Số ngày lịch sử cần lấy
        min_bars: Tối thiểu bars hợp lệ (default 60)

    Returns:
        DataFrame với columns [date, open, high, low, close, volume]

    Raises:
        ValueError: Nếu tất cả sources fail.
    """
    symbol   = symbol.upper().strip()
    failures = []

    sources = [
        (_fetch_from_analyzer, "analyzer.get_price_data"),
        (_fetch_entrade,       "Entrade v2"),
        (_fetch_dnse_v3,       "DNSE v3/SSI"),
        (_fetch_fireant,       "Fireant"),
        (_fetch_tcbs,          "TCBS"),
        (_fetch_vnstock,       "vnstock"),
    ]

    for fn, name in sources:
        try:
            df = fn(symbol, days)
            if df is not None and len(df) >= min_bars:
                df = _clean_ohlcv(df)
                if len(df) >= min_bars:
                    logger.info(f"load_vn_ohlcv({symbol}): SUCCESS via {name} | {len(df)} bars")
                    return df
                else:
                    msg = f"{name}: {len(df)} bars sau clean (< {min_bars})"
                    failures.append(msg)
                    logger.warning(f"load_vn_ohlcv: {msg}")
            elif df is not None:
                msg = f"{name}: chỉ {len(df)} bars (< {min_bars})"
                failures.append(msg)
                logger.warning(f"load_vn_ohlcv: {msg}")
            else:
                failures.append(f"{name}: trả về None")
        except Exception as e:
            msg = f"{name}: {type(e).__name__}: {e}"
            failures.append(msg)
            logger.warning(f"load_vn_ohlcv exception [{name}]: {e}")

    # Tất cả fail → raise với thông tin đầy đủ
    failure_detail = " | ".join(failures)
    raise ValueError(
        f"Khong the lay du lieu cho {symbol} ({days} ngay). "
        f"Da thu {len(sources)} sources. Chi tiet: {failure_detail}"
    )


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch DataFrame OHLCV: null, giá <= 0, duplicate, sort."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def get_vn_info(symbol: str) -> dict:
    """Lấy thông tin cơ bản của mã — dùng Fireant nếu có token."""
    token = os.environ.get("FIREANT_TOKEN", "").strip()
    base  = {"symbol": symbol, "name": symbol, "exchange": "HSX/HNX", "industry": "N/A"}
    if not token:
        return base
    try:
        headers = dict(_HEADERS)
        headers["Authorization"] = f"Bearer {token}"
        r = requests.get(
            f"https://restv2.fireant.vn/symbols/{symbol}",
            headers=headers, timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return {
            "symbol":   symbol,
            "name":     data.get("companyName", symbol),
            "exchange": data.get("exchange", "N/A"),
            "industry": data.get("industryName", "N/A"),
        }
    except Exception:
        return base


# ══════════════════════════════════════════════════════════════════════════════
# DEBUG HELPER — gọi từ /debug command trong bot
# ══════════════════════════════════════════════════════════════════════════════

def debug_sources(symbol: str, days: int = 100) -> str:
    """
    Kiểm tra tất cả sources và trả về báo cáo.
    Gọi từ bot: result = debug_sources("VCB")
    """
    import time
    lines  = [f"=== VN Loader Debug: {symbol} ({days}D) ==="]
    symbol = symbol.upper()

    sources = [
        (_fetch_from_analyzer, "analyzer.get_price_data"),
        (_fetch_entrade,       "Entrade v2"),
        (_fetch_dnse_v3,       "DNSE v3/SSI"),
        (_fetch_fireant,       "Fireant"),
        (_fetch_tcbs,          "TCBS"),
        (_fetch_vnstock,       "vnstock"),
    ]

    for fn, name in sources:
        t0 = time.time()
        try:
            df   = fn(symbol, days)
            elapsed = round(time.time() - t0, 1)
            if df is not None and len(df) > 0:
                last_close = df["close"].iloc[-1]
                last_date  = df["date"].iloc[-1].strftime("%Y-%m-%d") if hasattr(df["date"].iloc[-1], "strftime") else str(df["date"].iloc[-1])
                lines.append(f"  ✅ {name}: {len(df)} bars | last={last_close:,.0f} @ {last_date} ({elapsed}s)")
            else:
                lines.append(f"  ❌ {name}: None/empty ({elapsed}s)")
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            lines.append(f"  ❌ {name}: {type(e).__name__}: {str(e)[:80]} ({elapsed}s)")

    return "\n".join(lines)
