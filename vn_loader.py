"""
vn_loader.py — DataLoader cho thị trường chứng khoán Việt Nam.

Tái sử dụng logic fetch từ analyzer.py (Entrade/DNSE primary, Fireant fallback).
Output: pandas DataFrame với columns [date, open, high, low, close, volume].

Usage:
    from vn_loader import load_vn_ohlcv
    df = load_vn_ohlcv("VCB", days=365)
"""

from __future__ import annotations

import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
ENTRADE_BASE  = "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
FIREANT_BASE  = "https://restv2.fireant.vn"
TCBS_BASE     = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock"
DEFAULT_DAYS  = 365
REQUEST_TIMEOUT = 15


def _to_date_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


# ── Source 1: Entrade/DNSE ────────────────────────────────────────────────────
def _fetch_entrade(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """
    Lấy OHLCV từ Entrade API (primary source, không cần auth).
    """
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 30)  # buffer thêm 30 ngày phòng holiday

    params = {
        "resolution": "D",
        "symbol":     symbol,
        "from":       int(start_dt.timestamp()),
        "to":         int(end_dt.timestamp()),
    }
    try:
        r = requests.get(ENTRADE_BASE, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        if not data.get("t"):
            logger.debug(f"Entrade: empty data for {symbol}")
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
        logger.warning(f"Entrade fail for {symbol}: {e}")
        return None


# ── Source 2: Fireant ─────────────────────────────────────────────────────────
def _fetch_fireant(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """
    Lấy OHLCV từ Fireant API (Bearer token từ env FIREANT_TOKEN).
    """
    token = os.environ.get("FIREANT_TOKEN", "")
    if not token:
        logger.debug("FIREANT_TOKEN chưa set, bỏ qua Fireant source")
        return None

    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 30)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    params = {
        "startDate": _to_date_str(start_dt),
        "endDate":   _to_date_str(end_dt),
        "offset":    0,
        "limit":     days + 50,
    }
    try:
        url = f"{FIREANT_BASE}/symbols/{symbol}/historical-quotes"
        r   = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        raw = r.json()

        if not raw:
            return None

        records = []
        for item in raw:
            records.append({
                "date":   pd.to_datetime(item.get("date", item.get("tradingDate", ""))),
                "open":   float(item.get("priceOpen",  item.get("open",  0))) * 1000,
                "high":   float(item.get("priceHigh",  item.get("high",  0))) * 1000,
                "low":    float(item.get("priceLow",   item.get("low",   0))) * 1000,
                "close":  float(item.get("priceClose", item.get("close", 0))) * 1000,
                "volume": float(item.get("totalVolume", item.get("volume", 0))),
            })

        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        df = df.tail(days).reset_index(drop=True)
        logger.info(f"Fireant OK: {symbol} {len(df)} bars")
        return df

    except Exception as e:
        logger.warning(f"Fireant fail for {symbol}: {e}")
        return None


# ── Source 3: TCBS ────────────────────────────────────────────────────────────
def _fetch_tcbs(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """
    Lấy OHLCV từ TCBS API (không cần auth, fallback cuối).
    """
    try:
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=days + 30)

        url = f"{TCBS_BASE}/{symbol}/bars-long-term"
        params = {
            "resolution": "D",
            "from":       int(start_dt.timestamp()),
            "to":         int(end_dt.timestamp()),
            "type":       "stock",
        }
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        bars = data.get("data", [])
        if not bars:
            return None

        records = []
        for b in bars:
            records.append({
                "date":   pd.to_datetime(b.get("tradingDate", b.get("t", "")), unit="s"
                          if isinstance(b.get("t"), (int, float)) else None),
                "open":   float(b.get("open",  0)),
                "high":   float(b.get("high",  0)),
                "low":    float(b.get("low",   0)),
                "close":  float(b.get("close", 0)),
                "volume": float(b.get("volume", 0)),
            })

        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        df = df.tail(days).reset_index(drop=True)
        logger.info(f"TCBS OK: {symbol} {len(df)} bars")
        return df

    except Exception as e:
        logger.warning(f"TCBS fail for {symbol}: {e}")
        return None


# ── Source 4: vnstock fallback ────────────────────────────────────────────────
def _fetch_vnstock(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """Fallback dùng vnstock nếu cài sẵn."""
    try:
        from vnstock import Vnstock  # type: ignore
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(days=days + 30)
        stock    = Vnstock().stock(symbol=symbol, source="VCI")
        df_raw   = stock.quote.history(
            start=_to_date_str(start_dt),
            end=_to_date_str(end_dt),
            interval="1D",
        )
        if df_raw is None or df_raw.empty:
            return None

        # Chuẩn hóa column names
        col_map = {}
        for c in df_raw.columns:
            cl = c.lower()
            if "time" in cl or "date" in cl:
                col_map[c] = "date"
            elif cl in ("open", "o"):
                col_map[c] = "open"
            elif cl in ("high", "h"):
                col_map[c] = "high"
            elif cl in ("low", "l"):
                col_map[c] = "low"
            elif cl in ("close", "c"):
                col_map[c] = "close"
            elif cl in ("volume", "vol", "v"):
                col_map[c] = "volume"
        df_raw = df_raw.rename(columns=col_map)

        needed = ["date", "open", "high", "low", "close", "volume"]
        if not all(c in df_raw.columns for c in needed):
            return None

        df = df_raw[needed].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df = df.tail(days).reset_index(drop=True)
        logger.info(f"vnstock OK: {symbol} {len(df)} bars")
        return df

    except Exception as e:
        logger.warning(f"vnstock fail for {symbol}: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────
def load_vn_ohlcv(
    symbol: str,
    days: int = DEFAULT_DAYS,
    min_bars: int = 60,
) -> pd.DataFrame:
    """
    Load OHLCV data cho mã chứng khoán Việt Nam.

    Waterfall: Entrade → Fireant → TCBS → vnstock

    Args:
        symbol:   Mã cổ phiếu (VD: "VCB", "HPG")
        days:     Số ngày lịch sử cần lấy
        min_bars: Số bars tối thiểu để coi là hợp lệ

    Returns:
        DataFrame với columns [date, open, high, low, close, volume]
        date là UTC naive datetime.

    Raises:
        ValueError: Nếu tất cả sources đều fail hoặc data không đủ.
    """
    symbol = symbol.upper().strip()

    for source_fn, name in [
        (_fetch_entrade,  "Entrade"),
        (_fetch_fireant,  "Fireant"),
        (_fetch_tcbs,     "TCBS"),
        (_fetch_vnstock,  "vnstock"),
    ]:
        try:
            df = source_fn(symbol, days)
            if df is not None and len(df) >= min_bars:
                df = _clean_ohlcv(df)
                logger.info(f"load_vn_ohlcv({symbol}): {name} OK, {len(df)} bars")
                return df
        except Exception as e:
            logger.warning(f"load_vn_ohlcv: {name} exception: {e}")
            continue

    raise ValueError(
        f"Khong the lay du lieu cho {symbol}. "
        f"Tat ca sources deu that bai hoac co it hon {min_bars} bars."
    )


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch DataFrame OHLCV:
    - Xóa hàng có giá trị null hoặc <= 0
    - Chuẩn hóa dtype
    - Đảm bảo tăng dần theo date
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    # Loại bỏ hàng giá null hoặc <= 0
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df


# ── Utility: lấy thông tin cơ bản ────────────────────────────────────────────
def get_vn_info(symbol: str) -> dict:
    """
    Lấy thông tin cơ bản của mã: tên công ty, sàn, ngành.
    Dùng Fireant nếu có token, fallback về dict rỗng.
    """
    token = os.environ.get("FIREANT_TOKEN", "")
    if not token:
        return {"symbol": symbol, "name": symbol, "exchange": "HSX/HNX", "industry": "N/A"}

    try:
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(
            f"{FIREANT_BASE}/symbols/{symbol}",
            headers=headers, timeout=10
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
        return {"symbol": symbol, "name": symbol, "exchange": "HSX/HNX", "industry": "N/A"}
