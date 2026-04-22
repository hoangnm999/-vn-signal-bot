"""
vn_market.py — Vietnam Stock Market Data Loader for Vibe-Trading (HKUDS)

Implement DataLoaderProtocol chuẩn của Vibe-Trading.
Đặt file này tại: agent/backtest/loaders/vn_market.py

Data sources (theo thứ tự ưu tiên):
  1. Entrade/DNSE API   — OHLCV free, không cần key, ít bị block
  2. vnstock (KBS/VCI) — fallback, cần VNAI_API_KEY để unlock đầy đủ

Quy ước symbol:
  VCB.VN, HPG.VN, FPT.VN  → suffix .VN cho HOSE/HNX
  VCB, HPG, FPT            → cũng được chấp nhận (tự thêm .VN)

Backtest rules cho TTCK VN:
  - T+2.5: mua hôm nay, bán sớm nhất T+3
  - Phí giao dịch: 0.15% một chiều (standard retail)
  - Giới hạn biên độ: ±7% (HOSE), ±10% (HNX/UPCOM)
  - Đơn vị giao dịch tối thiểu: 100 cổ phiếu/lô

Cách đăng ký vào registry:
  Thêm "backtest.loaders.vn_market" vào _loader_modules trong registry.py
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

# Import từ Vibe-Trading framework
from backtest.loaders.base import validate_date_range
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

# ─── Cấu hình ────────────────────────────────────────────────────────────────

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Referer":         "https://entrade.com.vn/",
    "Origin":          "https://entrade.com.vn",
}

# VN transaction costs cho backtest engine
VN_BACKTEST_CONFIG = {
    "commission_rate": 0.0015,    # 0.15% một chiều
    "settlement_days": 3,          # T+3 (thực tế T+2.5, làm tròn lên T+3)
    "lot_size": 100,               # 1 lô = 100 cổ phiếu
    "price_limit_hose": 0.07,      # ±7% HOSE
    "price_limit_hnx":  0.10,      # ±10% HNX/UPCOM
    "market": "vn_equity",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_symbol(code: str) -> str:
    """
    Chuẩn hóa symbol:
      VCB.VN → VCB
      vcb    → VCB
      VCB    → VCB
    """
    s = code.strip().upper()
    if s.endswith(".VN"):
        return s[:-3]
    return s


def _fetch_entrade(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Lấy OHLCV từ Entrade/DNSE public API.
    Không cần API key, hoạt động tốt từ datacenter IP (Railway, Render...).
    """
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts   = int(datetime.strptime(end_date,   "%Y-%m-%d").timestamp()) + 86400

    url = (
        f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
        f"?from={start_ts}&to={end_ts}&symbol={symbol}&resolution=D"
    )
    try:
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=12)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ValueError(f"Entrade API error for {symbol}: {e}") from e

    if "t" not in data or not data["t"]:
        raise ValueError(f"Entrade returned empty data for {symbol}")

    df = pd.DataFrame({
        "trade_date": pd.to_datetime(data["t"], unit="s").strftime("%Y-%m-%d"),
        "open":       [float(x) for x in data["o"]],
        "high":       [float(x) for x in data["h"]],
        "low":        [float(x) for x in data["l"]],
        "close":      [float(x) for x in data["c"]],
        "volume":     [float(x) for x in data["v"]],
    })
    df = df.sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def _fetch_vnstock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fallback: vnstock library (KBS source).
    Cần cài: pip install vnstock==3.5.1
    """
    try:
        from vnstock import Vnstock
    except ImportError:
        raise ValueError("vnstock not installed. Run: pip install vnstock==3.5.1")

    for source in ["KBS", "VCI"]:
        try:
            stock = Vnstock().stock(symbol=symbol, source=source)
            df = stock.quote.history(start=start_date, end=end_date, interval="1D")
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                if "time" in df.columns:
                    df = df.rename(columns={"time": "trade_date"})
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                return df.sort_values("trade_date").reset_index(drop=True)
        except Exception as e:
            logger.debug("vnstock %s failed for %s: %s", source, symbol, e)
            continue

    raise ValueError(f"vnstock could not fetch data for {symbol}")


# ─── DataLoader Class ─────────────────────────────────────────────────────────

@register
class VNMarketLoader:
    """
    Vietnam Stock Market Data Loader cho Vibe-Trading framework.

    Implement DataLoaderProtocol của HKUDS/Vibe-Trading.
    Hỗ trợ toàn bộ mã trên HOSE, HNX, UPCOM.

    Ví dụ sử dụng trong Vibe-Trading:
        vibe-trading run -p "Backtest VCB.VN RSI strategy from 2023-01-01 to 2024-12-31"
        vibe-trading run -p "Analyze HPG.VN using technical_analysis_panel swarm"
    """

    name          = "vn_market"
    markets       = {"vn_equity"}
    requires_auth = False          # Entrade không cần auth

    def __init__(self) -> None:
        self._vnai_key = os.environ.get("VNAI_API_KEY", "")
        # Setup VNAI nếu có key
        if self._vnai_key:
            try:
                import vnai
                vnai.setup_api_key(self._vnai_key)
            except Exception:
                pass

    def is_available(self) -> bool:
        """
        Check xem Entrade API có accessible không.
        Thử fetch 1 ngày data của VCB để verify.
        """
        try:
            test_ts = int(time.time())
            url = (
                f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
                f"?from={test_ts - 86400}&to={test_ts}&symbol=VCB&resolution=D"
            )
            resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data cho danh sách mã TTCK VN.

        Args:
            codes:      List mã cổ phiếu, ví dụ ["VCB.VN", "HPG.VN", "FPT"]
            start_date: Ngày bắt đầu "YYYY-MM-DD"
            end_date:   Ngày kết thúc "YYYY-MM-DD"
            interval:   Độ phân giải (hiện chỉ hỗ trợ "1D")
            fields:     Ignored (giữ cho compatible với Protocol)

        Returns:
            Dict {symbol: DataFrame(trade_date, open, high, low, close, volume)}
            trade_date là pd.Timestamp, giá đơn vị nghìn VND (×1000 để ra VND thực)
        """
        validate_date_range(start_date, end_date)

        if interval not in ("1D", "D", "day"):
            logger.warning(
                "VNMarketLoader only supports daily interval. "
                "Got '%s', falling back to 1D.", interval
            )

        results: Dict[str, pd.DataFrame] = {}

        for code in codes:
            symbol = _normalize_symbol(code)
            df = self._fetch_single(symbol, start_date, end_date)
            if df is not None and not df.empty:
                results[code] = df
            else:
                logger.warning("No data returned for %s", code)

        return results

    def _fetch_single(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch 1 mã với fallback chain:
          Entrade (primary) → vnstock KBS (fallback)
        """
        # Source 1: Entrade
        try:
            df = _fetch_entrade(symbol, start_date, end_date)
            logger.debug("Entrade OK for %s: %d rows", symbol, len(df))
            return self._validate_ohlcv(df)
        except Exception as e:
            logger.warning("Entrade failed for %s: %s. Trying vnstock...", symbol, e)

        # Source 2: vnstock
        try:
            df = _fetch_vnstock(symbol, start_date, end_date)
            logger.debug("vnstock OK for %s: %d rows", symbol, len(df))
            return self._validate_ohlcv(df)
        except Exception as e:
            logger.error("All sources failed for %s: %s", symbol, e)
            return None

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate và chuẩn hóa DataFrame theo format Vibe-Trading yêu cầu.
        Output: columns = [trade_date, open, high, low, close, volume]
        trade_date là pd.Timestamp, không có NaN.
        """
        required = ["trade_date", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        df = df[required].copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # Bỏ rows có giá = 0 hoặc NaN (ngày không giao dịch)
        df = df[
            (df["close"] > 0) &
            (df["open"] > 0) &
            (df["volume"] >= 0)
        ].dropna(subset=["open", "high", "low", "close"])

        return df.sort_values("trade_date").reset_index(drop=True)


# ─── Backtest Engine Config cho TTCK VN ──────────────────────────────────────

def get_vn_backtest_params() -> dict:
    """
    Trả về config backtest chuẩn cho TTCK VN.
    Dùng khi gọi backtest engine của Vibe-Trading.

    Ví dụ:
        params = get_vn_backtest_params()
        # Truyền vào backtest command trong Vibe-Trading
    """
    return {
        "market":           "vn_equity",
        "data_source":      "vn_market",
        "commission":       0.0015,      # 0.15% một chiều
        "slippage":         0.001,       # 0.1% slippage estimate
        "settlement_days":  3,           # T+3
        "lot_size":         100,         # lô tối thiểu
        "initial_capital":  100_000_000, # 100 triệu VND default
        "currency":         "VND",
        "price_unit":       1000,        # Giá Entrade đơn vị nghìn VND
        "timezone":         "Asia/Ho_Chi_Minh",
        "trading_hours":    "09:00-11:30,13:00-14:45",
        "calendar":         "HOSE",
    }
