import os
import logging
_logger = logging.getLogger(__name__)
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# ── Vibe-Trading availability check — import thực sự ở dòng ~840
# 13 HKUDS engines + 2 context agents = 15 agents tổng ────────────────────────
_VIBE_AVAILABLE = False  # placeholder, sẽ được set lại khi import thực tế bên dưới

DEEPSEEK_API_KEY  = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_URL      = "https://api.deepseek.com/v1/chat/completions"
VNAI_API_KEY      = os.environ.get("VNAI_API_KEY", "")
FIREANT_TOKEN     = os.environ.get("FIREANT_TOKEN", "")   # Bearer token từ fireant.vn

# ── Lock cho _market_cache để tránh race condition ────────────────────────────
_market_cache_lock = Lock()

# ── Init VNAI key ngay khi module load ────────────────────────────────────────
if VNAI_API_KEY:
    try:
        import vnai
        vnai.setup_api_key(VNAI_API_KEY)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# DATA LAYER  — multi-source với fallback
# Primary:  Entrade (DNSE broker) — ít bị Cloudflare block nhất
# Fallback: vnstock KBS/VCI (cần VNAI key)
# ══════════════════════════════════════════════════════════════════════════════

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Referer":         "https://entrade.com.vn/",
    "Origin":          "https://entrade.com.vn",
}


def _price_from_entrade(symbol: str, days: int) -> pd.DataFrame:
    """Lấy OHLCV từ Entrade/DNSE public API"""
    import time
    to_ts   = int(time.time())
    from_ts = to_ts - days * 86400
    url = (
        f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
        f"?from={from_ts}&to={to_ts}&symbol={symbol}&resolution=D"
    )
    r = requests.get(url, headers=_BROWSER_HEADERS, timeout=12)
    r.raise_for_status()
    data = r.json()

    # Response: {"t":[ts,..], "o":[..], "h":[..], "l":[..], "c":[..], "v":[..]}
    if "t" not in data or not data["t"]:
        raise ValueError("Entrade trả về data rỗng")

    df = pd.DataFrame({
        "time":   pd.to_datetime(data["t"], unit="s").strftime("%Y-%m-%d"),
        "open":   data["o"],
        "high":   data["h"],
        "low":    data["l"],
        "close":  data["c"],
        "volume": data["v"],
    })
    return df.sort_values("time").reset_index(drop=True)


def _price_from_vnstock(symbol: str, days: int) -> pd.DataFrame:
    """Fallback: vnstock KBS source"""
    from vnstock import Vnstock
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    for src in ["KBS", "VCI"]:
        try:
            stock = Vnstock().stock(symbol=symbol, source=src)
            df = stock.quote.history(start=start, end=end, interval="1D")
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                return df.sort_values("time").reset_index(drop=True)
        except Exception:
            continue
    raise ValueError("Không lấy được dữ liệu từ vnstock KBS/VCI")


def get_price_data(symbol: str, days: int = 90) -> dict:
    # Thử Entrade trước, fallback vnstock
    for fetcher in [_price_from_entrade, _price_from_vnstock]:
        try:
            df = fetcher(symbol, days)
            if df is not None and not df.empty:
                return {"success": True, "df": df, "source": fetcher.__name__}
        except Exception as e:
            last_err = str(e)
    return {"success": False, "error": last_err}


def _fundamental_from_vnstock(symbol: str):
    """Lấy ratio từ vnstock KBS source — cần VNAI key để unlock"""
    from vnstock import Vnstock
    for period in ["quarter", "annual"]:
        try:
            stock = Vnstock().stock(symbol=symbol, source="KBS")
            ratio = stock.finance.ratio(period=period)
            if ratio is not None and not ratio.empty:
                return ratio
        except Exception:
            continue
    return None


def get_fundamental_data(symbol: str) -> dict:
    """
    Lấy fundamental data.
    Source 1: vnstock KBS (cần VNAI key unlock)
    Source 2: Dùng LLM tổng hợp kiến thức có sẵn về công ty VN
    """
    # Source 1: vnstock KBS
    try:
        ratio = _fundamental_from_vnstock(symbol)
        if ratio is not None and not ratio.empty:
            latest = ratio.iloc[0]
            cols   = list(latest.index)

            def _find(patterns, d=0):
                for p in patterns:
                    for c in cols:
                        if p.lower() in c.lower():
                            try:
                                v = float(latest[c])
                                if not pd.isna(v):
                                    return v
                            except Exception:
                                pass
                return d

            return {
                "success":        True,
                "pe":             round(_find(["P/E", "pe", "priceToEarning"]), 2),
                "pb":             round(_find(["P/B", "pb", "priceToBook"]), 2),
                "roe":            round(_find(["ROE", "roe"]) * 100, 2),
                "eps":            round(_find(["EPS", "eps", "earningPerShare"]), 0),
                "revenue_growth": round(_find(["revenueGrowth", "revenue_growth"]) * 100, 2),
                "profit_growth":  round(_find(["profitGrowth", "profit_growth"]) * 100, 2),
                "debt_equity":    round(_find(["debtOnEquity", "debt_equity", "D/E"]), 2),
                "source":         "vnstock KBS",
            }
    except Exception:
        pass

    # Source 2: LLM knowledge-based fundamental analysis
    # Dùng khi API không có — LLM có kiến thức về các công ty VN lớn
    return {
        "success":  True,
        "pe":       0,
        "pb":       0,
        "roe":      0,
        "eps":      0,
        "revenue_growth": 0,
        "profit_growth":  0,
        "debt_equity":    0,
        "source":   "llm_knowledge",
        "llm_note": symbol,   # truyền symbol để agent tự phân tích
    }


def get_foreign_flow_data(symbol: str) -> dict:
    """
    Lấy foreign flow.
    Entrade OHLCV không có foreign columns (chỉ t,o,h,l,c,v).
    Thay vào đó: phân tích price+volume pattern để ước tính
    smart money behavior - đây là cách hợp lý khi không có raw data.
    """
    try:
        # Lấy price data đã có để phân tích volume pattern
        price = get_price_data(symbol, 30)
        if not price.get("success"):
            return {"success": False, "error": "Không có price data để phân tích"}

        df = price["df"]
        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # Volume-price analysis thay cho foreign flow trực tiếp
        # Ngày tăng giá + volume cao => tín hiệu mua mạnh (smart money)
        # Ngày giảm giá + volume cao => tín hiệu bán mạnh
        price_chg = close.pct_change()
        avg_vol   = volume.rolling(20).mean()
        vol_ratio = volume / avg_vol

        def _net_flow(chg_series, vr_series):
            """
            Tính net flow score cho 1 cửa sổ thời gian.
            buy_pressure  = trung bình (vol_ratio * sign_tăng)
            sell_pressure = trung bình (vol_ratio * sign_giảm)
            Nhân với avg_trade_val (tỷ VND/ngày) → ra tỷ VND ước tính
            """
            buy  = float(((chg_series > 0) * vr_series.clip(0, 3)).sum())
            sell = float(((chg_series < 0) * vr_series.clip(0, 3)).sum())
            n    = len(chg_series)
            # net = (buy - sell) / n  → tỉ lệ [-1, 1] mỗi phiên
            return round((buy - sell) / n, 4) if n > 0 else 0.0

        # avg_trade_val: giá trị giao dịch TB 1 phiên (tỷ VND)
        avg_trade_val = float((volume * close).tail(20).mean()) / 1e9

        # 3 khung thời gian THỰC SỰ khác nhau
        net_1d  = _net_flow(price_chg.tail(1),  vol_ratio.tail(1))
        net_5d  = _net_flow(price_chg.tail(5),  vol_ratio.tail(5))
        net_20d = _net_flow(price_chg.tail(20), vol_ratio.tail(20))

        # Quy đổi sang tỷ VND — nhân tỉ lệ với avg_trade_val
        # Ý nghĩa: nếu net = 0.5 và avg_trade_val = 100 tỷ/ngày
        # → ước tính ~50 tỷ VND net mua trong phiên
        # Tránh -0.0 khi làm tròn số âm rất nhỏ
        def _r(x): return 0.0 if abs(x) < 0.05 else round(x, 1)

        return {
            "success":       True,
            "net_today":     _r(net_1d  * avg_trade_val),
            "net_5d":        _r(net_5d  * avg_trade_val),
            "net_20d":       _r(net_20d * avg_trade_val),
            "avg_daily_val": round(avg_trade_val, 1),
            "source":        "volume_proxy",
            "note":          "uoc tinh tu volume-price pattern (ty VND/ngay)",
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:120]}


def _parse_close_series(data) -> pd.Series:
    """Parse close price series từ Entrade OHLCV response"""
    if isinstance(data, dict):
        for key in ["c", "close", "Close"]:
            if key in data:
                val = data[key]
                # Lọc None/0/null values
                if isinstance(val, list):
                    clean = [float(x) for x in val if x is not None and x != 0]
                    if clean:
                        return pd.Series(clean, dtype=float)
        # Log keys để debug
        keys_preview = {k: f"list[{len(v)}]" if isinstance(v, list) else str(v)[:20]
                        for k, v in list(data.items())[:8]}
        raise ValueError(f"Khong tim thay close hop le, keys={keys_preview}")
    raise ValueError(f"Response khong phai dict: {type(data)}")


# Cache market data — dùng lại ngoài giờ giao dịch
_market_cache = {"data": None, "date": None}

def get_commodity_data(symbol: str) -> dict:
    """
    Lấy giá hàng hóa liên quan và tính commodity context signal.
    [HKUDS] commodity-analysis/SKILL.md

    Logic: xác định commodity liên quan từ symbol VN, fetch giá qua yfinance/RSS,
    tính composite commodity score ảnh hưởng đến mã phân tích.

    Sector-commodity mapping (TTCK VN):
      Steel (HPG/HSG/NKG): iron ore + coking coal + steel futures
      Energy (GAS/PLX/PVD): crude oil (Brent)
      Consumer/agri (VNM/MSN): soybean + sugar + dairy
      Securities (SSI/VND): VN30 + global risk appetite (gold/VIX)
      Banking (VCB/BID): gold + USD/VND
    """
    _SECTOR_COMMODITY = {
        frozenset(["HPG","HSG","NKG","TLH","VGS","POM","SMC"]): {
            "name": "steel", "tickers": ["^HRC=F"],
            "label": "Steel futures", "direction": 1,   # giá thép tăng → tốt cho HPG
        },
        frozenset(["GAS","PLX","PVD","PVS","BSR","OIL","PVC"]): {
            "name": "energy", "tickers": ["BZ=F","CL=F"],
            "label": "Brent/WTI crude", "direction": 1,
        },
        frozenset(["VNM","MSN","QNS","KDC","SBT"]): {
            "name": "agri", "tickers": ["ZS=F","SB=F"],
            "label": "Soybean/Sugar futures", "direction": -1,  # input cost
        },
        frozenset(["VCB","BID","CTG","TCB","MBB","VPB","ACB"]): {
            "name": "banking_macro", "tickers": ["GC=F","DX-Y.NYB"],
            "label": "Gold/DXY", "direction": 0,  # mixed
        },
    }

    # Xác định commodity liên quan
    commodity_info = None
    for sector_set, info in _SECTOR_COMMODITY.items():
        if symbol in sector_set:
            commodity_info = info
            break

    if not commodity_info:
        return {"success": False, "error": "No commodity mapping for this symbol",
                "signal": 0, "detail": "Khong co hang hoa lien quan"}

    # Fetch giá qua yfinance (nếu có)
    try:
        import yfinance as yf
        prices = {}
        for ticker in commodity_info["tickers"][:1]:  # chỉ lấy 1 ticker chính
            hist = yf.download(ticker, period="3mo", interval="1d",
                               progress=False, auto_adjust=True)
            if hist is not None and not hist.empty and len(hist) >= 5:
                close = hist["Close"].squeeze()
                ret_1m = float((close.iloc[-1] / close.iloc[-20] - 1) * 100) if len(close) >= 20 else 0
                ret_1w = float((close.iloc[-1] / close.iloc[-5]  - 1) * 100) if len(close) >= 5  else 0
                prices[ticker] = {"price": float(close.iloc[-1]),
                                  "ret_1w": ret_1w, "ret_1m": ret_1m}

        if not prices:
            return {"success": False, "error": "yfinance fetch failed",
                    "signal": 0, "detail": commodity_info["label"] + ": data unavailable"}

        # Tính signal dựa trên commodity momentum
        main_ticker = commodity_info["tickers"][0]
        if main_ticker in prices:
            p = prices[main_ticker]
            direction = commodity_info["direction"]
            # 1-month momentum
            if direction == 1:   # input cost hoặc revenue driver positive
                if p["ret_1m"] > 5:   signal = 1
                elif p["ret_1m"] < -5: signal = -1
                else:                  signal = 0
            elif direction == -1:  # input cost (giá cao → margin squeeze)
                if p["ret_1m"] > 5:   signal = -1
                elif p["ret_1m"] < -5: signal = 1
                else:                  signal = 0
            else:
                signal = 0

            label = commodity_info["label"]
            detail = (f"{label}: {p['price']:.2f} | "
                      f"1W={p['ret_1w']:+.1f}% 1M={p['ret_1m']:+.1f}% | "
                      f"Impact={'POSITIVE' if signal>0 else 'NEGATIVE' if signal<0 else 'NEUTRAL'}")
            return {"success": True, "signal": signal, "detail": detail,
                    "sector": commodity_info["name"]}

    except Exception as e:
        pass

    return {"success": False, "error": "yfinance unavailable",
            "signal": 0, "detail": commodity_info["label"] + ": fetch failed (no yfinance)"}


def get_market_data() -> dict:
    """
    Lấy market regime từ VN-Index trực tiếp (SSI/cafef API) hoặc ETF proxy.
    Cache trong ngày — thread-safe.
    """
    global _market_cache
    from datetime import date as _date

    today = _date.today().isoformat()
    with _market_cache_lock:
        if _market_cache["data"] and _market_cache["date"] == today:
            return _market_cache["data"]

    # ── Thử lấy VN-Index trực tiếp từ SSI API ────────────────────────────────
    def _try_vnindex_direct():
        """Lấy VN-Index OHLCV từ SSI public chart API."""
        import time as _time
        to_ts   = int(_time.time())
        from_ts = to_ts - 90 * 86400
        url = (
            f"https://iboard-query.ssi.com.vn/v2/chart/overviewIndex"
            f"?indexId=VN30&fromTime={from_ts}&toTime={to_ts}&resolution=D"
        )
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept":     "application/json",
            "Referer":    "https://iboard.ssi.com.vn/",
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        # SSI trả về {"data": {"t":[..], "c":[..], ...}}
        inner = data.get("data", data)
        times  = inner.get("t", [])
        closes = inner.get("c", [])
        if not times or not closes:
            raise ValueError("SSI API trả về rỗng")
        close = pd.Series([float(x) for x in closes])
        return close, "VN30 (SSI)"

    def _try_vnindex_cafef():
        """Lấy VNINDEX từ cafef chart API."""
        import time as _time
        to_ts   = int(_time.time())
        from_ts = to_ts - 90 * 86400
        url = (
            f"https://s.cafef.vn/Lich-su-giao-dich-VNINDEX-1.chn"
        )
        # cafef không có JSON API dễ dùng — thử entrade với mã VNINDEX
        df = _price_from_entrade("VNINDEX", 90)
        if df is not None and not df.empty:
            close = df["close"].astype(float)
            return close, "VNINDEX (Entrade)"
        raise ValueError("VNINDEX từ Entrade rỗng")

    # ── Fallback: chỉ dùng index/ETF — KHÔNG dùng cổ phiếu đơn lẻ
    # Cổ phiếu đơn lẻ (VHM/VCB/FPT) không đại diện cho thị trường chung
    proxy_symbols = [
        ("VNINDEX",   "VN-Index Entrade"),
        ("VN30",      "VN30 Index"),
        ("E1VFVN30",  "ETF VN30"),
        ("FUEVFVND",  "ETF VNDiamond"),
        ("FUESSVFL",  "ETF SSIAM"),
        ("FUESSV30",  "ETF SSI VN30"),
        ("FUEVN100",  "ETF VN100"),
    ]

    def _compute_result(close: pd.Series, label: str) -> dict:
        """Tính market regime từ close series."""
        if len(close) < 5:
            raise ValueError(f"Khong du data ({len(close)} bars)")
        curr = float(close.iloc[-1])
        p5   = float(close.iloc[-5])  if len(close) >= 5  else curr
        p20  = float(close.iloc[-20]) if len(close) >= 20 else curr
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        return {
            "success":    True,
            "vnindex":    round(curr, 2),
            "change_5d":  round((curr - p5)  / p5  * 100, 2),
            "change_20d": round((curr - p20) / p20 * 100, 2),
            "above_ma20": bool(curr > ma20),
            "above_ma50": bool(curr > ma50) if ma50 else None,
            "ma20":       round(ma20, 2),
            "proxy":      label,
        }

    last_err = "Tat ca sources fail"

    # Thử SSI direct trước
    for fetcher, fname in [(_try_vnindex_direct, "SSI VN30"),
                            (_try_vnindex_cafef,  "Entrade VNINDEX")]:
        try:
            close, label = fetcher()
            result = _compute_result(close, label)
            with _market_cache_lock:
                _market_cache["data"] = result
                _market_cache["date"] = today
            return result
        except Exception as e:
            last_err = f"{fname}: {str(e)[:60]}"
            continue

    # Fallback: vnstock VNINDEX
    def _try_vnindex_vnstock():
        """Lấy VNINDEX từ vnstock KBS/VCI — backup khi Entrade lỗi."""
        from vnstock import Vnstock
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        for src in ["KBS", "VCI", "TCBS"]:
            try:
                stock = Vnstock().stock(symbol="VNINDEX", source=src)
                df = stock.quote.history(start=start, end=end, interval="1D")
                if df is not None and not df.empty:
                    df.columns = [c.lower() for c in df.columns]
                    close = df["close"].astype(float)
                    # vnstock KBS đôi khi trả về giá đã chia 1000
                    # (1.87 thay vì 1870) — chuẩn hoá lại
                    if len(close) > 0 and close.iloc[-1] < 100:
                        close = close * 1000
                    if len(close) >= 5:
                        result = _compute_result(close, f"VNINDEX (vnstock {src})")
                        with _market_cache_lock:
                            _market_cache["data"] = result
                            _market_cache["date"] = today
                        return result
            except Exception:
                continue
        raise ValueError("vnstock VNINDEX fail all sources")

    try:
        return _try_vnindex_vnstock()
    except Exception as e:
        last_err = f"vnstock VNINDEX: {str(e)[:60]}"

    # Fallback: proxy từng mã (chỉ ETF/Index)
    for symbol, label in proxy_symbols:
        try:
            df = _price_from_entrade(symbol, 90)
            if df is None or df.empty:
                last_err = f"{symbol}: empty"; continue
            close = df["close"].astype(float)
            result = _compute_result(close, label)
            with _market_cache_lock:
                _market_cache["data"] = result
                _market_cache["date"] = today
            return result
        except Exception as e:
            last_err = f"{symbol}: {str(e)[:60]}"
            continue

    # Cache cũ
    with _market_cache_lock:
        if _market_cache["data"]:
            cached = dict(_market_cache["data"])
            cached["proxy"] = cached.get("proxy", "") + " [cached]"
            return cached

    return {"success": False, "error": last_err}


# ══════════════════════════════════════════════════════════════════════════════
# VN MACRO CONTEXT — SBV interest rate, USD/VND, PMI VN
# [HKUDS] macro-analysis/SKILL.md + global-macro/SKILL.md
# Framework: GDP/CPI/PMI/FX → macro cycle → asset allocation tilt
# ══════════════════════════════════════════════════════════════════════════════

_macro_cache: dict = {"data": None, "date": None}
_macro_cache_lock = __import__("threading").Lock()

def get_vn_macro_data() -> dict:
    """
    Lấy VN macro indicators: USD/VND, gold price, SBV rate (proxy).
    [HKUDS] macro-analysis/SKILL.md + global-macro/SKILL.md

    VN macro framework (adapted từ HKUDS):
      - USD/VND: > 25,500 = VND depreciation pressure → bearish (capital outflow)
      - Gold (XAU/USD): > 2,500 = safe haven demand → risk-off environment
      - US 10Y yield: > 4.5% = USD strength → VN capital outflow pressure
      - VN Manufacturing PMI: > 50 = expansion (từ S&P Global monthly)

    Signal logic (macro-analysis SKILL):
      bull_macro: VND stable + global risk-on + gold neutral
      bear_macro: VND depreciating + risk-off + high US yields
    """
    global _macro_cache

    today = __import__("datetime").date.today().isoformat()
    with _macro_cache_lock:
        if _macro_cache["data"] and _macro_cache["date"] == today:
            return _macro_cache["data"]

    result = {"success": False, "signal": 0, "detail": "VN macro: N/A"}
    score = 0
    parts = []

    # ── USD/VND từ yfinance ────────────────────────────────────────────────
    try:
        import yfinance as yf
        fx = yf.download("USDVND=X", period="1mo", interval="1d",
                         progress=False, auto_adjust=True)
        if fx is not None and not fx.empty:
            close = fx["Close"].squeeze()
            usdvnd = float(close.iloc[-1])
            chg_1m = float((close.iloc[-1] / close.iloc[0] - 1) * 100) if len(close) > 5 else 0
            # USD/VND > 25,500 = VND yếu, > 26,000 = áp lực mạnh
            if usdvnd > 26000 or chg_1m > 2:
                score -= 1
                parts.append(f"USD/VND={usdvnd:.0f}(+{chg_1m:.1f}%,VND_yeu)")
            elif usdvnd < 24500 or chg_1m < -1:
                score += 1
                parts.append(f"USD/VND={usdvnd:.0f}({chg_1m:.1f}%,VND_manh)")
            else:
                parts.append(f"USD/VND={usdvnd:.0f}(on)")
    except Exception:
        pass

    # ── Gold XAU/USD ─────────────────────────────────────────────────────────
    try:
        import yfinance as yf
        gold = yf.download("GC=F", period="1mo", interval="1d",
                           progress=False, auto_adjust=True)
        if gold is not None and not gold.empty:
            g_close = gold["Close"].squeeze()
            g_val   = float(g_close.iloc[-1])
            g_chg1w = float((g_close.iloc[-1] / g_close.iloc[-5] - 1) * 100) if len(g_close) >= 5 else 0
            # Gold > 2500 = risk-off → bearish for equity
            if g_val > 2800 and g_chg1w > 2:
                score -= 1
                parts.append(f"Gold={g_val:.0f}(risk-off)")
            elif g_val < 2000 and g_chg1w < -2:
                score += 1
                parts.append(f"Gold={g_val:.0f}(risk-on)")
            else:
                parts.append(f"Gold={g_val:.0f}(neutral)")
    except Exception:
        pass

    # ── US 10Y yield (tác động vốn ngoại vào VN) ─────────────────────────────
    try:
        import yfinance as yf
        us10y = yf.download("^TNX", period="1mo", interval="1d",
                            progress=False, auto_adjust=True)
        if us10y is not None and not us10y.empty:
            y_close = us10y["Close"].squeeze()
            y_val   = float(y_close.iloc[-1])
            # US 10Y > 4.5% = USD strong = vốn ngoại rút khỏi EM/VN
            if y_val > 5.0:
                score -= 2
                parts.append(f"US10Y={y_val:.2f}%(qua_cao,outflow)")
            elif y_val > 4.5:
                score -= 1
                parts.append(f"US10Y={y_val:.2f}%(cao,pressure)")
            elif y_val < 3.5:
                score += 1
                parts.append(f"US10Y={y_val:.2f}%(thap,inflow)")
            else:
                parts.append(f"US10Y={y_val:.2f}%(neutral)")
    except Exception:
        pass

    # Tổng hợp macro signal
    if score >= 1:
        macro_sig = 1
        macro_label = "MACRO_BULLISH"
    elif score <= -1:
        macro_sig = -1
        macro_label = "MACRO_BEARISH"
    else:
        macro_sig = 0
        macro_label = "MACRO_NEUTRAL"

    result = {
        "success": len(parts) > 0,
        "signal":  macro_sig,
        "score":   score,
        "detail":  f"[macro-analysis+global-macro] " + " | ".join(parts) + f" | {macro_label}",
        "label":   macro_label,
    }

    with _macro_cache_lock:
        _macro_cache["data"] = result
        _macro_cache["date"] = today

    return result


# ══════════════════════════════════════════════════════════════════════════════
# NEWS — 12 nguồn song song
# ══════════════════════════════════════════════════════════════════════════════

def get_news_data(symbol: str) -> dict:
    """
    Thu thập tin tức từ RSS feeds (không bị block từ datacenter IP).

    Nguồn RSS hoạt động:
    - Google News RSS (tổng hợp từ tất cả báo VN)
    - CafeF RSS (tin tài chính chuyên sâu)
    - VnExpress RSS kinh doanh
    - VnEconomy RSS chứng khoán
    - Fireant API JSON (mạng trader VN)
    - ThanhNien RSS kinh tế
    - NDH.vn RSS chứng khoán (Nhịp cầu đầu tư)
    - Dantri RSS kinh doanh

    Nguồn không dùng RSS (bị block HTML scraping từ datacenter):
    - f319: không có RSS -> bù lại bằng Google News tìm "f319 VCB"
    - VCSC/HSC/ACBS/VNDIRECT/SSI: không có RSS -> Google News tìm báo cáo CTCK
    """
    import re
    import urllib.parse

    RSS_HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; Feedfetcher-Google; +http://www.google.com/feedfetcher.html)",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    # Các title vô nghĩa cần lọc
    _JUNK_TITLES = {
        "google tin tuc", "google news", "tin tuc", "vnexpress",
        "cafef", "vneconomy", "ndh.vn", "dantri", "thanh nien",
        "tin nhanh chung khoan", "tnck", "rss", "feed",
    }

    def parse_rss(url: str, prefix: str, max_items: int = 6) -> list:
        try:
            r = requests.get(url, headers=RSS_HEADERS, timeout=8)
            if r.status_code != 200:
                return []
            text = r.text
            # CDATA titles (dạng phổ biến nhất)
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", text, re.DOTALL)
            if not titles:
                # Plain titles
                titles = re.findall(r"<title>(.*?)</title>", text, re.DOTALL)
            # Clean HTML tags
            titles = [re.sub(r"<[^>]+>", "", t).strip() for t in titles]
            # Lọc: bỏ title đầu (tên feed), bỏ title quá ngắn, bỏ junk
            results = []
            for t in titles:
                if len(t) < 15:
                    continue
                if t.lower() in _JUNK_TITLES:
                    continue
                if any(j in t.lower() for j in ["google tin tuc", "google news"]):
                    continue
                results.append(f"[{prefix}] {t}")
                if len(results) >= max_items:
                    break
            return results
        except Exception:
            return []

    def fetch_google_news():
        """Google News RSS — queries đa dạng bao gồm tin CTCK và f319"""
        results = []
        queries = [
            f"{symbol} co phieu chung khoan",
            f"{symbol} ket qua kinh doanh loi nhuan",
            f"{symbol} phan tich khuyen nghi VCSC HSC SSI",
            f"{symbol} f319 dien dan",
        ]
        for q in queries:
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=vi&gl=VN&ceid=VN:vi"
            items = parse_rss(url, "GoogleNews", 3)
            results.extend(items)
        # Dedup
        seen, deduped = set(), []
        for item in results:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped[:12]

    def fetch_cafef_rss():
        """CafeF RSS feed"""
        results = parse_rss(
            f"https://cafef.vn/rss/{symbol.lower()}.rss",
            "CafeF", 5
        )
        if not results:
            results = parse_rss("https://cafef.vn/rss/thi-truong-chung-khoan.rss", "CafeF", 6)
            results = [r for r in results if symbol in r.upper()][:4]
        return results

    def fetch_vnexpress_rss():
        results = parse_rss("https://vnexpress.net/rss/kinh-doanh.rss", "VnExpress", 20)
        relevant = [t for t in results if symbol in t.upper()]
        return relevant[:4] if relevant else results[:2]

    def fetch_vneconomy_rss():
        results = parse_rss("https://vneconomy.vn/chung-khoan.rss", "VnEconomy", 20)
        relevant = [t for t in results if symbol in t.upper()]
        return relevant[:4] if relevant else results[:2]

    def fetch_ndh_rss():
        """NDH - Nhịp cầu đầu tư RSS"""
        results = parse_rss("https://ndh.vn/rss/chung-khoan.rss", "NDH", 10)
        relevant = [t for t in results if symbol in t.upper()]
        return relevant[:4] if relevant else results[:2]

    def fetch_dantri_rss():
        results = parse_rss("https://dantri.com.vn/kinh-doanh.rss", "DanTri", 20)
        relevant = [t for t in results if symbol in t.upper()]
        return relevant[:3]

    def fetch_thanhnien_rss():
        results = parse_rss("https://thanhnien.vn/rss/tai-chinh-kinh-doanh.rss", "ThanhNien", 20)
        relevant = [t for t in results if symbol in t.upper()]
        return relevant[:3]

    def fetch_tinnhanhchungkhoan_rss():
        """Tin nhanh chứng khoán RSS"""
        results = parse_rss("https://tinnhanhchungkhoan.vn/rss/chung-khoan.rss", "TNCK", 15)
        relevant = [t for t in results if symbol in t.upper()]
        return relevant[:4] if relevant else results[:2]

    def fetch_fireant_api():
        """
        Fireant JSON API - mạng trader VN.
        Cần FIREANT_TOKEN (Bearer) — đăng ký free tại fireant.vn.
        Nếu không có token → fallback scrape web Fireant.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0",
                "Accept": "application/json",
            }
            if FIREANT_TOKEN:
                headers["Authorization"] = f"Bearer {FIREANT_TOKEN}"

            r = requests.get(
                f"https://restv2.fireant.vn/posts?symbol={symbol}&limit=10&offset=0",
                headers=headers,
                timeout=8,
            )
            if r.status_code == 200:
                posts = r.json()
                if not isinstance(posts, list):
                    posts = posts.get("items", posts.get("data", []))
                results = []
                for p in posts[:8]:
                    content = re.sub(r"<[^>]+>", "", p.get("content", "") or "").strip()
                    content = " ".join(content.split())
                    if content and len(content) > 15:
                        results.append(f"[Fireant] {content[:180]}")
                return results
            elif r.status_code == 401:
                # Token sai hoặc hết hạn → fallback scrape web
                return _fetch_fireant_web(symbol)
        except Exception:
            pass
        return []

    def _fetch_fireant_web(sym):
        """Fallback: scrape trang web Fireant khi không có API token."""
        try:
            r = requests.get(
                f"https://fireant.vn/symbol/{sym}",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0",
                    "Accept": "text/html,application/xhtml+xml,*/*",
                    "Accept-Language": "vi-VN,vi;q=0.9",
                },
                timeout=10,
            )
            if r.status_code != 200:
                return []
            # Parse các đoạn text từ post/comment trên trang
            # Fireant render server-side một phần, tìm các thẻ chứa nội dung bài viết
            snippets = re.findall(
                r'(?:data-content|"content"|"body")["\s:>]+([^<"]{30,300})',
                r.text
            )
            results = []
            seen_fw = set()
            for s in snippets[:15]:
                s = s.strip().replace("\\n", " ").replace("\\t", " ")
                s = " ".join(s.split())
                if len(s) > 30 and s not in seen_fw:
                    seen_fw.add(s)
                    results.append(f"[Fireant] {s[:180]}")
                if len(results) >= 5:
                    break
            return results
        except Exception:
            return []

    def fetch_f319_html():
        """
        Scrape f319.com — diễn đàn trader lớn nhất VN.
        Render test cho thấy f319 trả 200 OK (không bị Cloudflare block).
        Parse các thread/post liên quan đến mã cổ phiếu.
        """
        results = []
        try:
            # Trang tìm kiếm theo mã
            r = requests.get(
                f"https://f319.com/search?q={symbol}&t=post",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0",
                    "Accept": "text/html,application/xhtml+xml,*/*",
                    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
                    "Referer": "https://f319.com/",
                },
                timeout=10,
            )
            if r.status_code != 200:
                return []

            html = r.text
            # Parse tiêu đề thread từ search results
            # f319 dùng class "title" hoặc thẻ <h3>/<h2> cho thread title
            titles = re.findall(
                r'<(?:h[23]|a)[^>]+class="[^"]*(?:title|thread-title|subject)[^"]*"[^>]*>(.*?)</(?:h[23]|a)>',
                html, re.IGNORECASE | re.DOTALL
            )
            # Fallback: lấy link text chứa mã symbol
            if not titles:
                all_links = re.findall(r'<a[^>]+href="/threads/[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
                titles = [t for t in all_links if symbol.upper() in t.upper()]

            seen_f = set()
            for t in titles[:10]:
                t_clean = re.sub(r"<[^>]+>", "", t).strip()
                t_clean = " ".join(t_clean.split())
                if len(t_clean) > 15 and t_clean not in seen_f:
                    seen_f.add(t_clean)
                    results.append(f"[f319] {t_clean[:180]}")
                if len(results) >= 5:
                    break

        except Exception:
            pass

        # Nếu search không có kết quả → thử trang chuyên mục mã
        if not results:
            try:
                r2 = requests.get(
                    f"https://f319.com/forums/{symbol.lower()}/",
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0"},
                    timeout=8,
                )
                if r2.status_code == 200:
                    thread_titles = re.findall(
                        r'<(?:h[23]|span)[^>]+class="[^"]*title[^"]*"[^>]*>(.*?)</(?:h[23]|span)>',
                        r2.text, re.IGNORECASE | re.DOTALL
                    )
                    for t in thread_titles[:8]:
                        t_clean = re.sub(r"<[^>]+>", "", t).strip()
                        t_clean = " ".join(t_clean.split())
                        if len(t_clean) > 15:
                            results.append(f"[f319] {t_clean[:180]}")
                        if len(results) >= 5:
                            break
            except Exception:
                pass

        return results

    # Chạy song song tất cả sources — thêm f319
    sources = [
        fetch_google_news,
        fetch_cafef_rss,
        fetch_vnexpress_rss,
        fetch_vneconomy_rss,
        fetch_ndh_rss,
        fetch_dantri_rss,
        fetch_thanhnien_rss,
        fetch_tinnhanhchungkhoan_rss,
        fetch_fireant_api,
        fetch_f319_html,
    ]
    source_names = [
        "GoogleNews", "CafeF", "VnExpress", "VnEconomy",
        "NDH", "DanTri", "ThanhNien", "TNCK", "Fireant", "f319",
    ]

    all_headlines, source_results = [], {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fn): name for fn, name in zip(sources, source_names)}
        for future, name in futures.items():
            try:
                items = future.result()
                source_results[name] = len(items)
                all_headlines.extend(items)
            except Exception:
                source_results[name] = 0

    # Dedup + clean
    seen, clean_headlines = set(), []
    for h in all_headlines:
        h = h.strip()
        if h and h not in seen and len(h) > 15:
            seen.add(h)
            clean_headlines.append(h)

    if not clean_headlines:
        return {"success": False, "error": "Không lấy được tin tức (tất cả RSS fail)"}

    return {
        "success":        True,
        "headlines":      clean_headlines[:20],
        "source_summary": source_results,
        "total":          len(clean_headlines),
    }


# ══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> dict:
    close  = df["close"].astype(float)
    volume = df["volume"].astype(float)

    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi   = float((100 - 100 / (1 + gain / loss)).iloc[-1])

    ema12     = close.ewm(span=12).mean()
    ema26     = close.ewm(span=26).mean()
    macd      = ema12 - ema26
    sig_line  = macd.ewm(span=9).mean()
    macd_hist = float((macd - sig_line).iloc[-1])

    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_upper = float((sma20 + 2 * std20).iloc[-1])
    bb_lower = float((sma20 - 2 * std20).iloc[-1])
    bb_mid   = float(sma20.iloc[-1])
    ma20     = float(sma20.iloc[-1])
    ma50     = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None

    avg_vol   = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

    curr = float(close.iloc[-1])
    p1w  = float(close.iloc[-5])  if len(close) >= 5  else curr
    p1m  = float(close.iloc[-20]) if len(close) >= 20 else curr

    return {
        "current_price":  round(curr, 2),
        "change_1w_pct":  round((curr - p1w) / p1w * 100, 2),
        "change_1m_pct":  round((curr - p1m) / p1m * 100, 2),
        "rsi":            round(rsi, 1),
        "macd_hist":      round(macd_hist, 4),
        "macd":           round(float(macd.iloc[-1]), 4),
        "macd_signal":    round(float(sig_line.iloc[-1]), 4),
        "bb_upper":       round(bb_upper, 2),
        "bb_lower":       round(bb_lower, 2),
        "bb_mid":         round(bb_mid, 2),
        "ma20":           round(ma20, 2),
        "ma50":           round(ma50, 2) if ma50 else None,
        "volume_ratio":   round(vol_ratio, 2),
        "resistance_20d": round(float(df["high"].astype(float).tail(20).max()), 2),
        "support_20d":    round(float(df["low"].astype(float).tail(20).min()), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEEPSEEK HELPER
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def call_gemini(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    """Fallback khi DeepSeek hết credit — Gemini 2.0 Flash free tier"""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY chưa được set")
    resp = requests.post(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    """Gọi LLM — ưu tiên DeepSeek, fallback Gemini nếu DeepSeek 402/429"""
    # Thử DeepSeek trước
    if DEEPSEEK_API_KEY:
        try:
            resp = requests.post(
                DEEPSEEK_URL,
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                         "Content-Type": "application/json"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            if resp.status_code == 402:
                raise ValueError("DeepSeek hết credit (402) — chuyển sang Gemini")
            if resp.status_code == 429:
                raise ValueError("DeepSeek rate limit (429) — chuyển sang Gemini")
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except ValueError as ve:
            if "402" in str(ve) or "429" in str(ve) or "Gemini" in str(ve):
                pass  # fallback
            else:
                raise

    # Fallback Gemini
    return call_gemini(system_prompt, user_prompt, max_tokens)


# Alias để không cần đổi tên ở 6 agents
def call_deepseek(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    return call_llm(system_prompt, user_prompt, max_tokens)


# ══════════════════════════════════════════════════════════════════════════════
# VIBE-TRADING AGENTS (HKUDS) — 7 engines thuần pandas
# Source: github.com/HKUDS/Vibe-Trading
# File:   vibe_skills.py (deploy cùng project)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from vibe_skills import run_vibe_agents as _run_vibe_agents
    _VIBE_AVAILABLE = True
except ImportError:
    _VIBE_AVAILABLE = False

# ── Context agents: Market Regime + News + Macro (dùng network data) ─────────

def run_trend_agent(symbol, ind):
    """
    Xu hướng — rule-based thuần túy từ indicators.
    Logic:
      TANG   : giá > MA20 VÀ (RSI > 55 HOẶC MACD hist > 0) VÀ 1W > 0
      GIAM   : giá < MA20 VÀ (RSI < 45 HOẶC MACD hist < 0) VÀ 1W < 0
      SIDEWAY: còn lại
    """
    price    = ind["current_price"]
    ma20     = ind["ma20"]
    ma50     = ind.get("ma50")
    rsi      = ind["rsi"]
    macd_h   = ind["macd_hist"]
    chg_1w   = ind["change_1w_pct"]
    chg_1m   = ind["change_1m_pct"]

    above_ma20 = price > ma20
    above_ma50 = (price > ma50) if ma50 else None

    bull_signals = sum([
        above_ma20,
        rsi > 55,
        macd_h > 0,
        chg_1w > 0,
        chg_1m > 2,
        above_ma50 == True,
    ])
    bear_signals = sum([
        not above_ma20,
        rsi < 45,
        macd_h < 0,
        chg_1w < 0,
        chg_1m < -2,
        above_ma50 == False,
    ])

    if bull_signals >= 4:
        v = "TANG"
        reason = f"Gia {'tren' if above_ma20 else 'duoi'} MA20={_fmt_price(ma20)}, RSI={rsi}, MACD hist={macd_h:+.4f}, 1W={chg_1w:+.1f}%"
    elif bear_signals >= 4:
        v = "GIAM"
        reason = f"Gia {'tren' if above_ma20 else 'duoi'} MA20={_fmt_price(ma20)}, RSI={rsi}, MACD hist={macd_h:+.4f}, 1W={chg_1w:+.1f}%"
    else:
        v = "SIDEWAY"
        reason = f"Tin hieu lan lon: bull={bull_signals}/6, bear={bear_signals}/6. MA20={_fmt_price(ma20)}, RSI={rsi}"

    txt = f"[Rule] {reason}"
    return txt, v


def run_volume_agent(symbol, ind):
    """
    Volume — rule-based.
    XAC NHAN : vol_ratio >= 1.2 (volume tăng xác nhận xu hướng)
    NGHI NGO : vol_ratio 0.5–1.2 (volume bình thường)
    PHAN KY  : vol_ratio < 0.5 (volume quá thấp, thiếu xác nhận)
    Thêm: nếu giá tăng mà volume giảm → PHAN KY
    """
    vol   = ind["volume_ratio"]
    chg   = ind["change_1w_pct"]
    bb_pos = (ind["current_price"] / ind["bb_mid"] - 1) * 100 if ind["bb_mid"] > 0 else 0

    # Phân kỳ giá-volume: giá tăng mạnh nhưng vol thấp
    price_vol_divergence = (chg > 3 and vol < 0.7) or (chg < -3 and vol < 0.7)

    if price_vol_divergence:
        v = "PHAN KY"
        reason = f"Phan ky: gia {chg:+.1f}% nhung vol chi {vol}x TB20"
    elif vol >= 1.5:
        v = "XAC NHAN"
        reason = f"Vol manh {vol}x TB20, xac nhan dong luc"
    elif vol >= 1.0:
        v = "XAC NHAN"
        reason = f"Vol binh thuong {vol}x TB20"
    elif vol >= 0.5:
        v = "NGHI NGO"
        reason = f"Vol thap {vol}x TB20, thieu xac nhan"
    else:
        v = "PHAN KY"
        reason = f"Vol rat thap {vol}x TB20, khong du luc"

    txt = f"[Rule] {reason} | BB pos: {bb_pos:+.1f}%"
    return txt, v


def run_risk_agent(symbol, ind):
    """
    Rủi ro — rule-based từ vị trí giá trong BB và khoảng cách support.
    THAP     : giá gần support, RSI < 60, còn room lên
    CAO      : giá gần resistance hoặc BB upper, RSI > 70
    TRUNG BINH: còn lại
    """
    price  = ind["current_price"]
    bb_up  = ind["bb_upper"]
    bb_low = ind["bb_lower"]
    bb_mid = ind["bb_mid"]
    rsi    = ind["rsi"]
    sup    = ind["support_20d"]
    res    = ind["resistance_20d"]

    bb_range  = bb_up - bb_low if bb_up > bb_low else 1
    bb_pct    = (price - bb_low) / bb_range * 100  # 0% = lower, 100% = upper

    dist_to_sup = (price - sup) / price * 100 if price > 0 else 0
    dist_to_res = (res - price) / price * 100 if price > 0 else 0

    risk_signals = sum([
        bb_pct > 80,          # Gần BB upper
        rsi > 70,             # Overbought
        dist_to_res < 2,      # Gần kháng cự
        dist_to_sup > 15,     # SL sẽ rất rộng
    ])
    safe_signals = sum([
        bb_pct < 30,          # Gần BB lower
        rsi < 50,             # Chưa overbought
        dist_to_sup < 5,      # Gần support (SL gần)
        dist_to_res > 8,      # Còn nhiều room lên
    ])

    if risk_signals >= 2:
        v = "CAO"
    elif safe_signals >= 3:
        v = "THAP"
    else:
        v = "TRUNG BINH"

    reason = (f"BB pos={bb_pct:.0f}%, RSI={rsi}, "
              f"cach support={dist_to_sup:.1f}%, cach khang cu={dist_to_res:.1f}%")
    txt = f"[Rule] {reason}"
    return txt, v


def run_news_agent(symbol, news):
    """
    News sentiment — đếm keyword tích cực/tiêu cực trong headlines.
    Không dùng LLM để tránh hallucinate.
    """
    if not news.get("success") or not news.get("headlines"):
        return "Khong co du lieu tin tuc", "TRUNG TINH"

    headlines = news["headlines"]
    total     = len(headlines)
    text_all  = " ".join(headlines).lower()

    # Keyword tích cực
    pos_kw = [
        "tang", "tăng", "loi nhuan", "lợi nhuận", "tang truong", "tăng trưởng",
        "ket qua tot", "kết quả tốt", "vuot ke hoach", "vượt kế hoạch",
        "mua vao", "mua vào", "khuyen nghi mua", "khuyến nghị mua",
        "tich cuc", "tích cực", "kha quan", "khả quan", "dot pha", "đột phá",
        "co tuc", "cổ tức", "chia co tuc", "chia cổ tức", "pha hanh", "phát hành",
        "hop dong lon", "hợp đồng lớn", "đơn hàng", "don hang",
    ]
    # Keyword tiêu cực
    neg_kw = [
        "giam", "giảm", "lo", "lỗ", "thua lo", "thua lỗ", "sut giam", "sụt giảm",
        "ban ra", "bán ra", "khuyen nghi ban", "khuyến nghị bán",
        "tieu cuc", "tiêu cực", "rui ro", "rủi ro", "canh bao", "cảnh báo",
        "thanh tra", "điều tra", "dieu tra", "vi pham", "vi phạm",
        "no xau", "nợ xấu", "mat thanh khoan", "mất thanh khoản",
        "thu hep", "thu nhep", "cat giam", "cắt giảm", "sa thai", "sa thải",
    ]

    pos_count = sum(1 for kw in pos_kw if kw in text_all)
    neg_count = sum(1 for kw in neg_kw if kw in text_all)

    # Tỷ lệ
    if pos_count == 0 and neg_count == 0:
        v = "TRUNG TINH"
        reason = f"Khong tim thay tu khoa ro rang ({total} headline)"
    elif pos_count > neg_count * 1.5:
        v = "TICH CUC"
        reason = f"Tu khoa tich cuc: {pos_count}, tieu cuc: {neg_count} ({total} headline)"
    elif neg_count > pos_count * 1.5:
        v = "TIEU CUC"
        reason = f"Tu khoa tieu cuc: {neg_count}, tich cuc: {pos_count} ({total} headline)"
    else:
        v = "TRUNG TINH"
        reason = f"Can bang: tich cuc={pos_count}, tieu cuc={neg_count} ({total} headline)"

    # Lấy vài headline tiêu biểu
    sample = " | ".join(h[:60] for h in headlines[:2])
    txt = f"[Rule] {reason}\nMau: {sample}"
    return txt, v


def run_market_regime_agent(market):
    """
    Market regime — rule-based từ proxy ETF/index.
    Không dùng LLM.
    """
    if not market.get("success"):
        err = market.get('error', '')
        return f"Khong lay duoc data thi truong: {err}", "UNKNOWN"

    chg_5d  = market["change_5d"]
    chg_20d = market["change_20d"]
    above_ma20 = market["above_ma20"]
    above_ma50 = market.get("above_ma50")

    bull = sum([
        chg_5d > 1,
        chg_20d > 3,
        above_ma20 == True,
        above_ma50 == True,
    ])
    bear = sum([
        chg_5d < -1,
        chg_20d < -3,
        above_ma20 == False,
        above_ma50 == False,
    ])

    if bull >= 3:
        v = "UPTREND"
    elif bear >= 3:
        v = "DOWNTREND"
    else:
        v = "SIDEWAYS"

    reason = (f"VN-Index proxy {chg_5d:+.1f}% (5D) / {chg_20d:+.1f}% (20D), "
              f"{'tren' if above_ma20 else 'duoi'} MA20"
              f"{', tren MA50' if above_ma50 else ', duoi MA50' if above_ma50 is False else ''}")
    txt = f"[Rule] {reason}"
    return txt, v


def run_macro_agent(news: dict) -> tuple:
    """
    Macro — rule-based, đếm keyword vĩ mô trong RSS headlines.
    Không dùng LLM.
    """
    import urllib.parse, re as _re

    RSS_HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; Feedfetcher-Google)",
        "Accept": "application/rss+xml, */*",
    }

    def _parse_rss_titles(url):
        try:
            r = requests.get(url, headers=RSS_HEADERS, timeout=8)
            if r.status_code != 200:
                return []
            text   = r.text
            titles = _re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", text, _re.DOTALL)
            if not titles:
                titles = _re.findall(r"<title>(.*?)</title>", text, _re.DOTALL)
            titles = [_re.sub(r"<[^>]+>", "", t).strip() for t in titles if len(t.strip()) > 15]
            return titles[1:8]
        except Exception:
            return []

    # Lấy headlines macro
    macro_headlines = _extract_macro_headlines(news.get("headlines", [])) if news.get("success") else []
    if len(macro_headlines) < 3:
        extra = []
        for url in [
            "https://vneconomy.vn/tai-chinh.rss",
            "https://vneconomy.vn/ngan-hang.rss",
            "https://cafef.vn/rss/vi-mo-dau-tu.rss",
        ]:
            titles = _parse_rss_titles(url)
            extra.extend([f"[Macro] {t}" for t in titles])
            if len(macro_headlines) + len(extra) >= 5:
                break
        macro_headlines = macro_headlines + extra

    if not macro_headlines:
        return "Khong co tin tuc macro", "TRUNG TINH"

    text_all = " ".join(macro_headlines).lower()

    # Keywords thuận lợi
    pos_macro = [
        "ha lai suat", "hạ lãi suất", "cat giam lai suat", "cắt giảm lãi suất",
        "no long", "nới lỏng", "bom tien", "bơm tiền", "tang cung tien",
        "tang truong gdp", "tăng trưởng gdp", "phuc hoi kinh te", "phục hồi kinh tế",
        "on dinh ty gia", "ổn định tỷ giá", "fed giu nguyen", "fed giữ nguyên",
        "room tin dung tang", "room tín dụng tăng",
    ]
    # Keywords rủi ro
    neg_macro = [
        "tang lai suat", "tăng lãi suất", "that chat", "thắt chặt",
        "lam phat tang", "lạm phát tăng", "ty gia tang", "tỷ giá tăng",
        "rui ro", "rủi ro", "suy thoai", "suy thoái", "khung hoang", "khủng hoảng",
        "fed tang lai", "fed tăng lãi", "ndt mat gia", "ndt mất giá",
        "room tin dung giam", "siết tín dụng",
    ]

    pos_count = sum(1 for kw in pos_macro if kw in text_all)
    neg_count = sum(1 for kw in neg_macro if kw in text_all)

    if pos_count > neg_count:
        v = "THUAN LOI"
    elif neg_count > pos_count:
        v = "RUI RO"
    else:
        v = "TRUNG TINH"

    sample = " | ".join(h[:50] for h in macro_headlines[:2])
    txt = f"[Rule] Macro: thuan loi={pos_count}, rui ro={neg_count} | {sample}"
    return txt, v




# ── _MACRO_KEYWORDS dùng cho run_macro_agent rule-based ──────────────────────
_MACRO_KEYWORDS = [
    "lai suat", "lãi suất", "ls dieu hanh", "lai suat tiet kiem",
    "lai suat cho vay", "ngan hang nha nuoc", "sbv", "nhnn",
    "ty gia", "tỷ giá", "usd", "vnd", "usd/vnd", "gia dollar", "gia usd",
    "fed", "federal reserve", "lai suat my", "powell", "fomc",
    "tin dung", "tín dụng", "room tin dung", "cung tien", "bom tien",
    "lam phat", "lạm phát", "cpi", "gdp", "tang truong kinh te",
    "trai phieu", "trái phiếu", "trai phieu chinh phu",
    "von hoa ngan hang", "an toan von", "basel",
]

def _extract_macro_headlines(headlines: list) -> list:
    macro = []
    for h in headlines:
        if any(kw in h.lower() for kw in _MACRO_KEYWORDS):
            macro.append(h)
    return macro




def _clean_agent_txt(txt: str) -> str:
    """Xoa dong Ket luan / — Ket luan khoi cuoi text agent."""
    lines = txt.splitlines()
    cleaned = []
    for line in lines:
        lo = line.lower().strip()
        if ("ket luan:" in lo or "k\u1ebft lu\u1eadn:" in lo
                or lo.startswith("\u2014 ket") or lo.startswith("- ket")):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    result = result.replace("[du lieu uoc tinh]", "").strip()
    return result


# Lớp 3: Budget ký tự tối đa mỗi agent trong phần CHI TIET
_AGENT_CHAR_BUDGET = 300


def _clip_agent_txt(txt: str, budget: int = _AGENT_CHAR_BUDGET) -> str:
    """
    Lớp 3: Clip text agent theo budget ký tự.
    Cắt tại ranh giới câu/dòng, không cắt giữa chừng.
    """
    if len(txt) <= budget:
        return txt
    # Thử cắt tại dấu chấm cuối câu gần nhất trước budget
    truncated = txt[:budget]
    last_dot = max(truncated.rfind(". "), truncated.rfind(".\n"))
    if last_dot > budget * 0.5:  # có câu hoàn chỉnh chiếm > 50% budget
        return txt[:last_dot + 1].strip()
    # Fallback: cắt tại dòng cuối hoàn chỉnh
    last_newline = truncated.rfind("\n")
    if last_newline > budget * 0.3:
        return txt[:last_newline].strip()
    # Last resort: cắt cứng nhưng thêm dấu hiệu
    return truncated.rstrip() + "..."


def _fmt_agent(label: str, verdict: str, txt: str) -> str:
    """Format 1 agent: [Label -> VERDICT] tren header, noi dung ben duoi.
    Ap dung Lop 3 clip truoc khi format."""
    clean    = _clean_agent_txt(txt)
    clipped  = _clip_agent_txt(clean)
    lines    = clipped.splitlines()
    if not lines:
        return f"[{label} -> {verdict}] (khong co noi dung)"
    first = lines[0].strip()
    rest  = "\n  ".join(l.strip() for l in lines[1:] if l.strip())
    body  = (first + "\n  " + rest) if rest else first
    return f"[{label} -> {verdict}]\n  {body}"


def _build_message_safe(
    emoji, symbol, now, ind, vnindex_str,
    trend_v, volume_v, risk_v, fund_v, smart_v, news_v, market_v, macro_v,
    trend_txt, volume_txt, risk_txt, fund_txt, smart_txt, news_txt, market_txt, macro_txt,
    final_v, conf, bull, bear, n_agents, summary, ap, negative,
    char_limit: int = 4000
) -> str:
    """
    Lớp 2: Smart truncate — đảm bảo KET LUAN + ACTION PLAN luôn nguyên vẹn.
    Thứ tự ưu tiên (quan trọng nhất → ít quan trọng nhất):
      1. Header (DU LIEU + 8 AGENTS summary)   — không bao giờ cắt
      2. KET LUAN + ACTION PLAN + LUU Y RUI RO — không bao giờ cắt
      3. CHI TIET AGENTS                        — cắt bớt nếu cần

    Chiến lược:
      - Build phần bắt buộc (header + footer) trước → tính remaining budget
      - Với remaining budget, fit càng nhiều chi tiết agent càng tốt
      - Nếu không đủ chỗ cho 1 agent, hiển thị "(Da rut gon)" thay vi cat giua
    """

    # ── Phần bắt buộc: HEADER ────────────────────────────────────────────────
    header = (
        f"{emoji} Phan tich {symbol} — {now}\n\n"
        f"DU LIEU:\n"
        f"  Gia: {_fmt_price(ind['current_price'])} | 1W: {ind['change_1w_pct']:+.1f}% | 1M: {ind['change_1m_pct']:+.1f}%\n"
        f"  RSI: {ind['rsi']} | MACD Hist: {ind['macd_hist']:+.4f} | Vol: {ind['volume_ratio']}x\n"
        f"  Ho tro: {_fmt_price(ind['support_20d'])} | Khang cu: {_fmt_price(ind['resistance_20d'])}\n"
        f"  VN-Index: {vnindex_str}\n\n"
        f"8 AGENTS:\n"
        f"  Xu huong:    {trend_v}\n"
        f"  Volume:      {volume_v}\n"
        f"  Rui ro:      {risk_v}\n"
        f"  Fundamental: {fund_v}\n"
        f"  Smart Money: {smart_v}\n"
        f"  News:        {news_v}\n"
        f"  Market:      {market_v}\n"
        f"  Macro:       {macro_v}\n"
    )

    # ── Phần bắt buộc: FOOTER (KET LUAN + ACTION PLAN + LUU Y) ──────────────
    footer = (
        f"\n{'='*30}\n"
        f"{emoji} KET LUAN TONG HOP: {final_v}\n"
        f"   Do tin cay: {conf}/10 | Dong thuan: {bull}/{n_agents} bullish, {bear}/{n_agents} bearish\n\n"
        f"LY DO CHINH: {summary}\n\n"
        f"{_format_action_plan(final_v, ap)}\n\n"
        f"LUU Y RUI RO: {negative}\n"
        f"{'='*30}"
    )

    mandatory_len = len(header) + len(footer)
    if mandatory_len >= char_limit:
        # Header + footer đã vượt limit (rất hiếm) → trả luôn, cắt header
        return (header[:char_limit - len(footer) - 20] + "\n[Rut gon]\n" + footer)[:char_limit]

    # ── Budget còn lại cho CHI TIET AGENTS ───────────────────────────────────
    detail_budget = char_limit - mandatory_len - 30  # 30 chars đệm an toàn

    agents = [
        ("Xu huong",    trend_v,  trend_txt),
        ("Volume",      volume_v, volume_txt),
        ("Rui ro",      risk_v,   risk_txt),
        ("Fundamental", fund_v,   fund_txt),
        ("Smart Money", smart_v,  smart_txt),
        ("News",        news_v,   news_txt),
        ("Market",      market_v, market_txt),
        ("Macro",       macro_v,  macro_txt),
    ]

    detail_header = "\nCHI TIET AGENTS:\n"
    detail_parts  = []
    used = len(detail_header)

    for label, verdict, txt in agents:
        block = _fmt_agent(label, verdict, txt) + "\n"
        if used + len(block) <= detail_budget:
            detail_parts.append(block)
            used += len(block)
        else:
            # Không đủ chỗ cho full block — thử version rút gọn
            short_block = f"[{label} -> {verdict}] (Da rut gon do gioi han ky tu)\n"
            if used + len(short_block) <= detail_budget:
                detail_parts.append(short_block)
                used += len(short_block)
            # Nếu cả short cũng không vừa → bỏ qua agent này

    detail_section = detail_header + "\n".join(detail_parts)
    return header + detail_section + footer


def _fmt_price(p):
    """Format giá thông minh: < 1000 giữ thập phân, >= 1000 làm tròn."""
    p = float(p)
    if p < 1000:
        return f"{p:,.2f}".rstrip('0').rstrip('.')
    return f"{p:,.0f}"


def _format_action_plan(verdict_label: str, ap: dict) -> str:
    """
    Format Action Plan theo verdict — action plan khớp với hướng signal.
    MUA  : Entry / Target / SL
    BAN  : Kháng cự / Điểm thoát / Hỗ trợ tiếp theo
    TRUNG LAP: Chờ xác nhận
    """
    if "MUA" in verdict_label:
        rr_warn = ""
        if ap.get("poor_rr"):
            rr_warn = f"\n  ⚠️  CANH BAO: R:R={ap['rr']} < 1.5 — rui ro/phan thuong chua tot, can than khi vao lenh"
        return (
            f"ACTION PLAN (MUA):\n"
            f"  Entry:  {_fmt_price(ap['entry_low'])} - {_fmt_price(ap['entry_high'])}\n"
            f"  Target: {_fmt_price(ap['tp'])} ({ap['tp_pct']:+.1f}%)\n"
            f"  SL:     {_fmt_price(ap['sl'])} ({ap['sl_pct']:+.1f}%)\n"
            f"  R:R   = 1:{ap['rr']}{rr_warn}"
        )
    elif "BAN" in verdict_label:
        r_lo  = ap.get("resist_low",  ap["entry_high"])
        r_hi  = ap.get("resist_high", ap["tp"])
        e_lo  = ap.get("exit_low",    ap["entry_low"])
        e_hi  = ap.get("exit_high",   ap["entry_high"])
        h_sl  = ap.get("hard_sl",     ap["sl"])
        h_pct = ap.get("hard_sl_pct", ap["sl_pct"])
        return (
            f"ACTION PLAN (BAN / THOAT HANG):\n"
            f"  Vung khang cu (KHONG mua them): {_fmt_price(r_lo)} - {_fmt_price(r_hi)}\n"
            f"  Vung thoat hang: {_fmt_price(e_lo)} - {_fmt_price(e_hi)}\n"
            f"  Cat lo cuong buc: Pha xuong {_fmt_price(h_sl)} ({h_pct:+.1f}%) → THOAT NGAY\n"
            f"  Khong nen mua them — cho tin hieu dao chieu"
        )
    else:
        return (
            f"ACTION PLAN (CHO XAC NHAN):\n"
            f"  Tin hieu lan lon — KHONG hanh dong ngay\n"
            f"  Theo doi vung ho tro: {_fmt_price(ap['sl'])} - {_fmt_price(ap['entry_low'])}\n"
            f"  Chi xem xet khi: >50% agents dong thuan + Volume xac nhan"
        )


def run_verdict_agent(symbol, verdicts, ind):
    """
    Verdict — Rule-based voting từ 6 agents thực tế.
    Agents: trend, volume, risk, news, market, macro.
    Đã loại: fundamental (hallucinate), smart_money (data không chính xác).

    Voting: mỗi agent +1 (bullish) / -1 (bearish) / 0 (neutral).
    Verdict dựa trên đếm thực tế, KHÔNG có threshold tùy ý.
    LLM CHỈ viết summary 2 câu sau khi verdict đã được tính xong.
    """
    def _v(val): return val.upper().strip()

    # ── Vote từng agent — rule-based, deterministic ───────────────────────────
    votes = {
        "trend":  ( 1 if _v(verdicts["trend"]) in ("TANG", "TĂNG")
               else -1 if _v(verdicts["trend"]) in ("GIAM", "GIẢM") else 0),
        "volume": ( 1 if _v(verdicts["volume"]) in ("XAC NHAN", "XÁC NHẬN")
               else -1 if _v(verdicts["volume"]) in ("PHAN KY", "PHÂN KỲ") else 0),
        "risk":   ( 1 if _v(verdicts["risk"]) in ("THAP", "THẤP")
               else -1 if _v(verdicts["risk"]) == "CAO" else 0),
        "news":   ( 1 if _v(verdicts["news"]) in ("TICH CUC", "TÍCH CỰC")
               else -1 if _v(verdicts["news"]) in ("TIEU CUC", "TIÊU CỰC") else 0),
        "market": ( 0 if _v(verdicts["market"]) == "UNKNOWN"
               else  1 if _v(verdicts["market"]) == "UPTREND"
               else -1 if _v(verdicts["market"]) == "DOWNTREND" else 0),
        "macro":  ( 1 if _v(verdicts["macro"]) in ("THUAN LOI", "THUẬN LỢI")
               else -1 if _v(verdicts["macro"]) in ("RUI RO", "RỦI RO") else 0),
    }

    # Bỏ market nếu UNKNOWN khỏi active agents
    active_agents = {k: v for k, v in votes.items()
                     if not (k == "market" and _v(verdicts["market"]) == "UNKNOWN")}
    n_active = len(active_agents)

    # ── Fix mục 2: Weighted scoring theo chất lượng engine ──────────────────
    # Trọng số dựa trên: độ phức tạp thuật toán, signal clarity, VN suitability
    # Scale 1.0 = trọng số chuẩn; > 1.0 = tin cậy cao hơn
    ENGINE_WEIGHTS = {
        # Kỹ thuật cổ điển (đã verify HKUDS, phù hợp VN)
        "Ichimoku":        1.3,  # multi-filter, ít false signal
        "TechnicalBasic":  1.2,  # 3-dim voting, robust
        "SMC":             1.2,  # structure-based, trend-following
        "Candlestick":     0.9,  # single-bar pattern, nhiễu nhiều
        # Wave / Pattern (phức tạp, ít tín hiệu hơn nhưng xác suất cao khi có)
        "ElliottWave":     1.3,
        "Harmonic":        1.2,
        "Chanlun":         1.3,  # Chinese structure theory, phù hợp TTCK VN
        # Quant / ML
        "MLStrategy":      1.1,  # RandomForest, walk-forward validated
        "MultiFactor":     1.0,  # ts-zscore khi single-symbol (limitation)
        "Volatility":      0.8,  # regime indicator, không directional mạnh
        "Seasonal":        0.7,  # statistical tendency, không causal
        # Context agents (thấp hơn vì external data có thể delay/noise)
        "MarketRegime":    1.0,
        "NewsSentiment":   0.8,  # keyword-based, prone to false positive
        "VNMacro":         0.9,
        "CommodityContext":0.9,
        "FundamentalFilter": 1.1,
    }

    # Weighted bull/bear scores
    w_bull = sum(ENGINE_WEIGHTS.get(k, 1.0) for k, v in active_agents.items() if v == 1)
    w_bear = sum(ENGINE_WEIGHTS.get(k, 1.0) for k, v in active_agents.items() if v == -1)
    w_total = sum(ENGINE_WEIGHTS.get(k, 1.0) for k in active_agents)

    bull_count  = sum(1 for v in active_agents.values() if v == 1)
    bear_count  = sum(1 for v in active_agents.values() if v == -1)
    net_score   = bull_count - bear_count

    # ── Verdict: đếm thực tế, không threshold ma thuật ────────────────────────
    majority = n_active / 2

    if bull_count > majority and bear_count == 0:
        verdict_label = "DONG THUAN MUA"
    elif bull_count > majority:
        verdict_label = "NGHIENG MUA"
    elif bear_count > majority and bull_count == 0:
        verdict_label = "DONG THUAN BAN"
    elif bear_count > majority:
        verdict_label = "NGHIENG BAN"
    else:
        verdict_label = "TRUNG LAP"

    # Weighted confidence (dùng w_bull/w_bear thay vì raw count)
    dominant_w = max(w_bull, w_bear)
    confidence_pct = round(dominant_w / w_total * 100) if w_total > 0 else 0

    # ── Action Plan — ATR-based, không thay đổi ───────────────────────────────
    price   = ind["current_price"]
    support = ind["support_20d"]
    resist  = ind["resistance_20d"]
    bb_low  = ind["bb_lower"]
    bb_up   = ind["bb_upper"]

    atr_proxy = (bb_up - bb_low) / 4.0
    if atr_proxy <= 0:
        atr_proxy = price * 0.03

    sl_atr = round(price - 1.5 * atr_proxy, 2)
    sl = round(min(sl_atr, support * 0.985), 2) if support >= price * 0.85 else sl_atr

    tp_candidate = max(resist, bb_up)
    tp = round(price + 2.0 * atr_proxy, 2) if tp_candidate < price * 1.02 else round(tp_candidate * 0.995, 2)

    entry_low  = round(price * 0.99, 2)
    entry_high = round(price * 1.01, 2)
    if tp <= entry_high * 1.02:
        tp = round(entry_high * 1.03 + 2.0 * atr_proxy, 2)

    risk   = price - sl
    reward = tp - price
    rr     = round(reward / risk, 1) if risk > 0 else 0

    sl_pct = round((sl - price) / price * 100, 1)
    if sl_pct < -15:
        sl     = round(price * 0.92, 2)
        sl_pct = round((sl - price) / price * 100, 1)
        risk   = price - sl
        rr     = round(reward / risk, 1) if risk > 0 else 0

    # ── R:R Guard: tự động thu hẹp SL nếu R:R < 1.5 ─────────────────────────
    # Nguyên tắc: không nên vào lệnh nếu R:R < 1.5 (rủi ro > 2/3 reward)
    # Cách fix: thu hẹp SL về ATR-based thay vì dùng support xa
    # Nếu vẫn không đạt → đánh dấu poor_rr để hiển thị cảnh báo
    MIN_RR = 1.5
    if rr > 0 and rr < MIN_RR:
        # Thử SL chặt hơn: 1.0× ATR thay 1.5×
        sl_tight  = round(price - 1.0 * atr_proxy, 2)
        risk_tight = price - sl_tight
        rr_tight   = round(reward / risk_tight, 1) if risk_tight > 0 else 0
        if rr_tight >= MIN_RR:
            sl     = sl_tight
            sl_pct = round((sl - price) / price * 100, 1)
            risk   = risk_tight
            rr     = rr_tight
        # Nếu vẫn < 1.5 (TP quá gần): đánh dấu poor_rr, không block signal
        # nhưng sẽ hiển thị cảnh báo trong format

    poor_rr = rr > 0 and rr < MIN_RR  # flag để _format_action_plan hiển thị cảnh báo

    # ── VaR/CVaR (Historical Simulation) ─────────────────────────────────────
    # [HKUDS] risk-analysis/SKILL.md
    # VaR = maximum expected loss at confidence level (historical simulation)
    # CVaR = expected loss beyond VaR (Expected Shortfall)
    var_95 = var_99 = cvar_95 = None
    try:
        from vibe_skills import _ENGINES
        close_s = price_data["df"]["close"] if "df" in locals() else None
    except Exception:
        close_s = None
    # Tính từ ind nếu có price data
    try:
        # Lấy returns từ price trong ind
        _c = ind.get("_close_series")  # sẽ thêm vào compute_indicators
        if _c is None and "df" in dir():
            _c = df_price["close"] if "df_price" in dir() else None
    except Exception:
        _c = None

    # Tính VaR từ indicators sẵn có (vol_20d proxy)
    vol_daily = ind.get("volatility_20d", price * 0.02 / 100)  # daily vol
    import scipy.stats as _st
    var_95_pct  = round(float(_st.norm.ppf(0.05) * vol_daily * -1) * 100, 2) if vol_daily else None
    var_99_pct  = round(float(_st.norm.ppf(0.01) * vol_daily * -1) * 100, 2) if vol_daily else None
    cvar_95_pct = round(float(_st.norm.pdf(_st.norm.ppf(0.05)) / 0.05 * vol_daily) * 100, 2) if vol_daily else None
    var_95_abs  = round(price * (var_95_pct  or 0) / 100, 2) if var_95_pct  else None
    var_99_abs  = round(price * (var_99_pct  or 0) / 100, 2) if var_99_pct  else None

    # Tính thêm các mức cho action plan BÁN (thoát hàng)
    resist_zone_lo  = round(price * 1.00, 2)
    resist_zone_hi  = round(resist * 0.995, 2)
    exit_lo         = round(price * 0.985, 2)
    exit_hi         = round(price * 1.005, 2)
    hard_sl         = round(support * 0.99, 2)
    hard_sl_pct     = round((hard_sl - price) / price * 100, 1)

    action_plan = {
        "entry_low":    entry_low,  "entry_high": entry_high,
        "tp":           tp,         "sl":         sl,
        "tp_pct":       round((tp - price) / price * 100, 1),
        "sl_pct":       sl_pct,     "rr":         rr,
        "poor_rr":      poor_rr,    # True nếu R:R < 1.5 sau khi đã thử tối ưu
        # Sell-side fields
        "resist_low":   resist_zone_lo, "resist_high": resist_zone_hi,
        "exit_low":     exit_lo,        "exit_high":   exit_hi,
        "hard_sl":      hard_sl,        "hard_sl_pct": hard_sl_pct,
        # Risk metrics — [HKUDS] risk-analysis/SKILL.md (Parametric VaR)
        "var_95_pct":   var_95_pct,   "var_95_abs":  var_95_abs,
        "var_99_pct":   var_99_pct,   "var_99_abs":  var_99_abs,
        "cvar_95_pct":  cvar_95_pct,
    }

    # ── LLM: CHỈ viết summary 2 câu — KHÔNG ảnh hưởng verdict ───────────────
    # Tất cả data truyền vào đã là kết quả rule-based, LLM chỉ diễn đạt lại
    bull_agents = [k for k, v in active_agents.items() if v == 1]
    bear_agents = [k for k, v in active_agents.items() if v == -1]

    sys_p = (
        "Viet 2 dong tom tat phan tich co phieu bang tieng Viet khong dau. "
        "Chi dua tren du lieu duoc cung cap, KHONG them thong tin moi. "
        "Format:\nSUMMARY: <2 cau mo ta du lieu thuc te>\n"
        "NEGATIVE: <1 cau rui ro chinh dua tren so lieu>"
    )
    user = (
        f"Co phieu: {symbol} | Gia: {_fmt_price(price)}\n"
        f"Ket qua vote: {bull_count}/{n_active} agents bullish, {bear_count}/{n_active} bearish\n"
        f"Agents ung ho: {', '.join(bull_agents) or 'khong co'}\n"
        f"Agents phan doi: {', '.join(bear_agents) or 'khong co'}\n"
        f"RSI={ind['rsi']}, Vol={ind['volume_ratio']}x, "
        f"MA20={'tren' if price > ind['ma20'] else 'duoi'}, "
        f"Market={verdicts.get('market','N/A')}\n"
        "Viet SUMMARY va NEGATIVE dua CHINH XAC tren du lieu tren, khong suy dien them."
    )
    try:
        llm_txt = call_deepseek(sys_p, user, max_tokens=200)
        summary  = ""
        negative = ""
        for line in llm_txt.splitlines():
            line = line.strip()
            if line.upper().startswith("SUMMARY:"):
                summary = line[8:].strip()
            elif line.upper().startswith("NEGATIVE:"):
                negative = line[9:].strip()
        if not summary:
            summary = llm_txt[:150].strip()
    except Exception:
        summary  = (f"{bull_count}/{n_active} agents bullish, {bear_count}/{n_active} bearish. "
                    f"RSI={ind['rsi']}, Vol={ind['volume_ratio']}x.")
        negative = ""

    # Negative fallback từ data thực nếu LLM fail
    if not negative:
        neg_parts = []
        if ind["volume_ratio"] < 0.7:
            neg_parts.append(f"volume thap {ind['volume_ratio']}x TB20")
        if ind["rsi"] > 68:
            neg_parts.append(f"RSI={ind['rsi']} gan vung qua mua")
        if bear_count > 0:
            neg_parts.append(f"{bear_count} agent phan doi ({', '.join(bear_agents)})")
        negative = "Luu y: " + ", ".join(neg_parts) if neg_parts else "Theo doi them de xac nhan tin hieu."

    return {
        "verdict_label":  verdict_label,
        "confidence_pct": confidence_pct,
        "bull_count":     bull_count,
        "bear_count":     bear_count,
        "active_agents":  n_active,
        "net_score":      net_score,
        "summary":        summary,
        "action_plan":    action_plan,
        "negative":       negative,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — dùng Vibe-Trading engines thay thế hoàn toàn agents cũ
# ══════════════════════════════════════════════════════════════════════════════

def _vibe_verdict(vibe: dict, ind: dict, market: dict, news: dict,
                  commodity_data: dict = None,
                  macro_data: dict = None) -> dict:
    """
    Tổng hợp verdict từ kết quả Vibe-Trading engines.
    Context agents:
      - MarketRegime    : VN-Index trend (SSI/vnstock)
      - NewsSentiment   : RSS news scoring
      - CommodityContext: giá hàng hóa liên quan (yfinance)
      - VNMacro         : USD/VND + Gold + US10Y (macro-analysis/global-macro SKILL)
    [HKUDS] macro-analysis/SKILL.md, global-macro/SKILL.md
    """
    signals = dict(vibe["signals"])  # 13 HKUDS-verified Vibe signals
    context_details = {}  # detail text cho context agents

    # ── Context signal: Market Regime ─────────────────────────────────────
    if market.get("success"):
        chg_5d  = market["change_5d"]
        above   = market.get("above_ma20", False)
        chg_20d = market.get("change_20d", 0)
        proxy   = market.get("proxy", "proxy")
        bull_mkt = sum([chg_5d > 1, chg_20d > 3, above])
        bear_mkt = sum([chg_5d < -1, chg_20d < -3, not above])
        if bull_mkt >= 2:
            signals["MarketRegime"] = 1
            mr_label = "UPTREND"
        elif bear_mkt >= 2:
            signals["MarketRegime"] = -1
            mr_label = "DOWNTREND"
        else:
            signals["MarketRegime"] = 0
            mr_label = "SIDEWAYS"
        context_details["MarketRegime"] = (
            f"{proxy} {market['vnindex']:,.1f} | "
            f"5D={chg_5d:+.1f}% 20D={chg_20d:+.1f}% | "
            f"{'Tren' if above else 'Duoi'} MA20 | Regime={mr_label} "
            f"(bull={bull_mkt}/3 bear={bear_mkt}/3)"
        )
    else:
        signals["MarketRegime"] = 0
        context_details["MarketRegime"] = f"Khong lay duoc data thi truong: {market.get('error','unknown')}"

    # ── Context signal: News Sentiment ────────────────────────────────────
    # Dung keyword cu the hon de tranh false positive
    # Keyword ngắn hơn, khớp được cả có dấu và không dấu
    # Mỗi headline chỉ tính 1 điểm (không tính trùng)
    _POS_KW = [
        # Lợi nhuận / doanh thu tăng
        "tang truong", "tăng trưởng", "loi nhuan", "lợi nhuận",
        "doanh thu tang", "doanh thu tăng", "vuot ke hoach", "vượt kế hoạch",
        # Khuyến nghị / hành động mua
        "khuyen nghi mua", "khuyến nghị mua", "mua vao", "mua vào",
        "nang gia muc tieu", "nâng giá mục tiêu",
        # Sự kiện tích cực
        "co tuc", "cổ tức", "hop dong", "hợp đồng", "don hang", "đơn hàng",
        "tang von", "tăng vốn", "phat hanh", "phát hành",
        "tich cuc", "tích cực", "kha quan", "khả quan",
        "phuc hoi", "phục hồi", "tang manh", "tăng mạnh",
        "dot pha", "đột phá", "ky luc", "kỷ lục",
    ]
    _NEG_KW = [
        # Thua lỗ / sụt giảm
        "thua lo", "thua lỗ", "lo rong", "lỗ ròng",
        "giam manh", "giảm mạnh", "sut giam", "sụt giảm",
        "doanh thu giam", "doanh thu giảm", "loi nhuan giam", "lợi nhuận giảm",
        # Khuyến nghị bán / cảnh báo
        "khuyen nghi ban", "khuyến nghị bán", "ban ra", "bán ra",
        "ha gia muc tieu", "hạ giá mục tiêu",
        "canh bao", "cảnh báo", "rui ro", "rủi ro",
        # Sự kiện tiêu cực
        "vi pham", "vi phạm", "bi phat", "bị phạt",
        "no xau", "nợ xấu", "mat thanh khoan", "mất thanh khoản",
        "dinh chi", "đình chỉ", "thu hoi", "thu hồi",
        "tieu cuc", "tiêu cực", "suy giam", "suy giảm",
        "khung hoang", "khủng hoảng",
    ]
    if news.get("success") and news.get("headlines"):
        headlines = news["headlines"]
        total_hl  = len(headlines)
        pos = 0
        neg = 0
        # Đếm per-headline để tránh 1 bài báo count nhiều lần
        for hl in headlines:
            hl_lo = hl.lower()
            if any(kw in hl_lo for kw in _POS_KW):
                pos += 1
            elif any(kw in hl_lo for kw in _NEG_KW):
                neg += 1

        if pos > neg and pos >= 1:
            signals["NewsSentiment"] = 1
            ns_label = "TICH CUC"
        elif neg > pos and neg >= 1:
            signals["NewsSentiment"] = -1
            ns_label = "TIEU CUC"
        elif pos > 0 and pos == neg:
            signals["NewsSentiment"] = 0
            ns_label = "TRUNG LAP"
        else:
            signals["NewsSentiment"] = 0
            ns_label = "TRUNG LAP"
        # Fix mục 4: thêm source attribution để tránh hallucination
        active_sources = [k for k,v in news.get("source_summary",{}).items() if v > 0]
        src_str = ", ".join(active_sources[:5]) if active_sources else "N/A"
        context_details["NewsSentiment"] = (
            f"{total_hl} headlines | pos={pos} neg={neg} | {ns_label} | "
            f"Nguon: {src_str}"
        )
    else:
        signals["NewsSentiment"] = 0
        context_details["NewsSentiment"] = "Khong lay duoc tin tuc"

    # ── Context signal: Commodity Context ────────────────────────────────
    # [HKUDS] commodity-analysis/SKILL.md
    # Chỉ áp dụng khi symbol thuộc sector có commodity liên quan
    if commodity_data and commodity_data.get("success"):
        signals["CommodityContext"] = commodity_data["signal"]
        context_details["CommodityContext"] = commodity_data["detail"]
    else:
        # Symbol không có commodity mapping hoặc fetch failed → bỏ qua (không thêm signal 0)
        # Để tránh dilute verdict khi không có thông tin
        err = commodity_data.get("error", "N/A") if commodity_data else "not fetched"
        context_details["CommodityContext"] = f"N/A: {err}"
        # Không thêm vào signals → không count trong verdict

    # ── Context signal: VN Macro Context ─────────────────────────────────
    # [HKUDS] macro-analysis/SKILL.md + global-macro/SKILL.md
    # USD/VND + Gold + US 10Y → macro tilt cho TTCK VN
    if macro_data and macro_data.get("success"):
        signals["VNMacro"] = macro_data["signal"]
        context_details["VNMacro"] = macro_data["detail"]
    else:
        # Không count nếu không lấy được data
        err = macro_data.get("detail", "N/A") if macro_data else "not fetched"
        context_details["VNMacro"] = f"N/A: {err}"

    # ── Tổng hợp vote ─────────────────────────────────────────────────────
    n_active = len(signals)
    bull = sum(1 for v in signals.values() if v > 0)
    bear = sum(1 for v in signals.values() if v < 0)
    threshold_strong = 0.5   # >50% để ra signal rõ ràng
    threshold_lean   = 0.3   # >30% để ra "nghiêng"

    # Logic ưu tiên: bull=0 và có bear → không thể TRUNG LAP
    if bull == 0 and bear > 0:
        # Không có ai bullish, có người bearish → BAN
        if bear / n_active >= threshold_strong:
            verdict_label = "DONG THUAN BAN"
        else:
            verdict_label = "NGHIENG BAN"
    elif bear == 0 and bull > 0:
        # Không có ai bearish, có người bullish → MUA
        if bull / n_active >= threshold_strong:
            verdict_label = "DONG THUAN MUA"
        else:
            verdict_label = "NGHIENG MUA"
    elif bull > bear and bull / n_active >= threshold_strong:
        verdict_label = "DONG THUAN MUA" if bear == 0 else "NGHIENG MUA"
    elif bear > bull and bear / n_active >= threshold_strong:
        verdict_label = "DONG THUAN BAN" if bull == 0 else "NGHIENG BAN"
    elif bear > bull and bear / n_active >= threshold_lean:
        verdict_label = "NGHIENG BAN"
    elif bull > bear and bull / n_active >= threshold_lean:
        verdict_label = "NGHIENG MUA"
    else:
        verdict_label = "TRUNG LAP"

    conf_pct = round(max(bull, bear) / n_active * 100)

    # ── Action Plan (ATR-based) ───────────────────────────────────────────
    price  = ind["current_price"]
    bb_up  = ind["bb_upper"]; bb_low = ind["bb_lower"]
    sup    = ind["support_20d"]; res = ind["resistance_20d"]
    atr    = (bb_up - bb_low) / 4.0 or price * 0.03

    sl_raw = price - 1.5 * atr
    sl     = round(min(sl_raw, sup * 0.985), 2) if sup >= price * 0.85 else round(sl_raw, 2)
    tp_cand = max(res, bb_up)
    tp     = round(price + 2.0 * atr, 2) if tp_cand < price * 1.02 else round(tp_cand * 0.995, 2)
    e_lo   = round(price * 0.99, 2); e_hi = round(price * 1.01, 2)
    if tp <= e_hi * 1.02: tp = round(e_hi * 1.03 + 2.0 * atr, 2)
    sl_pct = round((sl - price) / price * 100, 1)
    if sl_pct < -15: sl = round(price * 0.92, 2); sl_pct = round((sl - price) / price * 100, 1)
    risk   = price - sl; reward = tp - price
    rr     = round(reward / risk, 1) if risk > 0 else 0

    # ── R:R Guard: tự động thu hẹp SL để đạt R:R >= 1.5 ─────────────────
    MIN_RR = 1.5
    if rr > 0 and rr < MIN_RR:
        sl_tight   = round(price - 1.0 * atr, 2)
        risk_tight = price - sl_tight
        rr_tight   = round(reward / risk_tight, 1) if risk_tight > 0 else 0
        if rr_tight >= MIN_RR:
            sl     = sl_tight
            sl_pct = round((sl - price) / price * 100, 1)
            rr     = rr_tight
    poor_rr = rr > 0 and rr < MIN_RR

    # ── Volume Gate: vol < 0.5x → override verdict sang CHO XAC NHAN VOL ─
    # Thanh khoản quá thấp khiến tín hiệu kỹ thuật không đáng tin
    vol_ratio = ind.get("volume_ratio", 1.0)
    low_volume_gate = False
    if vol_ratio < 0.5 and "MUA" in verdict_label:
        verdict_label = "CHO XAC NHAN VOL"
        low_volume_gate = True

    # Tính các mức cho action plan BÁN
    hard_sl     = round(sup * 0.99, 2)
    hard_sl_pct = round((hard_sl - price) / price * 100, 1)

    action_plan = {
        "entry_low":       e_lo,   "entry_high": e_hi,
        "tp":              tp,     "sl":         sl,
        "tp_pct":          round((tp - price) / price * 100, 1),
        "sl_pct":          sl_pct, "rr":         rr,
        "poor_rr":         poor_rr,
        "low_volume_gate": low_volume_gate,
        "vol_ratio":       round(vol_ratio, 2),
        # Sell-side fields
        "resist_low":   round(price * 1.00, 2),
        "resist_high":  round(res * 0.995, 2),
        "exit_low":     round(price * 0.985, 2),
        "exit_high":    round(price * 1.005, 2),
        "hard_sl":      hard_sl,
        "hard_sl_pct":  hard_sl_pct,
    }

    # ── LLM summary — chỉ diễn đạt lại số liệu, KHÔNG suy diễn ─────────────
    bull_agents = [k for k,v in signals.items() if v > 0]
    bear_agents = [k for k,v in signals.items() if v < 0]
    neut_agents = [k for k,v in signals.items() if v == 0]

    sys_p = (
        "Ban la bo phan viet bao cao ngan cho co phieu Viet Nam. "
        "Nhiem vu: chi dien dat lai dung so lieu duoc cap, KHONG them thong tin ngoai. "
        "TUYET DOI KHONG noi 'phan lon engine cho thay tang' neu so lieu KHONG ung ho dieu do. "
        "Format bat buoc:\n"
        "SUMMARY: <2 cau mo ta chinh xac so lieu: X/N bullish, Y/N bearish, ket luan la gi>\n"
        "NEGATIVE: <1 cau ve rui ro chinh dua tren so lieu>"
    )
    user_p = (
        f"Co phieu: {ind.get('symbol','?')} | Gia: {_fmt_price(price)}\n"
        f"KET LUAN: {verdict_label} (tin cay {conf_pct}%)\n"
        f"So lieu chinh xac: {bull}/{n_active} agents BULLISH, {bear}/{n_active} BEARISH, "
        f"{len(neut_agents)}/{n_active} NEUTRAL\n"
        f"Agents BULLISH: {', '.join(bull_agents) or 'KHONG CO'}\n"
        f"Agents BEARISH: {', '.join(bear_agents) or 'KHONG CO'}\n"
        f"RSI={ind['rsi']} (14D) | Vol={ind['volume_ratio']}x TB20 | "
        f"1W={ind['change_1w_pct']:+.1f}% | 1M={ind['change_1m_pct']:+.1f}%\n"
        f"Market Regime: {'UPTREND' if signals.get('MarketRegime',0)>0 else 'DOWNTREND' if signals.get('MarketRegime',0)<0 else 'SIDEWAYS'}\n"
        f"Viet SUMMARY phan anh CHINH XAC {bull}/{n_active} bullish va {bear}/{n_active} bearish."
    )
    try:
        llm_txt = call_deepseek(sys_p, user_p, max_tokens=200)
        summary = neg_txt = ""
        for line in llm_txt.splitlines():
            ls = line.strip()
            if ls.upper().startswith("SUMMARY:"): summary = ls[8:].strip()
            elif ls.upper().startswith("NEGATIVE:"): neg_txt = ls[9:].strip()
        if not summary: summary = llm_txt[:150].strip()
    except Exception:
        # Fallback hoàn toàn rule-based nếu LLM fail
        if verdict_label in ("DONG THUAN MUA", "NGHIENG MUA"):
            summary = (f"{bull}/{n_active} agents ung ho MUA ({', '.join(bull_agents[:3])}). "
                       f"RSI={ind['rsi']}, Vol={ind['volume_ratio']}x.")
        elif verdict_label in ("DONG THUAN BAN", "NGHIENG BAN"):
            summary = (f"{bear}/{n_active} agents canh bao BAN ({', '.join(bear_agents[:3])}). "
                       f"RSI={ind['rsi']}, xu huong yeu.")
        else:
            summary = (f"Tin hieu lan lon: {bull}/{n_active} bullish, {bear}/{n_active} bearish. "
                       f"Can them xac nhan truoc khi hanh dong.")
        neg_txt = ""

    if not neg_txt:
        parts = []
        if ind["volume_ratio"] < 0.7: parts.append(f"vol thap {ind['volume_ratio']}x TB20")
        if ind["rsi"] > 68: parts.append(f"RSI={ind['rsi']} gan vung qua mua")
        if ind["rsi"] < 35: parts.append(f"RSI={ind['rsi']} oversold, co the con giam")
        if bear > 0: parts.append(f"{bear} agent phan doi ({', '.join(bear_agents[:2])})")
        if ind["change_1m_pct"] < -10: parts.append(f"giam {ind['change_1m_pct']:.1f}% trong 1 thang")
        neg_txt = "Luu y: " + ", ".join(parts) if parts else "Theo doi them de xac nhan tin hieu."

    # ── Relative Weakness Alert ────────────────────────────────────────
    # Phân kỳ: thị trường UPTREND nhưng cổ phiếu yếu về kỹ thuật
    # Điều kiện: MarketRegime=MUA (1) + phần lớn kỹ thuật BAN + giá giảm
    mr_signal   = signals.get("MarketRegime", 0)
    tech_agents = ["Candlestick","Ichimoku","TechnicalBasic","ElliottWave",
                   "CrossMarket","PriceMomentum","Breakout"]
    tech_bear = sum(1 for a in tech_agents if signals.get(a, 0) < 0)
    tech_bull = sum(1 for a in tech_agents if signals.get(a, 0) > 0)

    # ── Fix mục 3: Giải thích mâu thuẫn giữa các engines ────────────────────
    bull_engines  = [k for k, v in signals.items() if v == 1]
    bear_engines  = [k for k, v in signals.items() if v == -1]
    contradictions = []
    _CONFLICT_PAIRS = [
        ("MultiFactor",    "FundamentalFilter",
         "MultiFactor thay dong luc tang nhung FundamentalFilter canh bao dinh gia cao"),
        ("TechnicalBasic", "Ichimoku",
         "TechnicalBasic va Ichimoku trai chieu — co the dang o vung chuyen pha"),
        ("ElliottWave",    "SMC",
         "Elliott bao hieu song tang nhung SMC phat hien CHoCH — cau truc co the bi pha"),
        ("Seasonal",       "TechnicalBasic",
         "Seasonal bias trai voi ky thuat — yeu to mua vu co the ghi de momentum"),
        ("MarketRegime",   "Ichimoku",
         "Thi truong UPTREND nhung Ichimoku BAN — co phieu yeu hon thi truong"),
        ("MLStrategy",     "Candlestick",
         "ML signal trai chieu nen — nhieu ngan han vs xu huong trung han"),
        ("Chanlun",        "TechnicalBasic",
         "Chan luan trai chieu chi bao co ban — hai khung thoi gian khac nhau"),
    ]
    for bull_e, bear_e, reason in _CONFLICT_PAIRS:
        if (bull_e in bull_engines and bear_e in bear_engines) or            (bull_e in bear_engines and bear_e in bull_engines):
            contradictions.append(f"⚡ {bull_e} >< {bear_e}: {reason}")

    relative_weakness = (
        mr_signal > 0               # thị trường tăng
        and tech_bear >= 3          # >= 3 agent kỹ thuật báo BAN
        and tech_bear > tech_bull   # kỹ thuật nghiêng về giảm
        and ind["change_1w_pct"] < 0  # giá đang giảm
    )
    relative_weakness_note = ""
    if relative_weakness:
        relative_weakness_note = (
            f"⚠️ CANH BAO PHAN KY: Co phieu dang YEU hon thi truong chung "
            f"(Market UPTREND nhung {tech_bear}/{len(tech_agents)} Agent ky thuat bao BAN). "
            f"Day la dau hieu Co phieu mat suc manh tuong doi (Relative Weakness) — "
            f"rui ro cao hon binh thuong du thi truong chung dang tang."
        )

    return {
        "verdict_label": verdict_label, "confidence_pct": conf_pct,
        "bull_count": bull, "bear_count": bear, "active_agents": n_active,
        "signals": signals,
        "context_details": context_details,
        "summary": summary, "negative": neg_txt, "action_plan": action_plan,
        "relative_weakness":      relative_weakness,
        "relative_weakness_note":  relative_weakness_note,
        "contradictions":          contradictions,
    }


def _build_vibe_message(
    emoji, symbol, now, ind, vnindex_str,
    vibe: dict, verdict: dict,
    char_limit: int = 4000
) -> str:
    """Build Telegram message với kết quả Vibe-Trading 16+2 agents."""
    ap  = verdict["action_plan"]
    fv  = verdict["verdict_label"]
    pct = verdict["confidence_pct"]
    bull = verdict["bull_count"]
    bear = verdict["bear_count"]
    n    = verdict["active_agents"]

    # Emoji từng signal
    def _sig_emoji(v): return "🟢" if v > 0 else "🔴" if v < 0 else "⚪"

    sigs    = verdict["signals"]
    details = dict(vibe.get("details", {}))
    details.update(verdict.get("context_details", {}))

    # Engines hiếm khi fire — khi có signal thì đáng chú ý đặc biệt
    _RARE_ENGINES = {"Ichimoku", "ElliottWave", "Harmonic", "Chanlun"}

    def _tl_note(name, det_str):
        """Trích TL reason từ detail string nếu có."""
        if not det_str:
            return ""
        # TL reason được embed trong detail với prefix "TL-ly-do:"
        if "TL-ly-do:" in det_str:
            idx = det_str.index("TL-ly-do:")
            reason = det_str[idx:].split("|")[0].strip()
            # Rút gọn để vừa 1 dòng
            return " — " + reason.replace("TL-ly-do:", "").strip()[:60]
        return ""

    agents_lines = "\n".join(
        f"  {_sig_emoji(v)} {name:<16} {'MUA' if v>0 else 'BAN' if v<0 else 'TL'}"
        + (_tl_note(name, details.get(name, "")) if v == 0 else
           (" ⭐" if name in _RARE_ENGINES and v != 0 else ""))
        for name, v in sigs.items()
    )

    # Rare Signal Highlight block — chỉ hiện khi có ít nhất 1 rare engine fire
    rare_fired = [(name, v) for name, v in sigs.items()
                  if name in _RARE_ENGINES and v != 0]
    rare_block = ""
    if rare_fired:
        rare_lines = []
        for name, v in rare_fired:
            direction = "MUA" if v > 0 else "BAN"
            det_short = details.get(name, "")[:80].replace("\n", " ")
            rare_lines.append(f"  ⭐ {name} → {direction}: {det_short}")
        rare_block = "\n⚡ HIGH-CONVICTION (engine hiem khi fire):\n" + "\n".join(rare_lines) + "\n"

    header = (
        f"{emoji} {symbol} — {now}\n\n"
        f"GIA & INDI:\n"
        f"  Gia: {_fmt_price(ind['current_price'])} | "
        f"1W: {ind['change_1w_pct']:+.1f}% | 1M: {ind['change_1m_pct']:+.1f}%\n"
        f"  RSI: {ind['rsi']} | MACD: {ind['macd_hist']:+.4f} | "
        f"Vol: {ind['volume_ratio']}x TB20\n"
        f"  Ho tro: {_fmt_price(ind['support_20d'])} | "
        f"Khang cu: {_fmt_price(ind['resistance_20d'])}\n"
        f"  VN-Index: {vnindex_str}\n\n"
        f"VIBE-TRADING ({n} AGENTS):\n{agents_lines}\n"
        + rare_block
    )

    verdict_block = (
        f"\n{'='*32}\n"
        f"{emoji} KET LUAN: {fv}\n"
        f"   {bull}/{n} bullish | {bear}/{n} bearish | Tin cay: {pct}%\n\n"
        f"TOM TAT: {verdict['summary']}\n\n"
    )

    # Action plan — theo chiều verdict
    if ap.get("low_volume_gate"):
        # Vol < 0.5x: tín hiệu MUA không đáng tin do thiếu thanh khoản
        action_block = (
            f"ACTION PLAN (CHO XAC NHAN VOL):\n"
            f"  ⚠️  KHONG VAO LENH — Volume {ap['vol_ratio']}x TB20 qua thap (<0.5x)\n"
            f"  Tin hieu ky thuat co the gia tao khi thanh khoan can kiet\n"
            f"  Dieu kien de chuyen sang MUA:\n"
            f"    1. Vol >= 0.7x TB20 it nhat 2 phien lien tiep\n"
            f"    2. Gia giu tren {_fmt_price(ap['sl'])} (SL tham chieu)\n"
            f"  Khi du dieu kien: Entry {_fmt_price(ap['entry_low'])} - {_fmt_price(ap['entry_high'])}"
            f" | Target {_fmt_price(ap['tp'])} | R:R=1:{ap['rr']}\n"
        )
    elif "MUA" in fv:
        # VaR display — [HKUDS] risk-analysis/SKILL.md
        var_str = ""
        if ap.get("var_95_pct"):
            var_str = f"  VaR 95%: {ap['var_95_pct']:+.2f}% (~{_fmt_price(ap.get('var_95_abs',0))} max loss/day)\n"
        rr_warn = ""
        if ap.get("poor_rr"):
            rr_warn = f"  ⚠️  CANH BAO: R:R={ap['rr']} < 1.5 — rui ro/phan thuong chua tot, can than khi vao lenh\n"
        action_block = (
            f"ACTION PLAN (MUA):\n"
            f"  Entry:  {_fmt_price(ap['entry_low'])} - {_fmt_price(ap['entry_high'])}\n"
            f"  Target: {_fmt_price(ap['tp'])} ({ap['tp_pct']:+.1f}%)\n"
            f"  SL:     {_fmt_price(ap['sl'])} ({ap['sl_pct']:+.1f}%)\n"
            f"  R:R   = 1:{ap['rr']}\n"
            + rr_warn + var_str
        )
    elif "BAN" in fv:
        # BÁN: kháng cự / thoát hàng / cắt lỗ cứng — không hiển thị vùng mua
        r_lo  = ap.get("resist_low",  ap["entry_high"])
        r_hi  = ap.get("resist_high", ap["tp"])
        e_lo  = ap.get("exit_low",    ap["entry_low"])
        e_hi  = ap.get("exit_high",   ap["entry_high"])
        h_sl  = ap.get("hard_sl",     ap["sl"])
        h_pct = ap.get("hard_sl_pct", ap["sl_pct"])
        action_block = (
            f"ACTION PLAN (BAN / THOAT HANG):\n"
            f"  Vung khang cu (KHONG mua them): {_fmt_price(r_lo)} - {_fmt_price(r_hi)}\n"
            f"  Vung thoat hang (dat lenh ban): {_fmt_price(e_lo)} - {_fmt_price(e_hi)}\n"
            f"  Cat lo cuong buc: Pha xuong {_fmt_price(h_sl)} ({h_pct:+.1f}%) → THOAT NGAY\n"
            f"  Khong nen mua them — cho tin hieu dao chieu (RSI<35 + Vol xac nhan)\n"
        )
    else:
        # TRUNG LAP: chờ xác nhận, không hành động
        var_str = ""
        if ap.get("var_95_pct"):
            var_str = f"  VaR 95%: {ap['var_95_pct']:+.2f}% — rui ro thua lo toi da trong 1 ngay\n"
        action_block = (
            f"ACTION PLAN (CHO XAC NHAN):\n"
            f"  Tin hieu lan lon — KHONG hanh dong ngay\n"
            f"  Theo doi vung ho tro: {_fmt_price(ap['sl'])} - {_fmt_price(ap['entry_low'])}\n"
            f"  Chi xem xet mua khi: >50% agents dong thuan + Volume xac nhan\n"
            + var_str
        )

    # Vibe details (rút gọn cho vừa Telegram)
    # Gộp: 14 engine details + 2 context agent details
    details = dict(vibe.get("details", {}))
    details.update(verdict.get("context_details", {}))
    detail_lines = []
    for name, v in sigs.items():
        desc = details.get(name, "")
        short = desc[:100].replace("\n", " ")
        detail_lines.append(f"  [{name}] {short}")
    detail_block = "\nCHI TIET:\n" + "\n".join(detail_lines) if detail_lines else ""

    # Footer — theo report-generate SKILL: kết luận rõ ràng + risk
    rw_note  = verdict.get("relative_weakness_note", "")
    contradictions = verdict.get("contradictions", [])
    neg_note = verdict.get("negative", "")
    # VaR summary cho risk section
    var_note = ""
    ap_ref   = verdict.get("action_plan", {})
    if ap_ref.get("var_95_pct") and ap_ref.get("cvar_95_pct"):
        var_note = (f"Risk (VaR 95%={ap_ref['var_95_pct']:+.2f}% "
                    f"CVaR={ap_ref['cvar_95_pct']:+.2f}%): "
                    f"{'Rui ro cao — quan ly vi the chat' if abs(ap_ref['var_95_pct']) > 3 else 'Rui ro binh thuong'}. ")
    footer = (
        f"\nLUU Y: {neg_note}\n"
        + (f"{var_note}\n" if var_note else "")
        + (f"\n{rw_note}\n" if rw_note else "")
        + ("\n".join(contradictions) + "\n" if contradictions else "")
        + f"{'='*32}"
    )

    # Smart truncate
    base = header + verdict_block + action_block + footer
    remaining = char_limit - len(base) - 5
    if remaining > 100 and detail_block:
        detail_trunc = detail_block[:remaining]
        msg = header + verdict_block + action_block + detail_trunc + footer
    else:
        msg = base

    return msg[:char_limit]



VN30_SYMBOLS = [
    "VCB","BID","CTG","TCB","MBB","VPB","ACB","HDB","STB","LPB",
    "VIC","VHM","VRE","MSN","MWG","FPT","HPG","HSG","NKG","GAS",
    "PLX","POW","PVD","REE","SAB","SSI","VND","HCM","VCI","DXG",
]


def _get_volume_spike_top5(exclude_watchlist: list) -> str:
    """Quét VN30, tìm top 5 mã có volume spike cao nhất so với TB 20 phiên"""
    results = []
    to_scan = [s for s in VN30_SYMBOLS if s not in exclude_watchlist]

    def _check_spike(sym):
        try:
            data = get_price_data(sym, 30)
            if not data.get("success"):
                return None
            df     = data["df"]
            volume = df["volume"].astype(float)
            close  = df["close"].astype(float)
            avg20  = float(volume.rolling(20).mean().iloc[-1])
            if avg20 <= 0:
                return None
            vol_ratio = float(volume.iloc[-1]) / avg20
            change_1d = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 else 0
            return {
                "symbol":    sym,
                "vol_ratio": round(vol_ratio, 2),
                "price":     round(float(close.iloc[-1]), 0),
                "change_1d": round(change_1d, 1),
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_check_spike, s): s for s in to_scan}
        for f in futures:
            r = f.result()
            if r and r["vol_ratio"] >= 1.5:   # chỉ lấy mã có vol spike >= 1.5x
                results.append(r)

    if not results:
        return "Khong co ma nao co volume spike dang ke trong phien nay."

    top5 = sorted(results, key=lambda x: x["vol_ratio"], reverse=True)[:5]
    lines = ["TOP 5 VOL SPIKE (VN30):"]
    for r in top5:
        em = "🟢" if r["change_1d"] > 0 else "🔴" if r["change_1d"] < 0 else "🟡"
        lines.append(
            f"  {em} {r['symbol']}  {r['price']:,.0f}  ({r['change_1d']:+.1f}%)  "
            f"Vol: {r['vol_ratio']}x TB20"
        )
    return "\n".join(lines)


def scan_watchlist(watchlist: list) -> str:
    market = get_market_data()
    results = []

    for symbol in watchlist:
        try:
            data = get_price_data(symbol, 30)
            if not data["success"]:
                results.append(f"❌ {symbol}: Không lấy được data")
                continue
            ind = compute_indicators(data["df"])
            rsi, change_1w, vol = ind["rsi"], ind["change_1w_pct"], ind["volume_ratio"]

            tags = []
            if rsi < 35:             tags.append("🔵 Quá bán")
            elif rsi > 70:           tags.append("🔺 Quá mua")
            if vol > 1.5:            tags.append(f"📊 Vol {vol}x")
            if abs(change_1w) > 5:   tags.append(f"⚡ {change_1w:+.1f}%")
            if not tags:              tags.append("➡️ Bình thường")

            em = "🟢" if change_1w > 1 else "🔴" if change_1w < -2 else "🟡"
            results.append(
                f"{em} *{symbol}* `{ind['current_price']:,.0f}` ({change_1w:+.1f}%)\n"
                f"   RSI:{rsi} Vol:{vol}x {' '.join(tags)}"
            )
        except Exception as e:
            results.append(f"❌ {symbol}: {str(e)[:40]}")

    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    vnindex_line = ""
    if market.get("success"):
        regime = ("🟢 UPTREND"   if market["above_ma20"] and market["change_5d"] > 0 else
                  "🔴 DOWNTREND" if not market["above_ma20"] and market["change_5d"] < 0 else
                  "🟡 SIDEWAYS")
        vnindex_line = (f"\n🌊 VN-Index: {market['vnindex']:,} "
                        f"({market['change_5d']:+.1f}% 5D) — {regime}\n{'─'*28}\n")

    # Volume spike toàn VN30
    spike_section = _get_volume_spike_top5(watchlist)

    header = f"📋 Scan Watchlist — {now}\n{'─'*28}{vnindex_line}"
    footer = f"\n\n{'─'*28}\n{spike_section}\n\n/check <MA> de phan tich sau 8 agents"
    return header + "\n".join(results) + footer


def load_previous_context(symbol: str) -> dict | None:
    """
    Tải bối cảnh phân tích gần nhất của symbol từ DB.

    Returns:
        dict với keys:
          verdict_label, confidence_pct, price, timestamp,
          days_ago, bull_count, bear_count, key_signals, summary_line
        hoặc None nếu không có record / DB không available.
    """
    try:
        from db import get_conn
    except ImportError:
        return None

    try:
        conn = get_conn()
        try:
            cur = conn.cursor()
            # Lấy record mới nhất của symbol (tối đa 30 ngày để còn relevant)
            # Query 1: ưu tiên signal chưa hết hạn
            cur.execute(
                """
                SELECT id, verdict_label, confidence_pct, entry_price,
                       bull_count, bear_count, agent_verdicts, summary,
                       created_at, expires_at,
                       (NOW() > expires_at) AS is_expired
                FROM   signals
                WHERE  symbol = %s
                  AND  created_at >= NOW() - INTERVAL '30 days'
                ORDER  BY
                    (NOW() > expires_at) ASC,  -- chưa hết hạn lên trước
                    created_at DESC
                LIMIT  1
                """,
                (symbol.upper()[:10],)
            )
            row = cur.fetchone()
        finally:
            conn.close()
    except Exception as e:
        _logger.debug(f"load_previous_context: DB query failed: {e}")
        return None

    if not row:
        return None

    try:
        # Row giờ có thêm expires_at và is_expired
        (sid, verdict_label, confidence_pct, price,
         bull_count, bear_count, agent_verdicts_raw, summary,
         created_at, expires_at, is_expired) = row

        # Tính số ngày đã qua
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        # created_at có thể là naive hoặc aware
        if hasattr(created_at, "tzinfo") and created_at.tzinfo is None:
            from datetime import timezone as tz
            created_at = created_at.replace(tzinfo=tz.utc)
        delta = now_utc - created_at
        days_ago  = delta.days
        hours_ago = int(delta.total_seconds() // 3600)

        # ── Shelf Life check ─────────────────────────────────────────────────
        # is_expired: DB đã tính (NOW() > expires_at), nhưng double-check Python-side
        # để tránh timezone mismatch
        _signal_expired = bool(is_expired)
        if expires_at is not None:
            _exp = expires_at
            if hasattr(_exp, "tzinfo") and _exp.tzinfo is None:
                from datetime import timezone as _tz
                _exp = _exp.replace(tzinfo=_tz.utc)
            _signal_expired = now_utc > _exp

        if _signal_expired:
            # Tín hiệu đã hết hạn → trả về None với thông báo rõ ràng
            # Không inject context cũ không còn giá trị vào analysis mới
            _logger.debug(
                f"load_previous_context: {symbol} signal expired "
                f"(created={days_ago}d ago, verdict={verdict_label})"
            )
            return {
                "expired":       True,
                "verdict_label": verdict_label,
                "confidence_pct": confidence_pct,
                "days_ago":      days_ago,
                "summary_line":  (
                    f"📜 Lan truoc ({days_ago} ngay truoc): "
                    f"{verdict_label} — [HET HAN, khong con hieu luc]"
                ),
                "agent_context": None,   # không inject vào agents
            }

        # Parse agent_verdicts (lưu dạng JSON string hoặc dict)
        import json
        if isinstance(agent_verdicts_raw, str):
            try:
                agent_verdicts = json.loads(agent_verdicts_raw)
            except Exception:
                agent_verdicts = {}
        elif isinstance(agent_verdicts_raw, dict):
            agent_verdicts = agent_verdicts_raw
        else:
            agent_verdicts = {}

        # Trích các signal quan trọng nhất (ưu tiên high-weight engines)
        _HIGH_WEIGHT = ["Ichimoku", "ElliottWave", "Chanlun", "SMC",
                        "TechnicalBasic", "MarketRegime", "FundamentalFilter"]
        key_signals = []
        for eng in _HIGH_WEIGHT:
            v = agent_verdicts.get(eng, "")
            if v and v != "TRUNG LAP":
                key_signals.append(f"{eng}={v}")
        if not key_signals:
            # fallback: bất kỳ signal nào khác TRUNG LAP
            key_signals = [f"{k}={v}" for k, v in list(agent_verdicts.items())[:4]
                           if v != "TRUNG LAP"]

        # Format thời gian đẹp
        if days_ago == 0:
            time_str = f"{hours_ago} gio truoc" if hours_ago > 0 else "vua xong"
        elif days_ago == 1:
            time_str = "1 ngay truoc"
        else:
            time_str = f"{days_ago} ngay truoc"

        # Dòng tóm tắt hiển thị đầu output
        price_str = f"{float(price):,.0f}" if price else "N/A"
        conf_str  = f"{int(confidence_pct)}%" if confidence_pct else "?"
        summary_line = (
            f"📜 Lan truoc ({time_str}): {verdict_label} {conf_str} | "
            f"Gia={price_str} | {', '.join(key_signals[:3]) or 'N/A'}"
        )

        # System-prompt context để inject vào agents
        agent_context = (
            f"[BOI CANH PHAN TICH TRUOC — {symbol}]\n"
            f"Thoi gian: {time_str} (ngay {created_at.strftime('%d/%m/%Y %H:%M')})\n"
            f"Ket luan cu: {verdict_label} (do tin cay {conf_str})\n"
            f"Gia luc do: {price_str} VND\n"
            f"Bull/Bear agents: {bull_count}/{bear_count}\n"
            f"Tin hieu chinh: {', '.join(key_signals[:5]) or 'N/A'}\n"
            f"Tom tat: {(summary or '')[:200]}\n"
            f"---\n"
            f"YEU CAU: So sanh voi phan tich HIEN TAI. Lam noi bat nhung thay doi "
            f"quan trong (verdict, momentum, cau truc). "
            f"Neu verdict thay doi chieu (vi du: tu BAN sang MUA), hay giai thich ro nguyen nhan."
        )

        return {
            "verdict_label":   verdict_label,
            "confidence_pct":  confidence_pct,
            "price":           price,
            "days_ago":        days_ago,
            "hours_ago":       hours_ago,
            "time_str":        time_str,
            "bull_count":      bull_count,
            "bear_count":      bear_count,
            "key_signals":     key_signals,
            "summary_line":    summary_line,
            "agent_context":   agent_context,
            "created_at":      created_at,
        }
    except Exception as e:
        _logger.debug(f"load_previous_context parse error: {e}")
        return None


def analyze_stock_full(symbol: str) -> tuple:
    """
    Phân tích dùng Vibe-Trading 13 engines + context agents.
    Trả về (msg, metadata) — metadata dùng để lưu DB qua save_signal() trong bot.py.

    Flow mới (Persistent Memory):
      0. load_previous_context() — tải bối cảnh phân tích trước từ DB
      1. Thu thập data song song
      2. Chạy 13 Vibe engines (với prev_context injected)
      3. Tổng hợp verdict
      4. Format message (có dòng "Lần trước..." nếu có context)
      5. Return metadata để bot.py lưu DB
    """
    # ── 0. Load bối cảnh phân tích trước ─────────────────────────────────────
    prev_ctx = load_previous_context(symbol)

    # 1. Thu thập data song song
    with ThreadPoolExecutor(max_workers=5) as ex:
        f_price     = ex.submit(get_price_data,    symbol, 500)
        f_market    = ex.submit(get_market_data)
        f_news      = ex.submit(get_news_data,     symbol)
        f_commodity = ex.submit(get_commodity_data, symbol)
        f_macro     = ex.submit(get_vn_macro_data)
        price_data     = f_price.result()
        market_data    = f_market.result()
        news_data      = f_news.result()
        commodity_data = f_commodity.result()
        macro_data     = f_macro.result()

    if not price_data["success"]:
        return f"Khong lay duoc du lieu gia {symbol}: {price_data['error']}", None

    try:
        ind = compute_indicators(price_data["df"])
        ind["symbol"] = symbol
    except Exception as e:
        return f"Loi tinh indicators {symbol}: {e}", None

    # 2. Chạy 13 Vibe-Trading engines
    # Nếu có context cũ, truyền vào engines để chúng có thể tham chiếu
    if not _VIBE_AVAILABLE:
        return "vibe_skills.py chua duoc deploy cung project.", None
    try:
        vibe = _run_vibe_agents(symbol, price_data["df"])
    except Exception as e:
        return f"Loi Vibe-Trading engines: {e}", None

    # 3. Tổng hợp verdict (13 Vibe + context agents)
    try:
        verdict = _vibe_verdict(vibe, ind, market_data, news_data,
                                     commodity_data=commodity_data,
                                     macro_data=macro_data)
    except Exception as e:
        return f"Loi tong hop verdict: {e}", None

    # ── So sánh với verdict cũ (chỉ khi signal chưa hết hạn) ────────────────
    verdict_change_note = ""
    if prev_ctx and not prev_ctx.get("expired"):
        old_v   = prev_ctx["verdict_label"]
        new_v   = verdict["verdict_label"]
        old_pct = prev_ctx["confidence_pct"] or 0
        new_pct = verdict["confidence_pct"]
        # Phát hiện đảo chiều
        _BULL = {"DONG THUAN MUA", "NGHIENG MUA"}
        _BEAR = {"DONG THUAN BAN", "NGHIENG BAN"}
        if old_v in _BULL and new_v in _BEAR:
            verdict_change_note = "⚠️  ĐẢO CHIỀU: MUA → BÁN"
        elif old_v in _BEAR and new_v in _BULL:
            verdict_change_note = "✅  ĐẢO CHIỀU: BÁN → MUA"
        elif old_v == new_v and abs(new_pct - old_pct) >= 15:
            direction = "manh len" if new_pct > old_pct else "yeu di"
            verdict_change_note = f"📈  Cung chieu nhung {direction} ({old_pct}% → {new_pct}%)"
        elif old_v != new_v:
            verdict_change_note = f"🔄  Thay doi: {old_v} → {new_v}"

    # 4. Format message
    fv    = verdict["verdict_label"]
    emoji = {"DONG THUAN MUA": "🟢", "NGHIENG MUA": "🟢",
             "DONG THUAN BAN": "🔴", "NGHIENG BAN": "🔴",
             "TRUNG LAP": "🟡"}.get(fv, "🟡")
    vnindex_str = (f"{market_data['vnindex']:,} ({market_data['change_5d']:+.1f}% 5D)"
                   if market_data.get("success") else "N/A")
    now = datetime.now().strftime("%d/%m %H:%M")

    # Đảm bảo vnindex_str nhất quán với MarketRegime detail
    # Lấy từ market_data đã được _vibe_verdict xử lý
    if market_data.get("success"):
        _vn = market_data["vnindex"]
        _c5 = market_data["change_5d"]
        vnindex_str = f"{_vn:,.1f} ({_c5:+.1f}% 5D)"
    else:
        vnindex_str = "N/A"
    msg = _build_vibe_message(emoji, symbol, now, ind, vnindex_str, vibe, verdict)

    # ── Thêm dòng "Lần trước" vào đầu message ────────────────────────────────
    if prev_ctx:
        if prev_ctx.get("expired"):
            # Signal cũ đã hết hạn — hiển thị thông báo nhưng không compare verdict
            memory_block = prev_ctx["summary_line"] + "\n"
        else:
            memory_lines = [prev_ctx["summary_line"]]
            if verdict_change_note:
                memory_lines.append(f"   {verdict_change_note}")
            memory_block = "\n".join(memory_lines) + "\n"
        # Chèn vào ngay sau dòng header (dòng 1: emoji + symbol + date)
        lines = msg.split("\n")
        insert_idx = next(
            (i + 1 for i, l in enumerate(lines)
             if symbol.upper() in l and any(c in l for c in ("🟢","🔴","🟡"))),
            1
        )
        lines.insert(insert_idx, memory_block)
        msg = "\n".join(lines)

    # 5. Metadata cho DB
    agent_verdicts = {k: ("MUA" if v > 0 else "BAN" if v < 0 else "TRUNG LAP")
                      for k, v in verdict["signals"].items()}
    metadata = {
        "verdict":        verdict,
        "ind":            ind,
        "agent_verdicts": agent_verdicts,
        # macro_v là dict để local_swarm.py có thể gọi .get()
        "macro_v":        {
            "label":         ("THUAN LOI" if verdict["signals"].get("MarketRegime", 0) > 0
                              else "RUI RO" if verdict["signals"].get("MarketRegime", 0) < 0
                              else "TRUNG TINH"),
            "market_regime": ("UPTREND"    if verdict["signals"].get("MarketRegime", 0) > 0
                              else "DOWNTREND" if verdict["signals"].get("MarketRegime", 0) < 0
                              else "SIDEWAY"),
            "signal":        verdict["signals"].get("MarketRegime", 0),
            "macro_label":   (macro_data.get("label", "MACRO_NEUTRAL")
                              if isinstance(macro_data, dict) and macro_data.get("success")
                              else "MACRO_NEUTRAL"),
            "detail":        (macro_data.get("detail", "")
                              if isinstance(macro_data, dict) else ""),
        },
        "vibe_result":    vibe,
        "commodity_data": commodity_data,
        "macro_data":     macro_data,
        "prev_context":   prev_ctx,            # Persistent Memory: bối cảnh cũ
        "verdict_change": verdict_change_note, # Persistent Memory: note thay đổi
    }
    return msg, metadata
