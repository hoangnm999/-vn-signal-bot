import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_URL     = "https://api.deepseek.com/v1/chat/completions"
VNAI_API_KEY     = os.environ.get("VNAI_API_KEY", "")

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

def get_market_data() -> dict:
    """
    Lấy market regime data dùng _price_from_entrade.
    Cache kết quả trong ngày — Entrade trả rỗng ngoài giờ GD
    nhưng dữ liệu ngày hôm qua vẫn hợp lệ để phân tích trend.
    """
    global _market_cache
    from datetime import date as _date

    # Trả cache nếu đã fetch thành công hôm nay
    today = _date.today().isoformat()
    if _market_cache["data"] and _market_cache["date"] == today:
        return _market_cache["data"]

    symbols_to_try = [
        ("E1VFVN30", "ETF VN30"),
        ("FUEVFVND", "ETF VNFIN"),
        ("FUESSVFL", "ETF VN30 2"),
        ("VCB",      "Proxy VCB"),
        ("HPG",      "Proxy HPG"),
        ("FPT",      "Proxy FPT"),
        ("MWG",      "Proxy MWG"),
        ("TCB",      "Proxy TCB"),
    ]

    last_err = "Tat ca symbols fail"
    for symbol, label in symbols_to_try:
        try:
            # Dùng lại _price_from_entrade đã hoạt động ổn thay vì parse thủ công
            df = _price_from_entrade(symbol, 90)
            if df is None or df.empty:
                last_err = f"{symbol}: empty dataframe"
                continue

            close = df["close"].astype(float)
            if len(close) < 5:
                last_err = f"{symbol}: khong du du lieu ({len(close)} bars)"
                continue

            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
            curr = float(close.iloc[-1])
            p5   = float(close.iloc[-5])  if len(close) >= 5  else curr
            p20  = float(close.iloc[-20]) if len(close) >= 20 else curr

            result = {
                "success":    True,
                "vnindex":    round(curr, 2),
                "change_5d":  round((curr - p5)  / p5  * 100, 2),
                "change_20d": round((curr - p20) / p20 * 100, 2),
                "above_ma20": bool(curr > ma20),
                "above_ma50": bool(curr > ma50) if ma50 else None,
                "ma20":       round(float(ma20), 2),
                "proxy":      label,
            }
            # Lưu cache
            _market_cache["data"] = result
            _market_cache["date"] = today
            return result
        except Exception as e:
            last_err = f"{symbol}: {str(e)[:80]}"
            continue

    # Nếu tất cả fail nhưng có cache cũ → dùng cache (ngoài giờ GD)
    if _market_cache["data"]:
        cached = dict(_market_cache["data"])
        cached["proxy"] = cached.get("proxy","") + " [cached]"
        return cached

    return {"success": False, "error": last_err}


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
        """Fireant JSON API - mạng trader VN"""
        try:
            r = requests.get(
                f"https://restv2.fireant.vn/posts?symbol={symbol}&limit=8&offset=0",
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=8,
            )
            if r.status_code == 200:
                posts = r.json()
                results = []
                for p in posts[:6]:
                    content = re.sub(r"<[^>]+>", "", p.get("content", "") or "").strip()
                    content = " ".join(content.split())
                    if content and len(content) > 15:
                        results.append(f"[Fireant] {content[:150]}")
                return results
        except Exception:
            pass
        return []

    # Chạy song song tất cả RSS sources
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
    ]
    source_names = [
        "GoogleNews", "CafeF", "VnExpress", "VnEconomy",
        "NDH", "DanTri", "ThanhNien", "TNCK", "Fireant",
    ]

    all_headlines, source_results = [], {}
    with ThreadPoolExecutor(max_workers=9) as ex:
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
# 6 AGENTS + VERDICT
# ══════════════════════════════════════════════════════════════════════════════

def run_trend_agent(symbol, ind):
    sys_p = ("Ban la chuyen gia phan tich xu huong gia co phieu Viet Nam. "
             "Tra loi CHINH XAC theo format sau, KHONG them gi ngoai format:\n"
             "NHAN XET: <toi da 150 ky tu, neu nhan xet chinh ve trend/MA/momentum>\n"
             "KET LUAN: TANG hoac GIAM hoac SIDEWAY")
    user = (f"Co phieu: {symbol}\n"
            f"Gia: {ind['current_price']:,} | 1W: {ind['change_1w_pct']:+.1f}% | 1M: {ind['change_1m_pct']:+.1f}%\n"
            f"MA20: {ind['ma20']:,} | MA50: {ind.get('ma50') or 'N/A'}\n"
            f"RSI(14): {ind['rsi']} | MACD Hist: {ind['macd_hist']:+.4f}\n"
            "Phan tich ngan. KET LUAN: TANG/GIAM/SIDEWAY")
    txt = call_deepseek(sys_p, user, max_tokens=120)
    u = txt.upper()
    v = "TĂNG" if "TANG" in u or "TĂNG" in u else "GIẢM" if "GIAM" in u or "GIẢM" in u else "SIDEWAY"
    return txt, v


def run_volume_agent(symbol, ind):
    sys_p = ("Ban la chuyen gia phan tich khoi luong giao dich chung khoan VN. "
             "QUY TAC BAT BUOC: Neu Volume/TB20 < 0.5x thi KET LUAN PHAI la NGHI NGO. "
             "Tra loi CHINH XAC theo format sau:\n"
             "NHAN XET: <toi da 150 ky tu, nhan xet ve volume va xac nhan xu huong>\n"
             "KET LUAN: XAC NHAN hoac NGHI NGO hoac PHAN KY")
    bb_pos = round((ind['current_price'] / ind['bb_mid'] - 1) * 100, 1)
    user = (f"Co phieu: {symbol}\n"
            f"Volume hom nay / TB20: {ind['volume_ratio']}x\n"
            f"RSI: {ind['rsi']} | Gia vs BB_mid: {bb_pos:+.1f}%\n"
            f"BB Upper: {ind['bb_upper']:,} | BB Lower: {ind['bb_lower']:,}\n"
            "KET LUAN: XAC NHAN/NGHI NGO/PHAN KY")
    txt = call_deepseek(sys_p, user, max_tokens=120)
    v = "XÁC NHẬN" if "XÁC NHẬN" in txt.upper() else "PHÂN KỲ" if "PHÂN KỲ" in txt.upper() else "NGHI NGỜ"
    # Guard cứng: volume quá thấp không thể XÁC NHẬN dù LLM nói gì
    if ind["volume_ratio"] < 0.5 and v == "XÁC NHẬN":
        v = "NGHI NGỜ"
        txt += "\n[Auto-override: Vol < 0.5x TB20, khong du de xac nhan]"
    return txt, v


def run_risk_agent(symbol, ind):
    sys_p = ("Ban la chuyen gia quan ly rui ro co phieu VN. "
             "Tra loi CHINH XAC theo format sau:\n"
             "NHAN XET: <toi da 150 ky tu, danh gia rui ro dua tren BB/RSI/support>\n"
             "KET LUAN: THAP hoac TRUNG BINH hoac CAO")
    user = (f"Co phieu: {symbol}\n"
            f"Gia: {ind['current_price']:,}\n"
            f"BB Upper: {ind['bb_upper']:,} | BB Lower: {ind['bb_lower']:,}\n"
            f"Khang cu 20D: {ind['resistance_20d']:,} | Ho tro 20D: {ind['support_20d']:,}\n"
            f"RSI: {ind['rsi']}\n"
            "KET LUAN: THAP/TRUNG BINH/CAO")
    txt = call_deepseek(sys_p, user, max_tokens=120)
    v = "THẤP" if "THAP" in txt.upper() or "THẤP" in txt.upper() else "CAO" if "CAO" in txt.upper() else "TRUNG BÌNH"
    return txt, v


# ── Lookup table tên công ty — tránh LLM hallucinate khi chỉ có mã 3 ký tự ──
_COMPANY_NAMES = {
    # Ngân hàng
    "VCB":"Vietcombank", "BID":"BIDV", "CTG":"VietinBank",
    "TCB":"Techcombank", "MBB":"MB Bank", "VPB":"VPBank",
    "ACB":"ACB", "HDB":"HDBank", "STB":"Sacombank",
    "LPB":"LienVietPostBank", "SHB":"SHB", "MSB":"MSB",
    "TPB":"TPBank", "VIB":"VIB", "EIB":"Eximbank",
    # Bất động sản
    "VIC":"Vingroup (tap doan da nganh: BDS, o to VinFast, ban le)",
    "VHM":"Vinhomes (BDS Vingroup)", "VRE":"Vincom Retail",
    "NVL":"Novaland", "PDR":"Phat Dat Real Estate",
    "DXG":"Dat Xanh Group", "KDH":"Khang Dien",
    "BCM":"Becamex IDC",
    # Thép
    "HPG":"Hoa Phat Group (thep)", "HSG":"Hoa Sen Group (ton thep)",
    "NKG":"Thep Nam Kim", "TLH":"Thep Tien Len",
    # Công nghệ
    "FPT":"Tap doan FPT (CNTT, vien thong, giao duc)",
    "CMG":"Tap doan CMC", "VGI":"Viettel Global",
    # Bán lẻ
    "MWG":"The Gioi Di Dong (ban le dien may, duoc pham Long Chau)",
    "PNJ":"PNJ (vang bac da quy)", "FRT":"FPT Retail",
    "DGW":"Digiworld",
    # Thực phẩm
    "VNM":"Vinamilk", "SAB":"Sabeco (bia)", "MSN":"Masan Group",
    "KDC":"Kido Group (thuc pham)", "MCH":"Masan Consumer",
    # Năng lượng
    "GAS":"PV GAS (khi dot)", "PLX":"Petrolimex (xang dau)",
    "POW":"PV Power (dien luc)", "PVD":"PV Drilling (khoan dau khi)",
    "BSR":"Binh Son Refining (loc hoa dau)",
    # Hàng không / Cảng / Vận tải
    "HVN":"Vietnam Airlines", "ACV":"ACV (cang hang khong)",
    "GMD":"Gemadept (cang bien logistics)",
    "HAH":"Hai An Transport (van tai container)",
    "DVP":"Cang Dinh Vu - CTCP Cang Dinh Vu (cang container tai Hai Phong, KHONG PHAI duoc pham)",
    "PHP":"Cang Hai Phong", "VSC":"Container Viet Nam",
    "TCO":"Cang Tan Cuong (TP HCM)",
    # Chứng khoán
    "SSI":"SSI Securities", "VND":"VNDirect",
    "HCM":"HSC (Chung khoan TP HCM)", "VCI":"Viet Capital Securities",
    "MBS":"MB Securities",
    # Điện / Tiện ích
    "REE":"REE Corporation (co dien lanh, dien)", "PC1":"PC1 Group (xay lap dien)",
    "GEG":"Gia Lai Electricity",
    # Xây dựng
    "CTD":"Coteccons (xay dung)", "HBC":"Hoa Binh Construction",
    "VCS":"Vicostone (da thach anh nhan tao)",
    # Nông nghiệp / Thủy sản
    "VHC":"Vinh Hoan (xuat khau ca tra)",
    "ANV":"Nam Viet (ca tra)", "IDI":"IDI Corporation (thuy san)",
    # Dược phẩm (NOTE: DVP ≠ dược — DVP là cảng)
    "DHG":"DHG Pharma (duoc Hau Giang)",
    "IMP":"Imexpharm", "DMC":"Domesco",
    "DVN":"Davipharm - Tong Cong ty Duoc Viet Nam (ma DVN, KHONG PHAI DVP)",
    "TRA":"Traphaco (duoc pham)",
    # Hóa chất / Phân bón
    "DGC":"Duc Giang Chemicals - Tap doan Hoa chat Duc Giang (hoa chat, photpho, phan bon)",
    "DCM":"Phan bon Dau khi Ca Mau (dam ure)",
    "DPM":"Phan bon va Hoa chat Dau khi (dam ure Phu My)",
    "CSV":"Hoa chat Viet Nam - CTCP Hoa chat Co ban Mien Nam",
    "SFG":"Phan bon Sieu Viet (phan bon)",
    # Cao su / Nông sản
    "PHR":"Cao su Phuoc Hoa", "DPR":"Cao su Dong Phu",
    "TRC":"Cao su Tay Ninh", "HRC":"Cao su Hoa Binh",
    "HAG":"Hoang Anh Gia Lai (nong nghiep, bat dong san)",
    "BAF":"BA F (chan nuoi heo)",
    # Dệt may / Giày dép
    "MSH":"May Song Hong", "TCM":"Det may Thanh Cong",
    "TNG":"May TNG Thai Nguyen", "VGT":"Vinatex (det may)",
    "EVE":"Everpia (chan ga goi dem)",
    # Giáo dục / Khác
    "ELC":"Elcom (cong nghe thong tin)",
    "FOX":"Fao Viet Nam (thuc pham chuc nang)",
    "VOS":"Van tai bien Viet Nam",
    "SGN":"Saigon Ground Services (phuc vu mat dat san bay)",
    "ASM":"Sao Mai Group (thuy san, BDS, nang luong)",
    "IDC":"KCN Viet Nam - Viglacera IDC",
    "KBC":"Khu cong nghiep Kinh Bac",
    "SZC":"KCN Sonadezi Chau Duc",
}

def _get_company_name(symbol: str) -> str:
    """Lấy tên công ty từ lookup table, fallback vnstock."""
    if symbol in _COMPANY_NAMES:
        return _COMPANY_NAMES[symbol]
    try:
        from vnstock import Vnstock
        info = Vnstock().stock(symbol=symbol, source="VCI").company.overview()
        if info is not None and not info.empty:
            row = info.iloc[0]
            name = str(row.get("short_name", symbol))
            ind  = str(row.get("industry_name", ""))
            return f"{name} - nganh {ind}" if ind else name
    except Exception:
        pass
    return f"Ma {symbol} (chua co trong lookup table — can xac minh ten cong ty)"


def run_fundamental_agent(symbol, fund):
    if not fund.get("success"):
        return f"Khong lay duoc fundamental: {fund.get('error','')}", "TRUNG TINH"

    company_name = _get_company_name(symbol)

    # Nếu có data thực từ API
    if fund.get("source") in ("vnstock KBS", "Entrade"):
        sys_p = ("Ban la chuyen gia phan tich co ban co phieu VN. "
                 "Tra loi CHINH XAC theo format sau:\n"
                 "NHAN XET: <toi da 180 ky tu, danh gia PE/PB/ROE/tang truong>\n"
                 "KET LUAN: TOT hoac TRUNG TINH hoac YEU")
        user = (f"Co phieu: {symbol} | Ten cong ty: {company_name}\n"
                f"PE: {fund['pe']} | PB: {fund['pb']} | ROE: {fund['roe']}%\n"
                f"EPS: {fund['eps']:,} | Tang truong DT: {fund['revenue_growth']:+.1f}%\n"
                f"Tang truong LN: {fund['profit_growth']:+.1f}% | No/Von: {fund['debt_equity']}\n"
                "KET LUAN: TOT/TRUNG TINH/YEU")
        txt = call_deepseek(sys_p, user, max_tokens=150)
        v = "TOT" if "TOT" in txt.upper() else "YEU" if "YEU" in txt.upper() else "TRUNG TINH"
        return txt, v

    # Không có API data — dùng LLM knowledge
    sys_p = ("Ban la chuyen gia tai chinh ve cac cong ty niem yet HOSE/HNX. "
             "TUYET DOI KHONG nham lan cong ty. Neu khong biet chinh xac, ghi 'Khong du thong tin'. "
             "Tra loi CHINH XAC theo format sau:\n"
             "NHAN XET: <toi da 200 ky tu, nganh nghe + vi the + KQKD gan nhat>\n"
             "KET LUAN: TOT hoac TRUNG TINH hoac YEU")
    user = (f"Co phieu: {symbol} — Ten chinh xac: {company_name}\n"
            f"Khong co real-time data. Phan tich dua tren kien thuc co san ve '{company_name}'.\n"
            "KET LUAN: TOT/TRUNG TINH/YEU")
    txt = call_deepseek(sys_p, user, max_tokens=200)
    v = "TOT" if "TOT" in txt.upper() else "YEU" if "YEU" in txt.upper() else "TRUNG TINH"
    return txt + f" [LLM knowledge - {company_name}]", v


def run_smart_money_agent(symbol, foreign):
    if not foreign.get("success"):
        return f"Khong lay duoc smart money data: {foreign.get('error','')}", "TRUNG LAP"

    is_proxy = foreign.get("source") == "volume_proxy"
    data_note = "(uoc tinh tu volume-price pattern)" if is_proxy else "(real foreign flow data)"

    sys_p = ("Ban la chuyen gia theo doi dong tien thong minh TTCK VN. "
             "Tra loi CHINH XAC theo format sau:\n"
             "NHAN XET: <toi da 150 ky tu, xu huong dong tien 1D/5D/20D>\n"
             "KET LUAN: MUA RONG hoac BAN RONG hoac TRUNG LAP")
    user = (f"Co phieu: {symbol} {data_note}\n"
            f"Net buy/sell hom nay: {foreign['net_today']:+.2f} ty\n"
            f"Net buy/sell 5 phien: {foreign['net_5d']:+.2f} ty\n"
            f"Net buy/sell 20 phien: {foreign['net_20d']:+.2f} ty\n"
            "KET LUAN: MUA RONG/BAN RONG/TRUNG LAP")
    txt = call_deepseek(sys_p, user, max_tokens=150)
    u = txt.upper()
    v = "MUA RONG" if "MUA RONG" in u else "BAN RONG" if "BAN RONG" in u else "TRUNG LAP"
    suffix = " [du lieu uoc tinh]" if is_proxy else ""
    return txt + suffix, v


_MACRO_KEYWORDS = [
    # Lãi suất
    "lai suat", "lãi suất", "ls dieu hanh", "ls co ban", "lai suat tiet kiem",
    "lai suat cho vay", "ngan hang nha nuoc", "sbv", "nhnn",
    # Tỷ giá
    "ty gia", "tỷ giá", "usd", "vnd", "usd/vnd", "gia dollar", "gia usd",
    "ty gia ngoai te", "ngoai te",
    # Fed / quốc tế
    "fed", "federal reserve", "lai suat my", "powell", "fomc",
    # Tín dụng / tiền tệ
    "tin dung", "tín dụng", "room tin dung", "cung tien", "bom tien", "hut tien",
    "thi truong mo", "nghiep vu thi truong mo",
    # Lạm phát / kinh tế
    "lam phat", "lạm phát", "cpi", "gdp", "tang truong kinh te",
    "chinh sach tien te", "vi mo", "kinh te vi mo",
    # Trái phiếu
    "trai phieu", "trái phiếu", "trai phieu chinh phu", "lai suat trai phieu",
    # Ngân hàng / tín dụng
    "von hoa ngan hang", "an toan von", "basel", "car",
]

def _extract_macro_headlines(headlines: list) -> list:
    """Lọc các headline liên quan đến macro/tiền tệ từ danh sách chung"""
    macro = []
    for h in headlines:
        h_lower = h.lower()
        if any(kw in h_lower for kw in _MACRO_KEYWORDS):
            macro.append(h)
    return macro


def run_macro_agent(news: dict) -> tuple:
    """
    Macro Agent — đánh giá môi trường vĩ mô từ RSS headlines có sẵn.
    Nếu không đủ tin → fetch thêm RSS VnEconomy macro.
    """
    import urllib.parse, re

    RSS_HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; Feedfetcher-Google)",
        "Accept": "application/rss+xml, application/xml, */*",
    }

    def _parse_rss_titles(url):
        try:
            r = requests.get(url, headers=RSS_HEADERS, timeout=8)
            if r.status_code != 200:
                return []
            text  = r.text
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", text, re.DOTALL)
            if not titles:
                titles = re.findall(r"<title>(.*?)</title>", text, re.DOTALL)
            titles = [re.sub(r"<[^>]+>", "", t).strip() for t in titles if len(t.strip()) > 15]
            return titles[1:8]   # bỏ title đầu (tên feed)
        except Exception:
            return []

    # Bước 1: filter từ headlines chung
    macro_headlines = _extract_macro_headlines(news.get("headlines", [])) if news.get("success") else []

    # Bước 2: nếu < 3 tin → fetch thêm RSS macro riêng
    if len(macro_headlines) < 3:
        extra = []
        macro_rss_sources = [
            "https://vneconomy.vn/tai-chinh.rss",
            "https://vneconomy.vn/ngan-hang.rss",
            "https://cafef.vn/rss/vi-mo-dau-tu.rss",
            (f"https://news.google.com/rss/search?q="
             f"{urllib.parse.quote('lai suat ty gia SBV ngan hang nha nuoc')}"
             f"&hl=vi&gl=VN&ceid=VN:vi"),
        ]
        for url in macro_rss_sources:
            titles = _parse_rss_titles(url)
            extra.extend([f"[Macro] {t}" for t in titles])
            if len(macro_headlines) + len(extra) >= 5:
                break
        macro_headlines = macro_headlines + extra

    if not macro_headlines:
        return "Khong co tin tuc macro trong phien nay — moi truong vi mo chua xac dinh", "TRUNG TINH"

    headlines_text = "\n".join(f"- {h}" for h in macro_headlines[:12])
    sys_p = (
        "Ban la chuyen gia kinh te vi mo Viet Nam. "
        "Danh gia moi truong vi mo anh huong den TTCK VN. "
        "Tra loi CHINH XAC theo format sau:\n"
        "NHAN XET: <toi da 150 ky tu, tom tat tin hieu lai suat/ty gia/Fed/SBV>\n"
        "KET LUAN: THUAN LOI hoac TRUNG TINH hoac RUI RO"
    )
    user = (
        f"Tin tuc vi mo ({len(macro_headlines)} tin):\n"
        f"{headlines_text}\n\n"
        "KET LUAN: THUAN LOI hoac TRUNG TINH hoac RUI RO"
    )
    txt = call_deepseek(sys_p, user, max_tokens=180)
    u = txt.upper()
    v = ("THUAN LOI" if "THUAN LOI" in u else
         "RUI RO"    if "RUI RO"    in u else "TRUNG TINH")
    return txt, v


def run_news_agent(symbol, news):
    if not news.get("success"):
        return f"Khong lay duoc tin tuc: {news.get('error','')}", "TRUNG TINH"
    headlines_text = "\n".join(f"- {h}" for h in news["headlines"])
    src_summary    = news.get("source_summary", {})
    active_sources = [k for k, v in src_summary.items() if v > 0]
    source_note    = f"Nguon: {', '.join(active_sources) or 'khong co'} | Tong: {news.get('total',0)} muc"
    sys_p = ("Ban la chuyen gia phan tich sentiment da nguon cho TTCK VN. "
             "Bao cao CTCK co trong so CAO HON y kien ca nhan. "
             "Tra loi CHINH XAC theo format sau:\n"
             "NHAN XET: <toi da 180 ky tu, tom tat sentiment chinh tu cac nguon>\n"
             "KET LUAN: TICH CUC hoac TRUNG TINH hoac TIEU CUC")
    user = (f"Co phieu: {symbol}\n{source_note}\n\n"
            f"Noi dung:\n{headlines_text}\n\n"
            "KET LUAN: TICH CUC hoac TRUNG TINH hoac TIEU CUC")
    txt = call_deepseek(sys_p, user, max_tokens=200)
    u = txt.upper()
    if "TICH CUC" in u or "TÍCH CỰC" in u or "POSITIVE" in u:
        v = "TICH CUC"
    elif "TIEU CUC" in u or "TIÊU CỰC" in u or "NEGATIVE" in u:
        v = "TIEU CUC"
    else:
        v = "TRUNG TINH"
    return txt, v


def run_market_regime_agent(market):
    if not market.get("success"):
        err = market.get('error', '')
        return f"Khong lay duoc data thi truong: {err}", "UNKNOWN"
    sys_p = ("Ban la chuyen gia phan tich vi mo TTCK VN. "
             "Tra loi CHINH XAC theo format sau:\n"
             "NHAN XET: <toi da 150 ky tu, danh gia market regime dua tren MA/trend>\n"
             "KET LUAN: UPTREND hoac DOWNTREND hoac SIDEWAYS")
    ma50_str = (f"Tren MA50: {'Co' if market.get('above_ma50') else 'Khong'}"
                if market.get("above_ma50") is not None else "MA50: N/A")
    user = (f"VN-Index proxy: {market['vnindex']:,}\n"
            f"Thay doi 5 phien: {market['change_5d']:+.2f}%\n"
            f"Thay doi 20 phien: {market['change_20d']:+.2f}%\n"
            f"Tren MA20: {'Co' if market['above_ma20'] else 'Khong'} (MA20: {market['ma20']:,})\n"
            f"{ma50_str}\n"
            "KET LUAN: UPTREND hoac DOWNTREND hoac SIDEWAYS")
    txt = call_deepseek(sys_p, user, max_tokens=150)
    v = "UPTREND" if "UPTREND" in txt.upper() else "DOWNTREND" if "DOWNTREND" in txt.upper() else "SIDEWAYS"
    return txt, v


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
    Format Action Plan theo verdict:
    - MUA/BAN: hiển thị đầy đủ Entry/TP/SL/RR
    - TRUNG LAP/PHAN BAC: không thể quyết định mua/bán tại thời điểm này
    """
    if "MUA" in verdict_label or "BAN" in verdict_label:
        action = "MUA" if "MUA" in verdict_label else "BAN"
        return (
            f"ACTION PLAN ({action}):\n"
            f"  Entry:  {_fmt_price(ap['entry_low'])} - {_fmt_price(ap['entry_high'])}\n"
            f"  Target: {_fmt_price(ap['tp'])} ({ap['tp_pct']:+.1f}%)\n"
            f"  SL:     {_fmt_price(ap['sl'])} ({ap['sl_pct']:+.1f}%)\n"
            f"  R:R   = 1:{ap['rr']}\n"
            f"\n  Vung theo doi them: {_fmt_price(ap['sl'])} - {_fmt_price(ap['entry_low'])}"
        )
    else:
        # TRUNG LAP hoac PHAN BAC
        return (
            f"ACTION PLAN:\n"
            f"  Khong the quyet dinh mua/ban tai thoi diem nay.\n"
            f"  Vung co the xem xet mua khi co tin hieu:\n"
            f"    Ho tro: {_fmt_price(ap['sl'])} - {_fmt_price(ap['entry_low'])}\n"
            f"  Muc TP neu vao duoc: {_fmt_price(ap['tp'])} ({ap['tp_pct']:+.1f}%)\n"
            f"  Dieu kien can them: RSI < 70 + Volume xac nhan + Market Regime UPTREND"
        )


def run_verdict_agent(symbol, verdicts, ind):
    """
    Verdict agent mới — trả về dict với 4 keys:
      verdict_label : ĐỒNG THUẬN MUA / ĐỒNG THUẬN BÁN / TRUNG LẬP / PHẢN BÁC
      confidence    : float 1.0-10.0
      summary       : 2-3 câu tổng hợp lý do
      action_plan   : dict {entry, tp, sl, rr}
      negative      : 1 dòng lý do không nên vào lệnh ngay
    """
    # ── Tính Confidence Score từ sự đồng thuận agents ────────────────────────
    # Mỗi agent vote: +1 nếu bullish, -1 nếu bearish, 0 nếu neutral
    # NOTE: so sánh KHÔNG DẤU vì agents trả về không dấu
    def _v(val): return val.upper().strip()

    votes = {
        "trend":       1 if _v(verdicts["trend"]) in ("TANG","TĂNG")
                  else -1 if _v(verdicts["trend"]) in ("GIAM","GIẢM") else 0,
        "volume":      1 if _v(verdicts["volume"]) in ("XAC NHAN","XÁC NHẬN")
                  else -1 if _v(verdicts["volume"]) in ("PHAN KY","PHÂN KỲ") else 0,
        "risk":        1 if _v(verdicts["risk"]) in ("THAP","THẤP")
                  else -1 if _v(verdicts["risk"]) in ("CAO") else 0,
        "fundamental": 1 if _v(verdicts["fundamental"]) in ("TOT","TỐT")
                  else -1 if _v(verdicts["fundamental"]) in ("YEU","YẾU") else 0,
        "smart_money": 1 if _v(verdicts["smart_money"]) in ("MUA RONG","MUA RÒNG")
                  else -1 if _v(verdicts["smart_money"]) in ("BAN RONG","BÁN RÒNG") else 0,
        "news":        1 if _v(verdicts["news"]) in ("TICH CUC","TÍCH CỰC","POSITIVE")
                  else -1 if _v(verdicts["news"]) in ("TIEU CUC","TIÊU CỰC","NEGATIVE") else 0,
        "market":      0 if _v(verdicts["market"]) == "UNKNOWN"   # agent fail → bỏ qua
                  else 1 if _v(verdicts["market"]) == "UPTREND"
                  else -1 if _v(verdicts["market"]) == "DOWNTREND" else 0,
    }
    market_weight = 0.0 if _v(verdicts["market"]) == "UNKNOWN" else 1.5

    # Market Regime + Smart Money có trọng số 1.5x
    # market_weight = 0 nếu agent fail (UNKNOWN), không ảnh hưởng score
    weighted_score = (
        votes["trend"]       * 1.0 +
        votes["volume"]      * 1.0 +
        votes["risk"]        * 1.0 +
        votes["fundamental"] * 1.0 +
        votes["smart_money"] * 1.5 +
        votes["news"]        * 1.0 +
        votes["market"]      * market_weight
    )
    max_score = 6.0 + 1.5 + market_weight  # 7.5 khi market OK, 6.0 khi UNKNOWN
    # Scale về 1-10, tâm 5.5
    confidence = round(5.5 + (weighted_score / max_score) * 4.5, 1)
    confidence = max(1.0, min(10.0, confidence))

    active_agents = 7 if _v(verdicts["market"]) != "UNKNOWN" else 6
    bull_count = sum(1 for v in votes.values() if v == 1)
    bear_count = sum(1 for v in votes.values() if v == -1)

    # ── Verdict label ─────────────────────────────────────────────────────────
    if weighted_score >= 3.0:
        verdict_label = "ĐỒNG THUẬN MUA"
    elif weighted_score <= -3.0:
        verdict_label = "ĐỒNG THUẬN BÁN"
    elif weighted_score >= 1.0:
        verdict_label = "TRUNG LẬP"
    elif weighted_score <= -1.0:
        verdict_label = "PHẢN BÁC"
    else:
        verdict_label = "TRUNG LẬP"

    # ── Action Plan — tính dựa trên ATR để tránh SL vô lý ──────────────────
    price   = ind["current_price"]
    support = ind["support_20d"]
    resist  = ind["resistance_20d"]
    bb_low  = ind["bb_lower"]
    bb_up   = ind["bb_upper"]

    # ATR proxy: dùng (bb_upper - bb_lower) / 4 làm đơn vị rủi ro
    atr_proxy = (bb_up - bb_low) / 4.0
    if atr_proxy <= 0:
        atr_proxy = price * 0.03   # fallback: 3% giá

    # SL: giá hiện tại - 1.5x ATR (không dùng support_20d vì có thể quá xa)
    sl_atr = round(price - 1.5 * atr_proxy, 0)
    # Nếu support_20d gần hơn (trong vòng 15% giá) thì dùng support
    if support >= price * 0.85:
        sl = round(min(sl_atr, support * 0.985), 0)
    else:
        sl = sl_atr   # support quá xa → chỉ dùng ATR

    # TP: kháng cự 20D hoặc BB Upper (lấy cái nào gần hơn và hợp lý)
    tp_candidate = max(resist, bb_up)
    # Nếu TP quá gần (< 2% upside) → dùng 2x ATR từ giá hiện tại
    if tp_candidate < price * 1.02:
        tp = round(price + 2.0 * atr_proxy, 0)
    else:
        tp = round(tp_candidate * 0.995, 0)

    # Entry: vùng hợp lý quanh giá hiện tại (không kéo về support xa)
    # Dùng 2 chữ số thập phân thay vì round(..., 0) để tránh entry_low = entry_high
    entry_low  = round(price * 0.99, 2)    # -1% giá hiện tại
    entry_high = round(price * 1.01, 2)    # +1% giá hiện tại

    # Đảm bảo TP luôn lớn hơn entry_high ít nhất 2%
    if tp <= entry_high * 1.02:
        tp = round(entry_high * 1.03 + 2.0 * atr_proxy, 2)

    # R:R
    risk   = price - sl
    reward = tp - price
    rr     = round(reward / risk, 1) if risk > 0 else 0

    # Cap SL pct để không hiển thị SL vô lý > 15%
    sl_pct = round((sl - price) / price * 100, 1)
    if sl_pct < -15:
        sl     = round(price * 0.92, 2)   # force SL tối đa -8%
        sl_pct = round((sl - price) / price * 100, 1)
        risk   = price - sl
        rr     = round(reward / risk, 1) if risk > 0 else 0

    action_plan = {
        "entry_low":  round(entry_low, 2),
        "entry_high": round(entry_high, 2),
        "tp":         round(tp, 2),
        "sl":         round(sl, 2),
        "tp_pct":     round((tp - price) / price * 100, 1),
        "sl_pct":     sl_pct,
        "rr":         rr,
    }

    # ── LLM: Summary + Negative Filter ───────────────────────────────────────
    sys_p = (
        "Ban la Chief Investment Analyst cho TTCK Viet Nam. "
        "Nhiem vu: viet 2 doan ngan gon bang tieng Viet khong dau (ASCII safe). "
        "DOAN 1 — SUMMARY: 2 cau ly do chinh cho ket luan. "
        "DOAN 2 — NEGATIVE: 1 cau ngan ve rui ro/ly do KHONG nen vao lenh ngay bay gio. "
        "Format bat buoc:\n"
        "SUMMARY: <noi dung>\n"
        "NEGATIVE: <noi dung>"
    )
    user = (
        f"Co phieu: {symbol} | Gia: {price:,}\n"
        f"Ket luan: {verdict_label} | Confidence: {confidence}/10\n"
        f"Agents dong thuan: {bull_count}/7 bullish, {bear_count}/7 bearish\n\n"
        f"Xu huong: {verdicts['trend']} | Volume: {verdicts['volume']} | "
        f"Rui ro: {verdicts['risk']}\n"
        f"Fundamental: {verdicts['fundamental']} | Smart Money: {verdicts['smart_money']}\n"
        f"News: {verdicts['news']} | Market Regime: {verdicts['market']}\n\n"
        f"VN-Index proxy: {verdicts['market']}\n"
        "Viet SUMMARY 2 cau va NEGATIVE 1 cau theo format yeu cau."
    )
    llm_txt = call_deepseek(sys_p, user, max_tokens=300)

    # Parse SUMMARY và NEGATIVE từ LLM output
    summary  = ""
    negative = ""
    for line in llm_txt.splitlines():
        line = line.strip()
        if line.upper().startswith("SUMMARY:"):
            summary = line[8:].strip()
        elif line.upper().startswith("NEGATIVE:"):
            negative = line[9:].strip()

    # Fallback nếu LLM không theo format
    if not summary:
        summary = llm_txt[:200].strip()
    if not negative:
        negative = f"Kiem tra lai khi thi truong xac nhan ro rang hon (confidence {confidence}/10)"

    return {
        "verdict_label": verdict_label,
        "confidence":    confidence,
        "bull_count":    bull_count,
        "bear_count":    bear_count,
        "active_agents": active_agents,
        "summary":       summary,
        "action_plan":   action_plan,
        "negative":      negative,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def analyze_stock(symbol: str) -> str:
    # 1. Thu thập data song song
    with ThreadPoolExecutor(max_workers=5) as ex:
        f_price   = ex.submit(get_price_data,        symbol, 90)
        f_fund    = ex.submit(get_fundamental_data,  symbol)
        f_foreign = ex.submit(get_foreign_flow_data, symbol)
        f_market  = ex.submit(get_market_data)
        f_news    = ex.submit(get_news_data,         symbol)
        price_data   = f_price.result()
        fund_data    = f_fund.result()
        foreign_data = f_foreign.result()
        market_data  = f_market.result()
        news_data    = f_news.result()

    if not price_data["success"]:
        return f"❌ Không lấy được dữ liệu giá {symbol}: {price_data['error']}"

    try:
        ind = compute_indicators(price_data["df"])
    except Exception as e:
        return f"❌ Lỗi tính indicators {symbol}: {e}"

    # 2. Chạy 8 agent song song (thêm Macro agent)
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            f1 = ex.submit(run_trend_agent,          symbol, ind)
            f2 = ex.submit(run_volume_agent,         symbol, ind)
            f3 = ex.submit(run_risk_agent,           symbol, ind)
            f4 = ex.submit(run_fundamental_agent,    symbol, fund_data)
            f5 = ex.submit(run_smart_money_agent,    symbol, foreign_data)
            f6 = ex.submit(run_news_agent,           symbol, news_data)
            f7 = ex.submit(run_market_regime_agent,          market_data)
            f8 = ex.submit(run_macro_agent,                  news_data)
            trend_txt,  trend_v  = f1.result()
            volume_txt, volume_v = f2.result()
            risk_txt,   risk_v   = f3.result()
            fund_txt,   fund_v   = f4.result()
            smart_txt,  smart_v  = f5.result()
            news_txt,   news_v   = f6.result()
            market_txt, market_v = f7.result()
            macro_txt,  macro_v  = f8.result()
    except Exception as e:
        return f"❌ Lỗi khi chạy agents {symbol}: {e}"

    # 3. Verdict
    try:
        verdict = run_verdict_agent(symbol, {
            "trend": trend_v, "volume": volume_v, "risk": risk_v,
            "fundamental": fund_v, "smart_money": smart_v,
            "news": news_v, "market": market_v,
        }, ind)
    except Exception as e:
        return f"❌ Lỗi verdict agent {symbol}: {e}"

    # 4. Format — dùng _build_message_safe (Lớp 2: smart truncate)
    final_v  = verdict["verdict_label"]
    conf     = verdict["confidence"]
    ap       = verdict["action_plan"]
    summary  = verdict["summary"]
    negative = verdict["negative"]
    bull     = verdict["bull_count"]
    bear     = verdict["bear_count"]
    n_agents = verdict.get("active_agents", 7)

    emoji = {"ĐỒNG THUẬN MUA": "🟢", "ĐỒNG THUẬN BÁN": "🔴",
             "PHẢN BÁC": "🔴", "TRUNG LẬP": "🟡"}.get(final_v, "🟡")
    vnindex_str = (f"{market_data['vnindex']:,} ({market_data['change_5d']:+.1f}% 5D)"
                   if market_data.get("success") else "N/A")
    now = datetime.now().strftime("%d/%m %H:%M")

    return _build_message_safe(
        emoji, symbol, now, ind, vnindex_str,
        trend_v, volume_v, risk_v, fund_v, smart_v, news_v, market_v, macro_v,
        trend_txt, volume_txt, risk_txt, fund_txt, smart_txt, news_txt, market_txt, macro_txt,
        final_v, conf, bull, bear, n_agents, summary, ap, negative,
    )


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


def analyze_stock_full(symbol: str) -> tuple:
    """
    Giống analyze_stock() nhưng trả về tuple (msg, metadata).
    metadata dùng để lưu DB qua save_signal() trong bot.py.
    metadata = None nếu có lỗi.
    """
    # 1. Thu thập data song song
    with ThreadPoolExecutor(max_workers=5) as ex:
        f_price   = ex.submit(get_price_data,        symbol, 90)
        f_fund    = ex.submit(get_fundamental_data,  symbol)
        f_foreign = ex.submit(get_foreign_flow_data, symbol)
        f_market  = ex.submit(get_market_data)
        f_news    = ex.submit(get_news_data,         symbol)
        price_data   = f_price.result()
        fund_data    = f_fund.result()
        foreign_data = f_foreign.result()
        market_data  = f_market.result()
        news_data    = f_news.result()

    if not price_data["success"]:
        return f"Khong lay duoc du lieu gia {symbol}: {price_data['error']}", None

    try:
        ind = compute_indicators(price_data["df"])
    except Exception as e:
        return f"Loi tinh indicators {symbol}: {e}", None

    # 2. Chạy 8 agent song song
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            f1 = ex.submit(run_trend_agent,        symbol, ind)
            f2 = ex.submit(run_volume_agent,       symbol, ind)
            f3 = ex.submit(run_risk_agent,         symbol, ind)
            f4 = ex.submit(run_fundamental_agent,  symbol, fund_data)
            f5 = ex.submit(run_smart_money_agent,  symbol, foreign_data)
            f6 = ex.submit(run_news_agent,         symbol, news_data)
            f7 = ex.submit(run_market_regime_agent,        market_data)
            f8 = ex.submit(run_macro_agent,                news_data)
            trend_txt,  trend_v  = f1.result()
            volume_txt, volume_v = f2.result()
            risk_txt,   risk_v   = f3.result()
            fund_txt,   fund_v   = f4.result()
            smart_txt,  smart_v  = f5.result()
            news_txt,   news_v   = f6.result()
            market_txt, market_v = f7.result()
            macro_txt,  macro_v  = f8.result()
    except Exception as e:
        return f"Loi khi chay agents {symbol}: {e}", None

    agent_verdicts = {
        "trend": trend_v, "volume": volume_v, "risk": risk_v,
        "fundamental": fund_v, "smart_money": smart_v,
        "news": news_v, "market": market_v,
    }

    # 3. Verdict
    try:
        verdict = run_verdict_agent(symbol, agent_verdicts, ind)
    except Exception as e:
        return f"Loi verdict agent {symbol}: {e}", None

    # 4. Format — dùng _build_message_safe (Lớp 2: smart truncate)
    final_v  = verdict["verdict_label"]
    conf     = verdict["confidence"]
    ap       = verdict["action_plan"]
    summary  = verdict["summary"]
    negative = verdict["negative"]
    bull     = verdict["bull_count"]
    bear     = verdict["bear_count"]
    n_agents = verdict.get("active_agents", 7)

    emoji = {"ĐỒNG THUẬN MUA": "🟢", "ĐỒNG THUẬN BÁN": "🔴",
             "PHẢN BÁC": "🔴", "TRUNG LẬP": "🟡"}.get(final_v, "🟡")
    vnindex_str = (f"{market_data['vnindex']:,} ({market_data['change_5d']:+.1f}% 5D)"
                   if market_data.get("success") else "N/A")
    now = datetime.now().strftime("%d/%m %H:%M")

    msg = _build_message_safe(
        emoji, symbol, now, ind, vnindex_str,
        trend_v, volume_v, risk_v, fund_v, smart_v, news_v, market_v, macro_v,
        trend_txt, volume_txt, risk_txt, fund_txt, smart_txt, news_txt, market_txt, macro_txt,
        final_v, conf, bull, bear, n_agents, summary, ap, negative,
    )

    metadata = {
        "verdict":        verdict,
        "ind":            ind,
        "agent_verdicts": agent_verdicts,
        "macro_v":        macro_v,
    }
    return msg, metadata
