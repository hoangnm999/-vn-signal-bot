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
    """Thử lấy ratio từ vnstock KBS/VCI, nhiều period để tránh 404"""
    from vnstock import Vnstock
    for src in ["KBS", "VCI"]:
        for period in ["quarter", "annual"]:
            try:
                stock = Vnstock().stock(symbol=symbol, source=src)
                ratio = stock.finance.ratio(period=period)
                if ratio is not None and not ratio.empty:
                    return ratio
            except Exception:
                continue
    return None


def _fundamental_from_entrade(symbol: str) -> dict:
    """Fallback: lấy fundamental từ Entrade company info API"""
    url = f"https://services.entrade.com.vn/dnse-order-service/companies/{symbol}/financial-ratio"
    r = requests.get(url, headers=_BROWSER_HEADERS, timeout=10)
    r.raise_for_status()
    data = r.json()
    # Entrade trả về list, lấy phần tử mới nhất
    if isinstance(data, list) and data:
        d = data[0]
    elif isinstance(data, dict):
        d = data
    else:
        raise ValueError("Entrade fundamental data rỗng")

    def _g(key, d2=0):
        v = d.get(key)
        try:
            return float(v) if v is not None else d2
        except Exception:
            return d2

    return {
        "success":        True,
        "pe":             round(_g("pe"), 2),
        "pb":             round(_g("pb"), 2),
        "roe":            round(_g("roe") * 100, 2),
        "eps":            round(_g("eps"), 0),
        "revenue_growth": round(_g("revenueGrowth", 0) * 100, 2),
        "profit_growth":  round(_g("profitGrowth", 0) * 100, 2),
        "debt_equity":    round(_g("debtEquity", 0), 2),
    }


def get_fundamental_data(symbol: str) -> dict:
    # Thử vnstock trước
    try:
        ratio = _fundamental_from_vnstock(symbol)
        if ratio is not None and not ratio.empty:
            latest = ratio.iloc[0]
            cols = list(latest.index) if hasattr(latest, "index") else []

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
            }
    except Exception:
        pass

    # Fallback Entrade
    try:
        return _fundamental_from_entrade(symbol)
    except Exception as e:
        return {"success": False, "error": f"Không có dữ liệu fundamental: {str(e)[:80]}"}


def _foreign_from_entrade(symbol: str, days: int) -> dict:
    """Lấy foreign flow từ Entrade — thử nhiều endpoint"""
    import time
    to_ts   = int(time.time())
    from_ts = to_ts - days * 86400

    # Endpoint v2 ohlcs có foreign columns
    url = (
        f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
        f"?from={from_ts}&to={to_ts}&symbol={symbol}&resolution=D"
    )
    r = requests.get(url, headers=_BROWSER_HEADERS, timeout=12)
    r.raise_for_status()
    data = r.json()

    # Tìm cột foreign buy/sell trong response
    # Entrade OHLCV thường có: t, o, h, l, c, v, fBuyVol, fSellVol, fBuyVal, fSellVal
    buy_key  = next((k for k in data if "buy" in k.lower() and "val" in k.lower()), None)
    sell_key = next((k for k in data if "sell" in k.lower() and "val" in k.lower()), None)

    if not buy_key or not sell_key:
        # Không có foreign data trong OHLCV — thử endpoint riêng
        url2 = (
            f"https://services.entrade.com.vn/chart-api/v2/foreign-trading"
            f"?from={from_ts}&to={to_ts}&symbol={symbol}&resolution=D"
        )
        r2 = requests.get(url2, headers=_BROWSER_HEADERS, timeout=10)
        r2.raise_for_status()
        data = r2.json()
        buy_key  = next((k for k in data if "buy" in k.lower()), None)
        sell_key = next((k for k in data if "sell" in k.lower()), None)
        if not buy_key or not sell_key:
            raise ValueError(f"Không tìm thấy cột foreign trong response: {list(data.keys())[:8]}")

    buy_arr  = data[buy_key]  if isinstance(data[buy_key],  list) else [data[buy_key]]
    sell_arr = data[sell_key] if isinstance(data[sell_key], list) else [data[sell_key]]
    net_arr  = [float(b) - float(s) for b, s in zip(buy_arr, sell_arr)]

    return {
        "success":   True,
        "net_today": round(net_arr[-1] / 1e9, 2)           if net_arr else 0,
        "net_5d":    round(sum(net_arr[-5:])  / 1e9, 2)    if len(net_arr) >= 5  else 0,
        "net_20d":   round(sum(net_arr[-20:]) / 1e9, 2)    if len(net_arr) >= 20 else 0,
    }


def get_foreign_flow_data(symbol: str) -> dict:
    try:
        return _foreign_from_entrade(symbol, 30)
    except Exception as e:
        return {"success": False, "error": str(e)[:120]}


def _parse_market_close_series(data: dict) -> pd.Series:
    """Parse close price series từ OHLCV response"""
    for key in ["c", "close", "Close"]:
        if key in data and data[key]:
            return pd.Series(data[key], dtype=float)
    raise ValueError(f"Không tìm thấy close price, keys: {list(data.keys())}")


def get_market_data() -> dict:
    """VN-Index từ nhiều endpoint với fallback"""
    import time
    to_ts   = int(time.time())
    from_ts = to_ts - 90 * 86400  # 90 ngày để đủ MA50

    endpoints = [
        # Entrade index endpoint
        (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
         f"?from={from_ts}&to={to_ts}&symbol=VNINDEX&resolution=D"),
        # Entrade dùng type=index
        (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/index"
         f"?from={from_ts}&to={to_ts}&symbol=VNINDEX&resolution=D"),
        # Thử VN30 nếu VNINDEX fail
        (f"https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
         f"?from={from_ts}&to={to_ts}&symbol=VN30&resolution=D"),
    ]

    last_err = "Không kết nối được"
    for url in endpoints:
        try:
            r = requests.get(url, headers=_BROWSER_HEADERS, timeout=12)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                continue
            data = r.json()
            close = _parse_market_close_series(data)
            if close.empty:
                continue

            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
            curr = float(close.iloc[-1])
            p5   = float(close.iloc[-5])  if len(close) >= 5  else curr
            p20  = float(close.iloc[-20]) if len(close) >= 20 else curr

            return {
                "success":    True,
                "vnindex":    round(curr, 2),
                "change_5d":  round((curr - p5)  / p5  * 100, 2),
                "change_20d": round((curr - p20) / p20 * 100, 2),
                "above_ma20": bool(curr > ma20),
                "above_ma50": bool(curr > ma50) if ma50 else None,
                "ma20":       round(float(ma20), 2),
            }
        except Exception as e:
            last_err = str(e)
            continue

    return {"success": False, "error": last_err}


# ══════════════════════════════════════════════════════════════════════════════
# NEWS — 12 nguồn song song
# ══════════════════════════════════════════════════════════════════════════════

def get_news_data(symbol: str) -> dict:
    """
    Thu thập tin tức từ RSS feeds — không bị block bởi Cloudflare.
    RSS hoạt động từ datacenter IP, scraping HTML thì không.
    """
    import re
    import xml.etree.ElementTree as ET

    RSS_HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; RSS reader)",
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }

    def parse_rss(url: str, prefix: str, max_items: int = 6) -> list:
        """Parse RSS feed và trả về list tiêu đề"""
        try:
            r = requests.get(url, headers=RSS_HEADERS, timeout=8)
            if r.status_code != 200:
                return []
            text = r.text
            # Parse CDATA titles
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", text)
            if not titles:
                # Parse thường
                titles = re.findall(r"<title>(.*?)</title>", text)
            # Bỏ title đầu tiên (thường là tên feed)
            titles = [re.sub(r"<[^>]+>", "", t).strip() for t in titles[1:max_items+1]]
            return [f"[{prefix}] {t}" for t in titles if len(t) > 10]
        except Exception:
            return []

    def fetch_google_news_vn():
        """Google News RSS — hoạt động tốt từ datacenter"""
        results = []
        queries = [
            f"{symbol} cổ phiếu",
            f"{symbol} chứng khoán",
        ]
        for q in queries:
            import urllib.parse
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(q)}&hl=vi&gl=VN&ceid=VN:vi"
            results.extend(parse_rss(url, "GoogleNews", 5))
        return list(dict.fromkeys(results))[:8]  # dedup, max 8

    def fetch_cafef_rss():
        """CafeF RSS feed theo mã"""
        url = f"https://cafef.vn/rss/{symbol.lower()}.rss"
        results = parse_rss(url, "CafeF", 5)
        if not results:
            # Fallback RSS chứng khoán chung
            url2 = "https://cafef.vn/rss/thi-truong-chung-khoan.rss"
            results = parse_rss(url2, "CafeF", 4)
        return results

    def fetch_vnexpress_rss():
        """VnExpress RSS kinh doanh"""
        url = "https://vnexpress.net/rss/kinh-doanh.rss"
        all_items = parse_rss(url, "VnExpress", 20)
        # Lọc những tin có mention mã
        relevant = [t for t in all_items if symbol in t.upper()]
        return relevant[:4] if relevant else all_items[:3]

    def fetch_vneconomy_rss():
        """VnEconomy RSS"""
        url = "https://vneconomy.vn/chung-khoan.rss"
        all_items = parse_rss(url, "VnEconomy", 20)
        relevant = [t for t in all_items if symbol in t.upper()]
        return relevant[:4] if relevant else all_items[:2]

    def fetch_fireant_api():
        """Fireant API — thường không bị block"""
        try:
            url = f"https://restv2.fireant.vn/posts?symbol={symbol}&limit=8&offset=0"
            r = requests.get(url, timeout=8, headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            })
            if r.status_code == 200:
                posts = r.json()
                results = []
                for p in posts[:6]:
                    content = re.sub(r"<[^>]+>", "", p.get("content", "") or "").strip()
                    content = re.sub(r"\s+", " ", content)
                    if content and len(content) > 15:
                        results.append(f"[Fireant] {content[:150]}")
                return results
        except Exception:
            pass
        return []

    def fetch_thanhnien_rss():
        """Thanh Niên RSS kinh tế"""
        url = "https://thanhnien.vn/rss/tai-chinh-kinh-doanh.rss"
        all_items = parse_rss(url, "ThanhNien", 20)
        relevant = [t for t in all_items if symbol in t.upper()]
        return relevant[:3] if relevant else []

    def fetch_nhipsongdoanhnghiep_rss():
        """Nhịp sống doanh nghiệp RSS"""
        url = "https://nhipsongdoanhnghiep.laodong.vn/rss/chung-khoan.rss"
        return parse_rss(url, "NhipSong", 4)

    # Chạy song song tất cả nguồn RSS
    sources = [
        fetch_google_news_vn,
        fetch_cafef_rss,
        fetch_vnexpress_rss,
        fetch_vneconomy_rss,
        fetch_fireant_api,
        fetch_thanhnien_rss,
        fetch_nhipsongdoanhnghiep_rss,
    ]
    source_names = [
        "GoogleNews", "CafeF", "VnExpress", "VnEconomy",
        "Fireant", "ThanhNien", "NhipSong",
    ]

    all_headlines, source_results = [], {}
    with ThreadPoolExecutor(max_workers=7) as ex:
        futures = {ex.submit(fn): name for fn, name in zip(sources, source_names)}
        for future, name in futures.items():
            try:
                items = future.result()
                source_results[name] = len(items)
                all_headlines.extend(items)
            except Exception:
                source_results[name] = 0

    seen, clean_headlines = set(), []
    for h in all_headlines:
        h = h.strip()
        if h and h not in seen and len(h) > 15:
            seen.add(h)
            clean_headlines.append(h)

    if not clean_headlines:
        return {"success": False, "error": "Không lấy được tin tức (tất cả RSS đều fail)"}

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
    sys_p = ("Bạn là chuyên gia phân tích xu hướng giá cổ phiếu Việt Nam. "
             "Phân tích ngắn gọn, tập trung trend, MA, momentum. "
             "Dòng cuối PHẢI là — Kết luận: TĂNG hoặc GIẢM hoặc SIDEWAY")
    user = (f"Cổ phiếu: {symbol}\n"
            f"Giá: {ind['current_price']:,} | 1W: {ind['change_1w_pct']:+.1f}% | 1M: {ind['change_1m_pct']:+.1f}%\n"
            f"MA20: {ind['ma20']:,} | MA50: {ind.get('ma50') or 'N/A'}\n"
            f"RSI(14): {ind['rsi']} | MACD Hist: {ind['macd_hist']:+.4f}\n"
            "Phân tích 2-3 câu. Kết luận: TĂNG/GIẢM/SIDEWAY")
    txt = call_deepseek(sys_p, user)
    v = "TĂNG" if "TĂNG" in txt.upper() else "GIẢM" if "GIẢM" in txt.upper() else "SIDEWAY"
    return txt, v


def run_volume_agent(symbol, ind):
    sys_p = ("Bạn là chuyên gia phân tích khối lượng giao dịch chứng khoán VN. "
             "Phân tích volume để xác nhận hay bác bỏ xu hướng giá. "
             "Dòng cuối PHẢI là — Kết luận: XÁC NHẬN hoặc NGHI NGỜ hoặc PHÂN KỲ")
    bb_pos = round((ind['current_price'] / ind['bb_mid'] - 1) * 100, 1)
    user = (f"Cổ phiếu: {symbol}\n"
            f"Volume hôm nay / TB20: {ind['volume_ratio']}x\n"
            f"RSI: {ind['rsi']} | Giá vs BB_mid: {bb_pos:+.1f}%\n"
            f"BB Upper: {ind['bb_upper']:,} | BB Lower: {ind['bb_lower']:,}\n"
            "Phân tích 2-3 câu. Kết luận: XÁC NHẬN/NGHI NGỜ/PHÂN KỲ")
    txt = call_deepseek(sys_p, user)
    v = "XÁC NHẬN" if "XÁC NHẬN" in txt.upper() else "PHÂN KỲ" if "PHÂN KỲ" in txt.upper() else "NGHI NGỜ"
    return txt, v


def run_risk_agent(symbol, ind):
    sys_p = ("Bạn là chuyên gia quản lý rủi ro cổ phiếu VN. "
             "Đánh giá rủi ro dựa trên vị trí giá, Bollinger Bands, RSI, support/resistance. "
             "Dòng cuối PHẢI là — Kết luận: THẤP hoặc TRUNG BÌNH hoặc CAO")
    user = (f"Cổ phiếu: {symbol}\n"
            f"Giá: {ind['current_price']:,}\n"
            f"BB Upper: {ind['bb_upper']:,} | BB Lower: {ind['bb_lower']:,}\n"
            f"Kháng cự 20D: {ind['resistance_20d']:,} | Hỗ trợ 20D: {ind['support_20d']:,}\n"
            f"RSI: {ind['rsi']}\n"
            "Đánh giá rủi ro 2-3 câu. Kết luận: THẤP/TRUNG BÌNH/CAO")
    txt = call_deepseek(sys_p, user)
    v = "THẤP" if "THẤP" in txt.upper() else "CAO" if "CAO" in txt.upper() else "TRUNG BÌNH"
    return txt, v


def run_fundamental_agent(symbol, fund):
    if not fund.get("success"):
        return f"⚠️ Không lấy được fundamental: {fund.get('error','')}", "TRUNG TÍNH"
    sys_p = ("Bạn là chuyên gia phân tích cơ bản cổ phiếu VN. "
             "Đánh giá sức khỏe tài chính dựa trên PE, PB, ROE, tăng trưởng. "
             "Dòng cuối PHẢI là — Kết luận: TỐT hoặc TRUNG TÍNH hoặc YẾU")
    user = (f"Cổ phiếu: {symbol}\n"
            f"PE: {fund['pe']} | PB: {fund['pb']} | ROE: {fund['roe']}%\n"
            f"EPS: {fund['eps']:,} | Tăng trưởng DT: {fund['revenue_growth']:+.1f}%\n"
            f"Tăng trưởng LN: {fund['profit_growth']:+.1f}% | Nợ/Vốn: {fund['debt_equity']}\n"
            "Đánh giá 2-3 câu. Kết luận: TỐT/TRUNG TÍNH/YẾU")
    txt = call_deepseek(sys_p, user)
    v = "TỐT" if "TỐT" in txt.upper() else "YẾU" if "YẾU" in txt.upper() else "TRUNG TÍNH"
    return txt, v


def run_smart_money_agent(symbol, foreign):
    if not foreign.get("success"):
        return f"⚠️ Không lấy được foreign flow: {foreign.get('error','')}", "TRUNG LẬP"
    sys_p = ("Bạn là chuyên gia theo dõi dòng tiền thông minh TTCK VN. "
             "Phân tích hành vi khối ngoại để đánh giá xu hướng dòng tiền lớn. "
             "Dòng cuối PHẢI là — Kết luận: MUA RÒNG hoặc BÁN RÒNG hoặc TRUNG LẬP")
    user = (f"Cổ phiếu: {symbol}\n"
            f"Khối ngoại hôm nay: {foreign['net_today']:+.2f} tỷ\n"
            f"Khối ngoại 5 phiên: {foreign['net_5d']:+.2f} tỷ\n"
            f"Khối ngoại 20 phiên: {foreign['net_20d']:+.2f} tỷ\n"
            "Phân tích 2-3 câu. Kết luận: MUA RÒNG/BÁN RÒNG/TRUNG LẬP")
    txt = call_deepseek(sys_p, user)
    v = "MUA RÒNG" if "MUA RÒNG" in txt.upper() else "BÁN RÒNG" if "BÁN RÒNG" in txt.upper() else "TRUNG LẬP"
    return txt, v


def run_news_agent(symbol, news):
    if not news.get("success"):
        return f"⚠️ Không lấy được tin tức: {news.get('error','')}", "TRUNG TÍNH"
    headlines_text = "\n".join(f"- {h}" for h in news["headlines"])
    src_summary    = news.get("source_summary", {})
    active_sources = [k for k, v in src_summary.items() if v > 0]
    source_note    = f"Nguồn có dữ liệu: {', '.join(active_sources) or 'không có'} | Tổng: {news.get('total',0)} mục"
    sys_p = ("Bạn là chuyên gia phân tích sentiment đa nguồn cho TTCK VN. "
             "Dữ liệu từ: báo tài chính (CafeF/VnEconomy/Stockbiz/Vietstock), "
             "diễn đàn trader (f319), mạng xã hội trader (Fireant), "
             "báo cáo CTCK (VCSC/HSC/ACBS/VNDIRECT/SSI). "
             "Báo cáo CTCK có trọng số CAO HƠN ý kiến cá nhân. "
             "Dòng cuối PHẢI là — Kết luận: TÍCH CỰC hoặc TRUNG TÍNH hoặc TIÊU CỰC")
    user = (f"Cổ phiếu: {symbol}\n{source_note}\n\n"
            f"Nội dung thu thập:\n{headlines_text}\n\n"
            "Phân tích 3-4 câu. Kết luận: TÍCH CỰC/TRUNG TÍNH/TIÊU CỰC")
    txt = call_deepseek(sys_p, user, max_tokens=350)
    v = ("TÍCH CỰC" if "TÍCH CỰC" in txt.upper() else
         "TIÊU CỰC" if "TIÊU CỰC" in txt.upper() else "TRUNG TÍNH")
    return txt, v


def run_market_regime_agent(market):
    if not market.get("success"):
        return f"⚠️ Không lấy được data thị trường: {market.get('error','')}", "SIDEWAYS"
    sys_p = ("Bạn là chuyên gia phân tích vĩ mô TTCK VN. "
             "Xác định market regime dựa trên VN-Index. "
             "Dòng cuối PHẢI là — Kết luận: UPTREND hoặc DOWNTREND hoặc SIDEWAYS")
    ma50_str = (f"Trên MA50: {'Có' if market.get('above_ma50') else 'Không'}"
                if market.get("above_ma50") is not None else "MA50: N/A")
    user = (f"VN-Index: {market['vnindex']:,}\n"
            f"Thay đổi 5 phiên: {market['change_5d']:+.2f}%\n"
            f"Thay đổi 20 phiên: {market['change_20d']:+.2f}%\n"
            f"Trên MA20: {'Có' if market['above_ma20'] else 'Không'} (MA20: {market['ma20']:,})\n"
            f"{ma50_str}\n"
            "Đánh giá 2-3 câu. Kết luận: UPTREND/DOWNTREND/SIDEWAYS")
    txt = call_deepseek(sys_p, user)
    v = "UPTREND" if "UPTREND" in txt.upper() else "DOWNTREND" if "DOWNTREND" in txt.upper() else "SIDEWAYS"
    return txt, v


def run_verdict_agent(symbol, verdicts):
    sys_p = ("Bạn là Chief Investment Analyst, tổng hợp ý kiến từ 6 chuyên gia. "
             "Trọng số cao hơn cho Market Regime và Smart Money. "
             "Dòng cuối PHẢI là — Kết luận: ĐỒNG THUẬN MUA hoặc ĐỒNG THUẬN BÁN hoặc TRUNG LẬP hoặc PHẢN BÁC")
    user = (f"Cổ phiếu: {symbol}\n\n"
            f"Xu hướng: {verdicts['trend']}\nVolume: {verdicts['volume']}\n"
            f"Rủi ro: {verdicts['risk']}\nFundamental: {verdicts['fundamental']}\n"
            f"Smart Money: {verdicts['smart_money']}\nNews: {verdicts['news']}\n"
            f"Market Regime: {verdicts['market']}\n\n"
            "Tổng hợp 3-4 câu. Kết luận: ĐỒNG THUẬN MUA/ĐỒNG THUẬN BÁN/TRUNG LẬP/PHẢN BÁC")
    txt = call_deepseek(sys_p, user, max_tokens=400)
    u = txt.upper()
    v = ("ĐỒNG THUẬN MUA" if "ĐỒNG THUẬN MUA" in u else
         "ĐỒNG THUẬN BÁN" if "ĐỒNG THUẬN BÁN" in u else
         "PHẢN BÁC"       if "PHẢN BÁC"       in u else "TRUNG LẬP")
    return txt, v


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

    # 2. Chạy 7 agent song song
    try:
        with ThreadPoolExecutor(max_workers=7) as ex:
            f1 = ex.submit(run_trend_agent,          symbol, ind)
            f2 = ex.submit(run_volume_agent,         symbol, ind)
            f3 = ex.submit(run_risk_agent,           symbol, ind)
            f4 = ex.submit(run_fundamental_agent,    symbol, fund_data)
            f5 = ex.submit(run_smart_money_agent,    symbol, foreign_data)
            f6 = ex.submit(run_news_agent,           symbol, news_data)
            f7 = ex.submit(run_market_regime_agent,          market_data)
            trend_txt,  trend_v  = f1.result()
            volume_txt, volume_v = f2.result()
            risk_txt,   risk_v   = f3.result()
            fund_txt,   fund_v   = f4.result()
            smart_txt,  smart_v  = f5.result()
            news_txt,   news_v   = f6.result()
            market_txt, market_v = f7.result()
    except Exception as e:
        return f"❌ Lỗi khi chạy agents {symbol}: {e}"

    # 3. Verdict
    try:
        verdict_txt, final_v = run_verdict_agent(symbol, {
            "trend": trend_v, "volume": volume_v, "risk": risk_v,
            "fundamental": fund_v, "smart_money": smart_v,
            "news": news_v, "market": market_v,
        })
    except Exception as e:
        return f"❌ Lỗi verdict agent {symbol}: {e}"

    # 4. Format
    emoji = {"ĐỒNG THUẬN MUA": "🟢", "ĐỒNG THUẬN BÁN": "🔴",
             "PHẢN BÁC": "🔴", "TRUNG LẬP": "🟡"}.get(final_v, "🟡")
    vnindex_str = (f"{market_data['vnindex']:,} ({market_data['change_5d']:+.1f}% 5D)"
                   if market_data.get("success") else "N/A")
    now = datetime.now().strftime("%d/%m %H:%M")

    msg = f"""{emoji} *Phân tích {symbol}* — {now}

📊 *Dữ liệu:*
• Giá: `{ind['current_price']:,.0f}` | 1W: `{ind['change_1w_pct']:+.1f}%` | 1M: `{ind['change_1m_pct']:+.1f}%`
• RSI: `{ind['rsi']}` | MACD Hist: `{ind['macd_hist']:+.4f}` | Vol: `{ind['volume_ratio']}x`
• Hỗ trợ: `{ind['support_20d']:,.0f}` | Kháng cự: `{ind['resistance_20d']:,.0f}`
• VN-Index: `{vnindex_str}`

🤖 *6 Agents:*
  📈 Xu hướng: `{trend_v}`
  💧 Volume: `{volume_v}`
  ⚠️ Rủi ro: `{risk_v}`
  🏦 Fundamental: `{fund_v}`
  🐋 Smart Money: `{smart_v}`
  📰 News: `{news_v}`
  🌊 Market Regime: `{market_v}`

📈 *Xu hướng:* {trend_txt}

💧 *Volume:* {volume_txt}

⚠️ *Rủi ro:* {risk_txt}

🏦 *Fundamental:* {fund_txt}

🐋 *Smart Money:* {smart_txt}

📰 *News Sentiment:* {news_txt}

🌊 *Market Regime:* {market_txt}

{emoji} *KẾT LUẬN: {final_v}*
{verdict_txt}""".strip()

    return msg[:4000] + "\n_[Cắt bớt]_" if len(msg) > 4000 else msg


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
        vnindex_line = (f"\n🌊 VN-Index: `{market['vnindex']:,}` "
                        f"({market['change_5d']:+.1f}% 5D) — {regime}\n{'─'*28}\n")

    header = f"📋 *Scan Watchlist* — {now}\n{'─'*28}{vnindex_line}"
    footer = "\n\n_/check <MÃ> để phân tích sâu 6 agents_"
    return header + "\n".join(results) + footer
