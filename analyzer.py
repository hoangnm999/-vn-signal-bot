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
        price_chg  = close.pct_change()
        avg_vol    = volume.rolling(20).mean()
        vol_ratio  = volume / avg_vol

        # 5 phiên gần nhất
        recent_chg = price_chg.tail(5)
        recent_vr  = vol_ratio.tail(5)

        # .values để tránh Series không convert được sang scalar float
        buy_signal  = float(((recent_chg > 0) * recent_vr.clip(0, 3)).mean())
        sell_signal = float(((recent_chg < 0) * recent_vr.clip(0, 3)).mean())
        net_signal  = round(buy_signal - sell_signal, 3)

        avg_trade_val = float((volume * close).tail(20).mean()) / 1e9

        net_20d_series = ((price_chg > 0) * vol_ratio.clip(0, 3) -
                          (price_chg < 0) * vol_ratio.clip(0, 3)).tail(20)
        net_est_20d = round(float(net_20d_series.mean()) * avg_trade_val * 0.3, 2)
        net_est_5d  = round(net_signal * avg_trade_val * 0.3, 2)

        return {
            "success":   True,
            "net_today": round(net_signal * avg_trade_val * 0.15, 2),
            "net_5d":    net_est_5d,
            "net_20d":   net_est_20d,
            "source":    "volume_proxy",
            "note":      "uoc tinh tu volume-price pattern",
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


def get_market_data() -> dict:
    """
    Lấy market regime data.
    VNINDEX không phải stock => dùng ETF proxy:
    - E1VFVN30: ETF track VN30, niêm yết HOSE, đồng hành VNINDEX
    - FUEVFVND: ETF VNFIN LEAD
    - Fallback: dùng 1 bluechip (VCB) làm proxy tương đối
    """
    import time
    to_ts   = int(time.time())
    from_ts = to_ts - 90 * 86400

    base = "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock"
    # ETF proxy symbols - track thị trường chung
    symbols_to_try = [
        ("E1VFVN30", "ETF VN30"),
        ("FUEVFVND", "ETF VNFIN"),
        ("FUESSVFL", "ETF VN30"),
        ("VCB",      "Proxy VCB"),   # fallback cuối - bluechip đại diện
    ]

    last_err = "Tat ca symbols fail"
    for symbol, label in symbols_to_try:
        try:
            url = f"{base}?symbol={symbol}&resolution=D&from={from_ts}&to={to_ts}"
            r = requests.get(url, headers=_BROWSER_HEADERS, timeout=12)
            if r.status_code != 200:
                last_err = f"{symbol} HTTP {r.status_code}"
                continue

            raw = r.json()

            # Debug: log tất cả keys và sample values
            if isinstance(raw, dict):
                # Entrade keys: t=time, o=open, h=high, l=low, c=close, v=volume
                # Tìm key close price (c hoặc close)
                close_list = None
                for k in ["c", "close", "Close"]:
                    if k in raw and isinstance(raw[k], list):
                        vals = [float(x) for x in raw[k] if x is not None and str(x) not in ("0","0.0","")]
                        if len(vals) >= 5:
                            close_list = vals
                            break

                if not close_list:
                    last_err = f"{symbol}: khong co close data, keys={list(raw.keys())[:6]}"
                    continue

                close = pd.Series(close_list, dtype=float)
            else:
                last_err = f"{symbol}: response khong phai dict ({type(raw)})"
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
                "proxy":      label,
            }
        except Exception as e:
            last_err = f"{symbol}: {str(e)[:80]}"
            continue

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
        return f"Khong lay duoc fundamental: {fund.get('error','')}", "TRUNG TINH"

    # Nếu có data thực từ API
    if fund.get("source") in ("vnstock KBS", "Entrade"):
        sys_p = ("Ban la chuyen gia phan tich co ban co phieu VN. "
                 "Danh gia suc khoe tai chinh dua tren PE, PB, ROE, tang truong. "
                 "Dong cuoi PHAI la — Ket luan: TOT hoac TRUNG TINH hoac YEU")
        user = (f"Co phieu: {symbol}\n"
                f"PE: {fund['pe']} | PB: {fund['pb']} | ROE: {fund['roe']}%\n"
                f"EPS: {fund['eps']:,} | Tang truong DT: {fund['revenue_growth']:+.1f}%\n"
                f"Tang truong LN: {fund['profit_growth']:+.1f}% | No/Von: {fund['debt_equity']}\n"
                "Danh gia 2-3 cau. Ket luan: TOT/TRUNG TINH/YEU")
        txt = call_deepseek(sys_p, user)
        v = "TOT" if "TOT" in txt.upper() else "YEU" if "YEU" in txt.upper() else "TRUNG TINH"
        return txt, v

    # Không có API data — dùng LLM knowledge về công ty
    sys_p = ("Ban la chuyen gia tai chinh voi kien thuc sau ve cac cong ty niem yet tren HOSE/HNX Viet Nam. "
             "Dua tren kien thuc co san cua ban ve cong ty nay (ket qua kinh doanh, vi the nganh, xu huong gan day), "
             "hay danh gia suc khoe co ban. "
             "Luu y ro rang day la danh gia dua tren kien thuc LLM, khong phai real-time data. "
             "Dong cuoi PHAI la — Ket luan: TOT hoac TRUNG TINH hoac YEU")
    user = (f"Co phieu: {symbol}\n"
            "Khong co real-time financial API data. "
            "Dua tren kien thuc cua ban ve {symbol}: nganh nghe, lich su KQKD, vi the canh tranh, "
            "cac chi so tai chinh thuong thay, hay danh gia fundamental 2-3 cau. "
            "Neu khong biet ro thi noi ro la khong co du thong tin. "
            "Ket luan: TOT/TRUNG TINH/YEU")
    user = user.replace("{symbol}", symbol)
    txt = call_deepseek(sys_p, user)
    v = "TOT" if "TOT" in txt.upper() else "YEU" if "YEU" in txt.upper() else "TRUNG TINH"
    return txt + " [LLM knowledge, khong co real-time data]", v


def run_smart_money_agent(symbol, foreign):
    if not foreign.get("success"):
        return f"Khong lay duoc smart money data: {foreign.get('error','')}", "TRUNG LAP"

    is_proxy = foreign.get("source") == "volume_proxy"
    data_note = "(uoc tinh tu volume-price pattern)" if is_proxy else "(real foreign flow data)"

    sys_p = ("Ban la chuyen gia theo doi dong tien thong minh TTCK VN. "
             "Phan tich xu huong dong tien lon dua tren du lieu cung cap. "
             "Dong cuoi PHAI la — Ket luan: MUA RONG hoac BAN RONG hoac TRUNG LAP")
    user = (f"Co phieu: {symbol} {data_note}\n"
            f"Net buy/sell hom nay: {foreign['net_today']:+.2f} ty\n"
            f"Net buy/sell 5 phien: {foreign['net_5d']:+.2f} ty\n"
            f"Net buy/sell 20 phien: {foreign['net_20d']:+.2f} ty\n"
            "Phan tich xu huong dong tien 2-3 cau. Ket luan: MUA RONG/BAN RONG/TRUNG LAP")
    txt = call_deepseek(sys_p, user)
    u = txt.upper()
    v = "MUA RONG" if "MUA RONG" in u else "BAN RONG" if "BAN RONG" in u else "TRUNG LAP"
    suffix = " [du lieu uoc tinh]" if is_proxy else ""
    return txt + suffix, v


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
    votes = {
        "trend":      1 if verdicts["trend"] == "TĂNG" else -1 if verdicts["trend"] == "GIẢM" else 0,
        "volume":     1 if verdicts["volume"] == "XÁC NHẬN" else -1 if verdicts["volume"] == "PHÂN KỲ" else 0,
        "risk":       1 if verdicts["risk"] == "THẤP" else -1 if verdicts["risk"] == "CAO" else 0,
        "fundamental":1 if verdicts["fundamental"] in ("TOT", "TỐT") else -1 if verdicts["fundamental"] in ("YEU", "YẾU") else 0,
        "smart_money":1 if verdicts["smart_money"] == "MUA RONG" else -1 if verdicts["smart_money"] == "BAN RONG" else 0,
        "news":       1 if verdicts["news"] == "TÍCH CỰC" else -1 if verdicts["news"] == "TIÊU CỰC" else 0,
        "market":     1 if verdicts["market"] == "UPTREND" else -1 if verdicts["market"] == "DOWNTREND" else 0,
    }

    # Market Regime + Smart Money có trọng số 1.5x
    weighted_score = (
        votes["trend"]       * 1.0 +
        votes["volume"]      * 1.0 +
        votes["risk"]        * 1.0 +
        votes["fundamental"] * 1.0 +
        votes["smart_money"] * 1.5 +
        votes["news"]        * 1.0 +
        votes["market"]      * 1.5
    )
    max_score = 9.0  # tổng trọng số tối đa
    # Scale về 1-10, tâm 5.5
    confidence = round(5.5 + (weighted_score / max_score) * 4.5, 1)
    confidence = max(1.0, min(10.0, confidence))

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

    # ── Action Plan từ indicators có sẵn ─────────────────────────────────────
    price   = ind["current_price"]
    support = ind["support_20d"]
    resist  = ind["resistance_20d"]
    bb_low  = ind["bb_lower"]
    bb_up   = ind["bb_upper"]

    # Entry: giữa giá hiện tại và BB_lower (vùng hợp lý để chờ pullback)
    entry_low  = round(max(support, bb_low) * 1.005, 0)   # +0.5% trên support
    entry_high = round(price * 1.005, 0)                   # ngay trên giá hiện tại

    # SL: dưới support 20D hoặc BB Lower (lấy mức thấp hơn)
    sl = round(min(support, bb_low) * 0.985, 0)            # -1.5% dưới support

    # TP: kháng cự 20D hoặc BB Upper
    tp = round(max(resist, bb_up) * 0.995, 0)              # sát dưới kháng cự

    # R:R
    risk   = price - sl
    reward = tp - price
    rr     = round(reward / risk, 1) if risk > 0 else 0

    action_plan = {
        "entry_low":  int(entry_low),
        "entry_high": int(entry_high),
        "tp":         int(tp),
        "sl":         int(sl),
        "tp_pct":     round((tp - price) / price * 100, 1),
        "sl_pct":     round((sl - price) / price * 100, 1),
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
        verdict = run_verdict_agent(symbol, {
            "trend": trend_v, "volume": volume_v, "risk": risk_v,
            "fundamental": fund_v, "smart_money": smart_v,
            "news": news_v, "market": market_v,
        }, ind)
    except Exception as e:
        return f"❌ Lỗi verdict agent {symbol}: {e}"

    # 4. Format
    final_v  = verdict["verdict_label"]
    conf     = verdict["confidence"]
    ap       = verdict["action_plan"]
    summary  = verdict["summary"]
    negative = verdict["negative"]
    bull     = verdict["bull_count"]
    bear     = verdict["bear_count"]

    emoji = {"ĐỒNG THUẬN MUA": "🟢", "ĐỒNG THUẬN BÁN": "🔴",
             "PHẢN BÁC": "🔴", "TRUNG LẬP": "🟡"}.get(final_v, "🟡")
    vnindex_str = (f"{market_data['vnindex']:,} ({market_data['change_5d']:+.1f}% 5D)"
                   if market_data.get("success") else "N/A")
    now = datetime.now().strftime("%d/%m %H:%M")

    msg = f"""{emoji} Phan tich {symbol} — {now}

DU LIEU:
  Gia: {ind['current_price']:,.0f} | 1W: {ind['change_1w_pct']:+.1f}% | 1M: {ind['change_1m_pct']:+.1f}%
  RSI: {ind['rsi']} | MACD Hist: {ind['macd_hist']:+.4f} | Vol: {ind['volume_ratio']}x
  Ho tro: {ind['support_20d']:,.0f} | Khang cu: {ind['resistance_20d']:,.0f}
  VN-Index: {vnindex_str}

7 AGENTS:
  Xu huong:    {trend_v}
  Volume:      {volume_v}
  Rui ro:      {risk_v}
  Fundamental: {fund_v}
  Smart Money: {smart_v}
  News:        {news_v}
  Market:      {market_v}

CHI TIET AGENTS:
  [Xu huong] {trend_txt}
  [Volume] {volume_txt}
  [Rui ro] {risk_txt}
  [Fundamental] {fund_txt}
  [Smart Money] {smart_txt}
  [News] {news_txt}
  [Market] {market_txt}

{'='*30}
{emoji} KET LUAN: {final_v}
CONFIDENCE: {conf}/10 ({bull}/7 bullish, {bear}/7 bearish)

LY DO: {summary}

ACTION PLAN:
  Entry:  {ap['entry_low']:,} - {ap['entry_high']:,}
  Target: {ap['tp']:,} ({ap['tp_pct']:+.1f}%)
  SL:     {ap['sl']:,} ({ap['sl_pct']:+.1f}%)
  R:R   = 1:{ap['rr']}

CANH BAO: {negative}
{'='*30}""".strip()

    return msg[:4000] + "\n[Cat bot]" if len(msg) > 4000 else msg


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
