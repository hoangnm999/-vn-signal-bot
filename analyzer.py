import os
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# ── Vibe-Trading 10 engines (candlestick, ichimoku, smc, elliott, harmonic,
#    technical-basic, volatility, seasonal, cross-market, multi-factor) ───────
try:
    from vibe_skills import run_vibe_agents as _run_vibe
    _VIBE_AVAILABLE = True
except Exception:
    _VIBE_AVAILABLE = False

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

def get_market_data() -> dict:
    """
    Lấy market regime data dùng _price_from_entrade.
    Cache kết quả trong ngày — thread-safe với Lock.
    """
    global _market_cache
    from datetime import date as _date

    today = _date.today().isoformat()

    # Check cache trước (read không cần lock hoàn toàn an toàn)
    with _market_cache_lock:
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
            # Lưu cache thread-safe
            with _market_cache_lock:
                _market_cache["data"] = result
                _market_cache["date"] = today
            return result
        except Exception as e:
            last_err = f"{symbol}: {str(e)[:80]}"
            continue

    # Nếu tất cả fail nhưng có cache cũ → dùng cache (ngoài giờ GD)
    with _market_cache_lock:
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
    return header + detail_section + vibe_section + footer


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

    bull_count  = sum(1 for v in active_agents.values() if v == 1)
    bear_count  = sum(1 for v in active_agents.values() if v == -1)
    net_score   = bull_count - bear_count  # đơn giản, không weighted tùy ý

    # ── Verdict: đếm thực tế, không threshold ma thuật ────────────────────────
    # Cần đa số rõ ràng (> 50% active agents đồng thuận 1 chiều)
    majority = n_active / 2  # ngưỡng đa số tự nhiên

    if bull_count > majority and bear_count == 0:
        verdict_label = "DONG THUAN MUA"   # đồng thuận tuyệt đối bullish
    elif bull_count > majority:
        verdict_label = "NGHIENG MUA"      # đa số bullish nhưng có phản đối
    elif bear_count > majority and bull_count == 0:
        verdict_label = "DONG THUAN BAN"   # đồng thuận tuyệt đối bearish
    elif bear_count > majority:
        verdict_label = "NGHIENG BAN"      # đa số bearish nhưng có phản đối
    else:
        verdict_label = "TRUNG LAP"        # không có đa số rõ ràng

    # Confidence = % agents đồng thuận chiều chính (0-100, không scale tùy ý)
    dominant = max(bull_count, bear_count)
    confidence_pct = round(dominant / n_active * 100) if n_active > 0 else 0

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

    action_plan = {
        "entry_low":  entry_low,
        "entry_high": entry_high,
        "tp":         tp,
        "sl":         sl,
        "tp_pct":     round((tp - price) / price * 100, 1),
        "sl_pct":     sl_pct,
        "rr":         rr,
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

def _vibe_verdict(vibe: dict, ind: dict, market: dict, news: dict) -> dict:
    """
    Tổng hợp verdict từ kết quả Vibe-Trading engines.
    Thêm 2 context signals: Market Regime + News Sentiment.
    """
    signals = dict(vibe["signals"])  # 7 Vibe signals

    # ── Context signal: Market Regime ─────────────────────────────────────
    if market.get("success"):
        chg_5d  = market["change_5d"]
        above   = market.get("above_ma20", False)
        chg_20d = market.get("change_20d", 0)
        bull_mkt = sum([chg_5d > 1, chg_20d > 3, above])
        bear_mkt = sum([chg_5d < -1, chg_20d < -3, not above])
        if bull_mkt >= 2:   signals["MarketRegime"] = 1
        elif bear_mkt >= 2: signals["MarketRegime"] = -1
        else:               signals["MarketRegime"] = 0
    else:
        signals["MarketRegime"] = 0

    # ── Context signal: News Sentiment ────────────────────────────────────
    if news.get("success") and news.get("headlines"):
        text_all = " ".join(news["headlines"]).lower()
        pos = sum(1 for kw in [
            "tang","loi nhuan","tang truong","ket qua tot","tich cuc",
            "mua vao","co tuc","hop dong lon","vuot ke hoach","dot pha",
        ] if kw in text_all)
        neg = sum(1 for kw in [
            "giam","thua lo","sut giam","tieu cuc","ban ra","rui ro",
            "canh bao","vi pham","no xau","mat thanh khoan","sa thai",
        ] if kw in text_all)
        if pos > neg * 1.5:   signals["NewsSentiment"] = 1
        elif neg > pos * 1.5: signals["NewsSentiment"] = -1
        else:                 signals["NewsSentiment"] = 0
    else:
        signals["NewsSentiment"] = 0

    # ── Tổng hợp vote ─────────────────────────────────────────────────────
    n_active = len(signals)
    bull = sum(1 for v in signals.values() if v > 0)
    bear = sum(1 for v in signals.values() if v < 0)
    majority = n_active / 2

    if bull > majority and bear == 0:
        verdict_label = "DONG THUAN MUA"
    elif bull > majority:
        verdict_label = "NGHIENG MUA"
    elif bear > majority and bull == 0:
        verdict_label = "DONG THUAN BAN"
    elif bear > majority:
        verdict_label = "NGHIENG BAN"
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

    action_plan = {
        "entry_low": e_lo, "entry_high": e_hi,
        "tp": tp, "sl": sl,
        "tp_pct": round((tp - price) / price * 100, 1),
        "sl_pct": sl_pct, "rr": rr,
    }

    # ── LLM summary (chỉ viết 2 câu, không vote) ─────────────────────────
    bull_agents = [k for k,v in signals.items() if v > 0]
    bear_agents = [k for k,v in signals.items() if v < 0]
    sys_p = (
        "Viet 2 dong tom tat phan tich co phieu bang tieng Viet khong dau. "
        "Chi dua tren du lieu duoc cung cap, KHONG them thong tin moi. "
        "Format:\nSUMMARY: <2 cau mo ta ket qua agents>\nNEGATIVE: <1 cau rui ro chinh>"
    )
    user_p = (
        f"Co phieu: {ind.get('symbol','?')} | Gia: {_fmt_price(price)}\n"
        f"Vibe-Trading 7 engines + 2 context: {bull}/{n_active} bullish, {bear}/{n_active} bearish\n"
        f"Agents ung ho: {', '.join(bull_agents) or 'khong co'}\n"
        f"Agents phan doi: {', '.join(bear_agents) or 'khong co'}\n"
        f"RSI={ind['rsi']}, Vol={ind['volume_ratio']}x, "
        f"Market={'UP' if signals.get('MarketRegime',0)>0 else 'DOWN' if signals.get('MarketRegime',0)<0 else 'SIDE'}"
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
        summary = f"Vibe-Trading {bull}/{n_active} bullish, {bear}/{n_active} bearish."
        neg_txt = ""

    if not neg_txt:
        parts = []
        if ind["volume_ratio"] < 0.7: parts.append(f"vol thap {ind['volume_ratio']}x")
        if ind["rsi"] > 68: parts.append(f"RSI={ind['rsi']} overbought")
        if bear > 0: parts.append(f"{bear} agents phan doi")
        neg_txt = "Luu y: " + ", ".join(parts) if parts else "Theo doi them tin hieu xac nhan."

    return {
        "verdict_label": verdict_label, "confidence_pct": conf_pct,
        "bull_count": bull, "bear_count": bear, "active_agents": n_active,
        "signals": signals,
        "summary": summary, "negative": neg_txt, "action_plan": action_plan,
    }


def _build_vibe_message(
    emoji, symbol, now, ind, vnindex_str,
    vibe: dict, verdict: dict,
    char_limit: int = 4000
) -> str:
    """Build Telegram message với kết quả Vibe-Trading 7+2 agents."""
    ap  = verdict["action_plan"]
    fv  = verdict["verdict_label"]
    pct = verdict["confidence_pct"]
    bull = verdict["bull_count"]
    bear = verdict["bear_count"]
    n    = verdict["active_agents"]

    # Emoji từng signal
    def _sig_emoji(v): return "🟢" if v > 0 else "🔴" if v < 0 else "⚪"

    sigs = verdict["signals"]
    agents_lines = "\n".join(
        f"  {_sig_emoji(v)} {name:<16} {'MUA' if v>0 else 'BAN' if v<0 else 'TL'}"
        for name, v in sigs.items()
    )

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
    )

    verdict_block = (
        f"\n{'='*32}\n"
        f"{emoji} KET LUAN: {fv}\n"
        f"   {bull}/{n} bullish | {bear}/{n} bearish | Tin cay: {pct}%\n\n"
        f"TOM TAT: {verdict['summary']}\n\n"
    )

    # Action plan
    action_label = "MUA" if "MUA" in fv else "BAN" if "BAN" in fv else None
    if action_label:
        action_block = (
            f"ACTION PLAN ({action_label}):\n"
            f"  Entry:  {_fmt_price(ap['entry_low'])} - {_fmt_price(ap['entry_high'])}\n"
            f"  Target: {_fmt_price(ap['tp'])} ({ap['tp_pct']:+.1f}%)\n"
            f"  SL:     {_fmt_price(ap['sl'])} ({ap['sl_pct']:+.1f}%)\n"
            f"  R:R   = 1:{ap['rr']}\n"
        )
    else:
        action_block = (
            f"ACTION PLAN (THEO DOI):\n"
            f"  Vung mua neu co tin hieu: {_fmt_price(ap['entry_low'])} - {_fmt_price(ap['entry_high'])}\n"
            f"  Muc TP neu vao: {_fmt_price(ap['tp'])} ({ap['tp_pct']:+.1f}%)\n"
            f"  Dieu kien: > 50% agents dong thuan\n"
        )

    # Vibe details (rút gọn cho vừa Telegram)
    details = vibe.get("details", {})
    detail_lines = []
    for name, desc in details.items():
        sig = sigs.get(name, 0)
        short = desc[:80].replace("\n", " ")
        detail_lines.append(f"  [{name}] {short}")
    detail_block = "\nCHI TIET:\n" + "\n".join(detail_lines) if detail_lines else ""

    footer = (
        f"\nLUU Y: {verdict['negative']}\n"
        f"{'='*32}"
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


def analyze_stock_full(symbol: str) -> tuple:
    """
    Giống analyze_stock() nhưng trả về tuple (msg, metadata).
    metadata dùng để lưu DB qua save_signal() trong bot.py.
    """
    # 1. Thu thập data (bỏ fundamental và foreign_flow)
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_price  = ex.submit(get_price_data,  symbol, 90)
        f_market = ex.submit(get_market_data)
        f_news   = ex.submit(get_news_data,   symbol)
        price_data  = f_price.result()
        market_data = f_market.result()
        news_data   = f_news.result()

    if not price_data["success"]:
        return f"Khong lay duoc du lieu gia {symbol}: {price_data['error']}", None

    try:
        ind = compute_indicators(price_data["df"])
    except Exception as e:
        return f"Loi tinh indicators {symbol}: {e}", None

    # 2. Chạy 6 rule-based agents + 10 Vibe-Trading engines (song song)
    try:
        with ThreadPoolExecutor(max_workers=7) as ex:
            f1 = ex.submit(run_trend_agent,         symbol, ind)
            f2 = ex.submit(run_volume_agent,        symbol, ind)
            f3 = ex.submit(run_risk_agent,          symbol, ind)
            f4 = ex.submit(run_news_agent,          symbol, news_data)
            f5 = ex.submit(run_market_regime_agent, market_data)
            f6 = ex.submit(run_macro_agent,         news_data)
            # Vibe-Trading engines chạy song song
            f_vibe = ex.submit(_run_vibe_safe, symbol, price_data["df"]) if _VIBE_AVAILABLE else None
            trend_txt,  trend_v  = f1.result()
            volume_txt, volume_v = f2.result()
            risk_txt,   risk_v   = f3.result()
            news_txt,   news_v   = f4.result()
            market_txt, market_v = f5.result()
            macro_txt,  macro_v  = f6.result()
            vibe_result = f_vibe.result() if f_vibe else None
    except Exception as e:
        return f"Loi khi chay agents {symbol}: {e}", None

    agent_verdicts = {
        "trend": trend_v, "volume": volume_v, "risk": risk_v,
        "news": news_v, "market": market_v, "macro": macro_v,
    }

    # 3. Verdict từ 6 rule-based agents
    try:
        verdict = run_verdict_agent(symbol, agent_verdicts, ind)
    except Exception as e:
        return f"Loi verdict {symbol}: {e}", None

    # 4. Format
    final_v  = verdict["verdict_label"]
    conf_pct = verdict["confidence_pct"]
    ap       = verdict["action_plan"]
    summary  = verdict["summary"]
    negative = verdict["negative"]
    bull     = verdict["bull_count"]
    bear     = verdict["bear_count"]
    n_agents = verdict["active_agents"]

    emoji = {
        "DONG THUAN MUA": "🟢", "NGHIENG MUA": "🟢",
        "DONG THUAN BAN": "🔴", "NGHIENG BAN": "🔴",
        "TRUNG LAP": "🟡",
    }.get(final_v, "🟡")

    vnindex_str = (f"{market_data['vnindex']:,} ({market_data['change_5d']:+.1f}% 5D)"
                   if market_data.get("success") else "N/A")
    now = datetime.now().strftime("%d/%m %H:%M")

    agents_list = [
        ("Xu huong", trend_v,  trend_txt),
        ("Volume",   volume_v, volume_txt),
        ("Rui ro",   risk_v,   risk_txt),
        ("News",     news_v,   news_txt),
        ("Market",   market_v, market_txt),
        ("Macro",    macro_v,  macro_txt),
    ]

    msg = _build_message_safe_v2(
        emoji, symbol, now, ind, vnindex_str,
        agents_list, final_v, conf_pct, bull, bear, n_agents,
        summary, ap, negative,
        vibe_result=vibe_result,
    )

    metadata = {
        "verdict":        verdict,
        "ind":            ind,
        "agent_verdicts": agent_verdicts,
        "macro_v":        macro_v,
        "vibe_result":    vibe_result,
    }
    return msg, metadata
def analyze_stock(symbol: str) -> str:
    """Phân tích mã chứng khoán dùng Vibe-Trading 7 engines + 2 context agents."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_price  = ex.submit(get_price_data,  symbol, 120)
        f_market = ex.submit(get_market_data)
        f_news   = ex.submit(get_news_data,   symbol)
        price_data  = f_price.result()
        market_data = f_market.result()
        news_data   = f_news.result()

    if not price_data["success"]:
        return f"Khong lay duoc du lieu gia {symbol}: {price_data['error']}"
    try:
        ind = compute_indicators(price_data["df"])
        ind["symbol"] = symbol
    except Exception as e:
        return f"Loi tinh indicators {symbol}: {e}"

    # ── Chạy Vibe-Trading 7 engines ──────────────────────────────────────────
    if not _VIBE_AVAILABLE:
        return "vibe_skills.py chua duoc deploy. Them file vibe_skills.py vao project."
    try:
        vibe = _run_vibe_agents(symbol, price_data["df"])
    except Exception as e:
        return f"Loi Vibe-Trading engines: {e}"

    # ── Tổng hợp verdict (bao gồm Market + News context) ─────────────────────
    try:
        verdict = _vibe_verdict(vibe, ind, market_data, news_data)
    except Exception as e:
        return f"Loi tong hop verdict: {e}"

    fv   = verdict["verdict_label"]
    emoji = {"DONG THUAN MUA":"🟢","NGHIENG MUA":"🟢",
             "DONG THUAN BAN":"🔴","NGHIENG BAN":"🔴","TRUNG LAP":"🟡"}.get(fv,"🟡")
    vnindex_str = (f"{market_data['vnindex']:,} ({market_data['change_5d']:+.1f}% 5D)"
                   if market_data.get("success") else "N/A")
    now = datetime.now().strftime("%d/%m %H:%M")

    return _build_vibe_message(emoji, symbol, now, ind, vnindex_str, vibe, verdict)


def analyze_stock_full(symbol: str) -> tuple:
    """Trả về (msg, metadata) cho bot.py — metadata dùng để lưu DB."""
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_price  = ex.submit(get_price_data,  symbol, 120)
        f_market = ex.submit(get_market_data)
        f_news   = ex.submit(get_news_data,   symbol)
        price_data  = f_price.result()
        market_data = f_market.result()
        news_data   = f_news.result()

    if not price_data["success"]:
        return f"Khong lay duoc du lieu gia {symbol}: {price_data['error']}", None
    try:
        ind = compute_indicators(price_data["df"])
        ind["symbol"] = symbol
    except Exception as e:
        return f"Loi tinh indicators {symbol}: {e}", None

    if not _VIBE_AVAILABLE:
        return "vibe_skills.py chua duoc deploy cung project.", None
    try:
        vibe = _run_vibe_agents(symbol, price_data["df"])
    except Exception as e:
        return f"Loi Vibe-Trading engines: {e}", None

    try:
        verdict = _vibe_verdict(vibe, ind, market_data, news_data)
    except Exception as e:
        return f"Loi tong hop verdict: {e}", None

    fv   = verdict["verdict_label"]
    emoji = {"DONG THUAN MUA":"🟢","NGHIENG MUA":"🟢",
             "DONG THUAN BAN":"🔴","NGHIENG BAN":"🔴","TRUNG LAP":"🟡"}.get(fv,"🟡")
    vnindex_str = (f"{market_data['vnindex']:,} ({market_data['change_5d']:+.1f}% 5D)"
                   if market_data.get("success") else "N/A")
    now = datetime.now().strftime("%d/%m %H:%M")

    msg = _build_vibe_message(emoji, symbol, now, ind, vnindex_str, vibe, verdict)

    # agent_verdicts theo format db.py expect
    agent_verdicts = {k: ("MUA" if v>0 else "BAN" if v<0 else "TRUNG LAP")
                      for k,v in verdict["signals"].items()}

    metadata = {
        "verdict":        verdict,
        "ind":            ind,
        "agent_verdicts": agent_verdicts,
        "macro_v":        "THUAN LOI" if verdict["signals"].get("MarketRegime",0)>0
                          else "RUI RO" if verdict["signals"].get("MarketRegime",0)<0
                          else "TRUNG TINH",
        "vibe_result":    vibe,
    }
    return msg, metadata
