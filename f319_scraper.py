"""
scrapers/f319_scraper.py
────────────────────────
Cào bài viết / thảo luận từ diễn đàn F319 (f319.com)
cho NewsSentimentAgent của analyzer.py.

Features:
  - cloudscraper để bypass Cloudflare JS challenge
  - Search qua Google site:f319.com (không bị block)
  - Fallback: scrape trực tiếp topic list + thread detail
  - Cache ngắn (5 phút) tránh hammer server
  - Trả về list[dict] chuẩn hoá dùng chung với Voz / Fireant
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional
from urllib.parse import quote_plus, urljoin

logger = logging.getLogger(__name__)

# ─── optional imports (graceful fallback) ────────────────────────────────────
try:
    import cloudscraper
    _HAS_CLOUDSCRAPER = True
except ImportError:
    import requests as cloudscraper          # type: ignore[no-redef]
    _HAS_CLOUDSCRAPER = False
    logger.warning("cloudscraper not installed, using plain requests (may get 403 on F319)")

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False
    logger.error("beautifulsoup4 not installed: pip install beautifulsoup4 lxml")

# ─── Constants ────────────────────────────────────────────────────────────────
F319_BASE       = "https://f319.com"
F319_SEARCH_URL = "https://f319.com/search/results?query={query}&t=post"
GOOGLE_SEARCH   = "https://www.google.com/search?q=site:f319.com+{query}&num=20"

HEADERS = {
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Simple in-memory cache: {cache_key: (timestamp, result)}
_CACHE: dict[str, tuple[float, list[dict]]] = {}
CACHE_TTL = 300   # 5 phút


# ─── Session factory ─────────────────────────────────────────────────────────

def _make_session():
    """Tạo cloudscraper session (bypass CF) hoặc requests bình thường."""
    if _HAS_CLOUDSCRAPER:
        return cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
    s = __import__("requests").Session()
    s.headers.update(HEADERS)
    return s


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _cache_key(symbol: str, max_posts: int) -> str:
    return hashlib.md5(f"f319:{symbol}:{max_posts}".encode()).hexdigest()


def _from_cache(key: str) -> Optional[list[dict]]:
    entry = _CACHE.get(key)
    if entry and (time.time() - entry[0]) < CACHE_TTL:
        return entry[1]
    return None


def _to_cache(key: str, data: list[dict]) -> None:
    _CACHE[key] = (time.time(), data)


# ─── Main public API ─────────────────────────────────────────────────────────

def search_f319(symbol: str, max_posts: int = 30) -> list[dict]:
    """
    Tìm bài viết F319 liên quan đến mã chứng khoán.

    Returns:
        list of {title, url, date, snippet, source, sentiment_hint}
        Trả [] nếu lỗi (không raise exception để không phá pipeline).
    """
    if not _HAS_BS4:
        logger.error("bs4 missing — F319 scraper disabled")
        return []

    key = _cache_key(symbol, max_posts)
    cached = _from_cache(key)
    if cached is not None:
        logger.debug("f319 cache hit: %s", symbol)
        return cached

    results: list[dict] = []

    # Strategy 1: Google site search (most reliable, no CF issue)
    try:
        results = _google_site_search(symbol, max_posts)
        logger.info("F319 Google search %s: %d posts", symbol, len(results))
    except Exception as e:
        logger.warning("F319 Google search failed: %s", e)

    # Strategy 2: F319 internal search (fallback)
    if len(results) < 5:
        try:
            direct = _f319_internal_search(symbol, max_posts)
            # merge, deduplicate by url
            urls = {r["url"] for r in results}
            results += [r for r in direct if r["url"] not in urls]
            logger.info("F319 internal search %s: total %d posts", symbol, len(results))
        except Exception as e:
            logger.warning("F319 internal search failed: %s", e)

    results = results[:max_posts]
    _to_cache(key, results)
    return results


def get_f319_thread_content(url: str, max_chars: int = 3000) -> str:
    """
    Lấy nội dung đầy đủ của một thread F319.
    Dùng khi cần chi tiết hơn snippet.

    Returns:
        str — nội dung plain text (truncated), hoặc "" nếu lỗi
    """
    if not _HAS_BS4:
        return ""
    try:
        session = _make_session()
        resp = session.get(url, timeout=10, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # F319 dùng XenForo — posts trong .message-body
        posts = soup.select(".message-body .bbWrapper")
        if not posts:
            posts = soup.select(".postbody")   # fallback cũ

        texts = []
        for p in posts[:5]:   # lấy 5 posts đầu
            text = p.get_text(separator=" ", strip=True)
            if len(text) > 50:
                texts.append(text)

        full_text = "\n\n---\n\n".join(texts)
        return full_text[:max_chars]
    except Exception as e:
        logger.warning("get_f319_thread_content error %s: %s", url, e)
        return ""


# ─── Strategy 1: Google site:f319.com search ─────────────────────────────────

def _google_site_search(symbol: str, max_posts: int) -> list[dict]:
    """Dùng Google để tìm: site:f319.com VCB"""
    import requests   # plain requests cho Google (không cần CF bypass)

    query = quote_plus(f'site:f319.com "{symbol}"')
    url = f"https://www.google.com/search?q={query}&num={min(max_posts, 20)}&hl=vi"
    headers = {
        **HEADERS,
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    results = []

    for g in soup.select("div.g"):
        a_tag = g.select_one("a[href]")
        title_tag = g.select_one("h3")
        snippet_tag = g.select_one("div[data-sncf], .VwiC3b, span.aCOpRe")

        if not a_tag or not title_tag:
            continue

        href = a_tag["href"]
        if "f319.com" not in href:
            continue

        title   = title_tag.get_text(strip=True)
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
        date    = _extract_date_from_snippet(snippet)

        results.append({
            "title":          title,
            "url":            href,
            "date":           date,
            "snippet":        snippet[:500],
            "source":         "F319",
            "sentiment_hint": _quick_sentiment(title + " " + snippet),
        })

    return results


# ─── Strategy 2: F319 internal search ────────────────────────────────────────

def _f319_internal_search(symbol: str, max_posts: int) -> list[dict]:
    """Gọi thẳng F319 search endpoint."""
    session = _make_session()
    query = quote_plus(symbol)
    url = F319_SEARCH_URL.format(query=query)

    resp = session.get(url, timeout=15, headers=HEADERS)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    results = []

    # XenForo search results: li.searchResult
    for item in soup.select("li.searchResult, .block-row.block-row--separated")[:max_posts]:
        a_title = item.select_one("h3.contentRow-title a, .title a")
        snippet_el = item.select_one(".contentRow-snippet, .snippet")
        date_el = item.select_one("time[datetime], .contentRow-minor time")

        if not a_title:
            continue

        href  = urljoin(F319_BASE, a_title.get("href", ""))
        title = a_title.get_text(strip=True)
        snip  = snippet_el.get_text(strip=True) if snippet_el else ""
        date  = (
            date_el.get("datetime", "")[:10]
            if date_el else datetime.now().strftime("%Y-%m-%d")
        )

        results.append({
            "title":          title,
            "url":            href,
            "date":           date,
            "snippet":        snip[:500],
            "source":         "F319",
            "sentiment_hint": _quick_sentiment(title + " " + snip),
        })

    return results


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_date_from_snippet(text: str) -> str:
    """Cố trích ngày từ snippet Google (dạng '25 thg 4, 2026')."""
    import re
    # dạng: "25 thg 4, 2026" hoặc "Apr 25, 2026"
    m = re.search(r"(\d{1,2})\s+thg\s+(\d{1,2}),?\s+(\d{4})", text)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        try:
            return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
        except ValueError:
            pass
    m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m2:
        return m2.group(0)
    return datetime.now().strftime("%Y-%m-%d")


# Từ khoá bullish / bearish nhanh (để hint cho LLM, không thay thế NLP)
_BULL_KW = ["tăng", "mua", "breakout", "bứt phá", "uptrend", "target", "tích lũy",
            "hỗ trợ tốt", "kết quả tốt", "lợi nhuận tăng"]
_BEAR_KW = ["giảm", "bán", "breakdown", "hỗ trợ thủng", "cắt lỗ", "margin call",
            "downtrend", "rủi ro", "lỗ", "xả"]


def _quick_sentiment(text: str) -> str:
    """Trả 'bullish' / 'bearish' / 'neutral' nhanh dựa trên từ khoá."""
    text_lower = text.lower()
    bull = sum(1 for kw in _BULL_KW if kw in text_lower)
    bear = sum(1 for kw in _BEAR_KW if kw in text_lower)
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"
