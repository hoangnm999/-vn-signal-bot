"""
scrapers/fireant_scraper.py
───────────────────────────
Lấy tin tức + bình luận từ Fireant API cho NewsSentimentAgent.

Endpoints dùng:
  1. /symbols/{symbol}/news         — tin tức cơ bản
  2. /symbols/{symbol}/activity     — activity feed (người dùng đăng)
  3. /posts?symbol={symbol}         — posts thảo luận (social feed)
  4. /symbols/{symbol}/fundamental  — chỉ số cơ bản (P/E, EPS, ...)

Yêu cầu: FIREANT_TOKEN trong env (Bearer token từ fireant.vn)
Fallback: nếu không có token, trả danh sách rỗng (không crash)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
FIREANT_BASE    = "https://restv2.fireant.vn"
FIREANT_TOKEN   = os.getenv("FIREANT_TOKEN", "")

_DEFAULT_HEADERS = {
    "Authorization": f"Bearer {FIREANT_TOKEN}" if FIREANT_TOKEN else "",
    "Accept": "application/json",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "User-Agent": "Mozilla/5.0 VNSignalBot/4.0",
}

# ─── Cache ────────────────────────────────────────────────────────────────────
_CACHE: dict[str, tuple[float, object]] = {}
CACHE_TTL = 600   # 10 phút


def _cache_key(*parts) -> str:
    return hashlib.md5(":".join(str(p) for p in parts).encode()).hexdigest()


def _from_cache(key: str):
    entry = _CACHE.get(key)
    if entry and (time.time() - entry[0]) < CACHE_TTL:
        return entry[1]
    return None


def _to_cache(key: str, data) -> None:
    _CACHE[key] = (time.time(), data)


# ─── HTTP helper ──────────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None, timeout: int = 10) -> dict | list | None:
    """GET Fireant API. Returns parsed JSON or None on error."""
    if not FIREANT_TOKEN:
        logger.debug("FIREANT_TOKEN not set, skipping: %s", endpoint)
        return None

    url = f"{FIREANT_BASE}{endpoint}"
    try:
        resp = requests.get(url, headers=_DEFAULT_HEADERS, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            logger.error("Fireant 401 Unauthorized — kiểm tra FIREANT_TOKEN")
        else:
            logger.warning("Fireant HTTP error %s: %s", endpoint, e)
    except Exception as e:
        logger.warning("Fireant request error %s: %s", endpoint, e)
    return None


# ─── Public API ───────────────────────────────────────────────────────────────

def get_fireant_news(symbol: str, max_posts: int = 30) -> list[dict]:
    """
    Lấy tin tức từ Fireant cho mã CK.

    Returns:
        list of {title, url, date, snippet, source, sentiment_hint}
    """
    key = _cache_key("news", symbol, max_posts)
    cached = _from_cache(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    data = _get(f"/symbols/{symbol}/news", params={"limit": max_posts, "offset": 0})
    if not data:
        return []

    results = []
    items = data if isinstance(data, list) else data.get("items", data.get("data", []))

    for item in items[:max_posts]:
        title   = item.get("title") or item.get("name") or ""
        url     = item.get("url") or item.get("link") or ""
        pub_raw = item.get("publishedDate") or item.get("date") or ""
        snippet = item.get("shortContent") or item.get("content") or item.get("body") or ""

        # Chuẩn hoá ngày
        date = _parse_date(pub_raw)

        results.append({
            "title":          title[:300],
            "url":            url,
            "date":           date,
            "snippet":        _clean_text(snippet)[:600],
            "source":         "Fireant-News",
            "sentiment_hint": _quick_sentiment(title + " " + snippet),
        })

    _to_cache(key, results)
    return results


def get_fireant_posts(symbol: str, max_posts: int = 20) -> list[dict]:
    """
    Lấy social posts / bình luận người dùng từ Fireant.

    Returns:
        list of {title, url, date, snippet, source, sentiment_hint, likes}
    """
    key = _cache_key("posts", symbol, max_posts)
    cached = _from_cache(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    # Thử endpoint /posts?symbol=VCB
    data = _get("/posts", params={"symbol": symbol, "limit": max_posts, "offset": 0})
    if not data:
        # Thử /symbols/{symbol}/activity
        data = _get(f"/symbols/{symbol}/activity", params={"limit": max_posts})
    if not data:
        return []

    results = []
    items = data if isinstance(data, list) else data.get("items", data.get("data", []))

    for item in items[:max_posts]:
        content = (
            item.get("content") or
            item.get("text") or
            item.get("body") or ""
        )
        if not content or len(content) < 20:
            continue

        pub_raw = item.get("date") or item.get("publishedDate") or item.get("createdAt") or ""
        post_id = item.get("id") or ""
        url = f"https://fireant.vn/feed/{post_id}" if post_id else "https://fireant.vn"
        likes = item.get("likeCount") or item.get("totalLikes") or 0

        # Lấy dòng đầu làm "title"
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        title = lines[0][:150] if lines else content[:150]

        results.append({
            "title":          title,
            "url":            url,
            "date":           _parse_date(pub_raw),
            "snippet":        _clean_text(content)[:600],
            "source":         "Fireant-Social",
            "sentiment_hint": _quick_sentiment(content),
            "likes":          int(likes),
        })

    # Sắp xếp theo likes giảm dần — post nhiều like có ảnh hưởng hơn
    results.sort(key=lambda x: x.get("likes", 0), reverse=True)

    _to_cache(key, results)
    return results


def get_fireant_fundamental(symbol: str) -> dict:
    """
    Lấy chỉ số cơ bản (P/E, EPS, ROE, ...) từ Fireant.
    Dùng cho FundamentalFilter trong analyzer.py.

    Returns:
        dict với các key: pe, pb, eps, roe, roa, revenue_growth, ...
        Trả {} nếu lỗi hoặc không có token.
    """
    key = _cache_key("fundamental", symbol)
    cached = _from_cache(key)
    if cached is not None:
        return cached  # type: ignore[return-value]

    data = _get(f"/symbols/{symbol}/fundamental")
    if not data:
        return {}

    # Chuẩn hoá tên field (Fireant hay đổi)
    mapping = {
        "pe":              ["pe", "priceToEarnings", "P/E"],
        "pb":              ["pb", "priceToBook", "P/B"],
        "eps":             ["eps", "earningsPerShare"],
        "roe":             ["roe", "returnOnEquity"],
        "roa":             ["roa", "returnOnAssets"],
        "revenue_growth":  ["revenueGrowth", "revenue_growth"],
        "profit_growth":   ["profitGrowth", "profit_growth"],
        "market_cap":      ["marketCap", "market_cap"],
        "dividend_yield":  ["dividendYield", "dividend_yield"],
    }

    result: dict = {}
    flat = data if isinstance(data, dict) else {}

    for our_key, candidates in mapping.items():
        for c in candidates:
            val = flat.get(c)
            if val is not None:
                result[our_key] = val
                break

    _to_cache(key, result)
    return result


def search_fireant_all(symbol: str, max_posts: int = 30) -> list[dict]:
    """
    Convenience: gộp news + social posts, đã deduplicated.
    Đây là hàm chính gọi từ NewsSentimentAgent.
    """
    news  = get_fireant_news(symbol, max_posts=max_posts // 2 + 5)
    posts = get_fireant_posts(symbol, max_posts=max_posts // 2 + 5)
    combined = news + posts
    return combined[:max_posts]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_date(raw: str) -> str:
    """Cố parse nhiều định dạng ngày, trả YYYY-MM-DD hoặc today."""
    if not raw:
        return datetime.now().strftime("%Y-%m-%d")
    import re
    # ISO format: 2026-04-25T14:30:00Z hoặc 2026-04-25
    m = re.match(r"(\d{4}-\d{2}-\d{2})", str(raw))
    if m:
        return m.group(1)
    # Unix timestamp (ms)
    try:
        ts = int(raw)
        if ts > 1e10:
            ts //= 1000
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OSError):
        pass
    return datetime.now().strftime("%Y-%m-%d")


def _clean_text(text: str) -> str:
    """Loại bỏ HTML tags đơn giản."""
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_BULL_KW = ["tăng", "mua", "khuyến nghị mua", "outperform", "target price",
            "kết quả tốt", "lợi nhuận tăng", "tích lũy", "hỗ trợ", "bứt phá"]
_BEAR_KW = ["giảm", "bán", "cắt lỗ", "underperform", "downgrade",
            "lỗ", "rủi ro", "xả", "margin call", "thủng hỗ trợ"]


def _quick_sentiment(text: str) -> str:
    text_lower = text.lower()
    bull = sum(1 for kw in _BULL_KW if kw in text_lower)
    bear = sum(1 for kw in _BEAR_KW if kw in text_lower)
    if bull > bear:
        return "bullish"
    if bear > bull:
        return "bearish"
    return "neutral"
