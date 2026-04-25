"""
news_sentiment_patch.py
────────────────────────
Patch tích hợp F319 + Fireant vào NewsSentimentAgent trong analyzer.py.

CÁCH DÙNG:
  Thay thế (hoặc bổ sung) method `_fetch_community_sentiment` trong class
  NewsSentimentAgent bằng đoạn code dưới đây.

  Nếu bạn không muốn sửa analyzer.py trực tiếp, import và gọi
  `patch_news_sentiment_agent()` sau khi import analyzer.

────────────────────────────────────────────────────────────────────────────
OPTION A — Patch monkey (không sửa analyzer.py)
────────────────────────────────────────────────────────────────────────────

    # Ở đầu bot.py, sau khi import analyzer:
    import analyzer
    from news_sentiment_patch import patch_news_sentiment_agent
    patch_news_sentiment_agent(analyzer)

────────────────────────────────────────────────────────────────────────────
OPTION B — Thêm thẳng vào analyzer.py (sạch hơn)
────────────────────────────────────────────────────────────────────────────

    Xem phần REPLACEMENT METHOD ở cuối file này.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ─── Import scrapers (graceful) ───────────────────────────────────────────────
try:
    from scrapers.f319_scraper import search_f319
    _F319_OK = True
except ImportError:
    _F319_OK = False
    logger.warning("F319 scraper not available")

try:
    from scrapers.fireant_scraper import search_fireant_all, get_fireant_fundamental
    _FIREANT_OK = True
except ImportError:
    _FIREANT_OK = False
    logger.warning("Fireant scraper not available")


# ─── Standalone fetch function (dùng bởi cả patch lẫn direct call) ───────────

def fetch_community_sentiment_enhanced(symbol: str, max_posts: int = 40) -> dict:
    """
    Lấy sentiment từ F319 + Fireant + các nguồn cũ song song.

    Returns:
        {
          "posts":         list[dict],   # all posts combined
          "bull_count":    int,
          "bear_count":    int,
          "neutral_count": int,
          "sources":       list[str],    # nguồn nào có data
          "top_posts":     list[dict],   # top 5 posts relevant nhất
          "fireant_fundamental": dict,   # P/E, ROE, ... (nếu có)
        }
    """
    results: list[dict] = []
    sources: list[str] = []
    fa_fundamental: dict = {}

    tasks = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        if _F319_OK:
            tasks["f319"] = ex.submit(search_f319, symbol, max_posts // 2)
        if _FIREANT_OK:
            tasks["fireant"] = ex.submit(search_fireant_all, symbol, max_posts // 2)
            tasks["fa_fund"] = ex.submit(get_fireant_fundamental, symbol)

        for name, future in tasks.items():
            try:
                data = future.result(timeout=15)
                if name == "fa_fund":
                    fa_fundamental = data or {}
                elif data:
                    results.extend(data)
                    sources.append(name)
            except TimeoutError:
                logger.warning("fetch_community_sentiment timeout: %s", name)
            except Exception as e:
                logger.warning("fetch_community_sentiment error %s: %s", name, e)

    # Count sentiments
    bull    = sum(1 for p in results if p.get("sentiment_hint") == "bullish")
    bear    = sum(1 for p in results if p.get("sentiment_hint") == "bearish")
    neutral = len(results) - bull - bear

    # Sort by recency + likes for top_posts
    def _score(p):
        like_score = min(p.get("likes", 0) / 10, 5)   # cap at 5
        return like_score

    top = sorted(results, key=_score, reverse=True)[:5]

    return {
        "posts":                  results,
        "bull_count":             bull,
        "bear_count":             bear,
        "neutral_count":          neutral,
        "sources":                sources,
        "top_posts":              top,
        "fireant_fundamental":    fa_fundamental,
        "total":                  len(results),
    }


def format_sentiment_for_llm(symbol: str, data: dict) -> str:
    """
    Format kết quả sentiment thành text ngắn gọn cho LLM prompt.
    Dùng trong NewsSentimentAgent._build_prompt() hoặc tương đương.
    """
    if not data or data.get("total", 0) == 0:
        return f"Không tìm thấy thảo luận cộng đồng cho {symbol}."

    lines = [
        f"📊 Community sentiment {symbol} ({data['total']} posts từ {', '.join(data['sources'])}):",
        f"  🟢 Bullish: {data['bull_count']}  🔴 Bearish: {data['bear_count']}  ⚪ Neutral: {data['neutral_count']}",
    ]

    # Thêm chỉ số cơ bản nếu có
    fa = data.get("fireant_fundamental", {})
    if fa:
        fa_parts = []
        for k, label in [("pe","P/E"), ("pb","P/B"), ("roe","ROE"), ("eps","EPS")]:
            if k in fa:
                val = fa[k]
                fa_parts.append(f"{label}={val:.1f}" if isinstance(val, float) else f"{label}={val}")
        if fa_parts:
            lines.append(f"  📈 Cơ bản: {', '.join(fa_parts)}")

    # Top posts
    top = data.get("top_posts", [])
    if top:
        lines.append("  💬 Bài viết nổi bật:")
        for p in top[:3]:
            src  = p.get("source", "")
            hint = p.get("sentiment_hint", "neutral")
            emoji = "🟢" if hint == "bullish" else ("🔴" if hint == "bearish" else "⚪")
            lines.append(f"    {emoji} [{src}] {p['title'][:100]}")

    return "\n".join(lines)


# ─── Monkey-patch function ────────────────────────────────────────────────────

def patch_news_sentiment_agent(analyzer_module) -> None:
    """
    Patch NewsSentimentAgent trong analyzer module đã import.

    Usage:
        import analyzer
        from news_sentiment_patch import patch_news_sentiment_agent
        patch_news_sentiment_agent(analyzer)
    """
    try:
        agent_cls = analyzer_module.NewsSentimentAgent
    except AttributeError:
        logger.error("patch_news_sentiment_agent: NewsSentimentAgent not found in analyzer")
        return

    def _patched_fetch(self, symbol: str) -> str:
        """
        Drop-in replacement cho _fetch_community_sentiment (hoặc tương đương).
        Trả về formatted string để nhét vào LLM prompt.
        """
        try:
            data = fetch_community_sentiment_enhanced(symbol, max_posts=40)
            return format_sentiment_for_llm(symbol, data)
        except Exception as e:
            logger.error("NewsSentimentAgent patched fetch error: %s", e)
            return f"Lỗi khi lấy community sentiment: {e}"

    # Thử patch các method name phổ biến
    for method_name in ["_fetch_community_sentiment", "_fetch_news", "_get_sentiment_data"]:
        if hasattr(agent_cls, method_name):
            setattr(agent_cls, method_name, _patched_fetch)
            logger.info("✅ Patched NewsSentimentAgent.%s with F319+Fireant", method_name)
            return

    # Nếu không tìm thấy method cụ thể, patch chung
    agent_cls._fetch_community_sentiment = _patched_fetch
    logger.info("✅ Added NewsSentimentAgent._fetch_community_sentiment (F319+Fireant)")


# ─── OPTION B: Replacement method (copy vào analyzer.py) ─────────────────────
#
# Trong class NewsSentimentAgent của analyzer.py,
# THÊM method này (hoặc THAY THẾ _fetch_community_sentiment hiện tại):
#
# def _fetch_community_sentiment(self, symbol: str) -> str:
#     """Lấy sentiment từ F319 + Fireant."""
#     try:
#         from news_sentiment_patch import (
#             fetch_community_sentiment_enhanced,
#             format_sentiment_for_llm,
#         )
#         data = fetch_community_sentiment_enhanced(symbol, max_posts=40)
#         return format_sentiment_for_llm(symbol, data)
#     except Exception as e:
#         logger.error("Community sentiment error: %s", e)
#         return f"Không thể lấy community sentiment: {e}"
