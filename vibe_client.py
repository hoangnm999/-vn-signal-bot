"""
vibe_client.py — HTTP client gọi Vibe-Trading API (HKUDS/Vibe-Trading)

Cấu trúc Vibe-Trading thực tế:
  71 Skills  : Technical(25) | Fundamental(15) | Macro(10) |
               Sentiment(8)  | Quant(10)       | Risk(5)   | Portfolio(8)
  69 Agents  : Leader / Researcher / Critic / Secretary per swarm
  7 Archetypes: Strategists | Technical Experts | Value Hunters |
                Macro Watchers | Risk Controllers | Sentiment Analysts | Backtester
  35 Swarms  : mỗi swarm gộp nhiều skills + agents liên quan

Env vars (Railway bot service):
    VIBE_API_URL  = https://vibe-trading-xxxx.railway.app
    VIBE_API_KEY  = <API_AUTH_KEY bên Vibe service>
"""

from __future__ import annotations
import os, time, logging, requests
from typing import Optional, Callable

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if url and not url.startswith("http"):
        url = "https://" + url
    return url


VIBE_API_URL    = _normalize_url(os.environ.get("VIBE_API_URL", ""))
VIBE_API_KEY    = os.environ.get("VIBE_API_KEY", "")
POLL_TIMEOUT    = 900
POLL_INTERVAL   = 8
REQUEST_TIMEOUT = 30


# ══════════════════════════════════════════════════════════════════════════════
# SKILLS COVERAGE — 71 skills map vào từng swarm
# ══════════════════════════════════════════════════════════════════════════════

SKILLS_COVERAGE: dict[str, list[str]] = {
    # Technical Analysis (25 skills) — 4 swarms
    "technical": [
        "candlestick_patterns", "chart_pattern_recognition",
        "moving_average_systems", "momentum_indicators", "trend_following",
        "ichimoku_cloud", "bollinger_bands", "volume_analysis",
        "support_resistance", "breakout_detection", "price_action",
        "gap_analysis", "divergence_detection", "multi_timeframe_analysis",
        "relative_strength",
    ],
    "elliott_harmonic": [
        "elliott_wave", "harmonic_patterns", "fibonacci_analysis",
        "chart_pattern_recognition",
    ],
    "volatility_seasonal": [
        "volatility_analysis", "seasonal_patterns", "market_breadth",
        "mean_reversion", "intermarket_analysis",
    ],
    "market_microstructure": [
        "market_microstructure", "options_flow", "volume_analysis",
        "market_breadth",
    ],
    # Fundamental Analysis (15 skills) — 4 swarms
    "fundamental": [
        "financial_statement_analysis", "ratio_analysis", "dcf_valuation",
        "comparable_analysis", "earnings_quality",
    ],
    "earnings": [
        "earnings_quality", "revenue_analysis", "margin_analysis",
        "growth_analysis", "financial_statement_analysis",
    ],
    "equity": [
        "comparable_analysis", "competitive_moat", "management_assessment",
        "industry_analysis", "esg_scoring",
    ],
    "valuation": [
        "dcf_valuation", "ratio_analysis", "cash_flow_analysis",
        "dividend_analysis", "debt_analysis",
    ],
    # Macro Policy (10 skills) — 7 swarms
    "macro": [
        "interest_rate_analysis", "inflation_analysis",
        "monetary_policy", "global_liquidity",
    ],
    "macro_rates": [
        "interest_rate_analysis", "currency_analysis",
        "fiscal_policy", "trade_policy",
    ],
    "sector":          ["industry_analysis", "commodity_macro", "intermarket_analysis"],
    "geopolitical":    ["geopolitical_risk", "trade_policy", "currency_analysis"],
    "commodity":       ["commodity_macro", "inflation_analysis", "intermarket_analysis"],
    "global_alloc":    ["global_liquidity", "gdp_growth_analysis", "monetary_policy", "fiscal_policy"],
    "global_equities": ["gdp_growth_analysis", "currency_analysis", "intermarket_analysis", "global_liquidity"],
    # Sentiment Analysis (8 skills) — 3 swarms
    "sentiment":    ["news_sentiment", "analyst_sentiment", "market_fear_greed", "retail_sentiment"],
    "social_alpha": ["social_media_sentiment", "retail_sentiment", "market_fear_greed"],
    "insider_flow": ["insider_trading", "fund_flow_analysis", "options_sentiment"],
    # Quant & Optimization (10 skills) — 6 swarms
    "quant":        ["factor_investing", "momentum_factor", "value_factor", "quality_factor"],
    "factor":       ["factor_investing", "momentum_factor", "value_factor", "quality_factor", "ml_signal_generation"],
    "pairs":        ["pairs_trading", "statistical_arbitrage", "correlation_analysis"],
    "stat_arb":     ["statistical_arbitrage", "pairs_trading", "mean_reversion"],
    "ml_quant":     ["ml_signal_generation", "backtesting_engine", "factor_investing"],
    "event_driven": ["backtesting_engine", "momentum_factor", "earnings_quality"],
    # Risk Management (5 skills) — 1 swarm
    "risk":         ["var_calculation", "drawdown_analysis", "stress_testing", "correlation_analysis", "position_sizing"],
    # Portfolio Utility (8 skills) — 3 swarms
    "investment":    ["portfolio_construction", "allocation_optimization", "benchmark_comparison", "performance_attribution"],
    "portfolio":     ["portfolio_construction", "rebalancing", "liquidity_management", "performance_attribution"],
    "portfolio_opt": ["portfolio_optimization", "risk_parity", "allocation_optimization", "tax_optimization", "reporting_generation"],
    # Asset Class
    "etf":              ["allocation_optimization", "benchmark_comparison", "liquidity_management"],
    "credit":           ["debt_analysis", "var_calculation", "stress_testing"],
    "derivatives":      ["options_flow", "options_sentiment", "var_calculation"],
    "fund_select":      ["performance_attribution", "benchmark_comparison", "ratio_analysis"],
    "convertible_bond": ["debt_analysis", "options_flow", "ratio_analysis"],
    # Crypto
    "crypto_research":  ["momentum_indicators", "social_media_sentiment", "market_fear_greed"],
    "crypto_trading":   ["breakout_detection", "volume_analysis", "momentum_indicators"],
}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT INFO — 69 agents, 4 roles per swarm, 7 archetypes
# ══════════════════════════════════════════════════════════════════════════════

AGENT_ROLES = {
    "Leader":     "Điều phối, ra kết luận cuối, tổng hợp debate",
    "Researcher": "Phân tích dữ liệu, tính toán, lập luận từ data",
    "Critic":     "Phản biện, tìm điểm yếu trong luận điểm",
    "Secretary":  "Ghi chép, tổng hợp, format output",
}

AGENT_ARCHETYPES = {
    "the_strategists":    "Xu hướng dài hạn, chiến lược vĩ mô, asset allocation",
    "technical_experts":  "Phân tích kỹ thuật, chart patterns, timing",
    "value_hunters":      "Định giá cơ bản, tìm cổ phiếu undervalued",
    "macro_watchers":     "Vĩ mô, lãi suất, tỷ giá, chính sách",
    "risk_controllers":   "Quản lý rủi ro, VaR, drawdown, sizing",
    "sentiment_analysts": "Tâm lý thị trường, news flow, social",
    "the_backtester":     "Backtest, validate signal, performance stats",
}

# swarm alias → (archetype, n_agents)
SWARM_AGENT_INFO: dict[str, tuple[str, int]] = {
    "technical":            ("technical_experts",  6),
    "elliott_harmonic":     ("technical_experts",  4),
    "volatility_seasonal":  ("technical_experts",  4),
    "market_microstructure":("technical_experts",  4),
    "fundamental":          ("value_hunters",       4),
    "earnings":             ("value_hunters",       4),
    "equity":               ("value_hunters",       4),
    "valuation":            ("value_hunters",       4),
    "macro":                ("macro_watchers",      4),
    "macro_rates":          ("macro_watchers",      4),
    "sector":               ("macro_watchers",      4),
    "geopolitical":         ("macro_watchers",      4),
    "commodity":            ("macro_watchers",      3),
    "global_alloc":         ("the_strategists",     4),
    "global_equities":      ("the_strategists",     4),
    "sentiment":            ("sentiment_analysts",  4),
    "social_alpha":         ("sentiment_analysts",  4),
    "insider_flow":         ("sentiment_analysts",  3),
    "quant":                ("the_backtester",      4),
    "factor":               ("the_backtester",      4),
    "pairs":                ("the_backtester",      4),
    "stat_arb":             ("the_backtester",      4),
    "ml_quant":             ("the_backtester",      3),
    "event_driven":         ("the_backtester",      3),
    "risk":                 ("risk_controllers",    4),  # + position_sizing = 5 skills
    "investment":           ("the_strategists",     4),
    "portfolio":            ("the_strategists",     4),
    "portfolio_opt":        ("the_strategists",     4),
    "etf":                  ("the_strategists",     4),
    "credit":               ("risk_controllers",    4),
    "derivatives":          ("risk_controllers",    3),
    "fund_select":          ("the_strategists",     3),
    "convertible_bond":     ("value_hunters",       4),
    "crypto_research":      ("technical_experts",   4),
    "crypto_trading":       ("technical_experts",   4),
}


# ══════════════════════════════════════════════════════════════════════════════
# SWARM ALIASES — alias → preset name trong Vibe-Trading server
# ══════════════════════════════════════════════════════════════════════════════

SWARM_ALIASES: dict[str, str] = {
    # Technical (4 swarms)
    "technical":             "technical_analysis_panel",
    "elliott_harmonic":      "elliott_harmonic_panel",
    "volatility_seasonal":   "volatility_seasonal_desk",
    "market_microstructure": "market_microstructure_desk",
    # Fundamental (4 swarms)
    "fundamental":           "fundamental_research_team",
    "earnings":              "earnings_research_desk",
    "equity":                "equity_research_team",
    "valuation":             "valuation_committee",
    # Macro (7 swarms)
    "macro":                 "macro_strategy_forum",
    "macro_rates":           "macro_rates_fx_desk",
    "sector":                "sector_rotation_team",
    "geopolitical":          "geopolitical_war_room",
    "commodity":             "commodity_research_team",
    "global_alloc":          "global_allocation_committee",
    "global_equities":       "global_equities_desk",
    # Sentiment (3 swarms)
    "sentiment":             "sentiment_intelligence_team",
    "social_alpha":          "social_alpha_team",
    "insider_flow":          "insider_flow_desk",
    # Quant (6 swarms)
    "quant":                 "quant_strategy_desk",
    "factor":                "factor_research_committee",
    "pairs":                 "pairs_research_lab",
    "stat_arb":              "statistical_arbitrage_desk",
    "ml_quant":              "ml_quant_lab",
    "event_driven":          "event_driven_task_force",
    # Risk (1 swarm)
    "risk":                  "risk_committee",
    # Portfolio (3 swarms)
    "investment":            "investment_committee",
    "portfolio":             "portfolio_review_board",
    "portfolio_opt":         "portfolio_optimization_desk",
    # Asset Class (5 swarms)
    "etf":                   "etf_allocation_desk",
    "credit":                "credit_research_team",
    "derivatives":           "derivatives_strategy_desk",
    "fund_select":           "fund_selection_panel",
    "convertible_bond":      "convertible_bond_team",
    # Crypto (2 swarms)
    "crypto_research":       "crypto_research_lab",
    "crypto_trading":        "crypto_trading_desk",
}


SWARM_LABELS: dict[str, str] = {
    "technical":             "Technical Analysis Panel (6 agents)",
    "elliott_harmonic":      "Elliott & Harmonic Panel (4 agents)",
    "volatility_seasonal":   "Volatility & Seasonal Desk (4 agents)",
    "market_microstructure": "Market Microstructure Desk (4 agents)",
    "fundamental":           "Fundamental Research Team (4 agents)",
    "earnings":              "Earnings Research Desk (4 agents)",
    "equity":                "Equity Research Team (4 agents)",
    "valuation":             "Valuation Committee (4 agents)",
    "macro":                 "Macro Strategy Forum (4 agents)",
    "macro_rates":           "Macro Rates & FX Desk (4 agents)",
    "sector":                "Sector Rotation Team (4 agents)",
    "geopolitical":          "Geopolitical War Room (4 agents)",
    "commodity":             "Commodity Research Team (3 agents)",
    "global_alloc":          "Global Allocation Committee (4 agents)",
    "global_equities":       "Global Equities Desk (4 agents)",
    "sentiment":             "Sentiment Intelligence Team (4 agents)",
    "social_alpha":          "Social Alpha Team (4 agents)",
    "insider_flow":          "Insider & Fund Flow Desk (3 agents)",
    "quant":                 "Quant Strategy Desk (4 agents)",
    "factor":                "Factor Research Committee (4 agents)",
    "pairs":                 "Pairs Research Lab (4 agents)",
    "stat_arb":              "Statistical Arbitrage Desk (4 agents)",
    "ml_quant":              "ML Quant Lab (3 agents)",
    "event_driven":          "Event-Driven Task Force (3 agents)",
    "risk":                  "Risk Committee (4 agents)",
    "investment":            "Investment Committee (4 agents)",
    "portfolio":             "Portfolio Review Board (4 agents)",
    "portfolio_opt":         "Portfolio Optimization Desk (4 agents)",
    "etf":                   "ETF Allocation Desk (4 agents)",
    "credit":                "Credit Research Team (4 agents)",
    "derivatives":           "Derivatives Strategy Desk (3 agents)",
    "fund_select":           "Fund Selection Panel (3 agents)",
    "convertible_bond":      "Convertible Bond Team (4 agents)",
    "crypto_research":       "Crypto Research Lab (4 agents)",
    "crypto_trading":        "Crypto Trading Desk (4 agents)",
}

# Nhóm hiển thị trong /vibe (không arg) — dùng emoji + skill count
SWARM_GROUPS: dict[str, list[str]] = {
    "📊 Technical (25 skills)": [
        "technical", "elliott_harmonic",
        "volatility_seasonal", "market_microstructure",
    ],
    "📈 Fundamental (15 skills)": [
        "fundamental", "earnings", "equity", "valuation",
    ],
    "🌍 Macro (10 skills)": [
        "macro", "macro_rates", "sector", "geopolitical",
        "commodity", "global_alloc", "global_equities",
    ],
    "💬 Sentiment (8 skills)": [
        "sentiment", "social_alpha", "insider_flow",
    ],
    "🔢 Quant (10 skills)": [
        "quant", "factor", "pairs",
        "stat_arb", "ml_quant", "event_driven",
    ],
    "🛡️ Risk (5 skills)": ["risk"],
    "💼 Portfolio (8 skills)": [
        "investment", "portfolio", "portfolio_opt",
    ],
    "🏦 Asset Class": [
        "etf", "credit", "derivatives",
        "fund_select", "convertible_bond",
    ],
    "₿ Crypto": ["crypto_research", "crypto_trading"],
}

# Quick alias — viết tắt thân thiện cho /vibe <symbol> <keyword>
QUICK_ALIASES: dict[str, str] = {
    "ta": "technical", "ky_thuat": "technical", "chart": "technical",
    "wave": "elliott_harmonic", "harmonic": "elliott_harmonic", "elliott": "elliott_harmonic",
    "vol": "volatility_seasonal", "seasonal": "volatility_seasonal", "hv": "volatility_seasonal",
    "flow": "market_microstructure", "micro": "market_microstructure",
    "cs": "fundamental", "co_ban": "fundamental",
    "dcf": "valuation", "gia_tri": "valuation",
    "loi_nhuan": "earnings", "eps": "earnings",
    "vi_mo": "macro", "lai_suat": "macro_rates", "ty_gia": "macro_rates",
    "nganh": "sector", "geo": "geopolitical", "hang_hoa": "commodity",
    "toan_cau": "global_alloc", "global": "global_equities",
    "tamly": "sentiment", "news": "sentiment",
    "social": "social_alpha", "insider": "insider_flow",
    "ml": "ml_quant", "arb": "stat_arb", "event": "event_driven",
    "rui_ro": "risk", "var": "risk",
    "danh_muc": "portfolio", "optimize": "portfolio_opt",
}


# ══════════════════════════════════════════════════════════════════════════════
# USER VARS BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _infer_sector(symbol: str) -> str:
    _MAP = {
        frozenset(["VCB","BID","CTG","TCB","MBB","VPB","ACB","HDB","STB","LPB",
                   "TPB","MSB","VIB","OCB","SHB","EIB"]): "banking",
        frozenset(["VIC","VHM","NVL","DXG","PDR","KDH","BCM","DIG","HDC"]): "real_estate",
        frozenset(["FPT","MWG","DGW","CMG","VGI","ELC"]): "technology_retail",
        frozenset(["HPG","HSG","NKG","TLH","VGS","SMC","POM"]): "steel_materials",
        frozenset(["GAS","PLX","PVD","PVS","BSR","OIL"]): "energy",
        frozenset(["VNM","SAB","MCH","MSN","QNS","KDC","SBT"]): "consumer",
        frozenset(["SSI","VND","HCM","VCI","BSI","FTS","MBS"]): "securities",
    }
    for s, sector in _MAP.items():
        if symbol in s:
            return sector
    return "diversified"


def _build_user_vars(alias: str, symbol: str, market: str,
                     timeframe: str, extra: dict) -> dict:
    vn_market = market or "Vietnam HOSE/HNX"
    skills    = SKILLS_COVERAGE.get(alias, [])
    archetype, n_agents = SWARM_AGENT_INFO.get(alias, ("the_strategists", 4))

    # ── Fix 1: Gán roles cụ thể cho từng agent ──────────────────────────
    # AGENT_ROLES đã define sẵn nhưng chưa được dùng — giờ inject vào user_vars
    # Server HKUDS dùng {{role_leader}}, {{role_researcher}}, ... trong prompt template
    role_assignments = {
        "role_leader":     AGENT_ROLES["Leader"],
        "role_researcher": AGENT_ROLES["Researcher"],
        "role_critic":     AGENT_ROLES["Critic"],
        "role_secretary":  AGENT_ROLES["Secretary"],
    }

    # ── Fix 2: Debate instructions — bắt buộc Critic phải challenge ─────
    # Không để agents dễ dàng đồng thuận — Critic PHẢI tìm điểm yếu trước
    # khi Leader được phép ra kết luận cuối
    debate_config = {
        "debate_mode":         "structured_challenge",
        "critic_instruction":  (
            "You are the Critic. Your role is to CHALLENGE the Researcher's thesis. "
            "You MUST identify at least 2 specific weaknesses or counter-arguments "
            "before any consensus can be reached. Do NOT agree immediately. "
            "Ask: What if the trend reverses? What does the bear case look like? "
            "Only after rigorous challenge should the Leader synthesize."
        ),
        "leader_instruction":  (
            "You are the Leader. Do NOT accept the first conclusion. "
            "You must weigh the Researcher's analysis AGAINST the Critic's objections. "
            "Your final verdict must explicitly address the strongest bear argument. "
            "If bull and bear cases are close, output NEUTRAL with clear conditions."
        ),
        "consensus_threshold": "Consensus requires Critic to be convinced, not just outvoted.",
    }

    base = {
        # ── Target context ───────────────────────────────────────────────
        "target":           f"{symbol} ({vn_market})",
        "market":           vn_market,
        "timeframe":        timeframe or "daily",
        "horizon":          "1 month",
        "goal":             f"Phan tich co phieu {symbol} tren {vn_market}",
        "risk_tolerance":   "medium",
        "agent_archetype":  archetype,
        "n_agents":         str(n_agents),
        "skills_used":      ", ".join(skills[:8]),
        # ── Sector & instrument context ──────────────────────────────────
        "commodity":        symbol,
        "crisis":           f"Tac dong den co phieu {symbol}",
        "factor_type":      "momentum, value, quality",
        "sector":           _infer_sector(symbol),
        "event_type":       "earnings, macro, policy",
        "view":             f"Neutral — phan tich khach quan {symbol}",
        "fund_type":        "equity",
        "strategy_type":    "balanced",
        "target_variable":  "5-day forward return",
        "portfolio":        f"{symbol} position",
        "review_period":    "last 30 days",
        "risk_profile":     "balanced",
        "pair_asset":       "VNINDEX",
        "benchmark":        "VNINDEX",
    }

    # Merge roles + debate config vào base
    base.update(role_assignments)
    base.update(debate_config)

    # ── Fix 3 (data injection): nếu extra chứa signal data từ /check ────
    # Caller (bot.py) có thể truyền vào extra_vars={
    #   "local_signals": "Bull:8/13 Bear:3/13 Verdict:NGHIENG_MUA",
    #   "price_context": "Close=64.5 RSI=52 MACD=+0.36 Vol=1.2xMA20",
    #   "support_resistance": "Ho tro:63.5 Khang cu:68.8",
    # }
    # Các key này sẽ được inject để agents có data thực thay vì general knowledge
    if extra.get("local_signals"):
        base["local_signal_context"] = (
            f"[VN Signal Bot /check output for {symbol}]\n"
            f"Signal summary: {extra['local_signals']}\n"
            f"Price context: {extra.get('price_context', 'N/A')}\n"
            f"Support/Resistance: {extra.get('support_resistance', 'N/A')}\n"
            f"NOTE: Use this as additional data point. Do NOT simply echo it — "
            f"critically evaluate whether you agree or disagree with each signal."
        )

    base.update({k: v for k, v in extra.items()
                 if k not in ("local_signals", "price_context", "support_resistance")})
    return base


# ══════════════════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if VIBE_API_KEY:
        h["Authorization"] = f"Bearer {VIBE_API_KEY}"
    return h


def is_available() -> bool:
    if not VIBE_API_URL:
        logger.warning("is_available: VIBE_API_URL chua duoc set")
        return False
    health_url = f"{VIBE_API_URL}/health"
    try:
        r = requests.get(health_url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return True
        logger.warning(f"is_available: /health HTTP {r.status_code}")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.error(f"is_available: Connection failed: {e}")
        return False
    except requests.exceptions.Timeout:
        logger.error(f"is_available: Timeout sau {REQUEST_TIMEOUT}s")
        return False
    except Exception as e:
        logger.error(f"is_available: {type(e).__name__}: {e}")
        return False


def list_presets() -> list[dict]:
    if not VIBE_API_URL:
        return []
    try:
        r = requests.get(f"{VIBE_API_URL}/swarm/presets",
                         headers=_headers(), timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"list_presets: {e}")
        return []


def resolve_alias(alias: str) -> Optional[str]:
    """Resolve alias → canonical alias (hỗ trợ quick alias + partial match)."""
    alias = alias.lower().strip()
    canonical = QUICK_ALIASES.get(alias, alias)
    if canonical in SWARM_ALIASES:
        return canonical
    for key in SWARM_ALIASES:
        if alias in key or key.startswith(alias):
            return key
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_local_context(symbol: str, check_result: dict) -> dict:
    """
    Build extra_vars từ kết quả /check để inject vào /vibe.
    Gọi trước start_swarm() nếu muốn agents có data thực của cổ phiếu.

    Args:
        symbol: mã cổ phiếu
        check_result: dict từ analyze_stock_full() hoặc run_vibe_agents()
            Cần có keys: signals, details, verdict, bull, bear, n, indicators

    Returns:
        dict extra_vars để truyền vào start_swarm(extra_vars=...)

    Usage trong bot.py:
        check_data = run_vibe_agents(symbol, df)
        ind        = get_indicators(symbol, df)
        extra      = build_local_context(symbol, {**check_data, "indicators": ind})
        run_id     = start_swarm("technical", symbol, extra_vars=extra)
    """
    signals    = check_result.get("signals", {})
    bull       = check_result.get("bull", 0)
    bear       = check_result.get("bear", 0)
    n          = check_result.get("n", 1) or 1
    verdict    = check_result.get("verdict", "")
    ind        = check_result.get("indicators", {})

    # ── Signal summary ────────────────────────────────────────────────────
    bull_engines = [k for k, v in signals.items() if v == 1]
    bear_engines = [k for k, v in signals.items() if v == -1]
    neutral_engines = [k for k, v in signals.items() if v == 0]
    conf_pct   = round(max(bull, bear) / n * 100) if n > 0 else 0

    local_signals = (
        f"Bull:{bull}/{n} Bear:{bear}/{n} Neutral:{n-bull-bear}/{n} "
        f"Verdict:{verdict} Confidence:{conf_pct}%\n"
        f"Bullish engines: {', '.join(bull_engines) or 'none'}\n"
        f"Bearish engines: {', '.join(bear_engines) or 'none'}\n"
        f"Neutral engines: {', '.join(neutral_engines) or 'none'}"
    )

    # ── Price context ─────────────────────────────────────────────────────
    price_parts = []
    if ind.get("current_price"):
        price_parts.append(f"Close={ind['current_price']}")
    if ind.get("rsi"):
        price_parts.append(f"RSI={round(ind['rsi'], 1)}")
    if ind.get("macd"):
        price_parts.append(f"MACD={round(ind['macd'], 4):+}")
    if ind.get("volume_ratio"):
        price_parts.append(f"Vol={round(ind['volume_ratio'], 2)}xMA20")
    if ind.get("change_1w_pct"):
        price_parts.append(f"1W={round(ind['change_1w_pct'], 1):+}%")
    if ind.get("change_1m_pct"):
        price_parts.append(f"1M={round(ind['change_1m_pct'], 1):+}%")
    price_context = " | ".join(price_parts) if price_parts else "N/A"

    # ── Support / Resistance ──────────────────────────────────────────────
    sr_parts = []
    if ind.get("support_20d"):
        sr_parts.append(f"Ho tro:{ind['support_20d']}")
    if ind.get("resistance_20d"):
        sr_parts.append(f"Khang cu:{ind['resistance_20d']}")
    if ind.get("bb_lower"):
        sr_parts.append(f"BB_low:{round(ind['bb_lower'],2)}")
    if ind.get("bb_upper"):
        sr_parts.append(f"BB_up:{round(ind['bb_upper'],2)}")
    support_resistance = " | ".join(sr_parts) if sr_parts else "N/A"

    return {
        "local_signals":      local_signals,
        "price_context":      price_context,
        "support_resistance": support_resistance,
    }


def start_swarm(alias: str, symbol: str,
                market: str = "Vietnam HOSE/HNX",
                timeframe: str = "daily",
                extra_vars: Optional[dict] = None) -> Optional[str]:
    if not VIBE_API_URL:
        logger.error("VIBE_API_URL chua set")
        return None
    canonical = resolve_alias(alias)
    if not canonical:
        logger.error(f"Alias '{alias}' khong ton tai")
        return None
    preset_name = SWARM_ALIASES[canonical]
    user_vars   = _build_user_vars(canonical, symbol, market, timeframe, extra_vars or {})
    try:
        r = requests.post(
            f"{VIBE_API_URL}/swarm/runs",
            headers=_headers(),
            json={"preset_name": preset_name, "user_vars": user_vars},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        run_id = r.json().get("id")
        logger.info(f"Swarm started: {preset_name} | {symbol} | run_id={run_id}")
        return run_id
    except Exception as e:
        logger.error(f"start_swarm error: {e}")
        return None


def poll_swarm(run_id: str,
               progress_callback: Optional[Callable] = None) -> dict:
    if not VIBE_API_URL:
        return {"status": "error", "error": "VIBE_API_URL chua set"}
    start          = time.time()
    last_done      = -1
    tasks_appeared = False
    while True:
        elapsed = time.time() - start
        if elapsed > POLL_TIMEOUT:
            return {"status": "error", "error": f"Timeout sau {int(POLL_TIMEOUT/60)} phut"}
        try:
            r = requests.get(f"{VIBE_API_URL}/swarm/runs/{run_id}",
                             headers=_headers(), timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning(f"poll retry ({int(elapsed)}s): {e}")
            time.sleep(POLL_INTERVAL)
            continue

        status  = data.get("status", "")
        tasks   = data.get("tasks", [])
        n_tasks = len(tasks)

        if n_tasks == 0 and not tasks_appeared and elapsed < 60:
            if progress_callback and int(elapsed) % 10 == 0:
                try: progress_callback(f"Khoi dong agents... ({int(elapsed)}s)")
                except Exception: pass
            time.sleep(3)
            continue
        if n_tasks > 0:
            tasks_appeared = True

        done = sum(1 for t in tasks
                   if t.get("status") in ("completed", "failed", "cancelled"))

        if progress_callback and done != last_done:
            last_done  = done
            in_prog    = [t.get("agent_id", t.get("id","?")) for t in tasks
                          if t.get("status") == "in_progress"]
            done_last3 = [t.get("agent_id", t.get("id","?")) for t in tasks
                          if t.get("status") == "completed"][-3:]
            try:
                lines = [f"Agents: {done}/{n_tasks} xong ({int(elapsed)}s)"]
                if in_prog:    lines.append(f"Dang chay: {', '.join(in_prog[:3])}")
                if done_last3: lines.append(f"Vua xong: {', '.join(done_last3)}")
                progress_callback("\n".join(lines))
            except Exception: pass

        if status == "completed":
            return {
                "status":          "completed",
                "final_report":    data.get("final_report", ""),
                "tasks":           tasks,
                "user_vars":       data.get("user_vars", {}),
                "elapsed_seconds": round(elapsed),
            }
        if status in ("failed", "cancelled"):
            failed = [t for t in tasks if t.get("status") == "failed"]
            err    = f" | {failed[0].get('error','')[:100]}" if failed else ""
            return {"status": status, "error": f"Swarm {status}{err}",
                    "tasks": tasks, "elapsed_seconds": round(elapsed)}
        time.sleep(POLL_INTERVAL)


def run_swarm_sync(alias: str, symbol: str,
                   market: str = "Vietnam HOSE/HNX",
                   timeframe: str = "daily",
                   extra_vars: Optional[dict] = None,
                   progress_callback: Optional[Callable] = None) -> dict:
    """Start + poll trong 1 lệnh. Hỗ trợ quick alias."""
    canonical = resolve_alias(alias)
    if not canonical:
        return {"status": "error",
                "error": f"Alias '{alias}' khong hop le. Dung /vibe de xem danh sach."}
    run_id = start_swarm(canonical, symbol, market, timeframe, extra_vars)
    if not run_id:
        return {"status": "error",
                "error": "Khong khoi dong duoc swarm. Kiem tra VIBE_API_URL va VIBE_API_KEY."}
    return poll_swarm(run_id, progress_callback=progress_callback)


def _condense_output(text: str, max_sentences: int = 3) -> str:
    """
    Thu gọn output của 1 agent thành 2-3 câu đủ nghĩa.
    Ưu tiên câu có keyword signal/verdict/warning.
    """
    import re as _re
    if not text or not text.strip():
        return ""
    raw_sents = _re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
    sents = [s.strip() for s in raw_sents if len(s.strip()) > 15]
    if not sents:
        return text.strip()[:300]
    if len(sents) <= max_sentences:
        return " ".join(sents)[:500]
    _KEY = ["signal","buy","sell","mua","ban","risk","rui ro","canh bao",
            "warning","bull","bear","uptrend","downtrend","ket luan",
            "conclusion","recommend","dissent","disagree","concern","lo ngai",
            "can than","luu y"]
    def _score(s):
        return sum(1 for k in _KEY if k in s.lower())
    first = sents[0]; last = sents[-1]; middle = sents[1:-1]
    if max_sentences <= 2 or not middle:
        return f"{first} {last}"[:500]
    best_mid = max(middle, key=_score)
    parts = [p for p in [first, best_mid, last] if p and p != first or p == first]
    # deduplicate while preserving order
    seen = set(); deduped = []
    for p in [first, best_mid, last]:
        if p not in seen: seen.add(p); deduped.append(p)
    return " ".join(deduped)[:500]


def _extract_role(task: dict, idx: int) -> str:
    """Trích role/name agent từ task dict, normalize tên."""
    import re as _re
    for field in ("agent_role", "role", "agent_name", "name", "agent_id", "id"):
        v = task.get(field)
        if v and isinstance(v, str) and v.strip():
            raw = _re.sub(r"^[\d_\-]+", "", v.strip()).strip()
            if raw:
                return raw[:30]
    return f"Agent-{idx + 1}"


_DISSENT_KW = {
    "disagree","dissent","concern","warning","caution","risk","however","but",
    "although","despite","pullback","correction","overvalued","overbought",
    "bearish","decline","weak","breakdown","resistance","reject","fail",
    "canh bao","lo ngai","rui ro","tuy nhien","nhung","giam","yeu","can than","luu y",
}
_BULL_KW = {
    "buy","bullish","uptrend","mua","tang","upside","breakout","strong",
    "momentum","support","nen","co hoi","tich luc","positive",
}


def _detect_sentiment(text: str) -> str:
    if not text: return "neutral"
    tl = text.lower()
    b = sum(1 for k in _BULL_KW if k in tl)
    d = sum(1 for k in _DISSENT_KW if k in tl)
    if b > d + 1: return "bull"
    if d > b + 1: return "bear"
    return "neutral"


def _build_meeting_transcript(tasks: list) -> str:
    """
    Build section "Diễn biến cuộc họp" từ tasks.
    Highlight bất đồng khi sentiment không đồng nhất.
    """
    completed = [t for t in tasks
                 if t.get("status") == "completed"
                 and (t.get("output") or t.get("summary") or t.get("reasoning"))]
    if not completed:
        return ""

    agent_items = []
    for i, t in enumerate(completed):
        raw = (t.get("output") or t.get("summary") or t.get("reasoning") or "").strip()
        if not raw:
            continue
        agent_items.append({
            "role":      _extract_role(t, i),
            "condensed": _condense_output(raw, 3),
            "sentiment": _detect_sentiment(raw),
        })
    if not agent_items:
        return ""

    sentiments  = {a["sentiment"] for a in agent_items}
    has_dissent = "bull" in sentiments and "bear" in sentiments

    lines = ["", "🗣️ DIEN BIEN CUOC HOP:", "─" * 34]
    if has_dissent:
        lines.append("⚡ Phat hien bat dong quan diem:")

    for a in agent_items:
        icon = "🟢" if a["sentiment"]=="bull" else "🔴" if a["sentiment"]=="bear" else "🔵"
        flag = " ⚠️" if has_dissent and a["sentiment"] == "bear" else ""
        lines.append(f"{icon} [{a['role']}]{flag}:")
        # Word-wrap tại 70 ký tự
        words = a["condensed"].split()
        line  = "   "
        for w in words:
            if len(line) + len(w) + 1 > 73:
                lines.append(line.rstrip()); line = "   " + w + " "
            else:
                line += w + " "
        if line.strip(): lines.append(line.rstrip())

    if has_dissent:
        bull_r = [a["role"] for a in agent_items if a["sentiment"] == "bull"]
        bear_r = [a["role"] for a in agent_items if a["sentiment"] == "bear"]
        lines += ["─" * 34,
                  f"📊 Dong thuan MUA: {', '.join(bull_r[:3]) or 'N/A'}",
                  f"📊 Phan bien BAN:  {', '.join(bear_r[:3]) or 'N/A'}"]
    return "\n".join(lines)


def format_swarm_result(result: dict, symbol: str, alias: str) -> str:
    """
    Format kết quả swarm thành text đầy đủ.

    Cấu trúc output:
      [HEADER]       — swarm name, agent count, thời gian
      [FINAL REPORT] — kết luận tổng hợp từ Leader
      [TRANSCRIPT]   — "Diễn biến cuộc họp": per-agent + highlight bất đồng

    KHÔNG truncate — caller (bot.py) dùng _smart_split() để tách nếu cần.
    """
    canonical = resolve_alias(alias) or alias
    label     = SWARM_LABELS.get(canonical, canonical.upper())
    status    = result.get("status", "unknown")

    if status != "completed":
        err = result.get("error", "Unknown error")
        return f"Vibe-Trading — {label}\nMa: {symbol} | {status}\nLoi: {err}"

    elapsed      = result.get("elapsed_seconds", 0)
    tasks        = result.get("tasks", [])
    n_done       = sum(1 for t in tasks if t.get("status") == "completed")
    n_total      = len(tasks)
    skills       = SKILLS_COVERAGE.get(canonical, [])
    archetype, _ = SWARM_AGENT_INFO.get(canonical, ("", 0))
    arch_desc    = AGENT_ARCHETYPES.get(archetype, "")

    header = (
        f"VIBE-TRADING — {label}\n"
        f"Ma: {symbol} | {n_done}/{n_total} agents | "
        f"{int(elapsed)//60}p{int(elapsed)%60}s\n"
        f"Skills: {', '.join(skills[:5])}{'...' if len(skills) > 5 else ''}\n"
        f"Role: {arch_desc}\n"
        f"{'='*36}\n\n"
    )

    final_report = result.get("final_report", "")
    if not final_report:
        completed_t = [t for t in tasks if t.get("status") == "completed"]
        if completed_t:
            last         = completed_t[-1]
            final_report = (f"[{_extract_role(last, len(completed_t)-1)}]\n"
                            f"{(last.get('output') or '')}")
        else:
            final_report = "Khong co bao cao."

    transcript = _build_meeting_transcript(tasks)
    return header + final_report + transcript


def get_swarm_info(alias: str) -> str:
    """Trả về thông tin chi tiết về 1 swarm (dùng cho /vibestatus <alias>)."""
    canonical = resolve_alias(alias)
    if not canonical:
        return f"Alias '{alias}' khong tim thay. Dung /vibe de xem danh sach."
    label            = SWARM_LABELS.get(canonical, canonical)
    skills           = SKILLS_COVERAGE.get(canonical, [])
    archetype, n_ag  = SWARM_AGENT_INFO.get(canonical, ("?", 0))
    arch_desc        = AGENT_ARCHETYPES.get(archetype, "")
    preset           = SWARM_ALIASES.get(canonical, "?")
    return "\n".join([
        f"Swarm: {label}",
        f"Preset: {preset}",
        f"Agents: {n_ag} ({archetype})",
        f"Archetype: {arch_desc}",
        f"Skills ({len(skills)}): {', '.join(skills)}",
    ])
