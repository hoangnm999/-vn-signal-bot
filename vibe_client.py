"""
vibe_client.py — HTTP client gọi Vibe-Trading API (29 swarms / 113 agents).

Env vars cần set trên Railway (bot service):
    VIBE_API_URL  = https://vibe-trading-xxxx.railway.app
    VIBE_API_KEY  = (giá trị API_AUTH_KEY bên Vibe service)

Dùng:
    from vibe_client import run_swarm_sync, format_swarm_result, SWARM_ALIASES
    result = run_swarm_sync("technical", "VCB", progress_callback=cb)
    text   = format_swarm_result(result, "VCB", "technical")
"""

from __future__ import annotations
import os, time, logging, requests
from typing import Optional, Callable

logger = logging.getLogger(__name__)

VIBE_API_URL     = os.environ.get("VIBE_API_URL", "").rstrip("/")
VIBE_API_KEY     = os.environ.get("VIBE_API_KEY", "")
POLL_TIMEOUT     = 900   # 15 phút
POLL_INTERVAL    = 8     # giây
REQUEST_TIMEOUT  = 30    # giây HTTP

# ── 29 Swarm aliases → preset name đầy đủ ────────────────────────────────────
# Nhóm TECHNICAL (phân tích kỹ thuật)
# Nhóm FUNDAMENTAL (phân tích cơ bản)
# Nhóm MACRO (vĩ mô)
# Nhóm QUANT (định lượng)
# Nhóm RISK (rủi ro)
# Nhóm PORTFOLIO (danh mục)
# Nhóm SENTIMENT (tâm lý)
# Nhóm CRYPTO (tiền mã hoá)
# Nhóm SPECIAL (chuyên biệt)

SWARM_ALIASES: dict[str, str] = {
    # ── Technical ──────────────────────────────────────────────────────────
    "technical":        "technical_analysis_panel",
    # ── Fundamental ────────────────────────────────────────────────────────
    "fundamental":      "fundamental_research_team",
    "earnings":         "earnings_research_desk",
    "equity":           "equity_research_team",
    # ── Macro ──────────────────────────────────────────────────────────────
    "macro":            "macro_strategy_forum",
    "macro_rates":      "macro_rates_fx_desk",
    "sector":           "sector_rotation_team",
    "geopolitical":     "geopolitical_war_room",
    "commodity":        "commodity_research_team",
    "global_alloc":     "global_allocation_committee",
    "global_equities":  "global_equities_desk",
    # ── Quant ──────────────────────────────────────────────────────────────
    "quant":            "quant_strategy_desk",
    "factor":           "factor_research_committee",
    "pairs":            "pairs_research_lab",
    "stat_arb":         "statistical_arbitrage_desk",
    "ml_quant":         "ml_quant_lab",
    "event_driven":     "event_driven_task_force",
    # ── Risk & Portfolio ───────────────────────────────────────────────────
    "risk":             "risk_committee",
    "investment":       "investment_committee",
    "portfolio":        "portfolio_review_board",
    # ── Sentiment ──────────────────────────────────────────────────────────
    "sentiment":        "sentiment_intelligence_team",
    "social_alpha":     "social_alpha_team",
    # ── Asset class ────────────────────────────────────────────────────────
    "etf":              "etf_allocation_desk",
    "credit":           "credit_research_team",
    "derivatives":      "derivatives_strategy_desk",
    "fund_select":      "fund_selection_panel",
    "convertible_bond": "convertible_bond_team",
    # ── Crypto ─────────────────────────────────────────────────────────────
    "crypto_research":  "crypto_research_lab",
    "crypto_trading":   "crypto_trading_desk",
}

# Human-readable labels
SWARM_LABELS: dict[str, str] = {
    "technical":       "Technical Analysis Panel (6 agents)",
    "fundamental":     "Fundamental Research Team (4 agents)",
    "earnings":        "Earnings Research Desk (4 agents)",
    "equity":          "Equity Research Team (4 agents)",
    "macro":           "Macro Strategy Forum (4 agents)",
    "macro_rates":     "Macro Rates & FX Desk (4 agents)",
    "sector":          "Sector Rotation Team (4 agents)",
    "geopolitical":    "Geopolitical War Room (4 agents)",
    "commodity":       "Commodity Research Team (3 agents)",
    "global_alloc":    "Global Allocation Committee (4 agents)",
    "global_equities": "Global Equities Desk (4 agents)",
    "quant":           "Quant Strategy Desk (4 agents)",
    "factor":          "Factor Research Committee (4 agents)",
    "pairs":           "Pairs Research Lab (4 agents)",
    "stat_arb":        "Statistical Arbitrage Desk (4 agents)",
    "ml_quant":        "ML Quant Lab (3 agents)",
    "event_driven":    "Event-Driven Task Force (3 agents)",
    "risk":            "Risk Committee (4 agents)",
    "investment":      "Investment Committee (4 agents)",
    "portfolio":       "Portfolio Review Board (4 agents)",
    "sentiment":       "Sentiment Intelligence Team (4 agents)",
    "social_alpha":    "Social Alpha Team (4 agents)",
    "etf":             "ETF Allocation Desk (4 agents)",
    "credit":          "Credit Research Team (4 agents)",
    "derivatives":     "Derivatives Strategy Desk (3 agents)",
    "fund_select":     "Fund Selection Panel (3 agents)",
    "convertible_bond":"Convertible Bond Team (4 agents)",
    "crypto_research": "Crypto Research Lab (4 agents)",
    "crypto_trading":  "Crypto Trading Desk (4 agents)",
}

# Nhóm để hiển thị trong /help
SWARM_GROUPS: dict[str, list[str]] = {
    "Technical":   ["technical"],
    "Fundamental": ["fundamental", "earnings", "equity"],
    "Macro":       ["macro", "macro_rates", "sector", "geopolitical",
                    "commodity", "global_alloc", "global_equities"],
    "Quant":       ["quant", "factor", "pairs", "stat_arb", "ml_quant", "event_driven"],
    "Risk/Portf":  ["risk", "investment", "portfolio"],
    "Sentiment":   ["sentiment", "social_alpha"],
    "Asset Class": ["etf", "credit", "derivatives", "fund_select", "convertible_bond"],
    "Crypto":      ["crypto_research", "crypto_trading"],
}

# user_vars builder — mỗi preset cần biến khác nhau
def _build_user_vars(alias: str, symbol: str, market: str,
                     timeframe: str, extra: dict) -> dict:
    """Build user_vars dict phù hợp với YAML template của từng preset."""
    vn_market = market or "Vietnam HOSE/HNX"
    base = {
        # Biến chung — hầu hết preset dùng ít nhất 1 trong số này
        "target":          f"{symbol} ({vn_market})",
        "market":          vn_market,
        "timeframe":       timeframe or "daily",
        "horizon":         "1 month",
        "goal":            f"Phan tich co phieu {symbol} tren {vn_market}",
        "risk_tolerance":  "medium",
        # Biến đặc biệt theo nhóm
        "commodity":       symbol,
        "crisis":          f"Tac dong den co phieu {symbol}",
        "factor_type":     "momentum,value,quality",
        "sector":          "financials",
        "event_type":      "earnings,macro,policy",
        "view":            f"Neutral to slightly bullish on {symbol}",
        "fund_type":       "equity",
        "strategy_type":   "balanced",
        "target_variable": "5-day forward return",
        "portfolio":       f"{symbol} position",
        "review_period":   "last 30 days",
        "risk_profile":    "balanced",
    }
    base.update(extra)
    return base


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if VIBE_API_KEY:
        h["Authorization"] = f"Bearer {VIBE_API_KEY}"
    return h


def is_available() -> bool:
    if not VIBE_API_URL:
        return False
    try:
        r = requests.get(f"{VIBE_API_URL}/health",
                         timeout=REQUEST_TIMEOUT)
        return r.status_code == 200
    except Exception:
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


# ── Core functions ────────────────────────────────────────────────────────────

def start_swarm(alias: str, symbol: str,
                market: str = "Vietnam HOSE/HNX",
                timeframe: str = "daily",
                extra_vars: Optional[dict] = None) -> Optional[str]:
    """
    Khởi động swarm run.
    Trả về run_id hoặc None nếu lỗi.
    """
    if not VIBE_API_URL:
        logger.error("VIBE_API_URL chua set")
        return None

    preset_name = SWARM_ALIASES.get(alias)
    if not preset_name:
        logger.error(f"Alias '{alias}' khong ton tai. Dung: {list(SWARM_ALIASES)}")
        return None

    user_vars = _build_user_vars(alias, symbol, market, timeframe, extra_vars or {})

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
    """
    Poll swarm run đến khi xong.
    Trả về dict: {status, final_report, tasks, elapsed_seconds}
    """
    if not VIBE_API_URL:
        return {"status": "error", "error": "VIBE_API_URL chua set"}

    start = time.time()
    last_done = -1

    while True:
        elapsed = time.time() - start
        if elapsed > POLL_TIMEOUT:
            return {"status": "error",
                    "error": f"Timeout sau {int(POLL_TIMEOUT/60)} phut"}

        try:
            r = requests.get(
                f"{VIBE_API_URL}/swarm/runs/{run_id}",
                headers=_headers(), timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning(f"poll retry: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        status = data.get("status", "")
        tasks  = data.get("tasks", [])
        total  = len(tasks)
        done   = sum(1 for t in tasks if t.get("status") in ("completed","failed"))

        # Progress update
        if progress_callback and done != last_done:
            last_done = done
            done_names = [t.get("agent_id","?") for t in tasks
                          if t.get("status") == "completed"][-3:]
            try:
                progress_callback(
                    f"Agents: {done}/{total} xong\n"
                    f"Vua hoan thanh: {', '.join(done_names) or '...'}"
                )
            except Exception:
                pass

        if status == "completed":
            return {
                "status":          "completed",
                "final_report":    data.get("final_report", ""),
                "tasks":           tasks,
                "user_vars":       data.get("user_vars", {}),
                "elapsed_seconds": round(elapsed),
            }
        if status in ("failed", "cancelled"):
            return {
                "status":  status,
                "error":   f"Swarm {status}",
                "tasks":   tasks,
                "elapsed_seconds": round(elapsed),
            }

        time.sleep(POLL_INTERVAL)


def run_swarm_sync(alias: str, symbol: str,
                   market: str = "Vietnam HOSE/HNX",
                   timeframe: str = "daily",
                   extra_vars: Optional[dict] = None,
                   progress_callback: Optional[Callable] = None) -> dict:
    """Start + poll trong 1 lệnh."""
    run_id = start_swarm(alias, symbol, market, timeframe, extra_vars)
    if not run_id:
        return {"status": "error",
                "error": "Khong khoi dong duoc swarm. Kiem tra VIBE_API_URL va VIBE_API_KEY."}
    return poll_swarm(run_id, progress_callback=progress_callback)


def format_swarm_result(result: dict, symbol: str, alias: str) -> str:
    """Format kết quả swarm thành text Telegram-ready (< 4000 ký tự)."""
    status = result.get("status", "unknown")

    if status != "completed":
        err = result.get("error", "Unknown error")
        label = SWARM_LABELS.get(alias, alias.upper())
        return (f"Vibe-Trading — {label}\n"
                f"Ma: {symbol} | Trang thai: {status}\n"
                f"Loi: {err}")

    elapsed  = result.get("elapsed_seconds", 0)
    tasks    = result.get("tasks", [])
    n_done   = sum(1 for t in tasks if t.get("status") == "completed")
    n_total  = len(tasks)
    label    = SWARM_LABELS.get(alias, alias.upper())

    header = (
        f"VIBE-TRADING — {label}\n"
        f"Ma: {symbol} | {n_done}/{n_total} agents | "
        f"{elapsed//60}p{elapsed%60}s\n"
        f"{'='*36}\n\n"
    )

    final_report = result.get("final_report", "")
    if not final_report:
        # Fallback: lấy output agent cuối
        completed = [t for t in tasks if t.get("status") == "completed"]
        if completed:
            last  = completed[-1]
            output = (last.get("output") or "")[:800]
            final_report = f"[{last.get('agent_id','?')}]\n{output}"
        else:
            final_report = "Khong co bao cao."

    max_body = 4000 - len(header) - 60
    body = final_report[:max_body]
    if len(final_report) > max_body:
        body += "\n...[xem day du tren Vibe-Trading dashboard]"

    return header + body
