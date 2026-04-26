"""
local_swarm.py — Local Swarm Panel v2.0
Hội đồng chuyên gia AI nội bộ, chiều sâu phân tích 90% Panel Swarm gốc.

KIẾN TRÚC v2.0:
  analyze_stock_full() → metadata (16 engines + indicators)
       → extract_technical_data() → TechnicalData (JSON đầy đủ)
       → SwarmOrchestrator (3 vòng tranh luận)
       → SwarmReport → format_swarm_report()

PROVIDERS (waterfall):
  DeepSeek → OpenRouter → Groq → Gemini → Ollama

TIMEOUT: 600s hard + per-call 120s (patched từ local_swarm_cmd.py)
"""

from __future__ import annotations

import os
import json
import re
import time
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DEBATE_ROUNDS    = 3
LLM_TIMEOUT      = 90        # giây — sẽ bị override bởi _patch_llm_timeout
LLM_MAX_TOKENS   = 800       # per expert response
MODERATOR_TOKENS = 2000      # moderator JSON
SWARM_VERSION    = "2.0"

_DEFAULT_MODELS = {
    "deepseek":   "deepseek-chat",
    "openrouter": "deepseek/deepseek-chat",
    "groq":       "llama-3.3-70b-versatile",
    "gemini":     "gemini-2.0-flash",
    "ollama":     "qwen2.5:7b",
}

# Biên độ giá hợp lệ: mọi mức giá phải nằm trong ±30% so với giá hiện tại
PRICE_BAND_PCT = 0.30


def _price_in_band(p: float, ref: float, band: float = PRICE_BAND_PCT) -> bool:
    """Kiểm tra p có nằm trong ±band% so với ref không."""
    if ref <= 0:
        return False
    return abs(p - ref) / ref <= band


def _sanitize_engine_detail(detail: str, ref_price: float) -> str:
    """
    Quét text engine detail, đánh dấu các số trông như giá cổ phiếu
    nhưng nằm ngoài ±30% ref_price bằng [~xxx] để LLM không dùng nhầm.
    Chỉ xử lý số >= ref_price*0.3 (tránh nhầm với %, RSI, v.v.)
    """
    if ref_price <= 0 or not detail:
        return detail
    lo        = ref_price * (1 - PRICE_BAND_PCT)
    hi        = ref_price * (1 + PRICE_BAND_PCT)
    threshold = ref_price * 0.30

    def _replace(m: re.Match) -> str:
        raw = m.group(0)
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            return raw
        if val < threshold:
            return raw          # quá nhỏ, không phải giá — giữ nguyên
        if lo <= val <= hi:
            return raw          # trong biên — giữ nguyên
        return f"[~{val:,.0f}]"  # ngoài biên — đánh dấu

    return re.sub(r"[\d,]+(?:\.\d+)?", _replace, detail)


# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL DATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_technical_data(symbol: str, meta: dict) -> dict:
    """
    Trích xuất toàn bộ dữ liệu kỹ thuật từ metadata analyze_stock_full().
    Trả về dict đầy đủ để truyền vào prompt từng chuyên gia.
    """
    ind     = meta.get("ind", {})
    verdict = meta.get("verdict", {})
    vibe    = meta.get("vibe_result", {})
    macro_v = meta.get("macro_v", {})
    news    = meta.get("news_data", {})
    comm    = meta.get("commodity_data", {})

    # ── Signals ─────────────────────────────────────────────────────────────
    signals: dict = {}
    raw_sigs = verdict.get("signals", {})
    if isinstance(raw_sigs, dict):
        for k, v in raw_sigs.items():
            try:
                signals[k] = int(v)
            except Exception:
                pass

    # ── Giá cơ sở — phải lấy trước để sanitize engine details ───────────────
    price   = float(ind.get("current_price", 0))
    sup_20d = float(ind.get("support_20d",    price * 0.95))
    res_20d = float(ind.get("resistance_20d", price * 1.05))
    bb_low  = float(ind.get("bb_lower",       price * 0.96))
    bb_up   = float(ind.get("bb_upper",       price * 1.04))
    ma20    = float(ind.get("ma20",           price))
    ma50_v  = ind.get("ma50")
    ma50    = float(ma50_v) if ma50_v else None
    atr     = price * 0.02

    # ── Lấy engine details — sanitize ngay giá ngoài biên ±30% ──────────────
    _raw_engine: dict = {}
    if isinstance(vibe, dict):
        raw_details = vibe.get("details", {})
        if isinstance(raw_details, dict):
            _raw_engine = {k: str(v)[:250] for k, v in raw_details.items()}
        ctx_details = verdict.get("context_details", {})
        if isinstance(ctx_details, dict):
            _raw_engine.update({k: str(v)[:250] for k, v in ctx_details.items()})
    # Sanitize: đánh dấu giá ngoài band trong text để LLM không dùng nhầm
    engine_details: dict = {
        k: _sanitize_engine_detail(v, price)
        for k, v in _raw_engine.items()
    }

    sr = _compute_sr_levels_v2(price, sup_20d, res_20d, bb_low, bb_up, ma20, ma50, atr)

    # ── Action plan từ /check — clamp về band hợp lệ ─────────────────────────
    ap       = verdict.get("action_plan", {})
    tp_raw   = float(ap.get("tp",        0))
    sl_raw   = float(ap.get("sl",        0))
    en_raw   = float(ap.get("entry_low", price))
    # Chỉ dùng nếu trong biên, không thì về 0 (moderator tự tính)
    tp_check = tp_raw if (tp_raw > 0 and _price_in_band(tp_raw, price)) else 0.0
    sl_check = sl_raw if (sl_raw > 0 and _price_in_band(sl_raw, price)) else 0.0
    en_check = en_raw if (en_raw > 0 and _price_in_band(en_raw, price)) else price

    news_headlines = (
        news.get("headlines", [])[:8]
        if isinstance(news, dict) and news.get("success") else []
    )

    return {
        "symbol":          symbol.upper(),
        "price":           price,
        "change_1d_pct":   float(ind.get("change_1d_pct", ind.get("change_1w_pct", 0))),
        "change_1w_pct":   float(ind.get("change_1w_pct", 0)),
        "change_1m_pct":   float(ind.get("change_1m_pct", 0)),
        "rsi":             float(ind.get("rsi", 50)),
        "macd":            float(ind.get("macd", 0)),
        "macd_hist":       float(ind.get("macd_hist", 0)),
        "macd_signal":     float(ind.get("macd_signal", 0)),
        "volume_ratio":    float(ind.get("volume_ratio", 1.0)),
        "ma20":            ma20,
        "ma50":            ma50,
        "bb_upper":        bb_up,
        "bb_lower":        bb_low,
        "bb_mid":          float(ind.get("bb_mid", ma20)),
        "atr":             atr,
        "support_20d":     sup_20d,
        "resistance_20d":  res_20d,
        "sr_levels":       sr,
        "tp_check":        tp_check,
        "sl_check":        sl_check,
        "entry_check":     en_check,
        "verdict_label":   verdict.get("verdict_label", "TRUNG LAP"),
        "confidence_pct":  float(verdict.get("confidence_pct", 50)),
        "bull_count":      int(verdict.get("bull_count", 0)),
        "bear_count":      int(verdict.get("bear_count", 0)),
        "active_agents":   int(verdict.get("active_agents", 0)),
        "contradictions":  verdict.get("contradictions", []),
        "signals":         signals,
        "engine_details":  engine_details,
        "market_regime":   macro_v.get("market_regime", "UNKNOWN") if isinstance(macro_v, dict) else "UNKNOWN",
        "macro_label":     macro_v.get("label", "") if isinstance(macro_v, dict) else "",
        "macro_detail":    macro_v.get("detail", "") if isinstance(macro_v, dict) else "",
        "news_headlines":  news_headlines,
        "news_sentiment":  verdict.get("news_sentiment", ""),
        "commodity_detail": comm.get("detail", "") if isinstance(comm, dict) else "",
        "commodity_signal": int(comm.get("signal", 0)) if isinstance(comm, dict) else 0,
    }


def _compute_sr_levels_v2(
    price: float, sup_20d: float, res_20d: float,
    bb_low: float, bb_up: float, ma20: float,
    ma50: Optional[float], atr: float,
) -> dict:
    """Tính S/R đa tầng, sắp xếp gần→xa, loại trùng 2.5%."""
    sup_cands: list[tuple[float, str]] = []
    res_cands: list[tuple[float, str]] = []

    def _add(p: float, reason: str):
        if p <= 0 or abs(p - price) / price > 0.25:
            return
        if p < price:
            sup_cands.append((p, reason))
        elif p > price:
            res_cands.append((p, reason))

    _add(sup_20d,  "Low 20 ngày")
    _add(res_20d,  "High 20 ngày")
    _add(bb_low,   "Bollinger Lower (2σ)")
    _add(bb_up,    "Bollinger Upper (2σ)")
    _add(ma20,     "SMA20")
    if ma50:
        _add(ma50, "SMA50")

    rng = res_20d - sup_20d
    if rng > 0:
        for fib, label in [
            (0.236, "Fib 23.6%"), (0.382, "Fib 38.2%"),
            (0.500, "Fib 50%"),   (0.618, "Fib 61.8%"),
            (0.786, "Fib 78.6%"),
        ]:
            _add(round(res_20d - fib * rng, 0), label)

    for mult, lbl in [(1.0, "ATR×1"), (1.5, "ATR×1.5"), (2.0, "ATR×2")]:
        _add(round(price - mult * atr, 0), f"Hỗ trợ {lbl}")
        _add(round(price + mult * atr, 0), f"Kháng cự {lbl}")

    def _dedupe(cands, desc: bool) -> list[dict]:
        srt = sorted(cands, key=lambda x: abs(x[0] - price))
        result: list[dict] = []
        for p, reason in srt:
            if any(abs(p - r["price"]) / max(r["price"], 1) < 0.025 for r in result):
                continue
            result.append({
                "price":    round(p, 0),
                "reason":   reason,
                "dist_pct": round((p - price) / price * 100, 1),
            })
            if len(result) >= 3:
                break
        while len(result) < 3:
            base  = result[-1]["price"] if result else price
            new_p = round(base * (1.03 if desc else 0.97), 0)
            result.append({
                "price":    new_p,
                "reason":   "Ước tính kỹ thuật",
                "dist_pct": round((new_p - price) / price * 100, 1),
            })
        return result

    return {
        "support":    _dedupe(sup_cands, desc=False),
        "resistance": _dedupe(res_cands, desc=True),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExpertOpinion:
    expert_id:   str
    role:        str
    stance:      str        # MUA / BAN / THEO DOI
    score:       int        # -5..+5
    confidence:  int        # 0-100
    key_points:  list[str]  # lý do cụ thể có số liệu
    concern:     str
    raw_text:    str


@dataclass
class DebateRound:
    round_num:  int
    exchanges:  list[dict]


@dataclass
class SwarmReport:
    symbol:            str
    timestamp:         str
    elapsed_s:         float
    llm_provider:      str
    llm_model:         str
    panel_verdict:     str
    panel_confidence:  float
    final_score:       float
    resonance_pct:     float
    consensus_level:   str
    vote_bull:         int
    vote_neutral:      int
    vote_bear:         int
    dissent_notes:     list[str]
    scenario_buy:      dict
    scenario_sell:     dict
    scenario_watch:    dict
    support_levels:    list
    resistance_levels: list
    rr_warning:        str
    main_risks:        list[str]
    key_catalysts:     list[str]
    shelf_life_days:   int
    expires_at:        str
    review_date:       str
    expert_opinions:   list[ExpertOpinion]
    debate_rounds:     list[DebateRound]
    moderator_summary: str
    moderator_raw_json: dict
    input_summary:     str
    tech_data:         dict


# ══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """Waterfall: DeepSeek → OpenRouter → Groq → Gemini → Ollama"""

    def __init__(self):
        self.provider, self.model = self._detect_provider()
        logger.info(f"[LLM] Provider={self.provider} Model={self.model}")

    def _detect_provider(self) -> tuple[str, str]:
        for provider, env_key in [
            ("deepseek",   "DEEPSEEK_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
            ("groq",       "GROQ_API_KEY"),
            ("gemini",     "GEMINI_API_KEY"),
        ]:
            key = os.environ.get(env_key, "").strip()
            if key:
                model = os.environ.get(f"{provider.upper()}_MODEL",
                                       _DEFAULT_MODELS[provider])
                return provider, model
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        try:
            r = requests.get(f"{ollama_url}/api/tags", timeout=3)
            if r.ok:
                return "ollama", os.environ.get("OLLAMA_MODEL", _DEFAULT_MODELS["ollama"])
        except Exception:
            pass
        raise RuntimeError(
            "Không tìm thấy LLM provider!\n"
            "Cần set: DEEPSEEK_API_KEY / OPENROUTER_API_KEY / GROQ_API_KEY / GEMINI_API_KEY"
        )

    def chat(self, system: str, user: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
        if self.provider == "deepseek":
            return self._openai_compat(
                "https://api.deepseek.com/v1/chat/completions",
                os.environ.get("DEEPSEEK_API_KEY", ""),
                system, user, max_tokens,
            )
        if self.provider == "openrouter":
            return self._openai_compat(
                "https://openrouter.ai/api/v1/chat/completions",
                os.environ.get("OPENROUTER_API_KEY", ""),
                system, user, max_tokens,
                extra_headers={
                    "HTTP-Referer": "https://github.com/vnsignalbot",
                    "X-Title": "VN Signal Bot",
                },
            )
        if self.provider == "groq":
            return self._openai_compat(
                "https://api.groq.com/openai/v1/chat/completions",
                os.environ.get("GROQ_API_KEY", ""),
                system, user, max_tokens,
            )
        if self.provider == "gemini":
            return self._gemini(
                os.environ.get("GEMINI_API_KEY", ""),
                system, user, max_tokens,
            )
        if self.provider == "ollama":
            return self._ollama(system, user, max_tokens)
        raise RuntimeError(f"Provider '{self.provider}' chưa implement")

    def _openai_compat(
        self, url: str, api_key: str,
        system: str, user: str, max_tokens: int,
        extra_headers: dict | None = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            **(extra_headers or {}),
        }
        payload = {
            "model":       self.model,
            "max_tokens":  max_tokens,
            "temperature": 0.6,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        r = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def _gemini(self, api_key: str, system: str, user: str, max_tokens: int) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.6},
        }
        r = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _ollama(self, system: str, user: str, max_tokens: int) -> str:
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        payload = {
            "model":  self.model,
            "prompt": f"<|system|>{system}<|user|>{user}<|assistant|>",
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.6},
        }
        r = requests.post(f"{url}/api/generate", json=payload, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json()["response"].strip()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

EXPERTS = [
    {
        "id":    "tech_analyst",
        "role":  "Chuyên gia Phân tích Kỹ thuật",
        "emoji": "📊",
        "focus": (
            "Bạn là chuyên gia phân tích kỹ thuật 15 năm kinh nghiệm HOSE/HNX. "
            "Trường phái: Classic TA, Ichimoku Cloud, Volume analysis, Candlestick patterns. "
            "Phân tích RSI, MACD, Bollinger, MA crossover, "
            "Ichimoku Tenkan/Kijun/Kumo, volume confirmation. "
            "KHÔNG được nói chung chung — PHẢI dẫn số liệu cụ thể từ dữ liệu được cấp."
        ),
        "skill_keys": ["TechnicalBasic", "Ichimoku", "Candlestick", "ElliottWave"],
    },
    {
        "id":    "smc_trader",
        "role":  "Smart Money Concepts Trader",
        "emoji": "💡",
        "focus": (
            "Bạn là SMC trader chuyên theo dòng tiền tổ chức trên HOSE. "
            "Trường phái: Order Blocks, Fair Value Gaps, BOS/ChoCH, Liquidity Pools, "
            "Wyckoff Method, Market Structure. "
            "Xác định điểm POI (Point of Interest) dựa trên SMC engine detail và structure. "
            "PHẢI dẫn chứng từ SMC engine detail — không tự suy diễn."
        ),
        "skill_keys": ["SMC", "Chanlun", "ElliottWave", "MultiFactor"],
    },
    {
        "id":    "macro_strategist",
        "role":  "Chiến lược gia Vĩ mô",
        "emoji": "🌏",
        "focus": (
            "Bạn là chiến lược gia vĩ mô từ Dragon Capital. "
            "Trường phái: Market Regime, VNMacro context, tác động Fed/SBV, "
            "tỷ giá USD/VND, giá vàng/dầu, News Sentiment. "
            "Nhìn big picture — skeptical với signals kỹ thuật đơn thuần. "
            "PHẢI phân tích market_regime, macro_label, và news headlines cụ thể."
        ),
        "skill_keys": ["MarketRegime", "VNMacro", "CommodityContext", "NewsSentiment"],
    },
    {
        "id":    "risk_manager",
        "role":  "Nhà Quản trị Rủi ro",
        "emoji": "🛡️",
        "focus": (
            "Bạn là risk manager chuyên nghiệp — LUÔN bảo vệ vốn trước tiên. "
            "Tính R:R chính xác, đặt SL/TP hợp lý dựa trên ATR và S/R, "
            "phân tích Volatility, xác định max drawdown tiềm năng. "
            "PHẢI phản đối bất kỳ kịch bản nào có R:R < 1.5. "
            "PHẢI dẫn chứng: ATR, BB width, khoảng cách đến SL, VaR."
        ),
        "skill_keys": ["Volatility", "TechnicalBasic", "MarketRegime"],
    },
    {
        "id":    "fundamental_filter",
        "role":  "Bộ lọc Cơ bản & Đa nhân tố",
        "emoji": "📋",
        "focus": (
            "Bạn là chuyên gia phân tích cơ bản và đa nhân tố. "
            "Trường phái: FundamentalFilter (P/E, P/B, ROE), MLStrategy (Random Forest signal), "
            "Seasonal patterns, sector rotation. "
            "PHẢI dẫn chứng từ FundamentalFilter detail và MLStrategy detail. "
            "Đặt câu hỏi: tại sao mua mã này mà không mua mã khác cùng ngành?"
        ),
        "skill_keys": ["FundamentalFilter", "MLStrategy", "Seasonal", "Harmonic"],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_data_block(td: dict) -> str:
    """Block dữ liệu kỹ thuật chung — truyền cho tất cả experts."""
    price = td["price"]
    sr    = td["sr_levels"]

    s_lines = "\n".join(
        f"  S{i+1}: {s['price']:,.0f}  ({s['dist_pct']:+.1f}%)  — {s['reason']}"
        for i, s in enumerate(sr["support"][:3])
    )
    r_lines = "\n".join(
        f"  R{i+1}: {r['price']:,.0f}  ({r['dist_pct']:+.1f}%)  — {r['reason']}"
        for i, r in enumerate(sr["resistance"][:3])
    )

    bull_engines = [k for k, v in td["signals"].items() if v > 0]
    bear_engines = [k for k, v in td["signals"].items() if v < 0]
    neu_engines  = [k for k, v in td["signals"].items() if v == 0]
    ma50_str     = f"{td['ma50']:,.0f}" if td["ma50"] else "N/A (<50 bars)"

    news_str = ""
    if td["news_headlines"]:
        news_str = "\nTIN TỨC GẦN ĐÂY:\n" + "\n".join(
            f"  • {h[:120]}" for h in td["news_headlines"][:4]
        )

    comm_str = ""
    if td["commodity_detail"]:
        sig_em = "🟢" if td["commodity_signal"] > 0 else "🔴" if td["commodity_signal"] < 0 else "⚪"
        comm_str = f"\nHÀNG HÓA ({sig_em}): {td['commodity_detail'][:120]}"

    return (
        f"=== DỮ LIỆU KỸ THUẬT: {td['symbol']} ===\n"
        f"Thời điểm: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"--- GIÁ & INDICATORS ---\n"
        f"Giá hiện tại  : {price:,.0f} VND\n"
        f"Thay đổi      : 1W={td['change_1w_pct']:+.1f}%  |  1M={td['change_1m_pct']:+.1f}%\n"
        f"RSI(14)       : {td['rsi']:.1f}"
        f"{'  ⚠️ Quá mua' if td['rsi']>70 else '  ⚠️ Quá bán' if td['rsi']<30 else ''}\n"
        f"MACD          : {td['macd']:.4f}  |  Hist={td['macd_hist']:+.4f}  |  Signal={td['macd_signal']:.4f}\n"
        f"Volume        : {td['volume_ratio']:.2f}x TB20\n"
        f"MA20={td['ma20']:,.0f}  |  MA50={ma50_str}\n"
        f"BB Upper={td['bb_upper']:,.0f}  |  Lower={td['bb_lower']:,.0f}  |  Mid={td['bb_mid']:,.0f}\n\n"
        f"--- HỖ TRỢ (gần → xa) ---\n{s_lines}\n\n"
        f"--- KHÁNG CỰ (gần → xa) ---\n{r_lines}\n\n"
        f"--- KẾT QUẢ 16 ENGINES ---\n"
        f"Phán quyết: {td['verdict_label']}  ({td['confidence_pct']:.0f}%)\n"
        f"Bull: {td['bull_count']}/{td['active_agents']} → {', '.join(bull_engines) or 'Không có'}\n"
        f"Bear: {td['bear_count']}/{td['active_agents']} → {', '.join(bear_engines) or 'Không có'}\n"
        f"Neutral: {len(neu_engines)} engines ({', '.join(neu_engines[:4])})\n\n"
        f"--- VĨ MÔ ---\n"
        f"Market Regime : {td['market_regime']}\n"
        f"Macro context : {td['macro_label']} {td['macro_detail'][:80] if td['macro_detail'] else ''}"
        f"{comm_str}{news_str}\n\n"
        f"--- MÂU THUẪN ---\n"
        f"{chr(10).join(td['contradictions'][:4]) if td['contradictions'] else 'Không có mâu thuẫn'}\n\n"
        f"--- ACTION PLAN TỪ /CHECK ---\n"
        f"TP={td['tp_check']:,.0f}  |  SL={td['sl_check']:,.0f}  |  Entry={td['entry_check']:,.0f}"
    ).strip()


def _get_skill_block(expert: dict, td: dict) -> str:
    """Trích skill details theo từng chuyên gia."""
    lines = []
    for sk in expert["skill_keys"]:
        detail = td["engine_details"].get(sk, "")
        if detail and len(detail) > 15:
            lines.append(f"  [{sk}] {detail[:200]}")
    return "\n".join(lines) if lines else "  (Không có detail từ engine này)"


def _build_round1_prompt(
    expert: dict, data_block: str, skill_block: str, td: dict
) -> tuple[str, str]:
    price_lo = round(td["price"] * (1 - PRICE_BAND_PCT), 0)
    price_hi = round(td["price"] * (1 + PRICE_BAND_PCT), 0)
    system = (
        f"Bạn là {expert['role']} trong Hội đồng Chuyên gia Phân tích Cổ phiếu VN.\n"
        f"{expert['focus']}\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. Dẫn chứng SỐ LIỆU CỤ THỂ — tên chỉ báo + mức giá + vị trí.\n"
        "   TỐT: 'RSI=62 tiếp cận quá mua, MACD hist=+0.0012 đang tăng'\n"
        "   XẤU: 'các chỉ báo cho thấy tích cực' (quá chung chung)\n"
        "2. STANCE=MUA/BAN → bắt buộc Entry, SL, TP, R:R cụ thể.\n"
        "3. R:R = (TP-Entry)/(Entry-SL). Nếu R:R < 1.5 → BẮT BUỘC đổi THEO DOI.\n"
        "4. SCORE từ -5 đến +5: âm=bearish, dương=bullish, 0=neutral.\n"
        "5. Chỉ phân tích theo trường phái của mình.\n"
        f"6. ⚠️ RÀNG BUỘC GIÁ CỨNG: Mọi mức giá bạn đề xuất (Entry, SL, TP, S/R)\n"
        f"   PHẢI nằm trong biên độ [{price_lo:,.0f} – {price_hi:,.0f}] (±30% giá hiện tại).\n"
        f"   CHỈ sử dụng dữ liệu được cung cấp. KHÔNG tự suy diễn giá.\n"
        f"   Nếu thiếu dữ liệu → ghi rõ 'không đủ thông tin' thay vì đoán.\n"
        f"   Số có dấu [~xxx] trong SKILL DETAILS = giá ngoài biên, KHÔNG được dùng.\n\n"
        "ĐỊNH DẠNG (bắt buộc, đúng thứ tự):\n"
        "STANCE: [MUA/BAN/THEO DOI]\n"
        "SCORE: [-5..+5]\n"
        "CONFIDENCE: [0-100]\n"
        "REASON: [1 câu có số liệu cụ thể]\n"
        "ENTRY: [giá hoặc 0]\n"
        "SL: [giá]  TP: [giá]  RR: [số]\n"
        "TRIGGER: [điều kiện kích hoạt có giá + indicator]\n"
        "RISK: [rủi ro có số liệu]\n"
        "KEY_DATA: [2-3 bullet từ skill của mình, mỗi bullet có số liệu]"
    )
    user = (
        f"Phân tích {td['symbol']} từ góc nhìn {expert['role']}:\n\n"
        f"{data_block}\n\n"
        f"=== SKILL DETAILS CHO {expert['role'].upper()} ===\n"
        f"{skill_block}\n\n"
        "Phân tích dựa trên SKILL DETAILS trên. "
        "Dẫn chứng số liệu cụ thể, tính R:R thực tế. "
        "R:R < 1.5 → BẮT BUỘC đổi THEO DOI."
    )
    return system, user


def _build_round2_prompt(
    expert: dict, data_block: str, skill_block: str,
    r1_opinions: list[ExpertOpinion], own_r1: ExpertOpinion, td: dict,
) -> tuple[str, str]:
    others = []
    for op in r1_opinions:
        if op.expert_id == expert["id"]:
            continue
        kp = op.key_points[0][:100] if op.key_points else op.concern[:80]
        others.append(f"[{op.role}] {op.stance}({op.confidence}%, Score={op.score:+d}) — {kp}")

    price_lo = round(td["price"] * (1 - PRICE_BAND_PCT), 0)
    price_hi = round(td["price"] * (1 + PRICE_BAND_PCT), 0)
    system = (
        f"Bạn là {expert['role']} — VÒNG 2 PHẢN BIỆN.\n"
        f"{expert['focus']}\n\n"
        "YÊU CẦU:\n"
        "1. Bảo vệ hoặc điều chỉnh stance sau khi nghe experts khác.\n"
        "2. PHẢN BIỆN trực tiếp ít nhất 1 expert đối lập — dẫn số liệu cụ thể.\n"
        "3. Ghi rõ: đồng ý với ai, điểm gì.\n"
        "4. Nếu Risk Manager chỉ ra R:R < 1.5 → BẮT BUỘC điều chỉnh.\n"
        "5. Cập nhật SCORE nếu cần.\n"
        f"6. ⚠️ Mọi giá đề xuất PHẢI trong [{price_lo:,.0f} – {price_hi:,.0f}]. "
        f"KHÔNG tự bịa giá. Số [~xxx] = giá ngoài biên, KHÔNG dùng.\n\n"
        "ĐỊNH DẠNG:\n"
        "STANCE: [MUA/BAN/THEO DOI]\n"
        "SCORE: [-5..+5]\n"
        "CONFIDENCE: [0-100]\n"
        "REASON: [1 câu có số liệu cập nhật]\n"
        "ENTRY: [giá]  SL: [giá]  TP: [giá]  RR: [số]\n"
        "REBUTS: [phản biện ai, điểm gì, dẫn số liệu]\n"
        "AGREES: [đồng ý với ai, điểm gì]\n"
        "TRIGGER: [điều kiện kích hoạt]\n"
        "RISK: [rủi ro cập nhật]"
    )
    user = (
        f"Vòng 1 của bạn: {own_r1.stance}({own_r1.confidence}%, Score={own_r1.score:+d})\n"
        f"Lý do: {own_r1.key_points[0] if own_r1.key_points else 'N/A'}\n\n"
        f"Ý kiến experts khác:\n" + "\n".join(others) +
        f"\n\nSkill của bạn:\n{skill_block[:400]}\n\n"
        "Cập nhật stance có dẫn chứng số liệu cụ thể. R:R < 1.5 → THEO DOI."
    )
    return system, user


def _build_round3_prompt(
    expert: dict, data_block: str, skill_block: str,
    r2_opinions: list[ExpertOpinion], own_r2: ExpertOpinion, td: dict,
) -> tuple[str, str]:
    others = [
        f"[{op.role}] {op.stance}({op.confidence}%, Score={op.score:+d})"
        for op in r2_opinions if op.expert_id != expert["id"]
    ]
    price_lo = round(td["price"] * (1 - PRICE_BAND_PCT), 0)
    price_hi = round(td["price"] * (1 + PRICE_BAND_PCT), 0)
    system = (
        f"Bạn là {expert['role']} — VÒNG 3 KẾT LUẬN CUỐI.\n"
        f"{expert['focus']}\n\n"
        "ĐÂY LÀ VÒNG CUỐI:\n"
        "1. Xác nhận stance cuối — không đổi trừ khi có lý do số liệu mạnh.\n"
        "2. Đề xuất kịch bản cụ thể nhất với Entry/SL/TP rõ ràng.\n"
        "3. Nêu 1 rủi ro quan trọng nhất mà các expert khác bỏ qua.\n"
        "4. Viết KEY_TAKEAWAY 1-2 câu — dẫn số liệu.\n"
        f"5. ⚠️ Mọi giá PHẢI trong [{price_lo:,.0f} – {price_hi:,.0f}]. "
        f"KHÔNG tự bịa. Số [~xxx] = ngoài biên, KHÔNG dùng.\n\n"
        "ĐỊNH DẠNG NGẮN GỌN:\n"
        "STANCE: [MUA/BAN/THEO DOI]\n"
        "SCORE: [-5..+5]\n"
        "CONFIDENCE: [0-100]\n"
        "FINAL_REASON: [1-2 câu có số liệu]\n"
        "SCENARIO: Entry=[giá] SL=[giá] TP=[giá] RR=[số]\n"
        "ALERT: [rủi ro bị bỏ qua]\n"
        "KEY_TAKEAWAY: [1-2 câu kết luận có số liệu]"
    )
    user = (
        f"Vòng 2 của bạn: {own_r2.stance}({own_r2.confidence}%, Score={own_r2.score:+d})\n"
        f"Experts khác: {' | '.join(others)}\n\n"
        "Xác nhận kịch bản cuối. R:R = (TP-Entry)/(Entry-SL). R:R < 1.5 → THEO DOI."
    )
    return system, user


def _build_moderator_prompt(
    td: dict, data_block: str,
    r1: list[ExpertOpinion],
    r2: list[ExpertOpinion],
    r3: list[ExpertOpinion],
) -> tuple[str, str]:
    final    = r3 or r2
    stances  = [op.stance for op in final]
    scores   = [op.score  for op in final]
    buy_c    = stances.count("MUA")
    sell_c   = stances.count("BAN")
    watch_c  = stances.count("THEO DOI")
    n_exp    = len(stances) or 5
    avg_score = sum(scores) / len(scores) if scores else 0
    dominant  = max(buy_c, sell_c, watch_c)
    avg_conf  = sum(op.confidence for op in final) / len(final) if final else 60
    base_conf = max(40, round((dominant / n_exp) * avg_conf))

    price  = td["price"]
    sr     = td["sr_levels"]
    s1     = sr["support"][0]["price"]
    r1_p   = sr["resistance"][0]["price"]
    tp_est = td["tp_check"] or r1_p
    sl_est = td["sl_check"] or s1
    rr_est = round((tp_est - price) / max(price - sl_est, 1), 2) if tp_est > sl_est else 2.0

    opinions_text = ""
    for op in final:
        em = next((e["emoji"] for e in EXPERTS if e["id"] == op.expert_id), "👤")
        kp = "; ".join(op.key_points[:2])[:150]
        opinions_text += (
            f"\n{em} {op.role}: {op.stance}({op.confidence}%, Score={op.score:+d})\n"
            f"  {kp}\n"
            f"  Rủi ro: {op.concern[:80]}\n"
        )

    sr_text = (
        f"S1={s1:,.0f} S2={sr['support'][1]['price']:,.0f} S3={sr['support'][2]['price']:,.0f}\n"
        f"R1={r1_p:,.0f} R2={sr['resistance'][1]['price']:,.0f} R3={sr['resistance'][2]['price']:,.0f}"
    )

    # Nếu dominant là THEO DOI → scenario_watch là ưu tiên
    watch_dominant = (watch_c >= buy_c and watch_c >= sell_c and watch_c > 0)
    sc_watch_prob  = 60 if watch_dominant else 30
    sc_buy_prob    = 15 if watch_dominant else 50
    sc_sell_prob   = 15 if watch_dominant else 20
    price_lo = round(price * (1 - PRICE_BAND_PCT), 0)
    price_hi = round(price * (1 + PRICE_BAND_PCT), 0)

    watch_rule = (
        f"9. ⚠️ LOGIC STANCE: {watch_c}/{n_exp} experts THEO DOI → "
        f"scenario_watch phải là ưu tiên (probability ~{sc_watch_prob}%).\n"
        f"   scenario_buy/sell chỉ là dự phòng (prob ~{sc_buy_prob}% / {sc_sell_prob}%).\n"
        if watch_dominant else ""
    )

    system = (
        "Bạn là MODERATOR Hội đồng Chuyên gia Phân tích Cổ phiếu VN.\n"
        "Tổng hợp 3 vòng tranh luận → JSON THUẦN (không markdown, không cắt).\n\n"
        "QUY TẮC:\n"
        f"1. panel_confidence >= {base_conf}.\n"
        "2. R:R = (tp1-entry)/(entry-sl). R:R < 1.5 → entry=0, rr_warning có nội dung.\n"
        "3. support_levels: S1 gần nhất dưới giá, cách nhau >= 2.5%.\n"
        "4. resistance_levels: R1 gần nhất trên giá, cách nhau >= 2.5%.\n"
        "5. main_risks và key_catalysts: PHẢI dẫn số liệu cụ thể.\n"
        "6. moderator_summary: TEXT THUẦN tiếng Việt 80-120 từ, KHÔNG JSON.\n"
        f"7. final_score = {round(avg_score, 1)}\n"
        f"8. ⚠️ RÀNG BUỘC GIÁ: Mọi entry/sl/tp/support/resistance PHẢI trong "
        f"[{price_lo:,.0f} – {price_hi:,.0f}] (±30% giá {price:,.0f}). KHÔNG tự bịa giá.\n"
        + watch_rule +
        "OUTPUT JSON THUẦN:\n"
        "{\n"
        '  "panel_verdict": "MUA|BAN|THEO DOI",\n'
        f'  "panel_confidence": {base_conf},\n'
        f'  "final_score": {round(avg_score, 1)},\n'
        '  "rr_warning": "",\n'
        '  "consensus_level": "DONG THUAN|PHAN BIEN|CHIA RE",\n'
        f'  "scenario_buy": {{"probability": {sc_buy_prob}, "entry": {round(price,0)}, '
        f'"tp1": {round(tp_est,0)}, "tp2": {round(tp_est*1.04,0)}, '
        f'"sl": {round(sl_est,0)}, "rr": {rr_est}, '
        '"trigger": "MUA khi [giá]+[volume]+[indicator]", "catalyst": "[cụ thể]"}},\n'
        f'  "scenario_sell": {{"probability": {sc_sell_prob}, "entry": 0, '
        f'"tp": {round(sl_est*0.97,0)}, "sl": {round(r1_p*1.02,0)}, "rr": 2.0, '
        '"trigger": "BÁN khi [điều kiện]", "catalyst": "[cụ thể]"}},\n'
        f'  "scenario_watch": {{"probability": {sc_watch_prob}, "trigger": "Chờ [điều kiện cụ thể]", '
        '"watch_for": "[tín hiệu có số liệu]"}},\n'
        f'  "support_levels": [{{"price": {s1}, "reason": "[lý do kỹ thuật]", '
        f'"dist_pct": {round((s1-price)/price*100,1)}}}, '
        f'{{"price": {sr["support"][1]["price"]}, "reason": "[lý do]", "dist_pct": 0}}, '
        f'{{"price": {sr["support"][2]["price"]}, "reason": "[lý do]", "dist_pct": 0}}],\n'
        f'  "resistance_levels": [{{"price": {r1_p}, "reason": "[lý do kỹ thuật]", '
        f'"dist_pct": {round((r1_p-price)/price*100,1)}}}, '
        f'{{"price": {sr["resistance"][1]["price"]}, "reason": "[lý do]", "dist_pct": 0}}, '
        f'{{"price": {sr["resistance"][2]["price"]}, "reason": "[lý do]", "dist_pct": 0}}],\n'
        '  "main_risks": ["[rủi ro 1 có số liệu]", "[rủi ro 2]", "[rủi ro 3]"],\n'
        '  "key_catalysts": ["[catalyst 1 có số liệu]", "[catalyst 2]"],\n'
        '  "shelf_life_days": 7,\n'
        '  "moderator_summary": "TEXT THUẦN 80-120 từ tiếng Việt, không JSON."\n'
        "}"
    )
    user = (
        f"Mã: {td['symbol']} | Giá: {price:,.0f}\n"
        f"Vote 3 vòng: MUA={buy_c} BAN={sell_c} THEO DOI={watch_c}\n"
        f"Avg Score: {avg_score:+.1f}  |  Confidence floor: {base_conf}%\n\n"
        f"Ý kiến vòng cuối:{opinions_text}\n"
        f"S/R:\n{sr_text}\n"
        f"TP={tp_est:,.0f}  SL={sl_est:,.0f}  R:R={rr_est:.2f}\n\n"
        "Trả về JSON ĐẦY ĐỦ, đóng tất cả ngoặc. "
        "moderator_summary là TEXT THUẦN, không JSON."
    )
    return system, user


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_expert_response(expert: dict, raw: str, round_num: int) -> ExpertOpinion:
    lines      = raw.strip().split("\n")
    stance     = "THEO DOI"
    score      = 0
    confidence = 50
    key_points: list[str] = []
    concern    = ""
    entry = sl = tp = rr = 0.0

    for line in lines:
        l  = line.strip()
        lu = l.upper()

        if lu.startswith("STANCE:"):
            s = lu.split(":", 1)[1].strip()
            stance = "MUA" if "MUA" in s else ("BAN" if "BAN" in s else "THEO DOI")

        elif lu.startswith("SCORE:"):
            try:
                score = int(float(re.sub(r"[^\d\.\-\+]", "", l.split(":", 1)[1].strip()[:5])))
                score = max(-5, min(5, score))
            except Exception:
                pass

        elif lu.startswith("CONFIDENCE:"):
            try:
                c = int("".join(filter(str.isdigit, l.split(":", 1)[1].strip()[:5])))
                confidence = max(0, min(100, c))
            except Exception:
                pass

        elif lu.startswith(("REASON:", "FINAL_REASON:")):
            txt = l.split(":", 1)[1].strip()
            if txt:
                key_points.insert(0, txt[:150])

        elif lu.startswith(("KEY_TAKEAWAY:", "KEY_DATA:")):
            txt = l.split(":", 1)[1].strip()
            if txt:
                key_points.append(txt[:150])

        elif lu.startswith("ENTRY:"):
            m = re.search(r"[\d]+\.?\d*", l.split(":", 1)[1].replace(",", ""))
            if m:
                try:
                    entry = float(m.group())
                except Exception:
                    pass

        elif lu.startswith("SL:"):
            parts = l.split("SL:", 1)[1]
            sl_part = parts.split("TP:")[0] if "TP:" in parts.upper() else parts
            m = re.search(r"[\d]+\.?\d*", sl_part.replace(",", ""))
            if m:
                try:
                    sl = float(m.group())
                except Exception:
                    pass
            if "TP:" in parts.upper():
                tp_part = parts.upper().split("TP:")[1].split("RR:")[0] if "RR:" in parts.upper() else parts.upper().split("TP:")[1]
                m2 = re.search(r"[\d]+\.?\d*", tp_part.replace(",", ""))
                if m2:
                    try:
                        tp = float(m2.group())
                    except Exception:
                        pass
            if "RR:" in parts.upper():
                rr_part = parts.upper().split("RR:")[1]
                m3 = re.search(r"[\d\.]+", rr_part)
                if m3:
                    try:
                        rr = float(m3.group())
                    except Exception:
                        pass

        elif lu.startswith("TP:"):
            m = re.search(r"[\d]+\.?\d*", l.split(":", 1)[1].replace(",", ""))
            if m:
                try:
                    tp = float(m.group())
                except Exception:
                    pass

        elif lu.startswith("RR:"):
            m = re.search(r"[\d\.]+", l.split(":", 1)[1])
            if m:
                try:
                    rr = float(m.group())
                except Exception:
                    pass

        elif lu.startswith("SCENARIO:"):
            for kv in l.split(":", 1)[1].split():
                if "=" in kv:
                    k_s, v_str = kv.split("=", 1)
                    m = re.search(r"[\d\.]+", v_str.replace(",", ""))
                    if m:
                        try:
                            val = float(m.group())
                            ku  = k_s.upper()
                            if ku == "ENTRY": entry = val
                            elif ku == "SL":  sl    = val
                            elif ku == "TP":  tp    = val
                            elif ku == "RR":  rr    = val
                        except Exception:
                            pass

        elif lu.startswith(("TRIGGER:", "TRIGGER CẬP NHẬT:")):
            txt = l.split(":", 1)[1].strip()[:150]
            if txt:
                key_points.append(f"Trigger: {txt}")

        elif lu.startswith(("RISK:", "ALERT:")):
            txt = l.split(":", 1)[1].strip()[:150]
            if txt:
                concern = txt

        elif lu.startswith("REBUTS:"):
            txt = l.split(":", 1)[1].strip()[:120]
            if txt:
                key_points.append(f"Phản biện: {txt}")

        elif l.startswith("- ") and len(key_points) < 5:
            key_points.append(l[2:].strip()[:120])

    # Kiểm tra sơ bộ: nếu entry/sl/tp spread quá rộng (>3x) → LLM có thể sai mã
    if entry > 0 and sl > 0 and tp > 0:
        ratio_max = max(entry, sl, tp) / max(min(entry, sl, tp), 1)
        if ratio_max > 3.0:
            entry = sl = tp = 0.0
            concern = concern or "Giá entry/sl/tp spread bất thường — có thể nhầm mã"

    # Validate R:R
    if entry > 0 and tp > 0 and sl > 0 and entry != sl:
        rr_calc = round(abs(tp - entry) / abs(entry - sl), 2)
        if stance == "MUA" and (entry <= sl or entry >= tp or rr_calc < 1.5):
            stance = "THEO DOI"
            concern = concern or f"R:R={rr_calc:.1f} hoặc giá không hợp lệ"
            entry  = 0
            score  = max(0, score - 1)
        elif stance == "BAN" and (entry >= sl or entry <= tp or rr_calc < 1.5):
            stance = "THEO DOI"
            concern = concern or f"R:R={rr_calc:.1f} không hợp lệ"
            entry  = 0

    if not key_points:
        for sent in raw.split("."):
            s = sent.strip()
            if len(s) > 20:
                key_points.append(s[:150])
                break

    return ExpertOpinion(
        expert_id  = expert["id"],
        role       = expert["role"],
        stance     = stance,
        score      = score,
        confidence = confidence,
        key_points = (key_points or ["Không có lý do"])[:5],
        concern    = concern or "Xem phân tích chi tiết",
        raw_text   = raw,
    )


def _parse_moderator_json(raw: str, td: dict) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:]).rsplit("```", 1)[0].strip()

    start = text.find("{")
    end   = text.rfind("}") + 1

    if start != -1 and end > start:
        json_text = text[start:end]
    elif start != -1:
        json_text = text[start:]
        open_n = json_text.count("{") - json_text.count("}")
        json_text += "}" * max(0, open_n)
        json_text = re.sub(r",\s*}", "}", json_text)
        json_text = re.sub(r",\s*]", "]", json_text)
    else:
        json_text = "{}"

    data = None
    for attempt in [json_text, json_text + "}", json_text + "}}"]:
        try:
            data = json.loads(attempt)
            break
        except json.JSONDecodeError:
            pass

    if data is None:
        logger.warning("Moderator JSON parse failed — using fallback")
        data = _moderator_fallback(td)

    summary = str(data.get("moderator_summary", ""))
    if "{" in summary and ("panel_verdict" in summary or '"entry"' in summary):
        summary = summary[:summary.find("{")].strip()
    data["moderator_summary"] = summary[:600] or "Hội đồng đã hoàn tất phân tích 3 vòng."
    data["panel_confidence"]  = max(25.0, float(data.get("panel_confidence", 60)))
    data["shelf_life_days"]   = max(5, int(data.get("shelf_life_days", 7)))

    data, rr_warning = _validate_rr(data, td)
    data["rr_warning"] = rr_warning
    data = _fix_sr(data, td)
    data = _validate_prices_in_output(data, td)  # kiểm tra giá ngoài biên
    return data


def _validate_rr(data: dict, td: dict) -> tuple[dict, str]:
    rr_warning = ""
    for sc_key in ["scenario_buy", "scenario_sell"]:
        sc = data.get(sc_key, {})
        if not sc:
            continue
        entry = float(sc.get("entry", 0) or 0)
        sl    = float(sc.get("sl",    0) or 0)
        tp1   = float(sc.get("tp1",   sc.get("tp", 0)) or 0)
        if entry <= 0 or sl <= 0 or tp1 <= 0:
            continue
        is_buy = sc_key == "scenario_buy"
        if (is_buy and (entry <= sl or entry >= tp1)) or \
           (not is_buy and (entry >= sl or entry <= tp1)):
            sc["entry"] = 0
            rr_warning  = f"Entry không hợp lệ kịch bản {'MUA' if is_buy else 'BÁN'}"
        else:
            rr_calc = round(abs(tp1 - entry) / abs(entry - sl), 2)
            sc["rr"] = rr_calc
            if rr_calc < 1.5:
                rr_warning = f"R:R={rr_calc:.1f} < 1.5 — chưa tối ưu"
                sc["entry"] = 0
                data["panel_confidence"] = max(25, data.get("panel_confidence", 60) - 10)
            elif rr_calc < 2.0 and not rr_warning:
                rr_warning = f"R:R={rr_calc:.1f} dưới ngưỡng tối ưu 2.0"
        data[sc_key] = sc
    return data, rr_warning


def _validate_prices_in_output(data: dict, td: dict) -> dict:
    """
    Kiểm tra tính nhất quán: entry/sl/tp trong scenarios phải nằm trong ±30% giá.
    Nếu không → reset về 0 và gắn rr_warning.
    Cũng enforce scenario_watch priority nếu panel_verdict là THEO DOI.
    """
    price    = td["price"]
    warnings = []

    for sc_key in ["scenario_buy", "scenario_sell"]:
        sc = data.get(sc_key, {})
        if not sc:
            continue
        for field in ["entry", "sl", "tp1", "tp", "tp2"]:
            val = float(sc.get(field, 0) or 0)
            if val > 0 and not _price_in_band(val, price):
                logger.warning(
                    f"[PriceGuard] {sc_key}.{field}={val:,.0f} ngoài band "
                    f"±{PRICE_BAND_PCT*100:.0f}% (giá={price:,.0f}) → reset 0"
                )
                sc[field] = 0
                warnings.append(
                    f"{sc_key}.{field}={val:,.0f} ngoài biên ±30%"
                )
        # Nếu entry đã bị reset → xóa luôn rr
        if float(sc.get("entry", 0) or 0) == 0:
            sc["entry"] = 0
        data[sc_key] = sc

    if warnings:
        existing = data.get("rr_warning", "")
        extra    = " | ".join(warnings)
        data["rr_warning"] = (f"{existing} | {extra}" if existing else extra)[:200]

    # Enforce: nếu panel_verdict là THEO DOI,
    # scenario_watch phải có probability cao nhất
    verdict = data.get("panel_verdict", "")
    if verdict == "THEO DOI":
        sc_w = data.get("scenario_watch", {})
        sc_b = data.get("scenario_buy",   {})
        sc_s = data.get("scenario_sell",  {})
        p_w  = int(sc_w.get("probability", 0))
        p_b  = int(sc_b.get("probability", 0))
        p_s  = int(sc_s.get("probability", 0))
        if p_w <= p_b or p_w <= p_s:
            # Điều chỉnh: watch lấy phần lớn nhất
            total     = max(p_w + p_b + p_s, 100)
            new_p_w   = max(50, round(total * 0.60))
            remaining = 100 - new_p_w
            new_p_b   = remaining // 2
            new_p_s   = remaining - new_p_b
            if sc_w: sc_w["probability"] = new_p_w
            if sc_b: sc_b["probability"] = new_p_b
            if sc_s: sc_s["probability"] = new_p_s
            data["scenario_watch"] = sc_w
            data["scenario_buy"]   = sc_b
            data["scenario_sell"]  = sc_s
            logger.info(
                f"[ScenarioFix] THEO DOI dominant → watch={new_p_w}% "
                f"buy={new_p_b}% sell={new_p_s}%"
            )

    return data


def _fix_sr(data: dict, td: dict) -> dict:
    price    = td["price"]
    computed = td["sr_levels"]

    def _fix(levels: list, direction: str) -> list:
        result: list[dict] = []
        if isinstance(levels, list):
            for lv in levels:
                if not isinstance(lv, dict):
                    continue
                p = float(lv.get("price", 0))
                if p <= 0:
                    continue
                if direction == "support"    and p >= price: continue
                if direction == "resistance" and p <= price: continue
                if any(abs(p - r["price"]) / max(r["price"], 1) < 0.025 for r in result):
                    continue
                result.append({
                    "price":    round(p, 0),
                    "reason":   str(lv.get("reason", ""))[:80],
                    "dist_pct": round((p - price) / price * 100, 1),
                })
            result.sort(key=lambda x: abs(x["price"] - price))
        for fb in computed[direction]:
            if len(result) >= 3: break
            if not any(abs(fb["price"] - r["price"]) / max(r["price"], 1) < 0.025 for r in result):
                result.append(fb)
        return result[:3]

    data["support_levels"]    = _fix(data.get("support_levels",    []), "support")
    data["resistance_levels"] = _fix(data.get("resistance_levels", []), "resistance")
    return data


def _moderator_fallback(td: dict) -> dict:
    price = td["price"]
    sr    = td["sr_levels"]
    s1    = sr["support"][0]["price"]
    r1_p  = sr["resistance"][0]["price"]
    tp_e  = td["tp_check"] or r1_p
    sl_e  = td["sl_check"] or s1
    rr_e  = round((tp_e - price) / max(price - sl_e, 1), 2)
    vl    = td["verdict_label"].upper()
    verdict_word = "MUA" if "MUA" in vl else ("BAN" if "BAN" in vl else "THEO DOI")

    return {
        "panel_verdict":    verdict_word,
        "panel_confidence": max(40.0, float(td["confidence_pct"])),
        "final_score":      float(td["bull_count"] - td["bear_count"]),
        "rr_warning":       "" if rr_e >= 1.5 else f"R:R={rr_e:.1f} < 1.5",
        "consensus_level":  "PHAN BIEN",
        "scenario_buy": {
            "probability": max(td["bull_count"] * 12, 35),
            "entry": round(price, 0) if rr_e >= 1.5 else 0,
            "tp1": round(tp_e, 0), "tp2": round(tp_e * 1.04, 0),
            "sl":  round(sl_e, 0), "rr":  rr_e,
            "trigger":  f"MUA khi giá close trên {r1_p:,.0f} + Volume > 1.5x TB20",
            "catalyst": f"Breakout {r1_p:,.0f} xác nhận uptrend",
        },
        "scenario_sell": {
            "probability": max(td["bear_count"] * 12, 20),
            "entry": 0, "tp": round(sl_e * 0.97, 0),
            "sl": round(r1_p * 1.02, 0), "rr": 2.0,
            "trigger":  f"BÁN khi phá vỡ {s1:,.0f} + Volume > 2x TB20",
            "catalyst": f"Breakdown {s1:,.0f} xác nhận downtrend",
        },
        "scenario_watch": {
            "probability": 30,
            "trigger":   f"Theo dõi vùng {round(price*0.99,0):,.0f}–{round(price*1.01,0):,.0f}",
            "watch_for": f"RSI={td['rsi']:.0f}, chờ Volume > 1.5x TB20 + close qua MA20={td['ma20']:,.0f}",
        },
        "support_levels":    sr["support"],
        "resistance_levels": sr["resistance"],
        "main_risks": [
            f"Giá phá {s1:,.0f} → xác nhận downtrend (RSI={td['rsi']:.0f})",
            f"Volume={td['volume_ratio']:.1f}x TB20 chưa đủ xác nhận xu hướng",
            f"Market Regime {td['market_regime']} — rủi ro đảo chiều",
        ],
        "key_catalysts": [
            f"Breakout {r1_p:,.0f} + Volume > 1.5x TB20 xác nhận",
            f"Kết quả kinh doanh {td['symbol']} — catalyst nội tại",
        ],
        "shelf_life_days": 7,
        "moderator_summary": (
            f"Fallback: {td['verdict_label']}, RSI={td['rsi']:.0f}, "
            f"Vol={td['volume_ratio']:.1f}x. Bull={td['bull_count']}/{td['active_agents']}, "
            f"Bear={td['bear_count']}/{td['active_agents']}. "
            f"Regime={td['market_regime']}."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SWARM ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class SwarmOrchestrator:
    def __init__(self, llm: LLMClient, progress_cb=None):
        self.llm  = llm
        self._cb  = progress_cb
        self._idx = 0

    def _progress(self, msg: str):
        self._idx += 1
        if self._cb:
            try:
                self._cb(f"[{self._idx}] {msg}")
            except Exception:
                pass
        logger.info(f"[Swarm] {msg}")

    def _call_expert(
        self, expert: dict, system: str, user: str, round_num: int
    ) -> ExpertOpinion:
        t0 = time.time()
        try:
            raw     = self.llm.chat(system, user, LLM_MAX_TOKENS)
            op      = _parse_expert_response(expert, raw, round_num)
            elapsed = round(time.time() - t0, 1)
            logger.info(f"[Expert][R{round_num}] {expert['id']} → {op.stance} Score={op.score:+d} ({elapsed}s)")
            return op
        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            logger.warning(f"[Expert][R{round_num}] {expert['id']} FAIL ({elapsed}s): {e}")
            return ExpertOpinion(
                expert_id  = expert["id"],
                role       = expert["role"],
                stance     = "THEO DOI",
                score      = 0,
                confidence = 40,
                key_points = [f"Lỗi LLM: {str(e)[:80]}"],
                concern    = "N/A",
                raw_text   = str(e),
            )

    def run(self, td: dict) -> SwarmReport:
        t0         = time.time()
        symbol     = td["symbol"]
        data_block = _build_data_block(td)

        # ── VÒNG 1 ──────────────────────────────────────────────────────────
        self._progress(f"📊 Vòng 1/3 — {len(EXPERTS)} chuyên gia phân tích độc lập...")
        r1: list[ExpertOpinion] = []
        for expert in EXPERTS:
            self._progress(f"  {expert['emoji']} {expert['role']}...")
            skill_block = _get_skill_block(expert, td)
            sys_p, usr_p = _build_round1_prompt(expert, data_block, skill_block, td)
            op = self._call_expert(expert, sys_p, usr_p, 1)
            r1.append(op)
            self._progress(f"  → {expert['emoji']} {op.stance} Score={op.score:+d} ({op.confidence}%)")

        # ── VÒNG 2 ──────────────────────────────────────────────────────────
        self._progress("💬 Vòng 2/3 — Phản biện chéo có dẫn chứng...")
        r2: list[ExpertOpinion] = []
        debate_r2 = DebateRound(round_num=2, exchanges=[])
        for expert in EXPERTS:
            own_r1 = next((op for op in r1 if op.expert_id == expert["id"]), r1[0])
            self._progress(f"  {expert['emoji']} {expert['role']} phản biện...")
            skill_block  = _get_skill_block(expert, td)
            sys_p, usr_p = _build_round2_prompt(expert, data_block, skill_block, r1, own_r1, td)
            op = self._call_expert(expert, sys_p, usr_p, 2)
            r2.append(op)
            change = " ⚡ ĐỔI!" if own_r1.stance != op.stance else ""
            self._progress(f"  → {expert['emoji']}: {own_r1.stance}→{op.stance}{change}")
            debate_r2.exchanges.append({
                "expert":    expert["id"],
                "role":      expert["role"],
                "stance_r1": own_r1.stance,
                "stance_r2": op.stance,
                "changed":   own_r1.stance != op.stance,
                "text":      op.raw_text[:200],
            })

        # ── VÒNG 3 ──────────────────────────────────────────────────────────
        self._progress("🏁 Vòng 3/3 — Kết luận cuối + kịch bản...")
        r3: list[ExpertOpinion] = []
        debate_r3 = DebateRound(round_num=3, exchanges=[])
        for expert in EXPERTS:
            own_r2 = next((op for op in r2 if op.expert_id == expert["id"]), r2[0])
            self._progress(f"  {expert['emoji']} {expert['role']} kết luận...")
            skill_block  = _get_skill_block(expert, td)
            sys_p, usr_p = _build_round3_prompt(expert, data_block, skill_block, r2, own_r2, td)
            op = self._call_expert(expert, sys_p, usr_p, 3)
            r3.append(op)
            self._progress(f"  → {expert['emoji']} FINAL: {op.stance} Score={op.score:+d}")
            debate_r3.exchanges.append({
                "expert":    expert["id"],
                "role":      expert["role"],
                "stance_r2": own_r2.stance,
                "stance_r3": op.stance,
                "changed":   own_r2.stance != op.stance,
                "text":      op.raw_text[:200],
            })

        # ── MODERATOR ────────────────────────────────────────────────────────
        self._progress("🔍 Moderator tổng hợp 3 vòng → JSON...")
        t_mod = time.time()
        try:
            sys_p, usr_p = _build_moderator_prompt(td, data_block, r1, r2, r3)
            raw_mod  = self.llm.chat(sys_p, usr_p, MODERATOR_TOKENS)
            mod_data = _parse_moderator_json(raw_mod, td)
            logger.info(f"[Moderator] done in {round(time.time()-t_mod,1)}s")
        except Exception as e:
            logger.warning(f"[Moderator] FAIL: {e}")
            mod_data = _moderator_fallback(td)
            mod_data["rr_warning"] = ""

        # ── BUILD REPORT ─────────────────────────────────────────────────────
        elapsed = round(time.time() - t0, 1)

        stances_final = [op.stance for op in r3]
        scores_final  = [op.score  for op in r3]
        vote_bull     = stances_final.count("MUA")
        vote_bear     = stances_final.count("BAN")
        vote_neutral  = stances_final.count("THEO DOI")
        n_exp         = len(r3) or 5

        final_score = float(mod_data.get(
            "final_score",
            sum(scores_final) / max(len(scores_final), 1),
        ))
        final_score = max(-5.0, min(5.0, final_score))

        dominant_stance = max(stances_final, key=stances_final.count) if stances_final else "THEO DOI"
        dom_ops         = [op for op in r3 if op.stance == dominant_stance]
        dom_avg_conf    = sum(op.confidence for op in dom_ops) / len(dom_ops) if dom_ops else 60
        resonance_pct   = round((len(dom_ops) / n_exp) * dom_avg_conf, 1)
        dom_count       = len(dom_ops)

        conf_floor = max(40, round((dom_count / n_exp) * dom_avg_conf))
        if dom_count == n_exp:       conf_floor = max(conf_floor, 65)
        elif dom_count >= n_exp - 1: conf_floor = max(conf_floor, 55)
        final_confidence = max(float(mod_data.get("panel_confidence", 60)), conf_floor)

        consensus_level = (
            "DONG THUAN" if dom_count == n_exp else
            "PHAN BIEN"  if dom_count >= n_exp - 1 else
            "CHIA RE"
        )

        dissent_notes: list[str] = []
        for op in r3:
            if op.stance != dominant_stance:
                em   = next((e["emoji"] for e in EXPERTS if e["id"] == op.expert_id), "👤")
                note = op.key_points[0][:100] if op.key_points else op.concern[:80]
                dissent_notes.append(f"{em} {op.role}: {note}")

        shelf_life  = max(5, int(mod_data.get("shelf_life_days", 7)))
        review_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        expires_at  = (datetime.now() + timedelta(days=shelf_life)).strftime("%Y-%m-%d")

        return SwarmReport(
            symbol            = symbol,
            timestamp         = datetime.now().strftime("%Y-%m-%d %H:%M"),
            elapsed_s         = elapsed,
            llm_provider      = self.llm.provider,
            llm_model         = self.llm.model,
            panel_verdict     = mod_data.get("panel_verdict", dominant_stance),
            panel_confidence  = final_confidence,
            final_score       = final_score,
            resonance_pct     = resonance_pct,
            consensus_level   = consensus_level,
            vote_bull         = vote_bull,
            vote_neutral      = vote_neutral,
            vote_bear         = vote_bear,
            dissent_notes     = dissent_notes,
            scenario_buy      = mod_data.get("scenario_buy",   {}),
            scenario_sell     = mod_data.get("scenario_sell",  {}),
            scenario_watch    = mod_data.get("scenario_watch", {}),
            support_levels    = mod_data.get("support_levels",    []),
            resistance_levels = mod_data.get("resistance_levels", []),
            rr_warning        = mod_data.get("rr_warning", ""),
            main_risks        = mod_data.get("main_risks",     []),
            key_catalysts     = mod_data.get("key_catalysts",  []),
            shelf_life_days   = shelf_life,
            expires_at        = expires_at,
            review_date       = review_date,
            expert_opinions   = r3,
            debate_rounds     = [debate_r2, debate_r3],
            moderator_summary = mod_data.get("moderator_summary", ""),
            moderator_raw_json= mod_data,
            input_summary     = (
                f"{symbol} | {td['verdict_label']} {td['confidence_pct']:.0f}% | "
                f"RSI={td['rsi']:.0f} Vol={td['volume_ratio']:.1f}x"
            ),
            tech_data = td,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

_SEP  = "═" * 32
_SEP2 = "─" * 32


def _score_bar(score: float) -> str:
    filled = round(abs(score))
    empty  = 5 - filled
    return ("🟢" * filled + "⬜" * empty) if score >= 0 else ("⬜" * empty + "🔴" * filled)


def format_swarm_report(report: SwarmReport) -> str:
    """Format SwarmReport thành text Telegram-ready, không cắt chữ."""
    v_em = {"MUA": "🟢", "BAN": "🔴", "THEO DOI": "🟡"}.get(report.panel_verdict, "🟡")

    lines: list[str] = [
        _SEP,
        f"🤖 LOCAL SWARM v{SWARM_VERSION}: {report.symbol}",
        f"📅 {report.timestamp}  ⏱️ {report.elapsed_s:.0f}s",
        f"🔗 {report.llm_provider}/{report.llm_model}",
        _SEP,
        "",
        "RESONANCE PANEL:",
        f"  Verdict   : {v_em} {report.panel_verdict}",
        f"  Score     : {report.final_score:+.1f}/5  {_score_bar(report.final_score)}",
        f"  Resonance : {report.resonance_pct:.0f}%",
        f"  Confidence: {report.panel_confidence:.0f}%",
        f"  Consensus : {report.consensus_level}",
        f"  Vote      : 🟢{report.vote_bull} ⚪{report.vote_neutral} 🔴{report.vote_bear}",
        "",
    ]

    # ── Bảng chuyên gia ─────────────────────────────────────────────────────
    lines += [_SEP2, "CHUYÊN GIA (3 VÒNG):", _SEP2]
    for op in report.expert_opinions:
        em   = next((e["emoji"] for e in EXPERTS if e["id"] == op.expert_id), "👤")
        s_em = {"MUA": "🟢", "BAN": "🔴", "THEO DOI": "🟡"}.get(op.stance, "🟡")
        lines.append(f"{em} {op.role}")
        lines.append(f"   {s_em} {op.stance}  Score={op.score:+d}  Conf={op.confidence}%")
        # Key takeaway — không cắt chữ giữa chừng
        kp = op.key_points[0] if op.key_points else op.concern
        kp = kp.strip()
        if len(kp) > 90:
            last_sp = kp.rfind(" ", 0, 87)
            kp = (kp[:last_sp] + "…") if last_sp > 50 else (kp[:87] + "…")
        lines.append(f"   └─ {kp}")
        if op.concern and op.concern not in ("N/A", "Xem phân tích chi tiết"):
            lines.append(f"   ⚠️ {op.concern.strip()[:80]}")
        lines.append("")

    # Debate changes
    changed_all = [x for rd in report.debate_rounds for x in rd.exchanges if x.get("changed")]
    if changed_all:
        lines.append(f"⚡ ĐỔI Ý KIẾN: {len(changed_all)} lần qua 3 vòng")
        for x in changed_all[-3:]:
            r1_s = x.get("stance_r1", x.get("stance_r2", "?"))
            r2_s = x.get("stance_r2", x.get("stance_r3", "?"))
            lines.append(f"  • {x['role'][:20]}: {r1_s}→{r2_s}")
        lines.append("")

    # ── Dissent Notes ────────────────────────────────────────────────────────
    if report.dissent_notes:
        lines += [_SEP2, "DISSENT NOTE:", _SEP2]
        for note in report.dissent_notes[:3]:
            lines.append(textwrap.fill(note, width=72, subsequent_indent="    "))
        lines.append("")

    # ── Hỗ trợ / Kháng cự ───────────────────────────────────────────────────
    sup_levels = report.support_levels    or []
    res_levels = report.resistance_levels or []
    if sup_levels or res_levels:
        lines += [_SEP2, "HỖ TRỢ / KHÁNG CỰ (gần → xa):", _SEP2]
        for i, s in enumerate(sup_levels[:3], 1):
            if isinstance(s, dict):
                dist     = s.get("dist_pct", "")
                dist_str = f" ({dist:+.1f}%)" if isinstance(dist, (int, float)) else ""
                reason   = str(s.get("reason", ""))[:55]
                lines.append(f"  🔵 S{i}: {s.get('price',0):,.0f}{dist_str}  — {reason}")
        for i, r in enumerate(res_levels[:3], 1):
            if isinstance(r, dict):
                dist     = r.get("dist_pct", "")
                dist_str = f" ({dist:+.1f}%)" if isinstance(dist, (int, float)) else ""
                reason   = str(r.get("reason", ""))[:55]
                lines.append(f"  🔴 R{i}: {r.get('price',0):,.0f}{dist_str}  — {reason}")
        lines.append("")

    # ── Kịch bản ─────────────────────────────────────────────────────────────
    lines += [_SEP2, "KỊCH BẢN ĐẦU TƯ:", _SEP2]
    all_sc = sorted([
        ("MUA",      report.scenario_buy   or {}, "🟢"),
        ("BAN",      report.scenario_sell  or {}, "🔴"),
        ("THEO DOI", report.scenario_watch or {}, "🟡"),
    ], key=lambda x: x[1].get("probability", 0), reverse=True)

    for idx, (sc_type, sc, sc_em) in enumerate(all_sc):
        if not sc:
            continue
        prob = sc.get("probability", 0)
        if prob < 5:
            lines.append(f"{sc_em} {sc_type}: xác suất thấp ({prob}%) — bỏ qua")
            continue
        priority = "ƯU TIÊN" if idx == 0 else "DỰ PHÒNG"
        lines.append(f"{sc_em} {priority} — {sc_type} ({prob}%):")

        entry = float(sc.get("entry", 0) or 0)
        tp1   = float(sc.get("tp1",   sc.get("tp", 0)) or 0)
        tp2   = float(sc.get("tp2", 0) or 0)
        sl    = float(sc.get("sl",   0) or 0)
        rr    = float(sc.get("rr",   0) or 0)

        lines.append(
            f"  Entry    : {f'{entry:,.0f}' if entry > 0 else '— (chờ trigger)'}"
        )
        if tp1 > 0:
            lines.append(f"  TP       : {tp1:,.0f}" + (f" / {tp2:,.0f}" if tp2 > 0 else ""))
        if sl > 0:
            lines.append(f"  Stop Loss: {sl:,.0f}")
        if rr > 0:
            rr_flag = " ⚠️ <1.5" if rr < 1.5 else (" ⚠️ <2.0" if rr < 2.0 else "")
            lines.append(f"  R:R      : {rr:.1f}{rr_flag}")

        trigger = sc.get("trigger", sc.get("condition", ""))
        if trigger:
            lines.append(f"  🎯 {str(trigger)[:100]}")
        catalyst = sc.get("catalyst", sc.get("watch_for", ""))
        if catalyst:
            lines.append(f"  💡 {str(catalyst)[:90]}")
        lines.append("")

    if report.rr_warning:
        lines += [f"⚠️ {report.rr_warning}", ""]

    # ── Risks & Catalysts ────────────────────────────────────────────────────
    if report.main_risks:
        lines += [_SEP2, "RỦI RO CHÍNH:", _SEP2]
        for i, risk in enumerate(report.main_risks[:4], 1):
            lines.append(f"  {i}. {str(risk)[:100]}")
        lines.append("")

    if report.key_catalysts:
        lines.append("CATALYST:")
        for cat in report.key_catalysts[:3]:
            lines.append(f"  + {str(cat)[:90]}")
        lines.append("")

    # ── Shelf life + Summary ─────────────────────────────────────────────────
    lines += [
        _SEP2,
        f"Hạn tín hiệu : {report.shelf_life_days} ngày",
        f"Đánh giá lại : {report.review_date}",
        f"Hết hạn      : {report.expires_at}",
        _SEP2, "",
        "TỔNG HỢP:", "",
    ]

    summary = report.moderator_summary or ""
    if "{" in summary and "panel_verdict" in summary:
        summary = summary[:summary.find("{")].strip()
    if summary:
        for para in summary.split("\n"):
            para = para.strip()
            if para:
                lines.append(textwrap.fill(para, width=56))
    lines.append("")

    lines += [
        _SEP,
        "⚠️ Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.",
        f"📋 Local Swarm v{SWARM_VERSION} | /local_swarm {report.symbol}",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_local_swarm(
    symbol:      str,
    meta:        dict | None = None,
    vibe_result: dict | None = None,
    progress_cb  = None,
) -> tuple[str, SwarmReport]:
    """
    Entry point chính — chạy Local Swarm Panel v2.0.

    Args:
        symbol:      Mã cổ phiếu
        meta:        dict từ analyze_stock_full() [ưu tiên — nhiều detail nhất]
        vibe_result: dict từ run_vibe_agents() [fallback]
        progress_cb: callback(msg) để stream progress về Telegram

    Returns:
        (formatted_text, SwarmReport)
    """
    if meta is None and vibe_result is None:
        raise ValueError("Cần cung cấp meta HOẶC vibe_result")

    td = (
        extract_technical_data(symbol, meta)
        if meta is not None
        else _td_from_vibe_result(symbol, vibe_result)
    )

    llm          = LLMClient()
    orchestrator = SwarmOrchestrator(llm, progress_cb=progress_cb)
    report       = orchestrator.run(td)
    text         = format_swarm_report(report)
    return text, report


def _td_from_vibe_result(symbol: str, vibe_result: dict) -> dict:
    """Fallback khi không có meta đầy đủ."""
    ind     = vibe_result.get("indicators", {})
    signals = vibe_result.get("signals", {})
    details = vibe_result.get("details", {})
    price   = float(ind.get("current_price", 0))
    bull    = sum(1 for v in signals.values() if v > 0)
    bear    = sum(1 for v in signals.values() if v < 0)
    n       = len(signals)
    conf    = round((max(bull, bear) / n * 100) if n > 0 else 50, 1)
    if n > 0:
        verdict = (
            "DONG THUAN MUA" if bull / n > 0.6 else
            "NGHIENG MUA"    if bull > bear else
            "DONG THUAN BAN" if bear / n > 0.6 else
            "NGHIENG BAN"    if bear > bull else
            "TRUNG LAP"
        )
    else:
        verdict = "TRUNG LAP"

    sup_20d = float(ind.get("support",    price * 0.95))
    res_20d = float(ind.get("resistance", price * 1.05))
    bb_low  = float(ind.get("bb_lower",   price * 0.96))
    bb_up   = float(ind.get("bb_upper",   price * 1.04))
    ma20    = float(ind.get("sma20",      price))
    ma50    = ind.get("sma50")
    atr     = price * 0.02
    sr      = _compute_sr_levels_v2(price, sup_20d, res_20d, bb_low, bb_up, ma20,
                                     float(ma50) if ma50 else None, atr)
    sig_ints = {}
    for k, v in signals.items():
        try:
            sig_ints[k] = int(v)
        except Exception:
            pass

    return {
        "symbol":           symbol.upper(),
        "price":            price,
        "change_1d_pct":    0.0,
        "change_1w_pct":    float(ind.get("change_1w_pct", 0)),
        "change_1m_pct":    float(ind.get("change_1m_pct", 0)),
        "rsi":              float(ind.get("rsi", 50)),
        "macd":             float(ind.get("macd", 0)),
        "macd_hist":        float(ind.get("macd_hist", 0)),
        "macd_signal":      float(ind.get("macd_signal", 0)),
        "volume_ratio":     float(ind.get("volume_ratio", 1.0)),
        "ma20":             ma20,
        "ma50":             float(ma50) if ma50 else None,
        "bb_upper":         bb_up,
        "bb_lower":         bb_low,
        "bb_mid":           ma20,
        "atr":              atr,
        "support_20d":      sup_20d,
        "resistance_20d":   res_20d,
        "sr_levels":        sr,
        "tp_check":         float(ind.get("tp", 0)),
        "sl_check":         float(ind.get("sl", 0)),
        "entry_check":      price,
        "verdict_label":    verdict,
        "confidence_pct":   conf,
        "bull_count":       bull,
        "bear_count":       bear,
        "active_agents":    n,
        "contradictions":   vibe_result.get("contradictions", []),
        "signals":          sig_ints,
        "engine_details":   {k: str(v)[:200] for k, v in details.items()},
        "market_regime":    ind.get("market_regime", "UNKNOWN"),
        "macro_label":      "",
        "macro_detail":     "",
        "news_headlines":   [],
        "news_sentiment":   "",
        "commodity_detail": "",
        "commodity_signal": 0,
    }


def check_local_swarm_available() -> tuple[bool, str]:
    """Kiểm tra Local Swarm có thể chạy không."""
    try:
        llm  = LLMClient()
        resp = llm.chat("Trả lời đúng 1 chữ.", "Viết chữ 'OK'", max_tokens=5)
        return True, f"{llm.provider}/{llm.model} (test: {resp.strip()[:10]})"
    except RuntimeError as e:
        return False, str(e)
    except Exception as e:
        return False, f"LLM error: {str(e)[:100]}"
