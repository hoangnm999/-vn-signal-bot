"""
local_swarm.py — Local Swarm Panel: Hội đồng chuyên gia AI nội bộ.

Mô phỏng cơ chế tranh luận của Vibe-Trading nhưng chạy hoàn toàn local,
dùng LLM qua OpenRouter / Groq / DeepSeek / Ollama — không phụ thuộc
Vibe-Trading server.

KIẾN TRÚC:
  ┌─────────────────────────────────────────────────────────────┐
  │           analyze_stock_full() → SwarmInput                 │
  │  (16 skills: signals, indicators, verdict, contradictions)  │
  └──────────────────────┬──────────────────────────────────────┘
                         │
         ┌───────────────▼──────────────────┐
         │         SwarmOrchestrator        │
         │  ┌──────────────────────────┐    │
         │  │  Expert 1: TechAnalyst   │    │
         │  │  Expert 2: MacroStrategist│   │
         │  │  Expert 3: RiskManager   │    │
         │  │  Expert 4: SMC_Trader    │    │
         │  │  Expert 5: FundaFilter   │    │
         │  └──────────────┬───────────┘    │
         │   Round 1: Ý kiến độc lập        │
         │   Round 2: Phản biện chéo        │
         │   Moderator: Tổng hợp verdict    │
         └───────────────┬──────────────────┘
                         │
              ┌──────────▼───────────┐
              │    SwarmReport       │
              │ • Verdict + prob     │
              │ • Entry/SL/TP/RR     │
              │ • Scenarios          │
              │ • Signal shelf life  │
              └──────────────────────┘

ENV VARS (ưu tiên lần lượt):
  DEEPSEEK_API_KEY   → api.deepseek.com (primary)
  OPENROUTER_API_KEY → openrouter.ai    (fallback 1)
  GROQ_API_KEY       → api.groq.com     (fallback 2, rất nhanh, free)
  GEMINI_API_KEY     → generativelanguage.googleapis.com (fallback 3)
  OLLAMA_URL         → localhost:11434  (fallback 4, fully local)
"""

from __future__ import annotations

import os
import json
import time
import logging
import asyncio
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

DEBATE_ROUNDS    = 3        # tăng lên 3 vòng để sâu hơn (Fix #5)
LLM_TIMEOUT      = 90       # giây timeout per LLM call
LLM_MAX_TOKENS   = 900      # token tối đa mỗi response chuyên gia
MODERATOR_TOKENS = 1800     # đủ cho JSON đầy đủ + summary
SWARM_VERSION    = "1.2"

# Mô hình mặc định theo provider
_DEFAULT_MODELS = {
    "deepseek":    "deepseek-chat",
    "openrouter":  "deepseek/deepseek-chat",
    "groq":        "llama-3.3-70b-versatile",
    "gemini":      "gemini-2.0-flash",
    "ollama":      "qwen2.5:7b",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SwarmInput:
    """Dữ liệu đầu vào từ analyze_stock_full() / vibe_skills.run_vibe_agents()."""
    symbol:          str
    current_price:   float
    verdict_label:   str          # DONG THUAN MUA / BAN / TRUNG LAP
    confidence_pct:  float
    bull_count:      int
    bear_count:      int
    active_agents:   int

    # Signals từ 16 engines {engine_name: +1/0/-1}
    signals:         dict[str, int]  = field(default_factory=dict)
    # Details từ từng engine
    details:         dict[str, str]  = field(default_factory=dict)

    # Indicators cơ bản
    rsi:             float = 0.0
    macd:            float = 0.0
    atr:             float = 0.0
    volume_ratio:    float = 1.0
    change_1d_pct:   float = 0.0
    change_1w_pct:   float = 0.0
    change_1m_pct:   float = 0.0
    sma20:           float = 0.0
    sma50:           float = 0.0

    # Context thị trường
    market_regime:   str = "UNKNOWN"    # UPTREND / DOWNTREND / SIDEWAYS
    macro_label:     str = ""
    news_sentiment:  str = ""
    contradictions:  list[str] = field(default_factory=list)

    # Các điểm kỹ thuật
    support:         float = 0.0
    resistance:      float = 0.0
    tp:              float = 0.0        # take profit từ /check
    sl:              float = 0.0        # stop loss từ /check
    entry:           float = 0.0

    # Thông tin bổ sung từ 16 skills để expert dùng (Fix #5)
    bb_upper:        float = 0.0
    bb_lower:        float = 0.0
    bb_mid:          float = 0.0
    macd_hist:       float = 0.0
    macd_signal:     float = 0.0
    resistance_20d:  float = 0.0
    support_20d:     float = 0.0
    # Skill details: dict {engine_name: detail_text} từ 16 engines
    skill_details:   dict  = field(default_factory=dict)

    @classmethod
    def from_analyze_result(cls, symbol: str, meta: dict) -> "SwarmInput":
        """Build SwarmInput từ meta dict của analyze_stock_full()."""
        v   = meta.get("verdict", {})
        ind = meta.get("ind", {})
        av  = meta.get("agent_verdicts", {})

        # Lấy signals từ agent_verdicts (nếu dạng dict {name: score})
        signals: dict[str, int] = {}
        if isinstance(av, dict):
            for k, val in av.items():
                try:
                    signals[k] = int(val)
                except Exception:
                    pass
        elif isinstance(av, str):
            try:
                signals = json.loads(av)
            except Exception:
                pass

        price = float(ind.get("current_price", 0))
        atr   = float(ind.get("atr", price * 0.02 if price else 0))

        # Lấy skill details từ details dict trong meta (nếu có)
        details_raw = meta.get("details", {})
        skill_details: dict = {}
        if isinstance(details_raw, dict):
            skill_details = {k: str(v)[:200] for k, v in details_raw.items()}

        return cls(
            symbol         = symbol.upper(),
            current_price  = price,
            verdict_label  = v.get("verdict_label", "TRUNG LAP"),
            confidence_pct = float(v.get("confidence_pct", 50)),
            bull_count     = int(v.get("bull_count", 0)),
            bear_count     = int(v.get("bear_count", 0)),
            active_agents  = int(v.get("active_agents", 0)),
            signals        = signals,
            details        = {},
            rsi            = float(ind.get("rsi", 50)),
            macd           = float(ind.get("macd", 0)),
            atr            = atr,
            volume_ratio   = float(ind.get("volume_ratio", 1.0)),
            change_1d_pct  = float(ind.get("change_1d_pct", 0)),
            change_1w_pct  = float(ind.get("change_1w_pct", 0)),
            change_1m_pct  = float(ind.get("change_1m_pct", 0)),
            sma20          = float(ind.get("sma20", ind.get("ma20", 0))),
            sma50          = float(ind.get("sma50", ind.get("ma50", 0)) or 0),
            market_regime  = meta.get("macro_v", {}).get("market_regime", "UNKNOWN") if isinstance(meta.get("macro_v"), dict) else "UNKNOWN",
            macro_label    = meta.get("macro_v", {}).get("label", "") if isinstance(meta.get("macro_v"), dict) else str(meta.get("macro_v", "")),
            news_sentiment = "",
            contradictions = v.get("contradictions", []),
            support        = float(ind.get("support", ind.get("support_20d", price * 0.95))),
            resistance     = float(ind.get("resistance", ind.get("resistance_20d", price * 1.05))),
            tp             = float(v.get("tp", 0)),
            sl             = float(v.get("sl", 0)),
            entry          = float(v.get("entry_price", price)),
            bb_upper       = float(ind.get("bb_upper", 0)),
            bb_lower       = float(ind.get("bb_lower", 0)),
            bb_mid         = float(ind.get("bb_mid", 0)),
            macd_hist      = float(ind.get("macd_hist", 0)),
            macd_signal    = float(ind.get("macd_signal", 0)),
            resistance_20d = float(ind.get("resistance_20d", 0)),
            support_20d    = float(ind.get("support_20d", 0)),
            skill_details  = skill_details,
        )

    @classmethod
    def from_vibe_result(cls, symbol: str, vibe_result: dict) -> "SwarmInput":
        """Build SwarmInput từ run_vibe_agents() result dict."""
        signals = vibe_result.get("signals", {})
        details = vibe_result.get("details", {})
        ind     = vibe_result.get("indicators", {})
        price   = float(ind.get("current_price", 0))

        bull = sum(1 for v in signals.values() if v > 0)
        bear = sum(1 for v in signals.values() if v < 0)
        n    = len(signals)
        conf = round((max(bull, bear) / n * 100) if n > 0 else 50, 1)

        if bull > bear:
            verdict = "DONG THUAN MUA" if bull / n > 0.6 else "NGHIENG MUA"
        elif bear > bull:
            verdict = "DONG THUAN BAN" if bear / n > 0.6 else "NGHIENG BAN"
        else:
            verdict = "TRUNG LAP"

        atr = float(ind.get("atr", price * 0.02 if price else 0))
        return cls(
            symbol        = symbol.upper(),
            current_price = price,
            verdict_label = verdict,
            confidence_pct= conf,
            bull_count    = bull,
            bear_count    = bear,
            active_agents = n,
            signals       = signals,
            details       = details,
            rsi           = float(ind.get("rsi", 50)),
            macd          = float(ind.get("macd", 0)),
            atr           = atr,
            volume_ratio  = float(ind.get("volume_ratio", 1.0)),
            change_1d_pct = float(ind.get("change_1d_pct", 0)),
            change_1w_pct = float(ind.get("change_1w_pct", 0)),
            change_1m_pct = float(ind.get("change_1m_pct", 0)),
            sma20         = float(ind.get("sma20", 0)),
            sma50         = float(ind.get("sma50", 0)),
            market_regime = ind.get("market_regime", "UNKNOWN"),
            macro_label   = "",
            contradictions= vibe_result.get("contradictions", []),
            support       = float(ind.get("support", price * 0.95)),
            resistance    = float(ind.get("resistance", price * 1.05)),
            tp            = float(ind.get("tp", 0)),
            sl            = float(ind.get("sl", 0)),
            entry         = price,
        )


@dataclass
class ExpertOpinion:
    expert_id:   str
    role:        str
    stance:      str        # BUY / SELL / WATCH
    confidence:  int        # 0-100
    key_points:  list[str]
    concern:     str        # rủi ro chính thấy được
    raw_text:    str        # full response từ LLM


@dataclass
class DebateRound:
    round_num:    int
    exchanges:    list[dict]   # [{"expert": str, "rebuts": str, "text": str}]


@dataclass
class SwarmReport:
    symbol:          str
    timestamp:       str
    elapsed_s:       float
    llm_provider:    str
    llm_model:       str

    # Kết luận hội đồng
    panel_verdict:   str       # MUA / BAN / THEO DOI
    panel_confidence: float    # 0-100
    consensus_level: str       # DONG THUAN / PHAN BIEN / CHIA RE

    # Scenarios
    scenario_buy:    dict      # {prob, entry, tp, sl, rr, condition, catalyst}
    scenario_sell:   dict
    scenario_watch:  dict

    # Rủi ro
    main_risks:      list[str]
    key_catalysts:   list[str]

    # Signal shelf life
    shelf_life_days: int
    expires_at:      str
    review_date:     str           # ngày hiện tại + 7 ngày

    # Support/resistance đã sắp xếp theo khoảng cách (Fix #2)
    support_levels:    list        # [{price, reason, dist_pct}] gần → xa
    resistance_levels: list        # [{price, reason, dist_pct}] gần → xa

    # R:R warning (Fix #1)
    rr_warning: str

    # Resonance panel (Fix #4)
    final_score:    float          # -5..+5
    resonance_pct:  float          # 0-100
    vote_bull:      int
    vote_neutral:   int
    vote_bear:      int
    dissent_notes:  list[str]      # cảnh báo từ chuyên gia thiểu số

    # Expert opinions
    expert_opinions: list[ExpertOpinion]
    debate_rounds:   list[DebateRound]

    # Moderator summary (TEXT THUẦN, không chứa JSON)
    moderator_summary: str

    # Raw JSON — chỉ lưu DB, KHÔNG hiển thị user (Fix #6 từ session trước)
    moderator_raw_json: dict

    # Raw input echo
    input_summary:   str


# ══════════════════════════════════════════════════════════════════════════════
# LLM CLIENT — waterfall qua các providers
# ══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """
    Universal LLM client: thử từng provider theo thứ tự ưu tiên.
    Ưu tiên: DeepSeek → OpenRouter → Groq → Gemini → Ollama
    """

    def __init__(self):
        self.provider, self.model = self._detect_provider()
        logger.info(f"LLM Client: {self.provider} / {self.model}")

    def _detect_provider(self) -> tuple[str, str]:
        """Phát hiện provider có sẵn theo env vars."""
        checks = [
            ("deepseek",   "DEEPSEEK_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
            ("groq",       "GROQ_API_KEY"),
            ("gemini",     "GEMINI_API_KEY"),
        ]
        for provider, env_key in checks:
            key = os.environ.get(env_key, "").strip()
            if key:
                model_env = os.environ.get(
                    f"{provider.upper()}_MODEL", _DEFAULT_MODELS[provider]
                )
                return provider, model_env

        # Ollama — không cần API key
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        try:
            r = requests.get(f"{ollama_url}/api/tags", timeout=3)
            if r.ok:
                model = os.environ.get("OLLAMA_MODEL", _DEFAULT_MODELS["ollama"])
                return "ollama", model
        except Exception:
            pass

        raise RuntimeError(
            "Khong tim thay LLM provider nao!\n"
            "Set 1 trong: DEEPSEEK_API_KEY, OPENROUTER_API_KEY, "
            "GROQ_API_KEY, GEMINI_API_KEY, hoac OLLAMA_URL"
        )

    def chat(self, system: str, user: str, max_tokens: int = LLM_MAX_TOKENS) -> str:
        """Gọi LLM và trả về text response."""
        if self.provider == "deepseek":
            return self._call_openai_compat(
                "https://api.deepseek.com/v1/chat/completions",
                os.environ.get("DEEPSEEK_API_KEY", ""),
                system, user, max_tokens,
            )
        if self.provider == "openrouter":
            return self._call_openai_compat(
                "https://openrouter.ai/api/v1/chat/completions",
                os.environ.get("OPENROUTER_API_KEY", ""),
                system, user, max_tokens,
                extra_headers={
                    "HTTP-Referer": "https://github.com/vnsignalbot",
                    "X-Title": "VN Signal Bot",
                },
            )
        if self.provider == "groq":
            return self._call_openai_compat(
                "https://api.groq.com/openai/v1/chat/completions",
                os.environ.get("GROQ_API_KEY", ""),
                system, user, max_tokens,
            )
        if self.provider == "gemini":
            return self._call_gemini(
                os.environ.get("GEMINI_API_KEY", ""),
                system, user, max_tokens,
            )
        if self.provider == "ollama":
            return self._call_ollama(system, user, max_tokens)

        raise RuntimeError(f"Provider '{self.provider}' chua duoc implement")

    def _call_openai_compat(
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
            "model":      self.model,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
        }
        r = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    def _call_gemini(self, api_key: str, system: str, user: str, max_tokens: int) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7},
        }
        r = requests.post(url, json=payload, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _call_ollama(self, system: str, user: str, max_tokens: int) -> str:
        url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        payload = {
            "model":  self.model,
            "prompt": f"<|system|>{system}<|user|>{user}<|assistant|>",
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.7},
        }
        r = requests.post(f"{url}/api/generate", json=payload, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json()["response"].strip()


# ══════════════════════════════════════════════════════════════════════════════
# EXPERT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

EXPERTS = [
    {
        "id":     "tech_analyst",
        "role":   "Chuyên gia Phân tích Kỹ thuật",
        "emoji":  "📊",
        "focus":  (
            "Bạn là chuyên gia phân tích kỹ thuật với 15 năm kinh nghiệm giao dịch HOSE/HNX. "
            "Chuyên về: Elliott Wave, Ichimoku, SMC (Smart Money Concepts), price action, "
            "candlestick patterns, và volume analysis. "
            "Bạn TIN VÀO dữ liệu kỹ thuật và signals từ các engines. "
            "Bạn đặc biệt chú ý đến: RSI overbought/oversold, MACD divergence, "
            "volume confirmation, và breakout/breakdown."
        ),
    },
    {
        "id":     "macro_strategist",
        "role":   "Chiến lược gia Vĩ mô",
        "emoji":  "🌏",
        "focus":  (
            "Bạn là chiến lược gia vĩ mô với nền tảng từ Dragon Capital và VinaCapital. "
            "Chuyên về: xu hướng thị trường toàn cầu, tác động của Fed/SBV, "
            "tỷ giá USD/VND, giá vàng/dầu, và chu kỳ kinh tế VN. "
            "Bạn nhìn BIG PICTURE và thường skeptical với signals kỹ thuật đơn thuần. "
            "Bạn đặc biệt quan tâm market regime (bull/bear/sideways) và liquidity."
        ),
    },
    {
        "id":     "risk_manager",
        "role":   "Nhà Quản trị Rủi ro",
        "emoji":  "🛡️",
        "focus":  (
            "Bạn là risk manager chuyên nghiệp, đến từ bộ phận quản lý rủi ro của một quỹ lớn. "
            "Nhiệm vụ: bảo vệ vốn bằng mọi giá. Bạn LUÔN đặt câu hỏi về downside trước upside. "
            "Chuyên về: position sizing, R:R ratio, stop loss placement, max drawdown, "
            "correlation risk và tail risk. "
            "Bạn sẽ PHẢN ĐỐI quyết định nếu R:R < 1.5 hoặc không có SL rõ ràng."
        ),
    },
    {
        "id":     "smc_trader",
        "role":   "Smart Money Concepts Trader",
        "emoji":  "💡",
        "focus":  (
            "Bạn là SMC trader chuyên theo dõi dòng tiền tổ chức (smart money) trên HOSE. "
            "Chuyên về: Order Blocks, Fair Value Gaps, Liquidity Pools, BOS/CHoCH, "
            "Market Structure, và Wyckoff Method. "
            "Bạn tin rằng giá luôn đi về nơi có liquidity. "
            "Bạn tìm kiếm vùng premium/discount và điểm POI (Point of Interest) "
            "để xác định entry chính xác nhất."
        ),
    },
    {
        "id":     "fundamental_filter",
        "role":   "Bộ lọc Cơ bản",
        "emoji":  "📋",
        "focus":  (
            "Bạn là chuyên gia phân tích cơ bản và ngành với 10 năm nghiên cứu doanh nghiệp VN. "
            "Chuyên về: sector rotation, câu chuyện doanh nghiệp, earnings momentum, "
            "định giá tương đối P/E/P/B so ngành, và catalyst nội tại. "
            "Bạn đặt câu hỏi: 'Tại sao MUA mã này MÀ KHÔNG MUA mã khác cùng ngành?' "
            "Bạn cũng chú ý seasonality và thời điểm trong năm."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_context_block(inp: SwarmInput) -> str:
    """Tạo block dữ liệu chung cho tất cả experts."""
    bull_engines = [k for k, v in inp.signals.items() if v > 0]
    bear_engines = [k for k, v in inp.signals.items() if v < 0]
    neutral      = [k for k, v in inp.signals.items() if v == 0]

    bull_str = ", ".join(bull_engines) or "Không có"
    bear_str = ", ".join(bear_engines) or "Không có"

    price = inp.current_price
    atr   = inp.atr or price * 0.02

    # ── Tính S/R gần nhất (Fix #2) ────────────────────────────────────────
    sr = _compute_sr_levels(inp)
    s_lines = "\n".join(
        f"  S{i+1}: {s['price']:,.0f}  ({s['dist_pct']:+.1f}%)  — {s['reason']}"
        for i, s in enumerate(sr["support"][:3])
    )
    r_lines = "\n".join(
        f"  R{i+1}: {r['price']:,.0f}  ({r['dist_pct']:+.1f}%)  — {r['reason']}"
        for i, r in enumerate(sr["resistance"][:3])
    )

    tp  = inp.tp  or round(sr["resistance"][0]["price"] if sr["resistance"] else price * 1.05, 0)
    sl  = inp.sl  or round(sr["support"][0]["price"]    if sr["support"]    else price * 0.95, 0)
    rr  = round((tp - price) / (price - sl), 2) if (price - sl) > 0 else 0

    # ── Skill details cho experts (Fix #5) ────────────────────────────────
    # Chỉ lấy các skill quan trọng, tóm tắt ngắn
    skill_block = ""
    if inp.skill_details:
        priority_skills = [
            "TechnicalBasic", "Ichimoku", "SMC", "Chanlun",
            "MarketRegime", "FundamentalFilter", "NewsSentiment",
            "ElliottWave", "Volatility",
        ]
        lines_sk = []
        for sk in priority_skills:
            detail = inp.skill_details.get(sk, "")
            if detail and len(detail) > 10:
                lines_sk.append(f"  [{sk}] {detail[:120]}")
        if lines_sk:
            skill_block = "\n--- CHI TIẾT 16 SKILLS ---\n" + "\n".join(lines_sk[:8])

    # ── BB info ──────────────────────────────────────────────────────────
    bb_info = ""
    if inp.bb_upper and inp.bb_lower:
        bb_info = (f"\nBollinger Bands: Upper={inp.bb_upper:,.0f}  "
                   f"Mid={inp.bb_mid:,.0f}  Lower={inp.bb_lower:,.0f}")

    ctx = f"""
=== DỮ LIỆU PHÂN TÍCH: {inp.symbol} ===
Thời điểm: {datetime.now().strftime("%Y-%m-%d %H:%M")}

--- GIÁ & KỸ THUẬT ---
Giá hiện tại  : {price:,.0f} VND
RSI(14)        : {inp.rsi:.1f}
MACD           : {inp.macd:.4f}  |  MACD Hist: {inp.macd_hist:.4f}  |  Signal: {inp.macd_signal:.4f}
ATR(14)        : {inp.atr:,.0f}
Volume ratio   : {inp.volume_ratio:.2f}x TB20
Thay đổi 1D/1W/1M: {inp.change_1d_pct:+.1f}% / {inp.change_1w_pct:+.1f}% / {inp.change_1m_pct:+.1f}%
SMA20={inp.sma20:,.0f}  |  SMA50={inp.sma50:,.0f}{bb_info}

--- HỖ TRỢ (gần → xa) ---
{s_lines}

--- KHÁNG CỰ (gần → xa) ---
{r_lines}

TP gợi ý: {tp:,.0f}  |  SL gợi ý: {sl:,.0f}  |  R:R: {rr:.2f}

--- KẾT QUẢ 16 ENGINES ---
Phán quyết: {inp.verdict_label} ({inp.confidence_pct:.0f}%)
Bull: {inp.bull_count}/{inp.active_agents} → {bull_str}
Bear: {inp.bear_count}/{inp.active_agents} → {bear_str}
Neutral: {len(neutral)} engines

--- VĨ MÔ & THỊ TRƯỜNG ---
Market Regime  : {inp.market_regime}
Macro context  : {inp.macro_label or "N/A"}
News sentiment : {inp.news_sentiment or "N/A"}

--- MÂU THUẪN ---
{chr(10).join(inp.contradictions) if inp.contradictions else "Không có mâu thuẫn rõ ràng"}
{skill_block}
""".strip()
    return ctx


def _compute_sr_levels(inp: SwarmInput) -> dict:
    """
    Tính mức hỗ trợ/kháng cự gần nhất từ indicators có sẵn.
    Sắp xếp: S1 = gần nhất dưới giá, R1 = gần nhất trên giá.
    Đảm bảo mỗi mức cách nhau ít nhất 2.5%.
    """
    price = inp.current_price
    atr   = inp.atr or price * 0.02

    # Thu thập tất cả mức tiềm năng từ indicators
    sup_candidates: list[tuple[float, str]] = []
    res_candidates: list[tuple[float, str]] = []

    def _add(p: float, reason: str):
        if p <= 0 or abs(p - price) / price > 0.25:  # bỏ quá xa 25%
            return
        if p < price:
            sup_candidates.append((p, reason))
        elif p > price:
            res_candidates.append((p, reason))

    # Từ SwarmInput indicators
    if inp.sma20 > 0:    _add(inp.sma20,    "SMA20")
    if inp.sma50 > 0:    _add(inp.sma50,    "SMA50")
    if inp.support > 0:  _add(inp.support,  "Hỗ trợ /check")
    if inp.resistance > 0: _add(inp.resistance, "Kháng cự /check")
    if inp.bb_lower > 0: _add(inp.bb_lower, "Bollinger Lower Band")
    if inp.bb_upper > 0: _add(inp.bb_upper, "Bollinger Upper Band")
    if inp.bb_mid > 0:   _add(inp.bb_mid,   "Bollinger Mid (SMA20)")
    if inp.support_20d > 0:    _add(inp.support_20d,    "Low 20 ngày")
    if inp.resistance_20d > 0: _add(inp.resistance_20d, "High 20 ngày")
    if inp.sl > 0:       _add(inp.sl,       "Stop Loss từ /check")
    if inp.tp > 0:       _add(inp.tp,       "Target từ /check")

    # Fibonacci từ range 20D
    h20 = inp.resistance_20d or (price * 1.05)
    l20 = inp.support_20d    or (price * 0.95)
    rng = h20 - l20
    if rng > 0:
        for fib, label in [(0.236, "Fib 23.6%"), (0.382, "Fib 38.2%"),
                           (0.5,   "Fib 50%"),   (0.618, "Fib 61.8%"),
                           (0.786, "Fib 78.6%")]:
            _add(round(h20 - fib * rng, 0), label)

    # ATR-based fallback (nếu thiếu)
    for mult, label in [(1.0, "ATR×1 hỗ trợ"), (2.0, "ATR×2 hỗ trợ"),
                        (3.0, "ATR×3 hỗ trợ")]:
        _add(round(price - mult * atr, 0), label)
    for mult, label in [(1.0, "ATR×1 kháng cự"), (2.0, "ATR×2 kháng cự"),
                        (3.0, "ATR×3 kháng cự")]:
        _add(round(price + mult * atr, 0), label)

    def _dedupe_and_sort(candidates: list, descending: bool) -> list[dict]:
        """Sắp xếp, loại duplicate (trong 2.5%), lấy 3 mức gần nhất."""
        if not candidates:
            return []
        # Sắp xếp gần giá nhất lên đầu
        srt = sorted(candidates, key=lambda x: abs(x[0] - price))
        result = []
        for p, reason in srt:
            # Loại bỏ nếu quá gần mức đã có (< 2.5%)
            if any(abs(p - r["price"]) / r["price"] < 0.025 for r in result):
                continue
            dist_pct = round((p - price) / price * 100, 1)
            result.append({"price": round(p, 0), "reason": reason, "dist_pct": dist_pct})
            if len(result) >= 3:
                break
        # Pad nếu thiếu
        while len(result) < 3:
            if result:
                if descending:  # resistance: thêm xa hơn
                    base = result[-1]["price"]
                    new_p = round(base * 1.03, 0)
                    new_reason = f"Fib/kháng cự ước tính (+{len(result)*3}%)"
                else:           # support: thêm xa hơn xuống
                    base = result[-1]["price"]
                    new_p = round(base * 0.97, 0)
                    new_reason = f"Fib/hỗ trợ ước tính (-{len(result)*3}%)"
            else:
                new_p = round(price * (1.03 if descending else 0.97), 0)
                new_reason = "ATR-based estimate"
            dist_pct = round((new_p - price) / price * 100, 1)
            result.append({"price": new_p, "reason": new_reason, "dist_pct": dist_pct})
        return result

    return {
        "support":    _dedupe_and_sort(sup_candidates, descending=False),
        "resistance": _dedupe_and_sort(res_candidates, descending=True),
    }


def _get_expert_skill_context(expert: dict, inp: SwarmInput) -> str:
    """Trích skill details phù hợp với từng chuyên gia (Fix #5)."""
    skill_map = {
        "tech_analyst":       ["TechnicalBasic", "Ichimoku", "ElliottWave", "Candlestick"],
        "macro_strategist":   ["MarketRegime", "CrossMarket", "CommodityContext", "VNMacro"],
        "risk_manager":       ["Volatility", "TechnicalBasic", "MarketRegime"],
        "smc_trader":         ["SMC", "Chanlun", "ElliottWave", "MultiFactor"],
        "fundamental_filter": ["FundamentalFilter", "NewsSentiment", "Seasonal", "MLStrategy"],
    }
    relevant = skill_map.get(expert["id"], [])
    lines = []
    for sk in relevant:
        detail = inp.skill_details.get(sk, "")
        if detail and len(detail) > 10:
            lines.append(f"  [{sk}] {detail[:150]}")
    return "\n".join(lines) if lines else "  (Không có skill details)"


def _build_expert_prompt_round1(expert: dict, ctx: str, inp: SwarmInput | None = None) -> tuple[str, str]:
    """System + user prompt cho vòng 1: ý kiến độc lập. Fix #3 (2-dòng output), Fix #5 (skill context)."""
    skill_ctx = _get_expert_skill_context(expert, inp) if inp else "  (Không có)"

    system = (
        f"Bạn là {expert['role']} trong Hội đồng Chuyên gia Phân tích Cổ phiếu VN.\n"
        f"{expert['focus']}\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        "1. STANCE=THEO DOI → entry=0. Trigger phải có: [giá cụ thể]+[volume]+[chỉ báo].\n"
        "2. STANCE=MUA/BAN → phải có Entry, SL, TP, R:R.\n"
        "3. R:R=(TP-Entry)/(Entry-SL). Nếu R:R<1.5 → BẮT BUỘC hạ sang THEO DOI.\n"
        "4. Nếu Entry>TP hoặc Entry<SL → báo lỗi, chuyển THEO DOI.\n"
        "5. Rủi ro và Catalyst phải gắn với số liệu cụ thể của mã này.\n\n"
        "ĐỊNH DẠNG (bắt buộc, đúng thứ tự):\n"
        "STANCE: [MUA/BAN/THEO DOI]\n"
        "CONFIDENCE: [0-100]\n"
        "REASON: [1 câu ngắn, lý do chính]\n"
        "ENTRY: [giá] hoặc 0 nếu THEO DOI\n"
        "SL: [giá]\n"
        "TP: [giá]\n"
        "RR: [số thực, vd 2.1]\n"
        "TRIGGER: [điều kiện kích hoạt]\n"
        "RISK: [rủi ro chính gắn với mã]\n"
        "CATALYST: [catalyst gắn với mã]"
    )
    symbol = inp.symbol if inp else ctx.split()[4] if len(ctx.split()) > 4 else "mã"
    user = (
        f"Phân tích {symbol} từ góc nhìn {expert['role']}:\n\n"
        f"{ctx}\n\n"
        f"Skills chuyên môn của bạn ({expert['role']}):\n{skill_ctx}\n\n"
        "Tính R:R thực tế. Nếu R:R<1.5 BẮT BUỘC chuyển THEO DOI. "
        "Rủi ro/Catalyst phải cụ thể, có số liệu."
    )
    return system, user


def _build_expert_prompt_round2(
    expert: dict, ctx: str,
    other_opinions: list[ExpertOpinion],
    own_round1: ExpertOpinion,
    inp: SwarmInput | None = None,
) -> tuple[str, str]:
    """Vòng 2: phản biện. Fix #5."""
    others_summary = []
    for op in other_opinions:
        if op.expert_id == expert["id"]:
            continue
        others_summary.append(
            f"[{op.role}] {op.stance}({op.confidence}%) — {'; '.join(op.key_points[:1])}"
        )

    skill_ctx = _get_expert_skill_context(expert, inp) if inp else ""

    system = (
        f"Bạn là {expert['role']} trong Hội đồng Chuyên gia.\n"
        f"{expert['focus']}\n\n"
        "VÒNG 2 — PHẢN BIỆN:\n"
        "1. Bảo vệ hoặc điều chỉnh stance sau khi nghe các expert khác\n"
        "2. Phản biện điểm yếu nhất của 1 expert đối lập\n"
        "3. Ghi 1 điểm ĐỒNG Ý với expert khác\n"
        "4. Nếu Risk Manager chỉ ra R:R<1.5 → phải điều chỉnh\n\n"
        "ĐỊNH DẠNG:\n"
        "STANCE: [MUA/BAN/THEO DOI]\n"
        "CONFIDENCE: [0-100]\n"
        "REASON: [1 câu lý do sau tranh luận]\n"
        "ENTRY: [giá] hoặc 0\n"
        "SL: [giá]  TP: [giá]  RR: [số]\n"
        "REBUTS: [phản biện expert nào, điểm gì]\n"
        "AGREES: [đồng ý với ai, điểm gì]\n"
        "TRIGGER: [điều kiện kích hoạt cập nhật]"
    )
    user = (
        f"Vòng 1 của bạn: {own_round1.stance}({own_round1.confidence}%) "
        f"— {'; '.join(own_round1.key_points[:1])}\n\n"
        f"Ý kiến các experts khác:\n" + "\n".join(others_summary) +
        f"\n\nSkills của bạn:\n{skill_ctx[:300]}\n\n"
        "Cập nhật stance nếu cần. R:R<1.5 → THEO DOI."
    )
    return system, user


def _build_expert_prompt_round3(
    expert: dict, ctx: str,
    r2_opinions: list[ExpertOpinion],
    own_r2: ExpertOpinion,
    inp: SwarmInput | None = None,
) -> tuple[str, str]:
    """Vòng 3 (mới — Fix #5): Kết luận cuối và xác nhận kịch bản."""
    others = []
    for op in r2_opinions:
        if op.expert_id == expert["id"]:
            continue
        others.append(f"[{op.role}] {op.stance}({op.confidence}%)")

    system = (
        f"Bạn là {expert['role']} — VÒNG 3 CUỐI CÙNG.\n"
        f"{expert['focus']}\n\n"
        "Sau 2 vòng tranh luận, đây là cơ hội cuối để:\n"
        "1. Xác nhận stance cuối cùng (không đổi nếu không có lý do mạnh)\n"
        "2. Đề xuất kịch bản cụ thể nhất với Entry/SL/TP khả thi\n"
        "3. Nêu 1 rủi ro quan trọng nhất mà các expert khác bỏ qua\n\n"
        "ĐỊNH DẠNG NGẮN GỌN:\n"
        "STANCE: [MUA/BAN/THEO DOI]\n"
        "CONFIDENCE: [0-100]\n"
        "FINAL_REASON: [1-2 câu kết luận]\n"
        "SCENARIO: Entry=[giá] SL=[giá] TP=[giá] RR=[số]\n"
        "ALERT: [rủi ro bị bỏ qua]"
    )
    user = (
        f"Vòng 2 của bạn: {own_r2.stance}({own_r2.confidence}%)\n"
        f"Các expert khác: {' | '.join(others)}\n\n"
        "Xác nhận kịch bản cuối. Tính R:R = (TP-Entry)/(Entry-SL). "
        "Nếu R:R<1.5 → THEO DOI."
    )
    return system, user


def _build_moderator_prompt(
    inp: SwarmInput,
    ctx: str,
    r1_opinions: list[ExpertOpinion],
    r2_opinions: list[ExpertOpinion],
    r3_opinions: list[ExpertOpinion] | None = None,
) -> tuple[str, str]:
    """Prompt cho moderator tổng hợp kết luận cuối. Dùng vòng 3 nếu có."""
    final_opinions = r3_opinions or r2_opinions
    stances = {op.expert_id: op.stance for op in final_opinions}
    confs   = {op.expert_id: op.confidence for op in final_opinions}

    buy_count   = sum(1 for s in stances.values() if s == "MUA")
    sell_count  = sum(1 for s in stances.values() if s == "BAN")
    watch_count = sum(1 for s in stances.values() if s == "THEO DOI")
    n_experts   = len(stances) or 5

    dominant    = max(buy_count, sell_count, watch_count)
    avg_conf    = sum(confs.values()) / len(confs) if confs else 60
    base_conf   = max(40, round((dominant / n_experts) * avg_conf))
    if dominant == n_experts:
        base_conf = max(base_conf, 65)
    elif dominant >= n_experts - 1:
        base_conf = max(base_conf, 55)

    price  = inp.current_price
    atr    = inp.atr or price * 0.02
    sr     = _compute_sr_levels(inp)
    s1     = sr["support"][0]["price"]    if sr["support"]    else round(price * 0.97, 0)
    r1_p   = sr["resistance"][0]["price"] if sr["resistance"] else round(price * 1.03, 0)
    tp_est = inp.tp or r1_p
    sl_est = inp.sl or s1
    rr_est = round((tp_est - price) / (price - sl_est), 2) if (price - sl_est) > 0 else 2.0

    opinions_text = ""
    for op in final_opinions:
        ename = next((e["emoji"] + " " + e["role"] for e in EXPERTS if e["id"] == op.expert_id), op.expert_id)
        opinions_text += (
            f"\n{ename}: {op.stance}({op.confidence}%)\n"
            f"  {'; '.join(op.key_points[:2])}\n"
            f"  Rủi ro: {op.concern[:80]}\n"
        )

    sr_text = (
        f"S1={s1:,.0f} | S2={sr['support'][1]['price']:,.0f} | S3={sr['support'][2]['price']:,.0f}\n"
        f"R1={r1_p:,.0f} | R2={sr['resistance'][1]['price']:,.0f} | R3={sr['resistance'][2]['price']:,.0f}"
        if len(sr["support"]) >= 3 and len(sr["resistance"]) >= 3 else ""
    )

    system = (
        "Bạn là MODERATOR Hội đồng Chuyên gia Phân tích Cổ phiếu VN.\n"
        "Trả về JSON ĐẦY ĐỦ — không cắt giữa chừng. KHÔNG dùng markdown.\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        f"1. panel_confidence >= {base_conf} (đồng thuận {dominant}/{n_experts} experts).\n"
        "2. R:R=(tp1-entry)/(entry-sl). R:R<1.5 → entry=0, chuyển THEO DOI, thêm rr_warning.\n"
        "3. Nếu entry>tp hoặc entry<sl → entry=0, thêm rr_warning.\n"
        "4. support_levels: S1 gần nhất dưới giá, S2/S3 xa dần. Cách nhau ≥2.5%.\n"
        "5. resistance_levels: R1 gần nhất trên giá, R2/R3 xa dần. Cách nhau ≥2.5%.\n"
        "6. main_risks và key_catalysts: cụ thể với mã, có số liệu.\n"
        "7. moderator_summary: TEXT THUẦN tiếng Việt 80-120 từ, KHÔNG chứa JSON.\n"
        "8. shelf_life_days >= 7.\n\n"
        "OUTPUT JSON (thuần, không markdown, không cắt):\n"
        "{\n"
        '  "panel_verdict": "MUA|BAN|THEO DOI",\n'
        f'  "panel_confidence": {base_conf},\n'
        '  "rr_warning": "",\n'
        '  "consensus_level": "DONG THUAN|PHAN BIEN|CHIA RE",\n'
        '  "scenario_buy": {'
        f'"probability": 50, "entry": {round(price,0)}, "tp1": {round(tp_est,0)}, "tp2": {round(tp_est*1.04,0)}, "sl": {round(sl_est,0)}, "rr": {rr_est}, '
        '"trigger": "MUA khi [giá]+[volume]+[chỉ báo]", "catalyst": "[cụ thể với mã]"},\n'
        '  "scenario_sell": {'
        f'"probability": 25, "entry": 0, "tp": {round(sl_est*0.97,0)}, "sl": {round(r1_p*1.02,0)}, "rr": 2.0, '
        '"trigger": "BÁN khi [điều kiện cụ thể]", "catalyst": "[cụ thể]"},\n'
        '  "scenario_watch": {'
        '"probability": 25, "trigger": "Chờ khi [giá]+[volume]+[chỉ báo]", "watch_for": "[tín hiệu cụ thể]"},\n'
        '  "support_levels": ['
        f'{{"price": {s1}, "reason": "[lý do kỹ thuật: SMA/Fib/OrderBlock]", "dist_pct": {round((s1-price)/price*100,1)}}}, '
        f'{{"price": {sr["support"][1]["price"] if len(sr["support"])>1 else round(s1*0.97,0)}, "reason": "[lý do]", "dist_pct": 0}}, '
        f'{{"price": {sr["support"][2]["price"] if len(sr["support"])>2 else round(s1*0.94,0)}, "reason": "[lý do]", "dist_pct": 0}}],\n'
        '  "resistance_levels": ['
        f'{{"price": {r1_p}, "reason": "[lý do kỹ thuật]", "dist_pct": {round((r1_p-price)/price*100,1)}}}, '
        f'{{"price": {sr["resistance"][1]["price"] if len(sr["resistance"])>1 else round(r1_p*1.03,0)}, "reason": "[lý do]", "dist_pct": 0}}, '
        f'{{"price": {sr["resistance"][2]["price"] if len(sr["resistance"])>2 else round(r1_p*1.06,0)}, "reason": "[lý do]", "dist_pct": 0}}],\n'
        '  "main_risks": ["[rủi ro cụ thể mã này 1]", "[rủi ro 2]", "[rủi ro 3]"],\n'
        '  "key_catalysts": ["[catalyst cụ thể mã này 1]", "[catalyst 2]"],\n'
        '  "shelf_life_days": 7,\n'
        '  "moderator_summary": "TEXT THUẦN tiếng Việt, không JSON, 80-120 từ.",\n'
        '  "expert_alignment": {"tech_analyst": "AGREE", "macro_strategist": "AGREE", "risk_manager": "AGREE", "smc_trader": "AGREE", "fundamental_filter": "AGREE"}\n'
        "}"
    )
    user = (
        f"Mã: {inp.symbol} | Giá: {price:,.0f}\n"
        f"Vote 3 vòng: MUA={buy_count} BAN={sell_count} THEO DOI={watch_count}\n"
        f"Confidence floor: {base_conf}% (đồng thuận {dominant}/{n_experts})\n\n"
        f"Ý kiến chuyên gia vòng cuối:{opinions_text}\n"
        f"S/R tính sẵn:\n{sr_text}\n"
        f"TP={tp_est:,.0f} SL={sl_est:,.0f} R:R={rr_est:.2f}\n\n"
        "Trả về JSON ĐẦY ĐỦ. moderator_summary là TEXT THUẦN, không JSON. "
        "Đảm bảo dấu ngoặc } cuối cùng có mặt."
    )
    return system, user


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_expert_response(expert: dict, raw_text: str, round_num: int) -> ExpertOpinion:
    """Parse text response của expert. Hỗ trợ format mới (REASON/ENTRY/SL/TP/RR)."""
    lines = raw_text.strip().split("\n")

    stance     = "THEO DOI"
    confidence = 50
    key_points = []
    concern    = ""
    entry      = 0.0
    sl         = 0.0
    tp         = 0.0
    rr         = 0.0
    trigger    = ""

    for i, line in enumerate(lines):
        l = line.strip()
        lu = l.upper()

        if lu.startswith("STANCE:"):
            s = lu.split(":", 1)[1].strip()
            if "MUA" in s:   stance = "MUA"
            elif "BAN" in s: stance = "BAN"
            else:            stance = "THEO DOI"

        elif lu.startswith("CONFIDENCE:"):
            try:
                c = int("".join(filter(str.isdigit, l.split(":", 1)[1][:5])))
                confidence = max(0, min(100, c))
            except Exception:
                pass

        elif lu.startswith("REASON:") or lu.startswith("FINAL_REASON:"):
            reason_text = l.split(":", 1)[1].strip()
            if reason_text:
                key_points.insert(0, reason_text[:120])

        elif lu.startswith("ENTRY:"):
            try:
                val = float("".join(c for c in l.split(":", 1)[1] if c.isdigit() or c == "."))
                entry = val
            except Exception:
                pass

        elif lu.startswith("SL:"):
            # Có thể dạng "SL: 60000  TP: 65000  RR: 2.1"
            parts = l.split("SL:", 1)[1]
            try:
                sl_part = parts.split("TP:")[0] if "TP:" in parts.upper() else parts
                sl = float("".join(c for c in sl_part if c.isdigit() or c == "."))
            except Exception:
                pass
            if "TP:" in parts.upper():
                try:
                    tp_part = parts.upper().split("TP:")[1].split("RR:")[0] if "RR:" in parts.upper() else parts.upper().split("TP:")[1]
                    tp = float("".join(c for c in tp_part if c.isdigit() or c == "."))
                except Exception:
                    pass
            if "RR:" in parts.upper():
                try:
                    rr_part = parts.upper().split("RR:")[1]
                    rr = float("".join(c for c in rr_part if c.isdigit() or c == "."))
                except Exception:
                    pass

        elif lu.startswith("TP:"):
            try:
                tp_part = l.split(":", 1)[1].split()[0]
                tp = float("".join(c for c in tp_part if c.isdigit() or c == "."))
            except Exception:
                pass

        elif lu.startswith("RR:"):
            try:
                rr_part = l.split(":", 1)[1].strip().split()[0]
                rr = float(rr_part)
            except Exception:
                pass

        elif lu.startswith("TRIGGER:") or lu.startswith("TRIGGER CẬP NHẬT:"):
            trigger = l.split(":", 1)[1].strip()[:150]
            key_points.append(f"Trigger: {trigger}") if trigger else None

        elif lu.startswith("RISK:") or "RỦI RO" in lu and ":" in lu:
            concern = l.split(":", 1)[1].strip()[:150]

        elif lu.startswith("CATALYST:"):
            cat = l.split(":", 1)[1].strip()[:120]
            if cat:
                key_points.append(f"Catalyst: {cat}")

        elif lu.startswith("REBUTS:") or "PHẢN BIỆN" in lu and ":" in lu:
            key_points.append(l.split(":", 1)[1].strip()[:100])

        elif lu.startswith("ALERT:"):
            alert = l.split(":", 1)[1].strip()[:120]
            if alert:
                concern = f"⚠️ {alert}" if not concern else concern

        elif lu.startswith("SCENARIO:"):
            # Parse "Entry=62000 SL=60000 TP=65000 RR=2.1"
            sc_text = l.split(":", 1)[1]
            for kv in sc_text.split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    try:
                        val = float("".join(c for c in v if c.isdigit() or c == "."))
                        if k.upper() == "ENTRY": entry = val
                        elif k.upper() == "SL":  sl = val
                        elif k.upper() == "TP":  tp = val
                        elif k.upper() == "RR":  rr = val
                    except Exception:
                        pass

        elif l.startswith("- ") and len(key_points) < 4:
            key_points.append(l[2:].strip()[:120])

    # Validate và tính R:R
    rr_actual = 0.0
    if entry > 0 and tp > 0 and sl > 0 and entry != sl:
        rr_actual = round((tp - entry) / abs(entry - sl), 2)
        # Fix #1: entry phải nằm giữa sl và tp
        if stance == "MUA":
            if entry <= sl or entry >= tp:
                stance = "THEO DOI"
                concern = f"R:R không hợp lệ: Entry={entry} SL={sl} TP={tp}"
                entry = 0
            elif rr_actual < 1.5:
                stance = "THEO DOI"
                concern = f"R:R={rr_actual:.1f} < 1.5, hạ xuống THEO DOI"
                entry = 0
        elif stance == "BAN":
            if entry >= sl or entry <= tp:
                stance = "THEO DOI"
                concern = f"R:R không hợp lệ khi BAN: Entry={entry} SL={sl} TP={tp}"
                entry = 0
            elif rr_actual < 1.5:
                stance = "THEO DOI"
                concern = f"R:R={rr_actual:.1f} < 1.5, hạ xuống THEO DOI"
                entry = 0

    if not key_points:
        # Fallback: lấy câu đầu tiên có nghĩa
        for sent in raw_text.split("."):
            s = sent.strip()
            if len(s) > 20:
                key_points.append(s[:120])
                break

    return ExpertOpinion(
        expert_id  = expert["id"],
        role       = expert["role"],
        stance     = stance,
        confidence = confidence,
        key_points = (key_points or ["Không có lý do rõ ràng"])[:4],
        concern    = concern or "Xem chi tiết phân tích",
        raw_text   = raw_text,
    )


def _parse_moderator_json(raw_text: str, inp: SwarmInput) -> dict:
    """Parse JSON từ moderator. Tự sửa JSON bị cắt, post-process R:R + S/R."""
    import re as _re

    text = raw_text.strip()
    # Strip markdown
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:]).rsplit("```", 1)[0].strip()

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        json_text = text[start:end]
    elif start != -1:
        # JSON bị cắt — tự repair
        json_text = text[start:]
        open_n = json_text.count("{") - json_text.count("}")
        json_text += "}" * max(0, open_n)
        json_text = _re.sub(r',\s*}', '}', json_text)
        json_text = _re.sub(r',\s*]', ']', json_text)
    else:
        json_text = "{}"

    data = None
    for attempt in [json_text, json_text + "}", json_text + "}"]:
        try:
            data = json.loads(attempt)
            break
        except json.JSONDecodeError:
            pass

    if data is None:
        logger.warning("Moderator JSON parse failed, using fallback")
        data = _moderator_fallback(inp)

    # ── Post-process summary (chỉ text thuần) ────────────────────────────
    summary = str(data.get("moderator_summary", ""))
    if "{" in summary and ("panel_verdict" in summary or "\"entry\"" in summary):
        summary = summary[:summary.find("{")].strip()
    data["moderator_summary"] = summary[:500] or "Hội đồng đã hoàn tất phân tích 3 vòng."

    # ── Post-process confidence ───────────────────────────────────────────
    data["panel_confidence"] = max(25.0, float(data.get("panel_confidence", 60)))

    # ── Post-process shelf_life ───────────────────────────────────────────
    data["shelf_life_days"] = max(7, int(data.get("shelf_life_days", 7)))

    # ── Post-process R:R (Fix #1) ─────────────────────────────────────────
    data, rr_warning = _validate_and_fix_rr(data, inp)
    data["rr_warning"] = rr_warning

    # ── Post-process S/R sort (Fix #2) ────────────────────────────────────
    data = _post_process_sr(data, inp)

    return data


def _validate_and_fix_rr(data: dict, inp: SwarmInput) -> tuple[dict, str]:
    """
    Fix #1: Validate R:R cho scenario_buy và scenario_sell.
    - R:R < 1.5 → entry=0, chuyển sang THEO DOI hint
    - entry > tp hoặc entry < sl → báo lỗi, entry=0
    """
    rr_warning = ""
    price = inp.current_price

    for sc_key in ["scenario_buy", "scenario_sell"]:
        sc = data.get(sc_key, {})
        if not sc:
            continue

        entry = float(sc.get("entry", 0) or 0)
        sl    = float(sc.get("sl",    0) or 0)
        tp1   = float(sc.get("tp1",   sc.get("tp", 0)) or 0)

        if entry <= 0 or sl <= 0 or tp1 <= 0:
            continue

        # Validate hướng
        is_buy = sc_key == "scenario_buy"
        valid = True
        if is_buy and (entry <= sl or entry >= tp1):
            valid = False
            rr_warning = f"Lỗi kịch bản MUA: Entry={entry:,.0f} SL={sl:,.0f} TP={tp1:,.0f} — không hợp lệ"
        elif not is_buy and (entry >= sl or entry <= tp1):
            valid = False
            rr_warning = f"Lỗi kịch bản BÁN: Entry={entry:,.0f} SL={sl:,.0f} TP={tp1:,.0f} — không hợp lệ"

        if not valid:
            sc["entry"] = 0
            sc["rr"] = 0
            data[sc_key] = sc
            continue

        # Tính R:R thực tế
        rr_calc = round(abs(tp1 - entry) / abs(entry - sl), 2)
        sc["rr"] = rr_calc

        if rr_calc < 1.5:
            rr_warning = f"R:R={rr_calc:.1f} < 1.5 — kịch bản không phải ưu tiên"
            sc["entry"] = 0   # giữ level nhưng bỏ entry
            # Hạ confidence 10 điểm
            data["panel_confidence"] = max(25, data.get("panel_confidence", 60) - 10)
        elif rr_calc < 2.0 and not rr_warning:
            rr_warning = f"R:R={rr_calc:.1f} dưới ngưỡng tối ưu 2.0"

        data[sc_key] = sc

    return data, rr_warning


def _post_process_sr(data: dict, inp: SwarmInput) -> dict:
    """
    Fix #2: Đảm bảo S/R được sắp xếp gần → xa và không trùng lặp.
    Lấy từ JSON hoặc tính từ _compute_sr_levels nếu thiếu.
    """
    price = inp.current_price
    sr_computed = _compute_sr_levels(inp)

    def _fix_levels(levels: list, direction: str) -> list:
        result = []
        if isinstance(levels, list) and len(levels) >= 1:
            for lv in levels:
                if isinstance(lv, dict) and float(lv.get("price", 0)) > 0:
                    p = float(lv["price"])
                    # Loại bỏ duplicate trong 2.5%
                    if any(abs(p - r["price"]) / r["price"] < 0.025 for r in result):
                        continue
                    # Chỉ lấy mức đúng hướng
                    if direction == "support"    and p >= price:  continue
                    if direction == "resistance" and p <= price:  continue
                    dist_pct = round((p - price) / price * 100, 1)
                    result.append({
                        "price":    round(p, 0),
                        "reason":   str(lv.get("reason", ""))[:80],
                        "dist_pct": dist_pct,
                    })
            # Sắp xếp gần → xa
            result.sort(key=lambda x: abs(x["price"] - price))

        # Nếu thiếu, bổ sung từ computed
        fallback = sr_computed[direction]
        for fb in fallback:
            if len(result) >= 3:
                break
            if not any(abs(fb["price"] - r["price"]) / r["price"] < 0.025 for r in result):
                result.append(fb)

        return result[:3]

    data["support_levels"]    = _fix_levels(data.get("support_levels", []),    "support")
    data["resistance_levels"] = _fix_levels(data.get("resistance_levels", []), "resistance")
    return data


def _moderator_fallback(inp: SwarmInput) -> dict:
    """Fallback hoàn chỉnh khi JSON parse thất bại."""
    price = inp.current_price
    atr   = inp.atr or price * 0.02
    sr    = _compute_sr_levels(inp)
    s1    = sr["support"][0]["price"]    if sr["support"]    else round(price * 0.97, 0)
    r1    = sr["resistance"][0]["price"] if sr["resistance"] else round(price * 1.03, 0)
    tp_e  = inp.tp or r1
    sl_e  = inp.sl or s1
    rr_e  = round((tp_e - price) / (price - sl_e), 2) if (price - sl_e) > 0 else 2.0

    return {
        "panel_verdict":    inp.verdict_label.split()[0] if inp.verdict_label else "THEO DOI",
        "panel_confidence": max(40.0, float(inp.confidence_pct)),
        "rr_warning":       "" if rr_e >= 1.5 else f"R:R={rr_e:.1f} < 1.5",
        "consensus_level":  "PHAN BIEN",
        "scenario_buy": {
            "probability": max(inp.bull_count * 12, 35),
            "entry":  round(price, 0) if rr_e >= 1.5 else 0,
            "tp1":    round(tp_e, 0), "tp2": round(tp_e * 1.04, 0),
            "sl":     round(sl_e, 0), "rr": rr_e,
            "trigger": f"MUA khi giá close trên {r1:,.0f} với volume > 1.5x TB20",
            "catalyst": f"Breakout {r1:,.0f} xác nhận uptrend",
        },
        "scenario_sell": {
            "probability": max(inp.bear_count * 12, 25),
            "entry": 0,
            "tp":    round(sl_e * 0.97, 0), "sl": round(r1 * 1.02, 0), "rr": 2.0,
            "trigger": f"BÁN khi giá phá vỡ {s1:,.0f} với volume > 2x TB20",
            "catalyst": f"Breakdown {s1:,.0f} xác nhận downtrend",
        },
        "scenario_watch": {
            "probability": 25,
            "trigger": f"Theo dõi vùng {round(price*0.99,0):,.0f}–{round(price*1.01,0):,.0f}",
            "watch_for": "Candlestick xác nhận + volume > 1.5x TB20",
        },
        "support_levels":    sr["support"],
        "resistance_levels": sr["resistance"],
        "main_risks": [
            f"Giá phá vỡ {s1:,.0f} → xác nhận downtrend (RSI={inp.rsi:.0f})",
            f"Volume ratio {inp.volume_ratio:.1f}x TB20 — chưa đủ xác nhận",
            f"Market regime {inp.market_regime} có thể đảo chiều",
        ],
        "key_catalysts": [
            f"Kết quả kinh doanh {inp.symbol} — catalyst nội tại",
            f"Breakout {r1:,.0f} với volume lớn",
        ],
        "shelf_life_days": 7,
        "moderator_summary": (
            f"Hội đồng 3 vòng: {inp.verdict_label}, RSI={inp.rsi:.0f}, "
            f"Vol={inp.volume_ratio:.1f}x TB20. Bull={inp.bull_count}/{inp.active_agents}, "
            f"Bear={inp.bear_count}/{inp.active_agents}. Regime={inp.market_regime}."
        ),
        "expert_alignment": {},
    }


# ══════════════════════════════════════════════════════════════════════════════
# SWARM ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class SwarmOrchestrator:
    """Điều phối toàn bộ quá trình tranh luận."""

    def __init__(self, llm: LLMClient, progress_cb=None):
        self.llm     = llm
        self._cb     = progress_cb  # callback(msg: str) để update Telegram
        self._cb_idx = 0

    def _progress(self, msg: str):
        self._cb_idx += 1
        if self._cb:
            try:
                self._cb(f"[{self._cb_idx}] {msg}")
            except Exception:
                pass
        logger.info(f"Swarm: {msg}")

    def run(self, inp: SwarmInput) -> SwarmReport:
        """Chạy 3 vòng tranh luận và trả về SwarmReport."""
        t0  = time.time()
        ctx = _build_context_block(inp)

        # ── VÒNG 1: Ý kiến độc lập + skill context ──────────────────────────
        self._progress(f"Vòng 1/3 — {len(EXPERTS)} chuyên gia phân tích từ skills...")
        r1_opinions: list[ExpertOpinion] = []
        for expert in EXPERTS:
            self._progress(f"  {expert['emoji']} {expert['role']}...")
            try:
                sys_p, usr_p = _build_expert_prompt_round1(expert, ctx, inp)
                raw = self.llm.chat(sys_p, usr_p, LLM_MAX_TOKENS)
                op  = _parse_expert_response(expert, raw, 1)
                r1_opinions.append(op)
                self._progress(f"  → {expert['emoji']} {op.stance}({op.confidence}%)")
            except Exception as e:
                logger.warning(f"Expert {expert['id']} R1 fail: {e}")
                r1_opinions.append(ExpertOpinion(
                    expert_id=expert["id"], role=expert["role"],
                    stance="THEO DOI", confidence=50,
                    key_points=["Lỗi lấy ý kiến"], concern="N/A", raw_text=str(e),
                ))

        # ── VÒNG 2: Phản biện ───────────────────────────────────────────────
        self._progress("Vòng 2/3 — Phản biện chéo...")
        r2_opinions: list[ExpertOpinion] = []
        debate_r2 = DebateRound(round_num=2, exchanges=[])
        for expert in EXPERTS:
            own_r1 = next((op for op in r1_opinions if op.expert_id == expert["id"]), r1_opinions[0])
            self._progress(f"  {expert['emoji']} {expert['role']} phản biện...")
            try:
                sys_p, usr_p = _build_expert_prompt_round2(expert, ctx, r1_opinions, own_r1, inp)
                raw = self.llm.chat(sys_p, usr_p, LLM_MAX_TOKENS)
                op  = _parse_expert_response(expert, raw, 2)
                r2_opinions.append(op)
                debate_r2.exchanges.append({
                    "expert": expert["id"], "role": expert["role"],
                    "stance_r1": own_r1.stance, "stance_r2": op.stance,
                    "changed": own_r1.stance != op.stance, "text": raw[:200],
                })
                change_note = " ⚡ ĐỔI!" if own_r1.stance != op.stance else ""
                self._progress(f"  → {expert['emoji']}: {own_r1.stance}→{op.stance}{change_note}")
            except Exception as e:
                logger.warning(f"Expert {expert['id']} R2 fail: {e}")
                r2_opinions.append(own_r1)

        # ── VÒNG 3: Kết luận cuối + xác nhận kịch bản ───────────────────────
        self._progress("Vòng 3/3 — Kết luận cuối và xác nhận kịch bản...")
        r3_opinions: list[ExpertOpinion] = []
        debate_r3 = DebateRound(round_num=3, exchanges=[])
        for expert in EXPERTS:
            own_r2 = next((op for op in r2_opinions if op.expert_id == expert["id"]), r2_opinions[0])
            self._progress(f"  {expert['emoji']} {expert['role']} kết luận...")
            try:
                sys_p, usr_p = _build_expert_prompt_round3(expert, ctx, r2_opinions, own_r2, inp)
                raw = self.llm.chat(sys_p, usr_p, LLM_MAX_TOKENS)
                op  = _parse_expert_response(expert, raw, 3)
                r3_opinions.append(op)
                debate_r3.exchanges.append({
                    "expert": expert["id"], "role": expert["role"],
                    "stance_r2": own_r2.stance, "stance_r3": op.stance,
                    "changed": own_r2.stance != op.stance, "text": raw[:200],
                })
                self._progress(f"  → {expert['emoji']} FINAL: {op.stance}({op.confidence}%)")
            except Exception as e:
                logger.warning(f"Expert {expert['id']} R3 fail: {e}")
                r3_opinions.append(own_r2)

        # ── MODERATOR ────────────────────────────────────────────────────────
        self._progress("Moderator tổng hợp 3 vòng...")
        try:
            sys_p, usr_p = _build_moderator_prompt(inp, ctx, r1_opinions, r2_opinions, r3_opinions)
            raw_mod  = self.llm.chat(sys_p, usr_p, MODERATOR_TOKENS)
            mod_data = _parse_moderator_json(raw_mod, inp)
        except Exception as e:
            logger.warning(f"Moderator fail: {e}")
            mod_data = _parse_moderator_json("", inp)

        # ── BUILD REPORT ─────────────────────────────────────────────────────
        elapsed    = round(time.time() - t0, 1)
        verdict    = mod_data.get("panel_verdict", "THEO DOI")
        shelf_life = max(7, int(mod_data.get("shelf_life_days", 7)))
        review_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        expires_at  = (datetime.now() + timedelta(days=shelf_life)).strftime("%Y-%m-%d")

        # Resonance (Fix #4)
        stances_final = [op.stance for op in r3_opinions]
        vote_bull    = stances_final.count("MUA")
        vote_bear    = stances_final.count("BAN")
        vote_neutral = stances_final.count("THEO DOI")
        n_exp        = len(stances_final) or 5
        final_score  = max(-5.0, min(5.0, float(vote_bull - vote_bear)))

        dominant_stance = max(stances_final, key=stances_final.count) if stances_final else "THEO DOI"
        dom_ops     = [op for op in r3_opinions if op.stance == dominant_stance]
        dom_avg_conf = sum(op.confidence for op in dom_ops) / len(dom_ops) if dom_ops else 60
        resonance_pct = round((len(dom_ops) / n_exp) * dom_avg_conf, 1)
        dom_count   = len(dom_ops)

        conf_floor = max(40, round((dom_count / n_exp) * dom_avg_conf))
        if dom_count == n_exp:        conf_floor = max(conf_floor, 65)
        elif dom_count >= n_exp - 1:  conf_floor = max(conf_floor, 55)
        final_confidence = max(float(mod_data.get("panel_confidence", 60)), conf_floor)

        if dom_count == n_exp:            consensus_level = "DONG THUAN"
        elif dom_count >= n_exp - 1:      consensus_level = "PHAN BIEN"
        else:                             consensus_level = "CHIA RE"

        # Dissent Notes (Fix #4)
        dissent_notes: list[str] = []
        for op in r3_opinions:
            if op.stance != dominant_stance:
                em   = next((e["emoji"] for e in EXPERTS if e["id"] == op.expert_id), "👤")
                note = op.key_points[0][:100] if op.key_points else op.concern[:80]
                dissent_notes.append(f"⚠️ Cảnh báo từ {em} {op.role}: {note}")

        input_summary = (
            f"{inp.symbol} | {inp.verdict_label} ({inp.confidence_pct:.0f}%) | "
            f"Bull={inp.bull_count} Bear={inp.bear_count} | "
            f"RSI={inp.rsi:.1f} | Vol={inp.volume_ratio:.1f}x"
        )
        self._progress(f"✅ {elapsed:.0f}s | {verdict} | Score={final_score:+.0f} | Resonance={resonance_pct:.0f}%")

        return SwarmReport(
            symbol             = inp.symbol,
            timestamp          = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_s          = elapsed,
            llm_provider       = self.llm.provider,
            llm_model          = self.llm.model,
            panel_verdict      = verdict,
            panel_confidence   = final_confidence,
            consensus_level    = consensus_level,
            scenario_buy       = mod_data.get("scenario_buy",   {}),
            scenario_sell      = mod_data.get("scenario_sell",  {}),
            scenario_watch     = mod_data.get("scenario_watch", {}),
            main_risks         = mod_data.get("main_risks",     []),
            key_catalysts      = mod_data.get("key_catalysts",  []),
            shelf_life_days    = shelf_life,
            expires_at         = expires_at,
            review_date        = review_date,
            support_levels     = mod_data.get("support_levels",    []),
            resistance_levels  = mod_data.get("resistance_levels", []),
            rr_warning         = mod_data.get("rr_warning", ""),
            final_score        = final_score,
            resonance_pct      = resonance_pct,
            vote_bull          = vote_bull,
            vote_neutral       = vote_neutral,
            vote_bear          = vote_bear,
            dissent_notes      = dissent_notes,
            expert_opinions    = r3_opinions,
            debate_rounds      = [debate_r2, debate_r3],
            moderator_summary  = mod_data.get("moderator_summary", ""),
            moderator_raw_json = mod_data,
            input_summary      = input_summary,
        )


# ══════════════════════════════════════════════════════════════════════════════
# REPORT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def format_swarm_report(report: SwarmReport) -> str:
    """
    Định dạng SwarmReport thành text cho Telegram.
    Fix #3: mỗi expert đúng 2 dòng.
    Fix #4: Resonance panel + Dissent Notes.
    Fix #6 (session trước): JSON raw ẩn.
    """
    SEP  = "═" * 38
    SEP2 = "─" * 38

    v_em = {"MUA": "🟢", "BAN": "🔴", "THEO DOI": "🟡"}.get(report.panel_verdict, "🟡")
    c_em = {"DONG THUAN": "✅", "PHAN BIEN": "⚡", "CHIA RE": "❌"}.get(report.consensus_level, "⚡")

    # ── Header ────────────────────────────────────────────────────────────────
    lines = [
        f"LOCAL SWARM v{SWARM_VERSION}: {report.symbol}",
        SEP,
        f"🤖 {report.llm_provider}/{report.llm_model} | ⏱️ {report.elapsed_s:.0f}s",
        f"📥 {report.input_summary}",
        SEP,
        "",
        f"PHÁN QUYẾT: {v_em} {report.panel_verdict}",
        f"Tin cậy    : {report.panel_confidence:.0f}% | {c_em} {report.consensus_level}",
    ]

    rr_warn = getattr(report, "rr_warning", "")
    if rr_warn:
        lines.append(f"⚠️  {rr_warn}")
    lines.append("")

    # ── Bảng Resonance (Fix #4) ───────────────────────────────────────────────
    vote_bull    = getattr(report, "vote_bull",    0)
    vote_neutral = getattr(report, "vote_neutral", 0)
    vote_bear    = getattr(report, "vote_bear",    0)
    final_score  = getattr(report, "final_score",  0.0)
    resonance    = getattr(report, "resonance_pct", 0.0)

    lines += [SEP2, "RESONANCE PANEL:", SEP2]
    score_bar = "█" * max(0, int(abs(final_score))) + "░" * (5 - max(0, int(abs(final_score))))
    score_dir = "+" if final_score >= 0 else "-"
    lines.append(f"Final Score : {score_dir}{abs(final_score):.0f}/5  [{score_bar}]")
    lines.append(f"Resonance   : {resonance:.0f}%")
    lines.append(f"Vote Split  : 🟢Bull={vote_bull}  ⚪Neutral={vote_neutral}  🔴Bear={vote_bear}")
    lines.append("")

    # ── Chuyên gia: đúng 2 dòng mỗi người (Fix #3) ───────────────────────────
    lines += [SEP2, "CHUYÊN GIA (3 vòng):", SEP2]
    for op in report.expert_opinions:
        em    = next((e["emoji"] for e in EXPERTS if e["id"] == op.expert_id), "👤")
        s_em  = {"MUA": "🟢", "BAN": "🔴", "THEO DOI": "🟡"}.get(op.stance, "🟡")
        # Dòng 1: tên + stance + confidence
        lines.append(f"{em} {op.role}: {s_em} {op.stance} ({op.confidence}%)")
        # Dòng 2: lý do chính, tối đa 80 ký tự, không cắt chữ
        reason = op.key_points[0] if op.key_points else op.concern
        reason = reason.strip()[:80]
        # Đảm bảo không cắt chữ giữa chừng
        if len(reason) == 80 and not reason[-1].isspace():
            last_space = reason.rfind(" ")
            if last_space > 50:
                reason = reason[:last_space] + "…"
            else:
                reason = reason + "…"
        lines.append(f"   └─ {reason}")
    lines.append("")

    # Debate changes tóm tắt
    changed_all = []
    for rd in report.debate_rounds:
        changed_all.extend([x for x in rd.exchanges if x.get("changed")])
    if changed_all:
        lines.append(f"⚡ ĐỔI Ý KIẾN: {len(changed_all)} lần qua 3 vòng")
        for x in changed_all[-3:]:  # chỉ show 3 gần nhất
            r1 = x.get("stance_r1", x.get("stance_r2", "?"))
            r2 = x.get("stance_r2", x.get("stance_r3", "?"))
            lines.append(f"  • {x['role'][:20]}: {r1}→{r2}")
        lines.append("")

    # ── Dissent Notes (Fix #4) ────────────────────────────────────────────────
    dissent = getattr(report, "dissent_notes", [])
    if dissent:
        lines += [SEP2, "DISSENT NOTE:", SEP2]
        for note in dissent[:3]:
            # Wrap tối đa 75 ký tự
            wrapped = textwrap.fill(note, width=75, subsequent_indent="    ")
            lines.append(wrapped)
        lines.append("")

    # ── Hỗ trợ / Kháng cự (Fix #2 — sắp gần→xa) ─────────────────────────────
    sup_levels = getattr(report, "support_levels",    []) or []
    res_levels = getattr(report, "resistance_levels", []) or []
    if sup_levels or res_levels:
        lines += [SEP2, "HỖ TRỢ / KHÁNG CỰ (gần → xa):", SEP2]
        for i, s in enumerate(sup_levels[:3], 1):
            if isinstance(s, dict):
                dist = s.get("dist_pct", "")
                dist_str = f" ({dist:+.1f}%)" if isinstance(dist, (int, float)) else ""
                lines.append(f"  🔵 S{i}: {s.get('price',0):,.0f}{dist_str}  — {str(s.get('reason',''))[:60]}")
        for i, r in enumerate(res_levels[:3], 1):
            if isinstance(r, dict):
                dist = r.get("dist_pct", "")
                dist_str = f" ({dist:+.1f}%)" if isinstance(dist, (int, float)) else ""
                lines.append(f"  🔴 R{i}: {r.get('price',0):,.0f}{dist_str}  — {str(r.get('reason',''))[:60]}")
        lines.append("")

    # ── Kịch bản — sắp xếp ưu tiên theo probability ──────────────────────────
    lines += [SEP2, "KỊCH BẢN ĐẦU TƯ:", SEP2]
    all_sc = sorted([
        ("MUA",      report.scenario_buy   or {}, "🟢"),
        ("BAN",      report.scenario_sell  or {}, "🔴"),
        ("THEO DOI", report.scenario_watch or {}, "🟡"),
    ], key=lambda x: x[1].get("probability", 0), reverse=True)

    for idx, (sc_type, sc, sc_em) in enumerate(all_sc):
        if not sc: continue
        prob     = sc.get("probability", 0)
        priority = "ƯU TIÊN" if idx == 0 else "DỰ PHÒNG"
        lines.append(f"{sc_em} {priority} — {sc_type} ({prob}%):")

        entry = float(sc.get("entry", 0) or 0)
        tp1   = float(sc.get("tp1", sc.get("tp", 0)) or 0)
        tp2   = float(sc.get("tp2", 0) or 0)
        sl    = float(sc.get("sl",  0) or 0)
        rr    = float(sc.get("rr",  0) or 0)

        if sc_type in ("MUA", "BAN") and entry > 0:
            lines.append(f"  Entry    : {entry:,.0f}")
        else:
            lines.append(f"  Entry    : — (chờ trigger)")

        if tp1 > 0:
            lines.append(f"  TP       : {tp1:,.0f}" + (f" / {tp2:,.0f}" if tp2 > 0 else ""))
        if sl > 0:
            lines.append(f"  Stop Loss: {sl:,.0f}")
        if rr > 0:
            rr_flag = " ⚠️ <1.5" if rr < 1.5 else (" ⚠️ <2.0" if rr < 2.0 else "")
            lines.append(f"  R:R      : {rr:.1f}{rr_flag}")

        trigger = sc.get("trigger", sc.get("condition", ""))
        if trigger:
            lines.append(f"  🎯 Trigger: {str(trigger)[:110]}")

        cat = sc.get("catalyst", sc.get("watch_for", ""))
        if cat:
            lines.append(f"  Catalyst : {str(cat)[:90]}")
        lines.append("")

    # ── Risks + Catalysts ─────────────────────────────────────────────────────
    if report.main_risks:
        lines += [SEP2, "RỦI RO CHÍNH:", SEP2]
        for i, risk in enumerate(report.main_risks[:4], 1):
            lines.append(f"  {i}. {str(risk)[:100]}")
        lines.append("")
    if report.key_catalysts:
        lines += ["CATALYST:"]
        for cat in report.key_catalysts[:3]:
            lines.append(f"  + {str(cat)[:90]}")
        lines.append("")

    # ── Shelf life ────────────────────────────────────────────────────────────
    review_date = getattr(report, "review_date", report.expires_at)
    lines += [
        SEP2,
        f"Hạn tín hiệu  : {report.shelf_life_days} ngày",
        f"Đánh giá lại  : {review_date}",
        f"Hết hạn       : {report.expires_at}",
        SEP2, "",
        "TỔNG HỢP:", "",
    ]

    summary = report.moderator_summary or ""
    # Guard: nếu summary chứa JSON artifacts
    if "{" in summary and "panel_verdict" in summary:
        summary = summary[:summary.find("{")].strip()
    if summary:
        for para in summary.split("\n"):
            para = para.strip()
            if para:
                lines.append(textwrap.fill(para, width=54))
    lines.append("")

    lines += [
        SEP,
        "⚠️ Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.",
        f"📋 Local Swarm v{SWARM_VERSION} | /local_swarm {report.symbol}",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_local_swarm(
    symbol: str,
    meta: dict | None = None,
    vibe_result: dict | None = None,
    progress_cb=None,
) -> tuple[str, SwarmReport]:
    """
    Entry point chính — chạy Local Swarm Panel.

    Args:
        symbol:      Mã cổ phiếu
        meta:        dict từ analyze_stock_full() [ưu tiên]
        vibe_result: dict từ run_vibe_agents() [fallback]
        progress_cb: callback(msg) để stream progress về Telegram

    Returns:
        (formatted_text, SwarmReport)
    """
    if meta:
        inp = SwarmInput.from_analyze_result(symbol, meta)
    elif vibe_result:
        inp = SwarmInput.from_vibe_result(symbol, vibe_result)
    else:
        raise ValueError("Cần cung cấp meta HOẶC vibe_result")

    llm         = LLMClient()
    orchestrator = SwarmOrchestrator(llm, progress_cb=progress_cb)
    report      = orchestrator.run(inp)
    text        = format_swarm_report(report)
    return text, report


def check_local_swarm_available() -> tuple[bool, str]:
    """
    Kiểm tra xem Local Swarm có thể chạy không.
    Returns: (available, provider_info)
    """
    try:
        llm = LLMClient()
        # Test call nhanh
        resp = llm.chat(
            "Trả lời đúng 1 chữ.",
            "Viết chữ 'OK'",
            max_tokens=5,
        )
        return True, f"{llm.provider}/{llm.model} (test: {resp.strip()[:10]})"
    except RuntimeError as e:
        return False, str(e)
    except Exception as e:
        return False, f"LLM error: {str(e)[:100]}"
