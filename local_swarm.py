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

DEBATE_ROUNDS    = 2        # số vòng tranh luận
LLM_TIMEOUT      = 90       # giây timeout per LLM call
LLM_MAX_TOKENS   = 900      # token tối đa mỗi response chuyên gia
MODERATOR_TOKENS = 1800     # tăng lên để JSON không bị cắt
SWARM_VERSION    = "1.1"

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
            sma20          = float(ind.get("sma20", 0)),
            sma50          = float(ind.get("sma50", 0)),
            market_regime  = meta.get("macro_v", {}).get("market_regime", "UNKNOWN"),
            macro_label    = meta.get("macro_v", {}).get("label", ""),
            news_sentiment = "",
            contradictions = v.get("contradictions", []),
            support        = float(ind.get("support", price * 0.95)),
            resistance     = float(ind.get("resistance", price * 1.05)),
            tp             = float(v.get("tp", 0)),
            sl             = float(v.get("sl", 0)),
            entry          = float(v.get("entry_price", price)),
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
    review_date:     str      # ngày hiện tại + 7 ngày

    # Support/resistance chi tiết
    support_levels:    list   # [{price, reason}, ...]
    resistance_levels: list   # [{price, reason}, ...]

    # R:R warning
    rr_warning: str

    # Expert opinions
    expert_opinions: list[ExpertOpinion]
    debate_rounds:   list[DebateRound]

    # Moderator summary (text thuần, KHÔNG chứa JSON)
    moderator_summary: str

    # Raw JSON từ moderator — CHỈ dùng để lưu DB, không hiển thị user
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
    # Tóm tắt signals
    bull_engines = [k for k, v in inp.signals.items() if v > 0]
    bear_engines = [k for k, v in inp.signals.items() if v < 0]
    neutral      = [k for k, v in inp.signals.items() if v == 0]

    bull_str    = ", ".join(bull_engines) or "Không có"
    bear_str    = ", ".join(bear_engines) or "Không có"

    # Tính hỗ trợ/kháng cự từ ATR nếu chưa có
    price   = inp.current_price
    support = inp.support or round(price * 0.95, 0)
    resist  = inp.resistance or round(price * 1.05, 0)
    tp      = inp.tp or round(price * 1.08, 0)
    sl      = inp.sl or round(price * 0.95, 0)
    rr      = round((tp - price) / (price - sl), 2) if (price - sl) > 0 else 0

    ctx = f"""
=== DỮ LIỆU PHÂN TÍCH: {inp.symbol} ===
Thời điểm: {datetime.now().strftime("%Y-%m-%d %H:%M")}

--- GIÁ & KỸ THUẬT ---
Giá hiện tại  : {price:,.0f} VND
RSI(14)        : {inp.rsi:.1f}
MACD           : {inp.macd:.2f}
ATR(14)        : {inp.atr:,.0f}
Volume ratio   : {inp.volume_ratio:.2f}x TB20
Thay đổi 1D   : {inp.change_1d_pct:+.2f}%
Thay đổi 1W   : {inp.change_1w_pct:+.2f}%
Thay đổi 1M   : {inp.change_1m_pct:+.2f}%
SMA20          : {inp.sma20:,.0f}  |  SMA50: {inp.sma50:,.0f}
Hỗ trợ        : {support:,.0f}   |  Kháng cự: {resist:,.0f}
TP gợi ý      : {tp:,.0f}   |  SL gợi ý: {sl:,.0f}   |  R:R: {rr:.2f}

--- KẾT QUẢ 16 ENGINES ---
Phán quyết tổng: {inp.verdict_label} ({inp.confidence_pct:.0f}%)
Bull signals  : {inp.bull_count}/{inp.active_agents} — Engines: {bull_str}
Bear signals  : {inp.bear_count}/{inp.active_agents} — Engines: {bear_str}
Neutral       : {len(neutral)} engines

--- VĨ MÔ & THỊ TRƯỜNG ---
Market Regime  : {inp.market_regime}
Macro context  : {inp.macro_label or "N/A"}
News sentiment : {inp.news_sentiment or "N/A"}

--- MÂU THUẪN PHÁT HIỆN ---
{chr(10).join(inp.contradictions) if inp.contradictions else "Không có mâu thuẫn rõ ràng"}
""".strip()
    return ctx


def _build_expert_prompt_round1(expert: dict, ctx: str) -> tuple[str, str]:
    """System + user prompt cho vòng 1: ý kiến độc lập."""
    system = (
        f"Bạn là {expert['role']} trong Hội đồng Chuyên gia Phân tích Cổ phiếu VN.\n"
        f"{expert['focus']}\n\n"
        "QUY TẮC BẮT BUỘC (vi phạm = output vô nghĩa):\n"
        "1. STANCE=THEO DOI → KHÔNG đặt entry = giá hiện tại. "
        "Trigger phải có: [giá cụ thể] + [volume điều kiện] + [chỉ báo kỹ thuật].\n"
        "2. STANCE=MUA hoặc BAN → phải có Entry, SL, TP, R:R đầy đủ.\n"
        "3. R:R < 2.0 → BẮT BUỘC hạ STANCE xuống THEO DOI và ghi rõ lý do.\n"
        "4. Mỗi mức hỗ trợ/kháng cự phải KHÁC NHAU ít nhất 3% giá trị. "
        "Không được có 2 mức giống nhau. Phải nêu lý do kỹ thuật cụ thể.\n"
        "5. Rủi ro và Catalyst phải gắn với MÃ NÀY cụ thể, "
        "không dùng cụm chung chung như 'biến động thị trường' hay 'rủi ro vĩ mô'.\n\n"
        "ĐỊNH DẠNG TRẢ LỜI (bắt buộc theo thứ tự):\n"
        "STANCE: [MUA/BAN/THEO DOI]\n"
        "CONFIDENCE: [0-100]\n"
        "ĐIỂM CHÍNH:\n"
        "- [điểm 1 gắn với chỉ báo/mức giá cụ thể của mã]\n"
        "- [điểm 2 gắn với chỉ báo/mức giá cụ thể của mã]\n"
        "- [điểm 3 gắn với chỉ báo/mức giá cụ thể của mã]\n"
        "HỖ TRỢ:\n"
        "- S1: [giá] — [lý do: SMA20/Kijun-sen/Fibonacci/OrderBlock/...]\n"
        "- S2: [giá, phải khác S1 ít nhất 3%] — [lý do khác S1]\n"
        "- S3: [giá, phải khác S2 ít nhất 3%] — [lý do khác S2]\n"
        "KHÁNG CỰ:\n"
        "- R1: [giá] — [lý do]\n"
        "- R2: [giá, phải khác R1 ít nhất 3%] — [lý do khác R1]\n"
        "- R3: [giá, phải khác R2 ít nhất 3%] — [lý do khác R2]\n"
        "TRIGGER: [Nếu THEO DOI: 'Chờ khi giá [X] VÀ volume [Y]x TB20 VÀ [chỉ báo Z]'. "
        "Nếu MUA/BAN: 'Vào lệnh khi [điều kiện cụ thể]']\n"
        "RỦI RO CHÍNH: [Gắn với mã cụ thể: mức giá nào, chỉ báo nào, sự kiện nào]\n"
        "CATALYST: [Gắn với mã cụ thể: kết quả KD, dòng tiền ngành, sự kiện sắp tới]\n"
        "LUẬN ĐIỂM: [80-120 từ, tiếng Việt, phân tích từ góc nhìn chuyên môn]"
    )
    symbol = ctx.split()[4] if len(ctx.split()) > 4 else "này"
    user = (
        f"Phân tích 16 engines cho mã {symbol}. Đưa ra ý kiến ĐỘC LẬP:\n\n{ctx}\n\n"
        "Lưu ý quan trọng:\n"
        "- Tính R:R từ entry/SL/TP thực tế. Nếu R:R < 2.0 thì phải THEO DOI.\n"
        "- Hỗ trợ/kháng cự: dùng SMA20, SMA50, ATR, Fibonacci từ dữ liệu trên.\n"
        "- Rủi ro phải gắn với mức giá hoặc chỉ báo cụ thể của mã này."
    )
    return system, user


def _build_expert_prompt_round2(
    expert: dict, ctx: str,
    other_opinions: list[ExpertOpinion],
    own_round1: ExpertOpinion,
) -> tuple[str, str]:
    """System + user prompt cho vòng 2: phản biện."""
    # Tóm tắt ý kiến các chuyên gia khác
    others_summary = []
    for op in other_opinions:
        if op.expert_id == expert["id"]:
            continue
        others_summary.append(
            f"[{op.role}] STANCE={op.stance} | {'; '.join(op.key_points[:2])}"
        )

    system = (
        f"Bạn là {expert['role']} trong Hội đồng Chuyên gia.\n"
        f"{expert['focus']}\n\n"
        "VÒNG 2 — PHẢN BIỆN:\n"
        "1. Bảo vệ hoặc điều chỉnh stance dựa trên lập luận của các expert khác\n"
        "2. Phản biện điểm yếu nhất trong ý kiến đối lập\n"
        "3. Chỉ ra 1 điều bạn ĐỒNG Ý với expert khác\n\n"
        "QUY TẮC GIỮ NGUYÊN:\n"
        "- THEO DOI → phải có trigger cụ thể (giá + volume + chỉ báo)\n"
        "- Nếu Risk Manager chỉ ra R:R < 2.0 và bạn không phản bác được → hạ stance\n"
        "- Rủi ro và catalyst phải gắn với số liệu cụ thể của mã này\n\n"
        "ĐỊNH DẠNG:\n"
        "STANCE: [MUA/BAN/THEO DOI]\n"
        "CONFIDENCE: [0-100]\n"
        "PHẢN BIỆN: [Expert nào, điểm gì, tại sao sai]\n"
        "ĐỒNG Ý VỚI: [Expert nào, điểm gì]\n"
        "TRIGGER CẬP NHẬT: [Điều kiện sau khi cân nhắc các ý kiến]\n"
        "KẾT LUẬN: [40-60 từ, tiếng Việt]"
    )

    user = (
        f"Dữ liệu: {ctx[:400]}...\n\n"
        f"Ý kiến vòng 1 của bạn: STANCE={own_round1.stance} ({own_round1.confidence}%)\n"
        f"Điểm chính: {'; '.join(own_round1.key_points[:2])}\n\n"
        f"Ý kiến các chuyên gia khác:\n"
        + "\n".join(others_summary)
        + "\n\nHãy phản biện và cập nhật trigger của bạn."
    )
    return system, user


def _build_moderator_prompt(
    inp: SwarmInput,
    ctx: str,
    r1_opinions: list[ExpertOpinion],
    r2_opinions: list[ExpertOpinion],
) -> tuple[str, str]:
    """Prompt cho moderator tổng hợp kết luận cuối."""
    stances = {op.expert_id: op.stance for op in r2_opinions}
    confs   = {op.expert_id: op.confidence for op in r2_opinions}

    buy_count   = sum(1 for s in stances.values() if s == "MUA")
    sell_count  = sum(1 for s in stances.values() if s == "BAN")
    watch_count = sum(1 for s in stances.values() if s == "THEO DOI")
    n_experts   = len(stances) or 5

    # Tính confidence dựa trên đồng thuận (Fix #2)
    dominant = max(buy_count, sell_count, watch_count)
    consensus_pct = dominant / n_experts  # 0.0–1.0
    avg_conf = sum(confs.values()) / len(confs) if confs else 60
    # confidence = đồng thuận * trung bình confidence experts, sàn 60% khi đồng thuận
    base_conf = round(consensus_pct * avg_conf)
    if dominant == n_experts:        # 5/5 đồng thuận
        base_conf = max(base_conf, 65)
    elif dominant >= n_experts - 1:  # 4/5
        base_conf = max(base_conf, 55)

    price  = inp.current_price
    atr    = inp.atr or price * 0.02

    # Tính các mức S/R đảm bảo cách nhau ít nhất 3% (Fix #3)
    s1 = inp.support    or round(price * 0.97, 0)
    s2 = round(min(s1 * 0.97, inp.sma20 or s1 * 0.97), 0)  # SMA20 hoặc -3%
    s3 = round(min(s2 * 0.97, inp.sma50 or s2 * 0.97), 0)  # SMA50 hoặc -3%
    r1 = inp.resistance or round(price * 1.03, 0)
    r2 = round(max(r1 * 1.03, r1 + atr), 0)
    r3 = round(max(r2 * 1.03, r2 + atr), 0)

    tp_est = inp.tp or round(r1, 0)
    sl_est = inp.sl or s1
    rr_est = round((tp_est - price) / (price - sl_est), 2) if (price - sl_est) > 0 else 0

    opinions_text = ""
    for op in r2_opinions:
        ename = next((e["emoji"] + " " + e["role"] for e in EXPERTS if e["id"] == op.expert_id), op.expert_id)
        opinions_text += (
            f"\n{ename}: {op.stance} ({op.confidence}%)\n"
            f"  Điểm: {'; '.join(op.key_points[:2])}\n"
            f"  Rủi ro: {op.concern}\n"
        )

    system = (
        "Bạn là MODERATOR Hội đồng Chuyên gia Phân tích Cổ phiếu VN.\n"
        "Trả về JSON HOÀN CHỈNH không cắt giữa chừng. Không dùng markdown.\n\n"
        "QUY TẮC BẮT BUỘC:\n"
        f"1. panel_confidence tối thiểu {base_conf} (đã tính từ đồng thuận {dominant}/{n_experts} experts).\n"
        "2. Nếu panel_verdict=THEO DOI: scenario_buy.entry=0, phải có trigger đầy đủ.\n"
        "3. Tính R:R=(tp1-entry)/(entry-sl). Nếu R:R<2.0: thêm rr_warning và giảm confidence 10.\n"
        "4. support/resistance: mỗi mức phải cách nhau ít nhất 3%. Phải có lý do kỹ thuật.\n"
        "5. main_risks và key_catalysts: gắn với MÃ CỤ THỂ, không dùng cụm chung chung.\n"
        "6. moderator_summary: chỉ là TEXT THUẦN 80-100 từ, không chứa JSON.\n"
        "7. shelf_life_days tối thiểu 7.\n\n"
        "JSON SCHEMA:\n"
        "{\n"
        '  "panel_verdict": "MUA|BAN|THEO DOI",\n'
        f'  "panel_confidence": {base_conf},\n'
        '  "rr_warning": "",\n'
        '  "consensus_level": "DONG THUAN|PHAN BIEN|CHIA RE",\n'
        '  "scenario_buy": {\n'
        '    "probability": 55,\n'
        f'    "entry": {round(price,0)},\n'
        f'    "tp1": {round(tp_est,0)}, "tp2": {round(tp_est*1.05,0)},\n'
        f'    "sl": {round(sl_est,0)},\n'
        f'    "rr": {rr_est},\n'
        '    "trigger": "MUA khi giá close trên [X] với volume > 1.5x TB20 và MACD dương",\n'
        '    "catalyst": "[Sự kiện/chỉ báo cụ thể của mã này]"\n'
        "  },\n"
        '  "scenario_sell": {\n'
        '    "probability": 25,\n'
        f'    "entry": {round(price,0)},\n'
        f'    "tp": {round(sl_est,0)},\n'
        f'    "sl": {round(r1,0)},\n'
        '    "rr": 2.0,\n'
        '    "trigger": "BÁN khi giá phá vỡ [hỗ trợ X] với volume > 2x TB20",\n'
        '    "catalyst": "[Điều kiện giảm cụ thể]"\n'
        "  },\n"
        '  "scenario_watch": {\n'
        '    "probability": 20,\n'
        '    "trigger": "Chờ khi giá [X] VÀ volume [Y]x TB20 VÀ [chỉ báo Z]",\n'
        '    "watch_for": "[Tín hiệu cụ thể]"\n'
        "  },\n"
        '  "support_levels": [\n'
        f'    {{"price": {s1}, "reason": "Hỗ trợ kỹ thuật — [lý do từ dữ liệu]"}},\n'
        f'    {{"price": {s2}, "reason": "SMA20 hoặc Fibonacci — [lý do]"}},\n'
        f'    {{"price": {s3}, "reason": "SMA50 hoặc đáy sóng — [lý do]"}}\n'
        "  ],\n"
        '  "resistance_levels": [\n'
        f'    {{"price": {r1}, "reason": "Kháng cự gần nhất — [lý do]"}},\n'
        f'    {{"price": {r2}, "reason": "Kháng cự thứ cấp — [lý do]"}},\n'
        f'    {{"price": {r3}, "reason": "Kháng cự mạnh — [lý do]"}}\n'
        "  ],\n"
        '  "main_risks": [\n'
        '    "[Rủi ro 1: mức giá/chỉ báo cụ thể của mã này]",\n'
        '    "[Rủi ro 2: sự kiện/ngành cụ thể]",\n'
        '    "[Rủi ro 3: kỹ thuật cụ thể]"\n'
        "  ],\n"
        '  "key_catalysts": [\n'
        '    "[Catalyst 1: sự kiện/kết quả cụ thể của mã]",\n'
        '    "[Catalyst 2: dòng tiền/ngành cụ thể]"\n'
        "  ],\n"
        '  "shelf_life_days": 7,\n'
        '  "moderator_summary": "TEXT THUẦN 80-100 từ tiếng Việt. Không JSON. Đồng thuận là gì, bất đồng ở đâu, lý do verdict.",\n'
        '  "expert_alignment": {"tech_analyst": "AGREE", "macro_strategist": "AGREE", "risk_manager": "AGREE", "smc_trader": "AGREE", "fundamental_filter": "AGREE"}\n'
        "}"
    )

    user = (
        f"Dữ liệu mã {inp.symbol}:\n{ctx}\n\n"
        f"Vote sau 2 vòng: MUA={buy_count} | BAN={sell_count} | THEO DOI={watch_count}\n"
        f"Đồng thuận: {dominant}/{n_experts} ({dominant/n_experts*100:.0f}%) "
        f"→ panel_confidence tối thiểu {base_conf}\n\n"
        f"Ý kiến chuyên gia vòng 2:{opinions_text}\n"
        f"Thông tin tính scenario: Giá={price:,.0f} | ATR={atr:,.0f} | "
        f"TP={tp_est:,.0f} | SL={sl_est:,.0f} | R:R={rr_est:.2f}\n\n"
        "Trả về JSON HOÀN CHỈNH. moderator_summary chỉ là TEXT, không JSON bên trong.\n"
        "Đảm bảo dấu ngoặc đóng cuối cùng } có mặt."
    )
    return system, user


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_expert_response(expert: dict, raw_text: str, round_num: int) -> ExpertOpinion:
    """Parse text response của expert thành ExpertOpinion."""
    lines = raw_text.strip().split("\n")

    stance     = "THEO DOI"
    confidence = 50
    key_points = []
    concern    = ""

    for i, line in enumerate(lines):
        line = line.strip()
        if line.upper().startswith("STANCE:"):
            s = line.split(":", 1)[1].strip().upper()
            if "MUA" in s:   stance = "MUA"
            elif "BAN" in s: stance = "BAN"
            else:             stance = "THEO DOI"
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                c = int("".join(filter(str.isdigit, line.split(":", 1)[1])))
                confidence = max(0, min(100, c))
            except Exception:
                pass
        elif line.startswith("- ") and len(key_points) < 3:
            key_points.append(line[2:].strip())
        elif "RỦI RO" in line.upper() and ":" in line:
            concern = line.split(":", 1)[1].strip()
        elif "PHẢN BIỆN:" in line.upper():
            concern = lines[i + 1].strip() if i + 1 < len(lines) else concern

    if not key_points:
        # Fallback: lấy 3 câu đầu từ raw_text
        sentences = [s.strip() for s in raw_text.split(".") if len(s.strip()) > 20]
        key_points = sentences[:3]

    return ExpertOpinion(
        expert_id  = expert["id"],
        role       = expert["role"],
        stance     = stance,
        confidence = confidence,
        key_points = key_points or ["Không có điểm chính"],
        concern    = concern or "Không xác định",
        raw_text   = raw_text,
    )


def _parse_moderator_json(raw_text: str, inp: SwarmInput) -> dict:
    """
    Parse JSON từ moderator.
    - Xử lý JSON bị cắt ngang (Fix #1): thử repair trước khi fallback
    - Đảm bảo moderator_summary là text thuần, không chứa JSON (Fix #6)
    - Đảm bảo confidence hợp lý (Fix #2)
    """
    text = raw_text.strip()

    # Strip markdown code block
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    # Tìm JSON block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        json_text = text[start:end]
    elif start != -1:
        # JSON bị cắt — thử repair bằng cách đếm ngoặc mở
        json_text = text[start:]
        open_count = json_text.count("{") - json_text.count("}")
        json_text += "}" * open_count  # đóng ngoặc còn thiếu
        # Xoá trailing comma trước ngoặc đóng
        import re as _re
        json_text = _re.sub(r',\s*}', '}', json_text)
        json_text = _re.sub(r',\s*]', ']', json_text)
    else:
        json_text = "{}"

    # Parse JSON
    data = None
    for attempt in [json_text, json_text + "}"]:
        try:
            data = json.loads(attempt)
            break
        except json.JSONDecodeError:
            pass

    if data is None:
        # Final fallback
        logger.warning("Moderator JSON parse failed after repair attempt")
        data = _moderator_fallback(inp)

    # ── Post-process: đảm bảo moderator_summary là TEXT THUẦN (Fix #6) ───────
    summary = data.get("moderator_summary", "")
    if isinstance(summary, str):
        # Nếu summary chứa JSON artifacts → strip chúng
        if "{" in summary or "panel_verdict" in summary:
            # Lấy phần text trước dấu { đầu tiên
            summary = summary[:summary.find("{")].strip()
        # Giới hạn độ dài
        if len(summary) > 500:
            summary = summary[:500]
        data["moderator_summary"] = summary or "Hội đồng đã thảo luận và đưa ra phán quyết dựa trên dữ liệu 16 engines."

    # ── Post-process: đảm bảo confidence hợp lý (Fix #2) ─────────────────────
    # Nếu JSON trả về confidence quá thấp so với mức đồng thuận → điều chỉnh
    # (moderator_prompt đã chỉ định base_conf nhưng LLM có thể ignore)
    raw_conf = float(data.get("panel_confidence", 60))
    data["panel_confidence"] = max(raw_conf, 25.0)  # sàn tuyệt đối 25%

    # ── Post-process: shelf_life tối thiểu 7 ngày ────────────────────────────
    data["shelf_life_days"] = max(7, int(data.get("shelf_life_days", 7)))

    # ── Post-process: đảm bảo S/R khác nhau ít nhất 3% (Fix #3) ─────────────
    price = inp.current_price
    atr   = inp.atr or price * 0.02
    data  = _ensure_sr_levels(data, price, atr, inp)

    # ── Post-process: R:R warning (Fix #4) ───────────────────────────────────
    sc_buy = data.get("scenario_buy", {})
    entry  = float(sc_buy.get("entry", price) or price)
    tp1    = float(sc_buy.get("tp1", sc_buy.get("tp", 0)) or 0)
    sl     = float(sc_buy.get("sl", 0) or 0)
    rr_warning = ""
    if entry > 0 and tp1 > 0 and sl > 0 and entry != sl:
        rr_calc = (tp1 - entry) / abs(entry - sl)
        sc_buy["rr"] = round(rr_calc, 2)
        if rr_calc < 2.0:
            rr_warning = f"R:R = {rr_calc:.1f} dưới ngưỡng 1:2 — confidence giảm tự động"
            data["panel_confidence"] = max(25, data["panel_confidence"] - 10)
    data["rr_warning"] = rr_warning or data.get("rr_warning", "")
    data["scenario_buy"] = sc_buy

    return data


def _ensure_sr_levels(data: dict, price: float, atr: float, inp: SwarmInput) -> dict:
    """Đảm bảo support/resistance levels có đủ 3 mức, khác nhau ít nhất 3%."""
    # Tính các mức mặc định
    s_base = inp.support    or round(price * 0.97, 0)
    r_base = inp.resistance or round(price * 1.03, 0)

    def _dedupe_levels(levels: list, base: float, direction: str) -> list:
        """Loại bỏ duplicate, đảm bảo cách nhau ít nhất 3%."""
        result = []
        prev   = None
        for lv in sorted(levels, key=lambda x: x.get("price", 0),
                         reverse=(direction == "resist")):
            p = float(lv.get("price", 0))
            if p <= 0:
                continue
            if prev is None or abs(p - prev) / prev >= 0.025:
                result.append(lv)
                prev = p
            if len(result) >= 3:
                break
        # Pad nếu thiếu
        while len(result) < 3:
            if direction == "support":
                if result:
                    new_p = round(result[-1]["price"] * 0.965, 0)
                else:
                    new_p = round(s_base, 0)
                reasons = ["SMA20 + tích lũy cũ", "Fibonacci 38.2% + Order Block", "SMA50 + đáy sóng Elliott"]
                result.append({"price": new_p, "reason": reasons[len(result)]})
            else:
                if result:
                    new_p = round(result[-1]["price"] * 1.035, 0)
                else:
                    new_p = round(r_base, 0)
                reasons = ["Kháng cự kỹ thuật gần + Ichimoku Kumo", "Fibonacci 61.8% + đỉnh cũ", "Kháng cự mạnh + vùng tâm lý"]
                result.append({"price": new_p, "reason": reasons[min(len(result), 2)]})
        return result[:3]

    sup_raw = data.get("support_levels", [])
    res_raw = data.get("resistance_levels", [])

    # Nếu chưa có → tạo từ ATR
    if not sup_raw:
        sup_raw = [
            {"price": round(s_base, 0),         "reason": "Hỗ trợ kỹ thuật chính (SMA20)"},
            {"price": round(s_base * 0.965, 0),  "reason": "Fibonacci 38.2% + Order Block"},
            {"price": round(s_base * 0.932, 0),  "reason": "SMA50 + đáy Elliott Wave"},
        ]
    if not res_raw:
        res_raw = [
            {"price": round(r_base, 0),         "reason": "Kháng cự gần + Ichimoku Kumo top"},
            {"price": round(r_base * 1.035, 0), "reason": "Fibonacci 61.8% + đỉnh cũ"},
            {"price": round(r_base * 1.072, 0), "reason": "Kháng cự mạnh + vùng tâm lý"},
        ]

    data["support_levels"]    = _dedupe_levels(sup_raw,  price, "support")
    data["resistance_levels"] = _dedupe_levels(res_raw, price, "resist")
    return data


def _moderator_fallback(inp: SwarmInput) -> dict:
    """Fallback hoàn chỉnh khi JSON parse thất bại."""
    price  = inp.current_price
    atr    = inp.atr or price * 0.02
    s1     = inp.support    or round(price * 0.97, 0)
    r1     = inp.resistance or round(price * 1.03, 0)
    tp_est = inp.tp or round(r1, 0)
    sl_est = inp.sl or s1
    rr_est = round((tp_est - price) / (price - sl_est), 2) if (price - sl_est) > 0 else 2.0

    return {
        "panel_verdict":    inp.verdict_label.split()[0] if inp.verdict_label else "THEO DOI",
        "panel_confidence": max(float(inp.confidence_pct), 40.0),
        "rr_warning":       "" if rr_est >= 2.0 else f"R:R={rr_est:.1f} dưới ngưỡng 1:2",
        "consensus_level":  "PHAN BIEN",
        "scenario_buy": {
            "probability": max(inp.bull_count * 12, 35),
            "entry":  round(price, 0) if inp.verdict_label not in ("TRUNG LAP", "THEO DOI") else 0,
            "tp1":    round(tp_est, 0), "tp2": round(tp_est * 1.04, 0),
            "sl":     round(sl_est, 0), "rr": rr_est,
            "trigger": f"MUA khi giá close trên {r1:,.0f} với volume > 1.5x TB20 và RSI < 65",
            "catalyst": f"Breakout {r1:,.0f} với volume xác nhận và MACD histogram dương",
        },
        "scenario_sell": {
            "probability": max(inp.bear_count * 12, 25),
            "entry":  0,
            "tp":     round(sl_est * 0.97, 0),
            "sl":     round(r1 * 1.02, 0), "rr": 2.0,
            "trigger": f"BÁN khi giá phá vỡ {s1:,.0f} với volume > 2x TB20 và RSI < 45",
            "catalyst": f"Breakdown {s1:,.0f} kéo theo SMA20 và MACD cắt xuống",
        },
        "scenario_watch": {
            "probability": 20,
            "trigger": f"Theo dõi vùng {round(price*0.99,0):,.0f}–{round(price*1.01,0):,.0f}",
            "watch_for": "Volume xác nhận và candlestick pattern rõ ràng",
        },
        "support_levels": [
            {"price": round(s1, 0),         "reason": "Hỗ trợ kỹ thuật chính (SMA20 + tích lũy)"},
            {"price": round(s1 * 0.965, 0), "reason": "Fibonacci 38.2% + Order Block"},
            {"price": round(s1 * 0.932, 0), "reason": "SMA50 + đáy Elliott Wave"},
        ],
        "resistance_levels": [
            {"price": round(r1, 0),          "reason": "Kháng cự gần nhất + Ichimoku Kumo"},
            {"price": round(r1 * 1.035, 0),  "reason": "Fibonacci 61.8% + đỉnh cũ"},
            {"price": round(r1 * 1.072, 0),  "reason": "Kháng cự mạnh + vùng tâm lý"},
        ],
        "main_risks":    [
            f"Giá phá vỡ hỗ trợ {s1:,.0f} với volume lớn → nguy cơ giảm tiếp",
            f"RSI hiện tại {inp.rsi:.0f} — cần theo dõi tín hiệu divergence",
            f"Volume ratio {inp.volume_ratio:.1f}x TB20 — xác nhận chưa đủ mạnh",
        ],
        "key_catalysts": [
            f"Kết quả kinh doanh {inp.symbol} — catalyst nội tại quan trọng nhất",
            f"Market regime {inp.market_regime} — dòng tiền ngành tác động trực tiếp",
        ],
        "shelf_life_days": 7,
        "moderator_summary": (
            f"Hội đồng đồng thuận về {inp.verdict_label} với RSI={inp.rsi:.0f} "
            f"và volume ratio {inp.volume_ratio:.1f}x TB20. "
            f"Bull signals: {inp.bull_count}/{inp.active_agents}, "
            f"bear signals: {inp.bear_count}/{inp.active_agents}. "
            f"Market regime: {inp.market_regime}."
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
        """Chạy toàn bộ swarm và trả về SwarmReport."""
        t0  = time.time()
        ctx = _build_context_block(inp)

        # ── VÒNG 1: Ý kiến độc lập ──────────────────────────────────────────
        self._progress(f"Vòng 1/2 — {len(EXPERTS)} chuyên gia đang phân tích...")
        r1_opinions: list[ExpertOpinion] = []
        for expert in EXPERTS:
            self._progress(f"  {expert['emoji']} {expert['role']} đang phân tích...")
            try:
                sys_p, usr_p = _build_expert_prompt_round1(expert, ctx)
                raw = self.llm.chat(sys_p, usr_p, LLM_MAX_TOKENS)
                opinion = _parse_expert_response(expert, raw, 1)
                r1_opinions.append(opinion)
                self._progress(
                    f"  → {expert['emoji']} {expert['role']}: "
                    f"{opinion.stance} ({opinion.confidence}%)"
                )
            except Exception as e:
                logger.warning(f"Expert {expert['id']} R1 fail: {e}")
                r1_opinions.append(ExpertOpinion(
                    expert_id=expert["id"], role=expert["role"],
                    stance="THEO DOI", confidence=50,
                    key_points=["Lỗi khi lấy ý kiến"], concern="N/A", raw_text=str(e),
                ))

        # ── VÒNG 2: Phản biện ───────────────────────────────────────────────
        self._progress("Vòng 2/2 — Tranh luận và phản biện...")
        r2_opinions: list[ExpertOpinion] = []
        debate_round = DebateRound(round_num=2, exchanges=[])

        for expert in EXPERTS:
            own_r1 = next((op for op in r1_opinions if op.expert_id == expert["id"]),
                         r1_opinions[0])
            self._progress(f"  {expert['emoji']} {expert['role']} đang phản biện...")
            try:
                sys_p, usr_p = _build_expert_prompt_round2(
                    expert, ctx, r1_opinions, own_r1
                )
                raw = self.llm.chat(sys_p, usr_p, LLM_MAX_TOKENS)
                opinion = _parse_expert_response(expert, raw, 2)
                r2_opinions.append(opinion)
                debate_round.exchanges.append({
                    "expert":   expert["id"],
                    "role":     expert["role"],
                    "stance_r1": own_r1.stance,
                    "stance_r2": opinion.stance,
                    "changed":  own_r1.stance != opinion.stance,
                    "text":     raw[:300],
                })
                self._progress(
                    f"  → {expert['emoji']}: "
                    f"R1={own_r1.stance} → R2={opinion.stance}"
                    + (" ⚡ THAY ĐỔI!" if own_r1.stance != opinion.stance else "")
                )
            except Exception as e:
                logger.warning(f"Expert {expert['id']} R2 fail: {e}")
                r2_opinions.append(own_r1)

        # ── MODERATOR TỔNG HỢP ───────────────────────────────────────────────
        self._progress("Moderator đang tổng hợp kết luận...")
        try:
            sys_p, usr_p = _build_moderator_prompt(inp, ctx, r1_opinions, r2_opinions)
            raw_mod = self.llm.chat(sys_p, usr_p, MODERATOR_TOKENS)
            mod_data = _parse_moderator_json(raw_mod, inp)
        except Exception as e:
            logger.warning(f"Moderator fail: {e}")
            mod_data = _parse_moderator_json("", inp)

        # ── BUILD REPORT ─────────────────────────────────────────────────────
        elapsed = round(time.time() - t0, 1)

        verdict    = mod_data.get("panel_verdict", "THEO DOI")
        shelf_life = max(7, int(mod_data.get("shelf_life_days", 7)))
        review_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        expires_at  = (datetime.now() + timedelta(days=shelf_life)).strftime("%Y-%m-%d")

        # ── Fix #2: Confidence cuối = max(mod_data, tính từ đồng thuận) ─────
        stances_r2  = [op.stance for op in r2_opinions]
        dominant_st = max(set(stances_r2), key=stances_r2.count) if stances_r2 else "THEO DOI"
        dom_count   = stances_r2.count(dominant_st)
        n_exp       = len(stances_r2) or 5
        avg_conf_r2 = sum(op.confidence for op in r2_opinions) / n_exp
        consensus_floor = max(40, round((dom_count / n_exp) * avg_conf_r2))
        if dom_count == n_exp:      # đồng thuận tuyệt đối
            consensus_floor = max(consensus_floor, 65)
        final_confidence = max(float(mod_data.get("panel_confidence", 60)), consensus_floor)

        # ── Consensus label ───────────────────────────────────────────────────
        if dom_count == n_exp:
            consensus_level = "DONG THUAN"
        elif dom_count >= n_exp - 1:
            consensus_level = "PHAN BIEN"
        else:
            consensus_level = "CHIA RE"
        # dùng moderator nếu có, override nếu sai
        mod_consensus = mod_data.get("consensus_level", consensus_level)
        if dom_count == n_exp and mod_consensus != "DONG THUAN":
            mod_consensus = "DONG THUAN"

        input_summary = (
            f"{inp.symbol} | {inp.verdict_label} ({inp.confidence_pct:.0f}%) | "
            f"Bull={inp.bull_count} Bear={inp.bear_count} | "
            f"RSI={inp.rsi:.1f} | Vol={inp.volume_ratio:.1f}x"
        )

        self._progress("✅ Hoàn tất! Đang định dạng báo cáo...")

        return SwarmReport(
            symbol             = inp.symbol,
            timestamp          = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_s          = elapsed,
            llm_provider       = self.llm.provider,
            llm_model          = self.llm.model,
            panel_verdict      = verdict,
            panel_confidence   = final_confidence,
            consensus_level    = mod_consensus,
            scenario_buy       = mod_data.get("scenario_buy", {}),
            scenario_sell      = mod_data.get("scenario_sell", {}),
            scenario_watch     = mod_data.get("scenario_watch", {}),
            main_risks         = mod_data.get("main_risks", []),
            key_catalysts      = mod_data.get("key_catalysts", []),
            shelf_life_days    = shelf_life,
            expires_at         = expires_at,
            review_date        = review_date,
            support_levels     = mod_data.get("support_levels", []),
            resistance_levels  = mod_data.get("resistance_levels", []),
            rr_warning         = mod_data.get("rr_warning", ""),
            expert_opinions    = r2_opinions,
            debate_rounds      = [debate_round],
            moderator_summary  = mod_data.get("moderator_summary", ""),
            moderator_raw_json = mod_data,   # chỉ lưu DB, không hiển thị user
            input_summary      = input_summary,
        )


# ══════════════════════════════════════════════════════════════════════════════
# REPORT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def format_swarm_report(report: SwarmReport) -> str:
    """
    Định dạng SwarmReport thành text đẹp để gửi Telegram.
    JSON raw KHÔNG được hiển thị (Fix #6) — chỉ dùng để lưu DB.
    """
    SEP  = "═" * 38
    SEP2 = "─" * 38

    verdict_emoji = {"MUA": "🟢", "BAN": "🔴", "THEO DOI": "🟡"}.get(report.panel_verdict, "🟡")
    consensus_emoji = {"DONG THUAN": "✅", "PHAN BIEN": "⚡", "CHIA RE": "❌"}.get(report.consensus_level, "⚡")

    # ── Header ────────────────────────────────────────────────────────────────
    lines = [
        f"LOCAL SWARM PANEL: {report.symbol}",
        SEP,
        f"🤖 {report.llm_provider}/{report.llm_model} | ⏱️ {report.elapsed_s:.0f}s",
        f"📥 {report.input_summary}",
        SEP,
        "",
        f"PHÁN QUYẾT HỘI ĐỒNG: {verdict_emoji} {report.panel_verdict}",
        f"Độ tin cậy   : {report.panel_confidence:.0f}%",
        f"Đồng thuận   : {consensus_emoji} {report.consensus_level}",
    ]

    # R:R warning
    rr_warn = getattr(report, "rr_warning", "")
    if rr_warn:
        lines.append(f"⚠️ {rr_warn}")

    lines.append("")

    # ── Expert opinions ───────────────────────────────────────────────────────
    lines += [SEP2, "CHUYÊN GIA (sau 2 vòng):", SEP2]
    for op in report.expert_opinions:
        em = next((e["emoji"] for e in EXPERTS if e["id"] == op.expert_id), "👤")
        s_em = {"MUA": "🟢", "BAN": "🔴", "THEO DOI": "🟡"}.get(op.stance, "🟡")
        lines.append(f"{em} {op.role}: {s_em} {op.stance} ({op.confidence}%)")
        if op.key_points:
            lines.append(f"   → {op.key_points[0][:90]}")
    lines.append("")

    # Debate changes
    if report.debate_rounds:
        rd = report.debate_rounds[0]
        changed = [x for x in rd.exchanges if x.get("changed")]
        if changed:
            lines.append(f"⚡ ĐẢO Ý KIẾN: {len(changed)} chuyên gia thay đổi stance")
            for x in changed:
                lines.append(f"  • {x['role']}: {x['stance_r1']} → {x['stance_r2']}")
            lines.append("")

    # ── Hỗ trợ / Kháng cự ────────────────────────────────────────────────────
    sup_levels = getattr(report, "support_levels", []) or []
    res_levels = getattr(report, "resistance_levels", []) or []
    if sup_levels or res_levels:
        lines += [SEP2, "MỨC HỖ TRỢ / KHÁNG CỰ:", SEP2]
        if sup_levels:
            for i, s in enumerate(sup_levels[:3], 1):
                if isinstance(s, dict):
                    lines.append(f"  🔵 S{i}: {s.get('price',0):,.0f}  — {s.get('reason','')}")
                else:
                    lines.append(f"  🔵 S{i}: {float(s):,.0f}")
        if res_levels:
            for i, r in enumerate(res_levels[:3], 1):
                if isinstance(r, dict):
                    lines.append(f"  🔴 R{i}: {r.get('price',0):,.0f}  — {r.get('reason','')}")
                else:
                    lines.append(f"  🔴 R{i}: {float(r):,.0f}")
        lines.append("")

    # ── Kịch bản đầu tư ──────────────────────────────────────────────────────
    lines += [SEP2, "KỊCH BẢN ĐẦU TƯ:", SEP2]

    # Sắp xếp theo probability giảm dần — kịch bản ưu tiên lên đầu
    all_scenarios = [
        ("MUA",      report.scenario_buy   or {}, "🟢"),
        ("BAN",      report.scenario_sell  or {}, "🔴"),
        ("THEO DOI", report.scenario_watch or {}, "🟡"),
    ]
    all_scenarios.sort(key=lambda x: x[1].get("probability", 0), reverse=True)

    for idx, (sc_type, sc, sc_em) in enumerate(all_scenarios):
        if not sc:
            continue
        prob    = sc.get("probability", 0)
        priority = "ƯU TIÊN" if idx == 0 else "DỰ PHÒNG"
        lines.append(f"{sc_em} KỊCH BẢN {priority} — {sc_type} ({prob}%):")

        entry = float(sc.get("entry", 0) or 0)
        tp1   = float(sc.get("tp1", sc.get("tp", 0)) or 0)
        tp2   = float(sc.get("tp2", 0) or 0)
        sl    = float(sc.get("sl", 0) or 0)
        rr    = float(sc.get("rr", 0) or 0)

        # Entry — Fix #4: THEO DOI không có entry tại thị trường
        if sc_type in ("MUA", "BAN") and entry > 0:
            lines.append(f"  Entry    : {entry:,.0f}")
        elif sc_type == "THEO DOI":
            lines.append(f"  Entry    : — (chờ trigger kích hoạt)")

        # TP / SL / R:R
        if tp1 > 0:
            lines.append(f"  TP1/TP2  : {tp1:,.0f}" + (f" / {tp2:,.0f}" if tp2 > 0 else ""))
        if sl > 0:
            lines.append(f"  Stop Loss: {sl:,.0f}")
        if rr > 0:
            rr_flag = " ⚠️ <1:2" if rr < 2.0 else ""
            lines.append(f"  R:R      : {rr:.1f}{rr_flag}")

        # Trigger (Fix #4)
        trigger = sc.get("trigger", sc.get("condition", ""))
        if trigger:
            lines.append(f"  🎯 Trigger: {trigger[:120]}")

        # Catalyst (Fix #5 — gắn với mã)
        cat = sc.get("catalyst", sc.get("watch_for", ""))
        if cat:
            lines.append(f"  Catalyst : {cat[:100]}")
        lines.append("")

    # ── Rủi ro (Fix #5 — cụ thể, gắn với mã) ────────────────────────────────
    risks = report.main_risks or []
    if risks:
        lines += [SEP2, "RỦI RO CHÍNH:", SEP2]
        for i, risk in enumerate(risks[:4], 1):
            lines.append(f"  {i}. {risk[:110]}")
        lines.append("")

    # ── Catalyst (Fix #5) ─────────────────────────────────────────────────────
    catalysts = report.key_catalysts or []
    if catalysts:
        lines += ["CATALYST TĂNG:"]
        for cat in catalysts[:3]:
            lines.append(f"  + {cat[:100]}")
        lines.append("")

    # ── Signal shelf life ─────────────────────────────────────────────────────
    review_date = getattr(report, "review_date", report.expires_at)
    lines += [
        SEP2,
        f"HẠN TÍN HIỆU  : {report.shelf_life_days} ngày",
        f"Đánh giá lại  : {review_date}",
        f"Hết hạn       : {report.expires_at}",
        SEP2, "",
        "TỔNG HỢP CUỘC HỌP:", "",
    ]

    # Moderator summary — TEXT THUẦN, không JSON (Fix #6)
    summary = report.moderator_summary or ""
    # Bảo vệ: nếu summary vẫn chứa dấu hiệu JSON → hiển thị message thay thế
    if "{" in summary and "panel_verdict" in summary:
        summary = "Hội đồng đã hoàn thành phân tích. Xem chi tiết ở các phần trên."
    if summary:
        for para in summary.split("\n"):
            para = para.strip()
            if para:
                lines.append(textwrap.fill(para, width=52))
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
