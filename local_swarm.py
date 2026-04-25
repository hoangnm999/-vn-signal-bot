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
LLM_TIMEOUT      = 60       # giây timeout per LLM call
LLM_MAX_TOKENS   = 800      # token tối đa mỗi response chuyên gia
MODERATOR_TOKENS = 1200     # token tổng hợp của moderator
SWARM_VERSION    = "1.0"

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
    scenario_buy:    dict
    scenario_sell:   dict
    scenario_watch:  dict
    scenario_primary:   dict   # kịch bản ưu tiên (probability cao nhất)
    scenario_secondary: dict   # kịch bản dự phòng

    # Hỗ trợ / Kháng cự chi tiết
    support_levels:    list    # [{price, reason}, ...]
    resistance_levels: list    # [{price, reason}, ...]

    # R:R warning
    rr_warning:      str

    # Rủi ro
    main_risks:      list[str]
    key_catalysts:   list[str]

    # Signal shelf life
    shelf_life_days: int
    expires_at:      str
    review_date:     str     # ngày hiện tại + 7 ngày — "đánh giá lại"

    # Expert opinions
    expert_opinions: list[ExpertOpinion]
    debate_rounds:   list[DebateRound]

    # Moderator summary
    moderator_summary: str

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
        f"Bạn là {expert['role']} trong một Hội đồng Chuyên gia Phân tích Cổ phiếu VN.\n"
        f"{expert['focus']}\n\n"
        "NGUYÊN TẮC BẮT BUỘC:\n"
        "1. Nếu stance là THEO DOI: KHÔNG được đưa Entry giá hiện tại. "
        "Thay vào đó BẮT BUỘC phải có TRIGGER rõ ràng (giá cụ thể + điều kiện volume/kỹ thuật).\n"
        "2. Entry chỉ được đặt khi stance là MUA hoặc BAN.\n"
        "3. Phải liệt kê ít nhất 3 mức hỗ trợ và 3 mức kháng cự với lý do trường phái xác nhận.\n"
        "4. R:R phải >= 1:2. Nếu R:R < 1:2 thì hạ stance xuống THEO DOI và ghi rõ lý do.\n"
        "5. Trigger phải gồm: mức giá cụ thể + điều kiện volume + điều kiện kỹ thuật bổ sung.\n\n"
        "ĐỊNH DẠNG TRẢ LỜI (bắt buộc):\n"
        "STANCE: [MUA/BAN/THEO DOI]\n"
        "CONFIDENCE: [0-100]\n"
        "ĐIỂM CHÍNH:\n"
        "- Điểm 1\n"
        "- Điểm 2\n"
        "- Điểm 3\n"
        "HỖ TRỢ:\n"
        "- S1: [giá] — [lý do: Fibonacci/SMA/SMC/Elliott/...]\n"
        "- S2: [giá] — [lý do]\n"
        "- S3: [giá] — [lý do]\n"
        "KHÁNG CỰ:\n"
        "- R1: [giá] — [lý do]\n"
        "- R2: [giá] — [lý do]\n"
        "- R3: [giá] — [lý do]\n"
        "TRIGGER: [Điều kiện kích hoạt: 'MUA/BAN khi giá [X] với volume > [Y]x TB20 và [điều kiện kỹ thuật]']\n"
        "  (Nếu THEO DOI: 'Chờ khi giá [X] VÀ volume [Y] VÀ [điều kiện kỹ thuật]')\n"
        "RR_CHECK: [Tính R:R = (TP-Entry)/(Entry-SL). Nếu < 2.0 ghi 'R:R DƯỚI NGƯỠNG — hạ confidence']\n"
        "RỦI RO CHÍNH: [1-2 câu về rủi ro lớn nhất]\n"
        "LUẬN ĐIỂM: [Phân tích chi tiết 100-150 từ, BẰNG TIẾNG VIỆT]"
    )
    user = (
        f"Dưới đây là kết quả phân tích 16 engines cho {ctx.split()[4]}. "
        f"Hãy đưa ra ý kiến ĐỘC LẬP của bạn:\n\n{ctx}\n\n"
        "Lưu ý: Nếu R:R < 1:2, BẮT BUỘC hạ stance xuống THEO DOI. "
        "Nếu stance THEO DOI, KHÔNG đặt entry ở giá hiện tại mà phải có trigger rõ ràng."
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
        "Bạn đã nghe ý kiến của các chuyên gia khác. "
        "Hãy:\n"
        "1. Bảo vệ hoặc điều chỉnh stance của bạn dựa trên lập luận của họ\n"
        "2. Phản biện điểm yếu nhất trong ý kiến đối lập\n"
        "3. Chỉ ra 1 điều bạn ĐỒNG Ý với expert khác\n\n"
        "NGUYÊN TẮC GIỮ NGUYÊN:\n"
        "- Nếu THEO DOI: vẫn PHẢI có trigger rõ ràng, KHÔNG entry giá hiện tại\n"
        "- Nếu R:R < 1:2 trong kịch bản của bạn: hạ confidence thêm 15 điểm\n"
        "- Nếu chuyên gia Risk Manager chỉ ra R:R thấp và bạn không phản bác được: hạ stance\n\n"
        "ĐỊNH DẠNG TRẢ LỜI:\n"
        "STANCE: [MUA/BAN/THEO DOI — có thể giữ nguyên hoặc thay đổi]\n"
        "CONFIDENCE: [0-100]\n"
        "PHẢN BIỆN: [Nhắm vào expert nào, điểm nào sai]\n"
        "ĐỒNG Ý VỚI: [Expert nào, điểm nào]\n"
        "TRIGGER CẬP NHẬT: [Điều kiện kích hoạt sau khi cân nhắc ý kiến các bên]\n"
        "KẾT LUẬN CỦA TÔI: [50-80 từ, bằng tiếng Việt]"
    )

    user = (
        f"Dữ liệu phân tích: {ctx[:500]}...\n\n"
        f"Ý kiến của bạn ở vòng 1:\n"
        f"STANCE={own_round1.stance}, CONFIDENCE={own_round1.confidence}\n"
        f"Điểm chính: {'; '.join(own_round1.key_points)}\n\n"
        f"Ý kiến các chuyên gia khác:\n"
        + "\n".join(others_summary)
        + "\n\nHãy phản biện và bảo vệ quan điểm của bạn."
    )
    return system, user


def _build_moderator_prompt(
    inp: SwarmInput,
    ctx: str,
    r1_opinions: list[ExpertOpinion],
    r2_opinions: list[ExpertOpinion],
) -> tuple[str, str]:
    """Prompt cho moderator tổng hợp kết luận cuối."""
    # Tổng hợp stance sau 2 vòng
    stances = {}
    for op in r2_opinions:
        stances[op.expert_id] = op.stance

    buy_count   = sum(1 for s in stances.values() if s == "MUA")
    sell_count  = sum(1 for s in stances.values() if s == "BAN")
    watch_count = sum(1 for s in stances.values() if s == "THEO DOI")

    price  = inp.current_price
    atr    = inp.atr or price * 0.02
    tp_est = inp.tp or round(price + 3 * atr, 0)
    sl_est = inp.sl or round(price - 1.5 * atr, 0)

    opinions_text = ""
    for op in r2_opinions:
        expert_name = next((e["emoji"] + " " + e["role"] for e in EXPERTS if e["id"] == op.expert_id), op.expert_id)
        opinions_text += (
            f"\n{expert_name}:\n"
            f"  Stance cuối: {op.stance} ({op.confidence}%)\n"
            f"  Điểm chính: {'; '.join(op.key_points[:2])}\n"
            f"  Rủi ro: {op.concern}\n"
        )

    system = (
        "Bạn là NGƯỜI ĐIỀU PHỐI (Moderator) của Hội đồng Chuyên gia Phân tích Cổ phiếu VN.\n"
        "Nhiệm vụ: Tổng hợp ý kiến, chỉ ra đồng thuận/bất đồng, "
        "và đưa ra kết luận CUỐI CÙNG dưới dạng JSON chuẩn.\n\n"
        "QUY TẮC BẮT BUỘC KHI TẠO JSON:\n"
        "1. Nếu panel_verdict = 'THEO DOI': trường 'entry' trong scenario_buy/sell PHẢI là 0 "
        "và trường 'trigger' PHẢI có điều kiện đầy đủ (giá + volume + kỹ thuật).\n"
        "2. Tính R:R = (tp1 - entry) / (entry - sl). Nếu R:R < 2.0: "
        "tự động giảm panel_confidence 15 điểm và thêm cảnh báo vào rr_warning.\n"
        "3. shelf_life_days tối thiểu 7 ngày cho khung Daily.\n"
        "4. Kịch bản ưu tiên (probability cao nhất) luôn để lên đầu.\n"
        "5. Mỗi kịch bản PHẢI có trigger đầy đủ: giá + volume + điều kiện kỹ thuật.\n"
        "6. Liệt kê đủ 3 mức hỗ trợ và 3 mức kháng cự với lý do rõ ràng.\n\n"
        "ĐỊNH DẠNG OUTPUT (JSON thuần, không markdown):\n"
        "{\n"
        '  "panel_verdict": "MUA|BAN|THEO DOI",\n'
        '  "panel_confidence": 75,\n'
        '  "rr_warning": "",\n'
        '  "consensus_level": "DONG THUAN|PHAN BIEN|CHIA RE",\n'
        '  "scenario_primary": {\n'
        '    "type": "MUA|BAN|THEO DOI",\n'
        '    "probability": 60,\n'
        '    "entry": 62000,\n'
        '    "tp1": 65000, "tp2": 68000,\n'
        '    "sl": 60000,\n'
        '    "rr": 2.1,\n'
        '    "trigger": "MUA khi giá close trên 62,500 với volume > 1.5x TB20 và MACD histogram chuyển dương",\n'
        '    "catalyst": "Catalyst thúc đẩy kịch bản này",\n'
        '    "fallback_condition": "Nếu giá phá vỡ SL 60,000 thì chuyển sang kịch bản dự phòng"\n'
        "  },\n"
        '  "scenario_secondary": {\n'
        '    "type": "MUA|BAN|THEO DOI",\n'
        '    "probability": 25,\n'
        '    "entry": 0,\n'
        '    "trigger": "Điều kiện kích hoạt kịch bản dự phòng",\n'
        '    "watch_for": "Tín hiệu cần theo dõi để chuyển kịch bản"\n'
        "  },\n"
        '  "scenario_buy": {\n'
        '    "probability": 60, "entry": 62000,\n'
        '    "tp1": 65000, "tp2": 68000, "sl": 60000, "rr": 2.1,\n'
        '    "condition": "Điều kiện vào lệnh MUA",\n'
        '    "trigger": "Trigger đầy đủ: giá + volume + kỹ thuật",\n'
        '    "catalyst": "Catalyst tăng giá"\n'
        "  },\n"
        '  "scenario_sell": {\n'
        '    "probability": 25, "entry": 62500,\n'
        '    "tp": 59000, "sl": 64000, "rr": 1.8,\n'
        '    "condition": "Điều kiện BÁN/TRÁNH",\n'
        '    "trigger": "Trigger đầy đủ: giá + volume + kỹ thuật",\n'
        '    "catalyst": "Trigger giảm giá"\n'
        "  },\n"
        '  "scenario_watch": {\n'
        '    "probability": 15,\n'
        '    "condition": "Khi nào thì THEO DÕI thêm",\n'
        '    "trigger": "Điều kiện chuyển sang hành động",\n'
        '    "watch_for": "Tín hiệu cần chờ đợi"\n'
        "  },\n"
        '  "support_levels": [\n'
        '    {"price": 60000, "reason": "SMA20 + vùng tích lũy cũ"},\n'
        '    {"price": 58000, "reason": "Fibonacci 38.2% + Order Block SMC"},\n'
        '    {"price": 55000, "reason": "SMA50 + đáy Elliott Wave"}\n'
        "  ],\n"
        '  "resistance_levels": [\n'
        '    {"price": 65000, "reason": "Đỉnh cũ tháng 3 + Fibonacci 61.8%"},\n'
        '    {"price": 68000, "reason": "Vùng kháng cự mạnh Ichimoku Kumo top"},\n'
        '    {"price": 72000, "reason": "All-time high vùng + kháng cự tâm lý"}\n'
        "  ],\n"
        '  "main_risks": ["Rủi ro 1", "Rủi ro 2", "Rủi ro 3"],\n'
        '  "key_catalysts": ["Catalyst 1", "Catalyst 2"],\n'
        '  "shelf_life_days": 7,\n'
        '  "moderator_summary": "Tóm tắt 100-150 từ bằng tiếng Việt: đồng thuận là gì, '
        'bất đồng chính ở đâu, kịch bản ưu tiên và lý do verdict cuối",\n'
        '  "expert_alignment": {\n'
        '    "tech_analyst": "AGREE|DISAGREE",\n'
        '    "macro_strategist": "AGREE|DISAGREE",\n'
        '    "risk_manager": "AGREE|DISAGREE",\n'
        '    "smc_trader": "AGREE|DISAGREE",\n'
        '    "fundamental_filter": "AGREE|DISAGREE"\n'
        "  }\n"
        "}"
    )

    user = (
        f"Dữ liệu gốc:\n{ctx}\n\n"
        f"Kết quả vote sau 2 vòng tranh luận:\n"
        f"MUA: {buy_count} | BAN: {sell_count} | THEO DOI: {watch_count}\n\n"
        f"Ý kiến chuyên gia sau vòng 2:\n{opinions_text}\n\n"
        f"Thông tin hỗ trợ tính scenario:\n"
        f"Giá hiện tại: {price:,.0f} | ATR: {atr:,.0f} | "
        f"TP gợi ý: {tp_est:,.0f} | SL gợi ý: {sl_est:,.0f}\n\n"
        "Hãy tổng hợp và trả về JSON thuần (không dùng markdown code block)."
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
    """Parse JSON từ moderator, với fallback nếu LLM không theo format."""
    # Strip markdown code block nếu có
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    # Tìm JSON block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: parse thủ công một số fields
        logger.warning("Moderator JSON parse failed, using fallback")
        price = inp.current_price
        atr   = inp.atr or price * 0.02
        rr_est = round((3 * atr) / (1.5 * atr), 2) if atr > 0 else 2.0
        return {
            "panel_verdict":    inp.verdict_label.split()[0] if inp.verdict_label else "THEO DOI",
            "panel_confidence": inp.confidence_pct,
            "rr_warning":       "" if rr_est >= 2.0 else "R:R dưới ngưỡng khuyến nghị — confidence đã giảm",
            "consensus_level":  "PHAN BIEN",
            "scenario_primary": {
                "type":              "THEO DOI",
                "probability":       max(inp.bull_count * 10, 40),
                "entry":             0,
                "trigger":           f"Chờ khi giá vượt {round(price * 1.02, 0):,.0f} với volume > 1.5x TB20 và RSI < 65",
                "catalyst":          "Breakout kháng cự với xác nhận volume",
                "fallback_condition": f"Nếu giá phá vỡ {round(price * 0.95, 0):,.0f} thì chuyển kịch bản dự phòng giảm",
            },
            "scenario_secondary": {
                "type":       "BAN",
                "probability": max(inp.bear_count * 10, 20),
                "entry":      0,
                "trigger":    f"Khi giá phá hỗ trợ {round(price * 0.95, 0):,.0f} với volume lớn",
                "watch_for":  "Breakdown với volume > 2x TB20",
            },
            "scenario_buy": {
                "probability": max(inp.bull_count * 10, 40),
                "entry":  round(price, 0),
                "tp1":    round(price + 3 * atr, 0),
                "tp2":    round(price + 5 * atr, 0),
                "sl":     round(price - 1.5 * atr, 0),
                "rr":     rr_est,
                "condition": "Giá giữ trên SMA20 với volume tăng",
                "trigger": f"MUA khi giá close trên {round(price * 1.01, 0):,.0f} với volume > 1.5x TB20 và MACD dương",
                "catalyst":  "Breakout kháng cự với xác nhận volume",
            },
            "scenario_sell": {
                "probability": max(inp.bear_count * 10, 30),
                "entry":  round(price, 0),
                "tp":     round(price - 3 * atr, 0),
                "sl":     round(price + 1.5 * atr, 0),
                "rr":     rr_est,
                "condition": "Giá phá vỡ SMA20 với volume lớn",
                "trigger": f"BÁN khi giá phá vỡ {round(price * 0.98, 0):,.0f} với volume > 2x TB20",
                "catalyst":  "Breakdown hỗ trợ chính",
            },
            "scenario_watch": {
                "probability": 20,
                "condition":   "Chờ tín hiệu rõ ràng hơn",
                "trigger":     f"Theo dõi khi giá test vùng {round(price * 0.99, 0):,.0f}–{round(price * 1.01, 0):,.0f}",
                "watch_for":   "Volume xác nhận và price action rõ ràng",
            },
            "support_levels": [
                {"price": round(inp.support or price * 0.95, 0), "reason": "Hỗ trợ kỹ thuật chính (SMA20/tích lũy)"},
                {"price": round((inp.support or price * 0.95) * 0.97, 0), "reason": "Hỗ trợ thứ cấp (Fibonacci 38.2%)"},
                {"price": round(inp.sl or price * 0.93, 0), "reason": "Hỗ trợ mạnh (SMA50/đáy sóng)"},
            ],
            "resistance_levels": [
                {"price": round(inp.resistance or price * 1.05, 0), "reason": "Kháng cự kỹ thuật chính"},
                {"price": round((inp.resistance or price * 1.05) * 1.03, 0), "reason": "Kháng cự thứ cấp (Fibonacci 61.8%)"},
                {"price": round(inp.tp or price * 1.10, 0), "reason": "Kháng cự mạnh (đỉnh cũ/TP mục tiêu)"},
            ],
            "main_risks":    ["Biến động thị trường chung", "Thiếu volume xác nhận", "Rủi ro vĩ mô toàn cầu"],
            "key_catalysts": ["Kết quả kinh doanh", "Dòng tiền khối ngoại"],
            "shelf_life_days": 7,
            "moderator_summary": raw_text[:400] if raw_text else "Không có tóm tắt",
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

        verdict        = mod_data.get("panel_verdict", "THEO DOI")
        shelf_life     = max(7, int(mod_data.get("shelf_life_days", 7)))
        review_date    = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        expires_at     = (datetime.now() + timedelta(days=shelf_life)).strftime("%Y-%m-%d")

        # Input summary
        input_summary = (
            f"{inp.symbol} | {inp.verdict_label} ({inp.confidence_pct:.0f}%) | "
            f"Bull={inp.bull_count} Bear={inp.bear_count} | "
            f"RSI={inp.rsi:.1f} | Vol={inp.volume_ratio:.1f}x"
        )

        self._progress("✅ Hoàn tất! Đang định dạng báo cáo...")

        # Tính R:R warning
        sc_buy = mod_data.get("scenario_buy", {})
        rr_val  = float(sc_buy.get("rr", 0))
        rr_warning = ""
        if 0 < rr_val < 2.0:
            rr_warning = f"R:R = {rr_val:.1f} dưới ngưỡng khuyến nghị 1:2 — confidence tự động giảm"
            # Tự động hạ confidence
            orig_conf = float(mod_data.get("panel_confidence", 60))
            mod_data["panel_confidence"] = max(30, orig_conf - 15)

        return SwarmReport(
            symbol              = inp.symbol,
            timestamp           = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_s           = elapsed,
            llm_provider        = self.llm.provider,
            llm_model           = self.llm.model,
            panel_verdict       = verdict,
            panel_confidence    = float(mod_data.get("panel_confidence", 60)),
            consensus_level     = mod_data.get("consensus_level", "PHAN BIEN"),
            scenario_buy        = mod_data.get("scenario_buy", {}),
            scenario_sell       = mod_data.get("scenario_sell", {}),
            scenario_watch      = mod_data.get("scenario_watch", {}),
            scenario_primary    = mod_data.get("scenario_primary", {}),
            scenario_secondary  = mod_data.get("scenario_secondary", {}),
            support_levels      = mod_data.get("support_levels", []),
            resistance_levels   = mod_data.get("resistance_levels", []),
            rr_warning          = rr_warning or mod_data.get("rr_warning", ""),
            main_risks          = mod_data.get("main_risks", []),
            key_catalysts       = mod_data.get("key_catalysts", []),
            shelf_life_days     = shelf_life,
            expires_at          = expires_at,
            review_date         = review_date,
            expert_opinions     = r2_opinions,
            debate_rounds       = [debate_round],
            moderator_summary   = mod_data.get("moderator_summary", raw_mod[:500] if 'raw_mod' in dir() else ""),
            input_summary       = input_summary,
        )


# ══════════════════════════════════════════════════════════════════════════════
# REPORT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def format_swarm_report(report: SwarmReport) -> str:
    """Định dạng SwarmReport thành text đẹp để gửi Telegram."""
    SEP  = "═" * 38
    SEP2 = "─" * 38

    verdict_emoji = {
        "MUA": "🟢", "BAN": "🔴", "THEO DOI": "🟡"
    }.get(report.panel_verdict, "🟡")

    consensus_emoji = {
        "DONG THUAN": "✅", "PHAN BIEN": "⚡", "CHIA RE": "❌"
    }.get(report.consensus_level, "⚡")

    # ── Header ────────────────────────────────────────────────────────────────
    lines = [
        f"LOCAL SWARM PANEL: {report.symbol}",
        SEP,
        f"🤖 {report.llm_provider}/{report.llm_model} | ⏱️ {report.elapsed_s:.0f}s",
        f"📥 {report.input_summary}",
        SEP,
        "",
        f"PHÁN QUYẾT HỘI ĐỒNG: {verdict_emoji} {report.panel_verdict}",
        f"Độ tin cậy  : {report.panel_confidence:.0f}%",
        f"Đồng thuận  : {consensus_emoji} {report.consensus_level}",
    ]

    # R:R warning nếu có
    rr_warning = getattr(report, "rr_warning", "") or ""
    if not rr_warning:
        # tự tính lại từ scenario
        sc = report.scenario_buy or {}
        rr_val = sc.get("rr", 0)
        if rr_val and float(rr_val) < 2.0:
            rr_warning = f"⚠️ R:R = {rr_val:.1f} — dưới ngưỡng 1:2 khuyến nghị"
    if rr_warning:
        lines.append(f"⚠️ {rr_warning}" if not rr_warning.startswith("⚠️") else rr_warning)

    lines.append("")

    # ── Expert opinions ───────────────────────────────────────────────────────
    lines += [SEP2, "BẢNG CHUYÊN GIA (sau 2 vòng):", SEP2]
    for op in report.expert_opinions:
        em = next((e["emoji"] for e in EXPERTS if e["id"] == op.expert_id), "👤")
        stance_em = {"MUA": "🟢", "BAN": "🔴", "THEO DOI": "🟡"}.get(op.stance, "🟡")
        lines.append(f"{em} {op.role}:")
        lines.append(f"  {stance_em} {op.stance} ({op.confidence}%)")
        if op.key_points:
            lines.append(f"  → {op.key_points[0]}")
        lines.append("")

    # Debate changes
    if report.debate_rounds:
        rd = report.debate_rounds[0]
        changed = [x for x in rd.exchanges if x.get("changed")]
        if changed:
            lines.append(f"⚡ ĐẢO Ý KIẾN SAU TRANH LUẬN: {len(changed)} chuyên gia")
            for x in changed:
                lines.append(f"  • {x['role']}: {x['stance_r1']} → {x['stance_r2']}")
            lines.append("")

    # ── Hỗ trợ / Kháng cự ────────────────────────────────────────────────────
    support_levels    = getattr(report, "support_levels",    None) or []
    resistance_levels = getattr(report, "resistance_levels", None) or []

    # Lấy từ mod_data nếu chưa có trong report
    if not support_levels:
        support_levels    = report.scenario_buy.get("support_levels", [])
    if not resistance_levels:
        resistance_levels = report.scenario_buy.get("resistance_levels", [])

    if support_levels or resistance_levels:
        lines += [SEP2, "BẢNG HỖ TRỢ / KHÁNG CỰ:", SEP2]
        if support_levels:
            lines.append("🔵 HỖ TRỢ:")
            for i, s in enumerate(support_levels[:3], 1):
                if isinstance(s, dict):
                    p = s.get("price", 0)
                    r = s.get("reason", "")
                    lines.append(f"  S{i}: {p:,.0f}  — {r}")
                else:
                    lines.append(f"  S{i}: {s:,.0f}")
        if resistance_levels:
            lines.append("🔴 KHÁNG CỰ:")
            for i, r in enumerate(resistance_levels[:3], 1):
                if isinstance(r, dict):
                    p = r.get("price", 0)
                    rs = r.get("reason", "")
                    lines.append(f"  R{i}: {p:,.0f}  — {rs}")
                else:
                    lines.append(f"  R{i}: {r:,.0f}")
        lines.append("")

    # ── Kịch bản ưu tiên & dự phòng ──────────────────────────────────────────
    lines += [SEP2, "KỊCH BẢN ĐẦU TƯ:", SEP2]

    sc_primary   = getattr(report, "scenario_primary",   None) or {}
    sc_secondary = getattr(report, "scenario_secondary", None) or {}

    def _fmt_scenario_block(sc: dict, label: str, priority: str) -> list[str]:
        """Format một kịch bản (primary hoặc secondary)."""
        if not sc:
            return []
        sc_type = sc.get("type", "THEO DOI")
        sc_em   = {"MUA": "🟢", "BAN": "🔴", "THEO DOI": "🟡"}.get(sc_type, "🟡")
        prob    = sc.get("probability", 0)
        out     = [f"{sc_em} KỊCH BẢN {priority} — {label} ({sc_type}, {prob}%):"]

        entry = sc.get("entry", 0)
        if entry and entry > 0 and sc_type in ("MUA", "BAN"):
            out.append(f"  Entry    : {entry:,.0f}")
            tp1 = sc.get("tp1", sc.get("tp", 0))
            tp2 = sc.get("tp2", 0)
            sl  = sc.get("sl", 0)
            rr  = sc.get("rr", 0)
            if tp1: out.append(f"  TP1/TP2  : {tp1:,.0f}" + (f" / {tp2:,.0f}" if tp2 else ""))
            if sl:  out.append(f"  Stop Loss: {sl:,.0f}")
            if rr:
                rr_f = float(rr)
                rr_warn = " ⚠️ <1:2" if rr_f < 2.0 else ""
                out.append(f"  R:R      : {rr_f:.1f}{rr_warn}")
        elif sc_type == "THEO DOI":
            out.append(f"  Entry    : — (chờ trigger)")

        trigger = sc.get("trigger", sc.get("condition", ""))
        if trigger:
            out.append(f"  🎯 Trigger: {trigger}")

        catalyst = sc.get("catalyst", "")
        if catalyst:
            out.append(f"  Catalyst : {catalyst}")

        fallback = sc.get("fallback_condition", sc.get("watch_for", ""))
        if fallback:
            out.append(f"  ↩️ Dự phòng: {fallback}")

        return out + [""]

    # Render primary + secondary nếu có
    if sc_primary:
        sc_type = sc_primary.get("type", "THEO DOI")
        lines += _fmt_scenario_block(sc_primary, sc_type, "ƯU TIÊN")
    if sc_secondary:
        sc_type2 = sc_secondary.get("type", "THEO DOI")
        lines += _fmt_scenario_block(sc_secondary, sc_type2, "DỰ PHÒNG")

    # Fallback: render scenario_buy/sell/watch cũ nếu không có primary/secondary
    if not sc_primary:
        sc_buy   = report.scenario_buy   or {}
        sc_sell  = report.scenario_sell  or {}
        sc_watch = report.scenario_watch or {}

        # Xác định kịch bản nào là ưu tiên
        scenarios_ranked = sorted(
            [
                ("MUA", sc_buy),
                ("BAN", sc_sell),
                ("THEO DOI", sc_watch),
            ],
            key=lambda x: x[1].get("probability", 0),
            reverse=True,
        )

        for idx, (sc_type, sc) in enumerate(scenarios_ranked):
            if not sc:
                continue
            priority = "ƯU TIÊN" if idx == 0 else "DỰ PHÒNG"
            sc_em    = {"MUA": "🟢", "BAN": "🔴", "THEO DOI": "🟡"}.get(sc_type, "🟡")
            prob     = sc.get("probability", 0)
            lines.append(f"{sc_em} KỊCH BẢN {priority} — {sc_type} ({prob}%):")

            if sc_type in ("MUA", "BAN"):
                entry = sc.get("entry", 0)
                tp1   = sc.get("tp1", sc.get("tp", 0))
                tp2   = sc.get("tp2", 0)
                sl    = sc.get("sl", 0)
                rr    = float(sc.get("rr", 0))
                if entry: lines.append(f"  Entry    : {entry:,.0f}")
                if tp1:   lines.append(f"  TP1/TP2  : {tp1:,.0f}" + (f" / {tp2:,.0f}" if tp2 else ""))
                if sl:    lines.append(f"  Stop Loss: {sl:,.0f}")
                if rr:
                    rr_warn = " ⚠️ <1:2" if rr < 2.0 else ""
                    lines.append(f"  R:R      : {rr:.1f}{rr_warn}")
            else:
                lines.append(f"  Entry    : — (chờ trigger)")

            trigger = sc.get("trigger", sc.get("condition", ""))
            if trigger:
                lines.append(f"  🎯 Trigger: {trigger}")
            catalyst = sc.get("catalyst", sc.get("watch_for", ""))
            if catalyst:
                lines.append(f"  Catalyst : {catalyst}")
            lines.append("")

    # ── Risks & Catalysts ─────────────────────────────────────────────────────
    if report.main_risks:
        lines += [SEP2, "RỦI RO CHÍNH:", SEP2]
        for i, risk in enumerate(report.main_risks[:4], 1):
            lines.append(f"  {i}. {risk}")
        lines.append("")

    if report.key_catalysts:
        lines += ["CATALYST TĂNG:", ""]
        for cat in report.key_catalysts[:3]:
            lines.append(f"  + {cat}")
        lines.append("")

    # ── Signal shelf life ─────────────────────────────────────────────────────
    review_date = getattr(report, "review_date", report.expires_at)
    lines += [
        SEP2,
        f"HẠN TÍN HIỆU   : {report.shelf_life_days} ngày (tối thiểu 7D)",
        f"Đánh giá lại   : {review_date}",
        f"Hết hạn        : {report.expires_at}",
        SEP2, "",
        "TỔNG HỢP CUỘC HỌP:", "",
    ]

    for para in report.moderator_summary.split("\n"):
        if para.strip():
            lines.append(textwrap.fill(para.strip(), width=50))
    lines.append("")

    lines.append(SEP)
    lines.append("⚠️ Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.")
    lines.append(f"📋 Local Swarm v{SWARM_VERSION} | /local_swarm {report.symbol}")

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
