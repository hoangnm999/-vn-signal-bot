"""
auto_context.py — Auto Context engine cho /backtest_rule <MA> (không có tham số rule).

Khi user gọi /backtest_rule VCB (không có entry/exit rule):
  1. Tra DB tìm bối cảnh gần nhất:
     - Ưu tiên /vibe context (có trade_plan) trong 7 ngày.
     - Fallback: /check context (chỉ có state_vector) trong 30 ngày.
  2. Nếu có trade_plan → tạo rule từ entry/stop/target, chạy backtest.
  3. Nếu chỉ có state_vector → tìm analog lịch sử, báo cáo forward returns.
  4. Nếu không có gì → hướng dẫn chạy /check hoặc /vibe trước.

TRADE PLAN PARSING:
  Vibe text thường có dạng:
    "Entry: 62,100 | Stop: 59,000 | Target: 68,500"
    "Điểm mua: 62.1k, Cắt lỗ: 59k, Mục tiêu: 68.5k"
  Parse bằng regex + heuristic, fallback về None nếu không tìm thấy.
"""

from __future__ import annotations

import re
import logging
import json
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Vibe context không được quá 7 ngày
VIBE_CONTEXT_TTL_DAYS = 7
# /check context không được quá 30 ngày
CHECK_CONTEXT_TTL_DAYS = 30


# ══════════════════════════════════════════════════════════════════════════════
# DB CONTEXT LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_auto_context(symbol: str) -> dict:
    """
    Tải bối cảnh gần nhất của symbol từ DB.

    Returns dict:
    {
        "found":       bool,
        "source":      "vibe" | "check" | None,
        "created_at":  datetime | None,
        "verdict":     str | None,
        "confidence":  float | None,
        "trade_plan":  dict | None,   # {entry, stop, target, strategy}
        "state_vector": dict | None,
        "raw_summary": str | None,
    }
    """
    symbol = symbol.upper()
    result = {
        "found":        False,
        "source":       None,
        "created_at":   None,
        "verdict":      None,
        "confidence":   None,
        "trade_plan":   None,
        "state_vector": None,
        "raw_summary":  None,
    }

    try:
        from db import get_conn  # type: ignore
        conn = get_conn()
        cur  = conn.cursor()

        # Tìm record mới nhất trong 30 ngày, ưu tiên có trade_plan
        sql = """
            SELECT
                verdict_label,
                confidence_pct,
                summary,
                created_at,
                state_vector,
                trade_plan
            FROM signals
            WHERE symbol = %s
              AND created_at >= NOW() - INTERVAL '30 days'
            ORDER BY
                (trade_plan IS NOT NULL AND trade_plan != 'null'::jsonb) DESC,
                created_at DESC
            LIMIT 1
        """
        cur.execute(sql, (symbol,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row:
            logger.info(f"auto_context({symbol}): không tìm thấy record trong DB")
            return result

        verdict_label, confidence_pct, summary, created_at, state_vector_raw, trade_plan_raw = row

        result["found"]      = True
        result["created_at"] = created_at
        result["verdict"]    = verdict_label
        result["confidence"] = confidence_pct
        result["raw_summary"] = summary

        # Parse state_vector
        if state_vector_raw:
            try:
                sv = state_vector_raw if isinstance(state_vector_raw, dict) else json.loads(state_vector_raw)
                result["state_vector"] = sv
            except Exception:
                pass

        # Parse trade_plan
        if trade_plan_raw:
            try:
                tp = trade_plan_raw if isinstance(trade_plan_raw, dict) else json.loads(trade_plan_raw)
                if tp and tp.get("entry"):
                    result["trade_plan"] = tp
                    result["source"] = "vibe"
            except Exception:
                pass

        # Nếu không có trade_plan → source là "check"
        if result["source"] is None:
            result["source"] = "check"

        # Kiểm tra TTL
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_days = (now - created_at).total_seconds() / 86400

        if result["source"] == "vibe" and age_days > VIBE_CONTEXT_TTL_DAYS:
            # Vibe context quá cũ → thử tìm check context
            logger.info(f"auto_context({symbol}): vibe context {age_days:.1f} ngày → expired")
            result["trade_plan"] = None
            result["source"]     = "check_expired_vibe"
            if age_days > CHECK_CONTEXT_TTL_DAYS:
                result["found"] = False

        return result

    except Exception as e:
        logger.warning(f"auto_context({symbol}): DB error: {e}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# TRADE PLAN PARSER
# ══════════════════════════════════════════════════════════════════════════════

# Regex patterns để parse giá từ text vibe report
_PRICE_PATTERNS = [
    # "Entry: 62,100" / "Entry: 62.1k" / "Entry price: 62100"
    (r'(?:entry|diem\s*mua|vao\s*lenh|buy\s*at|mua\s*tai|gia\s*vao)[:\s*]+([0-9][0-9,\.]+[k]?)',
     "entry"),
    # "Stop: 59,000" / "Stop loss: 59k" / "Cat lo: 59"
    (r'(?:stop[\s_]?loss|stop|cat\s*lo|sl)[:\s*]+([0-9][0-9,\.]+[k]?)',
     "stop"),
    # "Target: 68,500" / "Mục tiêu: 68.5k" / "Take profit: 68500"
    (r'(?:target|take[\s_]?profit|muc\s*tieu|chot\s*loi|tp)[:\s*]+([0-9][0-9,\.]+[k]?)',
     "target"),
]


def _parse_price_str(s: str) -> Optional[float]:
    """
    Chuyển chuỗi giá → float.
    Xử lý: "62,100" → 62100; "62.1k" → 62100; "68500" → 68500
    """
    s = s.strip().lower().replace(",", "")
    try:
        if s.endswith("k"):
            return float(s[:-1]) * 1000
        return float(s)
    except ValueError:
        return None


def parse_trade_plan_from_text(text: str) -> Optional[dict]:
    """
    Parse trade plan từ text của vibe report.

    Args:
        text: Nội dung báo cáo từ /vibe

    Returns:
        dict {"entry": float, "stop": float, "target": float, "strategy": str}
        hoặc None nếu không parse được.
    """
    if not text:
        return None

    text_lower = text.lower()
    found = {}

    for pattern, key in _PRICE_PATTERNS:
        m = re.search(pattern, text_lower, re.IGNORECASE)
        if m:
            price = _parse_price_str(m.group(1))
            if price and price > 0:
                found[key] = price

    # Cần ít nhất entry + (stop hoặc target)
    if "entry" not in found:
        return None
    if "stop" not in found and "target" not in found:
        return None

    # Sanity check: stop < entry < target (long only)
    entry = found.get("entry", 0)
    stop  = found.get("stop")
    target = found.get("target")

    if stop and stop >= entry:
        logger.debug(f"parse_trade_plan: stop ({stop}) >= entry ({entry}) → invalid")
        found.pop("stop", None)
        stop = None

    if target and target <= entry:
        logger.debug(f"parse_trade_plan: target ({target}) <= entry ({entry}) → invalid")
        found.pop("target", None)
        target = None

    if not stop and not target:
        return None

    # Nếu thiếu stop → tự set = entry * 0.95 (mặc định 5%)
    if not stop:
        found["stop"] = round(entry * 0.95, 0)

    # Nếu thiếu target → tự set = entry * 1.10 (mặc định 10%)
    if not target:
        found["target"] = round(entry * 1.10, 0)

    # Extract strategy description (lấy dòng đầu của report)
    first_line = text.strip().split("\n")[0][:100]
    found["strategy"] = first_line

    return found


def trade_plan_to_rules(
    trade_plan: dict,
    current_price: Optional[float] = None,
) -> tuple[str, str]:
    """
    Chuyển trade_plan thành entry/exit rules cho backtest_rule.

    Logic:
    - Entry: mua khi close < entry_price (giá chưa vượt) AND rsi < 60 (không overbought)
    - Exit:  chốt lời khi close > target, HOẶC cắt lỗ trailing stop

    Returns:
        (entry_rule: str, exit_rule: str)
    """
    entry  = float(trade_plan["entry"])
    stop   = float(trade_plan["stop"])
    target = float(trade_plan["target"])

    # Tính % stop loss và take profit
    sl_pct = round((entry - stop) / entry * 100, 1)
    tp_pct = round((target - entry) / entry * 100, 1)

    # Entry rule: giá đang dưới entry_price với buffer 2%
    entry_upper = round(entry * 1.02, 0)  # mua khi giá chưa vượt quá 2% entry
    entry_rule  = f"close < {entry_upper:.0f} and rsi < 65"

    # Exit rule: take profit hoặc stop loss
    # Dùng take_profit% và stop_loss% thay vì giá cứng để backtest đúng
    exit_parts = []
    if tp_pct > 0:
        exit_parts.append(f"take_profit({tp_pct}%)")
    if sl_pct > 0:
        exit_parts.append(f"stop_loss({sl_pct}%)")
    # Fallback: thoát khi RSI overbought
    exit_parts.append("rsi > 75")

    exit_rule = " or ".join(exit_parts)

    return entry_rule, exit_rule


# ══════════════════════════════════════════════════════════════════════════════
# DB SAVE HELPERS (gọi từ bot.py sau /check và /vibe)
# ══════════════════════════════════════════════════════════════════════════════

def save_state_vector_to_signal(signal_id: int, state_vector: dict) -> bool:
    """
    Lưu state_vector vào cột signals.state_vector.
    Gọi sau khi save_signal() trả về signal_id.
    """
    if not signal_id or not state_vector:
        return False
    try:
        from db import get_conn  # type: ignore
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute(
            "UPDATE signals SET state_vector = %s::jsonb WHERE id = %s",
            (json.dumps(state_vector), signal_id)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"save_state_vector_to_signal({signal_id}): {e}")
        return False


def save_trade_plan_to_signal(signal_id: int, trade_plan: dict) -> bool:
    """
    Lưu trade_plan vào cột signals.trade_plan.
    Gọi sau /vibe khi parse được entry/stop/target.
    """
    if not signal_id or not trade_plan:
        return False
    try:
        from db import get_conn  # type: ignore
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute(
            "UPDATE signals SET trade_plan = %s::jsonb WHERE id = %s",
            (json.dumps(trade_plan), signal_id)
        )
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"save_trade_plan_to_signal({signal_id}): {e}")
        return False


def save_vibe_signal(
    symbol:     str,
    verdict:    str,
    confidence: float,
    summary:    str,
    trade_plan: Optional[dict],
    state_vector: Optional[dict],
) -> int:
    """
    Lưu kết quả /vibe vào DB.
    Trả về signal_id (hoặc -1 nếu lỗi).
    """
    try:
        from db import get_conn  # type: ignore
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO signals
                (symbol, verdict_label, confidence_pct, summary,
                 trade_plan, state_vector, created_at)
            VALUES (%s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, NOW())
            RETURNING id
        """, (
            symbol,
            verdict,
            confidence,
            summary[:2000] if summary else "",
            json.dumps(trade_plan) if trade_plan else None,
            json.dumps(state_vector) if state_vector else None,
        ))
        sid = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"save_vibe_signal({symbol}): id={sid}")
        return sid
    except Exception as e:
        logger.warning(f"save_vibe_signal({symbol}): {e}")
        return -1


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def format_trade_plan_summary(trade_plan: dict) -> str:
    """Text tóm tắt trade plan để gửi cùng backtest."""
    entry  = trade_plan.get("entry", 0)
    stop   = trade_plan.get("stop", 0)
    target = trade_plan.get("target", 0)
    sl_pct = (entry - stop)   / entry * 100 if entry > 0 else 0
    tp_pct = (target - entry) / entry * 100 if entry > 0 else 0
    rr     = tp_pct / sl_pct if sl_pct > 0 else 0

    return "\n".join([
        "TRADE PLAN TU VIBE:",
        f"  Entry  : {entry:,.0f} VND",
        f"  Stop   : {stop:,.0f} VND  ({sl_pct:.1f}%)",
        f"  Target : {target:,.0f} VND (+{tp_pct:.1f}%)",
        f"  R:R    : 1:{rr:.1f}",
        f"  Strategy: {trade_plan.get('strategy','N/A')[:60]}",
    ])


def format_no_context_msg(symbol: str) -> str:
    return (
        f"Chua co du lieu phan tich cho {symbol}.\n\n"
        f"Hay chay mot trong hai lenh sau truoc:\n"
        f"  /check {symbol}     — Phan tich nhanh (30-60 giay)\n"
        f"  /vibe {symbol}      — Phan tich sau voi Trade Plan\n\n"
        f"Sau do chay lai:\n"
        f"  /backtest_rule {symbol}   — Tu dong backtest tu ket qua phan tich"
    )
