"""
portfolio.py — Portfolio Dashboard cho VN Signal Bot

Chức năng:
  1. Lưu/xóa vị thế (positions) vào PostgreSQL
  2. /buy <MA> <gia> <SL> <KL> [TP]   — ghi nhận lệnh mua
  3. /sell <MA> [gia]                  — đóng vị thế
  4. /portfolio [MA] [--history]       — xem tổng quan hoặc chi tiết 1 mã
  5. Portfolio Alert cron               — 2 tầng: critical push ngay, digest sáng

Alert logic:
  Tầng 1 — CRITICAL (push ngay, bất kể giờ):
    - Giá qua SL
    - Giá chạm TP (còn <=3%)
    - Volume đột biến >3x TB + giá giảm so với entry

  Tầng 2 — MORNING DIGEST (gộp vào 8:15 AM cùng scan_watchlist):
    - Wave flip: SONG TANG → SONG GIAM (2 ngày liên tiếp)
    - Stock Regime flip
    - RSI extreme: >72 hoặc <28

Cooldown: mỗi (symbol, alert_type) chỉ fire 1 lần / 24h
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS — positions table
# ══════════════════════════════════════════════════════════════════════════════

POSITIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id           SERIAL PRIMARY KEY,
    user_id      BIGINT       NOT NULL,
    symbol       VARCHAR(10)  NOT NULL,
    entry_price  NUMERIC      NOT NULL,
    quantity     INTEGER      NOT NULL,
    sl_price     NUMERIC      NOT NULL,
    tp_price     NUMERIC,
    entry_date   DATE         NOT NULL DEFAULT CURRENT_DATE,
    status       VARCHAR(10)  NOT NULL DEFAULT 'open',
    exit_price   NUMERIC,
    exit_date    DATE,
    note         TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_alert_log (
    id          SERIAL PRIMARY KEY,
    user_id     BIGINT      NOT NULL,
    symbol      VARCHAR(10) NOT NULL,
    alert_type  VARCHAR(30) NOT NULL,
    fired_at    TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_user    ON positions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_positions_symbol  ON positions(user_id, symbol, status);
CREATE INDEX IF NOT EXISTS idx_alert_log_lookup  ON portfolio_alert_log(user_id, symbol, alert_type, fired_at);
"""

WAVE_VERDICT_SCHEMA = """
CREATE TABLE IF NOT EXISTS wave_verdict_log (
    id         SERIAL PRIMARY KEY,
    symbol     VARCHAR(10) NOT NULL,
    verdict    VARCHAR(20) NOT NULL,
    logged_at  DATE        NOT NULL DEFAULT CURRENT_DATE,
    UNIQUE(symbol, logged_at)
);
"""


def init_portfolio_schema():
    """Khởi tạo bảng positions + alert log. Gọi trong init_db()."""
    from db import get_conn
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute(POSITIONS_SCHEMA)
        cur.execute(WAVE_VERDICT_SCHEMA)
        conn.commit()
        cur.close()
        logger.info("Portfolio schema initialized OK")
    except Exception as e:
        logger.error(f"init_portfolio_schema error: {e}")
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


# ── CRUD positions ─────────────────────────────────────────────────────────────

def add_position(user_id: int, symbol: str, entry_price: float,
                 quantity: int, sl_price: float,
                 tp_price: float | None = None) -> int:
    """
    Thêm vị thế mới.  Trả về position_id hoặc -1 nếu lỗi.
    Nếu đã có vị thế mở cho symbol này → trả về -2 (yêu cầu đóng trước).
    """
    from db import get_conn
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()

        # Kiểm tra đã có vị thế mở chưa
        cur.execute(
            "SELECT id FROM positions WHERE user_id=%s AND symbol=%s AND status='open'",
            (user_id, symbol.upper())
        )
        existing = cur.fetchone()
        if existing:
            cur.close()
            return -2   # đã có vị thế mở

        cur.execute("""
            INSERT INTO positions (user_id, symbol, entry_price, quantity, sl_price, tp_price)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (user_id, symbol.upper()[:10], float(entry_price),
              int(quantity), float(sl_price),
              float(tp_price) if tp_price else None))
        row = cur.fetchone()
        pos_id = int(row[0]) if row else -1
        conn.commit()
        cur.close()
        return pos_id
    except Exception as e:
        logger.error(f"add_position error: {e}")
        if conn:
            try: conn.rollback()
            except Exception: pass
        return -1
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


def close_position(user_id: int, symbol: str,
                   exit_price: float | None = None) -> dict:
    """
    Đóng vị thế mở cho symbol.
    Trả về {"ok": bool, "pnl_pct": float, "pnl_vnd": float, "msg": str}.
    """
    from db import get_conn
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()

        cur.execute("""
            SELECT id, entry_price, quantity, sl_price, tp_price
            FROM positions
            WHERE user_id=%s AND symbol=%s AND status='open'
            ORDER BY entry_date ASC
            LIMIT 1
        """, (user_id, symbol.upper()))
        row = cur.fetchone()
        if not row:
            return {"ok": False, "msg": f"Khong co vi the mo nao cho {symbol}"}

        pos_id, entry_price, quantity, sl_price, tp_price = row
        entry_price = float(entry_price)
        quantity    = int(quantity)

        # Nếu không truyền exit_price → lấy giá hiện tại
        if exit_price is None:
            from analyzer import get_price_data
            price_res = get_price_data(symbol.upper(), days=5)
            if price_res.get("success"):
                df = price_res["df"]
                exit_price = float(df["close"].iloc[-1])
            else:
                return {"ok": False, "msg": f"Khong lay duoc gia hien tai cho {symbol}"}
        else:
            exit_price = float(exit_price)

        pnl_pct = (exit_price - entry_price) / entry_price * 100
        pnl_vnd = (exit_price - entry_price) * quantity

        cur.execute("""
            UPDATE positions
            SET status='closed', exit_price=%s, exit_date=CURRENT_DATE
            WHERE id=%s
        """, (exit_price, pos_id))
        conn.commit()
        cur.close()

        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        return {
            "ok":      True,
            "symbol":  symbol.upper(),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_vnd": round(pnl_vnd, 0),
            "msg": (
                f"{emoji} Da dong {symbol.upper()}: "
                f"Entry {entry_price:,.0f} → Exit {exit_price:,.0f} "
                f"({pnl_pct:+.1f}% | {pnl_vnd:+,.0f} VND)"
            ),
        }
    except Exception as e:
        logger.error(f"close_position error: {e}")
        if conn:
            try: conn.rollback()
            except Exception: pass
        return {"ok": False, "msg": f"Loi: {e}"}
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


def get_open_positions(user_id: int) -> list[dict]:
    """Lấy tất cả vị thế đang mở của user."""
    from db import get_conn
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT id, symbol, entry_price, quantity, sl_price, tp_price, entry_date
            FROM positions
            WHERE user_id=%s AND status='open'
            ORDER BY entry_date ASC
        """, (user_id,))
        rows = cur.fetchall()
        cur.close()
        return [
            {
                "id":          r[0],
                "symbol":      r[1],
                "entry_price": float(r[2]),
                "quantity":    int(r[3]),
                "sl_price":    float(r[4]),
                "tp_price":    float(r[5]) if r[5] else None,
                "entry_date":  r[6],
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"get_open_positions error: {e}")
        return []
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


def get_closed_positions(user_id: int, limit: int = 20) -> list[dict]:
    """Lịch sử lệnh đã đóng."""
    from db import get_conn
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT symbol, entry_price, exit_price, quantity, entry_date, exit_date
            FROM positions
            WHERE user_id=%s AND status='closed'
            ORDER BY exit_date DESC
            LIMIT %s
        """, (user_id, limit))
        rows = cur.fetchall()
        cur.close()
        result = []
        for r in rows:
            ep   = float(r[1])
            xp   = float(r[2]) if r[2] else ep
            qty  = int(r[3])
            result.append({
                "symbol":      r[0],
                "entry_price": ep,
                "exit_price":  xp,
                "quantity":    qty,
                "entry_date":  r[4],
                "exit_date":   r[5],
                "pnl_pct":     round((xp - ep) / ep * 100, 2),
                "pnl_vnd":     round((xp - ep) * qty, 0),
            })
        return result
    except Exception as e:
        logger.error(f"get_closed_positions error: {e}")
        return []
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# PRICE FETCH — parallel
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_current_price(symbol: str) -> tuple[str, float | None]:
    """Worker: lấy giá đóng cửa gần nhất."""
    try:
        from analyzer import get_price_data
        res = get_price_data(symbol, days=10)
        if res.get("success"):
            price = float(res["df"]["close"].iloc[-1])
            return symbol, price
    except Exception as e:
        logger.warning(f"_fetch_current_price {symbol}: {e}")
    return symbol, None


def _fetch_prices_parallel(symbols: list[str]) -> dict[str, float | None]:
    """Lấy giá song song cho danh sách symbols."""
    if not symbols:
        return {}
    with ThreadPoolExecutor(max_workers=min(5, len(symbols))) as ex:
        results = list(ex.map(_fetch_current_price, symbols))
    return dict(results)


# ══════════════════════════════════════════════════════════════════════════════
# ENRICH — gắn wave + stock regime vào positions (dùng cache)
# ══════════════════════════════════════════════════════════════════════════════

def _get_wave_info(symbol: str) -> dict:
    """Lấy wave từ cache — KHÔNG recompute."""
    try:
        from wave_pattern import analyze_wave
        result = analyze_wave(symbol, force_rebuild=False)
        return {
            "verdict":    result.get("verdict", "KHONG RO"),
            "score_up":   result.get("score_up", 0.0),
            "score_down": result.get("score_down", 0.0),
            "confidence": result.get("confidence", 0.0),
            "amp_up":     result.get("amp_up_mean", 0.0),
            "amp_down":   result.get("amp_down_mean", 0.0),
        }
    except Exception as e:
        logger.warning(f"_get_wave_info {symbol}: {e}")
        return {"verdict": "KHONG RO", "score_up": 0.0, "score_down": 0.0,
                "confidence": 0.0, "amp_up": 0.0, "amp_down": 0.0}


def _get_sr_info(symbol: str) -> dict:
    """Lấy Stock Regime từ cache — KHÔNG recompute."""
    try:
        from stock_regime import get_stock_regime
        sr = get_stock_regime(symbol, force_rebuild=False)
        return {
            "label":      sr.get("label", "Khong xac dinh"),
            "sr":         sr.get("sr", 0),
            "confidence": sr.get("confidence", 0.0),
            "conf_label": sr.get("confidence_label", ""),
        }
    except Exception as e:
        logger.warning(f"_get_sr_info {symbol}: {e}")
        return {"label": "N/A", "sr": 0, "confidence": 0.0, "conf_label": ""}


def _get_rsi(symbol: str) -> float | None:
    """Lấy RSI hiện tại từ price data."""
    try:
        from analyzer import get_price_data
        import pandas as pd
        res = get_price_data(symbol, days=60)
        if not res.get("success"):
            return None
        close = res["df"]["close"].astype(float)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-9)
        rsi   = 100 - (100 / (1 + rs))
        return round(float(rsi.iloc[-1]), 1)
    except Exception:
        return None


def _get_volume_spike(symbol: str) -> float | None:
    """Volume hôm nay / TB 20 ngày."""
    try:
        from analyzer import get_price_data
        res = get_price_data(symbol, days=60)
        if not res.get("success"):
            return None
        vol  = res["df"]["volume"].astype(float)
        avg  = vol.iloc[-21:-1].mean()
        curr = vol.iloc[-1]
        if avg < 1:
            return None
        return round(curr / avg, 2)
    except Exception:
        return None


def enrich_position(pos: dict) -> dict:
    """Gắn wave + SR + RSI vào 1 vị thế."""
    symbol = pos["symbol"]
    pos["wave"] = _get_wave_info(symbol)
    pos["sr"]   = _get_sr_info(symbol)
    pos["rsi"]  = _get_rsi(symbol)
    pos["vol_spike"] = _get_volume_spike(symbol)
    return pos


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _wave_stars(score_up: float, score_down: float, verdict: str) -> str:
    if verdict == "SONG TANG":
        s = score_up
        d = "TANG"
    elif verdict == "SONG GIAM":
        s = score_down
        d = "GIAM"
    else:
        return "★☆☆ ?"
    stars = "★★★" if s >= 0.65 else "★★☆" if s >= 0.45 else "★☆☆"
    return f"{stars} {d}"


def _sr_emoji(sr: int) -> str:
    return {1: "🔵", 2: "🟢", 3: "🟡", 4: "🔴"}.get(sr, "⚪")


def _pnl_emoji(pnl_pct: float, near_sl: bool, past_sl: bool) -> str:
    if past_sl:   return "❌"
    if near_sl:   return "⚠️ "
    if pnl_pct >= 5: return "🟢"
    if pnl_pct >= 0: return "✅"
    return "🔴"


def _conflict_note(wave_verdict: str, sr: int) -> str:
    """Phát hiện mâu thuẫn Wave vs Stock Regime."""
    if wave_verdict == "SONG GIAM" and sr in (1, 2):
        return "← Wave GIAM nhung SR Accum/Markup"
    if wave_verdict == "SONG TANG" and sr in (3, 4):
        return "← Wave TANG nhung SR Distrib/Markdown"
    return ""


def format_portfolio_overview(positions: list[dict], prices: dict,
                               regime_label: str = "") -> str:
    """Render /portfolio tổng quan."""
    if not positions:
        return (
            "Portfolio trong.\n"
            "Dung /buy <MA> <gia> <SL> <KL> [TP] de ghi nhan vi the.\n"
            "Vi du: /buy HAH 57000 54000 10000"
        )

    today = datetime.now().strftime("%d/%m/%Y")
    regime_line = f"  Regime: {regime_label}" if regime_label else ""

    # Tính tổng P&L
    total_value   = 0.0
    total_cost    = 0.0
    total_pnl_vnd = 0.0

    enriched = []
    for pos in positions:
        sym    = pos["symbol"]
        ep     = pos["entry_price"]
        qty    = pos["quantity"]
        sl     = pos["sl_price"]
        tp     = pos["tp_price"]
        price  = prices.get(sym)
        wave   = pos.get("wave", {})
        sr     = pos.get("sr", {})

        if price is None:
            price = ep   # fallback

        pnl_pct  = (price - ep) / ep * 100
        pnl_vnd  = (price - ep) * qty
        cost     = ep * qty
        val      = price * qty

        past_sl  = price <= sl
        near_sl  = (not past_sl) and (price <= sl * 1.05)
        near_tp  = tp and (price >= tp * 0.97)

        total_cost    += cost
        total_value   += val
        total_pnl_vnd += pnl_vnd

        enriched.append({
            **pos,
            "price":    price,
            "pnl_pct":  pnl_pct,
            "pnl_vnd":  pnl_vnd,
            "past_sl":  past_sl,
            "near_sl":  near_sl,
            "near_tp":  near_tp,
        })

    total_pnl_pct = (total_value - total_cost) / total_cost * 100 if total_cost > 0 else 0

    # ── Header ────────────────────────────────────────────────────────────────
    lines = [
        f"PORTFOLIO — {today}{regime_line}",
        "=" * 38,
        "",
        "TONG QUAN:",
        f"  Von dang giu : {total_cost:>15,.0f} VND",
        f"  Gia tri HT   : {total_value:>15,.0f} VND",
        f"  P&L tong     : {total_pnl_vnd:>+15,.0f} VND  ({total_pnl_pct:+.1f}%)",
        "",
        "=" * 38,
    ]

    # ── Phân nhóm ────────────────────────────────────────────────────────────
    groups = {
        "critical": [],   # past SL
        "warning":  [],   # near SL hoặc wave xấu
        "profit":   [],   # đang lời ổn định
        "neutral":  [],   # trong trung
    }
    for p in enriched:
        if p["past_sl"]:
            groups["critical"].append(p)
        elif p["near_sl"] or (p.get("wave", {}).get("verdict") == "SONG GIAM"
                               and p["pnl_pct"] < 0):
            groups["warning"].append(p)
        elif p["pnl_pct"] >= 0:
            groups["profit"].append(p)
        else:
            groups["neutral"].append(p)

    def _pos_line(p) -> str:
        sym   = p["symbol"]
        price = p["price"]
        ep    = p["entry_price"]
        qty   = p["quantity"]
        sl    = p["sl_price"]
        tp    = p["tp_price"]
        pct   = p["pnl_pct"]
        past  = p["past_sl"]
        near  = p["near_sl"]
        ntp   = p["near_tp"]
        wave  = p.get("wave", {})
        sr    = p.get("sr", {})

        em      = _pnl_emoji(pct, near, past)
        stars   = _wave_stars(wave.get("score_up", 0), wave.get("score_down", 0),
                              wave.get("verdict", ""))
        sr_em   = _sr_emoji(sr.get("sr", 0))
        sr_lbl  = sr.get("label", "N/A")[:12]
        conflict = _conflict_note(wave.get("verdict", ""), sr.get("sr", 0))

        # SL buffer
        sl_buf = (price - sl) / sl * 100
        sl_tag = (f"❌QUA SL" if past else
                  f"⚠️ SL:{sl_buf:+.1f}%" if near else
                  f"SL:{sl_buf:+.1f}%")

        tp_tag = f" ⚡TP gần!" if ntp else ""

        line = (
            f"{em} {sym:<5} {pct:>+5.1f}%  {price:>8,.0f}"
            f" | KL {qty:,} | {sl_tag}{tp_tag}\n"
            f"   Wave {stars} | {sr_em}{sr_lbl}"
        )
        if conflict:
            line += f"\n   {conflict}"
        return line

    # In từng nhóm
    if groups["critical"]:
        lines.append("KHAN CAP — Qua SL:")
        for p in sorted(groups["critical"], key=lambda x: x["pnl_pct"]):
            lines.append(_pos_line(p))
        lines.append("")

    if groups["warning"]:
        lines.append("CAN CHU Y — Xem lai:")
        for p in sorted(groups["warning"], key=lambda x: x["pnl_pct"]):
            lines.append(_pos_line(p))
        lines.append("")

    if groups["profit"]:
        lines.append("DANG LOI — Giu:")
        for p in sorted(groups["profit"], key=lambda x: -x["pnl_pct"]):
            lines.append(_pos_line(p))
        lines.append("")

    if groups["neutral"]:
        lines.append("THEO DOI:")
        for p in sorted(groups["neutral"], key=lambda x: x["pnl_pct"]):
            lines.append(_pos_line(p))
        lines.append("")

    # ── Action suggestions ────────────────────────────────────────────────────
    actions = _build_action_suggestions(enriched, regime_label)
    if actions:
        lines.append("=" * 38)
        lines.append("HANH DONG HOM NAY:")
        lines.extend(actions)

    return "\n".join(lines)


def format_position_detail(pos: dict, price: float) -> str:
    """Render /portfolio <MA> chi tiết 1 mã."""
    sym   = pos["symbol"]
    ep    = pos["entry_price"]
    qty   = pos["quantity"]
    sl    = pos["sl_price"]
    tp    = pos["tp_price"]
    ed    = pos["entry_date"]
    wave  = pos.get("wave", {})
    sr    = pos.get("sr", {})
    rsi   = pos.get("rsi")
    vol   = pos.get("vol_spike")

    pnl_pct = (price - ep) / ep * 100
    pnl_vnd = (price - ep) * qty
    sl_buf  = (price - sl) / sl * 100
    val     = price * qty

    past_sl = price <= sl
    near_sl = (not past_sl) and (price <= sl * 1.05)
    near_tp = tp and (price >= tp * 0.97)

    pnl_em = "🟢" if pnl_pct >= 0 else "🔴"
    wave_stars = _wave_stars(wave.get("score_up", 0), wave.get("score_down", 0),
                             wave.get("verdict", ""))
    sr_em  = _sr_emoji(sr.get("sr", 0))
    sr_lbl = sr.get("label", "N/A")
    sr_conf = sr.get("conf_label", "")
    conflict = _conflict_note(wave.get("verdict", ""), sr.get("sr", 0))

    entry_date_str = ed.strftime("%d/%m/%Y") if hasattr(ed, "strftime") else str(ed)

    lines = [
        f"{sym} — Chi tiet vi the",
        "=" * 34,
        f"  Entry    : {ep:>10,.0f}  ({entry_date_str})",
        f"  Gia HT   : {price:>10,.0f}  ({pnl_pct:+.1f}%)  {pnl_em}",
        f"  SL       : {sl:>10,.0f}  (buffer {sl_buf:+.1f}%)"
        + ("  ❌ DA QUA SL!" if past_sl else "  ⚠️ GAN SL!" if near_sl else ""),
    ]

    if tp:
        tp_buf = (tp - price) / price * 100
        lines.append(
            f"  TP       : {tp:>10,.0f}  (con {tp_buf:+.1f}%)"
            + ("  ⚡ GAN TP!" if near_tp else "")
        )

    lines += [
        f"  KL       : {qty:>10,}  CP",
        f"  Gia tri  : {val:>10,.0f}  VND",
        f"  P&L      : {pnl_vnd:>+10,.0f}  VND",
        "",
        "─" * 34,
        f"  Wave     : {wave_stars}  (score up={wave.get('score_up',0):.2f})",
        f"  Stock SR : {sr_em}{sr_lbl}  {sr_conf}",
    ]

    if rsi is not None:
        rsi_note = ""
        if rsi > 72: rsi_note = "  ← Overbought"
        elif rsi < 28: rsi_note = "  ← Oversold"
        lines.append(f"  RSI      : {rsi:.1f}{rsi_note}")

    if vol is not None:
        vol_note = "  ← Dot bien!" if vol > 2.0 else ""
        lines.append(f"  Vol Spike: {vol:.1f}x{vol_note}")

    if conflict:
        lines.append(f"\n  ⚠️  {conflict}")

    # Gợi ý hành động
    lines.append("")
    lines.append("─" * 34)
    lines.append("GOI Y:")
    suggestion = _single_position_suggestion(pos, price)
    lines.append(f"  {suggestion}")

    return "\n".join(lines)


def _single_position_suggestion(pos: dict, price: float) -> str:
    sl    = pos["sl_price"]
    tp    = pos["tp_price"]
    wave  = pos.get("wave", {})
    sr    = pos.get("sr", {})
    pnl   = (price - pos["entry_price"]) / pos["entry_price"] * 100

    past_sl  = price <= sl
    near_sl  = (not past_sl) and price <= sl * 1.05
    near_tp  = tp and price >= tp * 0.97
    w_verdict = wave.get("verdict", "")
    sr_num   = sr.get("sr", 0)

    if past_sl:
        return "Cat lo ngay — gia da qua SL"
    if near_sl and w_verdict == "SONG GIAM":
        return "Wave GIAM + sap cham SL → cat lo hoac keo SL chat lai"
    if near_tp and w_verdict == "SONG TANG":
        return "Gan TP + Wave TANG → chot 50% hoac keo SL len breakeven"
    if near_tp:
        return "Gan TP → chot 50% de chot loi"
    if w_verdict == "SONG GIAM" and sr_num in (1, 2):
        return "Wave GIAM nhung SR tot → giu, SL cung, cho xac nhan them"
    if w_verdict == "SONG TANG" and sr_num in (3, 4):
        return "Wave TANG nhung SR xau → giu nhe, quan sat ky"
    if pnl >= 8 and w_verdict == "SONG TANG":
        return f"Loi {pnl:.1f}% + Wave TANG → keo SL len breakeven, de song tiep"
    if pnl < -3 and w_verdict == "SONG GIAM":
        return "Dang lo + Wave GIAM → xem xet cat lo, tranh thu lo lon"
    return "Tiep tuc theo doi, chua co tin hieu hanh dong ro rang"


def _build_action_suggestions(positions: list[dict],
                               regime_label: str = "") -> list[str]:
    """Sinh danh sách gợi ý hành động từ toàn bộ vị thế."""
    actions = []
    cut_symbols   = []
    profit_symbols = []

    for p in positions:
        sym   = p["symbol"]
        price = p["price"]
        sl    = p["sl_price"]
        tp    = p["tp_price"]
        wave  = p.get("wave", {})
        pnl   = p["pnl_pct"]

        if p["past_sl"]:
            cut_symbols.append(f"{sym}({pnl:+.1f}%)")
        elif p["near_tp"] and wave.get("verdict") == "SONG TANG":
            profit_symbols.append(f"{sym}(+{pnl:.1f}%)")

    if cut_symbols:
        actions.append(f"  ❌ Cat lo: {', '.join(cut_symbols)}")
    if profit_symbols:
        actions.append(f"  ✅ Cot loi 50%: {', '.join(profit_symbols)}")

    # Gợi ý theo Regime
    if "R4" in regime_label or "Bear Volatile" in regime_label:
        actions.append("  ⛔ R4 Bear Volatile — khong mo vi the moi")
    elif "R3" in regime_label or "Bear Quiet" in regime_label:
        actions.append("  ⚠️  R3 Bear Quiet — han che vi the moi, uu tien cash")
    elif "R2" in regime_label or "Bull Volatile" in regime_label:
        actions.append("  📊 R2 Bull Volatile — giu, chu y risk management")

    return actions


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY FORMAT
# ══════════════════════════════════════════════════════════════════════════════

def format_portfolio_history(positions: list[dict]) -> str:
    """Render lịch sử lệnh đã đóng."""
    if not positions:
        return "Chua co lenh nao da dong."

    total_pnl = sum(p["pnl_vnd"] for p in positions)
    wins  = sum(1 for p in positions if p["pnl_pct"] > 0)
    total = len(positions)

    lines = [
        f"LICH SU PORTFOLIO ({total} lenh)",
        f"Win rate: {wins}/{total} ({wins/total*100:.0f}%)",
        f"Total P&L: {total_pnl:+,.0f} VND",
        "─" * 38,
    ]
    for p in positions:
        em = "🟢" if p["pnl_pct"] > 0 else "🔴"
        ed = p["entry_date"]
        xd = p["exit_date"]
        ed_s = ed.strftime("%d/%m") if hasattr(ed, "strftime") else str(ed)[:5]
        xd_s = xd.strftime("%d/%m") if hasattr(xd, "strftime") else str(xd)[:5]
        lines.append(
            f"{em} {p['symbol']:<5}  {ed_s}→{xd_s}  "
            f"{p['entry_price']:,.0f}→{p['exit_price']:,.0f}  "
            f"{p['pnl_pct']:+.1f}%  ({p['pnl_vnd']:+,.0f})"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ALERT SYSTEM — 2 tầng
# ══════════════════════════════════════════════════════════════════════════════

def _was_alerted_recently(user_id: int, symbol: str,
                           alert_type: str, hours: int = 24) -> bool:
    """Kiểm tra cooldown: signal này đã fire trong `hours` giờ qua chưa?"""
    from db import get_conn
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            SELECT 1 FROM portfolio_alert_log
            WHERE user_id=%s AND symbol=%s AND alert_type=%s
              AND fired_at > NOW() - INTERVAL '1 hour' * %s
            LIMIT 1
        """, (user_id, symbol, alert_type, hours))
        found = cur.fetchone() is not None
        cur.close()
        return found
    except Exception as e:
        logger.warning(f"_was_alerted_recently error: {e}")
        return False
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


def _log_alert(user_id: int, symbol: str, alert_type: str):
    """Ghi nhận alert đã fire vào log (cooldown tracking)."""
    from db import get_conn
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO portfolio_alert_log (user_id, symbol, alert_type)
            VALUES (%s, %s, %s)
        """, (user_id, symbol, alert_type))
        conn.commit()
        cur.close()
    except Exception as e:
        logger.warning(f"_log_alert error: {e}")
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


def _get_wave_verdict_yesterday(symbol: str) -> str | None:
    """Lấy wave verdict ngày hôm qua từ log."""
    from db import get_conn
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        cur.execute("""
            SELECT verdict FROM wave_verdict_log
            WHERE symbol=%s AND logged_at=%s
        """, (symbol, yesterday))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None
    except Exception:
        return None
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


def _log_wave_verdict_today(symbol: str, verdict: str):
    """Ghi wave verdict hôm nay vào log (cho flip detection ngày mai)."""
    from db import get_conn
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()
        today = date.today().isoformat()
        cur.execute("""
            INSERT INTO wave_verdict_log (symbol, verdict, logged_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (symbol, logged_at) DO UPDATE SET verdict = EXCLUDED.verdict
        """, (symbol, verdict, today))
        conn.commit()
        cur.close()
    except Exception as e:
        logger.warning(f"_log_wave_verdict_today {symbol}: {e}")
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


def check_portfolio_alerts(user_id: int) -> tuple[list[str], list[str]]:
    """
    Kiểm tra tất cả positions của user, phân loại alerts.

    Returns:
        (critical_alerts, digest_items)
        critical_alerts : list[str] — push ngay
        digest_items    : list[str] — gộp vào morning digest
    """
    positions = get_open_positions(user_id)
    if not positions:
        return [], []

    symbols = [p["symbol"] for p in positions]
    prices  = _fetch_prices_parallel(symbols)

    critical = []
    digest   = []

    for pos in positions:
        sym    = pos["symbol"]
        ep     = pos["entry_price"]
        qty    = pos["quantity"]
        sl     = pos["sl_price"]
        tp     = pos["tp_price"]
        price  = prices.get(sym)
        if price is None:
            continue

        pnl_pct = (price - ep) / ep * 100
        pnl_vnd = (price - ep) * qty

        # ── Tầng 1: CRITICAL ─────────────────────────────────────────────────

        # 1a. Giá qua SL
        if price <= sl and not _was_alerted_recently(user_id, sym, "sl_breach"):
            msg = (
                f"🚨 [{sym}] GIA QUA SL\n"
                f"  Gia: {price:,.0f} | SL: {sl:,.0f}\n"
                f"  P&L: {pnl_pct:+.1f}% ({pnl_vnd:+,.0f} VND)\n"
                f"  → Can cat lo ngay"
            )
            critical.append(msg)
            _log_alert(user_id, sym, "sl_breach")

        # 1b. Giá chạm TP (còn <=3%)
        elif tp and price >= tp * 0.97 and not _was_alerted_recently(user_id, sym, "tp_near"):
            tp_buf = (tp - price) / price * 100
            msg = (
                f"⚡ [{sym}] GAN TP\n"
                f"  Gia: {price:,.0f} | TP: {tp:,.0f} (con {tp_buf:.1f}%)\n"
                f"  P&L: {pnl_pct:+.1f}% ({pnl_vnd:+,.0f} VND)\n"
                f"  → Can nhac chot loi"
            )
            critical.append(msg)
            _log_alert(user_id, sym, "tp_near")

        # 1c. Volume đột biến >3x + giá giảm vs entry
        else:
            vol_spike = _get_volume_spike(sym)
            if (vol_spike and vol_spike > 3.0 and price < ep
                    and not _was_alerted_recently(user_id, sym, "vol_spike_drop")):
                msg = (
                    f"🚨 [{sym}] VOLUME DOT BIEN + GIA GIAM\n"
                    f"  Volume: {vol_spike:.1f}x TB 20 ngay | Gia: {price:,.0f}\n"
                    f"  P&L: {pnl_pct:+.1f}% | → Kiem tra tin tuc ngay"
                )
                critical.append(msg)
                _log_alert(user_id, sym, "vol_spike_drop")

        # ── Tầng 2: DIGEST ───────────────────────────────────────────────────

        wave_info = _get_wave_info(sym)
        verdict   = wave_info.get("verdict", "")

        # Ghi log verdict hôm nay (để hôm sau detect flip)
        if verdict in ("SONG TANG", "SONG GIAM"):
            _log_wave_verdict_today(sym, verdict)

        # 2a. Wave flip: hôm qua TANG, hôm nay GIAM (confirmation 2 ngày)
        prev_verdict = _get_wave_verdict_yesterday(sym)
        if (prev_verdict == "SONG TANG" and verdict == "SONG GIAM"
                and not _was_alerted_recently(user_id, sym, "wave_flip_down", hours=48)):
            digest.append(f"⚠️  {sym} — Wave dao chieu TANG→GIAM")
            _log_alert(user_id, sym, "wave_flip_down")

        elif (prev_verdict == "SONG GIAM" and verdict == "SONG TANG"
              and not _was_alerted_recently(user_id, sym, "wave_flip_up", hours=48)):
            digest.append(f"✅ {sym} — Wave dao chieu GIAM→TANG")
            _log_alert(user_id, sym, "wave_flip_up")

        # 2b. RSI extreme
        rsi = _get_rsi(sym)
        if rsi is not None:
            if rsi > 72 and not _was_alerted_recently(user_id, sym, "rsi_overbought"):
                digest.append(f"📈 {sym} — RSI {rsi:.0f} (overbought)")
                _log_alert(user_id, sym, "rsi_overbought")
            elif rsi < 28 and not _was_alerted_recently(user_id, sym, "rsi_oversold"):
                digest.append(f"📉 {sym} — RSI {rsi:.0f} (oversold)")
                _log_alert(user_id, sym, "rsi_oversold")

    return critical, digest


def format_morning_digest(digest_items: list[str],
                           regime_label: str = "") -> str:
    """Format portfolio digest cho morning report."""
    if not digest_items:
        return ""

    lines = [
        "PORTFOLIO DIGEST:",
        "─" * 30,
    ]
    lines.extend(digest_items)
    if regime_label:
        lines.append(f"Regime: {regime_label}")
    lines.append("Dung /portfolio de xem chi tiet")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

async def buy_cmd(update, context):
    """
    /buy <MA> <gia> <SL> <KL> [TP]
    Vi du:
      /buy HAH 57000 54000 10000
      /buy DVP 45200 42000 5000 49000
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update); return

    args = context.args or []
    if len(args) < 4:
        await update.message.reply_text(
            "Cu phap: /buy <MA> <gia> <SL> <KL> [TP]\n"
            "Vi du:  /buy HAH 57000 54000 10000\n"
            "        /buy DVP 45200 42000 5000 49000\n\n"
            "  MA  : ma co phieu\n"
            "  gia : gia mua vao\n"
            "  SL  : stop loss (bat buoc)\n"
            "  KL  : so luong co phieu\n"
            "  TP  : take profit (tuy chon)"
        )
        return

    import re as _re
    symbol_raw = args[0].upper().strip()
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol_raw):
        await update.message.reply_text(f"Ma '{symbol_raw}' khong hop le.")
        return

    try:
        entry_price = float(args[1].replace(",", ""))
        sl_price    = float(args[2].replace(",", ""))
        quantity    = int(args[3].replace(",", ""))
        tp_price    = float(args[4].replace(",", "")) if len(args) >= 5 else None
    except ValueError:
        await update.message.reply_text(
            "Loi: gia/SL/KL/TP phai la so.\n"
            "Vi du: /buy HAH 57000 54000 10000"
        )
        return

    if sl_price >= entry_price:
        await update.message.reply_text(
            f"SL ({sl_price:,.0f}) phai nho hon gia mua ({entry_price:,.0f})."
        )
        return
    if quantity <= 0:
        await update.message.reply_text("So luong phai > 0.")
        return
    if tp_price and tp_price <= entry_price:
        await update.message.reply_text(
            f"TP ({tp_price:,.0f}) phai lon hon gia mua ({entry_price:,.0f})."
        )
        return

    user_id = update.effective_user.id

    pos_id = add_position(user_id, symbol_raw, entry_price,
                          quantity, sl_price, tp_price)

    if pos_id == -2:
        await update.message.reply_text(
            f"{symbol_raw} da co vi the mo roi.\n"
            f"Dung /sell {symbol_raw} de dong truoc, roi mua lai."
        )
        return

    if pos_id < 0:
        await update.message.reply_text(f"Loi khi luu vi the {symbol_raw}. Thu lai.")
        return

    sl_pct = (entry_price - sl_price) / entry_price * 100
    rr     = ""
    if tp_price:
        tp_pct = (tp_price - entry_price) / entry_price * 100
        rr_val = tp_pct / sl_pct if sl_pct > 0 else 0
        rr = f"\n  TP   : {tp_price:,.0f} (+{tp_pct:.1f}%)  |  R:R = 1:{rr_val:.1f}"

    cost = entry_price * quantity
    msg  = (
        f"✅ Da ghi nhan vi the #{pos_id}\n"
        f"  {symbol_raw}: {entry_price:,.0f} x {quantity:,} CP\n"
        f"  Gia tri: {cost:,.0f} VND\n"
        f"  SL   : {sl_price:,.0f} (-{sl_pct:.1f}%){rr}\n\n"
        f"Dung /portfolio de xem tong quan."
    )
    await update.message.reply_text(msg)


async def sell_cmd(update, context):
    """
    /sell <MA> [gia]
    Vi du:
      /sell HAH          → dong voi gia hien tai
      /sell HAH 59500    → dong voi gia cu the
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update); return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Cu phap: /sell <MA> [gia]\n"
            "Vi du:  /sell HAH\n"
            "        /sell HAH 59500"
        )
        return

    import re as _re
    symbol_raw = args[0].upper().strip()
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol_raw):
        await update.message.reply_text(f"Ma '{symbol_raw}' khong hop le.")
        return

    exit_price = None
    if len(args) >= 2:
        try:
            exit_price = float(args[1].replace(",", ""))
        except ValueError:
            await update.message.reply_text("Gia ban khong hop le.")
            return

    user_id = update.effective_user.id
    msg_obj = await update.message.reply_text(f"Dang xu ly /sell {symbol_raw}...")

    import asyncio
    result = await asyncio.to_thread(close_position, user_id, symbol_raw, exit_price)

    await msg_obj.edit_text(result["msg"])


async def portfolio_cmd(update, context):
    """
    /portfolio              — tong quan
    /portfolio <MA>         — chi tiet 1 ma
    /portfolio --history    — lich su lenh da dong
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update); return

    user_id = update.effective_user.id
    args    = context.args or []

    # ── /portfolio --history ─────────────────────────────────────────────────
    if "--history" in args:
        msg_obj = await update.message.reply_text("Dang tai lich su lenh...")
        import asyncio
        closed = await asyncio.to_thread(get_closed_positions, user_id)
        text   = format_portfolio_history(closed)
        await msg_obj.edit_text(text)
        return

    # ── /portfolio <MA> ──────────────────────────────────────────────────────
    import re as _re
    if args and _re.match(r'^[A-Z0-9]{2,10}$', args[0].upper()):
        symbol = args[0].upper()
        msg_obj = await update.message.reply_text(f"Dang tai chi tiet {symbol}...")

        import asyncio
        positions = await asyncio.to_thread(get_open_positions, user_id)
        pos = next((p for p in positions if p["symbol"] == symbol), None)

        if not pos:
            await msg_obj.edit_text(
                f"Khong co vi the mo nao cho {symbol}.\n"
                f"Dung /buy {symbol} <gia> <SL> <KL> de ghi nhan."
            )
            return

        pos = await asyncio.to_thread(enrich_position, pos)
        _, price = await asyncio.to_thread(_fetch_current_price, symbol)
        price = price or pos["entry_price"]

        text = format_position_detail(pos, price)
        await msg_obj.edit_text(text)
        return

    # ── /portfolio tổng quan ─────────────────────────────────────────────────
    msg_obj = await update.message.reply_text(
        "Dang tai portfolio... (lay gia song song)"
    )

    import asyncio

    positions = await asyncio.to_thread(get_open_positions, user_id)
    if not positions:
        await msg_obj.edit_text(
            "Portfolio trong.\n"
            "Dung /buy <MA> <gia> <SL> <KL> [TP] de them vi the.\n"
            "Vi du: /buy HAH 57000 54000 10000"
        )
        return

    # Fetch giá + enrich song song
    symbols = [p["symbol"] for p in positions]
    prices  = await asyncio.to_thread(_fetch_prices_parallel, symbols)

    def _enrich_all():
        result = []
        for pos in positions:
            result.append(enrich_position(pos))
        return result

    enriched = await asyncio.to_thread(_enrich_all)

    # Market Regime
    regime_label = ""
    try:
        from market_regime import get_market_regime
        mr = await asyncio.to_thread(get_market_regime)
        if mr:
            regime_label = mr.get("label", "")
    except Exception:
        pass

    text = format_portfolio_overview(enriched, prices, regime_label)

    # Split nếu dài
    if len(text) <= 4096:
        try:
            await msg_obj.edit_text(text)
        except Exception:
            await update.message.reply_text(text[:4096])
    else:
        try:
            await msg_obj.edit_text(text[:4000])
        except Exception:
            await update.message.reply_text(text[:4000])
        if len(text) > 4000:
            await update.message.reply_text(text[4000:4096])
