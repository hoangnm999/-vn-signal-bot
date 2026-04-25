"""
alert_system.py — Proactive Price Alert System cho VN Signal Bot

Chức năng:
  1. Bảng `alerts` trong PostgreSQL (schema + helpers)
  2. add_alert()       — đặt cảnh báo giá mới
  3. remove_alert()    — hủy cảnh báo theo ID
  4. list_alerts()     — liệt kê cảnh báo đang active
  5. check_and_fire()  — kiểm tra giá & gửi Telegram (dùng trong cron 15 phút)

Tích hợp vào bot.py:
  - CommandHandler("/alert", alert_cmd)
  - CommandHandler("/alerts", alerts_cmd)
  - asyncio.create_task(_start_alert_cron(app)) trong post_init
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

ALERTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id           SERIAL       PRIMARY KEY,
    chat_id      VARCHAR(30)  NOT NULL,
    symbol       VARCHAR(10)  NOT NULL,
    target_price FLOAT        NOT NULL,
    direction    VARCHAR(5)   NOT NULL CHECK (direction IN ('above', 'below')),
    created_at   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    is_active    BOOLEAN      NOT NULL DEFAULT TRUE,
    fired_at     TIMESTAMP WITH TIME ZONE,
    fired_price  FLOAT
);

CREATE INDEX IF NOT EXISTS idx_alerts_active ON alerts(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_alerts_chat   ON alerts(chat_id);
CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol);
"""

MAX_ALERTS_PER_USER = 10   # giới hạn để tránh spam
ALERT_CRON_INTERVAL = 900  # 15 phút (giây)


def init_alerts_schema() -> bool:
    """Tạo bảng alerts nếu chưa có. Gọi từ init_db() trong db.py."""
    try:
        from db import get_conn
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(ALERTS_SCHEMA_SQL)
            conn.commit()
            cur.close()
        finally:
            conn.close()
        logger.info("Alerts schema initialized OK")
        return True
    except Exception as e:
        logger.error(f"init_alerts_schema error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# CRUD
# ══════════════════════════════════════════════════════════════════════════════

def add_alert(chat_id: str, symbol: str,
              target_price: float, direction: str) -> dict:
    """
    Đặt cảnh báo giá mới.

    Returns:
        {"ok": True, "id": int, "msg": str}  — thành công
        {"ok": False, "msg": str}             — thất bại
    """
    symbol    = symbol.upper().strip()[:10]
    direction = direction.lower().strip()
    if direction not in ("above", "below"):
        return {"ok": False, "msg": f"direction phải là 'above' hoặc 'below', nhận: {direction}"}
    try:
        target_price = float(target_price)
    except (TypeError, ValueError):
        return {"ok": False, "msg": f"Gia khong hop le: {target_price}"}
    if target_price <= 0:
        return {"ok": False, "msg": "target_price phai > 0"}

    try:
        from db import get_conn
        conn = get_conn()
        try:
            cur = conn.cursor()
            # Kiểm tra giới hạn per-user
            cur.execute(
                "SELECT COUNT(*) FROM alerts WHERE chat_id=%s AND is_active=TRUE",
                (str(chat_id),)
            )
            row = cur.fetchone()
            active_count = int(row[0]) if row else 0
            if active_count >= MAX_ALERTS_PER_USER:
                return {
                    "ok": False,
                    "msg": (f"Ban dang co {active_count} canh bao active "
                            f"(toi da {MAX_ALERTS_PER_USER}). "
                            f"Dung /alerts de xem va huy bot.")
                }
            # Insert
            cur.execute(
                """
                INSERT INTO alerts (chat_id, symbol, target_price, direction)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (str(chat_id), symbol, float(target_price), direction)
            )
            row = cur.fetchone()
            alert_id = int(row[0]) if row else -1
            conn.commit()
            cur.close()
        finally:
            conn.close()

        dir_vn  = "vuot len tren" if direction == "above" else "giam xuong duoi"
        price_s = f"{target_price:,.0f}"
        return {
            "ok":  True,
            "id":  alert_id,
            "msg": (f"Da dat canh bao #{alert_id}: {symbol} khi gia {dir_vn} {price_s}.\n"
                    f"Dung /alerts de xem tat ca canh bao."),
        }
    except Exception as e:
        logger.error(f"add_alert error: {e}")
        return {"ok": False, "msg": f"Loi DB: {e}"}


def remove_alert(chat_id: str, alert_id: int) -> dict:
    """
    Hủy cảnh báo theo ID. Chỉ hủy được cảnh báo của chính mình.

    Returns:
        {"ok": True/False, "msg": str}
    """
    try:
        from db import get_conn
        conn = get_conn()
        try:
            cur = conn.cursor()
            # Kiểm tra ownership
            cur.execute(
                "SELECT symbol, target_price, direction, is_active "
                "FROM alerts WHERE id=%s AND chat_id=%s",
                (int(alert_id), str(chat_id))
            )
            row = cur.fetchone()
            if not row:
                return {"ok": False, "msg": f"Khong tim thay canh bao #{alert_id} cua ban."}
            symbol, price, direction, is_active = row
            if not is_active:
                return {"ok": False, "msg": f"Canh bao #{alert_id} da duoc huy truoc do."}

            cur.execute(
                "UPDATE alerts SET is_active=FALSE WHERE id=%s",
                (int(alert_id),)
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()

        dir_vn = "tren" if direction == "above" else "duoi"
        return {
            "ok": True,
            "msg": (f"Da huy canh bao #{alert_id}: "
                    f"{symbol} khi gia {dir_vn} {price:,.0f}."),
        }
    except Exception as e:
        logger.error(f"remove_alert error: {e}")
        return {"ok": False, "msg": f"Loi DB: {e}"}


def list_alerts(chat_id: str) -> str:
    """
    Trả về danh sách cảnh báo đang active của user.
    """
    try:
        from db import get_conn
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, symbol, target_price, direction, created_at
                FROM   alerts
                WHERE  chat_id=%s AND is_active=TRUE
                ORDER  BY created_at DESC
                """,
                (str(chat_id),)
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()

        if not rows:
            return ("Ban chua co canh bao nao dang hoat dong.\n"
                    "Dat canh bao: /alert STB 68000 [above|below]")

        lines = [f"CANH BAO GIA DANG ACTIVE ({len(rows)}/{MAX_ALERTS_PER_USER})",
                 "─" * 32]
        for row in rows:
            aid, symbol, price, direction, created_at = row
            dir_icon = "⬆️" if direction == "above" else "⬇️"
            dir_vn   = "vuot tren" if direction == "above" else "giam duoi"
            try:
                price_f = float(price) if price is not None else 0.0
            except (TypeError, ValueError):
                price_f = 0.0
            if created_at and hasattr(created_at, "strftime"):
                dt_s = created_at.strftime("%d/%m %H:%M")
            elif created_at:
                dt_s = str(created_at)[:16]
            else:
                dt_s = "N/A"
            lines.append(
                f"{dir_icon} #{aid}  {symbol}  {dir_vn}  {price_f:,.0f}  [{dt_s}]"
            )
        lines.append("─" * 32)
        lines.append("Huy canh bao: /alert cancel <ID>")
        return "\n".join(lines)

    except Exception as e:
        logger.error(f"list_alerts error: {e}")
        return f"Loi khi lay danh sach canh bao: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# CRON — check giá và fire alerts
# ══════════════════════════════════════════════════════════════════════════════

def _get_active_alerts() -> list[dict]:
    """Lấy tất cả alerts đang active từ DB."""
    try:
        from db import get_conn
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, chat_id, symbol, target_price, direction
                FROM   alerts
                WHERE  is_active = TRUE
                ORDER  BY symbol
                """
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()
        return [
            {"id": r[0], "chat_id": r[1], "symbol": r[2],
             "target_price": float(r[3]), "direction": r[4]}
            for r in rows
        ]
    except Exception as e:
        logger.error(f"_get_active_alerts error: {e}")
        return []


def _get_current_prices(symbols: list[str]) -> dict[str, float]:
    """
    Lấy giá hiện tại cho nhiều mã.
    Dùng get_price_data từ analyzer.py (Entrade).
    Trả về {symbol: price}.
    """
    from analyzer import get_price_data
    prices = {}
    for sym in symbols:
        try:
            result = get_price_data(sym, 5)   # chỉ cần 5 ngày gần nhất
            if result.get("success"):
                df = result["df"]
                prices[sym] = float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning(f"_get_current_prices {sym}: {e}")
    return prices


def _fire_alert(alert_id: int, fired_price: float) -> bool:
    """Đánh dấu alert đã fired trong DB."""
    try:
        from db import get_conn
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE alerts
                SET    is_active=FALSE,
                       fired_at=NOW(),
                       fired_price=%s
                WHERE  id=%s
                """,
                (float(fired_price), int(alert_id))
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()
        return True
    except Exception as e:
        logger.error(f"_fire_alert {alert_id}: {e}")
        return False


def _build_fire_message(symbol: str, target_price: float,
                        direction: str, current_price: float) -> str:
    """Build tin nhắn Telegram khi alert triggered."""
    dir_vn   = f"vuot tren {target_price:,.0f}" if direction == "above" \
               else f"giam duoi {target_price:,.0f}"
    dir_icon = "🚨📈" if direction == "above" else "🚨📉"
    diff_pct = (current_price - target_price) / target_price * 100
    return (
        f"{dir_icon} CANH BAO GIA: {symbol}\n"
        f"Gia hien tai: {current_price:,.0f} ({diff_pct:+.1f}%)\n"
        f"Da {dir_vn}\n"
        f"─────────────────────\n"
        f"Goi y: /check {symbol} de phan tich day du"
    )


async def check_and_fire(bot) -> int:
    """
    Lấy tất cả alerts active → kiểm tra giá → fire nếu điều kiện thỏa.
    Gọi trong cron loop mỗi 15 phút.

    Args:
        bot: telegram.Bot instance (từ application.bot)

    Returns:
        Số alerts đã fire.
    """
    alerts = _get_active_alerts()
    if not alerts:
        return 0

    # Group symbols để giảm API calls
    symbols = list({a["symbol"] for a in alerts})
    prices  = await asyncio.to_thread(_get_current_prices, symbols)

    if not prices:
        logger.warning("check_and_fire: không lấy được giá nào")
        return 0

    fired = 0
    for alert in alerts:
        sym    = alert["symbol"]
        price  = prices.get(sym)
        if price is None:
            continue

        target    = alert["target_price"]
        direction = alert["direction"]
        triggered = (
            (direction == "above" and price >= target) or
            (direction == "below" and price <= target)
        )
        if not triggered:
            continue

        # Fire: gửi Telegram + update DB
        msg = _build_fire_message(sym, target, direction, price)
        try:
            await bot.send_message(chat_id=alert["chat_id"], text=msg)
            _fire_alert(alert["id"], price)
            fired += 1
            logger.info(f"Alert fired: #{alert['id']} {sym} "
                        f"{direction} {target} | actual={price}")
        except Exception as e:
            logger.error(f"check_and_fire send failed alert #{alert['id']}: {e}")

    return fired


# ══════════════════════════════════════════════════════════════════════════════
# CRON LOOP — chạy trong bot.py
# ══════════════════════════════════════════════════════════════════════════════

async def _start_alert_cron(application) -> None:
    """
    Vòng lặp cron kiểm tra alerts mỗi ALERT_CRON_INTERVAL giây.
    Bắt đầu 30s sau khi bot start (để bot init xong).

    Tích hợp vào bot.py:
        async def post_init(application):
            asyncio.create_task(_start_cron())
            asyncio.create_task(_start_alert_cron(application))  # ← thêm dòng này
    """
    await asyncio.sleep(30)   # đợi bot init xong
    logger.info("Alert cron started (interval=15min)")

    while True:
        try:
            fired = await check_and_fire(application.bot)
            if fired:
                logger.info(f"Alert cron: {fired} alerts fired")
        except Exception as e:
            logger.error(f"Alert cron error: {e}")

        await asyncio.sleep(ALERT_CRON_INTERVAL)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM HANDLERS — thêm vào bot.py
# ══════════════════════════════════════════════════════════════════════════════

async def alert_cmd(update, context):
    """
    Handler cho lệnh /alert.

    Cú pháp:
        /alert STB 68000           → above (mặc định)
        /alert STB 62000 below     → below
        /alert cancel 5            → hủy alert #5
        /alert cancel all          → hủy tất cả alerts của mình
    """
    # Import ở đây để tránh circular import
    from bot import is_allowed, _deny, _validate_symbol

    if not is_allowed(update):
        await _deny(update); return

    chat_id = str(update.effective_chat.id)
    args    = context.args or []

    # ── /alert cancel <id|all> ────────────────────────────────────────────────
    if args and args[0].lower() == "cancel":
        if len(args) < 2:
            await update.message.reply_text(
                "Cu phap: /alert cancel <ID> hoac /alert cancel all\n"
                "Vi du:   /alert cancel 5"
            )
            return

        if args[1].lower() == "all":
            # Hủy tất cả alerts của user này
            try:
                from db import get_conn
                conn = get_conn()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE alerts SET is_active=FALSE "
                        "WHERE chat_id=%s AND is_active=TRUE",
                        (chat_id,)
                    )
                    n = cur.rowcount
                    conn.commit()
                    cur.close()
                finally:
                    conn.close()
                await update.message.reply_text(
                    f"Da huy {n} canh bao." if n > 0 else "Ban khong co canh bao nao dang active."
                )
            except Exception as e:
                await update.message.reply_text(f"Loi: {e}")
            return

        try:
            alert_id = int(args[1])
        except ValueError:
            await update.message.reply_text("ID phai la so nguyen. Vi du: /alert cancel 5")
            return

        result = remove_alert(chat_id, alert_id)
        await update.message.reply_text(
            ("✅ " if result["ok"] else "❌ ") + result["msg"]
        )
        return

    # ── /alert <symbol> <price> [above|below] ────────────────────────────────
    if len(args) < 2:
        await update.message.reply_text(
            "Cu phap:\n"
            "  /alert STB 68000          → canh bao khi vuot tren 68,000\n"
            "  /alert STB 62000 below    → canh bao khi giam duoi 62,000\n"
            "  /alert cancel <ID>        → huy canh bao\n"
            "  /alert cancel all         → huy tat ca\n"
            "  /alerts                   → xem danh sach canh bao"
        )
        return

    valid, symbol = _validate_symbol(args[0])
    if not valid:
        await update.message.reply_text(f"Ma co phieu khong hop le: {args[0]}")
        return

    try:
        target_price = float(args[1].replace(",", ""))
    except ValueError:
        await update.message.reply_text(
            f"Gia khong hop le: {args[1]}\n"
            "Vi du dung: /alert STB 68000"
        )
        return

    direction = "above"
    if len(args) >= 3 and args[2].lower() in ("below", "duoi", "b"):
        direction = "below"

    result = add_alert(chat_id, symbol, target_price, direction)
    dir_vn  = "vuot len tren" if direction == "above" else "giam xuong duoi"
    if result["ok"]:
        await update.message.reply_text(
            f"✅ Da dat canh bao #{result['id']}: "
            f"{symbol} khi gia {dir_vn} {target_price:,.0f}\n"
            f"Bot se thong bao trong vong 15 phut khi dieu kien xay ra.\n"
            f"/alerts de xem tat ca canh bao."
        )
    else:
        await update.message.reply_text(f"❌ {result['msg']}")


async def alerts_cmd(update, context):
    """
    Handler cho lệnh /alerts — xem danh sách cảnh báo đang active.
    """
    from bot import is_allowed, _deny
    if not is_allowed(update):
        await _deny(update); return

    chat_id = str(update.effective_chat.id)
    result  = list_alerts(chat_id)
    await update.message.reply_text(result)
