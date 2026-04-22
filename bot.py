import os
import logging
import asyncio
import re
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from analyzer import (analyze_stock_full, scan_watchlist,
                      get_price_data, get_market_data, get_news_data)
from db import init_db, save_signal, run_evaluation_cron, get_report, get_history

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

# ── Authorization ─────────────────────────────────────────────────────────────
def _allowed_ids() -> set:
    """
    Tập hợp chat_id/user_id được phép dùng bot.
    Set env var:
      CHAT_ID = "123456789"           (user chính)
      ALLOWED_CHAT_IDS = "111,222"    (thêm user/group khác, tùy chọn)
    """
    ids = set()
    main_id = os.environ.get("CHAT_ID", "").strip()
    if main_id:
        ids.add(main_id)
    for part in os.environ.get("ALLOWED_CHAT_IDS", "").split(","):
        part = part.strip()
        if part:
            ids.add(part)
    return ids


def is_allowed(update: Update) -> bool:
    """Chỉ cho phép chat_id trong whitelist. Từ chối tất cả nếu chưa cấu hình."""
    allowed = _allowed_ids()
    if not allowed:
        return False
    user_id = str(update.effective_user.id) if update.effective_user else ""
    chat_id = str(update.effective_chat.id) if update.effective_chat else ""
    return user_id in allowed or chat_id in allowed


async def _deny(update: Update):
    await update.message.reply_text("Khong co quyen su dung bot nay.")


# ── Rate limiting ─────────────────────────────────────────────────────────────
_last_heavy_cmd: dict[str, float] = {}
HEAVY_COOLDOWN_SECS = 30  # /check và /scan tốn API, giới hạn 1 lệnh/30s per user


def _check_rate_limit(user_id: str) -> float:
    """Trả về giây còn lại nếu đang cooldown, 0.0 nếu OK."""
    elapsed = time.time() - _last_heavy_cmd.get(user_id, 0)
    return max(0.0, HEAVY_COOLDOWN_SECS - elapsed)


def _record_cmd(user_id: str):
    _last_heavy_cmd[user_id] = time.time()


# ── Input validation ──────────────────────────────────────────────────────────
def _validate_symbol(raw: str) -> tuple[bool, str]:
    """
    Validate mã cổ phiếu: chỉ chấp nhận 2-10 ký tự chữ/số.
    Trả về (is_valid, cleaned_symbol).
    """
    s = raw.upper().strip()[:10]
    if not re.match(r'^[A-Z0-9]{2,10}$', s):
        return False, s
    return True, s


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_watchlist(context) -> list:
    default = [s.strip() for s in
               os.environ.get("WATCHLIST", "VCB,HPG,FPT,VNM,MWG,TCB").split(",")]
    return context.bot_data.get("watchlist", default)


def plain(text: str) -> str:
    """Strip markdown để tránh Telegram parse error."""
    text = re.sub(r"[*`]", "", text)
    text = text.replace("\\_", "_")
    return text


# ── /start ────────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await _deny(update); return
    msg = (
        "VN Signal Bot — San sang!\n\n"
        "DANH SACH LENH:\n"
        "/check <MA>    — Phan tich sau 1 ma (6 agents rule-based)\n"
        "  Vi du: /check VCB\n\n"
        "/scan          — Quet nhanh watchlist + Top 5 Vol Spike VN30\n"
        "/watchlist     — Xem danh sach ma dang theo doi\n"
        "/add <MA>      — Them ma vao watchlist (toi da 20 ma)\n"
        "/remove <MA>   — Xoa ma khoi watchlist\n"
        "/report [ngay] — Accuracy tung agent (mac dinh 30 ngay)\n"
        "/history <MA>  — Lich su signal cua 1 ma\n"
        "/status        — Kiem tra trang thai bot & API & DB\n"
        "/debug <MA>    — Debug tung data source\n"
        "/help          — Huong dan chi tiet\n"
        "/start         — Hien thi menu nay"
    )
    await update.message.reply_text(msg)


# ── /help ─────────────────────────────────────────────────────────────────────
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    msg = (
        "HUONG DAN VN SIGNAL BOT\n"
        "========================\n\n"
        "/check <MA>\n"
        "Phan tich 1 ma voi 6 agents rule-based (deterministic):\n"
        "  Agent 1: Xu huong gia (MA20/50, RSI, MACD)\n"
        "  Agent 2: Phan tich Volume (so voi TB20)\n"
        "  Agent 3: Danh gia Rui ro (BB, support/resistance)\n"
        "  Agent 4: News Sentiment (tu khoa RSS 10 nguon)\n"
        "  Agent 5: Market Regime (VN-Index proxy ETF)\n"
        "  Agent 6: Macro (lai suat, ty gia, SBV/Fed)\n"
        "  Ket luan: DONG THUAN/NGHIENG/TRUNG LAP + ACTION PLAN\n"
        "  Cooldown: 30 giay giua 2 lenh /check\n\n"
        "/scan\n"
        "Quet nhanh watchlist + Top 5 Vol Spike VN30:\n"
        "  RSI, Volume ratio, % thay doi 1 tuan\n\n"
        "/report [so_ngay]\n"
        "Thong ke do chinh xac tung agent:\n"
        "  Vi du: /report 60 (toi da 365 ngay)\n\n"
        "/history <MA>  — Lich su signal: /history VCB\n"
        "/watchlist     — Xem danh sach ma theo doi\n"
        "/add <MA>      — Them ma (toi da 20)\n"
        "/remove <MA>   — Xoa ma\n"
        "/status        — Kiem tra API + DB\n"
        "/debug <MA>    — Debug data source\n\n"
        "========================\n"
        "Luu y: Bot la cong cu ho tro phan tich,\n"
        "KHONG phai khuyen nghi dau tu chinh thuc."
    )
    await update.message.reply_text(msg)


# ── /watchlist, /add, /remove ─────────────────────────────────────────────────
async def watchlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    wl = _get_watchlist(context)
    await update.message.reply_text(f"Watchlist ({len(wl)} ma): {', '.join(wl)}")


async def add_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    if not context.args:
        await update.message.reply_text("Dung: /add VCB"); return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le (2-10 chu cai/so, VD: VCB)."); return

    wl = _get_watchlist(context)
    if len(wl) >= 20:
        await update.message.reply_text("Watchlist da day (toi da 20 ma). Xoa bot roi them."); return
    if symbol not in wl:
        wl.append(symbol)
        context.bot_data["watchlist"] = wl
        await update.message.reply_text(f"Da them {symbol} vao watchlist")
    else:
        await update.message.reply_text(f"{symbol} da co trong watchlist roi")


async def remove_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    if not context.args:
        await update.message.reply_text("Dung: /remove VCB"); return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le."); return

    wl = _get_watchlist(context)
    if symbol in wl:
        wl.remove(symbol)
        context.bot_data["watchlist"] = wl
        await update.message.reply_text(f"Da xoa {symbol} khoi watchlist")
    else:
        await update.message.reply_text(f"{symbol} khong co trong watchlist")


# ── /status ───────────────────────────────────────────────────────────────────
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    lines = ["Trang thai bot:\n"]
    lines.append(f"  DeepSeek API : {'Co' if os.environ.get('DEEPSEEK_API_KEY') else 'CHUA SET'}")
    lines.append(f"  Gemini API   : {'Co' if os.environ.get('GEMINI_API_KEY') else 'Chua set'}")
    lines.append(f"  VNAI API     : {'Co' if os.environ.get('VNAI_API_KEY') else 'Chua set'}")
    lines.append(f"  Fireant Token: {'Co' if os.environ.get('FIREANT_TOKEN') else 'Chua set'}")
    lines.append(f"  Watchlist    : {len(_get_watchlist(context))} ma")

    # Kiểm tra DB — KHÔNG expose DATABASE_URL
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        lines.append("  PostgreSQL   : CHUA SET DATABASE_URL")
    else:
        try:
            from db import get_conn
            conn = get_conn()
            conn.close()
            lines.append("  PostgreSQL   : Ket noi OK")
        except Exception as e:
            lines.append(f"  PostgreSQL   : LOI — {str(e)[:60]}")

    await update.message.reply_text("\n".join(lines))


# ── /debug <MA> ───────────────────────────────────────────────────────────────
async def debug_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    if not context.args:
        await update.message.reply_text("Dung: /debug VCB"); return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le."); return

    msg = await update.message.reply_text(f"Debug {symbol}... (15-30 giay)")
    lines = [f"=== DEBUG {symbol} ===\n"]

    # 1. Price data (Entrade)
    try:
        t0 = time.time()
        price = get_price_data(symbol, 30)
        elapsed = round(time.time() - t0, 1)
        if price["success"]:
            df = price["df"]
            lines.append(f"PRICE (Entrade): OK ({elapsed}s) | {len(df)} bars")
            lines.append(f"  Last close: {df['close'].iloc[-1]:.2f}")
        else:
            lines.append(f"PRICE: FAIL | {price['error'][:80]}")
    except Exception as e:
        lines.append(f"PRICE: ERROR | {str(e)[:80]}")

    # 2. Market data
    try:
        t0 = time.time()
        market = get_market_data()
        elapsed = round(time.time() - t0, 1)
        if market.get("success"):
            lines.append(f"MARKET: OK ({elapsed}s) proxy={market.get('proxy','?')}")
            lines.append(f"  VNINDEX={market['vnindex']} | 5D={market['change_5d']}%")
        else:
            lines.append(f"MARKET: FAIL | {market.get('error','?')[:80]}")
    except Exception as e:
        lines.append(f"MARKET: ERROR | {str(e)[:80]}")

    # 3. News (10 sources)
    try:
        t0 = time.time()
        news = get_news_data(symbol)
        elapsed = round(time.time() - t0, 1)
        if news.get("success"):
            src_summary = news.get("source_summary", {})
            working = [k for k, v in src_summary.items() if v > 0]
            lines.append(f"NEWS: OK ({elapsed}s) | total={news['total']}")
            lines.append(f"  Sources OK: {working}")
            if news['headlines']:
                lines.append(f"  Sample: {news['headlines'][0][:80]}")
        else:
            lines.append(f"NEWS: FAIL | {news.get('error','?')[:80]}")
    except Exception as e:
        lines.append(f"NEWS: ERROR | {str(e)[:80]}")

    result = "\n".join(lines)
    if len(result) > 3800:
        result = result[:3800] + "\n...[truncated]"
    try:
        await msg.edit_text(result)
    except Exception:
        await msg.edit_text(plain(result))


# ── /check ────────────────────────────────────────────────────────────────────
async def check_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    if not context.args:
        await update.message.reply_text("Vi du: /check VCB"); return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le (VD: VCB, HPG, FPT)."); return

    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    wait = _check_rate_limit(user_id)
    if wait > 0:
        await update.message.reply_text(f"Vui long cho {wait:.0f}s truoc khi /check tiep."); return

    _record_cmd(user_id)
    msg = await update.message.reply_text(f"Dang phan tich {symbol}... (~30-60 giay)")
    try:
        result, meta = await asyncio.to_thread(analyze_stock_full, symbol)
        result = plain(result)
        if len(result) > 4000:
            result = result[:3950] + "\n...[cat bot]"
        await msg.edit_text(result)

        if meta:
            try:
                sid = save_signal(
                    symbol,
                    meta["verdict"],
                    meta["ind"],
                    meta["agent_verdicts"],
                    meta["macro_v"],
                )
                if sid > 0:
                    logger.info(f"Signal saved: {symbol} id={sid}")
            except Exception as db_err:
                logger.warning(f"Khong luu duoc signal: {db_err}")

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        await msg.edit_text(f"Loi khi phan tich {symbol}: {str(e)[:200]}")


# ── /scan ─────────────────────────────────────────────────────────────────────
async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return

    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    wait = _check_rate_limit(user_id)
    if wait > 0:
        await update.message.reply_text(f"Vui long cho {wait:.0f}s."); return

    _record_cmd(user_id)
    wl = _get_watchlist(context)
    msg = await update.message.reply_text(f"Dang quet {len(wl)} ma... (~30 giay)")
    try:
        result = await asyncio.to_thread(scan_watchlist, wl)
        result = plain(result)
        if len(result) > 4000:
            result = result[:3950] + "\n...[cat bot]"
        await msg.edit_text(result)
    except Exception as e:
        logger.error(f"Scan error: {e}")
        await msg.edit_text(f"Loi khi quet watchlist: {str(e)[:200]}")


# ── /report ───────────────────────────────────────────────────────────────────
async def report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    msg = await update.message.reply_text("Dang tinh toan report...")
    try:
        days = 30
        if context.args:
            try:
                days = max(1, min(int(context.args[0]), 365))
            except ValueError:
                pass
        result = await asyncio.to_thread(get_report, days)
        if len(result) > 4000:
            result = result[:3950] + "\n...[cat bot]"
        await msg.edit_text(result)
    except Exception as e:
        await msg.edit_text(f"Loi khi lay report: {str(e)[:200]}")


# ── /history ──────────────────────────────────────────────────────────────────
async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    if not context.args:
        await update.message.reply_text("Vi du: /history VCB"); return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le."); return

    msg = await update.message.reply_text(f"Dang lay lich su {symbol}...")
    try:
        result = await asyncio.to_thread(get_history, symbol)
        if len(result) > 4000:
            result = result[:3950] + "\n...[cat bot]"
        await msg.edit_text(result)
    except Exception as e:
        await msg.edit_text(f"Loi khi lay history {symbol}: {str(e)[:200]}")


# ── Cron job 18:00 mỗi ngày ──────────────────────────────────────────────────
async def _start_cron():
    while True:
        import datetime as _dt
        now_dt = _dt.datetime.now()
        target = now_dt.replace(hour=18, minute=0, second=0, microsecond=0)
        if now_dt >= target:
            target += _dt.timedelta(days=1)
        wait_secs = (target - now_dt).total_seconds()
        logger.info(f"Cron: next run in {wait_secs/3600:.1f}h ({target.strftime('%d/%m %H:%M')})")
        await asyncio.sleep(wait_secs)
        try:
            updated = await asyncio.to_thread(run_evaluation_cron)
            logger.info(f"Cron xong: {updated} predictions da cham diem")
        except Exception as e:
            logger.error(f"Cron error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN chua duoc set")

    allowed = _allowed_ids()
    if not allowed:
        logger.warning("CANH BAO: CHAT_ID/ALLOWED_CHAT_IDS chua set — bot se tu choi tat ca!")
    else:
        logger.info(f"Authorization OK: {len(allowed)} id duoc phep")

    try:
        init_db()
        logger.info("DB initialized OK")
    except Exception as e:
        logger.warning(f"DB init failed: {e}")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",     start))
    app.add_handler(CommandHandler("help",      help_cmd))
    app.add_handler(CommandHandler("watchlist", watchlist_cmd))
    app.add_handler(CommandHandler("add",       add_cmd))
    app.add_handler(CommandHandler("remove",    remove_cmd))
    app.add_handler(CommandHandler("status",    status_cmd))
    app.add_handler(CommandHandler("debug",     debug_cmd))
    app.add_handler(CommandHandler("check",     check_cmd))
    app.add_handler(CommandHandler("scan",      scan_cmd))
    app.add_handler(CommandHandler("report",    report_cmd))
    app.add_handler(CommandHandler("history",   history_cmd))

    async def post_init(application):
        asyncio.create_task(_start_cron())

    app.post_init = post_init

    logger.info("Bot dang chay...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
