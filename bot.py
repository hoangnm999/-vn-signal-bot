import os
import logging
import asyncio
import re
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from analyzer import analyze_stock, scan_watchlist

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN")
ALLOWED_CHAT_ID = os.environ.get("CHAT_ID", "")
# Nếu ALLOWED_CHAT_IDS có nhiều ID cách nhau dấu phẩy → multi-user
ALLOWED_IDS = {
    cid.strip()
    for cid in os.environ.get("ALLOWED_CHAT_IDS", ALLOWED_CHAT_ID).split(",")
    if cid.strip()
}


def is_allowed(update: Update) -> bool:
    """Cho phép tất cả nếu ALLOWED_CHAT_IDS không được set, ngược lại check whitelist"""
    if not ALLOWED_IDS or ALLOWED_IDS == {""}:
        return True  # Không giới hạn
    return str(update.effective_chat.id) in ALLOWED_IDS


def escape_md(text: str) -> str:
    """
    Escape các ký tự Markdown v1 trong nội dung do AI tạo ra.
    Chỉ escape trong phần plain-text, không đụng đến các tag *bold* và `code`
    mà bot tự tạo.
    """
    if not text:
        return ""
    # Chỉ escape dấu _ và [ ] ( ) ~ > # + - = | { } . !
    # vì * và ` đã được dùng có chủ ý trong format output
    chars = r"_[]()~>#+=|{}.!"
    for ch in chars:
        text = text.replace(ch, f"\\{ch}")
    return text


def safe_md(text: str) -> str:
    """
    Làm sạch toàn bộ message trước khi gửi Telegram với parse_mode=Markdown.
    Chiến lược: bỏ parse_mode, gửi plain text — đơn giản và không bao giờ lỗi.
    """
    return text  # dùng với parse_mode=None bên dưới


async def send_safe(update_or_msg, text: str, parse_mode=None):
    """Gửi message, tự động fallback sang plain text nếu parse lỗi"""
    try:
        if hasattr(update_or_msg, "edit_text"):
            await update_or_msg.edit_text(text, parse_mode=parse_mode)
        else:
            await update_or_msg.message.reply_text(text, parse_mode=parse_mode)
    except Exception:
        # Fallback: strip tất cả markdown, gửi plain text
        plain = re.sub(r"[*_`\[\]]", "", text)
        try:
            if hasattr(update_or_msg, "edit_text"):
                await update_or_msg.edit_text(plain)
            else:
                await update_or_msg.message.reply_text(plain)
        except Exception as e2:
            logger.error(f"Không gửi được message: {e2}")


# ── Command handlers ──────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await update.message.reply_text("⛔ Bạn không có quyền truy cập bot này.")
        return
    msg = (
        "🤖 VN Signal Bot — Sẵn sàng!\n\n"
        "📋 DANH SÁCH LỆNH:\n\n"
        "/check <MÃ>  —  Phân tích sâu 1 mã (6 agents AI)\n"
        "    Ví dụ: /check VCB\n\n"
        "/scan  —  Quét nhanh toàn bộ watchlist\n\n"
        "/watchlist  —  Xem danh sách mã đang theo dõi\n\n"
        "/add <MÃ>  —  Thêm mã vào watchlist phiên này\n"
        "    Ví dụ: /add HPG\n\n"
        "/remove <MÃ>  —  Xoá mã khỏi watchlist phiên này\n\n"
        "/status  —  Kiểm tra trạng thái bot và API\n\n"
        "/help  —  Hiển thị hướng dẫn chi tiết\n\n"
        "/start  —  Hiển thị menu này"
    )
    await update.message.reply_text(msg)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await update.message.reply_text("⛔ Bạn không có quyền truy cập bot này.")
        return
    msg = (
        "📖 HƯỚNG DẪN VN SIGNAL BOT\n"
        "══════════════════════════\n\n"
        "🔍 /check <MÃ>\n"
        "Phân tích chuyên sâu 1 mã cổ phiếu với 6 agents AI độc lập:\n"
        "  • Agent 1: Xu hướng giá (MA, RSI, MACD)\n"
        "  • Agent 2: Phân tích Volume\n"
        "  • Agent 3: Đánh giá Rủi ro (Bollinger Bands)\n"
        "  • Agent 4: Fundamental (PE, ROE, EPS)\n"
        "  • Agent 5: Smart Money (dòng tiền ngoại)\n"
        "  • Agent 6: News Sentiment (tin tức đa nguồn)\n"
        "  • Agent 7: Market Regime (VN-Index trend)\n"
        "  → Verdict tổng hợp: ĐỒNG THUẬN MUA / BÁN / TRUNG LẬP / PHẢN BÁC\n"
        "Thời gian: ~1-2 phút\n\n"
        "📋 /scan\n"
        "Quét nhanh toàn bộ watchlist, hiển thị:\n"
        "  RSI, Volume ratio, % thay đổi 1 tuần\n"
        "  Tín hiệu: Quá mua / Quá bán / Bình thường\n"
        "Thời gian: ~30 giây\n\n"
        "📊 /watchlist\n"
        "Xem danh sách mã đang theo dõi\n\n"
        "➕ /add <MÃ>\n"
        "Thêm mã vào watchlist cho phiên hiện tại\n\n"
        "➖ /remove <MÃ>\n"
        "Xoá mã khỏi watchlist cho phiên hiện tại\n\n"
        "🔧 /status\n"
        "Kiểm tra trạng thái kết nối DeepSeek, Gemini, vnstock\n\n"
        "══════════════════════════\n"
        "⚠️ Lưu ý: Bot là công cụ hỗ trợ phân tích,\n"
        "không phải khuyến nghị đầu tư chính thức."
    )
    await update.message.reply_text(msg)


async def watchlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    wl = _get_watchlist(context)
    symbols = "  •  ".join(wl)
    await update.message.reply_text(f"📋 Watchlist ({len(wl)} mã):\n{symbols}")


async def add_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    if not context.args:
        await update.message.reply_text("⚠️ Dùng: /add VCB")
        return
    symbol = context.args[0].upper().strip()
    wl = _get_watchlist(context)
    if symbol not in wl:
        wl.append(symbol)
        context.bot_data["watchlist"] = wl
        await update.message.reply_text(f"✅ Đã thêm {symbol} vào watchlist")
    else:
        await update.message.reply_text(f"ℹ️ {symbol} đã có trong watchlist rồi")


async def remove_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    if not context.args:
        await update.message.reply_text("⚠️ Dùng: /remove VCB")
        return
    symbol = context.args[0].upper().strip()
    wl = _get_watchlist(context)
    if symbol in wl:
        wl.remove(symbol)
        context.bot_data["watchlist"] = wl
        await update.message.reply_text(f"✅ Đã xoá {symbol} khỏi watchlist")
    else:
        await update.message.reply_text(f"ℹ️ {symbol} không có trong watchlist")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    import os
    lines = ["🔧 Trạng thái bot:\n"]
    lines.append(f"  DeepSeek API key: {'✅ Có' if os.environ.get('DEEPSEEK_API_KEY') else '❌ Chưa set'}")
    lines.append(f"  Gemini API key:   {'✅ Có' if os.environ.get('GEMINI_API_KEY') else '⚠️ Chưa set (fallback unavailable)'}")
    lines.append(f"  VNAI API key:     {'✅ Có' if os.environ.get('VNAI_API_KEY') else '⚠️ Chưa set'}")
    wl = _get_watchlist(context)
    lines.append(f"  Watchlist:        {len(wl)} mã")
    lines.append(f"  Allowed IDs:      {', '.join(ALLOWED_IDS) if ALLOWED_IDS else 'Tất cả'}")
    await update.message.reply_text("\n".join(lines))


async def check_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await update.message.reply_text("⛔ Bạn không có quyền dùng lệnh này.")
        return
    if not context.args:
        await update.message.reply_text("⚠️ Vui lòng nhập mã. Ví dụ: /check VCB")
        return

    symbol = context.args[0].upper().strip()
    msg = await update.message.reply_text(f"🔍 Đang phân tích {symbol}... (~1-2 phút)")

    try:
        result = await asyncio.to_thread(analyze_stock, symbol)
        # Gửi plain text để tránh lỗi Markdown parse
        plain = re.sub(r"[*`]", "", result)   # bỏ bold/code markers
        plain = plain.replace("\\_", "_")      # unescape nếu có
        await msg.edit_text(plain)
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        await msg.edit_text(f"❌ Lỗi khi phân tích {symbol}: {str(e)[:200]}")


async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await update.message.reply_text("⛔ Bạn không có quyền dùng lệnh này.")
        return
    wl = _get_watchlist(context)
    msg = await update.message.reply_text(
        f"🔍 Đang quét {len(wl)} mã trong watchlist...\n(~30 giây)"
    )
    try:
        result = await asyncio.to_thread(scan_watchlist, wl)
        plain = re.sub(r"[*`]", "", result)
        await msg.edit_text(plain)
    except Exception as e:
        logger.error(f"Scan error: {e}")
        await msg.edit_text(f"❌ Lỗi khi quét watchlist: {str(e)[:200]}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_watchlist(context) -> list:
    default = os.environ.get("WATCHLIST", "VCB,HPG,FPT,VNM,MWG,TCB").split(",")
    return context.bot_data.get("watchlist", default.copy())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN chưa được set trong environment variables")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",     start))
    app.add_handler(CommandHandler("help",      help_cmd))
    app.add_handler(CommandHandler("watchlist", watchlist_cmd))
    app.add_handler(CommandHandler("add",       add_cmd))
    app.add_handler(CommandHandler("remove",    remove_cmd))
    app.add_handler(CommandHandler("status",    status_cmd))
    app.add_handler(CommandHandler("check",     check_cmd))
    app.add_handler(CommandHandler("scan",      scan_cmd))

    logger.info("Bot đang chạy...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,   # Bỏ qua updates cũ khi restart — fix Conflict error
    )


if __name__ == "__main__":
    main()
