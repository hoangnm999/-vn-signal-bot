import os
import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from analyzer import analyze_stock, scan_watchlist

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
ALLOWED_CHAT_ID = os.environ.get("CHAT_ID")

def is_allowed(update: Update) -> bool:
    return str(update.effective_chat.id) == str(ALLOWED_CHAT_ID)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    msg = (
        "🤖 *VN Signal Bot* — Sẵn sàng!\n\n"
        "📋 *Lệnh có sẵn:*\n"
        "`/check <MÃ>` — Phân tích 1 mã cổ phiếu\n"
        "Ví dụ: `/check VCB`\n\n"
        "`/scan` — Quét toàn bộ watchlist\n\n"
        "`/watchlist` — Xem danh sách theo dõi\n\n"
        "`/help` — Hướng dẫn sử dụng"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    msg = (
        "📖 *Hướng dẫn VN Signal Bot*\n\n"
        "*Kết quả phân tích gồm:*\n"
        "• Xu hướng giá (trend)\n"
        "• Volume & thanh khoản\n"
        "• Các chỉ báo kỹ thuật (RSI, MACD, BB)\n"
        "• Mức hỗ trợ / kháng cự\n"
        "• Đánh giá: 🟢 ĐỒNG THUẬN / 🟡 TRUNG LẬP / 🔴 PHẢN BÁC\n\n"
        "*Lưu ý:* Bot là công cụ hỗ trợ, không phải khuyến nghị đầu tư."
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

async def watchlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    watchlist = os.environ.get("WATCHLIST", "VCB,HPG,FPT,VNM,MWG,TCB").split(",")
    symbols = " • ".join(watchlist)
    await update.message.reply_text(f"📋 *Watchlist hiện tại:*\n• {symbols}", parse_mode='Markdown')

async def check_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    if not context.args:
        await update.message.reply_text("⚠️ Vui lòng nhập mã cổ phiếu. Ví dụ: `/check VCB`", parse_mode='Markdown')
        return

    symbol = context.args[0].upper().strip()
    msg = await update.message.reply_text(f"🔍 Đang phân tích *{symbol}*...", parse_mode='Markdown')

    try:
        result = await asyncio.to_thread(analyze_stock, symbol)
        await msg.edit_text(result, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        await msg.edit_text(f"❌ Lỗi khi phân tích {symbol}: {str(e)}")

async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return
    watchlist = os.environ.get("WATCHLIST", "VCB,HPG,FPT,VNM,MWG,TCB").split(",")
    msg = await update.message.reply_text(f"🔍 Đang quét {len(watchlist)} mã trong watchlist...\nQuá trình này mất 1-2 phút.")

    try:
        result = await asyncio.to_thread(scan_watchlist, watchlist)
        await msg.edit_text(result, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Scan error: {e}")
        await msg.edit_text(f"❌ Lỗi khi quét watchlist: {str(e)}")

def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN chưa được set")
    if not ALLOWED_CHAT_ID:
        raise ValueError("CHAT_ID chưa được set")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("watchlist", watchlist_cmd))
    app.add_handler(CommandHandler("check", check_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))

    logger.info("Bot đang chạy...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
