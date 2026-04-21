import os
import logging
import asyncio
import re
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from analyzer import (analyze_stock, scan_watchlist,
                       get_price_data, get_fundamental_data,
                       get_foreign_flow_data, get_market_data, get_news_data)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")


def is_allowed(update: Update) -> bool:
    return True


def _get_watchlist(context) -> list:
    default = os.environ.get("WATCHLIST", "VCB,HPG,FPT,VNM,MWG,TCB").split(",")
    return context.bot_data.get("watchlist", [s.strip() for s in default])


def plain(text: str) -> str:
    """Strip markdown markers để tránh Telegram parse error"""
    text = re.sub(r"[*`]", "", text)
    text = text.replace("\\_", "_")
    return text


# ── /start ────────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "VN Signal Bot — San sang!\n\n"
        "DANH SACH LENH:\n"
        "/check <MA>    — Phan tich sau 1 ma (6 agents AI)\n"
        "  Vi du: /check VCB\n\n"
        "/scan          — Quet nhanh toan bo watchlist\n"
        "/watchlist     — Xem danh sach ma dang theo doi\n"
        "/add <MA>      — Them ma vao watchlist\n"
        "/remove <MA>   — Xoa ma khoi watchlist\n"
        "/status        — Kiem tra trang thai bot & API keys\n"
        "/debug <MA>    — Debug tung data source (fundamental, foreign, market, news)\n"
        "/help          — Huong dan chi tiet\n"
        "/start         — Hien thi menu nay"
    )
    await update.message.reply_text(msg)


# ── /help ─────────────────────────────────────────────────────────────────────
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "HUONG DAN VN SIGNAL BOT\n"
        "========================\n\n"
        "/check <MA>\n"
        "Phan tich chuyen sau 1 ma voi 6 agents AI doc lap:\n"
        "  Agent 1: Xu huong gia (MA, RSI, MACD)\n"
        "  Agent 2: Phan tich Volume\n"
        "  Agent 3: Danh gia Rui ro (Bollinger Bands)\n"
        "  Agent 4: Fundamental (ROE, EPS, tang truong)\n"
        "  Agent 5: Smart Money (dong tien ngoai)\n"
        "  Agent 6: News Sentiment (tin tuc da nguon)\n"
        "  Agent 7: Market Regime (VN-Index trend)\n"
        "  Verdict: DONG THUAN MUA / BAN / TRUNG LAP / PHAN BAC\n"
        "  Thoi gian: ~1-2 phut\n\n"
        "/scan\n"
        "Quet nhanh toan bo watchlist:\n"
        "  RSI, Volume ratio, % thay doi 1 tuan\n"
        "  Tin hieu: Qua mua / Qua ban / Binh thuong\n"
        "  Thoi gian: ~30 giay\n\n"
        "/watchlist  — Xem danh sach theo doi\n"
        "/add <MA>   — Them ma (vi du: /add HPG)\n"
        "/remove <MA>— Xoa ma (vi du: /remove HPG)\n\n"
        "/status     — Kiem tra API keys con hoat dong khong\n\n"
        "/debug <MA> — Debug tung nguon du lieu:\n"
        "  Xem chinh xac Entrade tra ve gi cho fundamental,\n"
        "  foreign flow, market data, news\n"
        "  Dung khi mot agent bao 'khong lay duoc du lieu'\n\n"
        "========================\n"
        "Luu y: Bot la cong cu ho tro phan tich,\n"
        "khong phai khuyen nghi dau tu chinh thuc."
    )
    await update.message.reply_text(msg)


# ── /watchlist, /add, /remove ─────────────────────────────────────────────────
async def watchlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    wl = _get_watchlist(context)
    await update.message.reply_text(f"Watchlist ({len(wl)} ma): {', '.join(wl)}")


async def add_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Dung: /add VCB")
        return
    symbol = context.args[0].upper().strip()
    wl = _get_watchlist(context)
    if symbol not in wl:
        wl.append(symbol)
        context.bot_data["watchlist"] = wl
        await update.message.reply_text(f"Da them {symbol} vao watchlist")
    else:
        await update.message.reply_text(f"{symbol} da co trong watchlist roi")


async def remove_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Dung: /remove VCB")
        return
    symbol = context.args[0].upper().strip()
    wl = _get_watchlist(context)
    if symbol in wl:
        wl.remove(symbol)
        context.bot_data["watchlist"] = wl
        await update.message.reply_text(f"Da xoa {symbol} khoi watchlist")
    else:
        await update.message.reply_text(f"{symbol} khong co trong watchlist")


# ── /status ───────────────────────────────────────────────────────────────────
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["Trang thai bot:\n"]
    lines.append(f"  DeepSeek API : {'Co' if os.environ.get('DEEPSEEK_API_KEY') else 'CHUA SET'}")
    lines.append(f"  Gemini API   : {'Co' if os.environ.get('GEMINI_API_KEY') else 'Chua set (fallback off)'}")
    lines.append(f"  VNAI API     : {'Co' if os.environ.get('VNAI_API_KEY') else 'Chua set'}")
    lines.append(f"  Watchlist    : {len(_get_watchlist(context))} ma")
    await update.message.reply_text("\n".join(lines))


# ── /debug <MA> ───────────────────────────────────────────────────────────────
async def debug_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test từng data source, in raw response để xem lỗi chính xác"""
    if not context.args:
        await update.message.reply_text("Dung: /debug VCB")
        return

    symbol = context.args[0].upper().strip()
    msg = await update.message.reply_text(f"Debug {symbol}... (15-30 giay)")
    lines = [f"=== DEBUG {symbol} ===\n"]

    # 1. Price data
    try:
        t0 = time.time()
        price = get_price_data(symbol, 30)
        elapsed = round(time.time() - t0, 1)
        if price["success"]:
            df = price["df"]
            lines.append(f"PRICE: OK ({elapsed}s) | {len(df)} bars | cols: {list(df.columns)}")
            lines.append(f"  Last close: {df['close'].iloc[-1]:.1f}")
        else:
            lines.append(f"PRICE: FAIL | {price['error'][:80]}")
    except Exception as e:
        lines.append(f"PRICE: ERROR | {str(e)[:80]}")

    # 2. Fundamental
    try:
        t0 = time.time()
        fund = get_fundamental_data(symbol)
        elapsed = round(time.time() - t0, 1)
        if fund.get("success"):
            src = fund.get("source", "?")
            lines.append(f"FUND: OK ({elapsed}s) source={src}")
            lines.append(f"  PE={fund['pe']} PB={fund['pb']} ROE={fund['roe']}%")
            lines.append(f"  EPS={fund['eps']} RevG={fund['revenue_growth']}% ProfG={fund['profit_growth']}%")
        else:
            lines.append(f"FUND: FAIL | {fund.get('error','?')[:120]}")
    except Exception as e:
        lines.append(f"FUND: ERROR | {str(e)[:80]}")

    # 3. Foreign flow
    try:
        t0 = time.time()
        foreign = get_foreign_flow_data(symbol)
        elapsed = round(time.time() - t0, 1)
        if foreign.get("success"):
            lines.append(f"FOREIGN: OK ({elapsed}s) src={foreign.get('source','?')}")
            lines.append(f"  Today={foreign['net_today']}B | 5D={foreign['net_5d']}B | 20D={foreign['net_20d']}B")
        else:
            lines.append(f"FOREIGN: FAIL | {foreign.get('error','?')[:120]}")
    except Exception as e:
        lines.append(f"FOREIGN: ERROR | {str(e)[:80]}")

    # 4. Market data (VNINDEX)
    try:
        t0 = time.time()
        market = get_market_data()
        elapsed = round(time.time() - t0, 1)
        if market.get("success"):
            lines.append(f"MARKET: OK ({elapsed}s)")
            lines.append(f"  VNINDEX={market['vnindex']} | 5D={market['change_5d']}% | AboveMA20={market['above_ma20']}")
        else:
            lines.append(f"MARKET: FAIL | {market.get('error','?')[:120]}")
    except Exception as e:
        lines.append(f"MARKET: ERROR | {str(e)[:80]}")

    # 5. News
    try:
        t0 = time.time()
        news = get_news_data(symbol)
        elapsed = round(time.time() - t0, 1)
        if news.get("success"):
            src_summary = news.get("source_summary", {})
            working = {k: v for k, v in src_summary.items() if v > 0}
            lines.append(f"NEWS: OK ({elapsed}s) | total={news['total']}")
            lines.append(f"  Working sources: {working}")
            lines.append(f"  Sample: {news['headlines'][0][:80] if news['headlines'] else 'none'}")
        else:
            lines.append(f"NEWS: FAIL | {news.get('error','?')[:120]}")
    except Exception as e:
        lines.append(f"NEWS: ERROR | {str(e)[:80]}")

    result = "\n".join(lines)
    # Truncate nếu quá dài
    if len(result) > 3800:
        result = result[:3800] + "\n...[truncated]"

    try:
        await msg.edit_text(result)
    except Exception:
        await msg.edit_text(plain(result))


# ── /check ────────────────────────────────────────────────────────────────────
async def check_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Vi du: /check VCB")
        return
    symbol = context.args[0].upper().strip()
    msg = await update.message.reply_text(f"Dang phan tich {symbol}... (~1-2 phut)")
    try:
        result = await asyncio.to_thread(analyze_stock, symbol)
        result = plain(result)
        if len(result) > 4000:
            result = result[:3950] + "\n...[cat bot]"
        await msg.edit_text(result)
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        await msg.edit_text(f"Loi khi phan tich {symbol}: {str(e)[:200]}")


# ── /scan ─────────────────────────────────────────────────────────────────────
async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN chua duoc set")

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

    logger.info("Bot dang chay...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
