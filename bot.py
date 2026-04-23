import os
import logging
import asyncio
import re
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import Conflict, NetworkError, TimedOut
from analyzer import (analyze_stock_full, scan_watchlist,
                      get_price_data, get_market_data, get_news_data)
from db import init_db, save_signal, run_evaluation_cron, get_report, get_history
try:
    from vibe_client import (
        is_available as vibe_available,
        start_swarm, poll_swarm, format_swarm_result,
        SWARM_ALIASES, SWARM_LABELS, SWARM_GROUPS,
    )
    _VIBE_CLIENT = True
except ImportError:
    _VIBE_CLIENT = False

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
# ── /vibe — Gọi Vibe-Trading swarm agents thật sự ────────────────────────────
# Lệnh: /vibe <MA> [preset]
# Preset: technical (mặc định), investment, risk, macro, fundamental, sector
# Ví dụ: /vibe VCB
#         /vibe HPG investment
#         /vibe FPT risk

async def vibe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Chạy Vibe-Trading swarm agents (29 swarms / 113 agents)."""
    if not is_allowed(update): await _deny(update); return

    if not _VIBE_CLIENT:
        await update.message.reply_text(
            "vibe_client.py chua duoc deploy.\n"
            "Them file vibe_client.py vao repo va set VIBE_API_URL."
        ); return

    # Nếu không có args → hiện menu
    if not context.args:
        lines = ["VIBE-TRADING — 29 Swarms / 113 Agents", ""]
        for group, aliases in SWARM_GROUPS.items():
            lines.append(f"{group}:")
            for a in aliases:
                label = SWARM_LABELS.get(a, a)
                lines.append(f"  /vibe <MA> {a:<18} {label}")
            lines.append("")
        lines.append("Vi du: /vibe VCB technical")
        lines.append("       /vibe HPG investment")
        await update.message.reply_text("\n".join(lines))
        return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le (VD: VCB, HPG, FPT)."); return

    alias = context.args[1].lower() if len(context.args) > 1 else "technical"
    if alias not in SWARM_ALIASES:
        close = [a for a in SWARM_ALIASES if alias in a]
        tip = f"Y ban noi: {', '.join(close[:3])}?" if close else ""
        await update.message.reply_text(
            f"Swarm '{alias}' khong ton tai. {tip}\n"
            f"Goi /vibe de xem danh sach 29 swarms."
        ); return

    if not vibe_available():
        await update.message.reply_text(
            "Vibe-Trading service OFFLINE.\n"
            "Kiem tra /vibestatus"
        ); return

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    wait = _check_rate_limit(user_id)
    if wait > 0:
        await update.message.reply_text(f"Vui long cho {wait:.0f}s."); return
    _record_cmd(user_id)

    label   = SWARM_LABELS.get(alias, alias)
    chat_id = update.effective_chat.id
    msg = await update.message.reply_text(
        f"Vibe-Trading: {label}\n"
        f"Ma: {symbol} | Dang khoi dong...\n"
        f"(Co the mat 5-15 phut, bot van nhan lenh khac trong khi cho)"
    )
    msg_id = msg.message_id

    # Chạy swarm trong background task — KHÔNG block event loop
    async def _run_swarm_bg():
        try:
            # Start swarm
            run_id = await asyncio.to_thread(start_swarm, alias, symbol)
            if not run_id:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg_id,
                    text=f"Loi: Khong khoi dong duoc swarm {alias}.\n"
                         f"Kiem tra VIBE_API_KEY."
                )
                return

            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg_id,
                text=f"Vibe-Trading: {label}\n"
                     f"Ma: {symbol} | Run ID: {run_id[:12]}...\n"
                     f"Agents dang khoi dong..."
            )

            # Poll với progress update định kỳ
            start_t = time.time()
            last_update = 0

            while True:
                elapsed = time.time() - start_t
                if elapsed > 900:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg_id,
                        text=f"Timeout: Swarm chay qua 15 phut.\nThu lai sau."
                    )
                    return

                # Poll 1 lần
                try:
                    r = await asyncio.to_thread(
                        lambda: __import__('requests').get(
                            f"{__import__('vibe_client').VIBE_API_URL}/swarm/runs/{run_id}",
                            headers={"Authorization": f"Bearer {__import__('os').environ.get('VIBE_API_KEY','')}",
                                     "Content-Type": "application/json"},
                            timeout=15
                        )
                    )
                    data = r.json()
                except Exception as e:
                    logger.warning(f"poll error: {e}")
                    await asyncio.sleep(8)
                    continue

                status  = data.get("status", "")
                tasks   = data.get("tasks", [])
                n       = len(tasks)
                done    = sum(1 for t in tasks if t.get("status") in
                              ("completed","failed","cancelled"))
                in_prog = [t.get("agent_id", t.get("id","?")) for t in tasks
                           if t.get("status") == "in_progress"]

                # Update message mỗi 15s
                if time.time() - last_update > 15:
                    last_update = time.time()
                    prog = f"Agents: {done}/{n} xong ({int(elapsed)}s)"
                    if in_prog: prog += f"\nDang chay: {', '.join(in_prog[:3])}"
                    try:
                        await context.bot.edit_message_text(
                            chat_id=chat_id, message_id=msg_id,
                            text=f"Vibe-Trading: {label}\nMa: {symbol}\n\n{prog}"
                        )
                    except Exception:
                        pass

                if status == "completed":
                    final_report = data.get("final_report", "")
                    if not final_report and tasks:
                        # Lấy output của agent cuối
                        done_tasks = [t for t in tasks if t.get("status")=="completed"]
                        if done_tasks:
                            last_t = done_tasks[-1]
                            final_report = f"[{last_t.get('agent_id','?')}]\n{last_t.get('summary','')}"

                    header = (f"VIBE-TRADING — {label}\n"
                              f"Ma: {symbol} | {done}/{n} agents | "
                              f"{int(elapsed)//60}p{int(elapsed)%60}s\n"
                              f"{'='*32}\n\n")
                    body = (final_report or "Khong co bao cao.")[:3800 - len(header)]
                    output = header + body
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg_id,
                        text=output[:4000]
                    )
                    return

                if status in ("failed", "cancelled"):
                    failed = [t for t in tasks if t.get("status")=="failed"]
                    err = failed[0].get("error","")[:100] if failed else ""
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg_id,
                        text=f"Swarm {status}: {label}\nMa: {symbol}\n"
                             f"Loi: {err or 'Kiem tra Vibe-Trading logs'}"
                    )
                    return

                # Nếu tasks rỗng và < 60s → đang init
                if n == 0 and elapsed < 60:
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(8)

        except Exception as e:
            logger.error(f"_run_swarm_bg error: {e}")
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg_id,
                    text=f"Loi khi chay Vibe-Trading {alias}: {str(e)[:200]}"
                )
            except Exception:
                pass

    # Tạo background task — không await, event loop tiếp tục nhận lệnh khác
    asyncio.create_task(_run_swarm_bg())


async def vibe_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kiểm tra Vibe-Trading service + liệt kê swarms."""
    if not is_allowed(update): await _deny(update); return

    if not _VIBE_CLIENT:
        await update.message.reply_text("vibe_client.py chua duoc cai dat."); return

    # Lấy URL đã normalize từ vibe_client
    try:
        from vibe_client import VIBE_API_URL as _vibe_url
    except Exception:
        _vibe_url = os.environ.get("VIBE_API_URL", "")

    msg = await update.message.reply_text("Dang kiem tra Vibe-Trading service...")

    online  = vibe_available()
    lines = [
        "Vibe-Trading Service:",
        f"  URL    : {_vibe_url or 'CHUA SET VIBE_API_URL'}",
        f"  Status : {'Online ✓' if online else 'OFFLINE ✗'}",
        f"  API Key: {'Co' if os.environ.get('VIBE_API_KEY') else 'Chua set (dev mode)'}",
        f"  Swarms : {len(SWARM_ALIASES)} presets / 113 agents",
    ]
    if not online:
        lines += [
            "",
            "Nguyen nhan co the:",
            "  1. VIBE_API_URL sai (phai co https://)",
            "  2. Vibe Railway service chua chay",
            "  3. Network/firewall block",
            "",
            "Dung /vibetest de debug chi tiet",
        ]
    else:
        lines += ["", "Dung: /vibe <MA> <swarm>", "Xem list: /vibe"]

    await msg.edit_text("\n".join(lines))


async def vibetest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug chi tiết kết nối Vibe-Trading — gọi /health và in raw response."""
    if not is_allowed(update): await _deny(update); return

    if not _VIBE_CLIENT:
        await update.message.reply_text("vibe_client.py chua duoc cai dat."); return

    try:
        from vibe_client import VIBE_API_URL as _vibe_url
    except Exception:
        _vibe_url = os.environ.get("VIBE_API_URL", "")

    msg = await update.message.reply_text(f"Testing: {_vibe_url}/health ...")

    lines = [f"Vibe-Trading Debug", f"URL: {_vibe_url}"]

    if not _vibe_url:
        await msg.edit_text("VIBE_API_URL chua duoc set tren bot service!"); return

    # Test 1: /health
    import requests as _req
    try:
        r = _req.get(f"{_vibe_url}/health", timeout=15)
        lines.append(f"GET /health → HTTP {r.status_code}")
        if r.status_code == 200:
            lines.append(f"Response: {str(r.json())[:100]}")
        else:
            lines.append(f"Body: {r.text[:100]}")
    except _req.exceptions.ConnectionError as e:
        lines.append(f"GET /health → ConnectionError")
        lines.append(f"  {str(e)[:120]}")
    except _req.exceptions.Timeout:
        lines.append(f"GET /health → Timeout (15s)")
    except Exception as e:
        lines.append(f"GET /health → {type(e).__name__}: {str(e)[:100]}")

    # Test 2: /swarm/presets với auth header
    api_key = os.environ.get("VIBE_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r2 = _req.get(f"{_vibe_url}/swarm/presets",
                      headers=headers, timeout=15)
        lines.append(f"GET /swarm/presets → HTTP {r2.status_code}")
        if r2.status_code == 200:
            data = r2.json()
            n = len(data) if isinstance(data, list) else "?"
            lines.append(f"  {n} presets available")
        elif r2.status_code == 401:
            lines.append("  → 401 Unauthorized: VIBE_API_KEY sai hoac thieu")
        else:
            lines.append(f"  Body: {r2.text[:80]}")
    except Exception as e:
        lines.append(f"GET /swarm/presets → {type(e).__name__}: {str(e)[:80]}")

    await msg.edit_text("\n".join(lines))


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler — log lỗi nhưng KHÔNG crash bot."""
    err = context.error
    if isinstance(err, Conflict):
        # 2 instances đang chạy — chờ instance cũ tự dừng, không crash
        logger.warning(
            "Telegram 409 Conflict — co 2 bot instances dang chay. "
            "Instance nay se tiep tuc sau khi instance cu dung."
        )
        return
    if isinstance(err, (NetworkError, TimedOut)):
        logger.warning(f"Network error (se tu retry): {err}")
        return
    # Lỗi khác → log đầy đủ
    logger.error(f"Unhandled exception: {err}", exc_info=err)


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

    app.add_handler(CommandHandler("start",        start))
    app.add_handler(CommandHandler("help",         help_cmd))
    app.add_handler(CommandHandler("watchlist",    watchlist_cmd))
    app.add_handler(CommandHandler("add",          add_cmd))
    app.add_handler(CommandHandler("remove",       remove_cmd))
    app.add_handler(CommandHandler("status",       status_cmd))
    app.add_handler(CommandHandler("debug",        debug_cmd))
    app.add_handler(CommandHandler("check",        check_cmd))
    app.add_handler(CommandHandler("scan",         scan_cmd))
    app.add_handler(CommandHandler("report",       report_cmd))
    app.add_handler(CommandHandler("history",      history_cmd))
    app.add_handler(CommandHandler("vibe",         vibe_cmd))
    app.add_handler(CommandHandler("vibestatus",   vibe_status_cmd))
    app.add_handler(CommandHandler("vibetest",     vibetest_cmd))
    app.add_error_handler(_error_handler)

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
