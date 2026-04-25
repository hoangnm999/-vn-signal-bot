"""
local_swarm_cmd.py — Telegram command handler cho /local_swarm.

Tích hợp:
  1. /local_swarm STB           → Chạy thủ công
  2. /local_swarm STB --check   → Chạy /check trước, rồi swarm (kết quả đầy đủ nhất)
  3. Failover từ /vibe: khi Vibe-Trading offline → tự động chạy local_swarm

Thêm vào bot.py:
    from local_swarm_cmd import local_swarm_cmd, install_vibe_failover
    app.add_handler(CommandHandler("local_swarm", local_swarm_cmd))
    install_vibe_failover(app)   # patch vibe_cmd để tự failover
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Cooldown độc lập với /check và /vibe
LOCAL_SWARM_COOLDOWN = 60
_last_local_swarm: dict[str, float] = {}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_rate_limit(user_id: str) -> float:
    elapsed = time.time() - _last_local_swarm.get(user_id, 0)
    return max(0.0, LOCAL_SWARM_COOLDOWN - elapsed)


def _record(user_id: str):
    _last_local_swarm[user_id] = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS STREAMER — cập nhật Telegram message realtime
# ══════════════════════════════════════════════════════════════════════════════

class _TelegramProgressStreamer:
    """
    Stream progress messages lên Telegram trong khi swarm đang chạy.
    Chạy trong thread pool nên dùng asyncio.run_coroutine_threadsafe.
    """

    def __init__(self, bot, chat_id: int, message_id: int, loop):
        self._bot        = bot
        self._chat_id    = chat_id
        self._msg_id     = message_id
        self._loop       = loop
        self._lines: list[str] = []
        self._last_edit  = 0.0
        self._min_interval = 4.0    # tối thiểu 4s giữa 2 lần edit

    def __call__(self, msg: str):
        """Được gọi từ thread của swarm."""
        self._lines.append(msg)
        now = time.time()
        if now - self._last_edit >= self._min_interval:
            self._last_edit = now
            self._do_edit()

    def _do_edit(self):
        recent = self._lines[-8:]   # 8 dòng gần nhất
        text = (
            "🤖 LOCAL SWARM ĐANG CHẠY...\n"
            "─" * 30 + "\n"
            + "\n".join(recent)
            + "\n\n(Đang xử lý, vui lòng đợi...)"
        )
        text = text[:4000]

        async def _edit():
            try:
                await self._bot.edit_message_text(
                    chat_id=self._chat_id,
                    message_id=self._msg_id,
                    text=text,
                )
            except Exception:
                pass

        asyncio.run_coroutine_threadsafe(_edit(), self._loop)

    def flush(self):
        """Gọi lần cuối trước khi thay thế bằng kết quả."""
        self._do_edit()


# ══════════════════════════════════════════════════════════════════════════════
# CORE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

async def _run_local_swarm_flow(
    symbol: str,
    run_check_first: bool,
    context,
    chat_id: int,
    msg,
):
    """
    Luồng chạy chính — chạy trong background task.

    run_check_first=True  → gọi analyze_stock_full() trước để có meta đầy đủ
    run_check_first=False → lấy meta từ DB hoặc khởi tạo minimal
    """
    loop = asyncio.get_event_loop()

    try:
        from local_swarm import run_local_swarm

        # ── Bước 1: Lấy dữ liệu phân tích ──────────────────────────────────
        meta = None
        if run_check_first:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=f"⏳ Đang chạy /check {symbol} trước...\n(30-60 giây)"
            )
            try:
                from analyzer import analyze_stock_full
                _, meta = await asyncio.to_thread(analyze_stock_full, symbol)
                if meta is None:
                    raise ValueError("analyze_stock_full trả về None")
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=f"✅ /check {symbol} xong. Đang khởi động Hội đồng...",
                )
            except Exception as e:
                logger.warning(f"local_swarm check fail: {e}")
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=(
                        f"⚠️ Không lấy được dữ liệu /check ({e}).\n"
                        f"Sẽ dùng kết quả gần nhất từ DB hoặc minimal analysis..."
                    ),
                )

        # Nếu không có meta từ /check, thử lấy từ DB
        if meta is None:
            try:
                from db import get_latest_meta
                meta = get_latest_meta(symbol)
                if meta:
                    logger.info(f"local_swarm: dùng meta từ DB cho {symbol}")
            except Exception:
                pass

        if meta is None:
            # Tạo minimal meta từ price data
            try:
                from analyzer import get_price_data, get_indicators
                price_r = await asyncio.to_thread(get_price_data, symbol, 100)
                if price_r and price_r.get("success"):
                    df = price_r["df"]
                    ind = await asyncio.to_thread(get_indicators, symbol, df)
                    meta = {
                        "verdict": {
                            "verdict_label":  "TRUNG LAP",
                            "confidence_pct": 50,
                            "bull_count":     0,
                            "bear_count":     0,
                            "active_agents":  0,
                        },
                        "ind":          ind or {},
                        "agent_verdicts": {},
                        "macro_v":      {},
                    }
            except Exception as e:
                logger.warning(f"local_swarm minimal meta fail: {e}")

        if meta is None:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=(
                    f"❌ Không thể lấy dữ liệu cho {symbol}.\n"
                    f"Hãy thử: /local_swarm {symbol} --check"
                )
            )
            return

        # ── Bước 2: Setup progress streamer ─────────────────────────────────
        streamer = _TelegramProgressStreamer(
            context.bot, chat_id, msg.message_id, loop
        )

        # ── Bước 3: Chạy swarm trong thread ─────────────────────────────────
        try:
            text, report = await asyncio.to_thread(
                run_local_swarm, symbol, meta, None, streamer
            )
        except RuntimeError as e:
            # LLM không available
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=f"❌ Lỗi LLM:\n{str(e)[:500]}\n\nCần set 1 trong:\nDEEPSEEK_API_KEY / OPENROUTER_API_KEY / GROQ_API_KEY / GEMINI_API_KEY"
            )
            return

        # ── Bước 4: Gửi kết quả ─────────────────────────────────────────────
        # Import smart split từ bot.py
        try:
            from bot import _smart_split, plain
        except ImportError:
            import re
            def plain(t): return re.sub(r"[*`]", "", t)
            def _smart_split(t):
                return [t[:4000]] if len(t) > 4000 else [t]

        parts = _smart_split(plain(text))

        # Edit loading message thành phần đầu
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=parts[0],
            )
        except Exception:
            await context.bot.send_message(chat_id=chat_id, text=parts[0])

        # Gửi phần còn lại
        for part in parts[1:]:
            await context.bot.send_message(chat_id=chat_id, text=part)

        logger.info(
            f"local_swarm {symbol}: OK | "
            f"{report.panel_verdict} | {report.elapsed_s:.0f}s | "
            f"{report.llm_provider}"
        )

    except Exception as e:
        import traceback
        logger.error(f"local_swarm flow error: {e}\n{traceback.format_exc()}")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=f"❌ Lỗi khi chạy Local Swarm {symbol}:\n{str(e)[:300]}"
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

_HELP_TEXT = """\
🤖 LOCAL SWARM PANEL — Hội đồng Chuyên gia AI Nội bộ

Cú pháp:
  /local_swarm <MA>          — Dùng kết quả /check gần nhất
  /local_swarm <MA> --check  — Chạy /check trước (kết quả tốt nhất)
  /local_swarm status        — Kiểm tra LLM available

Ví dụ:
  /local_swarm VCB
  /local_swarm STB --check
  /local_swarm HPG

5 chuyên gia AI sẽ tranh luận 2 vòng:
  📊 Chuyên gia Kỹ thuật
  🌏 Chiến lược gia Vĩ mô
  🛡️ Nhà Quản trị Rủi ro
  💡 SMC Trader
  📋 Bộ lọc Cơ bản

Output: Verdict + Entry/SL/TP/RR + Scenarios + Shelf Life

Thời gian: ~2-4 phút (phụ thuộc LLM provider)
"""


async def local_swarm_cmd(update, context):
    """
    /local_swarm <MA> [--check]

    Args:
      MA       : mã cổ phiếu
      --check  : chạy /check trước để có dữ liệu mới nhất
    """
    from telegram import Update
    from telegram.ext import ContextTypes

    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update); return

    args = context.args or []

    # Help
    if not args or args[0].lower() in ("help", "--help", "-h", "?"):
        await update.message.reply_text(_HELP_TEXT)
        return

    # Status check
    if args[0].lower() == "status":
        msg = await update.message.reply_text("Đang kiểm tra LLM...")
        try:
            from local_swarm import check_local_swarm_available
            ok, info = await asyncio.to_thread(check_local_swarm_available)
            if ok:
                await msg.edit_text(f"✅ Local Swarm: READY\n{info}")
            else:
                await msg.edit_text(f"❌ Local Swarm: UNAVAILABLE\n{info}")
        except Exception as e:
            await msg.edit_text(f"Lỗi kiểm tra: {e}")
        return

    # Parse symbol
    import re
    raw_sym = args[0]
    symbol  = raw_sym.upper().strip()[:10]
    if not re.match(r'^[A-Z0-9]{2,10}$', symbol):
        await update.message.reply_text("Mã không hợp lệ (VD: VCB, STB, HPG)."); return

    run_check_first = "--check" in [a.lower() for a in args]

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    wait    = _get_rate_limit(user_id)
    if wait > 0:
        await update.message.reply_text(
            f"Vui lòng chờ {wait:.0f}s trước khi /local_swarm tiếp."
        ); return
    _record(user_id)

    chat_id = update.effective_chat.id

    # Loading message
    init_text = (
        f"🤖 LOCAL SWARM: {symbol}\n"
        f"{'─'*30}\n"
        f"{'⏳ Chạy /check trước...' if run_check_first else '⏳ Đang khởi động...'}\n\n"
        f"5 chuyên gia AI sẽ tranh luận 2 vòng.\n"
        f"Ước tính: 2-4 phút. Bot vẫn nhận lệnh khác trong khi chờ."
    )
    msg = await update.message.reply_text(init_text)

    # Chạy background
    asyncio.create_task(
        _run_local_swarm_flow(symbol, run_check_first, context, chat_id, msg)
    )


# ══════════════════════════════════════════════════════════════════════════════
# FAILOVER PATCH — tự động dùng Local Swarm khi Vibe offline
# ══════════════════════════════════════════════════════════════════════════════

def install_vibe_failover(app):
    """
    Patch vibe_cmd để tự động fallback sang Local Swarm khi Vibe server offline.

    Gọi sau khi đăng ký handler:
        install_vibe_failover(app)
    """
    try:
        import bot as _bot_module
        original_vibe_cmd = _bot_module.vibe_cmd

        async def _vibe_with_failover(update, context):
            """Wrapper: thử Vibe → fail → Local Swarm."""
            try:
                from vibe_client import is_available
                vibe_ok = is_available()
            except Exception:
                vibe_ok = False

            if not vibe_ok:
                # Vibe offline → thông báo và offer Local Swarm
                args    = context.args or []
                symbol  = args[0].upper() if args else "?"

                try:
                    from bot import is_allowed, _deny
                except ImportError:
                    def is_allowed(_): return True
                    async def _deny(_): pass

                if not is_allowed(update):
                    await _deny(update); return

                chat_id = update.effective_chat.id
                msg = await update.message.reply_text(
                    f"⚠️ Vibe-Trading server OFFLINE.\n\n"
                    f"🔄 Tự động chuyển sang LOCAL SWARM PANEL\n"
                    f"Mã: {symbol} | 5 chuyên gia AI nội bộ\n\n"
                    f"Đang khởi động... (~2-4 phút)"
                )

                user_id = str(update.effective_user.id) if update.effective_user else "unknown"
                asyncio.create_task(
                    _run_local_swarm_flow(symbol, True, context, chat_id, msg)
                )
                return

            # Vibe online → chạy bình thường
            await original_vibe_cmd(update, context)

        # Replace handler
        for handler_group in app.handlers.values():
            for handler in handler_group:
                from telegram.ext import CommandHandler
                if (isinstance(handler, CommandHandler) and
                        "vibe" in (handler.commands or set()) and
                        "vibestatus" not in (handler.commands or set())):
                    handler.callback = _vibe_with_failover
                    logger.info("Vibe failover patch installed on /vibe")
                    return

        logger.warning("Could not find /vibe handler to patch")

    except Exception as e:
        logger.warning(f"install_vibe_failover failed: {e}")
