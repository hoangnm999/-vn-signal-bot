"""
local_swarm_cmd.py — Handler Telegram cho /local_swarm.

ARCHITECTURE FIX (lỗi freeze 30 phút):
─────────────────────────────────────────────────────────────────────
Vấn đề gốc: progress_cb được gọi từ sync thread (asyncio.to_thread),
bên trong cb lại cố schedule coroutine vào event loop → deadlock.

Giải pháp: tách hoàn toàn 2 luồng:
  1. Thread worker (run_local_swarm) — sync, không biết gì về asyncio
     └─ Đẩy progress message vào queue (thread-safe)
     └─ Set done_event khi xong hoặc lỗi

  2. Async watcher (chạy trên event loop) — poll queue + done_event
     └─ Edit message Telegram mỗi khi có progress mới
     └─ Dừng ngay khi done_event được set
     └─ Hard timeout 600s

Không còn callback asyncio từ sync thread → không còn deadlock.
─────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SWARM_COOLDOWN   = 120      # giây giữa 2 lần /local_swarm cùng user
SWARM_TIMEOUT    = 600      # hard timeout 10 phút (Fix #2)
MAX_PROGRESS_MSG = 30       # tối đa 30 dòng progress trước khi im lặng (Fix #3)
PROGRESS_INTERVAL = 8       # giây cập nhật message Telegram
SPAM_WARNING     = "⚠️ Đang xử lý tác vụ phức tạp, vui lòng đợi thêm..."

# ── Global state ──────────────────────────────────────────────────────────────
_last_local_swarm: dict[str, float] = {}    # user_id → timestamp

# Lưu task đang chạy để /cancel có thể hủy (Fix #4)
# key: chat_id, value: {"future": Future, "cancel_event": Event, "symbol": str}
_active_swarm_tasks: dict[int, dict] = {}

# Thread pool riêng để chạy swarm (tránh block ThreadPoolExecutor chính)
_swarm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="swarm")


# ── Cancel command (Fix #4) ───────────────────────────────────────────────────

async def cancel_swarm_cmd(update, context):
    """
    /cancel — hủy tác vụ local_swarm đang chạy của chat này.
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update); return

    chat_id = update.effective_chat.id
    task = _active_swarm_tasks.get(chat_id)

    if not task:
        await update.message.reply_text("Không có tác vụ nào đang chạy.")
        return

    symbol = task.get("symbol", "?")
    task["cancel_event"].set()    # báo cho thread dừng
    future: Future = task["future"]
    future.cancel()               # cancel future nếu chưa start
    _active_swarm_tasks.pop(chat_id, None)

    await update.message.reply_text(
        f"✅ Đã gửi tín hiệu hủy tác vụ /local_swarm {symbol}.\n"
        "Kết quả hiện có sẽ được trả về nếu phân tích đã có một phần."
    )


# ── Main command handler ──────────────────────────────────────────────────────

async def local_swarm_cmd(update, context):
    """
    /local_swarm <MA> [--check]

    Flow:
      1. Validate + cooldown check
      2. Lấy meta từ analyze_stock_full (trong thread)
      3. Tạo progress_queue + done_event
      4. Submit run_local_swarm vào thread pool
      5. Async watcher: poll queue, cập nhật Telegram, timeout
      6. Khi done: gửi kết quả cuối
    """
    try:
        from bot import is_allowed, _deny, plain, _validate_symbol
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass
        def plain(t): return t
        def _validate_symbol(s):
            s = s.upper().strip()
            return (s.isalnum() and 2 <= len(s) <= 10), s

    if not is_allowed(update):
        await _deny(update); return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Cú pháp: /local_swarm <MA>\n"
            "Ví dụ : /local_swarm VCB\n"
            "Hủy   : /cancel"
        )
        return

    valid, symbol = _validate_symbol(args[0])
    if not valid:
        await update.message.reply_text(f"Mã '{args[0]}' không hợp lệ (2-10 ký tự)."); return

    check_mode = "--check" in [a.lower() for a in args]

    # ── Cooldown ──────────────────────────────────────────────────────────────
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_local_swarm.get(user_id, 0)
    if since < SWARM_COOLDOWN:
        wait = int(SWARM_COOLDOWN - since)
        await update.message.reply_text(f"⏳ Vui lòng chờ {wait}s trước khi chạy lại."); return
    _last_local_swarm[user_id] = time.time()

    chat_id = update.effective_chat.id

    # ── Kiểm tra có task đang chạy không ─────────────────────────────────────
    if chat_id in _active_swarm_tasks:
        await update.message.reply_text(
            f"⚠️ Đang có tác vụ /local_swarm {_active_swarm_tasks[chat_id].get('symbol','?')} "
            f"đang chạy.\nGõ /cancel để hủy hoặc đợi kết thúc."
        )
        return

    # ── Gửi loading message ───────────────────────────────────────────────────
    msg = await update.message.reply_text(
        f"🤖 LOCAL SWARM ĐANG CHẠY: {symbol}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏳ Đang lấy dữ liệu thị trường...\n"
        f"(Tối đa {SWARM_TIMEOUT//60} phút | /cancel để hủy)"
    )

    # ── Kiểm tra availability nhanh ───────────────────────────────────────────
    if check_mode:
        try:
            from local_swarm import check_local_swarm_available
            ok, info = check_local_swarm_available()
            if not ok:
                await msg.edit_text(f"❌ Local Swarm không khả dụng:\n{info}")
                return
            await context.bot.send_message(chat_id=chat_id, text=f"✅ LLM sẵn sàng: {info}")
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi kiểm tra: {e}")
            return

    # ── Setup thread-safe communication (Fix #1) ──────────────────────────────
    progress_queue: queue.Queue[str] = queue.Queue()
    done_event   = threading.Event()          # set khi swarm xong
    cancel_event = threading.Event()          # set khi user /cancel
    result_holder: dict = {"text": None, "error": None}

    def progress_cb(msg_text: str):
        """Callback an toàn: chỉ đẩy vào queue, không gọi asyncio."""
        if cancel_event.is_set():
            raise RuntimeError("Cancelled by user")
        progress_queue.put_nowait(str(msg_text))

    # ── Worker function chạy trong thread ────────────────────────────────────
    def _swarm_worker():
        """Chạy hoàn toàn synchronous, không biết gì về asyncio."""
        try:
            from analyzer import analyze_stock_full
            from local_swarm import run_local_swarm

            progress_cb(f"📊 Đang phân tích {symbol} qua 16 engines...")

            if cancel_event.is_set():
                result_holder["error"] = "Cancelled"
                return

            # Lấy meta từ analyze_stock_full
            result_text_check, meta = analyze_stock_full(symbol)
            if meta is None:
                result_holder["error"] = f"Không lấy được dữ liệu cho {symbol}. Thử /debug {symbol}"
                return

            if cancel_event.is_set():
                result_holder["error"] = "Cancelled"
                return

            progress_cb(f"✅ Dữ liệu OK | Bắt đầu hội đồng 5 chuyên gia (3 vòng)...")

            # Chạy swarm với timeout tích hợp (Fix #2)
            text, report = run_local_swarm(
                symbol,
                meta=meta,
                progress_cb=progress_cb,
            )
            result_holder["text"] = text

        except RuntimeError as e:
            if "Cancelled" in str(e):
                result_holder["error"] = "✅ Đã hủy theo yêu cầu."
            else:
                result_holder["error"] = f"Lỗi runtime: {e}"
            logger.warning(f"Swarm worker RuntimeError: {e}")
        except Exception as e:
            result_holder["error"] = f"Lỗi khi chạy Local Swarm: {e}"
            logger.error(f"Swarm worker error: {e}", exc_info=True)
        finally:
            done_event.set()    # BẮT BUỘC set dù thành công hay lỗi (Fix #1)

    # ── Submit vào thread pool ────────────────────────────────────────────────
    future = _swarm_executor.submit(_swarm_worker)
    _active_swarm_tasks[chat_id] = {
        "future":       future,
        "cancel_event": cancel_event,
        "symbol":       symbol,
        "started_at":   time.time(),
    }

    # ── Async watcher: poll queue + done_event (Fix #1, #3) ──────────────────
    await _watch_swarm_progress(
        context=context,
        chat_id=chat_id,
        msg=msg,
        progress_queue=progress_queue,
        done_event=done_event,
        cancel_event=cancel_event,
        result_holder=result_holder,
        symbol=symbol,
        plain=plain,
    )

    # Cleanup
    _active_swarm_tasks.pop(chat_id, None)


async def _watch_swarm_progress(
    context,
    chat_id: int,
    msg,
    progress_queue: queue.Queue,
    done_event: threading.Event,
    cancel_event: threading.Event,
    result_holder: dict,
    symbol: str,
    plain,
):
    """
    Async watcher chạy trên event loop.
    Poll progress_queue và cập nhật Telegram message.
    Dừng khi done_event được set hoặc timeout.
    (Fix #1: đồng bộ đúng; Fix #2: hard timeout; Fix #3: cap 30 lines)
    """
    started     = time.time()
    status_lines: list[str] = [
        f"🤖 LOCAL SWARM: {symbol}",
        "━" * 26,
    ]
    progress_count = 0      # số dòng progress đã nhận (Fix #3)
    last_edit_time = 0.0
    spam_mode      = False   # True khi vượt MAX_PROGRESS_MSG

    async def _try_edit(text: str):
        nonlocal last_edit_time
        now = time.time()
        if now - last_edit_time < 3.0:   # rate limit: không edit quá 1 lần/3s
            return
        last_edit_time = now
        try:
            await msg.edit_text(plain(text[:4000]))
        except Exception as e:
            if "message is not modified" not in str(e).lower():
                logger.debug(f"edit_text error: {e}")

    while True:
        # ── Kiểm tra timeout cứng (Fix #2) ───────────────────────────────────
        elapsed = time.time() - started
        if elapsed > SWARM_TIMEOUT:
            logger.warning(f"Swarm {symbol} TIMEOUT after {elapsed:.0f}s")
            cancel_event.set()
            result_holder["error"] = (
                f"⏰ Đã vượt quá thời gian xử lý ({SWARM_TIMEOUT//60} phút).\n"
                "Kết quả không đầy đủ. Thử lại sau hoặc dùng /check."
            )
            done_event.set()
            break

        # ── Drain progress queue ──────────────────────────────────────────────
        drained = False
        while True:
            try:
                line = progress_queue.get_nowait()
                progress_count += 1
                drained = True

                if progress_count <= MAX_PROGRESS_MSG:
                    # Thêm dòng mới vào status (Fix #3)
                    status_lines.append(line)
                    # Giữ tối đa 15 dòng gần nhất để message không quá dài
                    if len(status_lines) > 17:
                        status_lines = status_lines[:2] + status_lines[-15:]
                elif not spam_mode:
                    # Vượt giới hạn: chuyển sang spam_mode (Fix #3)
                    spam_mode = True
                    status_lines.append(SPAM_WARNING)

            except queue.Empty:
                break

        # ── Kiểm tra done_event (Fix #1) ─────────────────────────────────────
        if done_event.is_set():
            # Drain lần cuối
            while True:
                try:
                    line = progress_queue.get_nowait()
                    progress_count += 1
                    if progress_count <= MAX_PROGRESS_MSG:
                        status_lines.append(line)
                except queue.Empty:
                    break
            break

        # ── Cập nhật message nếu có thay đổi ─────────────────────────────────
        if drained:
            elapsed_s = int(time.time() - started)
            footer = f"\n⏱️ {elapsed_s}s | /cancel để hủy"
            await _try_edit("\n".join(status_lines) + footer)

        # Chờ interval ngắn rồi check lại
        await asyncio.sleep(2.0)

    # ── Gửi kết quả cuối (Fix #5) ────────────────────────────────────────────
    elapsed_total = int(time.time() - started)

    if result_holder["error"]:
        error_msg = str(result_holder["error"])
        # Xóa message loading, gửi lỗi
        try:
            await msg.edit_text(
                f"❌ {error_msg}\n\n"
                f"⏱️ Đã chạy {elapsed_total}s"
            )
        except Exception:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ {error_msg}"
            )
        return

    if result_holder["text"]:
        result_text = str(result_holder["text"])
        # Xóa message loading
        try:
            await msg.edit_text("✅ Hoàn tất! Đang gửi báo cáo...")
        except Exception:
            pass

        # Gửi kết quả dưới dạng message mới (Fix #5)
        # Nếu quá dài thì chia nhỏ
        chunks = _split_message(result_text, max_len=4000)
        for i, chunk in enumerate(chunks):
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=plain(chunk),
                )
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Gửi swarm result chunk {i}: {e}")
                break
    else:
        try:
            await msg.edit_text(
                f"⚠️ Local Swarm {symbol} kết thúc nhưng không có kết quả.\n"
                f"Thử lại: /local_swarm {symbol}"
            )
        except Exception:
            pass


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Chia text thành chunks không vượt quá max_len ký tự."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Cắt tại newline gần nhất
        cut = text.rfind("\n", 0, max_len)
        if cut < max_len // 2:
            cut = max_len
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


# ── Vibe failover (khi Vibe-Trading offline → dùng local swarm) ──────────────

def install_vibe_failover(app) -> None:
    """
    Patch /vibe command: khi Vibe API offline, tự động fallover sang local_swarm.
    Gọi sau khi đăng ký CommandHandler("vibe", vibe_cmd).
    """
    try:
        from bot import vibe_cmd as _orig_vibe_cmd
        from telegram.ext import CommandHandler

        async def _vibe_with_failover(update, context):
            try:
                from vibe_client import is_available
                if is_available():
                    return await _orig_vibe_cmd(update, context)
            except Exception:
                pass

            # Failover: chạy local_swarm thay thế
            await update.message.reply_text(
                "⚠️ Vibe-Trading API offline. Chuyển sang Local Swarm...\n"
                "(Chất lượng tương đương nhưng chạy chậm hơn ~3 phút)"
            )
            return await local_swarm_cmd(update, context)

        # Thay thế handler /vibe hiện tại
        # Lưu ý: chỉ hoạt động nếu được gọi sau khi add_handler("vibe")
        for handler in app.handlers.get(0, []):
            if hasattr(handler, "command") and "vibe" in (handler.command or []):
                handler.callback = _vibe_with_failover
                logger.info("✅ Vibe failover đã được cài vào /vibe handler")
                return

        # Nếu không tìm thấy, add mới
        app.add_handler(CommandHandler("vibe_local", local_swarm_cmd))
        logger.info("✅ Vibe failover: thêm /vibe_local handler")

    except Exception as e:
        logger.warning(f"install_vibe_failover: {e}")
