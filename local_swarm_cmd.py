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

DEBUG PATCH v2:
  - Log chi tiết từng bước với timestamp
  - Heartbeat 60s + cảnh báo 5 phút
  - Per-LLM-call timeout 120s (tránh hang vô tận)
  - /lswarm_fast: 2 chuyên gia, 1 vòng, tối đa 60s
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
SWARM_COOLDOWN    = 120      # giây giữa 2 lần /local_swarm cùng user
SWARM_TIMEOUT     = 600      # hard timeout 10 phút
FAST_TIMEOUT      = 90       # hard timeout cho /lswarm_fast
MAX_PROGRESS_MSG  = 30       # tối đa 30 dòng progress trước khi im lặng
PROGRESS_INTERVAL = 8        # giây cập nhật message Telegram
HEARTBEAT_INTERVAL = 60      # gửi heartbeat mỗi N giây
WARN_LONG_TASK    = 300      # cảnh báo sau 5 phút
LLM_CALL_TIMEOUT  = 120      # timeout cứng mỗi lần gọi LLM (giây)
SPAM_WARNING      = "⚠️ Đang xử lý tác vụ phức tạp, vui lòng đợi thêm..."

# ── Global state ──────────────────────────────────────────────────────────────
_last_local_swarm: dict[str, float] = {}    # user_id → timestamp

# Lưu task đang chạy để /cancel có thể hủy
# key: chat_id, value: {"future": Future, "cancel_event": Event, "symbol": str}
_active_swarm_tasks: dict[int, dict] = {}

# Thread pool riêng để chạy swarm (tránh block ThreadPoolExecutor chính)
_swarm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="swarm")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: log với timestamp rõ ràng
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log_step(step: str, symbol: str, extra: str = ""):
    """Log chi tiết từng bước với timestamp."""
    msg = f"[SWARM][{_ts()}] [{symbol}] {step}"
    if extra:
        msg += f" | {extra}"
    logger.info(msg)


# ─────────────────────────────────────────────────────────────────────────────
# CANCEL command
# ─────────────────────────────────────────────────────────────────────────────

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
    task["cancel_event"].set()
    future: Future = task["future"]
    future.cancel()
    _active_swarm_tasks.pop(chat_id, None)

    await update.message.reply_text(
        f"✅ Đã gửi tín hiệu hủy tác vụ /local_swarm {symbol}.\n"
        "Kết quả hiện có sẽ được trả về nếu phân tích đã có một phần."
    )


# ─────────────────────────────────────────────────────────────────────────────
# FAST MODE: /lswarm_fast — 2 chuyên gia, 1 vòng, tối đa 90s
# ─────────────────────────────────────────────────────────────────────────────

async def lswarm_fast_cmd(update, context):
    """
    /lswarm_fast <MA> — chế độ rút gọn:
      - Chỉ 2 chuyên gia (tech_analyst + risk_manager)
      - 1 vòng tranh luận
      - Timeout cứng 90 giây
      - Không cần cooldown dài
    Dùng để test xem vấn đề có phải do độ phức tạp prompt gốc không.
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
            "Cú pháp: /lswarm_fast <MA>\n"
            "Ví dụ : /lswarm_fast VCB\n"
            "⚡ Chế độ nhanh: 2 chuyên gia, 1 vòng, tối đa 90s."
        )
        return

    valid, symbol = _validate_symbol(args[0])
    if not valid:
        await update.message.reply_text(f"Mã '{args[0]}' không hợp lệ."); return

    chat_id = update.effective_chat.id
    _log_step("lswarm_fast STARTED", symbol)

    msg = await update.message.reply_text(
        f"⚡ FAST SWARM: {symbol}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏳ 2 chuyên gia | 1 vòng | tối đa {FAST_TIMEOUT}s\n"
        f"Đang lấy dữ liệu..."
    )

    progress_queue: queue.Queue[str] = queue.Queue()
    done_event    = threading.Event()
    cancel_event  = threading.Event()
    result_holder : dict = {"text": None, "error": None}

    def progress_cb(msg_text: str):
        if cancel_event.is_set():
            raise RuntimeError("Cancelled by user")
        progress_queue.put_nowait(str(msg_text))

    def _fast_worker():
        t_start = time.time()
        try:
            _log_step("fast_worker: import modules", symbol)
            from analyzer import analyze_stock_full

            progress_cb(f"📊 Lấy dữ liệu {symbol}...")
            _log_step("fast_worker: calling analyze_stock_full", symbol)

            try:
                result_text_check, meta = _call_with_timeout(
                    analyze_stock_full, [symbol], {},
                    timeout=60, label="analyze_stock_full"
                )
            except TimeoutError:
                result_holder["error"] = f"⏰ Lấy dữ liệu {symbol} quá 60s — API có thể đang chậm. Thử lại sau."
                _log_step("fast_worker: analyze_stock_full TIMEOUT", symbol)
                return
            except Exception as e:
                result_holder["error"] = f"❌ Lấy dữ liệu lỗi: {e}"
                _log_step("fast_worker: analyze_stock_full ERROR", symbol, str(e))
                return

            if meta is None:
                result_holder["error"] = f"Không lấy được dữ liệu cho {symbol}."
                return

            elapsed_data = round(time.time() - t_start, 1)
            _log_step(f"fast_worker: data OK in {elapsed_data}s", symbol)
            progress_cb(f"✅ Dữ liệu OK ({elapsed_data}s) | Bắt đầu fast swarm...")

            # Chạy fast swarm
            text = _run_fast_swarm(symbol, meta, progress_cb, cancel_event)
            result_holder["text"] = text
            elapsed_total = round(time.time() - t_start, 1)
            _log_step(f"fast_worker: DONE in {elapsed_total}s", symbol)

        except RuntimeError as e:
            if "Cancelled" in str(e):
                result_holder["error"] = "✅ Đã hủy theo yêu cầu."
            else:
                result_holder["error"] = f"Lỗi runtime: {e}"
            _log_step("fast_worker: RuntimeError", symbol, str(e))
        except Exception as e:
            result_holder["error"] = f"Lỗi fast swarm: {e}"
            _log_step("fast_worker: EXCEPTION", symbol, str(e))
            logger.error(f"fast_worker error [{symbol}]:", exc_info=True)
        finally:
            _log_step("fast_worker: setting done_event", symbol)
            done_event.set()

    future = _swarm_executor.submit(_fast_worker)
    _active_swarm_tasks[chat_id] = {
        "future": future,
        "cancel_event": cancel_event,
        "symbol": symbol,
        "started_at": time.time(),
    }

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
        timeout=FAST_TIMEOUT,
        mode="fast",
    )
    _active_swarm_tasks.pop(chat_id, None)


def _run_fast_swarm(symbol: str, meta: dict, progress_cb, cancel_event) -> str:
    """
    Fast swarm v3.0: chỉ 2 chuyên gia (tech_analyst + risk_manager), 1 vòng.
    Dùng API v3.0: extract_technical_data() + LLMClient + _build_round1_prompt()
    Không dùng SwarmOrchestrator 3 vòng → nhanh hơn ~3-4x.
    Trả về formatted text đơn giản.
    """
    # ── Import API v3.0 ────────────────────────────────────────────────────────
    from local_swarm import (
        LLMClient,
        EXPERTS,
        extract_technical_data,
        _get_skill_block,
        _build_round1_prompt,
        _parse_expert_response,
    )

    t0 = time.time()

    # ── Bước 1: extract technical data từ meta ────────────────────────────────
    progress_cb(f"📐 Chuẩn bị dữ liệu kỹ thuật {symbol}...")
    _log_step("fast_swarm: extract_technical_data", symbol)
    try:
        td = extract_technical_data(symbol, meta)
    except Exception as e:
        _log_step(f"fast_swarm: extract_technical_data ERROR: {e}", symbol)
        raise

    # ── Bước 2: chỉ 2 chuyên gia ─────────────────────────────────────────────
    fast_expert_ids = ("tech_analyst", "risk_manager")
    fast_experts    = [e for e in EXPERTS if e["id"] in fast_expert_ids]

    llm = LLMClient()
    _log_step(f"fast_swarm: LLM provider={llm.provider}/{llm.model}", symbol)

    opinions = []   # list of dict với keys: expert_id, role, emoji, stance, confidence, reason

    for expert in fast_experts:
        if cancel_event.is_set():
            _log_step("fast_swarm: cancelled before expert", symbol)
            break

        eid  = expert["id"]
        role = expert["role"]
        em   = expert.get("emoji", "👤")
        progress_cb(f"  {em} {role} phân tích...")
        t_llm = time.time()
        _log_step(f"fast_swarm: calling LLM for {eid}", symbol)

        try:
            # Build skill context block cho expert này
            skill_block = _get_skill_block(expert, td)

            # Build prompt vòng 1
            sys_p, usr_p = _build_round1_prompt(expert, td, skill_block)

            # Gọi LLM với timeout cứng
            raw = _llm_call_with_timeout(
                llm, sys_p, usr_p,
                max_tokens=700,
                timeout=LLM_CALL_TIMEOUT,
                symbol=symbol,
                expert_id=eid,
            )

            # Parse response
            parsed = _parse_expert_response(expert, raw, round_num=1)
            elapsed_llm = round(time.time() - t_llm, 1)

            stance     = parsed.get("stance",     "THEO DOI")
            confidence = parsed.get("confidence", 50)
            key_points = parsed.get("key_points", [])
            reason     = key_points[0] if key_points else parsed.get("concern", "Không rõ")

            opinions.append({
                "expert_id":  eid,
                "role":       role,
                "emoji":      em,
                "stance":     stance,
                "confidence": confidence,
                "reason":     str(reason)[:100],
            })

            _log_step(f"fast_swarm: {eid} done in {elapsed_llm}s → {stance}({confidence}%)", symbol)
            progress_cb(f"  → {em} {stance} ({confidence}%) [{elapsed_llm}s]")

        except TimeoutError:
            _log_step(f"fast_swarm: {eid} LLM TIMEOUT", symbol)
            progress_cb(f"  ⚠️ {role}: LLM timeout")
            opinions.append({
                "expert_id":  eid,
                "role":       role,
                "emoji":      em,
                "stance":     "THEO DOI",
                "confidence": 40,
                "reason":     "LLM timeout — không nhận được phản hồi",
            })
        except Exception as e:
            _log_step(f"fast_swarm: {eid} error: {e}", symbol)
            progress_cb(f"  ❌ {role}: {str(e)[:60]}")
            opinions.append({
                "expert_id":  eid,
                "role":       role,
                "emoji":      em,
                "stance":     "THEO DOI",
                "confidence": 40,
                "reason":     str(e)[:100],
            })

    # ── Bước 3: tổng hợp nhanh (không dùng moderator LLM) ────────────────────
    stances = [op["stance"] for op in opinions]
    buy_c   = stances.count("MUA")
    sell_c  = stances.count("BAN")
    watch_c = stances.count("THEO DOI")

    if buy_c > sell_c and buy_c > watch_c:
        verdict = "MUA"
    elif sell_c > buy_c and sell_c > watch_c:
        verdict = "BAN"
    else:
        verdict = "THEO DOI"

    avg_conf = int(sum(op["confidence"] for op in opinions) / len(opinions)) if opinions else 0
    elapsed  = round(time.time() - t0, 1)

    # ── Bước 4: format output ─────────────────────────────────────────────────
    lines = [
        f"⚡ FAST SWARM: {symbol}",
        "━" * 26,
        f"Kết luận : {verdict}  (tb {avg_conf}%)",
        f"Vote     : 🟢MUA={buy_c}  ⚪TD={watch_c}  🔴BAN={sell_c}",
        f"Thời gian: {elapsed}s",
        "",
        "CHUYÊN GIA:",
    ]

    for op in opinions:
        lines.append(f"{op['emoji']} {op['role']}: {op['stance']} ({op['confidence']}%)")
        lines.append(f"   └─ {op['reason']}")

    # Thêm context từ td nếu có
    price = td.get("price", 0)
    rsi   = td.get("rsi",   0)
    if price:
        lines += [
            "",
            f"Giá hiện tại: {price:,.0f}",
        ]
        if rsi:
            lines.append(f"RSI         : {rsi:.1f}")

    lines += [
        "",
        "⚠️ Fast mode — 2 chuyên gia, 1 vòng (không moderator).",
        f"Phân tích đầy đủ: /local_swarm {symbol}",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN command handler: /local_swarm
# ─────────────────────────────────────────────────────────────────────────────

async def local_swarm_cmd(update, context):
    """
    /local_swarm <MA> [--check]

    Flow:
      1. Validate + cooldown check
      2. Lấy meta từ analyze_stock_full (trong thread, timeout 90s)
      3. Tạo progress_queue + done_event
      4. Submit run_local_swarm vào thread pool
      5. Async watcher: poll queue, cập nhật Telegram, heartbeat, timeout
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
            "Nhanh  : /lswarm_fast VCB  (2 chuyên gia, ~60s)\n"
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

    _log_step("STARTED", symbol, f"user={user_id}")

    # ── Gửi loading message ───────────────────────────────────────────────────
    msg = await update.message.reply_text(
        f"🤖 LOCAL SWARM ĐANG CHẠY: {symbol}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏳ Đang lấy dữ liệu thị trường...\n"
        f"(Tối đa {SWARM_TIMEOUT//60} phút | /cancel để hủy)\n"
        f"⚡ Thử nhanh hơn: /lswarm_fast {symbol}"
    )

    # ── Kiểm tra availability nhanh ───────────────────────────────────────────
    if check_mode:
        try:
            from local_swarm import check_local_swarm_available
            _log_step("check_mode: testing LLM", symbol)
            ok, info = check_local_swarm_available()
            if not ok:
                await msg.edit_text(f"❌ Local Swarm không khả dụng:\n{info}")
                return
            await context.bot.send_message(chat_id=chat_id, text=f"✅ LLM sẵn sàng: {info}")
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi kiểm tra: {e}")
            return

    # ── Setup thread-safe communication ──────────────────────────────────────
    progress_queue: queue.Queue[str] = queue.Queue()
    done_event   = threading.Event()
    cancel_event = threading.Event()
    result_holder: dict = {"text": None, "error": None}

    def progress_cb(msg_text: str):
        """Callback an toàn: chỉ đẩy vào queue, không gọi asyncio."""
        if cancel_event.is_set():
            raise RuntimeError("Cancelled by user")
        progress_queue.put_nowait(str(msg_text))

    # ── Worker function chạy trong thread ────────────────────────────────────
    def _swarm_worker():
        """Chạy hoàn toàn synchronous, không biết gì về asyncio."""
        t_start = time.time()
        try:
            _log_step("worker: importing modules", symbol)
            from analyzer import analyze_stock_full
            from local_swarm import run_local_swarm

            progress_cb(f"📊 Đang phân tích {symbol} qua 16 engines...")
            _log_step("worker: calling analyze_stock_full", symbol)
            t_analyze = time.time()

            if cancel_event.is_set():
                result_holder["error"] = "Cancelled"
                return

            # analyze_stock_full với timeout cứng 90s
            try:
                result_text_check, meta = _call_with_timeout(
                    analyze_stock_full, [symbol], {},
                    timeout=90, label="analyze_stock_full"
                )
            except TimeoutError:
                elapsed = round(time.time() - t_analyze, 1)
                result_holder["error"] = (
                    f"⏰ analyze_stock_full({symbol}) timeout sau {elapsed}s.\n"
                    "Có thể nguồn dữ liệu đang chậm. Thử /debug_loader để kiểm tra."
                )
                _log_step("worker: analyze_stock_full TIMEOUT", symbol, f"{elapsed}s")
                return
            except Exception as e:
                result_holder["error"] = f"❌ Lấy dữ liệu lỗi: {type(e).__name__}: {e}"
                _log_step("worker: analyze_stock_full ERROR", symbol, str(e))
                logger.error(f"analyze_stock_full error [{symbol}]:", exc_info=True)
                return

            elapsed_analyze = round(time.time() - t_analyze, 1)
            _log_step(f"worker: analyze_stock_full DONE in {elapsed_analyze}s", symbol,
                      f"meta={'OK' if meta else 'None'}")

            if meta is None:
                result_holder["error"] = f"Không lấy được dữ liệu cho {symbol}. Thử /debug {symbol}"
                return

            if cancel_event.is_set():
                result_holder["error"] = "Cancelled"
                return

            progress_cb(f"✅ Dữ liệu OK ({elapsed_analyze}s) | Bắt đầu hội đồng 5 chuyên gia (3 vòng)...")
            _log_step("worker: calling run_local_swarm", symbol)
            t_swarm = time.time()

            # Patch LLM calls trong swarm để có timeout cứng per-call
            _patch_llm_timeout()

            text, report = run_local_swarm(
                symbol,
                meta=meta,
                progress_cb=progress_cb,
            )
            elapsed_swarm = round(time.time() - t_swarm, 1)
            elapsed_total = round(time.time() - t_start, 1)
            _log_step(f"worker: run_local_swarm DONE in {elapsed_swarm}s (total={elapsed_total}s)", symbol)

            result_holder["text"] = text

        except RuntimeError as e:
            if "Cancelled" in str(e):
                result_holder["error"] = "✅ Đã hủy theo yêu cầu."
            else:
                result_holder["error"] = f"Lỗi runtime: {e}"
            _log_step("worker: RuntimeError", symbol, str(e))
        except Exception as e:
            result_holder["error"] = f"Lỗi khi chạy Local Swarm: {type(e).__name__}: {e}"
            _log_step("worker: EXCEPTION", symbol, str(e))
            logger.error(f"Swarm worker error [{symbol}]:", exc_info=True)
        finally:
            elapsed_total = round(time.time() - t_start, 1)
            _log_step(f"worker: FINALLY - setting done_event (elapsed={elapsed_total}s)", symbol)
            done_event.set()    # BẮT BUỘC set dù thành công hay lỗi

    # ── Submit vào thread pool ────────────────────────────────────────────────
    _log_step("submitting to thread pool", symbol)
    future = _swarm_executor.submit(_swarm_worker)
    _active_swarm_tasks[chat_id] = {
        "future":       future,
        "cancel_event": cancel_event,
        "symbol":       symbol,
        "started_at":   time.time(),
    }

    # ── Async watcher: poll queue + done_event ────────────────────────────────
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
        timeout=SWARM_TIMEOUT,
        mode="full",
    )

    # Cleanup
    _active_swarm_tasks.pop(chat_id, None)
    _log_step("CLEANUP done", symbol)


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC WATCHER
# ─────────────────────────────────────────────────────────────────────────────

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
    timeout: int = SWARM_TIMEOUT,
    mode: str = "full",
):
    """
    Async watcher chạy trên event loop.
    Poll progress_queue và cập nhật Telegram message.
    Gửi heartbeat mỗi 60s. Cảnh báo khi quá 5 phút.
    Dừng khi done_event được set hoặc timeout.
    """
    started     = time.time()
    status_lines: list[str] = [
        f"🤖 LOCAL SWARM: {symbol}",
        "━" * 26,
    ]
    progress_count  = 0
    last_edit_time  = 0.0
    last_heartbeat  = started
    spam_mode       = False
    warned_long     = False

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

    async def _send_heartbeat(elapsed: int):
        """Gửi tin nhắn heartbeat mới (không edit) để user thấy bot còn sống."""
        minute = elapsed // 60
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"💓 Tác vụ {symbol} đã chạy được {minute} phút...\n"
                     f"(Gõ /cancel để hủy)"
            )
        except Exception as e:
            logger.debug(f"heartbeat send error: {e}")

    while True:
        now     = time.time()
        elapsed = now - started

        # ── Kiểm tra timeout cứng ─────────────────────────────────────────────
        if elapsed > timeout:
            logger.warning(f"[SWARM][{symbol}] TIMEOUT after {elapsed:.0f}s")
            cancel_event.set()
            result_holder["error"] = (
                f"⏰ Vượt quá thời gian xử lý ({timeout//60} phút).\n"
                "Kết quả không đầy đủ. Thử lại sau hoặc dùng /lswarm_fast."
            )
            done_event.set()
            break

        # ── Cảnh báo task lâu (sau 5 phút) ───────────────────────────────────
        if not warned_long and elapsed > WARN_LONG_TASK and mode == "full":
            warned_long = True
            _log_step(f"LONG TASK WARNING at {elapsed:.0f}s", symbol)
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        f"⚠️ Tác vụ {symbol} đang mất nhiều thời gian hơn dự kiến.\n"
                        f"Có thể hệ thống LLM đang quá tải.\n"
                        f"Bạn có thể:\n"
                        f"  • Gõ /cancel để hủy và thử /lswarm_fast {symbol}\n"
                        f"  • Đợi thêm (tối đa {timeout//60} phút)"
                    )
                )
            except Exception:
                pass

        # ── Heartbeat mỗi 60s ─────────────────────────────────────────────────
        if now - last_heartbeat >= HEARTBEAT_INTERVAL:
            last_heartbeat = now
            elapsed_int    = int(elapsed)
            _log_step(f"heartbeat at {elapsed_int}s", symbol)
            await _send_heartbeat(elapsed_int)

        # ── Drain progress queue ──────────────────────────────────────────────
        drained = False
        while True:
            try:
                line = progress_queue.get_nowait()
                progress_count += 1
                drained = True

                if progress_count <= MAX_PROGRESS_MSG:
                    status_lines.append(line)
                    # Giữ tối đa 17 dòng gần nhất
                    if len(status_lines) > 17:
                        status_lines = status_lines[:2] + status_lines[-15:]
                elif not spam_mode:
                    spam_mode = True
                    status_lines.append(SPAM_WARNING)

            except queue.Empty:
                break

        # ── Kiểm tra done_event ───────────────────────────────────────────────
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
            _log_step("done_event detected, breaking watcher loop", symbol)
            break

        # ── Cập nhật message nếu có thay đổi ─────────────────────────────────
        if drained:
            elapsed_s  = int(elapsed)
            footer     = f"\n⏱️ {elapsed_s}s | /cancel để hủy"
            await _try_edit("\n".join(status_lines) + footer)

        await asyncio.sleep(2.0)

    # ── Gửi kết quả cuối ─────────────────────────────────────────────────────
    elapsed_total = int(time.time() - started)
    _log_step(f"watcher: sending final result (elapsed={elapsed_total}s)", symbol,
              f"error={'Y' if result_holder['error'] else 'N'} text={'Y' if result_holder['text'] else 'N'}")

    if result_holder["error"]:
        error_msg = str(result_holder["error"])
        try:
            await msg.edit_text(
                f"❌ {error_msg}\n\n"
                f"⏱️ Đã chạy {elapsed_total}s\n"
                f"💡 Thử: /lswarm_fast {symbol}"
            )
        except Exception:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"❌ {error_msg}\n💡 Thử: /lswarm_fast {symbol}"
            )
        return

    if result_holder["text"]:
        result_text = str(result_holder["text"])
        try:
            await msg.edit_text("✅ Hoàn tất! Đang gửi báo cáo...")
        except Exception:
            pass

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
                logger.error(f"Gửi swarm result chunk {i} [{symbol}]: {e}")
                break
    else:
        try:
            await msg.edit_text(
                f"⚠️ Local Swarm {symbol} kết thúc nhưng không có kết quả.\n"
                f"Thử lại: /lswarm_fast {symbol}"
            )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: timeout wrapper cho blocking calls
# ─────────────────────────────────────────────────────────────────────────────

def _call_with_timeout(func, args: list, kwargs: dict, timeout: int, label: str):
    """
    Gọi func(*args, **kwargs) trong thread riêng với timeout cứng.
    Raise TimeoutError nếu quá timeout giây.
    Dùng trong sync context (không phải asyncio).
    """
    result_box = [None, None]  # [result, exception]
    finished   = threading.Event()

    def _runner():
        try:
            result_box[0] = func(*args, **kwargs)
        except Exception as e:
            result_box[1] = e
        finally:
            finished.set()

    t = threading.Thread(target=_runner, daemon=True, name=f"timeout_{label}")
    t.start()
    ok = finished.wait(timeout=timeout)

    if not ok:
        logger.warning(f"_call_with_timeout: '{label}' TIMED OUT after {timeout}s")
        raise TimeoutError(f"'{label}' timed out after {timeout}s")

    if result_box[1] is not None:
        raise result_box[1]

    return result_box[0]


def _llm_call_with_timeout(llm, system: str, user: str, max_tokens: int,
                            timeout: int, symbol: str, expert_id: str) -> str:
    """
    Gọi llm.chat() với timeout cứng per-call.
    Đây là nơi hay bị treo nhất — DeepSeek/Groq có thể hang vô tận.
    """
    label = f"LLM:{expert_id}"
    try:
        return _call_with_timeout(
            llm.chat,
            [system, user],
            {"max_tokens": max_tokens},
            timeout=timeout,
            label=label,
        )
    except TimeoutError:
        _log_step(f"LLM call TIMEOUT for {expert_id}", symbol, f"timeout={timeout}s")
        raise
    except Exception as e:
        _log_step(f"LLM call ERROR for {expert_id}: {e}", symbol)
        raise


def _patch_llm_timeout():
    """
    Monkey-patch LLMClient.chat() trong local_swarm để add timeout cứng.
    Gọi 1 lần trước khi chạy swarm worker.
    Idempotent — an toàn khi gọi nhiều lần.
    """
    try:
        import local_swarm as _ls

        if getattr(_ls.LLMClient, "_timeout_patched", False):
            return   # đã patch rồi

        original_chat = _ls.LLMClient.chat

        def _chat_with_timeout(self, system: str, user: str, max_tokens: int = _ls.LLM_MAX_TOKENS) -> str:
            label = f"LLM:{self.provider}"
            return _call_with_timeout(
                original_chat,
                [self, system, user],
                {"max_tokens": max_tokens},
                timeout=LLM_CALL_TIMEOUT,
                label=label,
            )

        _ls.LLMClient.chat          = _chat_with_timeout
        _ls.LLMClient._timeout_patched = True
        logger.info(f"✅ LLMClient.chat patched with {LLM_CALL_TIMEOUT}s per-call timeout")

    except Exception as e:
        logger.warning(f"_patch_llm_timeout failed (non-critical): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT MESSAGE
# ─────────────────────────────────────────────────────────────────────────────

def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Chia text thành chunks không vượt quá max_len ký tự."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, max_len)
        if cut < max_len // 2:
            cut = max_len
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# VIBE FAILOVER
# ─────────────────────────────────────────────────────────────────────────────

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

            await update.message.reply_text(
                "⚠️ Vibe-Trading API offline. Chuyển sang Local Swarm...\n"
                "(Chất lượng tương đương nhưng chạy chậm hơn ~3 phút)"
            )
            return await local_swarm_cmd(update, context)

        for handler in app.handlers.get(0, []):
            if hasattr(handler, "command") and "vibe" in (handler.command or []):
                handler.callback = _vibe_with_failover
                logger.info("✅ Vibe failover đã được cài vào /vibe handler")
                return

        app.add_handler(CommandHandler("vibe_local", local_swarm_cmd))
        logger.info("✅ Vibe failover: thêm /vibe_local handler")

    except Exception as e:
        logger.warning(f"install_vibe_failover: {e}")
