"""
morning_briefing.py — /morning command cho VN Signal Bot

Workflow tự động mỗi sáng (hoặc gọi thủ công):
  1. market_regime    → Gate keeper: có nên trade hôm nay không?
  2. scan_watchlist   → Top mã score cao nhất (tái dùng batch_scanner)
  3. /wave (cache)    → Filter: mã nào Wave GIAM rõ thì loại
  4. Format 1 message → Gửi tổng hợp, kèm gợi ý bước tiếp theo

Output ví dụ:
  MORNING BRIEFING — 28/04/2026
  Regime: R1 Bull Quiet — Den xanh

  TOP MA HOM NAY:
    HAH  Wave TANG ★★★ (conf 0.41) | Score 8.2  → /analog HAH
    DVP  Wave TANG ★★☆ (conf 0.31) | Score 7.8  → /analog DVP
    VCB  Wave KHONG RO (conf 0.14)  | Score 7.1  → can confirmation
    STB  Wave GIAM ★★★ (conf 0.38)  | Score 6.5  → bo qua

  → Uu tien: /analog HAH, /analog DVP
  → Portfolio: /portfolio

Thresholds:
  WAVE_CONF_CLEAR  = 0.30   Wave rõ ràng (TANG hoặc GIAM đáng tin)
  WAVE_CONF_NOISE  = 0.20   Dưới này → KHONG RO, bỏ qua verdict
  TOP_N_SCAN       = 7      Lấy top N mã từ scan để chạy wave
  TOP_N_RECOMMEND  = 5      Hiển thị tối đa N mã trong briefing
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
WAVE_CONF_CLEAR    = 0.30   # confidence >= này → verdict đáng tin
WAVE_CONF_NOISE    = 0.20   # confidence < này → KHONG RO
TOP_N_SCAN         = 7      # lấy top N mã từ scan để chạy wave
TOP_N_RECOMMEND    = 5      # hiển thị tối đa N mã
MORNING_HOUR       = 8
MORNING_MINUTE     = 15     # 8:15 AM — sau scan cron 8:00


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Market Regime
# ══════════════════════════════════════════════════════════════════════════════

def _get_regime() -> dict:
    """
    Lấy market regime hiện tại.
    Returns dict với keys: regime (int), label (str), emoji (str), is_bull (bool).
    """
    try:
        from market_regime import get_market_regime
        mr = get_market_regime()
        if mr:
            return mr
    except Exception as e:
        logger.warning(f"morning: get_market_regime fail: {e}")

    # Fallback minimal
    return {"regime": 0, "label": "Khong xac dinh", "emoji": "❓", "is_bull": False}


def _regime_gate(regime: int) -> tuple[bool, str]:
    """
    Quyết định có tiếp tục không dựa trên regime.
    Returns (should_continue, message).
    """
    gates = {
        1: (True,  "Den XANH — R1 Bull Quiet, tin hieu mua tin cay nhat"),
        2: (True,  "Den VANG — R2 Bull Volatile, giam size 20-30%"),
        3: (True,  "Den CAM — R3 Bear Quiet, chi trade setup rat ro, size nho"),
        4: (False, "Den DO — R4 Bear Volatile, KHONG mo position moi"),
        0: (True,  "Khong xac dinh duoc regime — can than"),
    }
    return gates.get(regime, (True, ""))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Scan top mã (tái dùng batch_scanner)
# ══════════════════════════════════════════════════════════════════════════════

def _get_top_symbols(n: int = TOP_N_SCAN) -> list[dict]:
    """
    Chạy batch scan và lấy top N mã theo score.
    Returns list of {"symbol": str, "score": float, "wr": float, "stats": dict}.
    Dùng cache nếu scan đã chạy gần đây (trong 30 phút).
    """
    try:
        from batch_scanner import load_watchlist, run_batch_scan
        symbols     = load_watchlist()
        scan_result = run_batch_scan(symbols)   # dùng cache internal của batch_scanner
        ranked      = scan_result.get("ranked", [])

        top = []
        for r in ranked[:n]:
            s = r.get("stats", {})
            top.append({
                "symbol":  r["symbol"],
                "score":   r.get("score", 0.0),
                "wr":      s.get("win_rate", 0.0),
                "mae":     s.get("median_mdd", 0.0),
                "n":       s.get("n_analogs", 0),
            })
        return top

    except Exception as e:
        logger.error(f"morning: _get_top_symbols fail: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Wave filter (parallel, dùng cache)
# ══════════════════════════════════════════════════════════════════════════════

def _get_wave_one(symbol: str) -> dict:
    """Worker: lấy wave info từ cache cho 1 mã."""
    try:
        from wave_pattern import analyze_wave
        result = analyze_wave(symbol, force_rebuild=False)
        verdict    = result.get("verdict", "KHONG RO")
        score_up   = result.get("score_up", 0.0)
        score_down = result.get("score_down", 0.0)
        confidence = result.get("confidence", abs(score_up - score_down))

        # Normalize verdict theo confidence
        if confidence < WAVE_CONF_NOISE:
            verdict = "KHONG RO"
        elif confidence < WAVE_CONF_CLEAR:
            verdict = verdict + "_WEAK"   # internal tag

        return {
            "symbol":     symbol,
            "verdict":    verdict,
            "score_up":   score_up,
            "score_down": score_down,
            "confidence": round(confidence, 3),
            "ok":         result.get("ok", False),
        }
    except Exception as e:
        logger.warning(f"morning: wave {symbol}: {e}")
        return {
            "symbol":     symbol,
            "verdict":    "KHONG RO",
            "score_up":   0.0,
            "score_down": 0.0,
            "confidence": 0.0,
            "ok":         False,
        }


def _get_wave_parallel(symbols: list[str]) -> dict[str, dict]:
    """Chạy wave song song cho danh sách symbols, tái dùng cache."""
    if not symbols:
        return {}
    n_workers = min(4, len(symbols))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_get_wave_one, symbols))
    return {r["symbol"]: r for r in results}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Classify + Format
# ══════════════════════════════════════════════════════════════════════════════

def _wave_stars(verdict: str, confidence: float) -> str:
    """Convert verdict + confidence → display string."""
    is_weak = verdict.endswith("_WEAK")
    base    = verdict.replace("_WEAK", "")

    if base == "SONG TANG":
        direction = "TANG"
        score_str = "★★★" if confidence >= WAVE_CONF_CLEAR * 1.3 else "★★☆"
    elif base == "SONG GIAM":
        direction = "GIAM"
        score_str = "★★★" if confidence >= WAVE_CONF_CLEAR * 1.3 else "★★☆"
    else:
        return "Wave ?"

    if is_weak:
        score_str = "★☆☆"

    return f"Wave {score_str} {direction} (conf {confidence:.2f})"


def _classify_symbol(sym_data: dict, wave_data: dict) -> dict:
    """
    Gộp scan score + wave verdict → xếp loại.

    Returns dict với:
      category: "recommend" | "watch" | "skip"
      reason:   str (hiển thị cho user)
      action:   str (gợi ý bước tiếp theo)
    """
    symbol    = sym_data["symbol"]
    score     = sym_data["score"]
    wave      = wave_data.get(symbol, {})
    verdict   = wave.get("verdict", "KHONG RO")
    conf      = wave.get("confidence", 0.0)
    base_v    = verdict.replace("_WEAK", "")
    is_weak   = verdict.endswith("_WEAK")

    # Gate 1: Wave GIAM rõ ràng → skip
    if base_v == "SONG GIAM" and conf >= WAVE_CONF_CLEAR:
        return {
            "category": "skip",
            "wave_str": _wave_stars(verdict, conf),
            "reason":   "Wave GIAM ro rang",
            "action":   "Bo qua",
        }

    # Gate 2: Wave TANG rõ ràng → recommend
    if base_v == "SONG TANG" and conf >= WAVE_CONF_CLEAR and not is_weak:
        return {
            "category": "recommend",
            "wave_str": _wave_stars(verdict, conf),
            "reason":   "Wave TANG ro rang",
            "action":   f"/analog {symbol}",
        }

    # Gate 3: Wave không rõ hoặc yếu → watch
    return {
        "category": "watch",
        "wave_str": _wave_stars(verdict, conf) if base_v != "KHONG RO" else "Wave KHONG RO",
        "reason":   "Can them confirmation",
        "action":   f"/wave {symbol} roi /analog {symbol}",
    }


def _regime_advice(regime: int) -> str:
    """Gợi ý quản lý vị thế theo regime."""
    advice = {
        1: "Full size binh thuong",
        2: "Giam size 20-30%, SL chat hon",
        3: "Size nho, chi setup rat ro rang",
        4: "Khong mo position moi",
        0: "Can than — khong xac dinh duoc thi truong",
    }
    return advice.get(regime, "")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BUILD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_morning_briefing() -> str:
    """
    Build toàn bộ morning briefing.
    Chạy trong thread (blocking), gọi bằng asyncio.to_thread().

    Returns: str — message gửi Telegram.
    """
    t0   = time.time()
    today = datetime.now().strftime("%d/%m/%Y")

    lines = [f"MORNING BRIEFING — {today}", "=" * 38, ""]

    # ── Step 1: Regime ────────────────────────────────────────────────────────
    regime_data = _get_regime()
    regime      = regime_data.get("regime", 0)
    r_label     = regime_data.get("label", "Khong xac dinh")
    r_emoji     = regime_data.get("emoji", "❓")
    should_continue, gate_msg = _regime_gate(regime)

    lines.append(f"MARKET REGIME:")
    lines.append(f"  {r_emoji} {r_label}")
    lines.append(f"  {gate_msg}")
    lines.append("")

    if not should_continue:
        lines.append("=" * 38)
        lines.append("KHONG TRADE HOM NAY.")
        lines.append("Uu tien bao ve von, cho regime thay doi.")
        lines.append("")
        lines.append("Dung /regime de xem chi tiet thi truong.")
        lines.append("Dung /portfolio de kiem tra vi the hien co.")
        return "\n".join(lines)

    # ── Step 2: Scan top mã ───────────────────────────────────────────────────
    lines.append("Dang scan watchlist...")
    top_symbols = _get_top_symbols(TOP_N_SCAN)

    if not top_symbols:
        lines[-1] = "Khong co ma nao du dieu kien hom nay."
        lines.append("")
        lines.append("Goi y: /scan_watchlist de chay scan thu cong.")
        return "\n".join(lines)

    # ── Step 3: Wave filter (parallel) ───────────────────────────────────────
    sym_list  = [s["symbol"] for s in top_symbols]
    wave_data = _get_wave_parallel(sym_list)

    # ── Step 4: Classify ──────────────────────────────────────────────────────
    classified = []
    for sym_data in top_symbols:
        cl = _classify_symbol(sym_data, wave_data)
        classified.append({**sym_data, **cl})

    recommend = [c for c in classified if c["category"] == "recommend"]
    watch     = [c for c in classified if c["category"] == "watch"]
    skip      = [c for c in classified if c["category"] == "skip"]

    # ── Format output ─────────────────────────────────────────────────────────
    lines[-1] = "TOP MA HOM NAY:"
    lines.append("─" * 38)

    # Ưu tiên: recommend trước, watch sau, skip cuối
    display_order = (recommend + watch + skip)[:TOP_N_RECOMMEND]

    for c in display_order:
        sym   = c["symbol"]
        score = c["score"]
        wr    = c["wr"]
        ws    = c["wave_str"]
        cat   = c["category"]

        if cat == "recommend":
            prefix = "✅"
        elif cat == "watch":
            prefix = "🟡"
        else:
            prefix = "⛔"

        lines.append(
            f"{prefix} {sym:<5} Score {score:.1f} | WR {wr:.0%}"
        )
        lines.append(f"   {ws}")
        if cat != "skip":
            lines.append(f"   → {c['action']}")
        lines.append("")

    # ── Action summary ────────────────────────────────────────────────────────
    lines.append("─" * 38)
    lines.append("BUOC TIEP THEO:")

    if recommend:
        top2 = recommend[:2]
        analog_cmds = " | ".join(f"/analog {c['symbol']}" for c in top2)
        lines.append(f"  Chay: {analog_cmds}")
    elif watch:
        top1 = watch[0]
        lines.append(f"  Xem them: /wave {top1['symbol']} roi /analog {top1['symbol']}")
    else:
        lines.append("  Khong co ma uu tien hom nay.")

    advice = _regime_advice(regime)
    if advice:
        lines.append(f"  Size: {advice}")

    lines.append("  Portfolio: /portfolio")

    # ── Footer ────────────────────────────────────────────────────────────────
    elapsed = round(time.time() - t0, 1)
    lines.append("")
    lines.append(f"[{elapsed}s | /morning de refresh]")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def morning_cmd(update, context):
    """
    /morning — Morning briefing: regime + top mã + wave filter + gợi ý hành động.

    Không cần tham số. Chạy khoảng 30-90 giây lần đầu (build wave cache).
    Các lần sau nhanh hơn vì dùng cache.
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update); return

    msg = await update.message.reply_text(
        "Dang chuan bi Morning Briefing...\n"
        "(Regime → Scan → Wave filter | ~30-90s lan dau)"
    )

    try:
        briefing = await asyncio.to_thread(build_morning_briefing)
    except Exception as e:
        import traceback
        logger.error(f"morning_cmd error: {e}\n{traceback.format_exc()}")
        await msg.edit_text(f"Loi khi chay /morning: {str(e)[:200]}")
        return

    if len(briefing) <= 4096:
        try:
            await msg.edit_text(briefing)
        except Exception:
            await update.message.reply_text(briefing[:4096])
    else:
        # Split tại dòng trống
        split_at = briefing.rfind("\n\n", 0, 4000)
        if split_at < 0:
            split_at = 4000
        part1 = briefing[:split_at].strip()
        part2 = briefing[split_at:].strip()
        try:
            await msg.edit_text(part1)
        except Exception:
            await update.message.reply_text(part1)
        if part2:
            await update.message.reply_text(part2[:4096])


# ══════════════════════════════════════════════════════════════════════════════
# CRON — 8:15 AM tự động (chạy sau scan cron 8:00)
# ══════════════════════════════════════════════════════════════════════════════

async def _start_morning_cron(bot, chat_ids: list[int]):
    """
    Cron job gửi morning briefing tự động lúc 8:15 AM.
    Chạy SAU scan cron (8:00) để có kết quả scan mới nhất.

    Gọi từ bot.py post_init.
    """
    import datetime as _dt

    logger.info(f"Morning cron started: {len(chat_ids)} users, daily 08:{MORNING_MINUTE:02d}")

    _sent_today: set[str] = set()   # track ngày đã gửi

    while True:
        try:
            now    = _dt.datetime.now()
            today  = now.strftime("%Y-%m-%d")

            # Chạy đúng giờ và chưa gửi hôm nay
            if (now.hour == MORNING_HOUR
                    and now.minute >= MORNING_MINUTE
                    and today not in _sent_today):

                logger.info("[MorningCron] Building briefing...")
                try:
                    briefing = await asyncio.to_thread(build_morning_briefing)
                except Exception as e:
                    briefing = f"Loi Morning Briefing: {str(e)[:200]}"
                    logger.error(f"[MorningCron] build error: {e}")

                for cid in chat_ids:
                    try:
                        # Split nếu dài
                        if len(briefing) <= 4096:
                            await bot.send_message(chat_id=cid, text=briefing)
                        else:
                            split_at = briefing.rfind("\n\n", 0, 4000)
                            if split_at < 0: split_at = 4000
                            await bot.send_message(chat_id=cid,
                                                   text=briefing[:split_at].strip())
                            await asyncio.sleep(0.5)
                            await bot.send_message(chat_id=cid,
                                                   text=briefing[split_at:].strip()[:4096])
                        logger.info(f"[MorningCron] Sent to {cid}")
                    except Exception as e:
                        logger.warning(f"[MorningCron] send {cid} fail: {e}")

                _sent_today.add(today)
                # Xóa ngày cũ khỏi set (giữ set nhỏ)
                _sent_today.discard(
                    (_dt.datetime.now() - _dt.timedelta(days=2)).strftime("%Y-%m-%d")
                )

        except Exception as e:
            logger.error(f"[MorningCron] outer error: {e}")

        # Check mỗi 5 phút
        await asyncio.sleep(5 * 60)
