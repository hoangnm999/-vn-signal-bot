"""
morning_briefing.py — /morning command cho VN Signal Bot

THIẾT KẾ ĐỂ NHANH (< 60 giây):
  - KHÔNG chạy full batch scan (mất 5-30 phút)
  - Tái dụng kết quả scan từ _last_scan_result (shared với scan cron 8:00)
  - Nếu chưa có scan result → dùng watchlist thô + wave cache
  - Wave: chỉ đọc cache (force_rebuild=False), cache miss → ghi chú, không chờ

Pipeline:
  1. market_regime  → cache 6h (~2s)
  2. Top mã         → từ _last_scan_result hoặc watchlist (~0.1s)
  3. Wave (cache)   → parallel 4 workers, timeout 15s/mã
  4. Format + gửi

Timeout cứng: 90 giây.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

logger = logging.getLogger(__name__)

WAVE_CONF_CLEAR    = 0.30
WAVE_CONF_NOISE    = 0.20
TOP_N_SCAN         = 7
TOP_N_RECOMMEND    = 5
MORNING_HOUR       = 8
MORNING_MINUTE     = 15
WAVE_TIMEOUT_SECS  = 8     # timeout cứng per symbol — cache miss → skip ngay
TOTAL_TIMEOUT_SECS = 45    # tổng timeout — bot PHẢI trả lời trong 45s

# POLICY: /morning CHỈ đọc cache có sẵn, KHÔNG chạy gì nặng
# - Scan cache: từ _last_scan_result (scan cron 8:00 hoặc /scan_watchlist)
# - Wave cache: chỉ đọc file _waves.json, KHÔNG build mới
# Nếu chưa có cache → hiển thị watchlist + hướng dẫn chạy scan/wave trước
MORNING_MIN_WEIGHTED_N = 5.0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Regime
# ══════════════════════════════════════════════════════════════════════════════

def _get_regime() -> dict:
    try:
        from market_regime import get_market_regime
        mr = get_market_regime()
        if mr:
            return mr
    except Exception as e:
        logger.warning(f"morning: regime fail: {e}")
    return {"regime": 0, "label": "Khong xac dinh", "emoji": "❓", "is_bull": False}


def _regime_gate(regime: int) -> tuple[bool, str]:
    gates = {
        1: (True,  "Den XANH — R1 Bull Quiet, tin hieu mua tin cay nhat"),
        2: (True,  "Den VANG — R2 Bull Volatile, giam size 20-30%"),
        3: (True,  "Den CAM — R3 Bear Quiet, chi trade setup rat ro, size nho"),
        4: (False, "Den DO — R4 Bear Volatile, KHONG mo position moi"),
        0: (True,  "Khong xac dinh duoc regime — can than"),
    }
    return gates.get(regime, (True, ""))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Top mã (không scan mới)
# ══════════════════════════════════════════════════════════════════════════════

def _get_top_symbols_fast(n: int = TOP_N_SCAN) -> tuple[list[dict], bool]:
    """
    Lấy top N mã TỪ CACHE — KHÔNG chạy scan mới.
    has_score=True nếu có cache scan, False nếu chỉ là watchlist thô.

    Đọc qua get_last_scan_result() — hỗ trợ cả in-memory lẫn file cache
    (tồn tại qua bot restart). Nếu chưa có cache → trả về watchlist thô.
    User cần chạy /scan_watchlist trước để có cache.
    """
    # Đọc qua get_last_scan_result() — PostgreSQL backend, tồn tại qua Render deploy
    try:
        from batch_scanner import get_last_scan_result
        # Thử watchlist trước, fallback hose nếu watchlist chưa có
        scan_result = get_last_scan_result(scan_type="watchlist")
        if not scan_result or not scan_result.get("ranked"):
            scan_result = get_last_scan_result(scan_type="hose")
        if scan_result and isinstance(scan_result, dict):
            ranked = scan_result.get("ranked", [])
            if ranked:
                top = []
                skipped = []
                for r in ranked:
                    if len(top) >= n:
                        break
                    s          = r.get("stats", {})
                    weighted_n = s.get("weighted_n", float(s.get("n", 0)))
                    if weighted_n < MORNING_MIN_WEIGHTED_N:
                        skipped.append(f"{r['symbol']}(wN={weighted_n:.1f})")
                        continue
                    top.append({
                        "symbol":     r["symbol"],
                        "score":      r.get("score", 0.0),
                        "wr":         s.get("win_rate", 0.0),
                        "rr":         s.get("rr", 0.0),
                        "exp":        s.get("weighted_exp", 0.0),
                        "weighted_n": weighted_n,
                    })
                if skipped:
                    logger.info(f"morning: loai {len(skipped)} ma wN<5: {', '.join(skipped)}")
                if top:
                    logger.info(f"morning: {len(top)} symbols from scan cache")
                    return top, True
    except Exception as e:
        logger.debug(f"morning: scan cache miss: {e}")

    # Fallback: watchlist thô (không có score — cần /scan_watchlist trước)
    try:
        from batch_scanner import load_watchlist
        symbols = load_watchlist()[:n]
        logger.info(f"morning: no scan cache, using raw watchlist ({len(symbols)} symbols)")
        return [{"symbol": s, "score": 0.0, "wr": 0.0, "rr": 0.0,
                 "exp": 0.0, "weighted_n": 0.0}
                for s in symbols], False
    except Exception as e:
        logger.warning(f"morning: load_watchlist fail: {e}")
        return [], False


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Wave (cache only)
# ══════════════════════════════════════════════════════════════════════════════

def _get_wave_cached(symbol: str) -> dict:
    """
    CHỈ đọc wave cache — KHÔNG build mới nếu chưa có.
    Cache miss → trả về KHONG RO ngay (< 1ms).
    Dùng timeout trong ThreadPoolExecutor để đảm bảo không block.
    """
    _miss = {"symbol": symbol, "verdict": "KHONG RO",
             "confidence": 0.0, "cache_miss": True}
    try:
        import pathlib
        cache_path = pathlib.Path("data") / f"{symbol.upper()}_waves.json"
        if not cache_path.exists():
            return _miss   # cache chưa có → skip ngay, không build

        from wave_pattern import analyze_wave
        result     = analyze_wave(symbol, force_rebuild=False)
        if not result or not result.get("ok"):
            return _miss

        verdict    = result.get("verdict", "KHONG RO")
        score_up   = result.get("score_up", 0.0)
        score_down = result.get("score_down", 0.0)
        confidence = result.get("confidence", abs(score_up - score_down))

        if confidence < WAVE_CONF_NOISE:
            verdict = "KHONG RO"
        elif confidence < WAVE_CONF_CLEAR:
            verdict = verdict + "_WEAK"

        return {"symbol": symbol, "verdict": verdict,
                "score_up": score_up, "score_down": score_down,
                "confidence": round(confidence, 3), "cache_miss": False}
    except Exception as e:
        logger.warning(f"morning: wave {symbol}: {e}")
        return _miss


def _get_wave_parallel_fast(symbols: list[str]) -> dict[str, dict]:
    """
    Parallel wave đọc cache. Timeout cứng WAVE_TIMEOUT_SECS per symbol.
    Cache miss → KHONG RO ngay (< 1ms). Không bao giờ block lâu.
    """
    if not symbols:
        return {}
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(symbols))) as ex:
        future_map = {ex.submit(_get_wave_cached, sym): sym for sym in symbols}
        for future in as_completed(future_map, timeout=WAVE_TIMEOUT_SECS * len(symbols)):
            sym = future_map[future]
            try:
                results[sym] = future.result(timeout=WAVE_TIMEOUT_SECS)
            except Exception:
                results[sym] = {"symbol": sym, "verdict": "KHONG RO",
                                "confidence": 0.0, "cache_miss": True}
    # Đảm bảo tất cả symbols đều có kết quả
    for sym in symbols:
        if sym not in results:
            results[sym] = {"symbol": sym, "verdict": "KHONG RO",
                            "confidence": 0.0, "cache_miss": True}
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFY + FORMAT
# ══════════════════════════════════════════════════════════════════════════════

def _wave_stars(verdict: str, confidence: float) -> str:
    is_weak   = verdict.endswith("_WEAK")
    base      = verdict.replace("_WEAK", "")
    if base == "SONG TANG":
        direction = "TANG"
    elif base == "SONG GIAM":
        direction = "GIAM"
    else:
        return "Wave ?"
    stars = "★★★" if confidence >= WAVE_CONF_CLEAR * 1.3 else "★★☆"
    if is_weak:
        stars = "★☆☆"
    return f"Wave {stars} {direction} (conf {confidence:.2f})"


def _classify(sym_data: dict, wave: dict) -> dict:
    symbol    = sym_data["symbol"]
    verdict   = wave.get("verdict", "KHONG RO")
    conf      = wave.get("confidence", 0.0)
    base_v    = verdict.replace("_WEAK", "")
    is_weak   = verdict.endswith("_WEAK")
    cache_miss = wave.get("cache_miss", False)

    if base_v == "SONG GIAM" and conf >= WAVE_CONF_CLEAR:
        return {"category": "skip",
                "wave_str": _wave_stars(verdict, conf),
                "action":   "Bo qua"}

    if base_v == "SONG TANG" and conf >= WAVE_CONF_CLEAR and not is_weak:
        return {"category": "recommend",
                "wave_str": _wave_stars(verdict, conf),
                "action":   f"/analog {symbol}"}

    if cache_miss:
        return {"category": "watch",
                "wave_str": "Wave chua co cache",
                "action":   f"/wave {symbol} truoc, roi /analog {symbol}"}

    return {"category": "watch",
            "wave_str": _wave_stars(verdict, conf) if base_v != "KHONG RO" else "Wave KHONG RO",
            "action":   f"/wave {symbol} roi /analog {symbol}"}


def _regime_advice(regime: int) -> str:
    return {
        1: "Full size binh thuong",
        2: "Giam size 20-30%, SL chat hon",
        3: "Size nho, chi setup rat ro rang",
        4: "Khong mo position moi",
        0: "Can than",
    }.get(regime, "")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BUILD
# ══════════════════════════════════════════════════════════════════════════════

def build_morning_briefing() -> str:
    t0    = time.time()
    today = datetime.now().strftime("%d/%m/%Y")
    lines = [f"MORNING BRIEFING — {today}", "=" * 38, ""]

    # Step 1: Regime
    regime_data    = _get_regime()
    regime         = regime_data.get("regime", 0)
    r_label        = regime_data.get("label", "Khong xac dinh")
    r_emoji        = regime_data.get("emoji", "❓")
    should_go, gate_msg = _regime_gate(regime)

    lines += [f"MARKET REGIME:", f"  {r_emoji} {r_label}", f"  {gate_msg}", ""]
    logger.info(f"morning step1 regime: {time.time()-t0:.1f}s")

    if not should_go:
        lines += ["=" * 38, "KHONG TRADE HOM NAY.",
                  "Uu tien bao ve von, cho regime thay doi.", "",
                  "Dung /regime de xem chi tiet.",
                  "Dung /portfolio de kiem tra vi the."]
        return "\n".join(lines)

    # Step 2: Top mã
    top_symbols, has_score = _get_top_symbols_fast(TOP_N_SCAN)
    logger.info(f"morning step2 symbols: {time.time()-t0:.1f}s (has_score={has_score})")

    if not top_symbols:
        lines += ["Khong lay duoc danh sach ma.",
                  "Goi y: /scan_watchlist de chay scan thu cong."]
        return "\n".join(lines)

    if not has_score:
        lines += [
            "Chua co ket qua scan — hien thi watchlist khong co score.",
            "Chay /scan_watchlist de co du lieu day du.",
            "",
        ]

    # Timeout check trước wave
    if time.time() - t0 > TOTAL_TIMEOUT_SECS - 20:
        lines += ["TOP MA HOM NAY (het thoi gian lay wave):", "─" * 38]
        for s in top_symbols[:TOP_N_RECOMMEND]:
            lines.append(f"🟡 {s['symbol']:<5} → /wave {s['symbol']} roi /analog {s['symbol']}")
        lines += ["", f"[{time.time()-t0:.0f}s | /morning de refresh]"]
        return "\n".join(lines)

    # Step 3: Wave parallel
    sym_list  = [s["symbol"] for s in top_symbols]
    wave_data = _get_wave_parallel_fast(sym_list)
    logger.info(f"morning step3 wave: {time.time()-t0:.1f}s")

    # Step 4: Classify + format
    classified  = [{**s, **_classify(s, wave_data.get(s["symbol"], {}))}
                   for s in top_symbols]
    recommend   = [c for c in classified if c["category"] == "recommend"]
    watch       = [c for c in classified if c["category"] == "watch"]
    skip        = [c for c in classified if c["category"] == "skip"]

    lines += ["TOP MA HOM NAY:", "─" * 38]

    for c in (recommend + watch + skip)[:TOP_N_RECOMMEND]:
        sym    = c["symbol"]
        score  = c.get("score", 0.0)
        wr     = c.get("wr", 0.0)
        rr     = c.get("rr", 0.0)
        exp    = c.get("exp", 0.0)
        cat    = c["category"]
        prefix = "✅" if cat == "recommend" else "🟡" if cat == "watch" else "⛔"

        meta = []
        if rr > 0:    meta.append(f"RR {rr:.1f}x")
        if wr > 0:    meta.append(f"WR {wr:.0%}")
        if exp != 0:  meta.append(f"Exp {exp:+.1f}%")
        meta_str = " | ".join(meta)

        lines.append(f"{prefix} {sym:<5} {meta_str}")
        lines.append(f"   {c['wave_str']}")
        if cat != "skip":
            lines.append(f"   → {c['action']}")
        lines.append("")

    # Action summary
    lines += ["─" * 38, "BUOC TIEP THEO:"]
    if recommend:
        cmds = " | ".join(f"/analog {c['symbol']}" for c in recommend[:2])
        lines.append(f"  Chay: {cmds}")
    elif watch:
        lines.append(f"  Xem them: /wave {watch[0]['symbol']} roi /analog {watch[0]['symbol']}")
    else:
        lines.append("  Khong co ma uu tien hom nay.")

    advice = _regime_advice(regime)
    if advice:
        lines.append(f"  Size: {advice}")
    lines.append("  Portfolio: /portfolio")

    if not has_score:
        lines += ["", "⚠️  Chua co ket qua scan. Score/WR se co sau /scan_watchlist."]

    elapsed = round(time.time() - t0, 1)
    lines += ["", f"[{elapsed}s | /morning de refresh]"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def morning_cmd(update, context):
    """
    /morning — Morning briefing nhanh (target < 30s, max 90s).
    Tái dụng scan cache + wave cache. Không chạy gì mới từ đầu.
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
        "(Chi doc cache — phai xong trong 30s)"
    )

    try:
        briefing = await asyncio.wait_for(
            asyncio.to_thread(build_morning_briefing),
            timeout=TOTAL_TIMEOUT_SECS
        )
    except asyncio.TimeoutError:
        briefing = (
            "Morning Briefing timeout (>90s).\n\n"
            "Nguyen nhan thuong gap:\n"
            "  Wave cache chua co — chay /wave <MA> cho tung ma truoc\n\n"
            "Goi y:\n"
            "  1. /scan_watchlist  (chay scan)\n"
            "  2. /wave HAH        (build wave cache)\n"
            "  3. /morning         (chay lai)"
        )
    except Exception as e:
        import traceback
        logger.error(f"morning_cmd: {e}\n{traceback.format_exc()}")
        briefing = f"Loi /morning: {str(e)[:300]}"

    if len(briefing) <= 4096:
        try:
            await msg.edit_text(briefing)
        except Exception:
            await update.message.reply_text(briefing[:4096])
    else:
        split_at = briefing.rfind("\n\n", 0, 4000)
        if split_at < 0: split_at = 4000
        try:
            await msg.edit_text(briefing[:split_at].strip())
        except Exception:
            await update.message.reply_text(briefing[:split_at].strip())
        await update.message.reply_text(briefing[split_at:].strip()[:4096])


# ══════════════════════════════════════════════════════════════════════════════
# CRON — 8:15 AM
# ══════════════════════════════════════════════════════════════════════════════

async def _start_morning_cron(bot, chat_ids: list[int]):
    """Cron 8:15 AM — chạy sau scan cron 8:00 để có _last_scan_result."""
    import datetime as _dt
    logger.info(f"Morning cron: {len(chat_ids)} users, daily 08:{MORNING_MINUTE:02d}")
    _sent_today: set[str] = set()

    while True:
        try:
            now   = _dt.datetime.now()
            today = now.strftime("%Y-%m-%d")
            if (now.hour == MORNING_HOUR
                    and now.minute >= MORNING_MINUTE
                    and today not in _sent_today):

                logger.info("[MorningCron] Building...")
                try:
                    briefing = await asyncio.wait_for(
                        asyncio.to_thread(build_morning_briefing),
                        timeout=TOTAL_TIMEOUT_SECS
                    )
                except Exception as e:
                    briefing = f"Loi Morning Briefing: {str(e)[:200]}"
                    logger.error(f"[MorningCron] {e}")

                for cid in chat_ids:
                    try:
                        if len(briefing) <= 4096:
                            await bot.send_message(chat_id=cid, text=briefing)
                        else:
                            sp = briefing.rfind("\n\n", 0, 4000)
                            if sp < 0: sp = 4000
                            await bot.send_message(chat_id=cid,
                                                   text=briefing[:sp].strip())
                            await asyncio.sleep(0.5)
                            await bot.send_message(chat_id=cid,
                                                   text=briefing[sp:].strip()[:4096])
                    except Exception as e:
                        logger.warning(f"[MorningCron] send {cid}: {e}")

                _sent_today.add(today)
                _sent_today.discard(
                    (_dt.datetime.now() - _dt.timedelta(days=2)).strftime("%Y-%m-%d")
                )
        except Exception as e:
            logger.error(f"[MorningCron] outer: {e}")

        await asyncio.sleep(5 * 60)
