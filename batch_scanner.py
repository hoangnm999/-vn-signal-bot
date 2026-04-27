"""
batch_scanner.py — Batch Historical Analog Scanner với 5-layer Guard Rails.

Tích hợp:
  - /scan_watchlist  : chạy thủ công qua Telegram
  - Cron 8:00 AM     : tự động mỗi sáng trước giờ mở cửa

Pipeline mỗi mã:
  1. Load watchlist từ WATCHLIST env var
  2. Kiểm tra cache vector + /check còn mới không
  3. find_similar() với bậc thang ngưỡng 80→75→70%
  4. 5 lớp Guard Rails (guardrails.py)
  5. Rank + format output → gửi Telegram

Guard Rails:
  Layer 1 — Data Quality Gates (volume, cache, age)
  Layer 2 — Statistical Sanity (outlier, dispersion, dead-cat, recency)
  Layer 3 — Reference Score với penalties/caps
  Layer 4 — Output framing (P25/P75, framing ngôn từ)
  Layer 5 — System safeguards (market regime, no-opportunity)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SCAN_TIMEOUT_SECS  = 600    # 10 phút hard limit
MAX_WORKERS        = 3      # parallel workers
CRON_HOUR          = 8      # 8:00 AM
CRON_MINUTE        = 0
COOLDOWN_SECS      = 300    # 5 phút giữa 2 lần scan thủ công

_last_scan_time: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_watchlist() -> list[str]:
    """
    Load danh sách mã từ WATCHLIST env var.
    Fallback về danh sách mặc định nếu không có.
    """
    raw = os.environ.get("WATCHLIST", "VCB,HPG,FPT,HAH,STB,DVP")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    # Deduplicate giữ thứ tự
    seen, unique = set(), []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    logger.info(f"[BatchScanner] Watchlist: {len(unique)} mã: {', '.join(unique)}")
    return unique


# ══════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL WORKER (chạy trong thread)
# ══════════════════════════════════════════════════════════════════════════════

def _scan_one_symbol(symbol: str) -> dict:
    """
    Xử lý đầy đủ một mã: check cache → find_similar → guardrails.
    Chạy trong ThreadPoolExecutor.
    Returns dict kết quả đầy đủ.
    """
    t0 = time.time()
    result_base = {
        "symbol": symbol, "gate": "UNKNOWN", "elapsed": 0.0,
        "analogs": [], "stats": {}, "flags": [], "score": 0.0,
        "penalties": [], "risk_tier": "EXCLUDED", "n_total": 0,
        "volume_avg_bill": None, "check_age_hours": None, "cache_days": None,
    }

    try:
        # ── 1. Load dữ liệu giá để lấy volume + giá hiện tại ─────────────
        volume_avg_bill = None
        cache_days      = None
        check_age_hours = None
        current_price   = 0.0

        try:
            from vn_loader import load_vn_ohlcv
            df = load_vn_ohlcv(symbol, days=60, min_bars=20)
            if df is not None and len(df) >= 20:
                close_arr  = df["close"].values[-20:]
                vol_arr    = df["volume"].values[-20:]
                # Volume TB 20 phiên × giá → tỷ VND
                avg_vol_vnd = float((vol_arr * close_arr).mean())
                volume_avg_bill = round(avg_vol_vnd / 1e9, 2)
                current_price   = float(df["close"].iloc[-1])
        except Exception as _ve:
            logger.debug(f"[{symbol}] volume load fail: {_ve}")

        # ── 2. Kiểm tra /check gần nhất ──────────────────────────────────
        try:
            from db import get_conn
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT created_at FROM signals "
                    "WHERE symbol=? ORDER BY created_at DESC LIMIT 1",
                    (symbol,)
                ).fetchone()
                if row and row[0]:
                    created = row[0]
                    if isinstance(created, str):
                        created = datetime.fromisoformat(created)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    check_age_hours = (
                        datetime.now(timezone.utc) - created
                    ).total_seconds() / 3600
        except Exception as _de:
            logger.debug(f"[{symbol}] db check fail: {_de}")

        # ── 3. Nếu /check quá cũ (> 48h) → chạy lại ─────────────────────
        from guardrails import MAX_CHECK_AGE_HOURS
        if check_age_hours is None or check_age_hours > MAX_CHECK_AGE_HOURS:
            try:
                logger.info(f"[{symbol}] /check cũ ({check_age_hours:.0f}h → refresh)" if check_age_hours is not None else f"[{symbol}] /check chưa có → chạy mới")
                from analyzer import analyze_stock_full
                analyze_stock_full(symbol)
                check_age_hours = 0.1   # vừa refresh
            except Exception as _ae:
                logger.warning(f"[{symbol}] analyze fail: {_ae}")

        # ── 4. Kiểm tra cache vector ──────────────────────────────────────
        try:
            from historical_analog import cache_exists, build_vector_cache, _cache_path
            import pathlib, pandas as pd
            if cache_exists(symbol):
                cp   = _cache_path(symbol)
                meta = pd.read_csv(cp, usecols=["date"])
                cache_days = len(meta)
            else:
                logger.info(f"[{symbol}] cache chưa có → build")
                ok, msg = build_vector_cache(symbol)
                if ok:
                    meta       = pd.read_csv(_cache_path(symbol), usecols=["date"])
                    cache_days = len(meta)
                else:
                    result_base.update({
                        "gate": "REJECT", "elapsed": round(time.time() - t0, 1),
                        "reason": f"Build cache thất bại: {msg[:100]}",
                    })
                    return result_base
        except Exception as _ce:
            logger.warning(f"[{symbol}] cache check fail: {_ce}")

        # ── 5. Load state vector ─────────────────────────────────────────
        # Ưu tiên: load_auto_context (hàm chính thống) → DB query → compute trực tiếp
        state_vec = None
        sv_source = "none"

        # 5a. Thử load_auto_context — dùng cùng logic với backtest_rule
        try:
            from auto_context import load_auto_context
            ctx = load_auto_context(symbol)
            if ctx and ctx.get("found") and ctx.get("state_vector"):
                state_vec = ctx["state_vector"]
                sv_source = "auto_context"
                logger.info(f"[{symbol}] state_vec from auto_context OK")
        except Exception as _ace:
            logger.debug(f"[{symbol}] auto_context fail: {_ace}")

        # 5b. Fallback: query DB trực tiếp (thử nhiều column names)
        if state_vec is None:
            try:
                import json as _json
                from db import get_conn
                with get_conn() as conn:
                    # Thử column "state_vector" trước, sau đó "state_vec"
                    for col in ("state_vector", "state_vec", "vector_json"):
                        try:
                            row = conn.execute(
                                f"SELECT {col} FROM signals "
                                f"WHERE symbol=? AND {col} IS NOT NULL "
                                f"ORDER BY created_at DESC LIMIT 1",
                                (symbol,)
                            ).fetchone()
                            if row and row[0]:
                                sv = row[0]
                                state_vec = _json.loads(sv) if isinstance(sv, str) else sv
                                sv_source = f"db.{col}"
                                logger.info(f"[{symbol}] state_vec from {sv_source}")
                                break
                        except Exception:
                            continue
            except Exception as _dbe:
                logger.warning(f"[{symbol}] db state_vec fail: {_dbe}")

        # 5c. Fallback cuối: tính vector từ OHLCV trực tiếp
        if state_vec is None:
            try:
                from state_vector import compute_state_vector_from_df
                from vn_loader import load_vn_ohlcv
                df_sv     = load_vn_ohlcv(symbol, days=120, min_bars=60)
                state_vec = compute_state_vector_from_df(df_sv) if df_sv is not None else None
                if state_vec:
                    sv_source = "computed_direct"
                    logger.info(f"[{symbol}] state_vec computed directly")
            except Exception as _fbe:
                logger.warning(f"[{symbol}] compute state_vec fail: {_fbe}")

        if state_vec is None:
            reason = f"Không lấy được state vector (tried: auto_context, db, compute)"
            logger.warning(f"[{symbol}] REJECT: {reason}")
            result_base.update({
                "gate": "REJECT", "elapsed": round(time.time() - t0, 1),
                "reason": reason,
            })
            return result_base

        # ── 6. find_similar với bậc thang ngưỡng ─────────────────────────
        analogs = None
        try:
            from historical_analog import find_similar
            analogs = find_similar(
                symbol        = symbol,
                target_vector = state_vec,
                years         = 5,
                exclude_days  = 90,
                min_results   = 3,
            )
        except Exception as _fe:
            logger.warning(f"[{symbol}] find_similar fail: {_fe}")

        if not analogs:
            reason = f"find_similar trả về None (cache_days={cache_days}, sv_source={sv_source})"
            logger.warning(f"[{symbol}] REJECT: {reason}")
            result_base.update({
                "gate": "REJECT", "elapsed": round(time.time() - t0, 1),
                "reason": reason,
            })
            return result_base

        # Gắn giá hiện tại vào analogs nếu chưa có
        if current_price > 0:
            for a in analogs:
                if a.get("close", 0) == 0:
                    a["close"] = current_price

        result_base.update({
            "gate":            "PASS",   # BUG FIX: was left as "UNKNOWN" → excluded from valid_results
            "analogs":         analogs,
            "volume_avg_bill": volume_avg_bill,
            "cache_days":      cache_days,
            "check_age_hours": check_age_hours,
            "elapsed":         round(time.time() - t0, 1),
        })
        return result_base

    except Exception as e:
        import traceback
        logger.error(f"[{symbol}] _scan_one_symbol ERROR: {e}\n{traceback.format_exc()}")
        result_base.update({
            "gate": "ERROR", "elapsed": round(time.time() - t0, 1),
            "reason": str(e)[:200],
        })
        return result_base


# ══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_batch_scan(
    symbols:    list[str],
    progress_cb = None,    # callback(msg: str) cho Telegram progress
) -> dict:
    """
    Chạy scan song song tối đa MAX_WORKERS workers.
    Timeout cứng SCAN_TIMEOUT_SECS.

    Returns:
        {
          "ranked":   list[dict],   # mã qua guard rails, sort by score
          "excluded": list[dict],   # mã MDD quá cao
          "rejected": list[dict],   # mã không qua gate
          "errors":   list[str],
          "elapsed":  float,
          "partial":  bool,         # True nếu bị timeout
        }
    """
    t_global = time.time()
    raw_results: dict[str, dict] = {}

    def _progress(msg: str):
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass
        logger.info(f"[BatchScan] {msg}")

    _progress(f"Bắt đầu scan {len(symbols)} mã với {MAX_WORKERS} workers...")

    # Chạy song song
    partial = False
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_scan_one_symbol, sym): sym for sym in symbols}
        deadline = t_global + SCAN_TIMEOUT_SECS

        for future in as_completed(futures, timeout=SCAN_TIMEOUT_SECS):
            sym = futures[future]
            try:
                res = future.result(timeout=max(1, deadline - time.time()))
                raw_results[sym] = res
                elapsed = res.get("elapsed", 0)
                gate    = res.get("gate", "?")
                _progress(f"  {sym}: {gate} ({elapsed:.0f}s)")
            except FuturesTimeout:
                raw_results[sym] = {"symbol": sym, "gate": "TIMEOUT",
                                    "reason": "Vượt quá thời gian 10 phút"}
                partial = True
                _progress(f"  {sym}: TIMEOUT")
            except Exception as e:
                raw_results[sym] = {"symbol": sym, "gate": "ERROR",
                                    "reason": str(e)[:100]}
                _progress(f"  {sym}: ERROR - {e}")

    # Tập hợp stats để tính z-score
    all_stats = []
    valid_results = {}
    for sym, res in raw_results.items():
        if res.get("gate") not in ("REJECT", "TIMEOUT", "ERROR", "UNKNOWN") \
                and res.get("analogs"):
            from guardrails import compute_base_stats
            stats = compute_base_stats(res["analogs"])
            if stats.get("valid"):
                all_stats.append(stats)
                valid_results[sym] = res

    _progress(f"Tính Guard Rails cho {len(valid_results)} mã hợp lệ...")

    # Chạy guardrails với z-score đầy đủ
    ranked   = []
    excluded = []
    rejected = []
    errors   = []

    for sym, res in raw_results.items():
        if res.get("gate") in ("TIMEOUT", "ERROR"):
            errors.append(f"{sym}: {res.get('reason', '?')}")
            continue

        if res.get("gate") == "REJECT":
            rejected.append({
                "symbol": sym,
                "reason": res.get("reason", "Gate reject"),
            })
            continue

        if sym not in valid_results:
            rejected.append({"symbol": sym, "reason": "Không có analogs hợp lệ"})
            continue

        from guardrails import run_guardrails_for_symbol
        gr = run_guardrails_for_symbol(
            symbol          = sym,
            analogs         = res["analogs"],
            all_stats       = all_stats,
            volume_avg_bill = res.get("volume_avg_bill"),
            cache_days      = res.get("cache_days"),
            check_age_hours = res.get("check_age_hours"),
        )

        if gr["gate"] == "REJECT":
            rejected.append({"symbol": sym, "reason": gr.get("reason", "Gate reject")})
            continue

        if gr["risk_tier"] == "EXCLUDED":
            excluded.append(gr)
        else:
            ranked.append(gr)

    # Sort by score desc
    ranked.sort(key=lambda x: x["score"], reverse=True)
    excluded.sort(key=lambda x: x["score"], reverse=True)

    elapsed = round(time.time() - t_global, 1)
    _progress(f"Scan xong: {len(ranked)} mã vào rank, {len(excluded)} excluded, "
              f"{len(rejected)} reject | {elapsed}s")

    return {
        "ranked":   ranked,
        "excluded": excluded,
        "rejected": rejected,
        "errors":   errors,
        "elapsed":  elapsed,
        "partial":  partial,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM OUTPUT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def format_scan_report(scan_result: dict) -> list[str]:
    """
    Format kết quả scan thành list messages (tự chia nếu > 4000 ký tự).
    Returns list[str] để gửi nhiều message nếu cần.
    """
    from guardrails import (
        format_full_report, check_no_opportunity, check_market_regime,
        SCORE_OPPORTUNITY_MIN,
    )

    ranked   = scan_result["ranked"]
    excluded = scan_result["excluded"]
    rejected = scan_result["rejected"]
    errors   = scan_result["errors"]
    partial  = scan_result["partial"]
    elapsed  = scan_result["elapsed"]

    # Layer 5: Market regime check
    market_warn = None
    try:
        from vn_loader import load_vn_ohlcv
        df_vni = load_vn_ohlcv("VNINDEX", days=10, min_bars=5)
        if df_vni is not None and len(df_vni) >= 6:
            close_5d = df_vni["close"].values
            chg_5d   = (close_5d[-1] - close_5d[-6]) / close_5d[-6] * 100
            market_warn = check_market_regime(chg_5d)
    except Exception:
        pass

    scan_date = datetime.now().strftime("%d/%m/%Y %H:%M")

    # Kiểm tra no-opportunity
    if check_no_opportunity(ranked):
        no_opp_msg = (
            f"🔍 SCAN WATCHLIST ({scan_date})\n"
            f"Thời gian: {elapsed:.0f}s\n\n"
        )
        if market_warn:
            no_opp_msg += f"⚠️ {market_warn}\n\n"
        no_opp_msg += (
            f"Không tìm thấy cơ hội nào đủ tin cậy hôm nay "
            f"(ngưỡng Score > {SCORE_OPPORTUNITY_MIN}/10).\n"
            f"Hãy kiên nhẫn chờ đợi."
        )
        return [no_opp_msg]

    # Main report
    main_report = format_full_report(
        ranked         = ranked,
        excluded       = excluded,
        market_warning = market_warn,
        scan_date      = scan_date,
    )

    # Footer: rejected + errors + timing
    footer_lines = [f"\n⏱️ Thời gian scan: {elapsed:.0f}s"]
    if partial:
        footer_lines.append("⚠️ Kết quả chưa đầy đủ — một số mã bị timeout.")
    if rejected:
        footer_lines.append(f"⏭️ Bị loại ({len(rejected)} mã):")
        for r in rejected[:6]:
            reason = r.get("reason", "unknown")[:60]
            footer_lines.append(f"  • {r['symbol']}: {reason}")
        if len(rejected) > 6:
            footer_lines.append(f"  ... và {len(rejected)-6} mã khác")
    if errors:
        footer_lines.append(f"❌ Lỗi: {'; '.join(errors[:3])}")

    full = main_report + "\n".join(footer_lines)

    # Tự chia messages nếu > 4000 ký tự
    MAX_LEN = 4000
    if len(full) <= MAX_LEN:
        return [full]

    # Chia: main report + footer riêng
    msgs = []
    if len(main_report) <= MAX_LEN:
        msgs.append(main_report)
    else:
        # Chia theo từng mã (split by "\n\n#")
        parts = main_report.split("\n\n")
        buf   = ""
        for part in parts:
            if len(buf) + len(part) + 2 > MAX_LEN:
                if buf:
                    msgs.append(buf.strip())
                buf = part
            else:
                buf = (buf + "\n\n" + part).strip() if buf else part
        if buf:
            msgs.append(buf.strip())

    footer_str = "\n".join(footer_lines)
    if footer_str.strip():
        if msgs and len(msgs[-1]) + len(footer_str) <= MAX_LEN:
            msgs[-1] += footer_str
        else:
            msgs.append(footer_str)

    return msgs if msgs else [full[:MAX_LEN]]


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def scan_watchlist_cmd(update, context):
    """
    /scan_watchlist — Chạy batch scan thủ công.
    Có cooldown 5 phút giữa các lần gọi.
    """
    global _last_scan_time

    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    # Cooldown
    since = time.time() - _last_scan_time
    if since < COOLDOWN_SECS:
        wait = int(COOLDOWN_SECS - since)
        await update.message.reply_text(
            f"⏳ Vui lòng chờ {wait}s trước khi scan lại.\n"
            f"(Cooldown {COOLDOWN_SECS//60} phút)"
        )
        return

    _last_scan_time = time.time()
    chat_id = update.effective_chat.id

    symbols = load_watchlist()
    msg = await update.message.reply_text(
        f"🔍 Đang scan {len(symbols)} mã trong watchlist...\n"
        f"Tối đa {SCAN_TIMEOUT_SECS // 60} phút. Vui lòng chờ."
    )

    # Progress callback (edit message liên tục)
    progress_lines: list[str] = [
        f"🔍 Scan {len(symbols)} mã — đang chạy...\n"
    ]

    async def _progress_async(line: str):
        progress_lines.append(line)
        try:
            preview = "\n".join(progress_lines[-8:])
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg.message_id,
                text=preview[:4000],
            )
        except Exception:
            pass

    # Wrapper sync → async (ThreadPoolExecutor gọi sync callback)
    loop = asyncio.get_event_loop()

    def _progress_sync(line: str):
        asyncio.run_coroutine_threadsafe(_progress_async(line), loop)

    # Chạy scan trong thread
    try:
        scan_result = await asyncio.to_thread(
            run_batch_scan, symbols, _progress_sync
        )
    except Exception as e:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=f"❌ Lỗi scan: {str(e)[:300]}",
        )
        return

    # Format và gửi
    messages = format_scan_report(scan_result)

    # Edit message đầu bằng msg[0]
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=msg.message_id,
            text=messages[0][:4000],
        )
    except Exception:
        await context.bot.send_message(chat_id=chat_id, text=messages[0][:4000])

    # Gửi thêm nếu có nhiều message
    for extra_msg in messages[1:]:
        try:
            await context.bot.send_message(chat_id=chat_id, text=extra_msg[:4000])
        except Exception as e:
            logger.warning(f"scan_watchlist_cmd: send extra msg fail: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CRON JOB — 8:00 AM
# ══════════════════════════════════════════════════════════════════════════════

async def _start_scan_cron(bot, chat_ids: list[int]):
    """
    Async loop cron — gọi từ bot.py khi khởi động.
    Chạy scan tự động lúc CRON_HOUR:CRON_MINUTE mỗi ngày.

    Args:
        bot:      telegram.Bot instance
        chat_ids: list chat_id để gửi kết quả
    """
    import datetime as _dt

    while True:
        now    = _dt.datetime.now()
        target = now.replace(
            hour=CRON_HOUR, minute=CRON_MINUTE, second=0, microsecond=0
        )
        if now >= target:
            target += _dt.timedelta(days=1)

        wait_secs = (target - now).total_seconds()
        logger.info(
            f"[ScanCron] Next run in {wait_secs/3600:.1f}h "
            f"({target.strftime('%d/%m %H:%M')})"
        )
        await asyncio.sleep(wait_secs)

        logger.info("[ScanCron] Bắt đầu chạy auto scan...")
        try:
            symbols     = load_watchlist()
            scan_result = await asyncio.to_thread(run_batch_scan, symbols)
            messages    = format_scan_report(scan_result)

            for cid in chat_ids:
                for m in messages:
                    try:
                        await bot.send_message(chat_id=cid, text=m[:4000])
                        await asyncio.sleep(0.3)   # tránh flood
                    except Exception as _se:
                        logger.warning(f"[ScanCron] send to {cid} fail: {_se}")

        except Exception as e:
            import traceback
            logger.error(f"[ScanCron] ERROR: {e}\n{traceback.format_exc()}")
            err_msg = f"❌ Auto scan lỗi ({_dt.datetime.now().strftime('%H:%M')}): {str(e)[:200]}"
            for cid in chat_ids:
                try:
                    await bot.send_message(chat_id=cid, text=err_msg)
                except Exception:
                    pass
