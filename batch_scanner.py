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
SCAN_TIMEOUT_SECS  = 1800   # 30 phút — đủ cho ~50 mã / 3 workers (~60s/mã)
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


def build_hose_watchlist(
    top_n:           int   = 200,
    min_vol_billion: float = 3.0,
    days:            int   = 20,
    progress_cb      = None,   # callback(msg: str) để báo tiến độ cho Telegram
) -> list[str]:
    """
    Tự động lấy top N mã HOSE theo thanh khoản trung bình 20 phiên giao dịch.

    Strategy:
      1. Lấy danh sách mã từ vnstock listing (HOSE)
      2. Với mỗi mã, load OHLCV → tính avg(volume × price) trên đúng 20 phiên cuối
      3. Lọc >= min_vol_billion tỷ/ngày
      4. Sort desc, lấy top_n

    Returns:
        List mã sorted by thanh khoản desc.
        Trả về [] nếu fail (caller dùng fallback).
    """
    logger.info(f"[HoseWatchlist] Building top {top_n} HOSE symbols "
                f"(min_vol={min_vol_billion}ty, sessions={days})...")
    t0 = time.time()

    def _progress(msg: str):
        if progress_cb:
            try: progress_cb(msg)
            except Exception: pass
        logger.info(f"[HoseWatchlist] {msg}")

    # ── Bước 1: Lấy danh sách mã HOSE ────────────────────────────────────────
    all_symbols: list[str] = []
    _progress("Buoc 1/3: Lay danh sach ma HOSE tu vnstock...")
    try:
        from vnstock import Vnstock
        listing = Vnstock().stock(symbol="VCB", source="VCI").listing.symbols_by_exchange()
        hose_df = listing[listing["exchange"].str.upper() == "HOSE"]
        all_symbols = hose_df["symbol"].str.upper().tolist()
        _progress(f"Buoc 1/3: Got {len(all_symbols)} ma HOSE (VCI)")
    except Exception as e:
        logger.warning(f"[HoseWatchlist] VCI listing fail: {e}, trying KBS...")
        try:
            from vnstock import Vnstock
            listing = Vnstock().stock(symbol="VCB", source="KBS").listing.symbols_by_exchange()
            hose_df = listing[listing["exchange"].str.upper() == "HOSE"]
            all_symbols = hose_df["symbol"].str.upper().tolist()
            _progress(f"Buoc 1/3: Got {len(all_symbols)} ma HOSE (KBS fallback)")
        except Exception as e2:
            logger.error(f"[HoseWatchlist] All listing sources fail: {e2}")
            return []

    if not all_symbols:
        logger.error("[HoseWatchlist] Empty symbol list")
        return []

    total = len(all_symbols)

    # ── Bước 2: Load OHLCV tuần tự, tính thanh khoản đúng 20 phiên cuối ──────
    # Sequential thay vì ThreadPoolExecutor để tránh deadlock với asyncio.to_thread
    # analyzer.get_price_data() rất nhanh (~0.1-0.2s/mã) → sequential OK
    from vn_loader import load_vn_ohlcv

    vol_map: dict[str, float] = {}

    def _check_vol(sym: str) -> float:
        try:
            df = load_vn_ohlcv(sym, days=days + 10, min_bars=days)
            if df is None or len(df) < days:
                return 0.0
            close_arr = df["close"].values[-days:]
            vol_arr   = df["volume"].values[-days:]
            avg_vnd   = float((vol_arr * close_arr).mean()) * 1000
            return round(avg_vnd / 1e9, 3)
        except SystemExit:
            # vnai/vnstock gọi sys.exit() khi rate limit — bắt ở đây để không crash bot
            logger.warning(f"[HoseWatchlist] {sym}: rate limit hit (SystemExit) — skip")
            time.sleep(60)   # chờ 60s rồi tiếp tục
            return 0.0
        except BaseException as e:
            logger.warning(f"[HoseWatchlist] {sym}: {type(e).__name__}: {e}")
            return 0.0

    _progress(f"Buoc 2/3: Kiem tra thanh khoan {total} ma ({days} phien cuoi, min={min_vol_billion}ty)...")

    try:
        for i, sym in enumerate(all_symbols, 1):
            vol = _check_vol(sym)
            if vol >= min_vol_billion:
                vol_map[sym] = vol
            if i % 25 == 0 or i == total:
                _progress(
                    f"Buoc 2/3: {i}/{total} ma "
                    f"({len(vol_map)} dat nguong {min_vol_billion}ty)"
                )
    except SystemExit:
        logger.warning(f"[HoseWatchlist] SystemExit trong vol loop sau {len(vol_map)} ma — dung lai, dung ket qua tam")
    except BaseException as e:
        logger.error(f"[HoseWatchlist] BaseException trong vol loop: {type(e).__name__}: {e}")

    # ── Bước 3: Sort và lấy top_n ────────────────────────────────────────────
    sorted_syms = sorted(vol_map.items(), key=lambda x: x[1], reverse=True)
    result = [s for s, _ in sorted_syms[:top_n]]

    elapsed = round(time.time() - t0, 1)
    _progress(
        f"Buoc 3/3: Xong! {len(vol_map)}/{total} ma dat nguong, "
        f"lay top {len(result)} | {elapsed}s"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL WORKER (chạy trong thread)
# ══════════════════════════════════════════════════════════════════════════════

def _scan_one_symbol(symbol: str, current_regime: int = 0) -> dict:
    """
    Xử lý đầy đủ một mã: check cache → find_similar → guardrails.
    Chạy trong ThreadPoolExecutor.

    current_regime: pre-loaded từ run_batch_scan (0 = disable regime filter)
    → tránh mỗi worker tự load VNINDEX riêng gây rate limit.
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
                # close tu vn_loader don vi NGHIN DONG (VD: 41.9 = 41,900d)
                # → nhan them 1000 de doi ra dong truoc khi tinh thanh tien
                avg_vol_vnd = float((vol_arr * close_arr).mean()) * 1000
                volume_avg_bill = round(avg_vol_vnd / 1e9, 2)
                # current_price nhan 1000 → don vi dong (dung cho ke hoach hanh dong)
                current_price   = float(df["close"].iloc[-1]) * 1000
                logger.info(f"[{symbol}] volume_avg_bill={volume_avg_bill:.2f}ty "
                            f"current_price={current_price:.1f}k")
        except Exception as _ve:
            logger.debug(f"[{symbol}] volume load fail: {_ve}")

        # ── 2. Kiểm tra /check gần nhất ──────────────────────────────────
        try:
            from db import get_conn
            conn = get_conn()
            cur  = conn.cursor()
            cur.execute(
                "SELECT created_at FROM signals "
                "WHERE symbol=%s ORDER BY created_at DESC LIMIT 1",
                (symbol,)
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
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

        # ── 3. Nếu /check quá cũ (> 48h) → bỏ qua trong batch scan ────────
        # Không tự trigger analyze_stock_full() trong batch vì:
        # - 50 mã × 30-60s/mã = vượt timeout
        # - Gây rate limit API khi chạy song song
        # → Chỉ log warning, để user tự chạy /check_all hoặc /check <MA>
        from guardrails import MAX_CHECK_AGE_HOURS
        if check_age_hours is None or check_age_hours > MAX_CHECK_AGE_HOURS:
            if check_age_hours is None:
                logger.info(f"[{symbol}] /check chua co — can chay /check {symbol} truoc")
            else:
                logger.info(f"[{symbol}] /check cu ({check_age_hours:.0f}h > {MAX_CHECK_AGE_HOURS}h) — nen refresh")

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
                conn = get_conn()
                cur  = conn.cursor()
                for col in ("state_vector", "state_vec", "vector_json"):
                    try:
                        cur.execute(
                            f"SELECT {col} FROM signals "
                            f"WHERE symbol=%s AND {col} IS NOT NULL "
                            f"ORDER BY created_at DESC LIMIT 1",
                            (symbol,)
                        )
                        row = cur.fetchone()
                        if row and row[0]:
                            sv = row[0]
                            state_vec = _json.loads(sv) if isinstance(sv, str) else sv
                            sv_source = f"db.{col}"
                            logger.info(f"[{symbol}] state_vec from {sv_source}")
                            break
                    except Exception:
                        continue
                cur.close()
                conn.close()
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
                symbol         = symbol,
                target_vector  = state_vec,
                years          = 5,
                exclude_days   = 90,
                min_results    = 3,
                current_regime = current_regime,   # pre-loaded, tránh re-fetch VNINDEX
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
    t_global  = time.time()
    raw_results: dict[str, dict] = {}
    total_syms  = len(symbols)
    done_count  = 0   # đếm số mã đã hoàn thành (dùng trong closure)

    def _progress(msg: str):
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass
        logger.info(f"[BatchScan] {msg}")

    _progress(f"Bat dau scan {total_syms} ma voi {MAX_WORKERS} workers...")

    # ── Pre-load market regime 1 lần — tránh mỗi worker tự load VNINDEX riêng ──
    current_regime = 0
    try:
        from market_regime import get_market_regime
        _rdata = get_market_regime()
        if _rdata.get("ok"):
            current_regime = int(_rdata.get("regime", 0))
            _progress(f"Market regime: R{current_regime} {_rdata.get('label','').split('—')[-1].strip()}")
        else:
            _progress("Market regime: load fail → regime filter disabled")
    except Exception as _re:
        logger.warning(f"[BatchScan] regime pre-load fail: {_re}")
        _progress("Market regime: error → regime filter disabled")

    # Chạy song song
    partial = False
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_scan_one_symbol, sym, current_regime): sym for sym in symbols}
        deadline = t_global + SCAN_TIMEOUT_SECS

        # BUG FIX: as_completed() raise TimeoutError ra ngoài vòng for khi
        # hết SCAN_TIMEOUT_SECS — crash toàn bộ run_batch_scan thay vì
        # bỏ qua futures chưa xong. Dùng try/except bọc iterator để catch.
        try:
            for future in as_completed(futures, timeout=SCAN_TIMEOUT_SECS):
                sym = futures[future]
                try:
                    res = future.result(timeout=max(1, deadline - time.time()))
                    raw_results[sym] = res
                    elapsed = res.get("elapsed", 0)
                    gate    = res.get("gate", "?")
                except FuturesTimeout:
                    raw_results[sym] = {"symbol": sym, "gate": "TIMEOUT",
                                        "reason": "Vuot qua thoi gian cho phep"}
                    partial = True
                    gate, elapsed = "TIMEOUT", 0
                except Exception as e:
                    raw_results[sym] = {"symbol": sym, "gate": "ERROR",
                                        "reason": str(e)[:100]}
                    gate, elapsed = "ERROR", 0

                done_count += 1
                _progress(f"  ({done_count}/{total_syms}) {sym}: {gate} ({elapsed:.0f}s)")

        except Exception as _outer_timeout:
            # as_completed() hết thời gian — đánh dấu các mã chưa có kết quả
            partial = True
            for sym in symbols:
                if sym not in raw_results:
                    raw_results[sym] = {"symbol": sym, "gate": "TIMEOUT",
                                        "reason": f"Scan timeout ({SCAN_TIMEOUT_SECS}s)"}
                    done_count += 1
                    _progress(f"  ({done_count}/{total_syms}) {sym}: TIMEOUT (global)")

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

    # Debug: log mã nào KHÔNG vào valid_results dù gate=PASS
    for sym, res in raw_results.items():
        if res.get("gate") == "PASS" and sym not in valid_results:
            analogs = res.get("analogs")
            n_ana   = len(analogs) if analogs else 0
            if analogs:
                from guardrails import compute_base_stats as _cbs
                st = _cbs(analogs)
                logger.warning(f"[BatchScan] {sym} gate=PASS analogs={n_ana} "
                               f"but valid=False: stats={st}")
            else:
                logger.warning(f"[BatchScan] {sym} gate=PASS but no analogs")
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
            logger.warning(f"[BatchScan] {sym} gate={res.get('gate')} NOT in valid_results")
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
            reason = gr.get("reason", "Gate reject")
            logger.warning(f"[BatchScan] {sym} guardrail REJECT: {reason}")
            _progress(f"  REJECT {sym}: {reason[:80]}")
            rejected.append({"symbol": sym, "reason": reason})
            continue

        if gr["risk_tier"] == "EXCLUDED":
            # Log lý do để debug
            pen = gr.get("penalties", [])
            pen_str = pen[0][:80] if pen else "score=0"
            s = gr.get("stats", {})
            logger.info(
                f"[BatchScan] {sym} EXCLUDED: exp={s.get('expectancy',0):+.1f}% "
                f"pf={s.get('profit_factor',0):.2f} wr={s.get('win_rate',0):.0%} "
                f"| {pen_str}"
            )
            _progress(f"  EXCLUDED {sym}: {pen_str[:60]}")
            gr["_exclude_reason"] = pen_str   # attach lý do vào gr để formatter dùng
            excluded.append(gr)
        else:
            ranked.append(gr)

    # Sort by score desc
    ranked.sort(key=lambda x: x["score"], reverse=True)
    excluded.sort(key=lambda x: x["score"], reverse=True)

    elapsed = round(time.time() - t_global, 1)
    _progress(f"Scan xong: {len(ranked)} mã vào rank, {len(excluded)} excluded, "
              f"{len(rejected)} reject | {elapsed}s")

    # ── DEBUG: log chi tiết để diagnose kết quả rỗng ─────────────────────────
    logger.info(
        f"[BatchScan] raw_results summary: total={len(raw_results)} "
        f"pass={sum(1 for r in raw_results.values() if r.get('gate')=='PASS')} "
        f"reject={sum(1 for r in raw_results.values() if r.get('gate')=='REJECT')} "
        f"timeout={sum(1 for r in raw_results.values() if r.get('gate')=='TIMEOUT')} "
        f"error={sum(1 for r in raw_results.values() if r.get('gate')=='ERROR')}"
    )
    logger.info(f"[BatchScan] valid_results={len(valid_results)} all_stats={len(all_stats)}")
    for sym, res in raw_results.items():
        if res.get("gate") == "PASS":
            analogs = res.get("analogs") or []
            fwd30_none = sum(1 for a in analogs if a.get("fwd_30") is None)
            logger.info(
                f"[BatchScan] PASS {sym}: analogs={len(analogs)} "
                f"fwd30_none={fwd30_none} "
                f"in_valid={sym in valid_results}"
            )
        elif res.get("gate") == "REJECT":
            logger.info(f"[BatchScan] REJECT {sym}: {res.get('reason','?')[:80]}")
    # ── END DEBUG ─────────────────────────────────────────────────────────────

    # Market regime check — tính ở đây 1 lần, không lặp lại trong formatter
    market_warn = None
    try:
        from vn_loader import load_vn_ohlcv
        from guardrails import check_market_regime
        df_vni = load_vn_ohlcv("VNINDEX", days=10, min_bars=5)
        if df_vni is not None and len(df_vni) >= 6:
            close_5d = df_vni["close"].values
            chg_5d   = (close_5d[-1] - close_5d[-6]) / close_5d[-6] * 100
            market_warn = check_market_regime(chg_5d)
    except Exception as _mre:
        logger.debug(f"market_regime check fail: {_mre}")

    return {
        "ranked":       ranked,
        "excluded":     excluded,
        "rejected":     rejected,
        "errors":       errors,
        "elapsed":      elapsed,
        "market_warn":  market_warn,
        "partial":  partial,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DUAL SCAN — Regime ON + Regime OFF song song
# ══════════════════════════════════════════════════════════════════════════════

# Ngưỡng tối thiểu raw_n để hiển thị mã trong bảng "No Filter"
DUAL_SCAN_MIN_RAW_N = 3   # trader muốn thấy mọi mã có >=3 mẫu raw


def run_dual_scan(
    symbols:    list[str],
    progress_cb = None,
) -> dict:
    """
    Chạy 2 scan song song:
      - Scan A: Regime filter ON  (current_regime từ market_regime)
      - Scan B: Regime filter OFF (current_regime=0)

    Hai scan chạy trong ThreadPoolExecutor — tổng thời gian ≈ max(A, B),
    không phải A+B.

    Returns:
        {
          "regime_on":  dict,   # kết quả scan A (format giống run_batch_scan)
          "regime_off": dict,   # kết quả scan B
          "current_regime": int,
          "regime_label": str,
          "elapsed": float,
        }
    """
    import threading
    t0 = time.time()

    # Pre-load regime 1 lần dùng chung
    current_regime = 0
    regime_label   = "Khong xac dinh"
    try:
        from market_regime import get_market_regime
        rdata = get_market_regime()
        if rdata.get("ok"):
            current_regime = int(rdata.get("regime", 0))
            regime_label   = rdata.get("label", "")
    except Exception as e:
        logger.warning(f"[DualScan] regime load fail: {e}")

    def _progress_a(msg: str):
        if progress_cb:
            try: progress_cb(f"[RegimeON]  {msg}")
            except Exception: pass
        logger.info(f"[DualScan-ON]  {msg}")

    def _progress_b(msg: str):
        if progress_cb:
            try: progress_cb(f"[RegimeOFF] {msg}")
            except Exception: pass
        logger.info(f"[DualScan-OFF] {msg}")

    result_a: dict = {}
    result_b: dict = {}
    err_a: list   = []
    err_b: list   = []

    def _run_a():
        try:
            # Scan A: dùng regime filter ON — truyền current_regime trực tiếp
            # để không pre-load lại VNINDEX
            result_a.update(_run_batch_scan_internal(
                symbols        = symbols,
                current_regime = current_regime,
                progress_cb    = _progress_a,
            ))
        except Exception as e:
            logger.error(f"[DualScan-ON] error: {e}")
            err_a.append(str(e))

    def _run_b():
        try:
            # Scan B: regime filter OFF (current_regime=0)
            result_b.update(_run_batch_scan_internal(
                symbols        = symbols,
                current_regime = 0,   # disable regime filter
                progress_cb    = _progress_b,
            ))
        except Exception as e:
            logger.error(f"[DualScan-OFF] error: {e}")
            err_b.append(str(e))

    # Chạy song song
    t_a = threading.Thread(target=_run_a, daemon=True)
    t_b = threading.Thread(target=_run_b, daemon=True)
    t_a.start()
    t_b.start()
    t_a.join(timeout=SCAN_TIMEOUT_SECS + 60)
    t_b.join(timeout=SCAN_TIMEOUT_SECS + 60)

    elapsed = round(time.time() - t0, 1)
    logger.info(f"[DualScan] Done: {elapsed}s | "
                f"ON={len(result_a.get('ranked',[]))} ranked | "
                f"OFF={len(result_b.get('ranked',[]))} ranked")

    return {
        "regime_on":      result_a if result_a else {"ranked": [], "excluded": [], "rejected": [], "errors": [], "partial": False},
        "regime_off":     result_b if result_b else {"ranked": [], "excluded": [], "rejected": [], "errors": [], "partial": False},
        "current_regime": current_regime,
        "regime_label":   regime_label,
        "elapsed":        elapsed,
        "market_warn":    result_a.get("market_warn") or result_b.get("market_warn"),
    }


def _run_batch_scan_internal(
    symbols:        list[str],
    current_regime: int,
    progress_cb     = None,
) -> dict:
    """
    Core scan logic tách ra từ run_batch_scan để dual scan gọi được.
    Không pre-load regime (đã được caller cung cấp).
    """
    t_global    = time.time()
    raw_results: dict[str, dict] = {}
    total_syms  = len(symbols)
    done_count  = 0

    def _progress(msg: str):
        if progress_cb:
            try: progress_cb(msg)
            except Exception: pass
        logger.debug(f"[BatchScanInternal] {msg}")

    partial = False
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures  = {pool.submit(_scan_one_symbol, sym, current_regime): sym
                    for sym in symbols}
        deadline = t_global + SCAN_TIMEOUT_SECS

        try:
            for future in as_completed(futures, timeout=SCAN_TIMEOUT_SECS):
                sym = futures[future]
                try:
                    res = future.result(timeout=max(1, deadline - time.time()))
                    raw_results[sym] = res
                    elapsed = res.get("elapsed", 0)
                    gate    = res.get("gate", "?")
                except FuturesTimeout:
                    raw_results[sym] = {"symbol": sym, "gate": "TIMEOUT",
                                        "reason": "Timeout"}
                    partial = True
                    gate, elapsed = "TIMEOUT", 0
                except Exception as e:
                    raw_results[sym] = {"symbol": sym, "gate": "ERROR",
                                        "reason": str(e)[:100]}
                    gate, elapsed = "ERROR", 0

                done_count += 1
                _progress(f"  ({done_count}/{total_syms}) {sym}: {gate} ({elapsed:.0f}s)")

        except Exception:
            partial = True
            for sym in symbols:
                if sym not in raw_results:
                    raw_results[sym] = {"symbol": sym, "gate": "TIMEOUT",
                                        "reason": "Global timeout"}

    # Stats + guardrails
    all_stats    = []
    valid_results = {}
    for sym, res in raw_results.items():
        if res.get("gate") not in ("REJECT", "TIMEOUT", "ERROR", "UNKNOWN") \
                and res.get("analogs"):
            from guardrails import compute_base_stats
            stats = compute_base_stats(res["analogs"])
            if stats.get("valid"):
                all_stats.append(stats)
                valid_results[sym] = res

    ranked   = []
    excluded = []
    rejected = []
    errors   = []

    for sym, res in raw_results.items():
        if res.get("gate") in ("TIMEOUT", "ERROR"):
            errors.append(f"{sym}: {res.get('reason', '?')}")
            continue
        if res.get("gate") == "REJECT":
            rejected.append({"symbol": sym, "reason": res.get("reason", "Gate reject")})
            continue
        if sym not in valid_results:
            rejected.append({"symbol": sym, "reason": "Khong co analogs hop le"})
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

        pen = gr.get("penalties", [])
        pen_str = pen[0][:80] if pen else "score=0"
        gr["_exclude_reason"] = pen_str

        if gr["risk_tier"] == "EXCLUDED":
            excluded.append(gr)
        else:
            ranked.append(gr)

    ranked.sort(key=lambda x: x["score"], reverse=True)
    excluded.sort(key=lambda x: x["score"], reverse=True)

    elapsed = round(time.time() - t_global, 1)

    # ── DEBUG: log chi tiết để diagnose kết quả rỗng ─────────────────────────
    logger.info(
        f"[BatchScanInternal] raw_results summary: total={len(raw_results)} "
        f"pass={sum(1 for r in raw_results.values() if r.get('gate')=='PASS')} "
        f"reject={sum(1 for r in raw_results.values() if r.get('gate')=='REJECT')} "
        f"timeout={sum(1 for r in raw_results.values() if r.get('gate')=='TIMEOUT')} "
        f"error={sum(1 for r in raw_results.values() if r.get('gate')=='ERROR')}"
    )
    logger.info(f"[BatchScanInternal] valid_results={len(valid_results)} all_stats={len(all_stats)}")
    for sym, res in raw_results.items():
        if res.get("gate") == "PASS":
            analogs = res.get("analogs") or []
            fwd30_none = sum(1 for a in analogs if a.get("fwd_30") is None)
            logger.info(
                f"[BatchScanInternal] PASS {sym}: analogs={len(analogs)} "
                f"fwd30_none={fwd30_none} "
                f"in_valid={sym in valid_results}"
            )
        elif res.get("gate") == "REJECT":
            logger.info(f"[BatchScanInternal] REJECT {sym}: {res.get('reason','?')[:80]}")
    # ── END DEBUG ─────────────────────────────────────────────────────────────

    # Market warn
    market_warn = None
    try:
        from vn_loader import load_vn_ohlcv
        from guardrails import check_market_regime
        df_vni = load_vn_ohlcv("VNINDEX", days=10, min_bars=5)
        if df_vni is not None and len(df_vni) >= 6:
            close_5d = df_vni["close"].values
            chg_5d   = (close_5d[-1] - close_5d[-6]) / close_5d[-6] * 100
            market_warn = check_market_regime(chg_5d)
    except Exception:
        pass

    return {
        "ranked":      ranked,
        "excluded":    excluded,
        "rejected":    rejected,
        "errors":      errors,
        "elapsed":     elapsed,
        "market_warn": market_warn,
        "partial":     partial,
    }


def format_dual_scan_report(dual_result: dict) -> list[str]:
    """
    Format kết quả dual scan thành list messages cho Telegram.

    Layout:
      MSG 1: Header + Bảng A (Regime ON) + Bảng B excerpt (Regime OFF)
      MSG 2+: Chi tiết nếu cần

    Bảng B chỉ hiển thị mã:
      - Không có trong ranked/excluded bảng A (tránh trùng lặp)
      - raw_n >= DUAL_SCAN_MIN_RAW_N (>= 3 mẫu)
      - Expectancy > 0 và WR >= 40%
    """
    from guardrails import SCORE_OPPORTUNITY_MIN

    regime_on    = dual_result.get("regime_on", {})
    regime_off   = dual_result.get("regime_off", {})
    cur_regime   = dual_result.get("current_regime", 0)
    regime_label = dual_result.get("regime_label", "")
    elapsed      = dual_result.get("elapsed", 0)
    market_warn  = dual_result.get("market_warn")

    scan_date    = datetime.now().strftime("%d/%m/%Y %H:%M")
    regime_names = {1: "R1 Bull Quiet", 2: "R2 Bull Volatile",
                    3: "R3 Bear Quiet",  4: "R4 Bear Volatile"}
    r_name       = regime_names.get(cur_regime, f"R{cur_regime}" if cur_regime else "?")

    ranked_on  = regime_on.get("ranked",   [])
    excl_on    = regime_on.get("excluded", [])
    rej_on     = regime_on.get("rejected", [])
    err_on     = regime_on.get("errors",   [])
    partial_on = regime_on.get("partial",  False)

    ranked_off = regime_off.get("ranked",   [])
    excl_off   = regime_off.get("excluded", [])

    total_scan = (len(ranked_on) + len(excl_on) +
                  len(rej_on) + len(err_on))

    # ── Build messages ────────────────────────────────────────────────────────
    lines = []

    # Header
    partial_tag = " [KET QUA CHUA DAY DU]" if partial_on else ""
    lines.append(f"SCAN WATCHLIST ({scan_date}){partial_tag}")
    lines.append(f"Tong: {total_scan} ma | {int(elapsed)}s | Regime: {r_name}")
    lines.append("=" * 38)

    if market_warn:
        lines.append(f"CANH BAO: {market_warn}")
        lines.append("")

    # ── BẢNG A: REGIME FILTER ON ──────────────────────────────────────────────
    lines.append(f"BANG A — REGIME FILTER ON ({r_name}):")
    lines.append("(Mau duoc can theo regime hien tai)")
    lines.append("")

    def _get_rr(r):
        """Risk/Reward ratio = Exp / |MAE30|. Trả về 0 nếu không tính được."""
        s   = r.get("stats", {})
        exp = s.get("expectancy", 0)
        mae = s.get("median_mdd_30d", s.get("median_mdd", 0))
        if mae >= 0 or exp <= 0:
            return 0.0
        return round(exp / abs(mae), 2)

    def _render(title, items, show_wn=True):
        if not items:
            return
        lines.append(title)
        hdr = f"  {'Ma':<6} {'WR':>5}  {'Exp':>6}  {'MAE30':>7}  {'RR':>5}"
        if show_wn:
            hdr += f"  {'wN':>4}"
        lines.append(hdr)
        lines.append("  " + "-" * (40 if show_wn else 35))
        for x in items:
            s    = x.get("stats", {})
            wr   = s.get("win_rate", 0)
            exp  = s.get("expectancy", 0)
            mae  = s.get("median_mdd_30d", s.get("median_mdd", 0))
            wn   = s.get("weighted_n", 0)
            rr   = _get_rr(x)
            row  = (f"  {x['symbol']:<6} "
                    f"{wr:>4.0%}  {exp:>+5.1f}%  {mae:>+6.1f}%  {rr:>4.1f}x")
            if show_wn:
                row += f"  {wn:>4.1f}"
            lines.append(row)
        lines.append("")

    # Group ranked ON theo RR thay vì score
    # UU TIEN: RR >= 2.0 (Exp ít nhất gấp đôi MAE)
    # TIEM NANG: 1.0 <= RR < 2.0 (Exp > MAE nhưng chưa vượt trội)
    # THAM KHAO: RR < 1.0 (MAE >= Exp — không có edge rõ ràng)
    grp_priority  = sorted([r for r in ranked_on if _get_rr(r) >= 2.0],
                            key=_get_rr, reverse=True)
    grp_potential = sorted([r for r in ranked_on if 1.0 <= _get_rr(r) < 2.0],
                            key=_get_rr, reverse=True)
    grp_ref       = sorted([r for r in ranked_on if _get_rr(r) < 1.0],
                            key=_get_rr, reverse=True)

    if not ranked_on:
        lines.append("  Khong co ma nao du dieu kien voi regime filter ON.")
        lines.append("")
    else:
        _render("UU TIEN (RR >= 2.0 | Exp > 2x MAE):", grp_priority)
        _render("TIEM NANG (1.0 <= RR < 2.0 | Exp > MAE):", grp_potential)
        if grp_ref:
            ref_syms = ", ".join(
                f"{r['symbol']}({_get_rr(r):.1f}x)"
                for r in grp_ref[:12]
            )
            lines.append(f"THAM KHAO (RR < 1.0 — MAE >= Exp): {ref_syms}")
            lines.append("")

    # Excluded ON — compact
    if excl_on:
        lines.append(f"LOAI (Exp am / PF<1 / WR<40% / wN<5): {len(excl_on)} ma")
        exc_syms = ", ".join(
            f"{r['symbol']}({r.get('stats',{}).get('weighted_n',0):.1f}wN)"
            for r in excl_on[:15]
        )
        lines.append(f"  {exc_syms}")
        lines.append("")

    lines.append("─" * 38)

    # ── BẢNG B: REGIME FILTER OFF — độc lập hoàn toàn với Bảng A ────────────
    lines.append("BANG B — REGIME FILTER OFF (khong loc regime):")
    lines.append("(Ket qua doc lap — trader tu quyet dinh)")
    lines.append("")

    ranked_off_all = ranked_off

    if not ranked_off_all:
        lines.append("  Khong co ma nao du dieu kien.")
        lines.append("")
    else:
        def _get_rr_raw(r):
            s   = r.get("stats", {})
            exp = s.get("expectancy", 0)
            mae = s.get("median_mdd_30d", s.get("median_mdd", 0))
            if mae >= 0 or exp <= 0:
                return 0.0
            return round(exp / abs(mae), 2)

        def _render_b(title, items):
            if not items:
                return
            lines.append(title)
            lines.append(f"  {'Ma':<6} {'WR':>5}  {'Exp':>6}  {'MAE30':>7}  {'RR':>5}  {'rawN':>5}")
            lines.append("  " + "-" * 44)
            for x in items:
                s     = x.get("stats", {})
                wr    = s.get("win_rate_raw", s.get("win_rate", 0))
                exp   = s.get("expectancy", 0)
                mae   = s.get("median_mdd_30d", s.get("median_mdd", 0))
                raw_n = s.get("n", 0)
                rr    = _get_rr_raw(x)
                lines.append(
                    f"  {x['symbol']:<6} "
                    f"{wr:>4.0%}  {exp:>+5.1f}%  {mae:>+6.1f}%  {rr:>4.1f}x  {raw_n:>5}"
                )
            lines.append("")

        grp_b_priority  = sorted([r for r in ranked_off_all if _get_rr_raw(r) >= 2.0],
                                   key=_get_rr_raw, reverse=True)
        grp_b_potential = sorted([r for r in ranked_off_all if 1.0 <= _get_rr_raw(r) < 2.0],
                                   key=_get_rr_raw, reverse=True)
        grp_b_ref       = [r for r in ranked_off_all if _get_rr_raw(r) < 1.0]

        _render_b("UU TIEN (RR >= 2.0):", grp_b_priority)
        _render_b("TIEM NANG (1.0 <= RR < 2.0):", grp_b_potential)
        if grp_b_ref:
            lines.append(f"THAM KHAO (RR < 1.0): {len(grp_b_ref)} ma — dung /analog <MA> --raw de xem")
            lines.append("")

    # Mã không đủ điều kiện (thanh khoản, cache...)
    non_timeout_rej = [r for r in rej_on
                       if "timeout" not in r.get("reason", "").lower()]
    if non_timeout_rej:
        lines.append(f"KHONG DU DIEU KIEN ({len(non_timeout_rej)} ma):")
        for r in non_timeout_rej[:8]:
            lines.append(f"  {r['symbol']:<6} {r.get('reason','?')[:55]}")
        lines.append("")

    lines.append("* Score la chi so tham khao, khong phai khuyen nghi mua/ban.")
    lines.append("* Bang B hien thi raw data — khong co regime weighting.")

    full = "\n".join(lines)

    # Split messages
    MAX_LEN = 4000
    if len(full) <= MAX_LEN:
        return [full]

    # Split tại separator
    msgs  = []
    split = full.find("BANG B —")
    if 0 < split < MAX_LEN:
        msgs.append(full[:split].strip())
        msgs.append(full[split:].strip()[:MAX_LEN])
    else:
        # Generic split
        buf = ""
        for line in full.split("\n"):
            if len(buf) + len(line) + 1 > MAX_LEN:
                msgs.append(buf.strip())
                buf = line + "\n"
            else:
                buf += line + "\n"
        if buf.strip():
            msgs.append(buf.strip())

    return msgs if msgs else [full[:MAX_LEN]]

def _format_overview_table(
    ranked:   list[dict],
    excluded: list[dict],
    rejected: list[dict],
    errors:   list[str],
    partial:  bool,
) -> str:
    """
    Tạo bảng tổng quan TẤT CẢ mã, phân nhóm theo score.
    Dùng monospace-friendly format cho Telegram plain text.
    """
    # Gom tất cả mã có score vào 1 list để sort
    all_scored = []
    for r in ranked:
        s = r.get("stats", {})
        all_scored.append({
            "symbol":   r["symbol"],
            "score":    r["score"],
            "wr":       s.get("win_rate", 0),
            "mae_med":  s.get("median_mdd", 0),   # median_mdd = MAE median trong guardrails
            "tier":     r["risk_tier"],
            "group":    "ranked",
        })
    for r in excluded:
        s = r.get("stats", {})
        all_scored.append({
            "symbol":   r["symbol"],
            "score":    r["score"],
            "wr":       s.get("win_rate", 0),
            "mae_med":  s.get("median_mdd", 0),
            "tier":     "EXCLUDED",
            "group":    "excluded",
        })

    # Sort theo score giảm dần
    all_scored.sort(key=lambda x: x["score"], reverse=True)

    # Phân nhóm
    grp_priority  = [x for x in all_scored if x["score"] >= 5.0 and x["group"] == "ranked"]
    grp_potential = [x for x in all_scored if 4.0 <= x["score"] < 5.0 and x["group"] == "ranked"]
    grp_ref       = [x for x in all_scored if x["score"] < 4.0 and x["group"] == "ranked"]
    grp_excluded  = [x for x in all_scored if x["group"] == "excluded"]

    lines = []

    def _render_group(title: str, items: list, show_excluded: bool = False):
        if not items:
            return
        lines.append(title)
        lines.append(f"  {'Ma':<6} {'Score':>5}  {'WR':>5}  {'MAE 90D':>8}")
        lines.append("  " + "-" * 32)
        for x in items:
            exc_note = " [RR cao]" if show_excluded else ""
            lines.append(
                f"  {x['symbol']:<6} {x['score']:>5.1f}  "
                f"{x['wr']:>4.0%}  {x['mae_med']:>+7.1f}%{exc_note}"
            )
        lines.append("")

    _render_group("UU TIEN (Score >= 5.0):", grp_priority)
    _render_group("TIEM NANG (4.0 - 4.9):", grp_potential)
    _render_group("THAM KHAO (< 4.0):", grp_ref)
    _render_group("RUI RO CAO (Excluded):", grp_excluded, show_excluded=True)

    # Debug: lý do từng mã bị excluded — giúp tune threshold
    if grp_excluded:
        lines.append("  Chi tiet excluded:")
        for r in excluded[:15]:
            s      = r.get("stats", {})
            exp    = s.get("expectancy", 0)
            pf     = s.get("profit_factor", 0)
            wr     = s.get("win_rate", 0)
            reason = r.get("_exclude_reason", "?")[:50]
            lines.append(
                f"  {r['symbol']:<6} Exp:{exp:+.1f}% "
                f"PF:{pf:.1f} WR:{wr:.0%} | {reason}"
            )
        lines.append("")

    # Mã không có kết quả
    non_timeout_rej = [r for r in rejected if "timeout" not in r.get("reason","").lower()]
    timeout_rej     = [r for r in rejected if "timeout" in r.get("reason","").lower()]
    timeout_errs    = [e.split(":")[0] for e in errors if "timeout" in e.lower()]

    if non_timeout_rej:
        lines.append(f"KHONG DU DIEU KIEN ({len(non_timeout_rej)} ma):")
        for r in non_timeout_rej[:10]:
            reason = r.get("reason","?")[:55]
            lines.append(f"  {r['symbol']:<6} {reason}")
        if len(non_timeout_rej) > 10:
            lines.append(f"  ... va {len(non_timeout_rej)-10} ma khac")
        lines.append("")

    all_timeouts = [r["symbol"] for r in timeout_rej] + timeout_errs
    if all_timeouts:
        lines.append(f"TIMEOUT ({len(all_timeouts)} ma): {', '.join(all_timeouts)}")
        lines.append("")

    return "\n".join(lines)



def format_scan_report(scan_result: dict) -> list[str]:
    """
    Format ket qua scan thanh list messages (tu chia neu > 4000 ky tu).
    Hien thi TOAN BO watchlist phan nhom theo score — khong an khi khong co ma >= 5.0.
    """
    from guardrails import (
        format_full_report, check_no_opportunity,
        SCORE_OPPORTUNITY_MIN,
    )

    ranked      = scan_result["ranked"]
    excluded    = scan_result["excluded"]
    rejected    = scan_result["rejected"]
    errors      = scan_result["errors"]
    partial     = scan_result["partial"]
    elapsed     = scan_result["elapsed"]
    # Market warn đã được tính trong run_batch_scan — không fetch lại ở đây
    market_warn = scan_result.get("market_warn")

    scan_date  = datetime.now().strftime("%d/%m/%Y %H:%M")
    no_opp     = check_no_opportunity(ranked)
    total_scan = len(ranked) + len(excluded) + len(rejected) + len(errors)

    # ── Header ────────────────────────────────────────────────────────
    hdr = []
    hdr.append("SCAN WATCHLIST (" + scan_date + ")")
    suffix = " | [KET QUA CHUA DAY DU]" if partial else ""
    hdr.append("Tong: " + str(total_scan) + " ma | " + str(int(elapsed)) + "s" + suffix)
    hdr.append("=" * 38)
    if market_warn:
        hdr.append("CANH BAO: " + market_warn)
        hdr.append("")
    if no_opp:
        hdr.append("Khong co ma nao dat nguong Score > " + str(SCORE_OPPORTUNITY_MIN) + "/10 hom nay.")
        hdr.append("Danh sach day du ben duoi de tham khao:")
        hdr.append("")
    header = "\n".join(hdr)

    # ── Bang tong quan ────────────────────────────────────────────────
    overview = _format_overview_table(ranked, excluded, rejected, errors, partial)

    # ── Detail top 5 uu tien (chi khi co ma >= 5.0) ──────────────────
    detail = ""
    top5 = [r for r in ranked if r["score"] >= 5.0][:5]
    if top5:
        full_rep = format_full_report(
            ranked         = ranked,
            excluded       = [],
            market_warning = None,
            scan_date      = scan_date,
        )
        detail = "\n" + full_rep

    # ── Footer ────────────────────────────────────────────────────────
    footer = "\n* Score la chi so tham khao, khong phai khuyen nghi mua/ban."

    full = header + "\n" + overview + detail + footer

    # ── Tu chia messages neu > 4000 ky tu ────────────────────────────
    MAX_LEN = 4000
    if len(full) <= MAX_LEN:
        return [full]

    msgs  = []
    part1 = header + "\n" + overview + footer
    if len(part1) <= MAX_LEN:
        msgs.append(part1)
    else:
        buf = header + "\n"
        for line in (overview + footer).split("\n"):
            if len(buf) + len(line) + 1 > MAX_LEN:
                msgs.append(buf.strip())
                buf = line + "\n"
            else:
                buf += line + "\n"
        if buf.strip():
            msgs.append(buf.strip())

    if detail.strip():
        parts = detail.split("\n\n")
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

    return msgs if msgs else [full[:MAX_LEN]]


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def scan_watchlist_cmd(update, context):
    """
    /scan_watchlist        — Dual scan: Regime ON + Regime OFF song song
    /scan_watchlist --raw  — Chỉ chạy scan đơn (regime OFF, behavior cũ)
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
            f"Vui long cho {wait}s truoc khi scan lai.\n"
            f"(Cooldown {COOLDOWN_SECS//60} phut)"
        )
        return

    _last_scan_time = time.time()
    chat_id  = update.effective_chat.id
    args     = context.args or []
    raw_mode = "--raw" in args   # chỉ chạy scan đơn nếu có --raw

    symbols = load_watchlist()

    if raw_mode:
        msg = await update.message.reply_text(
            f"Dang scan {len(symbols)} ma (Regime filter OFF)...\n"
            f"Toi da {SCAN_TIMEOUT_SECS // 60} phut."
        )
    else:
        msg = await update.message.reply_text(
            f"Dang dual scan {len(symbols)} ma...\n"
            f"Bang A: Regime filter ON | Bang B: Regime filter OFF\n"
            f"(2 scan chay song song — tong thoi gian ~bang 1 scan don)"
        )

    # Progress callback
    progress_lines: list[str] = [
        f"Scan {len(symbols)} ma — dang chay...\n"
    ]

    async def _progress_async(line: str):
        progress_lines.append(line)
        try:
            preview = "\n".join(progress_lines[-6:])
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=preview[:4000],
            )
        except Exception:
            pass

    loop = asyncio.get_event_loop()

    def _progress_sync(line: str):
        asyncio.run_coroutine_threadsafe(_progress_async(line), loop)

    try:
        if raw_mode:
            # Scan đơn regime OFF (behavior cũ)
            scan_result = await asyncio.to_thread(
                run_batch_scan, symbols, _progress_sync
            )
            messages = format_scan_report(scan_result)
        else:
            # Dual scan
            dual_result = await asyncio.to_thread(
                run_dual_scan, symbols, _progress_sync
            )
            # Cache regime_on result cho morning_briefing
            try:
                import batch_scanner as _bs_self
                _bs_self._last_scan_result = dual_result["regime_on"]
            except Exception:
                pass
            messages = format_dual_scan_report(dual_result)

    except Exception as e:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=f"Loi scan: {str(e)[:300]}",
        )
        return

    # Gửi messages
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=messages[0][:4000],
        )
    except Exception:
        await context.bot.send_message(chat_id=chat_id, text=messages[0][:4000])

    for extra_msg in messages[1:]:
        try:
            await context.bot.send_message(chat_id=chat_id, text=extra_msg[:4000])
        except Exception as e:
            logger.warning(f"scan_watchlist_cmd: send extra fail: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND — /scan_hose
# ══════════════════════════════════════════════════════════════════════════════

_last_hose_scan_time: float = 0.0
HOSE_COOLDOWN_SECS = 1800   # 30 phút giữa 2 lần scan thủ công


async def scan_hose_cmd(update, context):
    """
    /scan_hose            — Scan top 200 mã HOSE theo thanh khoản (>= 3 tỷ/ngày)
    /scan_hose --top 150  — Chỉ lấy top 150 mã
    /scan_hose --vol 5    — Ngưỡng thanh khoản tối thiểu 5 tỷ/ngày
    """
    global _last_hose_scan_time

    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    # Cooldown
    since = time.time() - _last_hose_scan_time
    if since < HOSE_COOLDOWN_SECS:
        wait = int(HOSE_COOLDOWN_SECS - since)
        await update.message.reply_text(
            f"Vui long cho {wait}s ({wait//60}ph) truoc khi scan HOSE lai."
        )
        return

    # Parse args: --top N và --vol X
    args     = context.args or []
    top_n    = HOSE_TOP_N
    min_vol  = HOSE_MIN_VOL
    try:
        if "--top" in args:
            top_n = int(args[args.index("--top") + 1])
        if "--vol" in args:
            min_vol = float(args[args.index("--vol") + 1])
    except (ValueError, IndexError):
        await update.message.reply_text(
            "Cu phap: /scan_hose [--top N] [--vol X]\n"
            "Vi du: /scan_hose --top 150 --vol 3.0"
        )
        return

    _last_hose_scan_time = time.time()
    chat_id = update.effective_chat.id

    msg = await update.message.reply_text(
        f"🔍 HOSE Scan (top {top_n}, vol >= {min_vol}ty/ngay, 20 phien)\n"
        f"Buoc 1/3: Dang lay danh sach ma..."
    )

    # Progress callback — update message trực tiếp
    _hdr = f"🔍 HOSE Scan (top {top_n}, vol>={min_vol}ty)\n"

    async def _progress_async(line: str):
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=(_hdr + line)[:4000],
            )
        except Exception:
            pass

    loop = asyncio.get_event_loop()

    def _progress_sync(line: str):
        asyncio.run_coroutine_threadsafe(_progress_async(line), loop)

    try:
        # Bước 1+2: build watchlist (với progress callback)
        symbols = await asyncio.to_thread(
            build_hose_watchlist, top_n, min_vol, 20, _progress_sync
        )
    except (SystemExit, BaseException) as e:
        import traceback
        logger.error(f"scan_hose_cmd build_watchlist ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=f"❌ Loi lay danh sach HOSE: {type(e).__name__}\n(Bot van hoat dong binh thuong)"
            )
        except Exception:
            pass
        return

    if not symbols:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text="❌ Khong lay duoc danh sach HOSE. Thu lai sau."
        )
        return

    await _progress_async(
        f"✅ Got {len(symbols)} ma HOSE\n"
        f"Buoc 3/3: Dang dual scan...\n"
        f"(Bang A: Regime ON | Bang B: Regime OFF)"
    )

    try:
        # Bước 3: dual scan với progress hiện (X/N)
        dual_result = await asyncio.to_thread(
            run_dual_scan, symbols, _progress_sync
        )
        # Format và gửi
        messages_out = format_dual_scan_report(dual_result)
        header = f"🏢 HOSE SCAN (top {top_n}, vol>={min_vol}ty)\n"
        messages_out[0] = header + messages_out[0]
    except (SystemExit, BaseException) as e:
        import traceback
        logger.error(f"scan_hose_cmd dual_scan ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=f"❌ Loi scan HOSE: {type(e).__name__}\n(Bot van hoat dong binh thuong)"
            )
        except Exception:
            pass
        return

    # Gửi kết quả
    try:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=msg.message_id,
            text=messages_out[0][:4000],
        )
    except Exception:
        await context.bot.send_message(chat_id=chat_id, text=messages_out[0][:4000])

    for extra_msg in messages_out[1:]:
        try:
            await context.bot.send_message(chat_id=chat_id, text=extra_msg[:4000])
        except Exception as e:
            logger.warning(f"scan_hose_cmd: send extra fail: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CRON JOB — 4:00 AM — FULL HOSE SCAN
# ══════════════════════════════════════════════════════════════════════════════

HOSE_CRON_HOUR   = 4
HOSE_CRON_MINUTE = 0
HOSE_TOP_N       = int(os.environ.get("HOSE_TOP_N",   "200"))
HOSE_MIN_VOL     = float(os.environ.get("HOSE_MIN_VOL", "3.0"))


async def _start_hose_cron(bot, chat_ids: list[int]):
    """
    Cron 4:00 AM — scan toàn bộ HOSE top N mã theo thanh khoản.

    Flow:
      1. build_hose_watchlist() → top 200 mã HOSE theo vol
      2. run_dual_scan() với danh sách đó
      3. Gửi kết quả qua Telegram

    Chạy lúc 4h sáng để tránh rate limit (ít traffic),
    cache vector sẽ được build sẵn trước khi thị trường mở.

    Env vars:
      HOSE_TOP_N   (default 200): số mã top lấy
      HOSE_MIN_VOL (default 5.0): thanh khoản tối thiểu tỷ/ngày
    """
    import datetime as _dt

    while True:
        now    = _dt.datetime.now()
        target = now.replace(
            hour=HOSE_CRON_HOUR, minute=HOSE_CRON_MINUTE, second=0, microsecond=0
        )
        if now >= target:
            target += _dt.timedelta(days=1)

        wait_secs = (target - now).total_seconds()
        logger.info(
            f"[HoseCron] Next run in {wait_secs/3600:.1f}h "
            f"({target.strftime('%d/%m %H:%M')}) | "
            f"top_n={HOSE_TOP_N} min_vol={HOSE_MIN_VOL}ty"
        )
        await asyncio.sleep(wait_secs)

        logger.info("[HoseCron] Bat dau build HOSE watchlist...")

        try:
            # Bước 1: build danh sách mã
            symbols = await asyncio.to_thread(
                build_hose_watchlist,
                HOSE_TOP_N,
                HOSE_MIN_VOL,
                20,    # đúng 20 phiên giao dịch
                None,  # cron không cần progress callback
            )

            if not symbols:
                err = "❌ [HoseCron] Khong lay duoc danh sach HOSE — skip scan"
                logger.error(err)
                for cid in chat_ids:
                    try:
                        await bot.send_message(chat_id=cid, text=err)
                    except Exception:
                        pass
                continue

            logger.info(f"[HoseCron] Got {len(symbols)} symbols, starting dual scan...")

            # Notify bắt đầu
            for cid in chat_ids:
                try:
                    await bot.send_message(
                        chat_id=cid,
                        text=f"🌙 [4AM] Bat dau scan {len(symbols)} ma HOSE "
                             f"(top {HOSE_TOP_N} theo thanh khoan)..."
                    )
                except Exception:
                    pass

            # Bước 2: dual scan
            dual_result = await asyncio.to_thread(run_dual_scan, symbols)

            # Cache cho morning_briefing
            try:
                import batch_scanner as _bs_self
                _bs_self._last_scan_result = dual_result["regime_on"]
            except Exception:
                pass

            # Bước 3: gửi kết quả
            messages = format_dual_scan_report(dual_result)
            header   = f"🌙 HOSE FULL SCAN ({_dt.datetime.now().strftime('%d/%m %H:%M')})\n"
            messages[0] = header + messages[0]

            for cid in chat_ids:
                for m in messages:
                    try:
                        await bot.send_message(chat_id=cid, text=m[:4000])
                        await asyncio.sleep(0.3)
                    except Exception as _se:
                        logger.warning(f"[HoseCron] send to {cid} fail: {_se}")

        except Exception as e:
            import traceback
            logger.error(f"[HoseCron] ERROR: {e}\n{traceback.format_exc()}")
            err_msg = (
                f"❌ HOSE scan lỗi ({_dt.datetime.now().strftime('%H:%M')}): "
                f"{str(e)[:200]}"
            )
            for cid in chat_ids:
                try:
                    await bot.send_message(chat_id=cid, text=err_msg)
                except Exception:
                    pass

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

        logger.info("[ScanCron] Bat dau chay auto dual scan...")
        try:
            symbols     = load_watchlist()
            dual_result = await asyncio.to_thread(run_dual_scan, symbols)

            # Cache regime_on cho morning_briefing
            try:
                import batch_scanner as _bs_self
                _bs_self._last_scan_result = dual_result["regime_on"]
            except Exception:
                pass

            messages = format_dual_scan_report(dual_result)

            for cid in chat_ids:
                for m in messages:
                    try:
                        await bot.send_message(chat_id=cid, text=m[:4000])
                        await asyncio.sleep(0.3)
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
