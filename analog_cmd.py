"""
analog_cmd.py — /analog command handler cho VN Signal Bot.

Cách dùng:
    /analog <MA>          → Phân tích tương đồng lịch sử với regime filter
    /analog <MA> --raw    → Tắt regime filter (backward compat)
    /analog <MA> --debug  → Hiện thêm meta info (weighted_n, regime cache size...)

Import vào bot.py:
    from analog_cmd import analog_cmd

Đăng ký trong main():
    app.add_handler(CommandHandler("analog", analog_cmd))

Pipeline:
    1. find_similar()     → tìm ngày lịch sử giống hiện tại (cosine sim, regime filter ON)
    2. compute_base_stats() → WR, Expectancy, MAE/MFE (weighted theo regime)
    3. guardrails         → hard gate Expectancy/WR/PF
    4. format output      → giống /backtest_rule nhưng rõ ràng hơn về sample source
    5. wave + stock regime → gắn thêm context (best-effort, không block)
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

ANALOG_COOLDOWN = 60   # giây giữa 2 lần gọi per user
_last_analog: dict[str, float] = {}


def _plain(text: str) -> str:
    import re
    text = re.sub(r"[*`]", "", text)
    return text.replace("\\_", "_")


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _format_analog_result(
    symbol:       str,
    analogs:      list,
    stats:        dict,
    wave_result:  dict | None,
    sr_result:    dict | None,
    debug:        bool = False,
) -> str:
    """
    Format output /analog — compact, actionable, giống /backtest nhưng
    source là historical analog (cosine similarity) thay vì rule-based.
    """
    import numpy as np

    n          = stats.get("n", 0)
    wr         = stats.get("win_rate", 0)
    wr_raw     = stats.get("win_rate_raw", wr)
    exp        = stats.get("expectancy", 0)
    pf         = stats.get("profit_factor", 0)
    pf_str     = "99.0" if pf >= 99 else f"{pf:.2f}"
    med_ret    = stats.get("median_ret", 0)
    med_mdd    = stats.get("median_mdd", 0)
    mae_avg    = stats.get("mae_avg", 0)
    mfe_avg    = stats.get("mfe_avg", 0)
    hold_med   = stats.get("hold_median", stats.get("hold_avg", 0))
    capture    = stats.get("mfe_capture_rate")
    p25        = stats.get("p25_ret")
    p75        = stats.get("p75_ret")
    rvr        = stats.get("return_vol_ratio", 0)
    med60      = stats.get("median_60")
    med90      = stats.get("median_90")
    weighted_n = stats.get("weighted_n", float(n))
    regime_on  = stats.get("regime_filter_active", False)
    match_pct  = stats.get("regime_match_pct")
    close_px   = stats.get("close", 0)

    # Meta từ _meta của analog đầu tiên
    meta = analogs[0].get("_meta", {}) if analogs else {}
    threshold  = meta.get("threshold_used", 0)
    sim_avg    = meta.get("avg_similarity", 0)
    total_raw  = meta.get("total_matches", 0)
    ind_n      = meta.get("independent_n", n)
    yr_used    = meta.get("years_used", 5)
    yr_exp     = meta.get("years_expanded", False)
    cur_regime = meta.get("current_regime", 0)
    regime_names = {1: "R1 Bull Quiet", 2: "R2 Bull Volatile",
                    3: "R3 Bear Quiet",  4: "R4 Bear Volatile"}

    # ── Header ────────────────────────────────────────────────────────────────
    sep = "═" * 32
    yr_tag  = f"{yr_used}Y" + (" [auto-expanded]" if yr_exp else "")
    sim_tag = f"Do TD: {threshold:.0%}" if threshold else ""
    n_tag   = f"{ind_n} mau" if ind_n == n else f"{ind_n}/{total_raw} mau"

    lines = [
        f"/analog {symbol}  [{yr_tag} | {n_tag} | {sim_tag}]",
        sep,
    ]

    # ── Regime filter info ────────────────────────────────────────────────────
    regime_fallback = meta.get("regime_fallback", False)

    if regime_fallback:
        lines.append(
            "Regime filter: FALLBACK — it mau cung regime, dung 8Y khong filter"
        )
        lines.append("Ket qua can than hon — mau tu nhieu regime khac nhau")
    elif regime_on:
        r_name = regime_names.get(cur_regime, f"R{cur_regime}")
        match_str = f" | Khop: {match_pct:.0%}" if match_pct is not None else ""
        lines.append(
            f"Regime: {r_name} | Weighted N: {weighted_n:.1f}{match_str}"
        )
        if abs(wr_raw - wr) > 0.03:
            lines.append(
                f"WR co trong so: {wr:.0%}  (raw: {wr_raw:.0%})"
            )
    else:
        lines.append("Regime filter: OFF")

    lines.append(sep)

    # ── Core stats ────────────────────────────────────────────────────────────
    lines.append(f"WR {wr:.0%}  |  PF {pf_str}  |  Expectancy {exp:+.1f}%")

    # Return timeline
    ret_line = f"LN: 30D {med_ret:+.1f}%"
    if med60 is not None: ret_line += f" | 60D {med60:+.1f}%"
    if med90 is not None: ret_line += f" | 90D {med90:+.1f}%"
    lines.append(ret_line)

    # MAE/MFE/Hold/Capture
    mfe_line = f"MAE {mae_avg:+.1f}%  |  MFE {mfe_avg:+.1f}%"
    if hold_med:
        mfe_line += f"  |  Hold ~{hold_med:.0f}D"
    if capture is not None:
        cap_note = " Exit som" if capture < 40 else " Hieu qua" if capture > 80 else ""
        mfe_line += f"  |  Capture {capture:.0f}%{cap_note}"
    lines.append(mfe_line)

    # P25/P75
    if p25 is not None and p75 is not None:
        lines.append(f"P25: {p25:+.1f}%  |  P75: {p75:+.1f}%")

    # ── Ke hoach hanh dong ────────────────────────────────────────────────────
    if close_px and mae_avg and mfe_avg:
        lines.append("─" * 32)
        sl_pct  = mae_avg - 2.0   # conservative buffer
        tp1_pct = med_ret
        tp2_pct = mfe_avg * 0.7   # 70% MFE làm TP2

        sl_px   = close_px * (1 + sl_pct / 100)
        tp1_px  = close_px * (1 + tp1_pct / 100)
        tp2_px  = close_px * (1 + tp2_pct / 100)

        rr1 = abs(tp1_pct / sl_pct) if sl_pct < 0 else 0
        rr2 = abs(tp2_pct / sl_pct) if sl_pct < 0 else 0

        lines.append(f"Entry: {close_px:,.0f}  |  SL: {sl_px:,.0f} ({sl_pct:.1f}%)")
        tp1_line = f"TP1: {tp1_px:,.0f} ({tp1_pct:+.1f}%)"
        if rr1: tp1_line += f"  R:R 1:{rr1:.1f}"
        lines.append(tp1_line)
        if tp2_pct > tp1_pct:
            tp2_line = f"TP2: {tp2_px:,.0f} ({tp2_pct:+.1f}%)"
            if rr2: tp2_line += f"  R:R 1:{rr2:.1f}"
            lines.append(tp2_line)

    # ── Wave ──────────────────────────────────────────────────────────────────
    if wave_result and wave_result.get("ok"):
        lines.append("─" * 32)
        wd    = wave_result.get("verdict", "KHONG RO")
        ws_up = wave_result.get("score_up_adj", wave_result.get("score_up", 0))
        ws_dn = wave_result.get("score_down_adj", wave_result.get("score_down", 0))
        wconf = wave_result.get("confidence", 0)
        wn    = min(wave_result.get("n_up", 0), wave_result.get("n_down", 0))

        def _stars(n, c):
            if n >= 20 and c >= 0.15: return "★★★"
            if n >= 15 and c >= 0.10: return "★★☆"
            if n >= 10 and c >= 0.08: return "★★☆"
            return "★☆☆"

        wrel = _stars(wn, wconf)
        w_em = "🟢" if wd == "SONG TANG" else "🔴" if wd == "SONG GIAM" else "🟡"
        lines.append(f"Wave: {w_em} {wd} {wrel} ({ws_dn:.0%} giam vs {ws_up:.0%} tang)")

        # Subtype
        if wd == "SONG TANG":
            st  = wave_result.get("subtype_stats_up", {})
            cmp = wave_result.get("subtype_compare_up", {})
            if st and st.get("total_known", 0) >= 6:
                pct     = st.get("pct", {})
                closest = cmp.get("closest", "")
                bot_pct = pct.get("BOTTOM_REAL", 0)
                ral_pct = pct.get("RELIEF_RALLY", 0)
                cl_lbl  = "Day that (tang ben vung)" if closest == "BOTTOM_REAL" else "Relief Rally (tang ngan)"
                cl_em   = "🟢" if closest == "BOTTOM_REAL" else "🟡"
                lines.append(f"   Loai: Day that {bot_pct:.0f}% | Relief Rally {ral_pct:.0f}% | {cl_em} Gan nhat: {cl_lbl}")
        elif wd == "SONG GIAM":
            st  = wave_result.get("subtype_stats_down", {})
            cmp = wave_result.get("subtype_compare_down", {})
            if st and st.get("total_known", 0) >= 6:
                pct      = st.get("pct", {})
                closest  = cmp.get("closest", "")
                peak_pct = pct.get("PEAK_REAL", 0)
                corr_pct = pct.get("CORRECTION", 0)
                cl_lbl   = "Peak that (giam sau)" if closest == "PEAK_REAL" else "Correction (giam roi len)"
                cl_em    = "🔴" if closest == "PEAK_REAL" else "🟡"
                lines.append(f"   Loai: Peak that {peak_pct:.0f}% | Correction {corr_pct:.0f}% | {cl_em} Gan nhat: {cl_lbl}")

    # ── Stock Regime ──────────────────────────────────────────────────────────
    if sr_result and sr_result.get("ok"):
        try:
            from stock_regime import format_stock_regime_for_backtest
            sr_block = format_stock_regime_for_backtest(sr_result)
            if sr_block:
                lines.append("─" * 32)
                lines.extend(sr_block.splitlines())
        except Exception:
            pass

    # ── Verdict ────────────────────────────────────────────────────────────────
    lines.append("─" * 32)

    # Cảnh báo sample nhỏ
    if weighted_n < 5:
        lines.append(f"Mau qua it (weighted N={weighted_n:.1f}) — ket qua chi tham khao")

    # Cảnh báo auto-expanded
    if yr_exp:
        lines.append(f"Da mo rong len {yr_used}Y de co du mau")

    # Kết luận
    if exp <= 0:
        conclusion = "Expectancy am — KHONG vao lenh"
        em = "🔴"
    elif wr < 0.50:
        conclusion = f"WR thap ({wr:.0%}) — chi vao neu R:R >= 1:2"
        em = "🟡"
    elif weighted_n < 5:
        conclusion = f"Mau it (N={weighted_n:.1f}) — vao voi size rat nho neu co"
        em = "🟡"
    elif wr >= 0.60 and exp > 0 and pf >= 1.5:
        conclusion = f"Co hoi tot: WR {wr:.0%}, Exp {exp:+.1f}%, PF {pf_str}"
        em = "✅"
    else:
        conclusion = f"Trung binh: WR {wr:.0%}, Exp {exp:+.1f}% — can xem them"
        em = "🟡"

    lines.append(f"{em} {conclusion}")
    lines.append(sep)

    # ── Debug info ────────────────────────────────────────────────────────────
    if debug:
        lines.append("[DEBUG]")
        lines.append(f"Avg similarity: {sim_avg:.3f} | Threshold: {threshold:.2f}")
        lines.append(f"Total raw matches: {total_raw} | Independent: {ind_n}")
        lines.append(f"Weighted N: {weighted_n:.2f} | Years: {yr_used}")
        if regime_on:
            lines.append(f"Current regime: {cur_regime} ({regime_names.get(cur_regime, '?')})")
        lines.append(sep)

    # ── 3 mẫu gần nhất ───────────────────────────────────────────────────────
    if analogs:
        lines.append("3 MAU TUONG DONG GAN NHAT:")
        for a in analogs[:3]:
            d    = a.get("date", "?")
            sim  = a.get("similarity", 0)
            f30  = a.get("fwd_30")
            rw   = a.get("regime_weight", 1.0)
            sr   = a.get("sample_regime", 0)
            em_a = "+" if (f30 or 0) > 0 else "-"
            rw_tag = f" [w={rw:.1f}]" if regime_on and rw != 1.0 else ""
            sr_tag = f" R{sr}" if sr > 0 else ""
            f30_str = f"{f30:+.1f}%" if f30 is not None else "N/A"
            lines.append(f"  {d}{sr_tag} | Sim {sim:.2f}{rw_tag} | 30D: {f30_str}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CORE RUNNER (sync, chạy trong thread)
# ══════════════════════════════════════════════════════════════════════════════

def _run_analog_sync(symbol: str, use_regime: bool = True) -> dict:
    """
    Pipeline hoàn chỉnh:
      1. Lấy state vector hiện tại (auto_context → db → compute)
      2. find_similar() với regime filter
      3. compute_base_stats() + sanity_check() + compute_reference_score()
      4. Hard gate: Expectancy <= 0, WR < 40%, PF < 1
      5. Trả về analogs + stats + verdict

    Returns dict với keys:
        status: "ok" | "error" | "warn_no_analog" | "warn_low_sample"
        analogs, stats, message (nếu error/warn)
    """
    # ── 1. Load state vector ──────────────────────────────────────────────────
    state_vec  = None
    close_px   = 0.0
    sv_source  = "none"

    # 1a. auto_context (cùng logic với backtest_rule)
    try:
        from auto_context import load_auto_context
        ctx = load_auto_context(symbol)
        if ctx and ctx.get("found") and ctx.get("state_vector"):
            state_vec = ctx["state_vector"]
            sv_source = "auto_context"
    except Exception as e:
        logger.debug(f"analog auto_context fail: {e}")

    # 1b. Fallback: compute trực tiếp từ OHLCV
    if state_vec is None:
        try:
            from vn_loader import load_vn_ohlcv
            from state_vector import compute_state_vector_from_df
            df = load_vn_ohlcv(symbol, days=120, min_bars=60)
            if df is not None and len(df) >= 60:
                state_vec = compute_state_vector_from_df(df)
                sv_source = "computed"
                close_px  = float(df["close"].iloc[-1]) * 1000   # vn_loader unit: nghìn đồng
        except Exception as e:
            logger.debug(f"analog compute sv fail: {e}")

    # 1c. Fallback: DB
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
                        break
                except Exception:
                    continue
            cur.close()
            conn.close()
        except Exception as e:
            logger.debug(f"analog db sv fail: {e}")

    if state_vec is None:
        return {
            "status":  "error",
            "message": (
                f"Khong lay duoc state vector cho {symbol}.\n"
                f"Thu chay /check {symbol} truoc de tao du lieu."
            ),
        }

    # Lấy close price nếu chưa có
    if close_px == 0:
        try:
            from analyzer import get_price_data
            res = get_price_data(symbol, days=5)
            if res.get("success"):
                close_px = float(res["df"]["close"].iloc[-1])
        except Exception:
            pass

    # ── 2. Kiểm tra và build cache nếu chưa có ───────────────────────────────
    try:
        from historical_analog import cache_exists, build_vector_cache
        if not cache_exists(symbol):
            logger.info(f"analog {symbol}: cache chua co → auto build...")
            ok, build_msg = build_vector_cache(symbol)
            if ok:
                logger.info(f"analog {symbol}: cache built OK — {build_msg[:80]}")
            else:
                logger.warning(f"analog {symbol}: build cache fail — {build_msg[:80]}")
                return {
                    "status":  "error",
                    "message": (
                        f"Cache chua co va khong build duoc cho {symbol}.\n"
                        f"Ly do: {build_msg[:200]}\n\n"
                        f"Thu chay /check {symbol} truoc de tao du lieu."
                    ),
                }
    except Exception as e:
        logger.warning(f"analog {symbol}: cache check/build fail: {e}")

    # ── 3. find_similar ───────────────────────────────────────────────────────
    try:
        from historical_analog import find_similar
        current_regime_arg = -1 if use_regime else 0   # -1=auto-detect, 0=disable
        analogs = find_similar(
            symbol         = symbol,
            target_vector  = state_vec,
            years          = 5,
            exclude_days   = 90,
            min_results    = 3,
            current_regime = current_regime_arg,
        )
    except Exception as e:
        logger.error(f"analog find_similar {symbol}: {e}")
        return {"status": "error", "message": f"find_similar loi: {str(e)[:200]}"}

    # Fallback: nếu regime filter làm weighted_n quá thấp → thử lại không filter
    regime_fallback_used = False
    if analogs is None and use_regime:
        logger.info(f"analog {symbol}: regime filter → None, retry without filter")
        try:
            analogs = find_similar(
                symbol         = symbol,
                target_vector  = state_vec,
                years          = 8,
                exclude_days   = 90,
                min_results    = 3,
                current_regime = 0,
            )
            if analogs:
                regime_fallback_used = True
                logger.info(f"analog {symbol}: fallback OK → {len(analogs)} samples")
        except Exception as e:
            logger.warning(f"analog {symbol}: fallback find_similar fail: {e}")

    if not analogs:
        return {
            "status":  "warn_no_analog",
            "message": (
                f"Khong tim duoc mau tuong dong cho {symbol}.\n\n"
                f"Da thu:\n"
                f"  1. Regime filter ON (5Y) → qua it mau\n"
                f"  2. Regime filter OFF (8Y) → van qua it mau\n\n"
                f"Nguyen nhan co the:\n"
                f"  - Cache chua co: /check {symbol} truoc\n"
                f"  - Ma qua it lich su (<300 phien)\n"
                f"  - Vector hien tai qua khac biet so voi lich su\n\n"
                f"Thu: /backtest_rule {symbol} de xem ket qua thay the."
            ),
        }

    # Gắn flag fallback vào meta để format output hiển thị warning
    if regime_fallback_used:
        for a in analogs:
            if "_meta" in a:
                a["_meta"]["regime_fallback"] = True

    # Gắn close_px vào analogs để format output dùng được
    for a in analogs:
        if a.get("close", 0) == 0:
            a["close"] = close_px

    # ── 3. Stats + guardrails ─────────────────────────────────────────────────
    try:
        from guardrails import (
            compute_base_stats, sanity_check,
            compute_reference_score, WINRATE_FLOOR, EXPECTANCY_FLOOR, PF_FLOOR,
        )
        stats = compute_base_stats(analogs)
        if not stats.get("valid"):
            return {
                "status":  "warn_no_analog",
                "message": f"Khong du du lieu thong ke cho {symbol}.",
            }

        # Gắn close vào stats để format output dùng
        if close_px and not stats.get("close"):
            stats["close"] = close_px

        flags = sanity_check(stats)

    except Exception as e:
        logger.error(f"analog guardrails {symbol}: {e}")
        return {"status": "error", "message": f"Guardrails loi: {str(e)[:200]}"}

    return {
        "status":  "ok",
        "analogs": analogs,
        "stats":   stats,
        "flags":   flags,
        "sv_source": sv_source,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def analog_cmd(update, context):
    """
    /analog <MA>          → Phân tích tương đồng lịch sử (regime filter ON)
    /analog <MA> --raw    → Tắt regime filter
    /analog <MA> --debug  → Thêm debug info

    Ví dụ:
        /analog HAH
        /analog VCB --debug
        /analog GAS --raw
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update); return

    args = context.args or []

    if not args:
        await update.message.reply_text(
            "Cu phap: /analog <MA> [--raw] [--debug]\n"
            "Vi du:\n"
            "  /analog HAH\n"
            "  /analog VCB --debug\n"
            "  /analog GAS --raw    (tat regime filter)\n\n"
            "Lenh nay tim cac ngay lich su co vector ky thuat\n"
            "giong hien tai va thong ke ket qua (WR, Expectancy, MAE/MFE)."
        )
        return

    import re as _re
    symbol_raw = args[0].upper().strip()
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol_raw):
        await update.message.reply_text(f"Ma '{symbol_raw}' khong hop le.")
        return

    symbol     = symbol_raw
    use_regime = "--raw"   not in args
    debug      = "--debug" in args

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "anon"
    since   = time.time() - _last_analog.get(user_id, 0)
    if since < ANALOG_COOLDOWN:
        wait = int(ANALOG_COOLDOWN - since)
        await update.message.reply_text(
            f"Vui long cho {wait}s truoc khi /analog tiep."
        )
        return
    _last_analog[user_id] = time.time()

    chat_id = update.effective_chat.id
    regime_tag = "" if use_regime else " [regime filter OFF]"
    msg = await update.message.reply_text(
        f"Dang phan tich tuong dong lich su: {symbol}{regime_tag}...\n"
        f"(Lan dau co the mat 60-120s neu phai build cache vector)"
    )

    async def _bg():
        try:
            # ── Core analog ─────────────────────────────────────────────────
            result = await asyncio.to_thread(_run_analog_sync, symbol, use_regime)

            if result["status"] in ("error", "warn_no_analog"):
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=_plain(result["message"])[:4096],
                )
                return

            analogs = result["analogs"]
            stats   = result["stats"]
            flags   = result.get("flags", [])

            # ── Wave (best-effort) ───────────────────────────────────────────
            wave_result = None
            try:
                from wave_pattern import analyze_wave
                wave_result = await asyncio.to_thread(analyze_wave, symbol, False)
            except Exception as we:
                logger.debug(f"analog: wave skip: {we}")

            # ── Stock Regime (best-effort) ───────────────────────────────────
            sr_result = None
            try:
                from stock_regime import get_stock_regime
                sr_result = await asyncio.to_thread(get_stock_regime, symbol)
            except Exception as sre:
                logger.debug(f"analog: stock_regime skip: {sre}")

            # ── Format output ────────────────────────────────────────────────
            output = _format_analog_result(
                symbol      = symbol,
                analogs     = analogs,
                stats       = stats,
                wave_result = wave_result,
                sr_result   = sr_result,
                debug       = debug,
            )

            # Flags từ sanity_check (append sau output chính)
            flag_labels = {
                "OUTLIER_RISK":    "Canh bao: Return cao bat thuong voi it mau",
                "HIGH_DISPERSION": "Canh bao: Ket qua qua phan tan",
                "DEAD_CAT":        "Canh bao: Tang ngan han nhung xu huong dai han yeu",
                "RECENCY_BIAS":    "Canh bao: Phan lon mau trong 12 thang gan — pattern moi",
            }
            flag_lines = []
            for f in flags:
                key = f.split(":")[0]
                if key in flag_labels:
                    flag_lines.append(f"⚠️  {flag_labels[key]}")
            if flag_lines:
                output += "\n" + "\n".join(flag_lines)

            # Gửi output
            if len(output) <= 4096:
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg.message_id,
                        text=_plain(output),
                    )
                except Exception:
                    await context.bot.send_message(
                        chat_id=chat_id, text=_plain(output)[:4096]
                    )
            else:
                split_at = output.rfind("\n", 0, 4000)
                if split_at < 0: split_at = 4000
                part1 = output[:split_at].strip()
                part2 = output[split_at:].strip()
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg.message_id,
                        text=_plain(part1),
                    )
                except Exception:
                    await context.bot.send_message(
                        chat_id=chat_id, text=_plain(part1)[:4096]
                    )
                if part2:
                    await context.bot.send_message(
                        chat_id=chat_id, text=_plain(part2)[:4096]
                    )

        except Exception as e:
            import traceback
            logger.error(f"analog_cmd bg error: {e}\n{traceback.format_exc()}")
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=f"Loi /analog {symbol}: {str(e)[:300]}",
                )
            except Exception:
                try:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"Loi /analog {symbol}: {str(e)[:300]}",
                    )
                except Exception:
                    pass

    asyncio.create_task(_bg())
