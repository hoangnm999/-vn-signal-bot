"""
backtest_rule_cmd.py — /backtest_rule command handler cho bot.py.

Import vào bot.py:
    from backtest_rule_cmd import backtest_rule_cmd, BACKTEST_RULE_COOLDOWN, _last_backtest_rule

Đăng ký trong main():
    app.add_handler(CommandHandler("backtest_rule", backtest_rule_cmd))
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import shutil
import tempfile
import time
import json as _json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BACKTEST_RULE_COOLDOWN = 120   # 2 phút per user
_last_backtest_rule: dict[str, float] = {}
# Đếm số lần run per (user, symbol) để cảnh báo overfitting
_backtest_run_count: dict[str, int] = {}   # key = f"{user_id}:{symbol}"
OVERFIT_WARN_THRESHOLD = 3   # cảnh báo sau 3 lần run cùng symbol

# ── Helpers tái sử dụng từ bot.py (import lúc runtime để tránh circular) ──────
def _plain(text: str) -> str:
    import re
    text = re.sub(r"[*`]", "", text)
    return text.replace("\\_", "_")


# ══════════════════════════════════════════════════════════════════════════════
# PARSE COMMAND ARGS
# ══════════════════════════════════════════════════════════════════════════════

_HELP_TEXT = """\
BACKTEST_RULE — Kiem tra rule tu hoi dong phan tich
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cu phap:
  /backtest_rule <MA> "<entry_rule>" "<exit_rule>" [ngay]

INDICATORS:
  close, open, high, low, volume
  sma20, sma50, ema12, ema26      (so bat ki: sma(N))
  rsi, rsi(N)                     (mac dinh N=14)
  macd, macd_signal, macd_hist
  bb_upper, bb_lower, bb_mid
  atr, stoch_k, stoch_d
  high20, low20                   (rolling N bar: highN, lowN)
  volume_sma20                    (volume MA)

FUNCTIONS:
  crossover(a, b)   — a cat len b
  crossunder(a, b)  — a cat xuong b
  rising(ind, N)    — tang N bar lien tiep
  falling(ind, N)   — giam N bar lien tiep
  breakout(high,N)  — close > high N bar truoc
  prev(ind, N)      — gia tri N bar truoc

EXIT ONLY:
  trailing_stop(5%) — trailing stop 5%
  take_profit(10%)  — chot loi 10%
  stop_loss(5%)     — cat lo 5%
  hold(20)          — giu toi da 20 bar

PHEP TINH: >, <, >=, <=, ==, and, or, not, +, -, *, /

VI DU:
  /backtest_rule VCB "rsi < 30" "rsi > 70"
  /backtest_rule HPG "rsi < 30 and close > sma20" "rsi > 70 or trailing_stop(5%)"
  /backtest_rule STB "crossover(ema12, ema26)" "crossunder(ema12, ema26)"
  /backtest_rule FPT "close > high20 and volume > volume_sma20 * 1.5" "take_profit(15%) or stop_loss(7%)"
  /backtest_rule VCB "close < 62100 and rsi < 30" "rsi > 70 or close > 68000" 500
"""

_EXAMPLES_SHORT = (
    "Vi du:\n"
    "  /backtest_rule VCB \"rsi < 30\" \"rsi > 70\"\n"
    "  /backtest_rule HPG \"rsi < 30 and close > sma20\" \"rsi > 70 or trailing_stop(5%)\"\n"
    "  /backtest_rule STB \"crossover(ema12, ema26)\" \"crossunder(ema12, ema26)\"\n"
    "  /backtest_rule FPT \"close < 62100 and rsi < 30\" \"close > 68000\""
)


def _parse_args(args: list[str]) -> tuple[str, str, str, int] | str:
    """
    Parse context.args → (symbol, entry_rule, exit_rule, days) hoặc error string.

    Hỗ trợ 2 cách nhập:
      1. Đã có quotes từ Telegram: args = ["VCB", "rsi < 30", "rsi > 70", "365"]
      2. Không có quotes (user gõ cả chuỗi): ghép lại và split bằng dấu "
    """
    if not args:
        return "error:empty"

    symbol = args[0].upper().strip()
    import re as _re
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol):
        return f"error:symbol:{symbol}"

    if len(args) < 3:
        return "error:notenough"

    # Telegram thường tách theo khoảng trắng, quotes nằm trong một arg
    # → ghép args[1:] lại và tìm các chuỗi trong ""
    joined = " ".join(args[1:])

    # Tìm các chuỗi trong dấu nháy kép
    import re
    quoted = re.findall(r'"([^"]*)"', joined)

    if len(quoted) >= 2:
        entry_rule = quoted[0].strip()
        exit_rule  = quoted[1].strip()
        # days là phần còn lại ngoài quotes
        remainder = re.sub(r'"[^"]*"', '', joined).strip()
        try:
            days = max(100, min(int(remainder), 1500)) if remainder else 365
        except ValueError:
            days = 365
    else:
        # Không có quotes — args[1] = entry, args[2] = exit, args[3] = days
        entry_rule = args[1].strip()
        exit_rule  = args[2].strip()
        try:
            days = max(100, min(int(args[3]), 1500)) if len(args) > 3 else 365
        except (ValueError, IndexError):
            days = 365

    if not entry_rule:
        return "error:empty_entry"
    if not exit_rule:
        return "error:empty_exit"

    return symbol, entry_rule, exit_rule, days


# ══════════════════════════════════════════════════════════════════════════════
# RESULT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════


def _calc_trade_analytics(trades: list, df) -> dict:
    """
    Hold Duration Stats và MAE/MFE từ danh sách trades + OHLCV df.
    Hold = số bars giữa BUY→SELL.
    MAE  = min(low trong window) / entry - 1
    MFE  = max(high trong window) / entry - 1
    MFE capture rate = actual_return / MFE × 100
    """
    import numpy as np
    sells = [t for t in trades if t.get("type") == "SELL"]
    buys  = [t for t in trades if t.get("type") == "BUY"]
    if not sells or df is None or len(df) < 2:
        return {}
    try:
        df_r = df.reset_index(drop=True)
        date_col = "date" if "date" in df_r.columns else df_r.columns[0]
        d2i = {str(row[date_col])[:10]: i for i, row in df_r.iterrows()}
    except Exception:
        return {}

    holds, maes, mfes, caps = [], [], [], []
    buy_idx = 0
    for sell in sells:
        sell_date = str(sell.get("date", ""))[:10]
        entry = None
        for b in buys[buy_idx:]:
            if str(b.get("date", ""))[:10] < sell_date:
                entry = b; buy_idx += 1
            else:
                break
        if entry is None:
            continue
        ep = float(entry.get("price", 0))
        xp = float(sell.get("price", 0))
        if ep <= 0:
            continue
        ed = str(entry.get("date", ""))[:10]
        i0 = d2i.get(ed); i1 = d2i.get(sell_date)
        if i0 is None or i1 is None or i1 <= i0:
            continue
        holds.append(i1 - i0)
        # MAE/MFE: chỉ tính trong holding period (bar SAU entry đến TRƯỚC bar exit)
        # Không bao gồm bar entry (giá đã biết lúc mua) và bar exit (đã thoát)
        win = df_r.iloc[i0 + 1: i1]
        if len(win) == 0:
            # Trade mở và đóng trong 1-2 bars — không đủ data cho MAE/MFE
            continue
        if "low" in win.columns and "high" in win.columns:
            lows  = win["low"].values.astype(float)
            highs = win["high"].values.astype(float)
            mae = (min(lows)  - ep) / ep * 100
            mfe = (max(highs) - ep) / ep * 100
            maes.append(round(mae, 2))
            mfes.append(round(mfe, 2))
            if mfe > 0:
                caps.append(round((xp - ep) / ep * 100 / mfe * 100, 1))

    result = {}
    if holds:
        result.update({"hold_avg": round(float(np.mean(holds)), 1),
                       "hold_median": round(float(np.median(holds)), 1),
                       "hold_max": int(max(holds)), "hold_min": int(min(holds))})
    if maes:
        result.update({"mae_avg": round(float(np.mean(maes)), 2),
                       "mae_median": round(float(np.median(maes)), 2),
                       "mae_worst": round(float(min(maes)), 2)})
    if mfes:
        result.update({"mfe_avg": round(float(np.mean(mfes)), 2),
                       "mfe_median": round(float(np.median(mfes)), 2),
                       "mfe_best": round(float(max(mfes)), 2)})
    if caps:
        result["mfe_capture_rate"] = round(float(np.mean(caps)), 1)
    return result

def _format_rule_result(
    metrics: dict,
    symbol: str,
    entry_rule: str,
    exit_rule: str,
    days: int,
    n_trades: int,
    exit_ec_entry,
    exit_ec_exit,
    trades_sample: list,
    n_buy_signals: int,
    n_sell_signals: int,
    bars: int,
    trade_analytics: dict = None,
    wave_result: dict = None,
    stock_regime_result: dict = None,
) -> str:
    """Tạo text output compact cho /backtest_rule."""
    ret  = metrics.get("total_return_pct", 0)
    wr   = metrics.get("win_rate_pct", 0)
    pf   = metrics.get("profit_factor", 0)
    sh   = metrics.get("sharpe_ratio", 0)
    mdd  = metrics.get("max_drawdown_pct", 0)
    aw   = metrics.get("avg_win_pct", 0)
    al   = metrics.get("avg_loss_pct", 0)

    ta   = trade_analytics or {}
    mae  = ta.get("mae_median", ta.get("mae_avg", 0))
    mfe  = ta.get("mfe_avg", 0)
    hold = ta.get("hold_avg", 0)
    cap  = ta.get("mfe_capture_rate", 0)

    # Dynamic exit summary
    dyn_parts = []
    _ts = getattr(exit_ec_exit, "trailing_stop_pct", None)
    _tp = getattr(exit_ec_exit, "take_profit_pct",  None)
    _sl = getattr(exit_ec_exit, "stop_loss_pct",    None)
    _hb = getattr(exit_ec_exit, "hold_bars",        None)
    if _ts: dyn_parts.append(f"TS {_ts:.1f}%")
    if _tp: dyn_parts.append(f"TP {_tp:.1f}%")
    if _sl: dyn_parts.append(f"SL {_sl:.1f}%")
    if _hb: dyn_parts.append(f"Hold {_hb}b")
    dyn_str = " | ".join(dyn_parts) if dyn_parts else ""

    # Entry/SL/TP từ trade_analytics nếu có, fallback tính từ metrics
    entry_price = 0.0
    try:
        sells = [t for t in trades_sample if t.get("type") == "SELL"]
        buys  = [t for t in trades_sample if t.get("type") == "BUY"]
        if buys:
            entry_price = float(buys[-1].get("price", 0))
    except Exception:
        pass

    sl_price  = entry_price * (1 + mae / 100 - 0.02) if entry_price and mae else 0
    tp1_price = entry_price * (1 + aw / 100)          if entry_price and aw  else 0
    rr_tp1    = abs(aw / (mae - 2)) if mae < 0 else 0

    # Verdict / kết luận
    em_ret = "✅" if ret > 0 and wr >= 55 and pf >= 1.5 else              "🟡" if ret > 0 else "🔴"
    if ret > 0 and wr >= 55 and pf >= 1.5:
        conclusion = f"Co hoi tot: WR {wr:.0f}%, R:R 1:{rr_tp1:.1f}, PF {pf:.2f}"
    elif ret > 0:
        conclusion = f"Co loi nhuan nhung WR ({wr:.0f}%) hoac PF ({pf:.2f}) chua manh"
    else:
        conclusion = f"Rule chua hieu qua trong {days}D — xem xet dieu chinh"

    cap_note = ""
    if cap > 0:
        cap_note = " ⚠️ Exit som" if cap < 40 else " ✅ Hieu qua" if cap > 80 else ""

    # Wave summary — 2 dòng nếu có
    wave_lines = []
    if wave_result and wave_result.get("ok"):
        wd      = wave_result.get("verdict", "KHONG RO")
        ws_up   = wave_result.get("score_up_adj",   wave_result.get("score_up",   0))
        ws_dn   = wave_result.get("score_down_adj", wave_result.get("score_down", 0))
        wconf   = wave_result.get("confidence", 0)
        wrel_n  = min(wave_result.get("n_up", 0), wave_result.get("n_down", 0))
        wrel_c  = wave_result.get("confidence", 0)
        def _wrel(n, c):
            if n >= 20 and c >= 0.15: return "★★★"
            if n >= 15 and c >= 0.10: return "★★☆"
            if n >= 10 and c >= 0.08: return "★★☆"
            return "★☆☆"
        wrel = _wrel(wrel_n, wrel_c)
        w_em = "🟢" if wd == "SONG TANG" else "🔴" if wd == "SONG GIAM" else "🟡"
        wave_lines.append(
            f"🌊 Wave: {w_em} {wd} {wrel} "
            f"({ws_dn:.0%} giam vs {ws_up:.0%} tang)"
        )
        # Subtype của chiều phù hợp verdict
        if wd == "SONG GIAM":
            st = wave_result.get("subtype_stats_down", {})
            cmp = wave_result.get("subtype_compare_down", {})
        elif wd == "SONG TANG":
            st = wave_result.get("subtype_stats_up", {})
            cmp = wave_result.get("subtype_compare_up", {})
        else:
            st = {}; cmp = {}
        if st and st.get("total_known", 0) >= 6:
            counts = st.get("counts", {})
            pct    = st.get("pct", {})
            if wd == "SONG GIAM":
                peak_pct = pct.get("PEAK_REAL", 0)
                corr_pct = pct.get("CORRECTION", 0)
                closest  = cmp.get("closest", "")
                cl_em    = "🔴" if closest == "PEAK_REAL" else "🟡"
                cl_lbl   = "Peak that (giam sau)" if closest == "PEAK_REAL" else "Correction (giam roi len)"
                wave_lines.append(
                    f"   Loai: Peak that {peak_pct:.0f}% | "
                    f"Correction {corr_pct:.0f}% | "
                    f"{cl_em} Gan nhat: {cl_lbl}"
                )
            elif wd == "SONG TANG":
                bot_pct  = pct.get("BOTTOM_REAL", 0)
                ral_pct  = pct.get("RELIEF_RALLY", 0)
                closest  = cmp.get("closest", "")
                cl_em    = "🟢" if closest == "BOTTOM_REAL" else "🟡"
                cl_lbl   = "Day that (tang ben vung)" if closest == "BOTTOM_REAL" else "Relief Rally (tang ngan)"
                wave_lines.append(
                    f"   Loai: Day that {bot_pct:.0f}% | "
                    f"Relief Rally {ral_pct:.0f}% | "
                    f"{cl_em} Gan nhat: {cl_lbl}"
                )

    # Stock Regime section (Phase 2 GMM)
    sr_lines = []
    if stock_regime_result and stock_regime_result.get("ok"):
        try:
            from stock_regime import format_stock_regime_for_backtest
            _sr_block = format_stock_regime_for_backtest(stock_regime_result)
            if _sr_block:
                sr_lines = _sr_block.splitlines()
        except Exception:
            pass

    # ── Assemble ──────────────────────────────────────────────────────
    sep = "═" * 32
    lines = [
        f"/backtest {symbol}  [{days}n | {n_trades} lenh | {bars}b]",
        sep,
        f"📊 WR {wr:.0f}%  |  PF {pf:.2f}  |  Sharpe {sh:.2f}",
        f"   LN: {ret:+.1f}% | MaxDD {mdd:.1f}% | "
        + (f"Hold ~{hold:.0f}b" if hold else f"WinAvg {aw:+.1f}%"),
        f"   MAE {mae:+.1f}%  |  MFE {mfe:+.1f}%"
        + (f"  |  Capture {cap:.0f}%{cap_note}" if cap else ""),
    ]

    # Entry/SL/TP nếu có giá
    if entry_price and sl_price and tp1_price:
        lines += [
            "─" * 32,
            f"📌 Entry: {entry_price:,.0f}"
            + (f"  |  SL: {sl_price:,.0f} ({mae-2:.1f}%)" if sl_price else ""),
            f"   TP1: {tp1_price:,.0f} (+{aw:.1f}%)"
            + (f"  R:R 1:{rr_tp1:.1f}" if rr_tp1 else ""),
        ]
        if dyn_str:
            lines.append(f"   Dynamic: {dyn_str}")

    # Wave section
    if wave_lines:
        lines.append("─" * 32)
        lines += wave_lines

    # Stock Regime section
    if sr_lines:
        lines.append("─" * 32)
        lines += sr_lines

    # Kết luận
    lines += [
        "─" * 32,
        f"{em_ret} {conclusion}",
        sep,
    ]

    return "\n".join(lines)


def _format_trades_msg(trades: list) -> str:
    """Format 5 lệnh gần nhất."""
    sells = [t for t in trades if t.get("type") == "SELL"][-5:]
    if not sells:
        return ""
    lines = ["5 LENH BAN GAN NHAT:"]
    for t in sells:
        em  = "✅" if t.get("pnl_vnd", 0) > 0 else "❌"
        by  = f" [{t['exit_by']}]" if t.get("exit_by") else ""
        lines.append(
            f"{em} {t['date']} | Gia: {t['price']:,.0f}{by}\n"
            f"   PnL: {t.get('pnl_pct', 0):+.2f}%"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# CORE RUNNER (sync, chạy trong thread)
# ══════════════════════════════════════════════════════════════════════════════

def _run_rule_backtest_sync(
    symbol: str,
    entry_rule: str,
    exit_rule: str,
    days: int,
) -> dict:
    """
    Pipeline hoàn chỉnh: load data → parse rules → run backtest → chart.
    Chạy trong asyncio.to_thread.
    """
    try:
        from vn_loader import load_vn_ohlcv
        from rule_engine import (
            generate_rule_signals, apply_dynamic_exit, parse_rule,
            format_rule_explanation,
        )
        from backtest_engine import _run_backtest_core, _calc_metrics, _draw_chart
    except ImportError as _ie:
        return {"status": "error", "error": f"Thieu module: {_ie}"}

    config = {
        "symbol":           symbol,
        "initial_capital":  100_000_000,
        "commission_pct":   0.0015,
        "slippage_pct":     0.001,
        "position_size":    1.0,
        "allow_short":      False,
        "days":             days,
    }

    # 1. Load data — thử vn_loader trước, fallback sang analyzer.get_price_data
    df = load_vn_ohlcv(symbol, days=days)
    if df is None or (hasattr(df, '__len__') and len(df) < 30):
        try:
            from analyzer import get_price_data
            result = get_price_data(symbol, days=min(days, 500))
            if isinstance(result, dict) and result.get("success"):
                df = result["df"]
                logger.info(f"backtest: fallback sang analyzer.get_price_data cho {symbol}")
        except Exception as _fe:
            logger.warning(f"backtest fallback get_price_data fail: {_fe}")
    if df is None or len(df) < 30:
        return {
            "status": "error",
            "error":  f"Khong du du lieu cho {symbol} (can toi thieu 30 bars). "
                      f"Thu /debug {symbol} de kiem tra data sources."
        }

    # 2. Parse & compile rules → signals
    # Wrap riêng để ParseError cho message rõ ràng hơn là generic exception
    try:
        signals, entry_ec, exit_ec, ctx = generate_rule_signals(df, entry_rule, exit_rule)
    except Exception as pe:
        return {
            "status": "error",
            "error":  f"Loi parse/compile rule: {str(pe)[:200]}\n"
                      f"ENTRY: {entry_rule[:80]}\nEXIT: {exit_rule[:80]}"
        }

    n_buy  = int((signals == 1).sum())
    n_sell = int((signals == -1).sum())

    if n_buy == 0:
        return {
            "status": "warn_no_signals",
            "message": (
                f"Rule ENTRY '{entry_rule}' khong tao duoc tin hieu nao "
                f"trong {days} ngay du lieu.\n"
                f"Thu no long rules (vd: rsi < 35 thay vi rsi < 30), "
                f"hoac tang so ngay."
            ),
        }

    # 3. Backtest — dùng dynamic exit nếu có trailing/tp/sl/hold
    if exit_ec.has_dynamic():
        bt = apply_dynamic_exit(df, signals, exit_ec, config)
    else:
        bt = _run_backtest_core(df, signals, config)

    equity = bt["equity"]
    trades = bt["trades"]

    # 4. Metrics
    metrics = _calc_metrics(equity, trades, config)

    # 5. Chart
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        chart_tmp = tf.name

    try:
        _draw_chart(df, signals, equity, metrics, config, chart_tmp)
        chart_ok = pathlib.Path(chart_tmp).exists()
    except Exception as ce:
        logger.warning(f"Chart warning: {ce}")
        chart_ok = False
        chart_tmp = ""

    return {
        "status":    "ok",
        "metrics":   metrics,
        "trades":    trades[-30:],
        "n_trades":  len([t for t in trades if t.get("type") == "SELL"]),
        "n_buy":     n_buy,
        "n_sell":    n_sell,
        "bars":      len(df),
        "chart_path": chart_tmp if chart_ok else "",
        "entry_ec":   entry_ec,
        "exit_ec":    exit_ec,
        "df":         df,
    }



# ══════════════════════════════════════════════════════════════════════════════
# AUTO CONTEXT HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def _handle_auto_context(update, context, symbol: str, plain_fn):
    """
    /backtest_rule VCB (không có rule):
      1. Load context từ DB → ưu tiên vibe (trade_plan) trong 7 ngày.
      2a. Có trade_plan → convert rule → backtest.
      2b. Chỉ có state_vector → analog search.
      3. Không có gì → hướng dẫn.
    """
    try:
        from auto_context import (
            load_auto_context, trade_plan_to_rules,
            format_trade_plan_summary, format_no_context_msg,
        )
        from historical_analog import (
            cache_exists, find_similar, format_analog_report, build_vector_cache,
        )
    except ImportError as _ie:
        await update.message.reply_text(
            f"Thieu module: {_ie}\n"
            f"De dung rule thu cong: /backtest_rule {symbol} \"rsi < 30\" \"rsi > 70\""
        )
        return

    NL      = "\n"
    chat_id = update.effective_chat.id
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"

    # Rate limit
    since = time.time() - _last_backtest_rule.get(user_id, 0)
    if since < BACKTEST_RULE_COOLDOWN:
        wait = int(BACKTEST_RULE_COOLDOWN - since)
        await update.message.reply_text(
            f"Vui long cho {wait}s truoc khi /backtest_rule tiep."
        )
        return
    _last_backtest_rule[user_id] = time.time()

    msg = await update.message.reply_text(
        f"Auto Context: dang tim boi canh phan tich cho {symbol}..."
    )

    try:
        # ── Bước 1: Load context từ DB ────────────────────────────────────
        ctx = await asyncio.to_thread(load_auto_context, symbol)

        if not ctx["found"]:
            await msg.edit_text(plain_fn(format_no_context_msg(symbol))[:4096])
            return

        source     = ctx["source"]
        trade_plan = ctx["trade_plan"]
        state_vec  = ctx["state_vector"]
        verdict    = ctx["verdict"] or "N/A"
        created    = ctx["created_at"]
        age_str    = ""
        if created:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_hours = (now - created).total_seconds() / 3600
            age_str   = f"{age_hours:.0f} gio truoc"

        # ── Bước 2a: Có trade_plan → backtest ────────────────────────────
        if trade_plan and trade_plan.get("entry"):
            entry_rule, exit_rule = trade_plan_to_rules(trade_plan)
            plan_summary          = format_trade_plan_summary(trade_plan)

            loading_lines = [
                f"Tim thay Trade Plan tu /vibe ({age_str}).",
                f"Verdict: {verdict}",
                "",
                plan_summary,
                "",
                "Dang chuyen thanh rule va chay backtest 365 ngay...",
            ]
            await msg.edit_text(plain_fn(NL.join(loading_lines))[:4096])

            result = await asyncio.to_thread(
                _run_rule_backtest_sync, symbol, entry_rule, exit_rule, 365
            )

            header_lines = [
                f"AUTO BACKTEST: {symbol}",
                f"Nguon: /vibe ({age_str}) | Verdict: {verdict}",
                plan_summary,
                f"Rule entry : {entry_rule[:70]}",
                f"Rule exit  : {exit_rule[:70]}",
                "=" * 40,
            ]
            header = NL.join(header_lines)

            if result["status"] == "warn_no_signals":
                warn_lines = [
                    header,
                    "",
                    "Canh bao: Rule khong tao duoc tin hieu nao trong 365 ngay.",
                    "Co the gia hien tai da vuot entry price.",
                    f"Thu /backtest_rule {symbol} voi rule thu cong.",
                ]
                await msg.edit_text(plain_fn(NL.join(warn_lines))[:4096])
                return

            if result["status"] == "error":
                err_lines = [header, "", f"Loi: {result.get('error','?')[:200]}"]
                await msg.edit_text(plain_fn(NL.join(err_lines))[:4096])
                return

            # Format kết quả
            metrics  = result["metrics"]
            n_trades = result["n_trades"]
            _ta_auto = _calc_trade_analytics(result["trades"], result.get("df"))
            _wave_auto = None
            try:
                from wave_pattern import analyze_wave
                _wave_auto = await asyncio.to_thread(analyze_wave, symbol, False)
            except Exception as _we:
                logger.debug(f"wave auto lookup skip: {_we}")
            bt_summary = _format_rule_result(
                metrics=metrics, symbol=symbol,
                entry_rule=entry_rule, exit_rule=exit_rule,
                days=365, n_trades=n_trades,
                exit_ec_entry=result["entry_ec"], exit_ec_exit=result["exit_ec"],
                trades_sample=result["trades"],
                n_buy_signals=result["n_buy"], n_sell_signals=result["n_sell"],
                bars=result["bars"],
                trade_analytics=_ta_auto,
                wave_result=_wave_auto,
            )
            full_text = plain_fn(header + NL + bt_summary)
            await msg.edit_text(full_text[:4096])

            # Chart
            chart_path = result.get("chart_path", "")
            if chart_path and pathlib.Path(chart_path).exists():
                cap_lines = [
                    f"Auto Backtest: {symbol} | Vibe Trade Plan",
                    f"Return: {metrics.get('total_return_pct',0):+.1f}% | Sharpe: {metrics.get('sharpe_ratio',0):.2f}",
                ]
                try:
                    with open(chart_path, "rb") as f:
                        await context.bot.send_photo(
                            chat_id=chat_id, photo=f,
                            caption=plain_fn(NL.join(cap_lines))[:1024],
                        )
                finally:
                    pathlib.Path(chart_path).unlink(missing_ok=True)

            # Trades
            trade_msg = _format_trades_msg(result["trades"])
            if trade_msg:
                await context.bot.send_message(
                    chat_id=chat_id, text=plain_fn(trade_msg)[:4096]
                )

        # ── Bước 2b: Chỉ có state_vector → analog search ─────────────────
        elif state_vec:
            loading_lines2 = [
                f"Tim thay boi canh tu /check ({age_str}).",
                f"Verdict: {verdict}",
                "Dang tim ngay lich su tuong dong...",
            ]
            await msg.edit_text(plain_fn(NL.join(loading_lines2))[:4096])

            cache_ok = await asyncio.to_thread(cache_exists, symbol)
            if not cache_ok:
                await msg.edit_text(
                    plain_fn(
                        f"Dang tao cache vector cho {symbol}..."
                        + NL +
                        "Vui long doi ~20-30 giay (chi can lam 1 lan)."
                    )[:4096]
                )
                ok, build_msg = await asyncio.to_thread(build_vector_cache, symbol)
                if not ok:
                    fail_lines = [
                        f"Khong tao duoc cache: {build_msg[:200]}",
                        f"Thu /backtest_rule {symbol} voi rule thu cong.",
                    ]
                    await msg.edit_text(plain_fn(NL.join(fail_lines))[:4096])
                    return

            analogs = await asyncio.to_thread(
                find_similar, symbol, state_vec, top_n=3, years=5
            )

            if not analogs:
                no_analog_lines = [
                    f"Khong tim thay ngay tuong dong cho {symbol}.",
                    "Co the cache chua du du lieu (can >= 6 thang).",
                    f"Thu /backtest_rule {symbol} voi rule thu cong.",
                ]
                await msg.edit_text(plain_fn(NL.join(no_analog_lines))[:4096])
                return

            from historical_analog import format_analog_report
            # Lay gia hien tai thuc te
            _current_price = 0.0
            try:
                from vn_loader import load_vn_ohlcv
                _df_price = load_vn_ohlcv(symbol, days=5, min_bars=1)
                if _df_price is not None and len(_df_price) > 0:
                    # Lưu theo nghìn đồng — nhất quán với analogs["close"] từ cache
                    _current_price = float(_df_price["close"].iloc[-1])
            except Exception as _pe:
                logger.warning(f"backtest_rule: lay gia hien tai {symbol} fail: {_pe}")
            # format_analog_report đã có header riêng — không cần wrapper
            report = format_analog_report(symbol, analogs, state_vec,
                                          current_price=_current_price)
            await msg.edit_text(plain_fn(report)[:4096])

        else:
            no_info_lines = [
                f"Boi canh {symbol} ({age_str}) khong co du thong tin.",
                f"Hay chay lai /check {symbol} hoac /vibe {symbol}.",
            ]
            await msg.edit_text(plain_fn(NL.join(no_info_lines))[:4096])

    except Exception as e:
        import traceback
        logger.error(f"_handle_auto_context error: {e}\n{traceback.format_exc()}")
        try:
            await msg.edit_text(
                plain_fn(f"Loi Auto Context {symbol}: {str(e)[:200]}")[:4096]
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def backtest_rule_cmd(update, context):
    """
    /backtest_rule <MA> ["<entry_rule>" "<exit_rule>"] [ngay]

    CHE DO AUTO CONTEXT (khong co rule):
      /backtest_rule VCB  → tu dong lay context tu /vibe hoac /check gan nhat

    CHE DO MANUAL (co rule):
      /backtest_rule VCB "rsi < 30" "rsi > 70"
      /backtest_rule HPG "rsi < 30 and close > sma20" "rsi > 70 or trailing_stop(5%)"
    """
    from telegram import Update
    from telegram.ext import ContextTypes

    try:
        from bot import is_allowed, _deny
        plain = _plain   # dùng local _plain thay vì import từ bot (tránh circular)
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass
        plain = _plain

    if not is_allowed(update):
        await _deny(update); return

    args = context.args or []

    if not args:
        await update.message.reply_text(_HELP_TEXT[:4096])
        return

    if args[0].lower() in ("help", "--help", "-h", "?"):
        await update.message.reply_text(_HELP_TEXT[:4096])
        return

    # Validate symbol
    symbol_raw = args[0].upper().strip()
    import re as _re2
    if not _re2.match(r'^[A-Z0-9]{2,10}$', symbol_raw):
        await update.message.reply_text(f"Ma '{symbol_raw}' khong hop le (2-10 chu/so)."); return
    symbol = symbol_raw

    # AUTO CONTEXT khi chỉ có 1 arg (symbol)
    if len(args) == 1:
        await _handle_auto_context(update, context, symbol, plain)
        return

    # MANUAL: parse đầy đủ
    parsed = _parse_args(args)
    if isinstance(parsed, str):
        if parsed == "error:empty":
            await update.message.reply_text(_HELP_TEXT[:4096]); return
        if parsed.startswith("error:symbol:"):
            sym = parsed.split(":")[-1]
            await update.message.reply_text(f"Ma '{sym}' khong hop le (2-10 chu/so)."); return
        if parsed == "error:notenough":
            await update.message.reply_text(
                "Thieu tham so. Can ca entry_rule va exit_rule.\n\n" + _EXAMPLES_SHORT
            ); return
        if parsed == "error:empty_entry":
            await update.message.reply_text("Entry rule khong duoc de trong."); return
        if parsed == "error:empty_exit":
            await update.message.reply_text("Exit rule khong duoc de trong."); return
        await update.message.reply_text(f"Loi: {parsed}"); return

    symbol, entry_rule, exit_rule, days = parsed

    # Validate rules trước khi bắt đầu (fast fail, không tốn thời gian load data)
    try:
        from rule_engine import parse_rule
        parse_rule(entry_rule)
        parse_rule(exit_rule)
    except Exception as pe:
        await update.message.reply_text(
            f"LOI PARSE RULE:\n{str(pe)}\n\n" + _EXAMPLES_SHORT
        )
        return

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_backtest_rule.get(user_id, 0)
    if since < BACKTEST_RULE_COOLDOWN:
        wait = int(BACKTEST_RULE_COOLDOWN - since)
        await update.message.reply_text(f"Vui long cho {wait}s truoc khi /backtest_rule tiep."); return
    _last_backtest_rule[user_id] = time.time()

    # Đếm số lần run để cảnh báo overfitting
    symbol_for_count = (args[0].upper() if args else "?")
    run_key  = f"{user_id}:{symbol_for_count}"
    run_count = _backtest_run_count.get(run_key, 0) + 1
    _backtest_run_count[run_key] = run_count
    overfit_warning = ""
    if run_count >= OVERFIT_WARN_THRESHOLD:
        overfit_warning = (
            f"⚠️ Ban da chay backtest {run_count} lan cho {symbol_for_count}.\n"
            f"   Ket qua co the bi overfit neu ban dang tune rule de maximize performance.\n"
            f"   Hay test rule tren du lieu chua tung xem truoc khi su dung that.\n"
        )

    chat_id = update.effective_chat.id

    # Gửi loading message
    msg = await update.message.reply_text(
        f"Dang chay rule backtest: {symbol} ({days} ngay)...\n"
        f"ENTRY: {entry_rule[:80]}\n"
        f"EXIT : {exit_rule[:80]}\n"
        f"Co the mat 30-90 giay."
    )

    # Chạy trong background task
    async def _bg():
        try:
            result = await asyncio.to_thread(
                _run_rule_backtest_sync, symbol, entry_rule, exit_rule, days
            )

            if result["status"] == "warn_no_signals":
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=plain(result["message"])[:4096],
                )
                return

            if result["status"] == "error":
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=f"Loi backtest: {result.get('error','?')[:300]}",
                )
                return

            # Build summary text
            _ta = _calc_trade_analytics(result["trades"], result.get("df"))
            # Load wave result nếu có (best-effort, không block nếu lỗi)
            _wave_res = None
            try:
                from wave_pattern import analyze_wave
                _wave_res = await asyncio.to_thread(analyze_wave, symbol, False)
            except Exception as _we:
                logger.debug(f"wave lookup skip: {_we}")

            # Load stock regime (best-effort, cache 24h)
            _sr_res = None
            try:
                from stock_regime import get_stock_regime
                _sr_res = await asyncio.to_thread(get_stock_regime, symbol)
            except Exception as _sre:
                logger.debug(f"stock regime lookup skip: {_sre}")

            summary = _format_rule_result(
                metrics          = result["metrics"],
                symbol           = symbol,
                entry_rule       = entry_rule,
                exit_rule        = exit_rule,
                days             = days,
                n_trades         = result["n_trades"],
                exit_ec_entry    = result["entry_ec"],
                exit_ec_exit     = result["exit_ec"],
                trades_sample    = result["trades"],
                n_buy_signals    = result["n_buy"],
                n_sell_signals   = result["n_sell"],
                bars             = result["bars"],
                trade_analytics  = _ta,
                wave_result      = _wave_res,
                stock_regime_result = _sr_res,
            )

            # Gửi metrics text
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=plain(summary)[:4096],
            )

            # Gửi overfit warning nếu user đã run nhiều lần
            if overfit_warning:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=plain(overfit_warning)[:500],
                )

            # Gửi chart
            chart_path = result.get("chart_path", "")
            if chart_path and pathlib.Path(chart_path).exists():
                m = result["metrics"]
                caption = (
                    f"Equity Curve: {symbol} | Rule Backtest | {days}D\n"
                    f"ENTRY: {entry_rule[:50]}\n"
                    f"EXIT : {exit_rule[:50]}\n"
                    f"Return: {m.get('total_return_pct',0):+.1f}% | "
                    f"Sharpe: {m.get('sharpe_ratio',0):.2f} | "
                    f"MaxDD: {m.get('max_drawdown_pct',0):.1f}%"
                )
                try:
                    with open(chart_path, "rb") as f:
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=f,
                            caption=plain(caption)[:1024],
                        )
                except Exception as pe:
                    logger.warning(f"Khong gui duoc chart: {pe}")
                finally:
                    pathlib.Path(chart_path).unlink(missing_ok=True)

            # Gửi 5 lệnh gần nhất
            trade_msg = _format_trades_msg(result["trades"])
            if trade_msg:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=plain(trade_msg)[:4096],
                )

        except Exception as e:
            import traceback
            logger.error(f"backtest_rule_cmd bg error: {e}\n{traceback.format_exc()}")
            err_text = f"Loi xu ly /backtest_rule: {str(e)[:300]}"
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=err_text,
                )
            except Exception:
                # edit fail (msg đã bị xoá hoặc timeout) → gửi message mới
                try:
                    await context.bot.send_message(chat_id=chat_id, text=err_text)
                except Exception:
                    pass

    # Chạy background — không await để bot tiếp tục nhận lệnh khác trong khi backtest chạy
    asyncio.create_task(_bg())
"""
PHẦN BỔ SUNG VÀO backtest_rule_cmd.py — TẦNG 1 HOÀN CHỈNH
=============================================================
Dán toàn bộ nội dung này vào CUỐI file backtest_rule_cmd.py.

Sau đó đăng ký trong bot.py:
    from backtest_rule_cmd import backtest_analog_cmd
    app.add_handler(CommandHandler("backtest_analog", backtest_analog_cmd))

LỆNH SỬ DỤNG:
    /backtest_analog HPG
    /backtest_analog VCB 1500

TẦNG 1: 15 combo × 7 ngưỡng = 105 experiments
  - 12 combo đơn (thiết kế theo ý nghĩa kinh tế)
  - 3 combo giao thoa (kết hợp 2 nhóm tín hiệu)
  Metrics: WR, MeanExp, MedianExp, MAE30, MaxDD, Sharpe, PF, n_signals, n_skip
"""

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_ANALOG_COOLDOWN = 300   # 5 phút per user
_last_backtest_analog: dict[str, float] = {}

# ── 15 combo (12 đơn + 3 giao thoa) ─────────────────────────────────────────
_ANALOG_COMBOS = [
    # ── 12 combo đơn ─────────────────────────────────────────────────────────
    {
        "name":       "Full Baseline",
        "group":      "baseline",
        "dims":       [
            "rsi_norm", "macd_hist_norm", "bb_position", "volume_spike",
            "trend_slope", "price_vs_sma20", "price_vs_sma50", "atr_ratio",
            "stoch_k_norm", "ema_cross", "momentum_5d", "momentum_20d",
            "high_low_pos", "vol_trend", "candle_body",
        ],
        "hypothesis": "Tat ca 15 chieu — dung lam baseline so sanh",
    },
    {
        "name":       "Pure Momentum",
        "group":      "momentum",
        "dims":       ["rsi_norm", "stoch_k_norm", "momentum_5d", "momentum_20d", "macd_hist_norm"],
        "hypothesis": "Chi dong luc ngan/trung han",
    },
    {
        "name":       "Trend Following",
        "group":      "trend",
        "dims":       ["trend_slope", "price_vs_sma20", "price_vs_sma50", "ema_cross", "momentum_20d"],
        "hypothesis": "Xu huong trung han",
    },
    {
        "name":       "Mean Reversion",
        "group":      "reversion",
        "dims":       ["bb_position", "high_low_pos", "rsi_norm", "stoch_k_norm", "price_vs_sma20"],
        "hypothesis": "Hoi phuc ve trung binh",
    },
    {
        "name":       "Volume Confirmed",
        "group":      "volume",
        "dims":       ["volume_spike", "vol_trend", "momentum_5d", "momentum_20d", "rsi_norm"],
        "hypothesis": "Breakout co volume xac nhan",
    },
    {
        "name":       "Oversold Bounce",
        "group":      "reversion",
        "dims":       ["rsi_norm", "stoch_k_norm", "bb_position", "high_low_pos", "momentum_5d"],
        "hypothesis": "Mua vung oversold",
    },
    {
        "name":       "Trend + Volume",
        "group":      "trend",
        "dims":       ["trend_slope", "price_vs_sma50", "ema_cross", "volume_spike", "vol_trend"],
        "hypothesis": "Xu huong + dong tien",
    },
    {
        "name":       "Volatility Aware",
        "group":      "volatility",
        "dims":       ["atr_ratio", "bb_position", "candle_body", "rsi_norm", "momentum_20d"],
        "hypothesis": "Dieu chinh theo bien dong",
    },
    {
        "name":       "No Volume",
        "group":      "momentum",
        "dims":       [
            "rsi_norm", "macd_hist_norm", "bb_position", "trend_slope",
            "price_vs_sma20", "price_vs_sma50", "stoch_k_norm", "momentum_20d",
        ],
        "hypothesis": "Bo qua volume — ky thuat thuan",
    },
    {
        "name":       "Momentum + Trend",
        "group":      "momentum",
        "dims":       ["momentum_5d", "momentum_20d", "trend_slope", "ema_cross", "price_vs_sma50"],
        "hypothesis": "Momentum trong xu huong",
    },
    {
        "name":       "Short-term Signal",
        "group":      "momentum",
        "dims":       ["rsi_norm", "momentum_5d", "volume_spike", "candle_body", "bb_position", "stoch_k_norm"],
        "hypothesis": "Tin hieu ngan han thuan",
    },
    {
        "name":       "Macro Trend",
        "group":      "trend",
        "dims":       ["trend_slope", "price_vs_sma50", "momentum_20d", "high_low_pos", "atr_ratio"],
        "hypothesis": "Buc tranh dai han",
    },
    # ── 3 combo giao thoa ────────────────────────────────────────────────────
    {
        "name":       "Oversold + Momentum",
        "group":      "crossover",
        "dims":       [
            "rsi_norm", "stoch_k_norm", "bb_position", "high_low_pos",
            "momentum_5d", "momentum_20d", "macd_hist_norm",
        ],
        "hypothesis": "Oversold NHUNG van co momentum tang — pullback trong uptrend",
    },
    {
        "name":       "Trend + Oversold",
        "group":      "crossover",
        "dims":       [
            "trend_slope", "price_vs_sma50", "ema_cross",
            "rsi_norm", "stoch_k_norm", "bb_position",
        ],
        "hypothesis": "Xu huong tang, dang pullback oversold — diem mua tot nhat",
    },
    {
        "name":       "Momentum + Volume",
        "group":      "crossover",
        "dims":       [
            "momentum_5d", "momentum_20d", "macd_hist_norm",
            "volume_spike", "vol_trend",
        ],
        "hypothesis": "Momentum co volume xac nhan — breakout chat luong cao",
    },
]

_ANALOG_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
# Tổng: 15 × 7 = 105 experiments

# ── Hard filter cho Backtest C ────────────────────────────────────────────────
# Format: {dim: (min, max)} — vector value phải trong khoảng [min, max]
# Ngưỡng dựa trên ý nghĩa kinh tế:
#   rsi_norm < -0.30    → RSI < 35
#   trend_slope > 0.20  → SMA20 > SMA50 + 2%
#   volume_spike > 0.25 → vol > 150% MA20
#   atr_ratio > 0.50    → ATR > 2.5% giá
#   momentum_5d < -0.20 → giảm > 2% trong 5 ngày
_COMBO_HARD_FILTERS = {
    "Full Baseline":       {},
    "Mean Reversion":      {"rsi_norm": (-1.0,-0.30), "bb_position": (0.0,0.25)},
    "Oversold Bounce":     {"rsi_norm": (-1.0,-0.30), "stoch_k_norm": (-1.0,-0.30)},
    "Oversold + Momentum": {"rsi_norm": (-1.0,-0.30), "stoch_k_norm": (-1.0,-0.30)},
    "Trend + Oversold":    {"trend_slope": (0.20,1.0), "rsi_norm": (-1.0,-0.30)},
    "Trend Following":     {"trend_slope": (0.20,1.0), "price_vs_sma50": (0.05,1.0)},
    "Trend + Volume":      {"trend_slope": (0.20,1.0), "volume_spike": (0.25,1.0)},
    "Momentum + Trend":    {"trend_slope": (0.20,1.0), "momentum_20d": (0.10,1.0)},
    "Macro Trend":         {"trend_slope": (0.20,1.0), "price_vs_sma50": (0.05,1.0)},
    "Volume Confirmed":    {"volume_spike": (0.25,1.0)},
    "Momentum + Volume":   {"volume_spike": (0.25,1.0), "momentum_5d": (-1.0,-0.20)},
    "Short-term Signal":   {"momentum_5d": (-1.0,-0.20), "volume_spike": (0.25,1.0)},
    "Volatility Aware":    {"atr_ratio": (0.50,1.0), "bb_position": (0.0,0.25)},
    "Pure Momentum":       {"momentum_5d": (-1.0,-0.20), "momentum_20d": (-1.0,-0.10)},
    "No Volume":           {"rsi_norm": (-1.0,-0.30), "bb_position": (0.0,0.25)},
}


def _check_hard_filter(vec: dict, combo_name: str) -> bool:
    """Kiểm tra ngày T có thỏa hard filter của combo không."""
    filters = _COMBO_HARD_FILTERS.get(combo_name)
    if not filters:
        return True
    for dim, (lo, hi) in filters.items():
        val = vec.get(dim, 0.0)
        if not (lo <= val <= hi):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# CORE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_experiment(
    combo, threshold, vectors, vector_indices, dates,
    close_arr, low_arr, n_bars,
    min_samples=5, use_hard_filter=False,
):
    """Chạy 1 experiment dùng chung cho A/B/C."""
    import numpy as np
    import pandas as pd

    WIN_THRESH = 1.0
    MDS_DAYS   = 30
    FWD_DAYS   = 30

    combo_name = combo["name"]
    dims       = combo["dims"]
    hypothesis = combo["hypothesis"]
    group      = combo.get("group", "")
    signals    = []
    n_skip     = 0

    for t_idx in range(120, n_bars - FWD_DAYS - 1, 7):
        if t_idx not in vectors:
            continue
        target_vec = vectors[t_idx]
        if use_hard_filter and not _check_hard_filter(target_vec, combo_name):
            n_skip += 1
            continue
        target_arr = np.array([target_vec.get(d, 0.0) for d in dims], dtype=float)
        t_norm     = np.linalg.norm(target_arr)
        if t_norm < 1e-9:
            continue
        exclude_cutoff = t_idx - 90
        candidates     = [i for i in vector_indices if i < exclude_cutoff]
        if len(candidates) < 10:
            continue
        sim_list = []
        for c_idx in candidates:
            c_vec  = vectors[c_idx]
            c_arr  = np.array([c_vec.get(d, 0.0) for d in dims], dtype=float)
            c_norm = np.linalg.norm(c_arr)
            if c_norm < 1e-9:
                continue
            sim = float(np.dot(target_arr, c_arr) / (t_norm * c_norm))
            if sim >= threshold:
                sim_list.append((c_idx, sim))
        if not sim_list:
            continue
        sim_list.sort(key=lambda x: -x[1])
        kept = []
        for c_idx, sim in sim_list:
            c_date    = pd.Timestamp(dates[c_idx])
            too_close = any(
                abs((c_date - pd.Timestamp(dates[k])).days) < MDS_DAYS
                for k, _ in kept
            )
            if not too_close:
                kept.append((c_idx, sim))
        if len(kept) < min_samples:
            n_skip += 1
            continue
        fwd_rets = []
        mae_vals = []
        for c_idx, _ in kept:
            fwd_idx = c_idx + FWD_DAYS
            if fwd_idx >= n_bars:
                continue
            entry = close_arr[c_idx]
            fwd_rets.append((close_arr[fwd_idx] - entry) / entry * 100)
            win_low = np.min(low_arr[c_idx + 1: fwd_idx + 1]) if fwd_idx > c_idx else entry
            mae_vals.append((win_low - entry) / entry * 100)
        if len(fwd_rets) < min_samples:
            n_skip += 1
            continue
        signals.append({"fwd_rets": fwd_rets, "mae_vals": mae_vals})

    if len(signals) < 5:
        return {
            "combo": combo_name, "group": group, "hypothesis": hypothesis,
            "threshold": threshold, "n_signals": len(signals), "n_skip": n_skip,
            "skip": True,
        }

    sig_rets = [float(np.median(s["fwd_rets"])) for s in signals]
    sig_maes = [float(np.median(s["mae_vals"])) for s in signals]
    wins     = [x for x in sig_rets if x >= WIN_THRESH]
    losses   = [x for x in sig_rets if x < WIN_THRESH]
    wr       = len(wins) / len(sig_rets)
    mean_exp = float(np.mean(sig_rets))
    med_exp  = float(np.median(sig_rets))
    std_ret  = float(np.std(sig_rets)) if len(sig_rets) > 1 else 1e-9
    sharpe   = mean_exp / std_ret * (52 ** 0.5) if std_ret > 0 else 0.0
    pos_sum  = sum(wins)
    neg_sum  = abs(sum(losses)) if losses else 1e-9
    pf       = pos_sum / neg_sum if neg_sum > 0 else 99.0
    mae30    = float(np.median(sig_maes))
    max_dd   = float(np.min(sig_rets))

    return {
        "combo": combo_name, "group": group, "hypothesis": hypothesis,
        "threshold": threshold, "n_signals": len(signals), "n_skip": n_skip,
        "wr": round(wr * 100, 1), "mean_exp": round(mean_exp, 2),
        "med_exp": round(med_exp, 2), "mae30": round(mae30, 2),
        "max_dd": round(max_dd, 1), "sharpe": round(sharpe, 3),
        "pf": round(pf, 2), "skip": False,
    }


def _run_analog_backtest_variants_sync(symbol: str, days: int = 1800) -> dict:
    """
    Chạy 3 backtest song song từ cùng 1 bộ data:
      A: Gốc       — MIN_SAMPLES=5, không filter
      B: MinS=10   — MIN_SAMPLES=10, không filter
      C: HardFlt   — MIN_SAMPLES=5, + hard filter theo combo
    """
    import numpy as np

    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=days, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data {symbol}: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu (can >= 200 bars)"}

    if "low" not in df.columns:
        df["low"] = df["close"]

    n_bars    = len(df)
    dates     = df["date"].values
    close_arr = df["close"].values.astype(float)
    low_arr   = df["low"].values.astype(float)

    try:
        from state_vector import compute_state_vector_for_date
    except ImportError as e:
        return {"status": "error", "error": f"Thieu state_vector: {e}"}

    vectors = {}
    for i in range(59, n_bars):
        vec = compute_state_vector_for_date(df, i)
        if vec is not None:
            vectors[i] = vec

    if len(vectors) < 100:
        return {"status": "error", "error": f"Chi co {len(vectors)} vectors"}

    vector_indices = sorted(vectors.keys())
    common_args    = (vectors, vector_indices, dates, close_arr, low_arr, n_bars)

    variants = {
        "A": {"min_samples": 5,  "use_hard_filter": False},
        "B": {"min_samples": 10, "use_hard_filter": False},
        "C": {"min_samples": 5,  "use_hard_filter": True},
    }

    all_variant_results = {}
    for tag, cfg in variants.items():
        results = []
        for combo in _ANALOG_COMBOS:
            for threshold in _ANALOG_THRESHOLDS:
                r = _run_one_experiment(
                    combo, threshold, *common_args,
                    min_samples=cfg["min_samples"],
                    use_hard_filter=cfg["use_hard_filter"],
                )
                results.append(r)

        valid = [r for r in results if not r.get("skip")]

        def _rank_key(r):
            return (
                int(r["mean_exp"] > 0 and r["pf"] >= 1.5),
                r["mean_exp"], r["pf"], r["sharpe"],
            )
        valid.sort(key=_rank_key, reverse=True)

        baseline = next(
            (r for r in results
             if r["combo"] == "Full Baseline"
             and abs(r["threshold"] - 0.70) < 0.001
             and not r.get("skip")),
            None,
        )
        all_variant_results[tag] = {
            "results": results, "valid": valid,
            "top": valid[:5], "baseline": baseline,
        }

    return {
        "status": "ok", "symbol": symbol,
        "n_bars": n_bars, "n_vectors": len(vectors),
        "variants": all_variant_results,
    }


def _format_analog_compare_result(res: dict) -> str:
    """Formatter so sánh 3 backtest A/B/C cạnh nhau."""
    symbol   = res["symbol"]
    n_bars   = res["n_bars"]
    n_vecs   = res["n_vectors"]
    variants = res["variants"]
    sep  = "=" * 36
    sep2 = "-" * 36

    tag_labels = {
        "A": "A (Goc)    ",
        "B": "B (MinS=10)",
        "C": "C (HardFlt)",
    }

    lines = [
        f"BACKTEST ANALOG COMPARE — {symbol}",
        f"{n_bars} bars | {n_vecs} vectors",
        sep,
        "PHUONG AN:",
        "  A (Goc)    : MIN_SAMPLES=5, khong filter",
        "  B (MinS=10): MIN_SAMPLES=10, khong filter",
        "  C (HardFlt): MIN_SAMPLES=5 + hard filter combo",
        sep,
        "TOP 1 MOI PHUONG AN:",
        "",
    ]

    best_results = {}
    for tag in ["A", "B", "C"]:
        v   = variants[tag]
        top = v["top"]
        if not top:
            lines.append(f"  [{tag_labels[tag]}] Khong co ket qua hop le.")
            continue
        best = top[0]
        best_results[tag] = best
        pass_filter = best["mean_exp"] > 0 and best["pf"] >= 1.5
        em = "OK" if pass_filter else "!!"
        lines.append(f"[{em}] [{tag_labels[tag]}]")
        lines.append(f"   {best['combo']} | nguong {best['threshold']:.2f}")
        lines.append(
            f"   WR {best['wr']:.0f}%  Exp {best['mean_exp']:+.2f}%  "
            f"PF {best['pf']:.2f}  n={best['n_signals']}"
        )
        lines.append(
            f"   MAE30 {best['mae30']:.1f}%  Worst {best['max_dd']:.1f}%  "
            f"Sharpe {best['sharpe']:.2f}"
        )
        lines.append("")

    lines.append(sep2)

    # Bảng so sánh trực tiếp
    combos = [best_results[t]["combo"] for t in ["A","B","C"] if t in best_results]
    if len(set(combos)) == 1:
        combo_name = combos[0]
        lines.append(f"BANG SO SANH — {combo_name}:")
        lines.append(f"  {'':12} {'n':>5} {'WR':>5} {'Exp':>7} {'PF':>6} {'MAE30':>7} {'Worst':>7}")
        lines.append(f"  {'-'*48}")
        for tag in ["A","B","C"]:
            if tag not in best_results:
                continue
            r = best_results[tag]
            lines.append(
                f"  {tag_labels[tag]} "
                f"{r['n_signals']:>5} {r['wr']:>4.0f}% "
                f"{r['mean_exp']:>+6.2f}% {r['pf']:>6.2f} "
                f"{r['mae30']:>6.1f}% {r['max_dd']:>6.1f}%"
            )
    else:
        lines.append("BANG SO SANH (combo khac nhau):")
        lines.append(f"  {'':12} {'Combo':25} {'Exp':>7} {'PF':>6} {'n':>5}")
        lines.append(f"  {'-'*54}")
        for tag in ["A","B","C"]:
            if tag not in best_results:
                continue
            r = best_results[tag]
            lines.append(
                f"  {tag_labels[tag]} {r['combo']:<25} "
                f"{r['mean_exp']:>+6.2f}% {r['pf']:>6.2f} {r['n_signals']:>5}"
            )

    lines.append("")
    lines.append(sep2)
    lines.append("NHAN XET:")

    if len(best_results) == 3:
        exp_a = best_results["A"]["mean_exp"]
        exp_b = best_results["B"]["mean_exp"]
        exp_c = best_results["C"]["mean_exp"]
        n_a   = best_results["A"]["n_signals"]
        n_b   = best_results["B"]["n_signals"]
        n_c   = best_results["C"]["n_signals"]

        if exp_b >= exp_a:
            lines.append(f"  B tot hon A: Exp {exp_b:+.2f}% vs {exp_a:+.2f}%, n: {n_a}->{n_b}")
        elif exp_b >= exp_a * 0.90:
            lines.append(f"  B gan bang A (Exp {exp_b:+.2f}% vs {exp_a:+.2f}%), n giam {n_a-n_b}")
        else:
            lines.append(f"  B kem hon A: Exp {exp_b:+.2f}% vs {exp_a:+.2f}%, giu A")

        if exp_c >= exp_a:
            lines.append(f"  C tot hon A: Exp {exp_c:+.2f}% vs {exp_a:+.2f}%, n: {n_a}->{n_c}")
        elif exp_c >= exp_a * 0.90:
            lines.append(f"  C gan bang A, n giam manh ({n_a}->{n_c}), hard filter khong them nhieu gia tri")
        else:
            lines.append(f"  C kem hon A: hard filter loai qua nhieu tin hieu tot, giu A")

        # Bộ lọc trước khi chọn best: n >= 20 VÀ PF <= 500
        MIN_N_COMPARE  = 20
        MAX_PF_COMPARE = 500
        valid_tags = [
            t for t in ["A", "B", "C"]
            if t in best_results
            and best_results[t]["n_signals"] >= MIN_N_COMPARE
            and best_results[t]["pf"] <= MAX_PF_COMPARE
        ]
        for t in ["A", "B", "C"]:
            if t not in best_results:
                continue
            r = best_results[t]
            if r["n_signals"] < MIN_N_COMPARE:
                lines.append(
                    f"  ⚠️  {tag_labels[t]} bi loai: "
                    f"n={r['n_signals']} < {MIN_N_COMPARE} (khong du y nghia thong ke)"
                )
            elif r["pf"] > MAX_PF_COMPARE:
                lines.append(
                    f"  ⚠️  {tag_labels[t]} bi loai: "
                    f"PF={r['pf']:.0f} > {MAX_PF_COMPARE} (dau hieu overfitting)"
                )
        if not valid_tags:
            lines.append(
                "  CANH BAO: Khong co phuong an nao hop le (n >= 20 va PF <= 500)."
            )
        else:
            best_tag = max(valid_tags,
                key=lambda t: (
                    int(best_results[t]["mean_exp"] > 0 and best_results[t]["pf"] >= 1.5),
                    best_results[t]["mean_exp"], best_results[t]["pf"],
                )
            )
            lines.append(f"  => Khuyen nghi: {tag_labels[best_tag]} — "
                         f"Exp {best_results[best_tag]['mean_exp']:+.2f}% "
                         f"PF {best_results[best_tag]['pf']:.2f} "
                         f"n={best_results[best_tag]['n_signals']}")

    lines += [
        "",
        "Day la backtest IN-SAMPLE. Dung /walkforward_analog de validate OOS.",
        sep,
    ]
    return "\n".join(lines)



def _run_analog_backtest_sync(symbol: str, days: int = 1800) -> dict:
    """
    Fetch data → tính vectors → walk-forward 105 experiments.
    Chạy trong asyncio.to_thread.
    """
    import numpy as np
    import pandas as pd
    from collections import Counter

    # 1. Fetch data ────────────────────────────────────────────────────────────
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=days, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data {symbol}: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu (can >= 200 bars)"}

    n_bars    = len(df)
    dates     = df["date"].values
    close_arr = df["close"].values.astype(float)
    low_arr   = df["low"].values.astype(float)

    # 2. Tính vectors ──────────────────────────────────────────────────────────
    try:
        from state_vector import compute_state_vector_for_date
    except ImportError as e:
        return {"status": "error", "error": f"Thieu state_vector: {e}"}

    vectors = {}
    for i in range(59, n_bars):
        vec = compute_state_vector_for_date(df, i)
        if vec is not None:
            vectors[i] = vec

    if len(vectors) < 100:
        return {"status": "error", "error": f"Chi co {len(vectors)} vectors, can >= 100"}

    vector_indices = sorted(vectors.keys())

    # 3. Walk-forward ──────────────────────────────────────────────────────────
    WIN_THRESH  = 1.0
    MIN_SAMPLES = 5
    MDS_DAYS    = 30
    FWD_DAYS    = 30

    results = []

    for combo in _ANALOG_COMBOS:
        combo_name = combo["name"]
        dims       = combo["dims"]
        hypothesis = combo["hypothesis"]
        group      = combo.get("group", "")

        for threshold in _ANALOG_THRESHOLDS:

            signals = []
            n_skip  = 0

            for t_idx in range(120, n_bars - FWD_DAYS - 1, 7):
                if t_idx not in vectors:
                    continue

                target_vec = vectors[t_idx]
                target_arr = np.array([target_vec.get(d, 0.0) for d in dims], dtype=float)
                t_norm     = np.linalg.norm(target_arr)
                if t_norm < 1e-9:
                    continue

                # Chỉ dùng data trước t_idx - 90 ngày
                exclude_cutoff = t_idx - 90
                candidates     = [i for i in vector_indices if i < exclude_cutoff]
                if len(candidates) < 10:
                    continue

                # Cosine similarity trên dims được chọn
                sim_list = []
                for c_idx in candidates:
                    c_vec  = vectors[c_idx]
                    c_arr  = np.array([c_vec.get(d, 0.0) for d in dims], dtype=float)
                    c_norm = np.linalg.norm(c_arr)
                    if c_norm < 1e-9:
                        continue
                    sim = float(np.dot(target_arr, c_arr) / (t_norm * c_norm))
                    if sim >= threshold:
                        sim_list.append((c_idx, sim))

                if not sim_list:
                    continue

                # Minimum Distance Sampling 30D
                sim_list.sort(key=lambda x: -x[1])
                kept = []
                for c_idx, sim in sim_list:
                    c_date    = pd.Timestamp(dates[c_idx])
                    too_close = any(
                        abs((c_date - pd.Timestamp(dates[k])).days) < MDS_DAYS
                        for k, _ in kept
                    )
                    if not too_close:
                        kept.append((c_idx, sim))

                if len(kept) < MIN_SAMPLES:
                    n_skip += 1
                    continue

                # Forward return 30D và MAE 30D cho từng analog
                fwd_rets = []
                mae_vals = []
                for c_idx, _ in kept:
                    fwd_idx = c_idx + FWD_DAYS
                    if fwd_idx >= n_bars:
                        continue
                    entry = close_arr[c_idx]
                    fwd_rets.append((close_arr[fwd_idx] - entry) / entry * 100)
                    # MAE: mức giảm tệ nhất trong window 30D
                    win_low = np.min(low_arr[c_idx + 1: fwd_idx + 1]) if fwd_idx > c_idx else entry
                    mae_vals.append((win_low - entry) / entry * 100)

                if len(fwd_rets) < MIN_SAMPLES:
                    n_skip += 1
                    continue

                signals.append({
                    "fwd_rets": fwd_rets,
                    "mae_vals": mae_vals,
                })

            # Tính metrics ─────────────────────────────────────────────────────
            if len(signals) < 5:
                results.append({
                    "combo": combo_name, "group": group,
                    "hypothesis": hypothesis, "threshold": threshold,
                    "n_signals": len(signals), "n_skip": n_skip,
                    "skip": True,
                })
                continue

            # Đại diện mỗi tín hiệu bằng median của analog samples
            sig_rets = [float(np.median(s["fwd_rets"])) for s in signals]
            sig_maes = [float(np.median(s["mae_vals"])) for s in signals]

            wins     = [x for x in sig_rets if x >= WIN_THRESH]
            losses   = [x for x in sig_rets if x < WIN_THRESH]
            wr       = len(wins) / len(sig_rets)
            mean_exp = float(np.mean(sig_rets))
            med_exp  = float(np.median(sig_rets))
            std_ret  = float(np.std(sig_rets)) if len(sig_rets) > 1 else 1e-9
            sharpe   = mean_exp / std_ret * (52 ** 0.5) if std_ret > 0 else 0.0
            pos_sum  = sum(wins)
            neg_sum  = abs(sum(losses)) if losses else 1e-9
            pf       = pos_sum / neg_sum if neg_sum > 0 else 99.0
            mae30    = float(np.median(sig_maes))

            # Worst Signal = return tệ nhất của 1 tín hiệu đơn lẻ
            # Đây là metric rủi ro đúng cho hệ thống analog:
            # Mỗi signal là 1 bet độc lập (overlap 4-5 signals đồng thời)
            # → equity curve tổng không có ý nghĩa, rủi ro thực = tệ nhất 1 lệnh
            max_dd   = float(np.min(sig_rets))

            results.append({
                "combo":     combo_name,
                "group":     group,
                "hypothesis":hypothesis,
                "threshold": threshold,
                "n_signals": len(signals),
                "n_skip":    n_skip,
                "wr":        round(wr * 100, 1),
                "mean_exp":  round(mean_exp, 2),
                "med_exp":   round(med_exp, 2),
                "mae30":     round(mae30, 2),
                "max_dd":    round(max_dd, 1),
                "sharpe":    round(sharpe, 3),
                "pf":        round(pf, 2),
                "skip":      False,
            })

    # 4. Xếp hạng ──────────────────────────────────────────────────────────────
    valid = [r for r in results if not r.get("skip")]
    # Xếp hạng: Exp > 0 và PF >= 1.5 trước, sau đó sort theo Exp DESC, PF tiebreak
    # Sharpe chỉ dùng tiebreak cuối cùng
    def _rank_key(r):
        exp_ok = r["mean_exp"] > 0
        pf_ok  = r["pf"] >= 1.5
        return (
            int(exp_ok and pf_ok),   # pass bộ lọc cứng lên trước
            r["mean_exp"],           # Exp cao hơn → tốt hơn
            r["pf"],                 # PF tiebreak
            r["sharpe"],             # Sharpe tiebreak cuối
        )
    valid.sort(key=_rank_key, reverse=True)

    baseline = next(
        (r for r in results
         if r["combo"] == "Full Baseline"
         and abs(r["threshold"] - 0.70) < 0.001
         and not r.get("skip")),
        None,
    )

    return {
        "status":        "ok",
        "symbol":        symbol,
        "n_bars":        n_bars,
        "n_vectors":     len(vectors),
        "n_experiments": len(results),
        "n_valid":       len(valid),
        "top_results":   valid[:10],
        "baseline":      baseline,
        "all_results":   results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def _format_analog_backtest_result(res: dict) -> str:
    import numpy as np
    from collections import Counter

    symbol   = res["symbol"]
    n_bars   = res["n_bars"]
    n_vecs   = res["n_vectors"]
    n_exp    = res["n_experiments"]
    n_valid  = res["n_valid"]
    top      = res.get("top_results", [])
    baseline = res.get("baseline")
    all_res  = res.get("all_results", [])

    sep  = "=" * 32
    sep2 = "-" * 32

    lines = [
        f"BACKTEST ANALOG T1 — {symbol}",
        f"{n_bars} bars | {n_vecs} vectors | {n_exp} exp ({n_valid} hop le)",
        sep,
    ]

    # Baseline
    lines.append("BASELINE (15 chieu | nguong 0.70):")
    if baseline:
        lines.append(
            f"  WR {baseline['wr']:.0f}%  "
            f"Exp {baseline['mean_exp']:+.2f}% (med {baseline['med_exp']:+.2f}%)  "
            f"Sharpe {baseline['sharpe']:.2f}"
        )
        lines.append(
            f"  MAE30 {baseline['mae30']:.1f}%  "
            f"Worst {baseline['max_dd']:.1f}%  "
            f"PF {baseline['pf']:.2f}  "
            f"n={baseline['n_signals']}"
        )
    else:
        lines.append("  Khong du tin hieu.")
    lines.append(sep2)

    # Top 5
    lines.append(f"TOP 5 / {n_valid} experiments:")
    lines.append("")

    if not top:
        lines.append("  Khong co experiment hop le (>= 5 tin hieu).")
    else:
        bl_sharpe = baseline["sharpe"] if baseline else 0.0
        for rank, r in enumerate(top[:5], 1):
            diff    = r["sharpe"] - bl_sharpe
            sign    = "+" if diff >= 0 else ""
            vs_base = f" [{sign}{diff:.2f} vs BL]"

            em = "✅" if r["wr"] >= 60 and r["mean_exp"] > 0 else \
                 "🟡" if r["mean_exp"] > 0 else "🔴"

            lines.append(f"{em} #{rank} {r['combo']} | nguong {r['threshold']:.2f}{vs_base}")
            lines.append(
                f"   WR {r['wr']:.0f}%  "
                f"Exp {r['mean_exp']:+.2f}% (med {r['med_exp']:+.2f}%)  "
                f"Sharpe {r['sharpe']:.2f}"
            )
            lines.append(
                f"   MAE30 {r['mae30']:.1f}%  "
                f"Worst {r['max_dd']:.1f}%  "
                f"PF {r['pf']:.2f}  "
                f"n={r['n_signals']} skip={r['n_skip']}"
            )
            lines.append(f"   >> {r['hypothesis']}")
            lines.append("")

    lines.append(sep2)

    # Phân tích combo giao thoa — sort theo Exp+PF
    cross_res = sorted(
        [r for r in all_res if r.get("group") == "crossover" and not r.get("skip")
         and r["mean_exp"] > 0],
        key=lambda x: (x["mean_exp"], x["pf"]), reverse=True,
    )
    if cross_res:
        bc      = cross_res[0]
        bl_exp  = baseline["mean_exp"] if baseline else 0.0
        diff    = bc["mean_exp"] - bl_exp
        verdict = "tot hon" if diff > 0 else "kem hon"
        lines.append("COMBO GIAO THOA (tot nhat theo Exp+PF):")
        lines.append(
            f"  {bc['combo']} | nguong {bc['threshold']:.2f} | "
            f"Exp {bc['mean_exp']:+.2f}%  PF {bc['pf']:.2f}  WR {bc['wr']:.0f}%"
        )
        lines.append(f"  → {verdict} baseline {abs(diff):.2f}% Exp")
        lines.append(sep2)

    # Nhận xét tự động
    lines.append("NHAN XET:")
    lines.append("  Xep hang: Exp (chinh) → PF (tiebreak). WR chi tham khao tam ly.")
    if top and baseline:
        best    = top[0]
        bl_exp  = baseline["mean_exp"] if baseline else 0.0
        exp_diff = best["mean_exp"] - bl_exp

        # Pass bộ lọc cứng không?
        if best["mean_exp"] > 0 and best["pf"] >= 1.5:
            if exp_diff > 0.5:
                lines.append(
                    f"  Chon loc chieu CO LOI ro rang: {best['combo']} "
                    f"Exp {best['mean_exp']:+.2f}% (baseline {bl_exp:+.2f}%), "
                    f"PF {best['pf']:.2f}."
                )
            elif exp_diff > 0:
                lines.append(
                    f"  Chon loc chieu loi nhat hoc: Exp chi tot hon baseline "
                    f"{exp_diff:+.2f}%. Full 15 chieu van chap nhan duoc."
                )
            else:
                lines.append(
                    f"  Full 15 chieu co Exp tot hon. Chon loc chieu KHONG cai thien Exp."
                )
        else:
            lines.append(
                f"  CANH BAO: Top combo chua pass bo loc cung "
                f"(Exp {best['mean_exp']:+.2f}%, PF {best['pf']:.2f}). "
                f"Ket qua nay khong nen dung cho live trading."
            )

        # WR note — chỉ nêu nếu WR thấp để cảnh báo tâm lý
        if best["wr"] < 50:
            lines.append(
                f"  Luu y tam ly: WR {best['wr']:.0f}% — "
                f"thuong thua nhieu hon thang, can ky luat cao."
            )

        avg_n = sum(r["n_signals"] for r in top[:5]) / max(len(top[:5]), 1)
        if avg_n < 15:
            lines.append(f"  CANH BAO: TB {avg_n:.0f} tin hieu/exp — thieu y nghia thong ke.")
        else:
            lines.append(f"  So tin hieu TB: {avg_n:.0f} — du de tham khao.")

        thresh_top5    = [r["threshold"] for r in top[:5]]
        best_thresh    = Counter(thresh_top5).most_common(1)[0][0]
        lines.append(f"  Nguong pho bien trong top 5: {best_thresh:.2f}")

        groups_top5    = [r["group"] for r in top[:5]]
        dominant_group = Counter(groups_top5).most_common(1)[0][0]
        group_labels   = {
            "momentum":  "Momentum", "trend": "Trend Following",
            "reversion": "Mean Reversion", "volume": "Volume",
            "volatility":"Volatility", "crossover": "Combo giao thoa",
            "baseline":  "Baseline",
        }
        lines.append(f"  Nhom thong tri top 5: {group_labels.get(dominant_group, dominant_group)}")
    else:
        lines.append("  Khong du du lieu de nhan xet.")

    lines.append("")
    lines.append("Day la backtest IN-SAMPLE. Nen dung ket qua nay de dinh huong,")
    lines.append("khong phai de ket luan cuoi cung. Tang 2 se verify overfitting.")
    lines.append(sep)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def backtest_analog_cmd(update, context):
    """
    /backtest_analog <MA> [days] [compare]
    Them "compare" de so sanh 3 phuong an A/B/C.
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []

    if not args:
        await update.message.reply_text(
            "Cu phap: /backtest_analog <MA> [days] [compare]\n\n"
            "Vi du:\n"
            "  /backtest_analog HPG\n"
            "  /backtest_analog MWG compare\n"
            "  /backtest_analog VCB 1500 compare\n\n"
            "compare: so sanh 3 phuong an A/B/C (~3x lau hon)\n"
            "Thoi gian: 4-10 phut (goc) | 12-30 phut (compare)."
        )
        return

    import re as _re

    do_compare = "compare" in [a.lower() for a in args]
    raw_args   = [a for a in args if a.lower() != "compare"]

    symbol = raw_args[0].upper().strip() if raw_args else ""
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol):
        await update.message.reply_text(f"Ma '{symbol}' khong hop le.")
        return

    days = 1800
    if len(raw_args) > 1:
        try:
            days = max(500, min(int(raw_args[1]), 2500))
        except ValueError:
            days = 1800

    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_backtest_analog.get(user_id, 0)
    if since < BACKTEST_ANALOG_COOLDOWN:
        wait = int(BACKTEST_ANALOG_COOLDOWN - since)
        await update.message.reply_text(f"Vui long cho {wait}s truoc khi chay tiep.")
        return
    _last_backtest_analog[user_id] = time.time()

    chat_id  = update.effective_chat.id
    est_time = "12-30 phut" if do_compare else "4-10 phut"
    mode_str = "so sanh 3 phuong an A/B/C" if do_compare else "105 experiments"
    msg = await update.message.reply_text(
        f"Backtest Analog T1: {symbol} ({days} ngay)\n"
        f"Mode: {mode_str}\n"
        f"Uoc tinh {est_time}, vui long doi..."
    )

    async def _bg():
        try:
            if do_compare:
                result = await asyncio.to_thread(
                    _run_analog_backtest_variants_sync, symbol, days
                )
                if result["status"] == "error":
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg.message_id,
                        text=f"Loi: {result.get('error','?')[:300]}",
                    )
                    return
                summary = _format_analog_compare_result(result)
            else:
                result = await asyncio.to_thread(_run_analog_backtest_sync, symbol, days)
                if result["status"] == "error":
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg.message_id,
                        text=f"Loi: {result.get('error','?')[:300]}",
                    )
                    return
                summary = _format_analog_backtest_result(result)

            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=_plain(summary)[:4096],
            )

        except Exception as e:
            import traceback
            logger.error(f"backtest_analog_cmd error: {e}\n{traceback.format_exc()}")
            err_text = f"Loi /backtest_analog {symbol}: {str(e)[:200]}"
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id, text=err_text,
                )
            except Exception:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=err_text)
                except Exception:
                    pass

    asyncio.create_task(_bg())

async def backtest_analog_batch_cmd(update, context):
    """
    /backtest_analog_batch HPG VCB STB CTG MWG FPT TCB [days]

    Chay backtest analog cho nhieu ma mot luot.
    Ket qua: bang so sanh + combo nhat quan + khuyen nghi walk-forward.

    Vi du:
      /backtest_analog_batch HPG VCB STB
      /backtest_analog_batch HPG VCB STB CTG MWG FPT TCB 1500
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []

    if not args:
        await update.message.reply_text(
            "Cu phap: /backtest_analog_batch MA1 MA2 MA3... [days]\n\n"
            "Vi du:\n"
            "  /backtest_analog_batch HPG VCB STB\n"
            "  /backtest_analog_batch HPG VCB STB CTG MWG FPT TCB\n\n"
            f"Toi da {BATCH_MAX_SYMBOLS} ma mot lan.\n"
            "Thoi gian: ~5-10 phut/ma. 7 ma ~ 35-70 phut.\n"
            "Ket qua: bang so sanh + combo nhat quan + khuyen nghi walk-forward."
        )
        return

    # Parse symbols và days
    import re as _re
    symbols = []
    days    = 1800
    for arg in args:
        if _re.match(r'^\d+$', arg):
            days = max(500, min(int(arg), 2500))
        elif _re.match(r'^[A-Z0-9]{2,10}$', arg.upper()):
            symbols.append(arg.upper())

    if not symbols:
        await update.message.reply_text("Khong tim thay ma hop le trong lenh.")
        return

    if len(symbols) > BATCH_MAX_SYMBOLS:
        await update.message.reply_text(
            f"Toi da {BATCH_MAX_SYMBOLS} ma mot lan. "
            f"Ban nhap {len(symbols)} ma — vui long chia thanh nhieu lenh."
        )
        return

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_backtest_analog_batch.get(user_id, 0)
    if since < BACKTEST_ANALOG_BATCH_COOLDOWN:
        wait = int(BACKTEST_ANALOG_BATCH_COOLDOWN - since)
        await update.message.reply_text(
            f"Vui long cho {wait}s truoc khi /backtest_analog_batch tiep."
        )
        return
    _last_backtest_analog_batch[user_id] = time.time()

    chat_id  = update.effective_chat.id
    est_mins = len(symbols) * 7   # ước tính ~7 phút/mã

    msg = await update.message.reply_text(
        f"Backtest Analog Batch: {len(symbols)} ma\n"
        f"Ma: {', '.join(symbols)}\n"
        f"105 experiments/ma | {days} ngay lich su\n"
        f"Uoc tinh ~{est_mins} phut. Vui long doi..."
    )

    async def _bg():
        try:
            # Chạy từng mã, cập nhật progress
            results = {}
            for i, symbol in enumerate(symbols, 1):
                # Cập nhật progress message
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg.message_id,
                        text=(
                            f"Backtest Analog Batch: {len(symbols)} ma\n"
                            f"Dang chay: {symbol} ({i}/{len(symbols)})\n"
                            f"Hoan thanh: {', '.join(symbols[:i-1]) if i > 1 else 'chua co'}\n"
                            f"Con lai: ~{(len(symbols)-i)*7} phut..."
                        ),
                    )
                except Exception:
                    pass

                # Chạy backtest cho symbol này
                try:
                    r = await asyncio.to_thread(_run_analog_backtest_sync, symbol, days)
                    results[symbol] = r
                except Exception as e:
                    results[symbol] = {"status": "error", "error": str(e)[:120]}

            # Format và gửi kết quả
            summary = _format_batch_result(results)

            # Nếu dài hơn 4096 → tách 2 message
            if len(summary) <= 4096:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    text=_plain(summary),
                )
            else:
                # Part 1: edit message loading
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    text=_plain(summary[:4000] + "\n...(tiep theo)"),
                )
                # Part 2: send message mới
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=_plain(summary[4000:])[:4096],
                )

        except Exception as e:
            import traceback
            logger.error(f"backtest_analog_batch_cmd error: {e}\n{traceback.format_exc()}")
            err_text = f"Loi /backtest_analog_batch: {str(e)[:200]}"
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id, text=err_text,
                )
            except Exception:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=err_text)
                except Exception:
                    pass

    asyncio.create_task(_bg())


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ANALOG DETAIL — full threshold table cho 1 mã + 1 combo
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_ANALOG_DETAIL_COOLDOWN = 180   # 3 phút per user
_last_backtest_analog_detail: dict[str, float] = {}

# Map tên combo → index trong _ANALOG_COMBOS (dùng để lookup)
def _find_combo(name_query: str) -> dict | None:
    """
    Tìm combo theo tên, hỗ trợ partial match không phân biệt hoa thường.
    Ví dụ: "oversold" → "Oversold + Momentum"
    """
    q = name_query.lower().strip()
    # Exact match trước
    for c in _ANALOG_COMBOS:
        if c["name"].lower() == q:
            return c
    # Partial match
    for c in _ANALOG_COMBOS:
        if q in c["name"].lower():
            return c
    # Match từng từ
    words = q.split()
    for c in _ANALOG_COMBOS:
        if all(w in c["name"].lower() for w in words):
            return c
    return None


def _run_analog_detail_sync(symbol: str, combo_name: str, days: int = 1800) -> dict:
    """
    Chạy backtest cho 1 mã + 1 combo trên TẤT CẢ 7 threshold.
    Tái sử dụng logic từ _run_analog_backtest_sync nhưng chỉ cho 1 combo.
    """
    import numpy as np
    import pandas as pd

    # Tìm combo
    combo = _find_combo(combo_name)
    if combo is None:
        return {
            "status": "error",
            "error":  f"Khong tim thay combo '{combo_name}'. "
                      f"Cac combo co san: {', '.join(c['name'] for c in _ANALOG_COMBOS)}"
        }

    # Fetch data
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=days, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data {symbol}: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu (can >= 200 bars)"}

    n_bars    = len(df)
    dates     = df["date"].values
    close_arr = df["close"].values.astype(float)
    low_arr   = df["low"].values.astype(float)

    # Tính vectors
    try:
        from state_vector import compute_state_vector_for_date
    except ImportError as e:
        return {"status": "error", "error": f"Thieu state_vector: {e}"}

    vectors = {}
    for i in range(59, n_bars):
        vec = compute_state_vector_for_date(df, i)
        if vec is not None:
            vectors[i] = vec

    if len(vectors) < 100:
        return {"status": "error", "error": f"Chi co {len(vectors)} vectors"}

    vector_indices = sorted(vectors.keys())

    WIN_THRESH  = 1.0
    MIN_SAMPLES = 5
    MDS_DAYS    = 30
    FWD_DAYS    = 30
    dims        = combo["dims"]

    threshold_results = []

    for threshold in _ANALOG_THRESHOLDS:
        signals = []
        n_skip  = 0

        for t_idx in range(120, n_bars - FWD_DAYS - 1, 7):
            if t_idx not in vectors:
                continue

            target_vec = vectors[t_idx]
            target_arr = np.array([target_vec.get(d, 0.0) for d in dims], dtype=float)
            t_norm     = np.linalg.norm(target_arr)
            if t_norm < 1e-9:
                continue

            exclude_cutoff = t_idx - 90
            candidates     = [i for i in vector_indices if i < exclude_cutoff]
            if len(candidates) < 10:
                continue

            sim_list = []
            for c_idx in candidates:
                c_vec  = vectors[c_idx]
                c_arr  = np.array([c_vec.get(d, 0.0) for d in dims], dtype=float)
                c_norm = np.linalg.norm(c_arr)
                if c_norm < 1e-9:
                    continue
                sim = float(np.dot(target_arr, c_arr) / (t_norm * c_norm))
                if sim >= threshold:
                    sim_list.append((c_idx, sim))

            if not sim_list:
                continue

            # MDS
            sim_list.sort(key=lambda x: -x[1])
            kept = []
            for c_idx, sim in sim_list:
                c_date    = pd.Timestamp(dates[c_idx])
                too_close = any(
                    abs((c_date - pd.Timestamp(dates[k])).days) < MDS_DAYS
                    for k, _ in kept
                )
                if not too_close:
                    kept.append((c_idx, sim))

            if len(kept) < MIN_SAMPLES:
                n_skip += 1
                continue

            fwd_rets = []
            mae_vals = []
            for c_idx, _ in kept:
                fwd_idx = c_idx + FWD_DAYS
                if fwd_idx >= n_bars:
                    continue
                entry   = close_arr[c_idx]
                fwd_rets.append((close_arr[fwd_idx] - entry) / entry * 100)
                win_low = np.min(low_arr[c_idx + 1: fwd_idx + 1]) if fwd_idx > c_idx else entry
                mae_vals.append((win_low - entry) / entry * 100)

            if len(fwd_rets) < MIN_SAMPLES:
                n_skip += 1
                continue

            signals.append({
                "fwd_rets": fwd_rets,
                "mae_vals": mae_vals,
            })

        # Metrics
        if len(signals) < 5:
            threshold_results.append({
                "threshold": threshold,
                "n_signals": len(signals),
                "n_skip":    n_skip,
                "skip":      True,
            })
            continue

        sig_rets = [float(np.median(s["fwd_rets"])) for s in signals]
        sig_maes = [float(np.median(s["mae_vals"])) for s in signals]

        wins     = [x for x in sig_rets if x >= WIN_THRESH]
        losses   = [x for x in sig_rets if x < WIN_THRESH]
        wr       = len(wins) / len(sig_rets)
        mean_exp = float(np.mean(sig_rets))
        med_exp  = float(np.median(sig_rets))
        std_ret  = float(np.std(sig_rets)) if len(sig_rets) > 1 else 1e-9
        sharpe   = mean_exp / std_ret * (52 ** 0.5) if std_ret > 0 else 0.0
        pos_sum  = sum(wins)
        neg_sum  = abs(sum(losses)) if losses else 1e-9
        pf       = pos_sum / neg_sum if neg_sum > 0 else 99.0
        mae30    = float(np.median(sig_maes))
        # Worst Signal = return tệ nhất của 1 tín hiệu (metric rủi ro đúng)
        max_dd   = float(np.min(sig_rets))

        threshold_results.append({
            "threshold": threshold,
            "n_signals": len(signals),
            "n_skip":    n_skip,
            "wr":        round(wr * 100, 1),
            "mean_exp":  round(mean_exp, 2),
            "med_exp":   round(med_exp, 2),
            "mae30":     round(mae30, 2),
            "max_dd":    round(max_dd, 1),
            "sharpe":    round(sharpe, 3),
            "pf":        round(pf, 2),
            "skip":      False,
        })

    return {
        "status":   "ok",
        "symbol":   symbol,
        "combo":    combo["name"],
        "dims":     dims,
        "hypothesis": combo["hypothesis"],
        "n_bars":   n_bars,
        "n_vectors":len(vectors),
        "threshold_results": threshold_results,
    }


def _format_detail_result(res: dict) -> str:
    """Format full threshold table + khuyến nghị walk-forward."""
    symbol    = res["symbol"]
    combo     = res["combo"]
    hypo      = res["hypothesis"]
    n_bars    = res["n_bars"]
    n_vecs    = res["n_vectors"]
    rows      = res["threshold_results"]

    sep  = "=" * 36
    sep2 = "-" * 36

    lines = [
        f"DETAIL: {symbol} — {combo}",
        f"{n_bars} bars | {n_vecs} vectors",
        f">> {hypo}",
        sep,
        f"{'Nguong':<8} {'WR':<6} {'Exp':<8} {'Med':<8} {'Sharpe':<8} {'PF':<6} {'n':<5} {'Worst':<8} {'skip'}",
        sep2,
    ]

    valid_rows = [r for r in rows if not r.get("skip")]

    for r in rows:
        if r.get("skip"):
            lines.append(
                f"{r['threshold']:.2f}    "
                f"{'— skip (< 5 tin hieu)'}"
                f"  skip={r['n_skip']}"
            )
            continue

        # Đánh dấu row tốt nhất: pass bộ lọc cứng (Exp>0, PF>=1.5) + Exp cao nhất
        pass_rows  = [r for r in valid_rows if r["mean_exp"] > 0 and r["pf"] >= 1.5]
        best_exp_pf = max(pass_rows, key=lambda x: (x["mean_exp"], x["pf"])) if pass_rows else None
        is_best = best_exp_pf and r["threshold"] == best_exp_pf["threshold"]
        marker  = " ◄ Exp+PF" if is_best else ""

        lines.append(
            f"{r['threshold']:.2f}    "
            f"{r['wr']:.0f}%   "
            f"{r['mean_exp']:+.2f}%  "
            f"{r['med_exp']:+.2f}%  "
            f"{r['sharpe']:.2f}    "
            f"{r['pf']:.2f}  "
            f"{r['n_signals']:<5}"
            f"{r['max_dd']:.1f}%  "
            f"skip={r['n_skip']}"
            f"{marker}"
        )

    lines.append(sep2)

    # Phân tích và khuyến nghị
    if not valid_rows:
        lines.append("Khong co threshold nao du tin hieu.")
        return "\n".join(lines)

    # Best theo Exp+PF (bộ lọc cứng trước)
    pass_rows = [r for r in valid_rows if r["mean_exp"] > 0 and r["pf"] >= 1.5]
    if not pass_rows:
        # Không có row nào pass → dùng tất cả valid, chọn Exp cao nhất
        pass_rows = valid_rows

    best_exp_row    = max(pass_rows, key=lambda x: (x["mean_exp"], x["pf"]))
    # Risk-adjusted: Exp >= 90% max nhưng MaxDD nhỏ hơn
    max_exp         = best_exp_row["mean_exp"]
    risk_candidates = [r for r in pass_rows if r["mean_exp"] >= max_exp * 0.90]
    best_risk_adj   = max(risk_candidates, key=lambda x: x["max_dd"])

    lines.append("")
    lines.append("PHAN TICH THRESHOLD:")
    lines.append("  Xep hang: Exp (chinh) → PF (tiebreak) → Sharpe (tiebreak cuoi)")
    lines.append("  WR chi tham khao tam ly, khong dung lam tieu chi loc chinh.")
    lines.append("")

    # Trend analysis
    exp_vals    = [r["mean_exp"] for r in valid_rows]
    pf_vals     = [r["pf"]      for r in valid_rows]
    max_dd_vals = [r["max_dd"]  for r in valid_rows]
    if len(exp_vals) >= 3:
        peak_idx = exp_vals.index(max(exp_vals))
        has_peak = 0 < peak_idx < len(exp_vals) - 1
        if has_peak:
            lines.append(
                f"  Exp dat dinh tai nguong {valid_rows[peak_idx]['threshold']:.2f} "
                f"— tang hoac giam nguong deu kem hon."
            )
        else:
            lines.append(
                f"  Exp tang dan theo threshold "
                f"(nguong cao = chat loc tot hon voi combo nay)."
            )
        dd_improving = all(
            max_dd_vals[i] >= max_dd_vals[i+1]
            for i in range(len(max_dd_vals)-1)
        )
        if dd_improving:
            lines.append(
                "  MaxDD giam dan khi tang threshold — it rui ro hon."
            )

    lines.append("")
    lines.append(
        f"  Exp+PF tot nhat : nguong {best_exp_row['threshold']:.2f} "
        f"(Exp {best_exp_row['mean_exp']:+.2f}%, PF {best_exp_row['pf']:.2f}, "
        f"Worst {best_exp_row['max_dd']:.1f}%)"
    )
    lines.append(
        f"  Risk-adjusted   : nguong {best_risk_adj['threshold']:.2f} "
        f"(Exp {best_risk_adj['mean_exp']:+.2f}%, PF {best_risk_adj['pf']:.2f}, "
        f"Worst {best_risk_adj['max_dd']:.1f}%)"
    )
    if best_exp_row["wr"] < 50:
        lines.append(
            f"  Luu y tam ly: WR {best_exp_row['wr']:.0f}% — "
            f"chap nhan cuoi thua nhieu hon thang nhung Exp va PF van duong."
        )
    lines.append("")
    lines.append("KHUYEN NGHI WALK-FORWARD:")
    if best_exp_row["threshold"] == best_risk_adj["threshold"]:
        lines.append(
            f"  → Dung nguong {best_exp_row['threshold']:.2f} "
            f"(vua Exp+PF tot nhat vua MaxDD tot)."
        )
    else:
        lines.append(
            f"  → Uu tien loi nhuan cao : {best_exp_row['threshold']:.2f} "
            f"(Exp {best_exp_row['mean_exp']:+.2f}%, PF {best_exp_row['pf']:.2f})"
        )
        lines.append(
            f"  → Uu tien rui ro thap  : {best_risk_adj['threshold']:.2f} "
            f"(Exp {best_risk_adj['mean_exp']:+.2f}%, Worst {best_risk_adj['max_dd']:.1f}%)"
        )
        exp_diff = best_exp_row["mean_exp"] - best_risk_adj["mean_exp"]
        dd_diff  = abs(best_exp_row["max_dd"] - best_risk_adj["max_dd"])
        lines.append(
            f"  → Goi y: dung {best_risk_adj['threshold']:.2f} cho live trading "
            f"(Exp chi kem {exp_diff:.2f}% nhung Worst tot hon {dd_diff:.1f}%)"
        )

    lines.append("")
    lines.append(
        "Day la IN-SAMPLE. Fix threshold truoc khi chay walk-forward,"
    )
    lines.append("khong dieu chinh sau khi thay ket qua.")
    lines.append(sep)

    return "\n".join(lines)


async def backtest_analog_detail_cmd(update, context):
    """
    /backtest_analog_detail <MA> <combo> [days]

    Hien thi full threshold table (7 nguong) cho 1 ma + 1 combo.
    Giup chon threshold chinh xac truoc khi walk-forward.

    Vi du:
      /backtest_analog_detail SSI "Oversold + Momentum"
      /backtest_analog_detail HPG "Volume Confirmed"
      /backtest_analog_detail HPG volume          (partial match)
      /backtest_analog_detail SSI oversold 1500
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []

    # List combos nếu không có args
    if not args:
        combo_list = "\n".join(f"  • {c['name']}" for c in _ANALOG_COMBOS)
        await update.message.reply_text(
            "Cu phap: /backtest_analog_detail <MA> <combo> [days]\n\n"
            "Vi du:\n"
            "  /backtest_analog_detail SSI oversold\n"
            "  /backtest_analog_detail HPG volume\n"
            "  /backtest_analog_detail HPG \"Volume Confirmed\" 1500\n\n"
            f"Cac combo co san:\n{combo_list}"
        )
        return

    import re as _re

    # Parse symbol
    symbol = args[0].upper().strip()
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol):
        await update.message.reply_text(f"Ma '{symbol}' khong hop le.")
        return

    if len(args) < 2:
        await update.message.reply_text(
            f"Can them ten combo. Vi du:\n"
            f"  /backtest_analog_detail {symbol} volume\n"
            f"  /backtest_analog_detail {symbol} oversold"
        )
        return

    # Parse combo name và days
    # Ghép args[1:] để hỗ trợ tên combo nhiều từ
    joined  = " ".join(args[1:])
    # Tách days nếu có số ở cuối
    days    = 1800
    parts   = joined.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        days       = max(500, min(int(parts[1]), 2500))
        combo_query = parts[0].strip().strip('"')
    else:
        combo_query = joined.strip().strip('"')

    # Validate combo
    combo = _find_combo(combo_query)
    if combo is None:
        combo_list = "\n".join(f"  • {c['name']}" for c in _ANALOG_COMBOS)
        await update.message.reply_text(
            f"Khong tim thay combo '{combo_query}'.\n\n"
            f"Cac combo co san:\n{combo_list}"
        )
        return

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_backtest_analog_detail.get(user_id, 0)
    if since < BACKTEST_ANALOG_DETAIL_COOLDOWN:
        wait = int(BACKTEST_ANALOG_DETAIL_COOLDOWN - since)
        await update.message.reply_text(f"Vui long cho {wait}s truoc khi chay tiep.")
        return
    _last_backtest_analog_detail[user_id] = time.time()

    chat_id = update.effective_chat.id
    msg     = await update.message.reply_text(
        f"Detail: {symbol} — {combo['name']}\n"
        f"7 threshold × 1 combo | {days} ngay\n"
        f"Uoc tinh ~2-4 phut..."
    )

    async def _bg():
        try:
            result = await asyncio.to_thread(
                _run_analog_detail_sync, symbol, combo_query, days
            )

            if result["status"] == "error":
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=f"Loi: {result.get('error','?')[:300]}",
                )
                return

            summary = _format_detail_result(result)
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=_plain(summary)[:4096],
            )

        except Exception as e:
            import traceback
            logger.error(f"backtest_analog_detail_cmd error: {e}\n{traceback.format_exc()}")
            err_text = f"Loi /backtest_analog_detail: {str(e)[:200]}"
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id, text=err_text,
                )
            except Exception:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=err_text)
                except Exception:
                    pass

    asyncio.create_task(_bg())


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD ANALOG — OOS test 01/01/2025 đến nay
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_ANALOG_WF_COOLDOWN = 300   # 5 phút per user
_last_backtest_analog_wf: dict[str, float] = {}

# Config cố định cho từng mã — kết quả từ backtest + detail
# KHÔNG thay đổi sau khi thấy kết quả walk-forward
# Config mặc định (hardcode) — fallback khi DB chưa có dữ liệu
_WF_SYMBOL_CONFIG_DEFAULT = {
    "FPT": {"combo": "Macro Trend",       "threshold": 0.55},
    "MWG": {"combo": "Oversold Bounce",   "threshold": 0.55},
    "STB": {"combo": "Oversold Bounce",   "threshold": 0.60},
    "HPG": {"combo": "Volume Confirmed",  "threshold": 0.80},
    "GAS": {"combo": "No Volume",         "threshold": 0.55},
    "DPM": {"combo": "Volatility Aware",  "threshold": 0.55},
    "DCM": {"combo": "Volatility Aware",  "threshold": 0.55},
}

# Config live — được merge từ hardcode + DB khi bot start
# Dùng dict mutable để _run_pipeline_wf_sync có thể cập nhật runtime
_WF_SYMBOL_CONFIG: dict = dict(_WF_SYMBOL_CONFIG_DEFAULT)


def _load_wf_config_from_db():
    """
    Load analog config từ DB và merge vào _WF_SYMBOL_CONFIG.
    DB config ưu tiên hơn hardcode (override).
    Gọi 1 lần khi bot start (từ bot.py sau init_db()).
    """
    global _WF_SYMBOL_CONFIG
    try:
        from db import load_analog_configs
        db_configs = load_analog_configs()
        if db_configs:
            _WF_SYMBOL_CONFIG = dict(_WF_SYMBOL_CONFIG_DEFAULT)
            _WF_SYMBOL_CONFIG.update(db_configs)
            logger.info(
                f"[WFConfig] Loaded {len(db_configs)} configs from DB. "
                f"Total: {len(_WF_SYMBOL_CONFIG)} symbols: {list(_WF_SYMBOL_CONFIG.keys())}"
            )
        else:
            logger.info("[WFConfig] No DB configs found, using hardcoded defaults.")
    except Exception as e:
        logger.warning(f"[WFConfig] load_wf_config_from_db error (using defaults): {e}")


# Giai đoạn OOS
_WF_START_DATE = "2025-01-01"


def _run_pipeline_wf_sync(symbol: str) -> dict:
    """
    Dùng cho mã CHƯA có trong _WF_SYMBOL_CONFIG:
      1. Chạy Tầng 1 (105 experiments) để tìm combo + threshold tối ưu
      2. Chạy Tầng 2 (walk-forward OOS) với config vừa tìm được
      3. Trả về cùng format với _run_walkforward_sync để dùng chung formatter

    Sau khi có kết quả PASS, admin có thể thêm vào _WF_SYMBOL_CONFIG và
    SIGNAL_SYMBOLS để đưa vào live trading — không cần nhờ Claude nữa.
    """
    import numpy as np
    import pandas as pd

    # Tầng 1: tìm combo + threshold tốt nhất
    bt_res = _run_analog_backtest_sync(symbol)
    if bt_res.get("status") == "error":
        return {"status": "error", "error": bt_res["error"]}

    valid = bt_res.get("top_results", [])
    pass_results = [
        r for r in valid
        if r["mean_exp"] > _PIPELINE_MIN_EXP
        and r["pf"] >= _PIPELINE_MIN_PF
        and r["max_dd"] >= _PIPELINE_MAX_DD
    ]
    if not pass_results:
        top = valid[0] if valid else {}
        return {
            "status": "skip",
            "reason": (
                f"Khong co combo nao qua bo loc "
                f"(top: Exp={top.get('mean_exp',0):+.2f}% PF={top.get('pf',0):.2f})"
            ),
        }

    best_bt    = pass_results[0]
    combo_name = best_bt["combo"]

    # Tầng 1b: detail để tìm threshold tối ưu
    detail_res = _run_analog_detail_sync(symbol, combo_name)
    if detail_res.get("status") != "error":
        thresh_rows = [
            r for r in detail_res.get("threshold_results", [])
            if not r.get("skip")
            and r["mean_exp"] > _PIPELINE_MIN_EXP
            and r["pf"] >= _PIPELINE_MIN_PF
            and r["max_dd"] >= _PIPELINE_MAX_DD
        ]
        if thresh_rows:
            # Chọn threshold cho Exp cao nhất, PF làm tiebreak
            # Không dùng risk-adjusted để tránh pick threshold Exp thấp hơn
            best_thresh_row = max(thresh_rows, key=lambda x: (x["mean_exp"], x["pf"]))
            threshold       = best_thresh_row["threshold"]
        else:
            threshold = best_bt["threshold"]
    else:
        threshold = best_bt["threshold"]

    # Tầng 2: walk-forward với config vừa tìm
    combo = _find_combo(combo_name)
    if combo is None:
        return {"status": "error", "error": f"Khong tim thay combo '{combo_name}'"}

    dims = combo["dims"]

    # Thêm tạm vào _WF_SYMBOL_CONFIG để _run_walkforward_sync dùng được
    _WF_SYMBOL_CONFIG[symbol] = {"combo": combo_name, "threshold": threshold}
    result = _run_walkforward_sync(symbol)

    # Kiểm tra kết quả: nếu không ok → xoá khỏi memory
    oos_pass = False
    if result.get("status") == "ok":
        oos_m    = result.get("oos_metrics") or {}
        oos_pass = oos_m.get("mean_exp", 0) > 0 and oos_m.get("pf", 0) >= 1.5

    if not oos_pass:
        _WF_SYMBOL_CONFIG.pop(symbol, None)

    # Đánh dấu kết quả đến từ auto-pipeline để formatter và /analog_approve biết
    if result.get("status") == "ok":
        result["auto_config"]      = True
        result["found_combo"]      = combo_name
        result["found_threshold"]  = threshold
        result["oos_pass"]         = oos_pass

    return result


def _run_walkforward_sync(symbol: str) -> dict:
    """
    Walk-forward cho 1 mã:
    - Training  : tất cả data TRƯỚC 2025-01-01
    - OOS       : 2025-01-01 đến nay
    - Config    : cố định từ _WF_SYMBOL_CONFIG
    - Step      : 7 ngày
    - So sánh  : WR/Exp/PF OOS vs in-sample
    """
    import numpy as np
    import pandas as pd

    symbol = symbol.upper()

    # Kiểm tra config
    if symbol not in _WF_SYMBOL_CONFIG:
        return {
            "status": "error",
            "error":  f"{symbol} chua co config. Co san: {', '.join(_WF_SYMBOL_CONFIG)}"
        }

    cfg        = _WF_SYMBOL_CONFIG[symbol]
    combo_name = cfg["combo"]
    threshold  = cfg["threshold"]

    # Tìm combo dims
    combo = _find_combo(combo_name)
    if combo is None:
        return {"status": "error", "error": f"Khong tim thay combo '{combo_name}'"}
    dims = combo["dims"]

    # Fetch data — lấy đủ dài để có cả training lẫn OOS
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu"}

    # Tính vectors toàn bộ
    try:
        from state_vector import compute_state_vector_for_date
    except ImportError as e:
        return {"status": "error", "error": f"Thieu state_vector: {e}"}

    vectors = {}
    for i in range(59, len(df)):
        vec = compute_state_vector_for_date(df, i)
        if vec is not None:
            vectors[i] = vec

    if len(vectors) < 100:
        return {"status": "error", "error": "Khong du vectors"}

    n_bars         = len(df)
    dates          = df["date"].values
    close_arr      = df["close"].values.astype(float)
    low_arr        = df["low"].values.astype(float)
    vector_indices = sorted(vectors.keys())

    # Xác định ranh giới OOS
    wf_start   = pd.Timestamp(_WF_START_DATE)
    wf_end     = pd.Timestamp.now().normalize()
    date_series= pd.to_datetime(df["date"])

    # Index đầu tiên của OOS
    oos_start_idx = next(
        (i for i, d in enumerate(date_series) if pd.Timestamp(d) >= wf_start),
        None
    )
    if oos_start_idx is None:
        return {"status": "error", "error": f"Khong co data sau {_WF_START_DATE}"}

    n_oos_bars = n_bars - oos_start_idx

    WIN_THRESH  = 1.0
    MIN_SAMPLES = 5
    # MDS_DAYS: 43 calendar days ≈ 30 trading days
    MDS_DAYS    = 43
    FWD_DAYS    = 30
    # LOOP_STEP = 1: check mỗi ngày như live trading thực tế
    # COOLDOWN_BARS = 5: khớp đúng với live trading
    # Live dùng SIGNAL_COOLDOWN_DAYS = 7 calendar days ≈ 5 trading bars
    LOOP_STEP     = 1
    COOLDOWN_BARS = 5

    oos_signals          = []
    train_signals        = []
    n_skip_oos           = 0
    n_skip_train         = 0
    n_skip_cooldown      = 0
    last_oos_signal_idx  = None

    for t_idx in range(120, n_bars - FWD_DAYS - 1, LOOP_STEP):
        if t_idx not in vectors:
            continue

        t_date    = pd.Timestamp(dates[t_idx])
        is_oos    = t_date >= wf_start

        # Simulate cooldown cho OOS — dùng bar index để chính xác
        # (calendar days không dùng được vì 7 bars ≈ 10 cal days > cooldown 7 ngày)
        if is_oos and last_oos_signal_idx is not None:
            bars_since = t_idx - last_oos_signal_idx
            if bars_since < COOLDOWN_BARS:
                n_skip_cooldown += 1
                continue

        target_vec = vectors[t_idx]

        # Hard filter — lọc ngày T không đúng regime của combo
        # Thiếu filter này → mọi ngày sau cooldown đều pass → n cơ học
        if not _check_hard_filter(target_vec, combo_name):
            continue

        target_arr = np.array([target_vec.get(d, 0.0) for d in dims], dtype=float)
        t_norm     = np.linalg.norm(target_arr)
        if t_norm < 1e-9:
            continue

        # Chỉ dùng data TRAINING (trước t_idx - 90 ngày VÀ trước OOS)
        # Đây là điểm quan trọng nhất của walk-forward:
        # Ngay cả khi t_idx ở OOS, analog vẫn chỉ tìm trong training
        exclude_cutoff = min(t_idx - 90, oos_start_idx - 1)
        candidates     = [i for i in vector_indices if i < exclude_cutoff]
        if len(candidates) < 10:
            continue

        # Cosine similarity
        sim_list = []
        for c_idx in candidates:
            c_vec  = vectors[c_idx]
            c_arr  = np.array([c_vec.get(d, 0.0) for d in dims], dtype=float)
            c_norm = np.linalg.norm(c_arr)
            if c_norm < 1e-9:
                continue
            sim = float(np.dot(target_arr, c_arr) / (t_norm * c_norm))
            if sim >= threshold:
                sim_list.append((c_idx, sim))

        if not sim_list:
            continue

        # MDS — dùng calendar days, 43 cal ≈ 30 trading days
        sim_list.sort(key=lambda x: -x[1])
        kept = []
        for c_idx, sim in sim_list:
            c_date    = pd.Timestamp(dates[c_idx])
            too_close = any(
                abs((c_date - pd.Timestamp(dates[k])).days) < MDS_DAYS
                for k, _ in kept
            )
            if not too_close:
                kept.append((c_idx, sim))

        if len(kept) < MIN_SAMPLES:
            if is_oos:
                n_skip_oos += 1
            else:
                n_skip_train += 1
            continue

        # Forward return 30D
        fwd_rets = []
        mae_vals = []
        for c_idx, _ in kept:
            fwd_idx = c_idx + FWD_DAYS
            if fwd_idx >= n_bars:
                continue
            entry = close_arr[c_idx]
            fwd_rets.append((close_arr[fwd_idx] - entry) / entry * 100)
            win_low = np.min(low_arr[c_idx + 1: fwd_idx + 1]) if fwd_idx > c_idx else entry
            mae_vals.append((win_low - entry) / entry * 100)

        if len(fwd_rets) < MIN_SAMPLES:
            if is_oos:
                n_skip_oos += 1
            else:
                n_skip_train += 1
            continue

        # Actual forward return tại ngày T (OOS: đây là kết quả thực tế)
        actual_fwd_idx = t_idx + FWD_DAYS
        if actual_fwd_idx >= n_bars:
            # Chưa có kết quả (chưa đủ 30 ngày) → ghi nhận pending
            signal = {
                "t_idx":    t_idx,
                "t_date":   str(t_date)[:10],
                "fwd_rets": fwd_rets,
                "mae_vals": mae_vals,
                "predicted":float(np.median(fwd_rets)),
                "actual":   None,   # chưa có kết quả
                "pending":  True,
            }
        else:
            actual = (close_arr[actual_fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100
            signal = {
                "t_idx":    t_idx,
                "t_date":   str(t_date)[:10],
                "fwd_rets": fwd_rets,
                "mae_vals": mae_vals,
                "predicted":float(np.median(fwd_rets)),
                "actual":   actual,
                "pending":  False,
            }

        if is_oos:
            oos_signals.append(signal)
            last_oos_signal_idx = t_idx   # cập nhật bar index (không dùng date)
        else:
            train_signals.append(signal)

    # ── Tính metrics OOS (chỉ tín hiệu đã có kết quả) ─────────────────────────
    def _calc_metrics(signals):
        completed = [s for s in signals if not s.get("pending") and s["actual"] is not None]
        if len(completed) < 3:
            return None
        actuals  = [s["actual"] for s in completed]
        wins     = [x for x in actuals if x >= WIN_THRESH]
        losses   = [x for x in actuals if x < WIN_THRESH]
        wr       = len(wins) / len(actuals)
        mean_exp = float(np.mean(actuals))
        med_exp  = float(np.median(actuals))
        std_ret  = float(np.std(actuals)) if len(actuals) > 1 else 1e-9
        sharpe   = mean_exp / std_ret * (52 ** 0.5) if std_ret > 0 else 0.0
        pos_sum  = sum(wins)
        neg_sum  = abs(sum(losses)) if losses else 1e-9
        pf       = pos_sum / neg_sum if neg_sum > 0 else 99.0
        # Median MAE
        all_maes = [float(np.median(s["mae_vals"])) for s in completed if s.get("mae_vals")]
        mae30    = float(np.median(all_maes)) if all_maes else 0.0
        # Worst Signal = return thực tế tệ nhất của 1 tín hiệu OOS
        max_dd   = float(np.min(actuals)) if actuals else 0.0
        return {
            "n":        len(completed),
            "n_pending":len([s for s in signals if s.get("pending")]),
            "wr":       round(wr * 100, 1),
            "mean_exp": round(mean_exp, 2),
            "med_exp":  round(med_exp, 2),
            "sharpe":   round(sharpe, 3),
            "pf":       round(pf, 2),
            "mae30":    round(mae30, 2),
            "max_dd":   round(max_dd, 1),
        }

    oos_metrics   = _calc_metrics(oos_signals)
    train_metrics = _calc_metrics(train_signals)

    return {
        "status":           "ok",
        "symbol":           symbol,
        "combo":            combo_name,
        "threshold":        threshold,
        "wf_start":         _WF_START_DATE,
        "n_oos_bars":       n_oos_bars,
        "n_skip_oos":       n_skip_oos,
        "n_skip_train":     n_skip_train,
        "n_skip_cooldown":  n_skip_cooldown,
        "cooldown_bars":    COOLDOWN_BARS,
        "oos_signals":      oos_signals,
        "oos_metrics":      oos_metrics,
        "train_metrics":    train_metrics,
    }


def _format_wf_result(res: dict) -> str:
    """Format kết quả walk-forward so sánh OOS vs Training."""
    symbol    = res["symbol"]
    combo     = res["combo"]
    threshold = res["threshold"]
    wf_start  = res["wf_start"]
    oos_m     = res.get("oos_metrics")
    train_m   = res.get("train_metrics")
    oos_sigs  = res.get("oos_signals", [])
    n_pending = sum(1 for s in oos_sigs if s.get("pending"))

    sep  = "=" * 36
    sep2 = "-" * 36

    lines = [
        f"WALK-FORWARD: {symbol}",
        f"Config: {combo} | nguong {threshold}",
        f"OOS: {wf_start} → hom nay",
        sep,
    ]

    # ── Training metrics ──────────────────────────────────────────────────────
    lines.append("TRAINING (in-sample, truoc 2025):")
    if train_m:
        lines.append(
            f"  WR {train_m['wr']:.0f}%  "
            f"Exp {train_m['mean_exp']:+.2f}%  "
            f"PF {train_m['pf']:.2f}  "
            f"Sharpe {train_m['sharpe']:.2f}  "
            f"n={train_m['n']}"
        )
        lines.append(
            f"  MAE30 {train_m['mae30']:.1f}%  Worst {train_m['max_dd']:.1f}%"
        )
    else:
        lines.append("  Khong du tin hieu training.")
    lines.append(sep2)

    # ── OOS metrics ───────────────────────────────────────────────────────────
    n_skip_cooldown = res.get("n_skip_cooldown", 0)
    cooldown_bars   = res.get("cooldown_bars", 10)
    lines.append(f"OUT-OF-SAMPLE ({wf_start} → hom nay):")
    lines.append(f"  [Simulate cooldown {cooldown_bars} bars (~1 tuan) — giong live trading]")
    if oos_m:
        lines.append(
            f"  WR {oos_m['wr']:.0f}%  "
            f"Exp {oos_m['mean_exp']:+.2f}%  "
            f"PF {oos_m['pf']:.2f}  "
            f"Sharpe {oos_m['sharpe']:.2f}  "
            f"n={oos_m['n']}"
            + (f" (+{n_pending} pending)" if n_pending else "")
        )
        lines.append(
            f"  MAE30 {oos_m['mae30']:.1f}%  Worst {oos_m['max_dd']:.1f}%"
        )
        if n_skip_cooldown:
            lines.append(
                f"  (Bo qua {n_skip_cooldown} nut scan vi cooldown — "
                f"giu lai {oos_m['n']} tin hieu thuc su)"
            )
    else:
        lines.append(
            f"  Chua du tin hieu OOS (can >= 3 tin hieu hoan thanh)."
            + (f" Co {n_pending} tin hieu dang cho ket qua (< 30 ngay)." if n_pending else "")
        )
    lines.append(sep2)

    # ── So sánh OOS vs Training ───────────────────────────────────────────────
    if oos_m and train_m:
        lines.append("SO SANH OOS vs TRAINING:")

        def _delta(oos_val, train_val, higher_is_better=True):
            diff = oos_val - train_val
            ok   = diff >= 0 if higher_is_better else diff <= 0
            sign = "+" if diff >= 0 else ""
            em   = "✅" if ok else "⚠️"
            return f"{em} {sign}{diff:.2f}"

        lines.append(
            f"  Exp   : OOS {oos_m['mean_exp']:+.2f}% vs Train {train_m['mean_exp']:+.2f}%  "
            f"{_delta(oos_m['mean_exp'], train_m['mean_exp'])}"
        )
        lines.append(
            f"  PF    : OOS {oos_m['pf']:.2f} vs Train {train_m['pf']:.2f}  "
            f"{_delta(oos_m['pf'], train_m['pf'])}"
        )
        lines.append(
            f"  WR    : OOS {oos_m['wr']:.0f}% vs Train {train_m['wr']:.0f}%  "
            f"{_delta(oos_m['wr'], train_m['wr'])}"
        )
        lines.append(
            f"  Worst : OOS {oos_m['max_dd']:.1f}% vs Train {train_m['max_dd']:.1f}%  "
            f"{_delta(oos_m['max_dd'], train_m['max_dd'], higher_is_better=False)}"
        )
        lines.append(sep2)

        # ── Verdict ───────────────────────────────────────────────────────────
        lines.append("VERDICT:")
        exp_ok  = oos_m["mean_exp"] > 0
        pf_ok   = oos_m["pf"] >= 1.5
        exp_deg = oos_m["mean_exp"] >= train_m["mean_exp"] * 0.60
        pf_deg  = oos_m["pf"] >= train_m["pf"] * 0.50

        if exp_ok and pf_ok and exp_deg and pf_deg:
            lines.append(f"  ✅ PASS — Pattern con gia tri tren OOS.")
            lines.append(f"     Exp {oos_m['mean_exp']:+.2f}% dương, PF {oos_m['pf']:.2f} >= 1.5.")
            if oos_m["mean_exp"] >= train_m["mean_exp"] * 0.80:
                lines.append("     Hieu suat OOS giu duoc >= 80% so voi training — rat tot.")
            else:
                lines.append("     Hieu suat OOS giam nhung van chap nhan duoc (>60% training).")
            lines.append("     → Co the tien toi live trading than trong.")
        elif exp_ok and pf_ok:
            lines.append(f"  🟡 PARTIAL PASS — Exp va PF van duong nhung suy giam nhieu.")
            lines.append(
                f"     Exp OOS = {oos_m['mean_exp'] / train_m['mean_exp'] * 100:.0f}% so voi training."
            )
            lines.append("     → Theo doi them, chua nen live trading.")
        else:
            lines.append("  🔴 FAIL — OOS khong xac nhan pattern training.")
            if not exp_ok:
                lines.append(f"     Exp OOS am ({oos_m['mean_exp']:+.2f}%) — pattern het hieu luc.")
            if not pf_ok:
                lines.append(f"     PF OOS {oos_m['pf']:.2f} < 1.5 — he thong khong co loi nhuan thuc.")
            lines.append("     → Khong dung cho live trading.")

        # Cảnh báo n nhỏ
        if oos_m["n"] < 20:
            lines.append(
                f"  ⚠️  Chi co {oos_m['n']} tin hieu OOS — "
                f"ket qua co the chua on dinh, can them thoi gian."
            )

        # ── Random Baseline (null hypothesis) ─────────────────────────────────
        baseline = _run_random_baseline(symbol)
        if baseline.get("status") == "ok":
            lines.append(sep2)
            lines.append("NULL HYPOTHESIS (random entry baseline):")
            lines.append(
                f"  Random {baseline['n_trials']} entry OOS: "
                f"WR {baseline['wr']:.0f}%  Exp {baseline['mean_exp']:+.2f}%  PF {baseline['pf']:.2f}"
            )
            alpha_exp = oos_m["mean_exp"] - baseline["mean_exp"]
            alpha_pf  = oos_m["pf"] - baseline["pf"]
            alpha_em  = "✅" if alpha_exp > 0.5 else ("⚠️" if alpha_exp > 0 else "🔴")
            lines.append(
                f"  Alpha thuc (WF - Random): "
                f"Exp {alpha_em} {alpha_exp:+.2f}%  PF {alpha_pf:+.2f}"
            )
            if alpha_exp <= 0:
                lines.append(
                    "  🔴 CANH BAO: WF OOS khong tot hon random entry "
                    "→ co the chi la market drift, khong co edge pattern."
                )
            elif alpha_exp < 0.5:
                lines.append(
                    "  ⚠️  Alpha nho (<0.5%) — can them n de xac nhan edge thuc su."
                )
            else:
                lines.append(
                    f"  ✅ Alpha duong ro rang ({alpha_exp:+.2f}%) — "
                    "co bang chung pattern matching them gia tri ngoai drift."
                )

    lines.append("")
    lines.append(
        "Config da duoc fix truoc khi chay OOS — ket qua nay co gia tri thong ke."
    )

    # Gợi ý thêm config khi đến từ auto-pipeline và PASS
    if res.get("auto_config"):
        oos_ok = oos_m and oos_m.get("mean_exp", 0) > 0 and oos_m.get("pf", 0) >= 1.5
        if oos_ok:
            lines.append(sep2)
            lines.append(f"💡 AUTO-PIPELINE — Tim thay config moi:")
            lines.append(f"   Combo: {combo} | Nguong: {threshold}")
            lines.append(f"   De them vao live trading, cap nhat 2 cho trong code:")
            lines.append(f"   1. _WF_SYMBOL_CONFIG: \"{symbol}\": {{\"combo\": \"{combo}\", \"threshold\": {threshold}}}")
            lines.append(f"   2. SIGNAL_SYMBOLS trong analog_signal.py (them dims tuong ung)")
            lines.append(f"   Sau do chay /analog_pipeline {symbol} de xac nhan lai.")

    lines.append(sep)
    return "\n".join(lines)


def _run_random_baseline(symbol: str, n_trials: int = 500) -> dict:
    """
    Null hypothesis test: chọn ngày OOS ngẫu nhiên (không dùng analog signal),
    tính WR/Exp/PF 30D để đo market drift tự nhiên.

    So sánh với WF result → đo alpha thực sự của analog system.
    Nếu random baseline ≈ WF → không có edge, chỉ là market drift.
    """
    import numpy as np
    import pandas as pd
    import random

    symbol = symbol.upper()
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu"}

    close_arr   = df["close"].values.astype(float)
    date_series = pd.to_datetime(df["date"])
    wf_start    = pd.Timestamp(_WF_START_DATE)

    # Chỉ sample trong OOS period
    oos_indices = [
        i for i, d in enumerate(date_series)
        if pd.Timestamp(d) >= wf_start and i + 30 < len(df)
    ]

    if len(oos_indices) < 10:
        return {"status": "error", "error": "Khong du ngay OOS de chay baseline"}

    WIN_THRESH = 1.0
    random.seed(42)   # reproducible

    actuals = []
    for _ in range(n_trials):
        t_idx  = random.choice(oos_indices)
        entry  = close_arr[t_idx]
        fwd    = close_arr[t_idx + 30]
        ret    = (fwd - entry) / entry * 100
        actuals.append(ret)

    wins    = [x for x in actuals if x >= WIN_THRESH]
    losses  = [x for x in actuals if x < WIN_THRESH]
    wr      = len(wins) / len(actuals) * 100
    mean_exp = float(np.mean(actuals))
    pos_sum  = sum(wins)
    neg_sum  = abs(sum(losses)) if losses else 1e-9
    pf       = pos_sum / neg_sum if neg_sum > 0 else 99.0

    return {
        "status":    "ok",
        "symbol":    symbol,
        "n_trials":  n_trials,
        "wr":        round(wr, 1),
        "mean_exp":  round(mean_exp, 2),
        "pf":        round(pf, 2),
        "n_oos_bars": len(oos_indices),
    }



    """Format kết quả walk-forward so sánh OOS vs Training."""
    symbol    = res["symbol"]
    combo     = res["combo"]
    threshold = res["threshold"]
    wf_start  = res["wf_start"]
    oos_m     = res.get("oos_metrics")
    train_m   = res.get("train_metrics")
    oos_sigs  = res.get("oos_signals", [])
    n_pending = sum(1 for s in oos_sigs if s.get("pending"))

    sep  = "=" * 36
    sep2 = "-" * 36

    lines = [
        f"WALK-FORWARD: {symbol}",
        f"Config: {combo} | nguong {threshold}",
        f"OOS: {wf_start} → hom nay",
        sep,
    ]

    # ── Training metrics ──────────────────────────────────────────────────────
    lines.append("TRAINING (in-sample, truoc 2025):")
    if train_m:
        lines.append(
            f"  WR {train_m['wr']:.0f}%  "
            f"Exp {train_m['mean_exp']:+.2f}%  "
            f"PF {train_m['pf']:.2f}  "
            f"Sharpe {train_m['sharpe']:.2f}  "
            f"n={train_m['n']}"
        )
        lines.append(
            f"  MAE30 {train_m['mae30']:.1f}%  Worst {train_m['max_dd']:.1f}%"
        )
    else:
        lines.append("  Khong du tin hieu training.")
    lines.append(sep2)

    # ── OOS metrics ───────────────────────────────────────────────────────────
    n_skip_cooldown = res.get("n_skip_cooldown", 0)
    cooldown_bars   = res.get("cooldown_bars", 10)
    lines.append(f"OUT-OF-SAMPLE ({wf_start} → hom nay):")
    lines.append(f"  [Simulate cooldown {cooldown_bars} bars (~1 tuan) — giong live trading]")
    if oos_m:
        lines.append(
            f"  WR {oos_m['wr']:.0f}%  "
            f"Exp {oos_m['mean_exp']:+.2f}%  "
            f"PF {oos_m['pf']:.2f}  "
            f"Sharpe {oos_m['sharpe']:.2f}  "
            f"n={oos_m['n']}"
            + (f" (+{n_pending} pending)" if n_pending else "")
        )
        lines.append(
            f"  MAE30 {oos_m['mae30']:.1f}%  Worst {oos_m['max_dd']:.1f}%"
        )
        if n_skip_cooldown:
            lines.append(
                f"  (Bo qua {n_skip_cooldown} nut scan vi cooldown — "
                f"giu lai {oos_m['n']} tin hieu thuc su)"
            )
    else:
        lines.append(
            f"  Chua du tin hieu OOS (can >= 3 tin hieu hoan thanh)."
            + (f" Co {n_pending} tin hieu dang cho ket qua (< 30 ngay)." if n_pending else "")
        )
    lines.append(sep2)

    # ── So sánh OOS vs Training ───────────────────────────────────────────────
    if oos_m and train_m:
        lines.append("SO SANH OOS vs TRAINING:")

        def _delta(oos_val, train_val, higher_is_better=True):
            diff = oos_val - train_val
            ok   = diff >= 0 if higher_is_better else diff <= 0
            sign = "+" if diff >= 0 else ""
            em   = "✅" if ok else "⚠️"
            return f"{em} {sign}{diff:.2f}"

        lines.append(
            f"  Exp   : OOS {oos_m['mean_exp']:+.2f}% vs Train {train_m['mean_exp']:+.2f}%  "
            f"{_delta(oos_m['mean_exp'], train_m['mean_exp'])}"
        )
        lines.append(
            f"  PF    : OOS {oos_m['pf']:.2f} vs Train {train_m['pf']:.2f}  "
            f"{_delta(oos_m['pf'], train_m['pf'])}"
        )
        lines.append(
            f"  WR    : OOS {oos_m['wr']:.0f}% vs Train {train_m['wr']:.0f}%  "
            f"{_delta(oos_m['wr'], train_m['wr'])}"
        )
        lines.append(
            f"  Worst : OOS {oos_m['max_dd']:.1f}% vs Train {train_m['max_dd']:.1f}%  "
            f"{_delta(oos_m['max_dd'], train_m['max_dd'], higher_is_better=False)}"
        )
        lines.append(sep2)

        # ── Verdict ───────────────────────────────────────────────────────────
        lines.append("VERDICT:")
        exp_ok  = oos_m["mean_exp"] > 0
        pf_ok   = oos_m["pf"] >= 1.5
        exp_deg = oos_m["mean_exp"] >= train_m["mean_exp"] * 0.60  # cho phép giảm tối đa 40%
        pf_deg  = oos_m["pf"] >= train_m["pf"] * 0.50

        if exp_ok and pf_ok and exp_deg and pf_deg:
            lines.append(
                f"  ✅ PASS — Pattern con gia tri tren OOS."
            )
            lines.append(
                f"     Exp {oos_m['mean_exp']:+.2f}% dương, PF {oos_m['pf']:.2f} >= 1.5."
            )
            if oos_m["mean_exp"] >= train_m["mean_exp"] * 0.80:
                lines.append("     Hieu suat OOS giu duoc >= 80% so voi training — rat tot.")
            else:
                lines.append("     Hieu suat OOS giam nhung van chap nhan duoc (>60% training).")
            lines.append("     → Co the tien toi live trading than trong.")
        elif exp_ok and pf_ok:
            lines.append(
                f"  🟡 PARTIAL PASS — Exp va PF van duong nhung suy giam nhieu."
            )
            lines.append(
                f"     Exp OOS = {oos_m['mean_exp'] / train_m['mean_exp'] * 100:.0f}% so voi training."
            )
            lines.append("     → Theo doi them, chua nen live trading.")
        else:
            lines.append("  🔴 FAIL — OOS khong xac nhan pattern training.")
            if not exp_ok:
                lines.append(f"     Exp OOS am ({oos_m['mean_exp']:+.2f}%) — pattern het hieu luc.")
            if not pf_ok:
                lines.append(f"     PF OOS {oos_m['pf']:.2f} < 1.5 — he thong khong co loi nhuan thuc.")
            lines.append("     → Khong dung cho live trading.")

        # Cảnh báo n nhỏ
        if oos_m["n"] < 20:
            lines.append(
                f"  ⚠️  Chi co {oos_m['n']} tin hieu OOS — "
                f"ket qua co the chua on dinh, can them thoi gian."
            )

    # ── Random Baseline (null hypothesis) ─────────────────────────────────────
    baseline = _run_random_baseline(symbol)
    if baseline.get("status") == "ok" and oos_m:
        lines.append(sep2)
        lines.append("NULL HYPOTHESIS (random entry baseline):")
        lines.append(
            f"  Random {baseline['n_trials']} entry OOS: "
            f"WR {baseline['wr']:.0f}%  Exp {baseline['mean_exp']:+.2f}%  PF {baseline['pf']:.2f}"
        )
        # Tính alpha thực sự
        alpha_exp = oos_m["mean_exp"] - baseline["mean_exp"]
        alpha_pf  = oos_m["pf"] - baseline["pf"]
        alpha_em  = "✅" if alpha_exp > 0.5 else ("⚠️" if alpha_exp > 0 else "🔴")
        lines.append(
            f"  Alpha thuc (WF - Random): "
            f"Exp {alpha_em} {alpha_exp:+.2f}%  PF {alpha_pf:+.2f}"
        )
        if alpha_exp <= 0:
            lines.append(
                "  🔴 CANH BAO: WF OOS khong tot hon random entry "
                "→ co the chi la market drift, khong co edge pattern."
            )
        elif alpha_exp < 0.5:
            lines.append(
                "  ⚠️  Alpha nho (<0.5%) — can them n de xac nhan edge thuc su."
            )
        else:
            lines.append(
                f"  ✅ Alpha duong ro rang ({alpha_exp:+.2f}%) — "
                "co bang chung pattern matching them gia tri ngoai drift."
            )

    lines.append("")
    lines.append(
        "Config da duoc fix truoc khi chay OOS — ket qua nay co gia tri thong ke."
    )

    # Gợi ý thêm config khi đến từ auto-pipeline và PASS
    if res.get("auto_config"):
        oos_ok = oos_m and oos_m.get("mean_exp", 0) > 0 and oos_m.get("pf", 0) >= 1.5
        if oos_ok:
            lines.append(sep2)
            lines.append(f"💡 AUTO-PIPELINE — Tim thay config moi:")
            lines.append(f"   Combo: {combo} | Nguong: {threshold}")
            lines.append(f"   De them vao live trading, cap nhat 2 cho trong code:")
            lines.append(f"   1. _WF_SYMBOL_CONFIG: \"{symbol}\": {{\"combo\": \"{combo}\", \"threshold\": {threshold}}}")
            lines.append(f"   2. SIGNAL_SYMBOLS trong analog_signal.py (them dims tuong ung)")
            lines.append(f"   Sau do chay /analog_pipeline {symbol} de xac nhan lai.")

    lines.append(sep)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# REGIME ANALYSIS — phân tích phân bố tín hiệu OOS
# ══════════════════════════════════════════════════════════════════════════════

def _analyze_oos_regime(res: dict) -> str:
    """
    Phân tích phân bố 42 tín hiệu OOS theo:
      1. Phân bố theo tháng — phát hiện clustering
      2. Hiệu suất theo volatility regime (ATR)
      3. Hiệu suất theo trend regime (trend_slope)
      4. Kết luận + gợi ý sizing/filter
    Input: kết quả từ _run_walkforward_sync (có oos_signals + df)
    """
    import numpy as np
    import pandas as pd
    from collections import defaultdict

    symbol    = res["symbol"]
    combo     = res["combo"]
    threshold = res["threshold"]
    oos_sigs  = res.get("oos_signals", [])
    df        = res.get("df")

    sep  = "=" * 36
    sep2 = "-" * 36
    lines = [
        f"REGIME ANALYSIS — {symbol}",
        f"Config: {combo} | nguong {threshold}",
        f"OOS: {res.get('wf_start','2025-01-01')} → hom nay",
        sep,
    ]

    completed = [s for s in oos_sigs if not s.get("pending") and s["actual"] is not None]
    n = len(completed)

    if n < 5:
        lines.append(f"Khong du tin hieu OOS de phan tich (can >= 5, co {n}).")
        return "\n".join(lines)

    # ── 1. Phân bố theo tháng ─────────────────────────────────────────────────
    lines.append("1. PHAN BO THEO THANG:")
    by_month = defaultdict(list)
    for s in completed:
        ym = s["t_date"][:7]   # "2025-01"
        by_month[ym].append(s["actual"])

    months_sorted = sorted(by_month.keys())
    max_count     = max(len(v) for v in by_month.values())

    for ym in months_sorted:
        rets   = by_month[ym]
        cnt    = len(rets)
        avg    = float(np.mean(rets))
        bar    = "█" * cnt + "░" * (max_count - cnt)
        em     = "✅" if avg > 0 else "🔴"
        lines.append(f"  {ym}  {bar}  {cnt:>2} sig  avg {avg:+.1f}%  {em}")

    # Phát hiện clustering: tháng nào có > 30% tổng tín hiệu
    lines.append("")
    cluster_months = [(ym, len(v)) for ym, v in by_month.items() if len(v) / n > 0.30]
    if cluster_months:
        for ym, cnt in cluster_months:
            lines.append(f"  ⚠️  CUM: {ym} chiem {cnt/n*100:.0f}% tin hieu ({cnt}/{n})")
        lines.append(f"  → Pattern co the bi phu thuoc vao 1 regime ngan han")
    else:
        span_months = len(months_sorted)
        lines.append(f"  ✅ Khong phat hien cum — {n} tin hieu trai deu {span_months} thang")

    lines.append(sep2)

    # ── 2. Hiệu suất theo volatility regime (ATR) ─────────────────────────────
    lines.append("2. HIEU SUAT THEO VOLATILITY (ATR tai ngay tin hieu):")

    atr_data_ok = False
    if df is not None and "atr_ratio" in (df.columns if hasattr(df, "columns") else []):
        atr_data_ok = True

    if df is not None:
        try:
            from state_vector import compute_state_vector_for_date
            # Lấy atr_ratio tại mỗi t_idx
            date_to_idx = {str(df["date"].values[i])[:10]: i for i in range(len(df))}
            low_vol, mid_vol, high_vol = [], [], []

            for s in completed:
                idx = date_to_idx.get(s["t_date"])
                if idx is None or idx < 59:
                    continue
                vec = compute_state_vector_for_date(df, idx)
                if vec is None:
                    continue
                atr = vec.get("atr_ratio", 0.5)   # đã chuẩn hóa 0-1, 0.5 = ATR ~2.5%
                if atr < 0.35:
                    low_vol.append(s["actual"])
                elif atr > 0.65:
                    high_vol.append(s["actual"])
                else:
                    mid_vol.append(s["actual"])

            def _regime_line(label, rets):
                if len(rets) < 3:
                    return f"  {label}: n={len(rets)} (khong du de phan tich)"
                wr  = sum(1 for x in rets if x >= 1.0) / len(rets) * 100
                avg = float(np.mean(rets))
                em  = "✅" if avg > 0 and wr >= 55 else "⚠️" if avg > 0 else "🔴"
                return f"  {label}: n={len(rets):>2}  WR {wr:.0f}%  Exp {avg:+.1f}%  {em}"

            lines.append(_regime_line("Low  vol (ATR < 1.8%)", low_vol))
            lines.append(_regime_line("Mid  vol (ATR 1.8-3%)  ", mid_vol))
            lines.append(_regime_line("High vol (ATR > 3%)   ", high_vol))

            # Cảnh báo nếu high-vol tệ hơn rõ ràng
            if len(high_vol) >= 3 and len(low_vol) >= 3:
                avg_lo = float(np.mean(low_vol))
                avg_hi = float(np.mean(high_vol))
                if avg_hi < 0 and avg_lo > 0:
                    lines.append(f"  ⚠️  Pattern BREAKDOWN khi high-vol — nen tam dung signal")
                elif avg_lo - avg_hi > 2.0:
                    lines.append(f"  ⚠️  Hieu suat giam {avg_lo-avg_hi:.1f}% khi high-vol — giam sizing")
                else:
                    lines.append(f"  ✅ Hieu suat on dinh qua cac muc volatility")

        except Exception as e:
            lines.append(f"  Khong tinh duoc (loi state_vector: {e})")
    else:
        lines.append("  Khong co df de phan tich ATR.")

    lines.append(sep2)

    # ── 3. Hiệu suất theo trend regime (trend_slope) ──────────────────────────
    lines.append("3. HIEU SUAT THEO TREND REGIME (trend_slope tai ngay tin hieu):")

    if df is not None:
        try:
            from state_vector import compute_state_vector_for_date
            date_to_idx = {str(df["date"].values[i])[:10]: i for i in range(len(df))}
            trending, sideways, downtrend = [], [], []

            for s in completed:
                idx = date_to_idx.get(s["t_date"])
                if idx is None or idx < 59:
                    continue
                vec = compute_state_vector_for_date(df, idx)
                if vec is None:
                    continue
                slope = vec.get("trend_slope", 0.0)
                if slope > 0.20:
                    trending.append((s["actual"], slope))
                elif slope < -0.10:
                    downtrend.append((s["actual"], slope))
                else:
                    sideways.append((s["actual"], slope))

            def _trend_line(label, pairs):
                rets = [x for x, _ in pairs]
                if len(rets) < 3:
                    return f"  {label}: n={len(rets)} (khong du)"
                wr  = sum(1 for x in rets if x >= 1.0) / len(rets) * 100
                avg = float(np.mean(rets))
                em  = "✅" if avg > 0 and wr >= 55 else "⚠️" if avg > 0 else "🔴"
                return f"  {label}: n={len(rets):>2}  WR {wr:.0f}%  Exp {avg:+.1f}%  {em}"

            lines.append(_trend_line("Uptrend  (slope > 0.20) ", trending))
            lines.append(_trend_line("Sideways (-0.10 to 0.20)", sideways))
            lines.append(_trend_line("Downtrend(slope < -0.10)", downtrend))

            # Gợi ý combo phù hợp theo regime
            if len(trending) >= 3 and len(sideways) >= 3:
                avg_trend   = float(np.mean([x for x, _ in trending]))
                avg_side    = float(np.mean([x for x, _ in sideways]))
                if avg_trend > avg_side + 1.5:
                    lines.append(f"  → Pattern hoat dong tot hon khi UPTREND (+{avg_trend-avg_side:.1f}%)")
                    lines.append(f"    Xem xet them dieu kien trend_slope > 0.10 lam pre-filter")
                elif avg_side > avg_trend + 1.5:
                    lines.append(f"  → Pattern hoat dong tot hon khi SIDEWAYS (+{avg_side-avg_trend:.1f}%)")
                    lines.append(f"    Day la mean-reversion thuan — hop ly voi combo {combo}")
                else:
                    lines.append(f"  ✅ Pattern on dinh ca uptrend va sideways")

        except Exception as e:
            lines.append(f"  Khong tinh duoc (loi: {e})")
    else:
        lines.append("  Khong co df de phan tich trend.")

    lines.append(sep2)

    # ── 4. Khoảng cách giữa các tín hiệu (inter-signal gap) ──────────────────
    lines.append("4. KHOANG CACH GIUA CAC TIN HIEU:")
    dates_sorted = sorted(s["t_date"] for s in completed)
    if len(dates_sorted) >= 2:
        gaps = []
        for i in range(1, len(dates_sorted)):
            d0 = pd.Timestamp(dates_sorted[i - 1])
            d1 = pd.Timestamp(dates_sorted[i])
            gaps.append((d1 - d0).days)

        avg_gap    = float(np.mean(gaps))
        min_gap    = int(np.min(gaps))
        n_close    = sum(1 for g in gaps if g < 14)
        lines.append(f"  Khoang cach TB  : {avg_gap:.1f} ngay")
        lines.append(f"  Khoang cach ngan nhat: {min_gap} ngay")
        lines.append(f"  Cap tin hieu < 14 ngay: {n_close}/{len(gaps)}")

        if n_close / len(gaps) > 0.35:
            lines.append(f"  ⚠️  {n_close/len(gaps)*100:.0f}% cap tin hieu rat gan nhau — co the cung 1 dot bien")
        else:
            lines.append(f"  ✅ Phan bo khoang cach hop ly — tin hieu doc lap nhau")
    lines.append(sep2)

    # ── 5. Kết luận tổng hợp ──────────────────────────────────────────────────
    lines.append("5. KET LUAN VA GIA Y:")

    actuals   = [s["actual"] for s in completed]
    overall_wr  = sum(1 for x in actuals if x >= 1.0) / n * 100
    overall_exp = float(np.mean(actuals))

    issues = []
    suggestions = []

    # Kiểm tra clustering tháng
    if cluster_months:
        issues.append("Pattern bi cum theo thang")
        suggestions.append("Giam sizing xuong 50% va theo doi them 2-3 thang")

    # Kiểm tra inter-signal gap
    if len(dates_sorted) >= 2 and n_close / max(len(gaps), 1) > 0.35:
        issues.append("Nhieu tin hieu xuat hien trong thoi gian ngan")
        suggestions.append("Xem xet tang cooldown len 14 ngay")

    if not issues:
        lines.append(f"  ✅ Pattern on dinh — {n} tin hieu OOS trai deu, khong bi cum")
        lines.append(f"  ✅ WR {overall_wr:.0f}%  Exp {overall_exp:+.1f}%  — du dieu kien deploy")
        lines.append(f"  → Giu nguyen config hien tai, sizing 100%")
    else:
        lines.append(f"  ⚠️  Phat hien {len(issues)} van de:")
        for iss in issues:
            lines.append(f"     - {iss}")
        lines.append(f"  Goi y:")
        for sug in suggestions:
            lines.append(f"     → {sug}")

    lines.append(sep)
    lines.append("Day la phan tich OOS — khong thay doi config, chi dung de hieu pattern.")
    return "\n".join(lines)


_last_regime_analysis: dict[str, float] = {}
REGIME_ANALYSIS_COOLDOWN = 180   # 3 phút

async def analog_regime_analysis_cmd(update, context):
    """
    /analog_regime_analysis <MA>

    Phân tích phân bố 42 tín hiệu OOS theo:
      - Clustering theo tháng
      - Hiệu suất theo volatility (ATR)
      - Hiệu suất theo trend regime (slope)
      - Khoảng cách giữa các tín hiệu

    Mục đích: hiểu khi nào pattern hoạt động tốt/kém
    để quyết định sizing và filter phù hợp.

    Thời gian: ~2 phút (chạy lại walk-forward).
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Cu phap: /analog_regime_analysis <MA>\n\n"
            "Vi du:\n"
            "  /analog_regime_analysis MWG\n"
            "  /analog_regime_analysis HPG\n\n"
            "Phan tich phan bo 42 tin hieu OOS theo thang, volatility, trend.\n"
            "Thoi gian: ~2 phut.\n\n"
            f"Ma co config: {', '.join(_WF_SYMBOL_CONFIG)}"
        )
        return

    import re as _re
    symbol = args[0].upper().strip()
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol):
        await update.message.reply_text(f"Ma khong hop le: {symbol}")
        return

    if symbol not in _WF_SYMBOL_CONFIG:
        await update.message.reply_text(
            f"{symbol} chua co config.\n"
            f"Chay /walkforward_analog {symbol} truoc.\n\n"
            f"Ma co config: {', '.join(_WF_SYMBOL_CONFIG)}"
        )
        return

    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_regime_analysis.get(user_id, 0)
    if since < REGIME_ANALYSIS_COOLDOWN:
        wait = int(REGIME_ANALYSIS_COOLDOWN - since)
        await update.message.reply_text(f"Vui long cho {wait}s truoc khi chay tiep.")
        return
    _last_regime_analysis[user_id] = time.time()

    cfg = _WF_SYMBOL_CONFIG[symbol]
    chat_id = update.effective_chat.id
    msg = await update.message.reply_text(
        f"Regime Analysis: {symbol}\n"
        f"Config: {cfg['combo']} | nguong {cfg['threshold']}\n"
        f"Dang chay walk-forward de lay tin hieu OOS (~2 phut)..."
    )

    async def _bg():
        try:
            # Chạy WF để lấy oos_signals + df
            res = await asyncio.to_thread(_run_walkforward_sync_with_df, symbol)

            if res["status"] == "error":
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    text=f"❌ {symbol}: {res.get('error','?')[:300]}"
                )
                return

            oos_sigs = res.get("oos_signals", [])
            completed = [s for s in oos_sigs if not s.get("pending")]
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg.message_id,
                text=f"Da lay {len(completed)} tin hieu OOS. Dang phan tich regime..."
            )

            text = _analyze_oos_regime(res)
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg.message_id,
                text=_plain(text)[:4096],
            )
        except Exception as e:
            import traceback
            logger.error(f"analog_regime_analysis_cmd error: {e}\n{traceback.format_exc()}")
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    text=f"Loi phan tich: {str(e)[:200]}"
                )
            except Exception:
                pass

    asyncio.create_task(_bg())


def _run_walkforward_sync_with_df(symbol: str) -> dict:
    """
    Wrapper _run_walkforward_sync — thêm df vào return dict
    để _analyze_oos_regime có thể tính state_vector.
    """
    import numpy as np
    import pandas as pd

    symbol = symbol.upper()
    if symbol not in _WF_SYMBOL_CONFIG:
        return {"status": "error", "error": f"{symbol} chua co config"}

    # Load df trước
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu"}

    # Chạy WF bình thường
    res = _run_walkforward_sync(symbol)

    # Đính kèm df vào result để _analyze_oos_regime dùng
    if res.get("status") == "ok":
        res["df"] = df

    return res


# ══════════════════════════════════════════════════════════════════════════════
# SIMILARITY DISTRIBUTION — chẩn đoán ngưỡng threshold
# ══════════════════════════════════════════════════════════════════════════════

def _run_sim_distribution_sync(symbol: str) -> dict:
    """
    Tính distribution cosine similarity của ngày hôm nay vs toàn bộ lịch sử.
    Mục đích: chẩn đoán xem threshold có quá dễ pass không.
    """
    import numpy as np
    import pandas as pd

    symbol = symbol.upper()
    if symbol not in _WF_SYMBOL_CONFIG:
        return {"status": "error", "error": f"{symbol} chua co config"}

    cfg       = _WF_SYMBOL_CONFIG[symbol]
    combo_name = cfg["combo"]
    threshold  = cfg["threshold"]

    combo = _find_combo(combo_name)
    if combo is None:
        return {"status": "error", "error": f"Khong tim thay combo '{combo_name}'"}
    dims = combo["dims"]

    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=1800, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Load data fail: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu"}

    try:
        from state_vector import compute_state_vector_from_df, compute_state_vector_for_date
    except ImportError as e:
        return {"status": "error", "error": f"Thieu state_vector: {e}"}

    # Vector ngày hôm nay
    target_vec = compute_state_vector_from_df(df)
    if target_vec is None:
        return {"status": "error", "error": "Khong tinh duoc vector hom nay"}

    target_arr = np.array([target_vec.get(d, 0.0) for d in dims], dtype=float)
    t_norm     = np.linalg.norm(target_arr)
    if t_norm < 1e-9:
        return {"status": "error", "error": "Vector hom nay = 0"}

    n_bars         = len(df)
    exclude_cutoff = n_bars - 90
    dates          = df["date"].values

    # Tính similarity tất cả ngày lịch sử
    all_sims  = []
    all_dates = []
    for i in range(59, exclude_cutoff):
        vec = compute_state_vector_for_date(df, i)
        if vec is None:
            continue
        arr  = np.array([vec.get(d, 0.0) for d in dims], dtype=float)
        norm = np.linalg.norm(arr)
        if norm < 1e-9:
            continue
        sim = float(np.dot(target_arr, arr) / (t_norm * norm))
        all_sims.append(sim)
        all_dates.append(str(dates[i])[:10])

    if not all_sims:
        return {"status": "error", "error": "Khong tinh duoc similarity"}

    sims = np.array(all_sims)
    n_total = len(sims)

    # % pass từng mức threshold
    thresholds_test = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    pct_pass = {t: float((sims >= t).mean() * 100) for t in thresholds_test}

    # Sau MDS 30 ngày: đếm analog độc lập thực sự pass threshold hiện tại
    pass_indices = [i for i, s in enumerate(all_sims) if s >= threshold]
    kept_mds = []
    for idx in sorted(pass_indices, key=lambda i: -all_sims[i]):
        d = pd.Timestamp(all_dates[idx])
        too_close = any(
            abs((d - pd.Timestamp(all_dates[k])).days) < 30
            for k in kept_mds
        )
        if not too_close:
            kept_mds.append(idx)

    # Phân bố sim theo bucket
    buckets = {}
    for lo in [x/10 for x in range(0, 10)]:
        hi  = lo + 0.1
        cnt = int(((sims >= lo) & (sims < hi)).sum())
        buckets[f"{lo:.1f}-{hi:.1f}"] = cnt

    return {
        "status":       "ok",
        "symbol":       symbol,
        "combo":        combo_name,
        "threshold":    threshold,
        "n_dims":       len(dims),
        "n_total":      n_total,
        "sim_median":   round(float(np.median(sims)), 3),
        "sim_mean":     round(float(np.mean(sims)), 3),
        "sim_p25":      round(float(np.percentile(sims, 25)), 3),
        "sim_p75":      round(float(np.percentile(sims, 75)), 3),
        "sim_p90":      round(float(np.percentile(sims, 90)), 3),
        "sim_p95":      round(float(np.percentile(sims, 95)), 3),
        "pct_pass":     pct_pass,
        "n_pass_raw":   int((sims >= threshold).sum()),
        "n_pass_mds":   len(kept_mds),
        "buckets":      buckets,
        "dims":         dims,
    }


def _format_sim_distribution(res: dict) -> str:
    """Format kết quả sim distribution thành Telegram message."""
    import numpy as np

    symbol    = res["symbol"]
    combo     = res["combo"]
    threshold = res["threshold"]
    n_total   = res["n_total"]
    n_dims    = res["n_dims"]
    sep  = "=" * 36
    sep2 = "-" * 36

    lines = [
        f"SIM DISTRIBUTION — {symbol}",
        f"Combo: {combo} ({n_dims} chieu) | nguong {threshold}",
        f"Tong ngay lich su: {n_total}",
        sep,
    ]

    # Phân bố similarity
    lines.append("PHAN BO COSINE SIMILARITY:")
    lines.append(f"  Median : {res['sim_median']:.3f}")
    lines.append(f"  P25/P75: {res['sim_p25']:.3f} / {res['sim_p75']:.3f}")
    lines.append(f"  P90/P95: {res['sim_p90']:.3f} / {res['sim_p95']:.3f}")
    lines.append("")

    # Histogram đơn giản
    lines.append("HISTOGRAM (bucket 0.1):")
    buckets = res["buckets"]
    max_cnt = max(buckets.values()) if buckets else 1
    for rng, cnt in sorted(buckets.items()):
        bar_len = int(cnt / max_cnt * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        marker  = " ← nguong" if abs(float(rng.split("-")[0]) - threshold) < 0.05 else ""
        lines.append(f"  {rng}  {bar}  {cnt:>4}{marker}")
    lines.append(sep2)

    # % pass từng threshold
    lines.append("% NGAY PASS TUNG NGUONG (truoc MDS):")
    pct_pass = res["pct_pass"]
    for t, pct in sorted(pct_pass.items()):
        marker = " ◄ hien tai" if t == threshold else ""
        em     = "🔴" if pct > 30 else "⚠️" if pct > 15 else "✅"
        lines.append(f"  {t:.2f}: {pct:>5.1f}% ngay pass  {em}{marker}")
    lines.append(sep2)

    # Sau MDS
    n_pass_raw = res["n_pass_raw"]
    n_pass_mds = res["n_pass_mds"]
    pct_raw    = n_pass_raw / n_total * 100
    pct_mds    = n_pass_mds / n_total * 100

    lines.append(f"SAU KHI LOC:")
    lines.append(f"  Pass threshold  : {n_pass_raw:>4} ngay ({pct_raw:.1f}%)")
    lines.append(f"  Sau MDS 30 ngay : {n_pass_mds:>4} analog doc lap ({pct_mds:.1f}%)")
    lines.append(sep2)

    # Chẩn đoán
    lines.append("CHAN DOAN:")
    pct_at_threshold = pct_pass.get(threshold, 0)

    if pct_at_threshold > 30:
        lines.append(f"  🔴 THRESHOLD QUA THAP — {pct_at_threshold:.1f}% ngay pass")
        lines.append(f"     Gan nhu ngay nao cung co the ra signal")
        # Gợi ý threshold hợp lý hơn (chỉ 10-15% ngày pass)
        suggested = next(
            (t for t, p in sorted(pct_pass.items()) if p <= 15.0),
            None
        )
        if suggested:
            lines.append(f"     Goi y: tang nguong len {suggested:.2f} (chi {pct_pass[suggested]:.1f}% pass)")
    elif pct_at_threshold > 15:
        lines.append(f"  ⚠️  THRESHOLD HƠI RONG — {pct_at_threshold:.1f}% ngay pass")
        lines.append(f"     Signal kha pho bien, nen ket hop them dieu kien")
    else:
        lines.append(f"  ✅ THRESHOLD HOP LY — {pct_at_threshold:.1f}% ngay pass")
        lines.append(f"     Signal co tinh chon loc tot")

    lines.append("")
    lines.append(f"  Analog doc lap sau MDS: {n_pass_mds} ({pct_mds:.1f}% lich su)")
    if n_pass_mds < 20:
        lines.append(f"  ⚠️  It analog — co the threshold qua cao, n < 20")
    elif n_pass_mds > 100:
        lines.append(f"  ⚠️  Qua nhieu analog — signal khong co tinh chon loc")
    else:
        lines.append(f"  ✅ So analog hop ly cho backtest co y nghia thong ke")

    lines.append(sep)
    return "\n".join(lines)


_last_sim_dist: dict[str, float] = {}
SIM_DIST_COOLDOWN = 120

async def analog_sim_dist_cmd(update, context):
    """
    /analog_sim_dist <MA>

    Chẩn đoán threshold: cosine similarity của hôm nay vs lịch sử.
    Trả lời câu hỏi: threshold có quá thấp không?
    Bao nhiêu % ngày lịch sử pass threshold hiện tại?
    Thời gian: ~1 phút.
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Cu phap: /analog_sim_dist <MA>\n\n"
            "Vi du:\n"
            "  /analog_sim_dist MWG\n"
            "  /analog_sim_dist DPM\n\n"
            "Chan doan: threshold co qua thap khong?\n"
            "Bao nhieu % ngay lich su pass threshold hien tai?\n\n"
            f"Ma co config: {', '.join(_WF_SYMBOL_CONFIG)}"
        )
        return

    import re as _re
    symbol = args[0].upper().strip()
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol):
        await update.message.reply_text(f"Ma khong hop le: {symbol}")
        return

    if symbol not in _WF_SYMBOL_CONFIG:
        await update.message.reply_text(
            f"{symbol} chua co config.\n"
            f"Ma co config: {', '.join(_WF_SYMBOL_CONFIG)}"
        )
        return

    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_sim_dist.get(user_id, 0)
    if since < SIM_DIST_COOLDOWN:
        wait = int(SIM_DIST_COOLDOWN - since)
        await update.message.reply_text(f"Vui long cho {wait}s truoc khi chay tiep.")
        return
    _last_sim_dist[user_id] = time.time()

    cfg     = _WF_SYMBOL_CONFIG[symbol]
    chat_id = update.effective_chat.id
    msg     = await update.message.reply_text(
        f"Sim Distribution: {symbol}\n"
        f"Combo: {cfg['combo']} | nguong {cfg['threshold']}\n"
        f"Dang tinh similarity vs {1800} ngay lich su (~1 phut)..."
    )

    async def _bg():
        try:
            res  = await asyncio.to_thread(_run_sim_distribution_sync, symbol)
            if res["status"] == "error":
                text = f"❌ {symbol}: {res.get('error','?')[:300]}"
            else:
                text = _format_sim_distribution(res)
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg.message_id,
                text=_plain(text)[:4096],
            )
        except Exception as e:
            import traceback
            logger.error(f"analog_sim_dist_cmd error: {e}\n{traceback.format_exc()}")
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=f"Loi: {str(e)[:200]}"
                )
            except Exception:
                pass

    asyncio.create_task(_bg())


async def walkforward_analog_cmd(update, context):
    """
    /walkforward_analog [MA1 MA2 ...]

    Walk-forward OOS test tu 2025-01-01 den hom nay.

    - Ma da co config (_WF_SYMBOL_CONFIG): chay nhanh voi config co san.
    - Ma CHUA co config: tu dong chay Pipeline (Tang 1 → Tang 2) de tim
      combo + threshold tot nhat, sau do luu vao config va chay WF.
      Khong can nhờ setup tay nua.

    Vi du:
      /walkforward_analog              (chay tat ca ma co config)
      /walkforward_analog DCM          (tu dong pipeline neu chua co config)
      /walkforward_analog FPT MWG DCM  (hon hop: co san + chua co config)
      /walkforward_analog VNM ACB      (ma moi hoan toan)
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []

    # Nếu không có args → chạy tất cả mã có config
    if not args:
        symbols = list(_WF_SYMBOL_CONFIG.keys())
    else:
        import re as _re
        symbols = [a.upper() for a in args if _re.match(r'^[A-Z0-9]{2,10}$', a.upper())]

    if not symbols:
        await update.message.reply_text(
            "Cu phap: /walkforward_analog [MA1 MA2 ...]\n\n"
            "Vi du:\n"
            "  /walkforward_analog              (tat ca ma co config)\n"
            "  /walkforward_analog DCM          (tu dong tim config neu chua co)\n"
            "  /walkforward_analog MWG STB DCM\n\n"
            f"Ma da co config: {', '.join(_WF_SYMBOL_CONFIG)}"
        )
        return

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_backtest_analog_wf.get(user_id, 0)
    if since < BACKTEST_ANALOG_WF_COOLDOWN:
        wait = int(BACKTEST_ANALOG_WF_COOLDOWN - since)
        await update.message.reply_text(f"Vui long cho {wait}s truoc khi chay tiep.")
        return
    _last_backtest_analog_wf[user_id] = time.time()

    # Phân loại mã: có config sẵn vs cần tự động pipeline
    known   = [s for s in symbols if s in _WF_SYMBOL_CONFIG]
    unknown = [s for s in symbols if s not in _WF_SYMBOL_CONFIG]

    info_parts = []
    if known:
        info_parts.append(f"Config san: {', '.join(known)}")
    if unknown:
        info_parts.append(f"Tu dong Pipeline (Tang 1+2): {', '.join(unknown)}")

    chat_id  = update.effective_chat.id
    est_mins = len(known) * 2 + len(unknown) * 10
    msg = await update.message.reply_text(
        f"Walk-forward OOS: {', '.join(symbols)}\n"
        + "\n".join(info_parts) + "\n"
        f"Giai doan: {_WF_START_DATE} → hom nay\n"
        f"Uoc tinh ~{est_mins} phut..."
    )

    async def _bg():
        try:
            all_results = {}
            for i, symbol in enumerate(symbols, 1):
                is_unknown = symbol not in _WF_SYMBOL_CONFIG

                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg.message_id,
                        text=(
                            f"Walk-forward: {symbol} ({i}/{len(symbols)})\n"
                            + (
                                f"Tu dong Pipeline: Tim combo + threshold tot nhat...\n"
                                if is_unknown else
                                f"Config: {_WF_SYMBOL_CONFIG[symbol]['combo']} "
                                f"| nguong {_WF_SYMBOL_CONFIG[symbol]['threshold']}\n"
                            )
                            + f"Dang tinh..."
                        ),
                    )
                except Exception:
                    pass

                if is_unknown:
                    # Chạy pipeline tự động: Tầng 1 tìm config → Tầng 2 WF
                    res = await asyncio.to_thread(_run_pipeline_wf_sync, symbol)
                else:
                    res = await asyncio.to_thread(_run_walkforward_sync, symbol)

                all_results[symbol] = res

                # Gửi kết quả từng mã
                if res["status"] == "error":
                    text = f"❌ {symbol}: {res.get('error','?')[:200]}"
                elif res["status"] == "skip":
                    text = (
                        f"⛔ {symbol}: Khong co combo nao qua bo loc\n"
                        f"   {res.get('reason','?')[:150]}"
                    )
                else:
                    text = _format_wf_result(res)

                if i == 1 and len(symbols) == 1:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg.message_id,
                        text=_plain(text)[:4096],
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=_plain(text)[:4096],
                    )

            # Tổng kết nếu nhiều mã
            if len(symbols) > 1:
                summary_lines = [
                    "TONG KET WALK-FORWARD",
                    f"Giai doan: {_WF_START_DATE} → hom nay",
                    "=" * 32,
                ]
                pass_syms    = []
                partial_syms = []
                fail_syms    = []

                for symbol, res in all_results.items():
                    if res["status"] in ("error", "skip"):
                        fail_syms.append(f"{symbol}({'loi' if res['status']=='error' else 'skip'})")
                        continue
                    oos_m = res.get("oos_metrics")
                    if not oos_m:
                        fail_syms.append(f"{symbol}(chua du tin hieu)")
                        continue
                    train_m = res.get("train_metrics", {}) or {}
                    exp_ok  = oos_m["mean_exp"] > 0
                    pf_ok   = oos_m["pf"] >= 1.5
                    exp_deg = oos_m["mean_exp"] >= (train_m.get("mean_exp", 0) or 0) * 0.60
                    cfg_tag = "" if symbol in _WF_SYMBOL_CONFIG else " [auto]"
                    line    = (
                        f"  {symbol}{cfg_tag}: "
                        f"Exp {oos_m['mean_exp']:+.2f}%  "
                        f"PF {oos_m['pf']:.2f}  "
                        f"WR {oos_m['wr']:.0f}%  "
                        f"n={oos_m['n']}"
                    )
                    if exp_ok and pf_ok and exp_deg:
                        summary_lines.append(f"✅ PASS   {line[4:]}")
                        pass_syms.append(symbol)
                    elif exp_ok and pf_ok:
                        summary_lines.append(f"🟡 PARTIAL{line[4:]}")
                        partial_syms.append(symbol)
                    else:
                        summary_lines.append(f"🔴 FAIL   {line[4:]}")
                        fail_syms.append(symbol)

                summary_lines.append("-" * 32)
                if pass_syms:
                    summary_lines.append(f"PASS    : {', '.join(pass_syms)}")
                if partial_syms:
                    summary_lines.append(f"PARTIAL : {', '.join(partial_syms)}")
                if fail_syms:
                    summary_lines.append(f"FAIL/SKIP: {', '.join(fail_syms)}")

                # Gợi ý lưu config cho mã auto pass
                auto_pass = [s for s in pass_syms if s not in _WF_SYMBOL_CONFIG]
                if auto_pass:
                    summary_lines.append("")
                    summary_lines.append(
                        f"💡 Ma moi PASS: {', '.join(auto_pass)}\n"
                        f"   Ket qua da luu tam thoi. "
                        f"Dung /analog_pipeline de xac nhan truoc khi them vao live."
                    )

                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    text=_plain("\n".join(summary_lines))[:4096],
                )

        except Exception as e:
            import traceback
            logger.error(f"walkforward_analog_cmd error: {e}\n{traceback.format_exc()}")
            err_text = f"Loi /walkforward_analog: {str(e)[:200]}"
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id, text=err_text,
                )
            except Exception:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=err_text)
                except Exception:
                    pass

    asyncio.create_task(_bg())


# ══════════════════════════════════════════════════════════════════════════════
# ANALOG PIPELINE — Tầng 1 + Tầng 2 tự động cho bất kỳ mã nào
# ══════════════════════════════════════════════════════════════════════════════

ANALOG_PIPELINE_COOLDOWN = 600   # 10 phút per user
_last_analog_pipeline: dict[str, float] = {}
PIPELINE_MAX_SYMBOLS = 5

# Ngưỡng lọc mã không phù hợp — dựa trên kinh nghiệm batch
_PIPELINE_MIN_EXP    = 0.0    # Exp phải dương
_PIPELINE_MIN_PF     = 1.5    # PF tối thiểu
_PIPELINE_MAX_DD     = -60.0  # MaxDD tệ hơn -60% → bỏ qua (như SSI)


def _run_pipeline_sync(symbol: str, days: int = 1800) -> dict:
    """
    Pipeline đầy đủ cho 1 mã:
      Tầng 1: Backtest 105 experiments → chọn top combo + threshold tối ưu
      Tầng 2: Walk-forward OOS 2025→nay với config từ Tầng 1
    Trả về dict tổng hợp cả 2 tầng.
    """
    import numpy as np

    # ── Tầng 1: Backtest ─────────────────────────────────────────────────────
    bt_res = _run_analog_backtest_sync(symbol, days)
    if bt_res.get("status") == "error":
        return {"status": "error", "stage": "backtest", "error": bt_res["error"]}

    valid = bt_res.get("top_results", [])
    if not valid:
        return {"status": "skip", "reason": "Khong co experiment nao hop le"}

    # Lọc theo bộ lọc cứng Exp + PF
    pass_results = [
        r for r in valid
        if r["mean_exp"] > _PIPELINE_MIN_EXP
        and r["pf"] >= _PIPELINE_MIN_PF
        and r["max_dd"] >= _PIPELINE_MAX_DD
    ]

    if not pass_results:
        # Lấy top dù không pass để báo lý do skip
        top = valid[0]
        reason = []
        if top["mean_exp"] <= 0:
            reason.append(f"Exp {top['mean_exp']:+.2f}% <= 0")
        if top["pf"] < _PIPELINE_MIN_PF:
            reason.append(f"PF {top['pf']:.2f} < 1.5")
        if top["max_dd"] < _PIPELINE_MAX_DD:
            reason.append(f"MaxDD {top['max_dd']:.1f}% < -60%")
        return {
            "status": "skip",
            "reason": " | ".join(reason),
            "top_exp": top["mean_exp"],
            "top_pf":  top["pf"],
            "top_dd":  top["max_dd"],
        }

    # Best combo: Exp → PF → Sharpe tiebreak (đã sort trong backtest)
    best_bt = pass_results[0]
    combo_name = best_bt["combo"]

    # ── Tầng 1b: Detail — tìm threshold tối ưu ───────────────────────────────
    detail_res = _run_analog_detail_sync(symbol, combo_name, days)
    if detail_res.get("status") == "error":
        # Fallback: dùng threshold từ backtest
        best_threshold = best_bt["threshold"]
        detail_ok      = False
    else:
        detail_ok  = True
        thresh_rows = [
            r for r in detail_res.get("threshold_results", [])
            if not r.get("skip")
            and r["mean_exp"] > _PIPELINE_MIN_EXP
            and r["pf"] >= _PIPELINE_MIN_PF
            and r["max_dd"] >= _PIPELINE_MAX_DD
        ]
        if thresh_rows:
            # Chọn threshold: Exp cao nhất, PF làm tiebreak — nhất quán với ranking T1
            best_thresh_row = max(thresh_rows, key=lambda x: (x["mean_exp"], x["pf"]))
            best_threshold  = best_thresh_row["threshold"]
            best_bt_detail  = best_thresh_row
        else:
            best_threshold = best_bt["threshold"]
            best_bt_detail = best_bt

    # ── Tầng 2: Walk-forward với config từ Tầng 1 ────────────────────────────
    # Tạo config tạm thời cho mã này mà không cần sửa _WF_SYMBOL_CONFIG
    import pandas as pd

    combo  = _find_combo(combo_name)
    if combo is None:
        return {"status": "error", "stage": "wf", "error": f"Khong tim thay combo {combo_name}"}

    dims      = combo["dims"]
    threshold = best_threshold

    # Fetch data
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=200)
    except Exception as e:
        return {"status": "error", "stage": "wf_data", "error": str(e)[:100]}

    if df is None or len(df) < 200:
        return {"status": "error", "stage": "wf_data", "error": "Khong du du lieu"}

    # Tính vectors
    try:
        from state_vector import compute_state_vector_for_date
    except ImportError as e:
        return {"status": "error", "stage": "wf_vec", "error": str(e)[:100]}

    vectors = {}
    for i in range(59, len(df)):
        vec = compute_state_vector_for_date(df, i)
        if vec is not None:
            vectors[i] = vec

    n_bars         = len(df)
    dates          = df["date"].values
    close_arr      = df["close"].values.astype(float)
    low_arr        = df["low"].values.astype(float)
    vector_indices = sorted(vectors.keys())
    date_series    = pd.to_datetime(df["date"])

    wf_start      = pd.Timestamp(_WF_START_DATE)
    oos_start_idx = next(
        (i for i, d in enumerate(date_series) if pd.Timestamp(d) >= wf_start), None
    )
    if oos_start_idx is None:
        return {"status": "error", "stage": "wf", "error": f"Khong co data sau {_WF_START_DATE}"}

    WIN_THRESH  = 1.0
    MIN_SAMPLES = 5
    # MDS_DAYS: 43 calendar days ≈ 30 trading days
    MDS_DAYS    = 43
    FWD_DAYS    = 30
    # LOOP_STEP=1, COOLDOWN_BARS=5: đồng bộ với _run_walkforward_sync
    # 5 bars ≈ 7 calendar days = SIGNAL_COOLDOWN_DAYS trong live trading
    LOOP_STEP     = 1
    COOLDOWN_BARS = 5

    oos_signals          = []
    train_signals        = []
    last_oos_signal_idx  = None

    for t_idx in range(120, n_bars - FWD_DAYS - 1, LOOP_STEP):
        if t_idx not in vectors:
            continue
        t_date = pd.Timestamp(dates[t_idx])
        is_oos = t_date >= wf_start

        # Simulate cooldown OOS — dùng bar index
        if is_oos and last_oos_signal_idx is not None:
            if t_idx - last_oos_signal_idx < COOLDOWN_BARS:
                continue

        target_vec = vectors[t_idx]

        # Hard filter — lọc ngày T không đúng regime của combo
        if not _check_hard_filter(target_vec, combo_name):
            continue

        target_arr = np.array([target_vec.get(d, 0.0) for d in dims], dtype=float)
        t_norm     = np.linalg.norm(target_arr)
        if t_norm < 1e-9:
            continue

        exclude_cutoff = min(t_idx - 90, oos_start_idx - 1)
        candidates     = [i for i in vector_indices if i < exclude_cutoff]
        if len(candidates) < 10:
            continue

        sim_list = []
        for c_idx in candidates:
            c_vec  = vectors[c_idx]
            c_arr  = np.array([c_vec.get(d, 0.0) for d in dims], dtype=float)
            c_norm = np.linalg.norm(c_arr)
            if c_norm < 1e-9:
                continue
            sim = float(np.dot(target_arr, c_arr) / (t_norm * c_norm))
            if sim >= threshold:
                sim_list.append((c_idx, sim))

        if not sim_list:
            continue

        sim_list.sort(key=lambda x: -x[1])
        kept = []
        for c_idx, sim in sim_list:
            c_date    = pd.Timestamp(dates[c_idx])
            # MDS: 43 calendar days ≈ 30 trading days
            too_close = any(
                abs((c_date - pd.Timestamp(dates[k])).days) < MDS_DAYS
                for k, _ in kept
            )
            if not too_close:
                kept.append((c_idx, sim))

        if len(kept) < MIN_SAMPLES:
            continue

        fwd_rets = []
        mae_vals = []
        for c_idx, _ in kept:
            fwd_idx = c_idx + FWD_DAYS
            if fwd_idx >= n_bars:
                continue
            entry = close_arr[c_idx]
            fwd_rets.append((close_arr[fwd_idx] - entry) / entry * 100)
            win_low = np.min(low_arr[c_idx + 1: fwd_idx + 1]) if fwd_idx > c_idx else entry
            mae_vals.append((win_low - entry) / entry * 100)

        if len(fwd_rets) < MIN_SAMPLES:
            continue

        actual_fwd_idx = t_idx + FWD_DAYS
        pending        = actual_fwd_idx >= n_bars
        actual         = None if pending else (
            (close_arr[actual_fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100
        )

        sig = {
            "t_date":    str(t_date)[:10],
            "predicted": float(np.median(fwd_rets)),
            "actual":    actual,
            "mae_vals":  mae_vals,
            "pending":   pending,
        }
        if is_oos:
            oos_signals.append(sig)
            last_oos_signal_idx = t_idx   # cập nhật cooldown tracker
        else:
            train_signals.append(sig)

    # Tính metrics
    def _calc(signals):
        done = [s for s in signals if not s.get("pending") and s["actual"] is not None]
        if len(done) < 3:
            return None
        actuals  = [s["actual"] for s in done]
        wins     = [x for x in actuals if x >= WIN_THRESH]
        losses   = [x for x in actuals if x < WIN_THRESH]
        wr       = len(wins) / len(actuals)
        mean_exp = float(np.mean(actuals))
        std_ret  = float(np.std(actuals)) if len(actuals) > 1 else 1e-9
        sharpe   = mean_exp / std_ret * (52 ** 0.5) if std_ret > 0 else 0.0
        pos_sum  = sum(wins)
        neg_sum  = abs(sum(losses)) if losses else 1e-9
        pf       = pos_sum / neg_sum if neg_sum > 0 else 99.0
        all_maes = [float(np.median(s["mae_vals"])) for s in done if s.get("mae_vals")]
        mae30    = float(np.median(all_maes)) if all_maes else 0.0
        # Worst Signal = return tệ nhất của 1 tín hiệu (metric rủi ro đúng)
        max_dd   = float(np.min(actuals)) if actuals else 0.0
        return {
            "n":         len(done),
            "n_pending": len([s for s in signals if s.get("pending")]),
            "wr":        round(wr * 100, 1),
            "mean_exp":  round(mean_exp, 2),
            "sharpe":    round(sharpe, 3),
            "pf":        round(pf, 2),
            "mae30":     round(mae30, 2),
            "max_dd":    round(max_dd, 1),
        }

    oos_m   = _calc(oos_signals)
    train_m = _calc(train_signals)

    # Verdict
    verdict = "unknown"
    if oos_m:
        exp_ok  = oos_m["mean_exp"] > 0
        pf_ok   = oos_m["pf"] >= 1.5
        exp_deg = oos_m["mean_exp"] >= (train_m["mean_exp"] if train_m else 0) * 0.60
        pf_deg  = oos_m["pf"] >= (train_m["pf"] if train_m else 0) * 0.50
        if exp_ok and pf_ok and exp_deg and pf_deg:
            verdict = "pass"
        elif exp_ok and pf_ok:
            verdict = "partial"
        else:
            verdict = "fail"
    elif len(oos_signals) > 0:
        verdict = "pending"   # có tín hiệu nhưng chưa đủ 30 ngày

    return {
        "status":       "ok",
        "symbol":       symbol,
        "combo":        combo_name,
        "threshold":    threshold,
        "verdict":      verdict,
        # Tầng 1 summary
        "t1_exp":       best_bt["mean_exp"],
        "t1_pf":        best_bt["pf"],
        "t1_max_dd":    best_bt["max_dd"],
        "t1_wr":        best_bt["wr"],
        "t1_n":         best_bt["n_signals"],
        # Tầng 2 OOS
        "oos_metrics":  oos_m,
        "train_metrics":train_m,
        "n_oos_sigs":   len(oos_signals),
        "n_pending":    sum(1 for s in oos_signals if s.get("pending")),
    }


def _format_pipeline_summary(results: dict) -> str:
    """Format tổng kết pipeline ngắn gọn — 1 dòng/mã + verdict."""
    from collections import Counter

    sep  = "=" * 40
    sep2 = "-" * 40

    pass_syms    = []
    partial_syms = []
    fail_syms    = []
    skip_syms    = []
    pending_syms = []

    lines = [
        "ANALOG PIPELINE — KET QUA",
        f"Giai doan OOS: {_WF_START_DATE} → hom nay",
        sep,
        f"{'Ma':<6} {'Combo':<22} {'Thr':<5} {'T1:Exp':>7} {'T1:PF':>6} {'T1:DD':>7} {'OOS:Exp':>8} {'OOS:PF':>7} {'OOS:n':>6}  Verdict",
        sep2,
    ]

    for symbol, res in results.items():
        if res["status"] == "error":
            lines.append(
                f"{'❌'}{symbol:<5} LOI: {res.get('error','?')[:50]}"
            )
            skip_syms.append(symbol)
            continue

        if res["status"] == "skip":
            lines.append(
                f"{'⛔'}{symbol:<5} SKIP: {res.get('reason','?')[:50]}"
            )
            skip_syms.append(symbol)
            continue

        combo_short = res["combo"][:20]
        thr         = res["threshold"]
        t1_exp      = res["t1_exp"]
        t1_pf       = res["t1_pf"]
        t1_dd       = res["t1_max_dd"]
        oos_m       = res.get("oos_metrics")
        verdict     = res["verdict"]

        # Emoji verdict
        em_map = {
            "pass":    "✅",
            "partial": "🟡",
            "fail":    "🔴",
            "pending": "⏳",
            "unknown": "❓",
        }
        em = em_map.get(verdict, "❓")

        oos_exp = f"{oos_m['mean_exp']:+.2f}%" if oos_m else "—"
        oos_pf  = f"{oos_m['pf']:.2f}"         if oos_m else "—"
        oos_n   = f"n={oos_m['n']}"             if oos_m else f"sig={res['n_oos_sigs']}"
        if res.get("n_pending", 0) > 0 and oos_m:
            oos_n += f"+{res['n_pending']}p"

        lines.append(
            f"{em}{symbol:<5} "
            f"{combo_short:<22} "
            f"{thr:<5} "
            f"{t1_exp:>+7.2f}% "
            f"{t1_pf:>6.2f} "
            f"{t1_dd:>6.1f}%  "
            f"{oos_exp:>8} "
            f"{oos_pf:>7} "
            f"{oos_n:>8}  "
            f"{verdict.upper()}"
        )

        if verdict == "pass":
            pass_syms.append(symbol)
        elif verdict == "partial":
            partial_syms.append(symbol)
        elif verdict == "fail":
            fail_syms.append(symbol)
        elif verdict == "pending":
            pending_syms.append(symbol)

    lines.append(sep2)
    lines.append("Columns: T1=Backtest in-sample | OOS=Walk-forward 2025→nay")
    lines.append(sep2)

    # Tổng kết
    if pass_syms:
        lines.append(f"✅ PASS    : {', '.join(pass_syms)}")
    if partial_syms:
        lines.append(f"🟡 PARTIAL : {', '.join(partial_syms)}")
    if fail_syms:
        lines.append(f"🔴 FAIL    : {', '.join(fail_syms)}")
    if pending_syms:
        lines.append(f"⏳ PENDING : {', '.join(pending_syms)} (chua du 30 ngay)")
    if skip_syms:
        lines.append(f"⛔ SKIP    : {', '.join(skip_syms)}")

    lines.append("")
    if pass_syms:
        lines.append(f"→ Du dieu kien live trading: {', '.join(pass_syms)}")
        lines.append("  SL goi y: MAE30 - 2% (xem chi tiet tung ma)")
    if partial_syms:
        lines.append(f"→ Theo doi them: {', '.join(partial_syms)}")
    if not pass_syms and not partial_syms:
        lines.append("→ Chua co ma nao pass OOS. Nen cho them data 2025.")

    lines.append(sep)
    return "\n".join(lines)


async def analog_pipeline_cmd(update, context):
    """
    /analog_pipeline MA1 MA2 MA3...

    Tu dong chay Tang 1 (backtest + detail) va Tang 2 (walk-forward OOS)
    cho bat ky ma nao. Khong can chay thu cong tung buoc.

    Vi du:
      /analog_pipeline FPT MWG STB HPG
      /analog_pipeline VCB TCB
      /analog_pipeline DPM HHS
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []

    if not args:
        await update.message.reply_text(
            "Cu phap: /analog_pipeline MA1 MA2 MA3...\n\n"
            "Vi du:\n"
            "  /analog_pipeline FPT MWG STB HPG\n"
            "  /analog_pipeline DPM HHS VNM\n\n"
            f"Toi da {PIPELINE_MAX_SYMBOLS} ma. Thoi gian ~8-12 phut/ma.\n"
            "Tu dong chay: Backtest → Detail → Walk-forward OOS.\n"
            "Ket qua: bang tong hop PASS/FAIL + Exp/PF/MaxDD."
        )
        return

    import re as _re
    symbols = [a.upper() for a in args if _re.match(r'^[A-Z0-9]{2,10}$', a.upper())]

    if not symbols:
        await update.message.reply_text("Khong tim thay ma hop le.")
        return

    if len(symbols) > PIPELINE_MAX_SYMBOLS:
        await update.message.reply_text(
            f"Toi da {PIPELINE_MAX_SYMBOLS} ma. Ban nhap {len(symbols)} ma.\n"
            f"Vui long chia thanh nhieu lenh."
        )
        return

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_analog_pipeline.get(user_id, 0)
    if since < ANALOG_PIPELINE_COOLDOWN:
        wait = int(ANALOG_PIPELINE_COOLDOWN - since)
        await update.message.reply_text(f"Vui long cho {wait}s truoc khi chay tiep.")
        return
    _last_analog_pipeline[user_id] = time.time()

    chat_id  = update.effective_chat.id
    est_mins = len(symbols) * 10

    msg = await update.message.reply_text(
        f"Analog Pipeline: {', '.join(symbols)}\n"
        f"Tang 1 (Backtest + Detail) → Tang 2 (Walk-forward OOS)\n"
        f"Uoc tinh ~{est_mins} phut. Vui long doi..."
    )

    async def _bg():
        results = {}
        try:
            for i, symbol in enumerate(symbols, 1):
                # Progress update
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg.message_id,
                        text=(
                            f"Pipeline: {symbol} ({i}/{len(symbols)})\n"
                            f"Tang 1: Backtest 105 experiments...\n"
                            f"Hoan thanh: {', '.join(symbols[:i-1]) or 'chua co'}\n"
                            f"Con lai: ~{(len(symbols)-i+1)*10} phut"
                        ),
                    )
                except Exception:
                    pass

                try:
                    res = await asyncio.to_thread(_run_pipeline_sync, symbol)
                except Exception as e:
                    res = {"status": "error", "error": str(e)[:100]}

                results[symbol] = res

            # Format và gửi tổng kết
            summary = _format_pipeline_summary(results)

            if len(symbols) == 1:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    text=_plain(summary)[:4096],
                )
            else:
                # Edit loading → tổng kết
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    text=_plain(summary)[:4096],
                )

        except Exception as e:
            import traceback
            logger.error(f"analog_pipeline_cmd error: {e}\n{traceback.format_exc()}")
            err_text = f"Loi /analog_pipeline: {str(e)[:200]}"
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id, text=err_text,
                )
            except Exception:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=err_text)
                except Exception:
                    pass

    asyncio.create_task(_bg())


# ══════════════════════════════════════════════════════════════════════════════
# /analog_approve — lưu config mới vào DB (không cần sửa code nữa)
# /analog_configs — xem tất cả config đang active
# /analog_remove  — xoá 1 mã khỏi config DB
# ══════════════════════════════════════════════════════════════════════════════

async def analog_approve_cmd(update, context):
    """
    /analog_approve <MA> [combo] [threshold] [mae30] [sizing]

    Lưu config analog của 1 mã vào DB — không cần sửa code nữa.

    Có 2 cách dùng:

    1. Sau khi /walkforward_analog tìm được config tự động (PASS):
       Bot đã gợi ý combo + threshold — chỉ cần approve:
         /analog_approve LPB

    2. Tự nhập tay (sau khi xem kết quả /backtest_analog_detail):
         /analog_approve LPB "Oversold Bounce" 0.60 -6.5 1.0
         /analog_approve TCB "Volume Confirmed" 0.75 -5.0 0.5

    Args:
        MA:        mã cổ phiếu
        combo:     tên combo (optional nếu đã chạy pipeline trước đó)
        threshold: ngưỡng cosine (optional)
        mae30:     MAE 30 ngày, số âm (optional, default 0)
        sizing:    hệ số vốn 0.5 hoặc 1.0 (optional, default 1.0)

    Sau khi approve:
      - Config được lưu vào DB (persist qua bot restart)
      - /walkforward_analog <MA> sẽ dùng config này ngay lập tức
      - Để đưa vào live signal cron, dùng /analog_approve rồi thêm
        dims thủ công vào SIGNAL_SYMBOLS trong analog_signal.py
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []
    if not args:
        combo_list = "\n".join(
            f"  {i:>2}. {c['name']}"
            for i, c in enumerate(_ANALOG_COMBOS)
        )
        await update.message.reply_text(
            "Cu phap:\n"
            "  /analog_approve <MA>                     (sau khi chay pipeline)\n"
            "  /analog_approve <MA> <index> <threshold> [mae30] [sizing]\n\n"
            "Vi du:\n"
            "  /analog_approve LPB\n"
            "  /analog_approve LPB 5 0.60\n"
            "  /analog_approve LPB 5 0.60 -6.5 0.5\n\n"
            f"Danh sach combo:\n{combo_list}\n\n"
            f"Config hien tai: {', '.join(_WF_SYMBOL_CONFIG.keys())}"
        )
        return

    import re as _re
    symbol = args[0].upper().strip()
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol):
        await update.message.reply_text(f"Ma khong hop le: {symbol}")
        return

    # Lấy combo + threshold từ args hoặc từ _WF_SYMBOL_CONFIG (đã được pipeline set)
    if len(args) >= 3:
        # Nhập tay: /analog_approve LPB 5 0.60 [-6.5] [0.5]
        # args[1] = index số hoặc tên combo (backward compat với dấu ")
        combo_arg = args[1]
        try:
            combo_idx = int(combo_arg)
            if combo_idx < 0 or combo_idx >= len(_ANALOG_COMBOS):
                await update.message.reply_text(
                    f"Index {combo_idx} khong hop le. "
                    f"Combo co so thu tu 0 den {len(_ANALOG_COMBOS)-1}.\n"
                    f"Goi /analog_approve de xem danh sach."
                )
                return
            combo = _ANALOG_COMBOS[combo_idx]["name"]
        except ValueError:
            # Backward compat: vẫn chấp nhận tên combo có dấu "
            import re
            full_args = " ".join(args[1:])
            quoted    = re.findall(r'"([^"]*)"', full_args)
            combo     = quoted[0] if quoted else combo_arg
        try:
            threshold = float(args[2]) if len(args) > 2 else 0.60
            mae30     = float(args[3]) if len(args) > 3 else 0.0
            sizing    = float(args[4]) if len(args) > 4 else 1.0
        except (ValueError, IndexError):
            await update.message.reply_text("threshold/mae30/sizing phai la so. Vi du: 5 0.60 -6.5 1.0")
            return
    elif symbol in _WF_SYMBOL_CONFIG:
        # Lấy từ config đã có trong memory (pipeline vừa tìm hoặc hardcode)
        cfg       = _WF_SYMBOL_CONFIG[symbol]
        combo     = cfg["combo"]
        threshold = cfg["threshold"]
        mae30     = cfg.get("mae30", 0.0)
        sizing    = cfg.get("sizing", 1.0)
    else:
        await update.message.reply_text(
            f"{symbol} chua co config trong memory.\n"
            f"Chay /walkforward_analog {symbol} truoc de pipeline tu dong tim config,\n"
            f"hoac nhap tay: /analog_approve {symbol} \"<combo>\" <threshold>"
        )
        return

    # Validate combo tồn tại
    combo_obj = _find_combo(combo)
    if combo_obj is None:
        combos_available = [c["name"] for c in _ANALOG_COMBOS]
        await update.message.reply_text(
            f"Combo '{combo}' khong ton tai.\n"
            f"Combo co san:\n" + "\n".join(f"  - {c}" for c in combos_available)
        )
        return

    dims = combo_obj["dims"]

    # Lưu vào DB
    try:
        from db import save_analog_config
        ok = save_analog_config(
            symbol=symbol, combo=combo, threshold=threshold,
            mae30=mae30, sizing=sizing, dims=dims,
        )
    except Exception as e:
        await update.message.reply_text(f"Loi luu DB: {e}")
        return

    if not ok:
        await update.message.reply_text(f"Luu DB that bai — kiem tra log.")
        return

    # Cập nhật memory ngay lập tức
    _WF_SYMBOL_CONFIG[symbol] = {"combo": combo, "threshold": threshold,
                                  "mae30": mae30, "sizing": sizing}

    sep = "─" * 32
    lines = [
        f"✅ DA LUU CONFIG: {symbol}",
        sep,
        f"  Combo     : {combo}",
        f"  Threshold : {threshold}",
        f"  MAE30     : {mae30:.1f}%",
        f"  Sizing    : {int(sizing*100)}%",
        f"  Dims ({len(dims)}): {', '.join(dims[:3])}{'...' if len(dims) > 3 else ''}",
        sep,
        f"Config da luu vao DB — persist qua bot restart.",
        f"/walkforward_analog {symbol} se dung config nay ngay.",
        "",
        f"Buoc tiep theo de them vao LIVE SIGNAL CRON:",
        f"  Them vao SIGNAL_SYMBOLS trong analog_signal.py:",
        f'  "{symbol}": {{',
        f'      "combo": "{combo}",',
        f'      "dims": {dims},',
        f'      "threshold": {threshold},',
        f'      "mae30": {mae30},',
        f'      "sizing": {sizing},',
        f'  }}',
    ]
    await update.message.reply_text("\n".join(lines)[:4096])


async def analog_configs_cmd(update, context):
    """
    /analog_configs
    Xem tất cả analog config đang active (hardcode + DB).
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    # Load fresh từ DB để hiển thị chính xác nhất
    try:
        from db import load_analog_configs
        db_configs = load_analog_configs()
    except Exception:
        db_configs = {}

    sep   = "─" * 36
    lines = [
        f"ANALOG CONFIGS ({len(_WF_SYMBOL_CONFIG)} ma)",
        f"DB: {len(db_configs)} ma | Hardcode: {len(_WF_SYMBOL_CONFIG_DEFAULT)} ma",
        sep,
    ]

    for symbol, cfg in sorted(_WF_SYMBOL_CONFIG.items()):
        source = "DB" if symbol in db_configs else "hardcode"
        mae30  = cfg.get("mae30", 0.0)
        sizing = cfg.get("sizing", 1.0)
        lines.append(
            f"{'📀' if source == 'DB' else '🔧'} {symbol:<5} [{source}]"
        )
        lines.append(
            f"   {cfg['combo']} | thr={cfg['threshold']}"
            + (f" | MAE={mae30:.1f}%" if mae30 else "")
            + (f" | size={int(sizing*100)}%" if sizing != 1.0 else "")
        )

    lines += [
        sep,
        "📀 = luu trong DB (persist)  🔧 = hardcode trong code",
        "",
        "Lenh quan ly:",
        "  /analog_approve <MA> [combo] [threshold] — them/sua config",
        "  /analog_remove <MA>                       — xoa config DB",
        "  /analog_configs                            — xem danh sach nay",
    ]
    await update.message.reply_text("\n".join(lines)[:4096])


async def analog_remove_cmd(update, context):
    """
    /analog_remove <MA>
    Xoá config của 1 mã khỏi DB. Không ảnh hưởng hardcode defaults.
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Cu phap: /analog_remove <MA>\n"
            "Vi du: /analog_remove LPB\n\n"
            "Chi xoa config luu trong DB.\n"
            "Neu ma co hardcode default, no se quay ve gia tri do."
        )
        return

    symbol = args[0].upper().strip()

    try:
        from db import delete_analog_config
        deleted = delete_analog_config(symbol)
    except Exception as e:
        await update.message.reply_text(f"Loi xoa DB: {e}")
        return

    # Cập nhật memory: nếu có hardcode thì quay về, không thì xoá
    if symbol in _WF_SYMBOL_CONFIG_DEFAULT:
        _WF_SYMBOL_CONFIG[symbol] = dict(_WF_SYMBOL_CONFIG_DEFAULT[symbol])
        status = f"Da xoa DB config. Quay ve hardcode default:\n  {_WF_SYMBOL_CONFIG[symbol]}"
    else:
        _WF_SYMBOL_CONFIG.pop(symbol, None)
        status = f"Da xoa khoi tat ca config. {symbol} se can chay pipeline lai."

    msg = f"{'✅' if deleted else '⚠️'} {symbol}: {status}"
    await update.message.reply_text(msg)
