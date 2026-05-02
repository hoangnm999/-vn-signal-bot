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

# ── Hard filter (optional, dùng khi use_hard_filter=True) ───────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
# ANALOG ENGINE V2
# ══════════════════════════════════════════════════════════════════════════════
#
# Thiết kế mới giải quyết 4 vấn đề từ Session 28:
#
#   Vấn đề 1+2: Threshold vô nghĩa vì pool luôn đủ lớn
#   → Fix: MIN_SAMPLES=15 + gate median(pool)>MIN_POOL_MEDIAN
#          → threshold phải lọc được chất lượng thực sự
#
#   Vấn đề 3: Backtest n quá cao (~91% bước tạo signal)
#   → Fix: Thêm breakdown skip rõ ràng (hard_filter / threshold / pool / median)
#
#   Vấn đề 4: Bất nhất Backtest vs WF
#   → Fix: Cùng hard filter optional, cùng MIN_SAMPLES, cùng gate,
#          cùng FWD_DAYS=30, cùng metrics = actual forward return
#
# State vector: V2 (11+2=13 dims, không overlap)
# ══════════════════════════════════════════════════════════════════════════════

# ── Constants ────────────────────────────────────────────────────────────────
_FWD_DAYS        = 30     # forward return window (bars)
_MDS_DAYS        = 43     # minimum distance sampling (cal days ≈ 30 trading)
_MIN_SAMPLES     = 15     # analog pool tối thiểu sau MDS
_WIN_THRESH      = 1.0    # win = actual_ret > 1%
_MIN_POOL_MEDIAN = 1.0    # gate: median(pool fwd_rets) > 1% mới tạo signal
_ANALOG_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

# Walk-forward
_WF_START_DATE         = "2025-01-01"
_WF_COOLDOWN_BARS      = 5     # 5 trading bars ≈ 7 calendar days (đồng bộ live)
BACKTEST_ANALOG_COOLDOWN    = 300
BACKTEST_ANALOG_WF_COOLDOWN = 300
ANALOG_PIPELINE_COOLDOWN    = 600
PIPELINE_MAX_SYMBOLS        = 5
_last_backtest_analog:    dict[str, float] = {}
_last_backtest_analog_wf: dict[str, float] = {}
_last_analog_pipeline:    dict[str, float] = {}

# Pipeline filter
_PIPELINE_MIN_EXP = 0.5   # mean actual return OOS > 0.5%
_PIPELINE_MIN_WR  = 0.50  # win rate OOS > 50%
_PIPELINE_MIN_PF  = 1.3   # profit factor OOS > 1.3

# ── WF Symbol Config ─────────────────────────────────────────────────────────
_WF_SYMBOL_CONFIG_DEFAULT: dict = {
    "FPT": {"combo": "Macro Trend",      "threshold": 0.65},
    "MWG": {"combo": "Oversold Bounce",  "threshold": 0.65},
    "STB": {"combo": "Oversold Bounce",  "threshold": 0.70},
    "HPG": {"combo": "Volume Confirmed", "threshold": 0.75},
    "GAS": {"combo": "Trend Following",  "threshold": 0.65},
    "DPM": {"combo": "Volatility Aware", "threshold": 0.65},
    "DCM": {"combo": "Volatility Aware", "threshold": 0.65},
}
_WF_SYMBOL_CONFIG: dict = dict(_WF_SYMBOL_CONFIG_DEFAULT)


def _load_wf_config_from_db():
    """Load analog config từ DB và merge vào _WF_SYMBOL_CONFIG."""
    global _WF_SYMBOL_CONFIG
    try:
        from db import load_analog_configs
        db_configs = load_analog_configs()
        if db_configs:
            _WF_SYMBOL_CONFIG = dict(_WF_SYMBOL_CONFIG_DEFAULT)
            _WF_SYMBOL_CONFIG.update(db_configs)
            logger.info(
                f"[WFConfig] Loaded {len(db_configs)} configs from DB. "
                f"Total: {len(_WF_SYMBOL_CONFIG)} symbols."
            )
    except Exception as e:
        logger.warning(f"[WFConfig] load_wf_config_from_db error: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_combo(name_query: str) -> dict | None:
    """Tìm combo theo tên (exact hoặc prefix, case-insensitive)."""
    q = name_query.strip().lower()
    for c in _ANALOG_COMBOS:
        if c["name"].lower() == q:
            return c
    for c in _ANALOG_COMBOS:
        if c["name"].lower().startswith(q):
            return c
    return None


def _build_target_arr(vec: dict, dims: list) -> "np.ndarray | None":
    import numpy as np
    arr = np.array([vec.get(d, 0.0) for d in dims], dtype=float)
    return arr if np.linalg.norm(arr) >= 1e-9 else None


def _cosine(a, b):
    import numpy as np
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _apply_mds(sim_list: list, dates, mds_days: int) -> list:
    """
    Minimum Distance Sampling: loại analog quá gần nhau về thời gian.
    sim_list: list of (idx, sim), đã sort giảm dần theo sim.
    """
    import pandas as pd
    kept = []
    for c_idx, sim in sim_list:
        c_date = pd.Timestamp(dates[c_idx])
        too_close = any(
            abs((c_date - pd.Timestamp(dates[k])).days) < mds_days
            for k, _ in kept
        )
        if not too_close:
            kept.append((c_idx, sim))
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# TẦNG 1: BACKTEST ENGINE V2
# ══════════════════════════════════════════════════════════════════════════════

def _run_one_experiment_v2(
    combo: dict,
    threshold: float,
    vectors: dict,
    vector_indices: list,
    dates,
    close_arr,
    low_arr,
    n_bars: int,
    use_hard_filter: bool = False,
) -> dict:
    """
    Chạy 1 experiment (1 combo × 1 threshold) — Engine V2.

    Skip breakdown rõ ràng:
      n_skip_hard  : regime filter loại
      n_skip_thresh: không đủ analog pass cosine threshold
      n_skip_pool  : pool < MIN_SAMPLES sau MDS
      n_skip_median: median(pool fwd_rets) <= MIN_POOL_MEDIAN
      n_signals    : số ngày thực sự tạo signal
    """
    import numpy as np
    import pandas as pd

    combo_name = combo["name"]
    dims       = combo["dims"]
    group      = combo.get("group", "")
    hypothesis = combo.get("hypothesis", "")

    n_skip_hard   = 0
    n_skip_thresh = 0
    n_skip_pool   = 0
    n_skip_median = 0
    signals       = []

    # Backtest loop: step=7 bars (simulate weekly review)
    for t_idx in range(120, n_bars - _FWD_DAYS - 1, 7):
        if t_idx not in vectors:
            continue

        target_vec = vectors[t_idx]

        # Gate 1: Hard filter (optional)
        if use_hard_filter and not _check_hard_filter(target_vec, combo_name):
            n_skip_hard += 1
            continue

        # Tính target array
        target_arr = _build_target_arr(target_vec, dims)
        if target_arr is None:
            continue

        # Pool analog: toàn bộ training, loại 90 bars gần nhất (tránh leakage)
        exclude_cutoff = t_idx - 90
        candidates = [i for i in vector_indices if i < exclude_cutoff]
        if len(candidates) < _MIN_SAMPLES:
            continue

        # Gate 2: Cosine similarity >= threshold
        sim_list = []
        for c_idx in candidates:
            c_arr = _build_target_arr(vectors[c_idx], dims)
            if c_arr is None:
                continue
            sim = _cosine(target_arr, c_arr)
            if sim >= threshold:
                sim_list.append((c_idx, sim))

        if not sim_list:
            n_skip_thresh += 1
            continue

        # MDS: loại analog quá gần nhau
        sim_list.sort(key=lambda x: -x[1])
        kept = _apply_mds(sim_list, dates, _MDS_DAYS)

        # Gate 3: Pool đủ lớn sau MDS
        if len(kept) < _MIN_SAMPLES:
            n_skip_pool += 1
            continue

        # Tính forward returns của pool (historical outcomes)
        fwd_rets = []
        mae_vals = []
        for c_idx, _ in kept:
            fwd_idx = c_idx + _FWD_DAYS
            if fwd_idx >= n_bars:
                continue
            entry = close_arr[c_idx]
            fwd_rets.append((close_arr[fwd_idx] - entry) / entry * 100)
            win_low = float(np.min(low_arr[c_idx + 1:fwd_idx + 1])) if fwd_idx > c_idx else entry
            mae_vals.append((win_low - entry) / entry * 100)

        if len(fwd_rets) < _MIN_SAMPLES:
            n_skip_pool += 1
            continue

        # Gate 4: median(pool) > _MIN_POOL_MEDIAN
        pool_median = float(np.median(fwd_rets))
        if pool_median <= _MIN_POOL_MEDIAN:
            n_skip_median += 1
            continue

        # Actual forward return tại ngày T (điểm đồng nhất với WF)
        actual_fwd_idx = t_idx + _FWD_DAYS
        if actual_fwd_idx >= n_bars:
            continue  # backtest: bỏ qua bar chưa có kết quả

        actual_ret = (close_arr[actual_fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100

        signals.append({
            "t_idx":      t_idx,
            "actual_ret": actual_ret,
            "pool_med":   pool_median,
            "mae":        float(np.median(mae_vals)),
        })

    # ── Tổng hợp metrics ─────────────────────────────────────────────────────
    n_sig = len(signals)
    skip_info = {
        "n_skip_hard":   n_skip_hard,
        "n_skip_thresh": n_skip_thresh,
        "n_skip_pool":   n_skip_pool,
        "n_skip_median": n_skip_median,
        "n_signals":     n_sig,
    }

    if n_sig < 5:
        return {
            "combo": combo_name, "group": group, "hypothesis": hypothesis,
            "threshold": threshold, "skip": True,
            **skip_info,
        }

    rets   = [s["actual_ret"] for s in signals]
    wins   = [r for r in rets if r >= _WIN_THRESH]
    losses = [r for r in rets if r < _WIN_THRESH]
    wr     = len(wins) / n_sig
    mean_r = float(np.mean(rets))
    med_r  = float(np.median(rets))
    std_r  = float(np.std(rets)) if n_sig > 1 else 1e-9
    sharpe = mean_r / std_r * (52 ** 0.5) if std_r > 0 else 0.0
    pos_s  = sum(wins)
    neg_s  = abs(sum(losses)) if losses else 1e-9
    pf     = pos_s / neg_s if neg_s > 0 else 99.0
    mae30  = float(np.median([s["mae"] for s in signals]))
    max_dd = float(np.min(rets))

    return {
        "combo": combo_name, "group": group, "hypothesis": hypothesis,
        "threshold": threshold, "skip": False,
        "wr":       round(wr * 100, 1),
        "mean_exp": round(mean_r, 2),
        "med_exp":  round(med_r, 2),
        "mae30":    round(mae30, 2),
        "max_dd":   round(max_dd, 1),
        "sharpe":   round(sharpe, 3),
        "pf":       round(pf, 2),
        **skip_info,
    }


def _run_analog_backtest_sync(symbol: str, days: int = 1800) -> dict:
    """
    Tầng 1: Chạy 15 combos × 7 thresholds = 105 experiments.
    Dùng hard_filter=False (optional) để tránh overfit.
    Trả về top results để Tầng 2 dùng.
    """
    import numpy as np
    import pandas as pd

    symbol = symbol.upper()
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=days, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu"}

    try:
        from state_vector import compute_state_vector_for_date
    except ImportError as e:
        return {"status": "error", "error": f"Thieu state_vector: {e}"}

    # Tính vectors toàn bộ training (trước OOS start)
    wf_start    = pd.Timestamp(_WF_START_DATE)
    date_series = pd.to_datetime(df["date"])
    oos_start   = next((i for i, d in enumerate(date_series) if d >= wf_start), len(df))

    vectors = {}
    for i in range(59, oos_start):
        vec = compute_state_vector_for_date(df, i)
        if vec is not None:
            vectors[i] = vec

    if len(vectors) < 150:
        return {"status": "error", "error": "Khong du vectors trong training period"}

    n_bars        = oos_start  # chỉ dùng training data
    close_arr     = df["close"].values[:oos_start].astype(float)
    low_arr       = df["low"].values[:oos_start].astype(float)
    dates         = df["date"].values[:oos_start]
    vector_indices = sorted(vectors.keys())

    # 105 experiments
    all_results = []
    for combo in _ANALOG_COMBOS:
        for thr in _ANALOG_THRESHOLDS:
            r = _run_one_experiment_v2(
                combo, thr, vectors, vector_indices,
                dates, close_arr, low_arr, n_bars,
                use_hard_filter=False,
            )
            if not r.get("skip"):
                all_results.append(r)

    if not all_results:
        return {"status": "error", "error": "Khong co experiment nao co du signal"}

    # Sort: mean_exp desc, pf desc
    all_results.sort(key=lambda x: (-x["mean_exp"], -x["pf"]))

    return {
        "status":      "ok",
        "symbol":      symbol,
        "n_bars":      n_bars,
        "n_vectors":   len(vectors),
        "top_results": all_results[:20],
        "all_results": all_results,
    }


def _format_analog_backtest_result(res: dict) -> str:
    """Format kết quả Tầng 1 cho Telegram."""
    if res.get("status") == "error":
        return f"❌ Lỗi backtest: {res['error']}"

    symbol = res["symbol"]
    top    = res.get("top_results", [])
    lines  = [
        f"📊 ANALOG BACKTEST V2 — {symbol}",
        f"Training: {res['n_bars']} bars | Vectors: {res['n_vectors']}",
        f"Gate: MIN_SAMPLES={_MIN_SAMPLES}, pool_median>{_MIN_POOL_MEDIAN}%, FWD={_FWD_DAYS}d",
        "─" * 38,
    ]

    for i, r in enumerate(top[:10], 1):
        skip_txt = (
            f"  skip: hard={r['n_skip_hard']} thr={r['n_skip_thresh']} "
            f"pool={r['n_skip_pool']} med={r['n_skip_median']}"
        )
        lines.append(
            f"{i:2}. [{r['combo']}] thr={r['threshold']:.2f}\n"
            f"   n={r['n_signals']} WR={r['wr']}% Exp={r['mean_exp']:+.2f}% "
            f"PF={r['pf']:.2f} MAE={r['mae30']:.1f}%\n"
            f"{skip_txt}"
        )

    lines.append("─" * 38)
    lines.append(f"Top 1: /backtest_analog_detail {symbol} \"{top[0]['combo']}\"" if top else "")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TẦNG 2: WALK-FORWARD ENGINE V2
# ══════════════════════════════════════════════════════════════════════════════

def _run_walkforward_sync(symbol: str) -> dict:
    """
    Walk-forward OOS V2:
      - Training : tất cả data TRƯỚC 2025-01-01
      - OOS      : 2025-01-01 → nay
      - Loop     : step=1 bar (simulate live trading)
      - Cooldown : 5 bars sau mỗi signal (≈ 7 calendar days)
      - Pool     : chỉ analog trong training (không dùng OOS làm analog)
      - Gate     : same as backtest (MIN_SAMPLES, pool_median, hard_filter=False)
      - Metrics  : actual forward return tại ngày T (đồng nhất với backtest)

    Skip breakdown:
      n_skip_cooldown: trong cooldown window
      n_skip_thresh  : không đủ analog pass cosine
      n_skip_pool    : pool < MIN_SAMPLES sau MDS
      n_skip_median  : median(pool) <= MIN_POOL_MEDIAN
    """
    import numpy as np
    import pandas as pd

    symbol = symbol.upper()
    if symbol not in _WF_SYMBOL_CONFIG:
        return {
            "status": "error",
            "error":  f"{symbol} chua co config. Co san: {', '.join(_WF_SYMBOL_CONFIG)}",
        }

    cfg        = _WF_SYMBOL_CONFIG[symbol]
    combo_name = cfg["combo"]
    threshold  = cfg["threshold"]

    combo = _find_combo(combo_name)
    if combo is None:
        return {"status": "error", "error": f"Khong tim thay combo '{combo_name}'"}
    dims = combo["dims"]

    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu"}

    try:
        from state_vector import compute_state_vector_for_date
    except ImportError as e:
        return {"status": "error", "error": f"Thieu state_vector: {e}"}

    # Tính vectors toàn bộ (cả training + OOS)
    vectors = {}
    for i in range(59, len(df)):
        vec = compute_state_vector_for_date(df, i)
        if vec is not None:
            vectors[i] = vec

    if len(vectors) < 100:
        return {"status": "error", "error": "Khong du vectors"}

    n_bars        = len(df)
    dates         = df["date"].values
    close_arr     = df["close"].values.astype(float)
    low_arr       = df["low"].values.astype(float)
    date_series   = pd.to_datetime(df["date"])
    vector_indices = sorted(vectors.keys())

    wf_start     = pd.Timestamp(_WF_START_DATE)
    oos_start_idx = next(
        (i for i, d in enumerate(date_series) if d >= wf_start), None
    )
    if oos_start_idx is None:
        return {"status": "error", "error": f"Khong co data sau {_WF_START_DATE}"}

    # Chỉ dùng training làm pool analog (không dùng OOS làm analog)
    train_indices = [i for i in vector_indices if i < oos_start_idx]

    # ── WF Loop ───────────────────────────────────────────────────────────────
    oos_signals         = []
    train_signals       = []
    n_skip_cooldown     = 0
    n_skip_thresh       = 0
    n_skip_pool         = 0
    n_skip_median       = 0
    last_signal_bar     = None

    for t_idx in range(120, n_bars - _FWD_DAYS - 1, 1):
        if t_idx not in vectors:
            continue

        t_date = pd.Timestamp(dates[t_idx])
        is_oos = t_date >= wf_start

        # Cooldown: chỉ áp dụng OOS (simulate live)
        if is_oos and last_signal_bar is not None:
            if (t_idx - last_signal_bar) < _WF_COOLDOWN_BARS:
                n_skip_cooldown += 1
                continue

        target_vec = vectors[t_idx]
        target_arr = _build_target_arr(target_vec, dims)
        if target_arr is None:
            continue

        # Pool analog: chỉ từ training, loại 90 bars gần nhất
        exclude_cutoff = min(t_idx - 90, oos_start_idx - 1)
        candidates = [i for i in train_indices if i < exclude_cutoff]
        if len(candidates) < _MIN_SAMPLES:
            continue

        # Gate: cosine similarity
        sim_list = []
        for c_idx in candidates:
            c_arr = _build_target_arr(vectors[c_idx], dims)
            if c_arr is None:
                continue
            sim = _cosine(target_arr, c_arr)
            if sim >= threshold:
                sim_list.append((c_idx, sim))

        if not sim_list:
            if is_oos:
                n_skip_thresh += 1
            continue

        # MDS
        sim_list.sort(key=lambda x: -x[1])
        kept = _apply_mds(sim_list, dates, _MDS_DAYS)

        if len(kept) < _MIN_SAMPLES:
            if is_oos:
                n_skip_pool += 1
            continue

        # Pool forward returns (historical)
        fwd_rets = []
        mae_vals = []
        for c_idx, _ in kept:
            fwd_idx = c_idx + _FWD_DAYS
            if fwd_idx >= n_bars:
                continue
            entry = close_arr[c_idx]
            fwd_rets.append((close_arr[fwd_idx] - entry) / entry * 100)
            win_low = float(np.min(low_arr[c_idx + 1:fwd_idx + 1])) if fwd_idx > c_idx else entry
            mae_vals.append((win_low - entry) / entry * 100)

        if len(fwd_rets) < _MIN_SAMPLES:
            if is_oos:
                n_skip_pool += 1
            continue

        # Gate: median(pool) > threshold
        pool_median = float(np.median(fwd_rets))
        if pool_median <= _MIN_POOL_MEDIAN:
            if is_oos:
                n_skip_median += 1
            continue

        # Actual forward return tại ngày T
        actual_fwd_idx = t_idx + _FWD_DAYS
        pending = actual_fwd_idx >= n_bars
        actual  = None if pending else (
            (close_arr[actual_fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100
        )

        sig = {
            "t_idx":      t_idx,
            "t_date":     str(t_date)[:10],
            "pool_med":   round(pool_median, 2),
            "pool_mae":   round(float(np.median(mae_vals)), 2),
            "actual":     actual,
            "pending":    pending,
        }

        if is_oos:
            oos_signals.append(sig)
            last_signal_bar = t_idx
        else:
            train_signals.append(sig)

    # ── Tính metrics ─────────────────────────────────────────────────────────
    def _calc(sigs):
        completed = [s for s in sigs if not s["pending"] and s["actual"] is not None]
        if not completed:
            return {}
        rets   = [s["actual"] for s in completed]
        wins   = [r for r in rets if r >= _WIN_THRESH]
        losses = [r for r in rets if r < _WIN_THRESH]
        wr     = len(wins) / len(rets)
        mean_r = float(np.mean(rets))
        pos_s  = sum(wins)
        neg_s  = abs(sum(losses)) if losses else 1e-9
        pf     = pos_s / neg_s if neg_s > 0 else 99.0
        return {
            "n":        len(completed),
            "n_pending":len(sigs) - len(completed),
            "wr":       round(wr * 100, 1),
            "mean_exp": round(mean_r, 2),
            "pf":       round(pf, 2),
            "max_dd":   round(float(np.min(rets)), 1),
            "mae30":    round(float(np.median([s["pool_mae"] for s in completed])), 2),
        }

    train_metrics = _calc(train_signals)
    oos_metrics   = _calc(oos_signals)

    return {
        "status":          "ok",
        "symbol":          symbol,
        "combo":           combo_name,
        "threshold":       threshold,
        "oos_start":       _WF_START_DATE,
        "train_metrics":   train_metrics,
        "oos_metrics":     oos_metrics,
        "oos_signals":     oos_signals,
        "train_signals":   train_signals,
        "n_skip_cooldown": n_skip_cooldown,
        "n_skip_thresh":   n_skip_thresh,
        "n_skip_pool":     n_skip_pool,
        "n_skip_median":   n_skip_median,
        "n_oos_bars":      n_bars - oos_start_idx,
    }


def _format_wf_result(res: dict) -> str:
    """Format kết quả Walk-forward V2 cho Telegram."""
    if res.get("status") == "error":
        return f"❌ Lỗi WF: {res['error']}"

    sym   = res["symbol"]
    combo = res["combo"]
    thr   = res["threshold"]
    tm    = res.get("train_metrics", {})
    oom   = res.get("oos_metrics", {})

    lines = [
        f"🔁 WALK-FORWARD V2 — {sym}",
        f"Combo: {combo} | Threshold: {thr}",
        f"OOS từ: {res['oos_start']} | FWD={_FWD_DAYS}d",
        "─" * 38,
        "📚 TRAINING:",
        f"  n={tm.get('n','?')} WR={tm.get('wr','?')}% "
        f"Exp={tm.get('mean_exp','?'):+}% PF={tm.get('pf','?')}",
        "─" * 38,
        "🎯 OOS:",
        f"  n={oom.get('n','?')} (pending={oom.get('n_pending','?')}) "
        f"WR={oom.get('wr','?')}% Exp={oom.get('mean_exp','?'):+}% PF={oom.get('pf','?')}",
        f"  MaxDD={oom.get('max_dd','?')}% MAE={oom.get('mae30','?')}%",
        "─" * 38,
        "📋 OOS SKIP BREAKDOWN:",
        f"  cooldown={res['n_skip_cooldown']} | "
        f"threshold={res['n_skip_thresh']} | "
        f"pool={res['n_skip_pool']} | "
        f"median={res['n_skip_median']}",
        "─" * 38,
    ]

    # Verdict
    n_oos = oom.get("n", 0)
    if n_oos < 3:
        verdict = "⚠️ Quá ít signal OOS — chưa đủ tin cậy"
    elif oom.get("mean_exp", 0) > _PIPELINE_MIN_EXP and oom.get("pf", 0) >= _PIPELINE_MIN_PF:
        verdict = "✅ OOS PASS — đủ điều kiện live trading"
    elif oom.get("mean_exp", 0) > 0:
        verdict = "🟡 OOS dương nhưng yếu — theo dõi thêm"
    else:
        verdict = "❌ OOS âm — không dùng config này"

    lines.append(verdict)

    # Recent OOS signals
    oos_sigs = res.get("oos_signals", [])
    if oos_sigs:
        lines.append("\n📅 Recent OOS signals:")
        for s in oos_sigs[-5:]:
            ret_txt = f"{s['actual']:+.1f}%" if s["actual"] is not None else "pending"
            lines.append(
                f"  {s['t_date']} pool_med={s['pool_med']:+.1f}% → actual={ret_txt}"
            )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ANALOG DETAIL — xem chi tiết 1 combo với 7 threshold levels
# ══════════════════════════════════════════════════════════════════════════════

def _run_analog_detail_sync(symbol: str, combo_name: str, days: int = 1800) -> dict:
    """Chạy 1 combo × 7 threshold trên training data để chọn threshold tối ưu."""
    import numpy as np
    import pandas as pd

    symbol = symbol.upper()
    combo  = _find_combo(combo_name)
    if combo is None:
        return {"status": "error", "error": f"Khong tim thay combo '{combo_name}'"}

    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=days, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu"}

    try:
        from state_vector import compute_state_vector_for_date
    except ImportError as e:
        return {"status": "error", "error": f"Thieu state_vector: {e}"}

    wf_start    = pd.Timestamp(_WF_START_DATE)
    date_series = pd.to_datetime(df["date"])
    oos_start   = next((i for i, d in enumerate(date_series) if d >= wf_start), len(df))

    vectors = {}
    for i in range(59, oos_start):
        vec = compute_state_vector_for_date(df, i)
        if vec is not None:
            vectors[i] = vec

    if len(vectors) < 150:
        return {"status": "error", "error": "Khong du vectors"}

    n_bars        = oos_start
    close_arr     = df["close"].values[:oos_start].astype(float)
    low_arr       = df["low"].values[:oos_start].astype(float)
    dates         = df["date"].values[:oos_start]
    vector_indices = sorted(vectors.keys())

    results = []
    for thr in _ANALOG_THRESHOLDS:
        r = _run_one_experiment_v2(
            combo, thr, vectors, vector_indices,
            dates, close_arr, low_arr, n_bars,
            use_hard_filter=False,
        )
        results.append(r)

    return {
        "status":            "ok",
        "symbol":            symbol,
        "combo":             combo["name"],
        "threshold_results": results,
    }


def _format_detail_result(res: dict) -> str:
    if res.get("status") == "error":
        return f"❌ {res['error']}"

    sym   = res["symbol"]
    combo = res["combo"]
    rows  = res.get("threshold_results", [])
    lines = [
        f"🔍 DETAIL — {sym} [{combo}]",
        f"{'Thr':>5} {'n':>4} {'WR':>6} {'Exp':>7} {'PF':>5} {'skip_thr':>8} {'skip_med':>8}",
        "─" * 50,
    ]
    for r in rows:
        if r.get("skip"):
            lines.append(f"{r['threshold']:>5.2f} {'—':>4}")
        else:
            lines.append(
                f"{r['threshold']:>5.2f} {r['n_signals']:>4} "
                f"{r['wr']:>5.1f}% {r['mean_exp']:>+6.2f}% {r['pf']:>5.2f} "
                f"{r['n_skip_thresh']:>8} {r['n_skip_median']:>8}"
            )
    # Chọn threshold tốt nhất
    valid = [r for r in rows if not r.get("skip") and r.get("mean_exp", 0) > 0]
    if valid:
        best = max(valid, key=lambda x: (x["mean_exp"], x["pf"]))
        lines.append("─" * 50)
        lines.append(f"→ Đề xuất: threshold={best['threshold']} "
                     f"(Exp={best['mean_exp']:+.2f}%, PF={best['pf']:.2f})")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE V2 — Tầng 1 + Tầng 2 tự động
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline_wf_sync(symbol: str) -> dict:
    """
    Pipeline cho mã CHƯA có config:
      Tầng 1 → tìm combo + threshold tốt nhất
      Tầng 1b → detail để chọn threshold tối ưu cho combo đó
      Tầng 2 → WF OOS validation
    """
    # Tầng 1
    bt_res = _run_analog_backtest_sync(symbol)
    if bt_res.get("status") == "error":
        return {"status": "error", "error": bt_res["error"]}

    valid = [
        r for r in bt_res.get("top_results", [])
        if r.get("mean_exp", 0) > _PIPELINE_MIN_EXP
        and r.get("pf", 0) >= _PIPELINE_MIN_PF
    ]
    if not valid:
        top = bt_res.get("top_results", [{}])[0]
        return {
            "status": "skip",
            "reason": (
                f"Khong co combo nao qua bo loc "
                f"(top: Exp={top.get('mean_exp', 0):+.2f}% PF={top.get('pf', 0):.2f})"
            ),
        }

    best_bt    = valid[0]
    combo_name = best_bt["combo"]

    # Tầng 1b: chọn threshold tối ưu
    detail_res = _run_analog_detail_sync(symbol, combo_name)
    threshold  = best_bt["threshold"]  # fallback
    if detail_res.get("status") != "error":
        thresh_rows = [
            r for r in detail_res.get("threshold_results", [])
            if not r.get("skip")
            and r.get("mean_exp", 0) > _PIPELINE_MIN_EXP
            and r.get("pf", 0) >= _PIPELINE_MIN_PF
        ]
        if thresh_rows:
            best_thr = max(thresh_rows, key=lambda x: (x["mean_exp"], x["pf"]))
            threshold = best_thr["threshold"]

    # Tầng 2: WF với config vừa tìm
    _WF_SYMBOL_CONFIG[symbol] = {"combo": combo_name, "threshold": threshold}
    result = _run_walkforward_sync(symbol)

    # Kiểm tra OOS pass
    oos_pass = False
    if result.get("status") == "ok":
        oom      = result.get("oos_metrics") or {}
        oos_pass = (
            oom.get("mean_exp", 0) > _PIPELINE_MIN_EXP
            and oom.get("pf", 0) >= _PIPELINE_MIN_PF
            and oom.get("wr", 0) >= _PIPELINE_MIN_WR * 100
        )

    if not oos_pass:
        _WF_SYMBOL_CONFIG.pop(symbol, None)

    if result.get("status") == "ok":
        result["auto_config"]     = True
        result["found_combo"]     = combo_name
        result["found_threshold"] = threshold
        result["oos_pass"]        = oos_pass

    return result


def _format_pipeline_summary(results: dict) -> str:
    """Format kết quả pipeline nhiều mã."""
    lines = ["📋 ANALOG PIPELINE V2 — Kết quả", "─" * 38]
    for sym, res in results.items():
        status = res.get("status")
        if status == "error":
            lines.append(f"❌ {sym}: {res['error']}")
        elif status == "skip":
            lines.append(f"⏭️ {sym}: {res['reason']}")
        else:
            oom  = res.get("oos_metrics", {})
            cfg  = f"{res.get('found_combo','?')} @ {res.get('found_threshold','?')}"
            flag = "✅" if res.get("oos_pass") else "🟡"
            lines.append(
                f"{flag} {sym}: {cfg}\n"
                f"   OOS n={oom.get('n','?')} WR={oom.get('wr','?')}% "
                f"Exp={oom.get('mean_exp','?'):+}% PF={oom.get('pf','?')}"
            )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

async def backtest_analog_cmd(update, context):
    """/backtest_analog <MA> [days] — Tầng 1: backtest 105 experiments."""
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    user_id = str(update.effective_user.id)
    now     = time.time()
    if now - _last_backtest_analog.get(user_id, 0) < BACKTEST_ANALOG_COOLDOWN:
        await update.message.reply_text("⏳ Vui lòng đợi 5 phút giữa các lần chạy.")
        return
    _last_backtest_analog[user_id] = now

    args   = context.args or []
    symbol = args[0].upper() if args else None
    days   = int(args[1]) if len(args) > 1 and args[1].isdigit() else 1800

    if not symbol:
        await update.message.reply_text(
            "Cú pháp: /backtest_analog <MA> [days]\nVí dụ: /backtest_analog MWG"
        )
        return

    msg = await update.message.reply_text(f"⏳ Đang chạy backtest V2 cho {symbol}...")

    async def _bg():
        try:
            res  = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _run_analog_backtest_sync(symbol, days)
            )
            text = _format_analog_backtest_result(res)
            await msg.edit_text(_plain(text))
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi: {e}")

    asyncio.create_task(_bg())


async def backtest_analog_detail_cmd(update, context):
    """/backtest_analog_detail <MA> "<combo>" — xem chi tiết 7 threshold."""
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args = context.args or []
    if len(args) < 2:
        await update.message.reply_text(
            'Cú pháp: /backtest_analog_detail <MA> "<combo>"\n'
            'Ví dụ: /backtest_analog_detail MWG "Oversold Bounce"'
        )
        return

    symbol     = args[0].upper()
    combo_name = " ".join(args[1:]).strip('"').strip("'")

    msg = await update.message.reply_text(
        f"⏳ Đang chạy detail [{combo_name}] cho {symbol}..."
    )

    async def _bg():
        try:
            res  = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _run_analog_detail_sync(symbol, combo_name)
            )
            text = _format_detail_result(res)
            await msg.edit_text(_plain(text))
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi: {e}")

    asyncio.create_task(_bg())


async def walkforward_analog_cmd(update, context):
    """/walkforward_analog [MA1 MA2 ...] — Walk-forward OOS V2."""
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    user_id = str(update.effective_user.id)
    now     = time.time()
    if now - _last_backtest_analog_wf.get(user_id, 0) < BACKTEST_ANALOG_WF_COOLDOWN:
        await update.message.reply_text("⏳ Vui lòng đợi 5 phút.")
        return
    _last_backtest_analog_wf[user_id] = now

    args    = context.args or []
    symbols = [a.upper() for a in args] if args else list(_WF_SYMBOL_CONFIG.keys())

    msg = await update.message.reply_text(
        f"⏳ Walk-forward V2: {', '.join(symbols)}..."
    )

    async def _bg():
        try:
            lines = []
            for sym in symbols:
                if sym in _WF_SYMBOL_CONFIG:
                    res = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sym: _run_walkforward_sync(s)
                    )
                else:
                    res = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sym: _run_pipeline_wf_sync(s)
                    )
                lines.append(_format_wf_result(res))
                lines.append("")

            await msg.edit_text(_plain("\n".join(lines))[:4000])
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi: {e}")

    asyncio.create_task(_bg())


async def analog_pipeline_cmd(update, context):
    """/analog_pipeline <MA1> [MA2 ...] — Pipeline tự động Tầng 1 + 2."""
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    user_id = str(update.effective_user.id)
    now     = time.time()
    if now - _last_analog_pipeline.get(user_id, 0) < ANALOG_PIPELINE_COOLDOWN:
        await update.message.reply_text("⏳ Vui lòng đợi 10 phút.")
        return
    _last_analog_pipeline[user_id] = now

    args    = context.args or []
    symbols = [a.upper() for a in args[:PIPELINE_MAX_SYMBOLS]]

    if not symbols:
        await update.message.reply_text(
            f"Cú pháp: /analog_pipeline <MA1> [MA2 ...] (tối đa {PIPELINE_MAX_SYMBOLS} mã)"
        )
        return

    msg = await update.message.reply_text(
        f"⏳ Pipeline V2: {', '.join(symbols)}..."
    )

    async def _bg():
        try:
            results = {}
            for sym in symbols:
                results[sym] = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sym: _run_pipeline_wf_sync(s)
                )
            text = _format_pipeline_summary(results)
            await msg.edit_text(_plain(text))
        except Exception as e:
            await msg.edit_text(f"❌ Lỗi: {e}")

    asyncio.create_task(_bg())


# ══════════════════════════════════════════════════════════════════════════════
# /analog_approve, /analog_configs, /analog_remove
# ══════════════════════════════════════════════════════════════════════════════

# ── Pending approve state ────────────────────────────────────────────────────
_pending_approve: dict[str, dict] = {}  # user_id -> {symbol, combo, threshold}


async def analog_approve_cmd(update, context):
    """
    /analog_approve <MA>              — tu dong tim config tot nhat tu backtest
    /analog_approve <MA> <combo> <thr> — chi dinh tay
    Hien thi confirm button Yes/No truoc khi luu DB.
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args    = context.args or []
    user_id = str(update.effective_user.id)

    if not args:
        await update.message.reply_text(
            "Cach dung:\n"
            "  /analog_approve <MA>                - tu tim config\n"
            "  /analog_approve <MA> <combo> <thr>  - chi dinh tay\n"
            "Vi du: /analog_approve MWG\n"
            "       /analog_approve MWG Volatility_Aware 0.75"
        )
        return

    symbol = args[0].upper()

    # ── Truong hop 1: chi dinh combo + threshold tay ──────────────────────────
    if len(args) >= 3:
        combo_name = " ".join(args[1:-1]).strip('"').strip("'")
        try:
            threshold = float(args[-1])
        except ValueError:
            await update.message.reply_text("Threshold phai la so, vi du: 0.75")
            return
        combo = _find_combo(combo_name)
        if combo is None:
            await update.message.reply_text(f"Khong tim thay combo '{combo_name}'")
            return
        cfg = {"combo": combo["name"], "threshold": threshold}

    # ── Truong hop 2: tu tim config tot nhat tu backtest ─────────────────────
    else:
        msg = await update.message.reply_text(
            f"Dang chay backtest de tim config cho {symbol}..."
        )
        try:
            loop = asyncio.get_event_loop()
            bt_res = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: _run_analog_backtest_sync(symbol)
                ),
                timeout=300,
            )
        except asyncio.TimeoutError:
            await msg.edit_text("Timeout — thu lai sau.")
            return
        except Exception as e:
            await msg.edit_text(f"Loi backtest: {e}")
            return

        if bt_res.get("status") == "error":
            await msg.edit_text(f"Loi: {bt_res['error']}")
            return

        valid = [
            r for r in bt_res.get("top_results", [])
            if not r.get("skip")
            and r.get("mean_exp", 0) > 0
            and r.get("pf", 0) >= 1.3
            and r.get("n_signals", 0) >= 5
        ]
        if not valid:
            top = bt_res.get("top_results", [{}])[0]
            await msg.edit_text(
                "Khong co combo nao du tieu chuan.\n"
                f"Top: [{top.get('combo','?')}] "
                f"Exp={top.get('mean_exp',0):+.2f}% PF={top.get('pf',0):.2f}"
            )
            return

        best = valid[0]
        cfg  = {"combo": best["combo"], "threshold": best["threshold"]}
        await msg.delete()

    # ── Hien thi confirm button Yes/No ────────────────────────────────────────
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    _pending_approve[user_id] = {
        "symbol":    symbol,
        "combo":     cfg["combo"],
        "threshold": cfg["threshold"],
    }

    old_cfg = _WF_SYMBOL_CONFIG.get(symbol)
    old_txt = (
        f"Config cu: [{old_cfg['combo']}] thr={old_cfg['threshold']}\n"
        if old_cfg else "Config cu: (chua co)\n"
    )

    text = (
        f"XAC NHAN LUU CONFIG — {symbol}\n"
        f"{'─' * 32}\n"
        f"{old_txt}"
        f"Config moi: [{cfg['combo']}] thr={cfg['threshold']}\n"
        f"{'─' * 32}\n"
        "Luu config nay vao DB khong?"
    )

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("Yes", callback_data=f"approve_yes_{user_id}"),
        InlineKeyboardButton("No",  callback_data=f"approve_no_{user_id}"),
    ]])
    await update.message.reply_text(text, reply_markup=keyboard)


async def analog_approve_callback(update, context):
    """Xu ly nut Yes/No tu analog_approve_cmd."""
    query   = update.callback_query
    await query.answer()
    data    = query.data
    user_id = str(query.from_user.id)

    if not data.endswith(f"_{user_id}"):
        await query.answer("Khong phai lenh cua ban.", show_alert=True)
        return

    pending = _pending_approve.pop(user_id, None)
    if pending is None:
        await query.edit_message_text("Phien xac nhan da het han. Chay lai /analog_approve.")
        return

    symbol    = pending["symbol"]
    combo     = pending["combo"]
    threshold = pending["threshold"]

    if data.startswith("approve_no_"):
        await query.edit_message_text(
            f"Da huy — config {symbol} [{combo}] thr={threshold} khong duoc luu."
        )
        return

    try:
        from db import save_analog_config
        save_analog_config(symbol, {"combo": combo, "threshold": threshold})
        _WF_SYMBOL_CONFIG[symbol] = {"combo": combo, "threshold": threshold}
        await query.edit_message_text(
            f"Da luu config {symbol}:\n"
            f"  Combo    : {combo}\n"
            f"  Threshold: {threshold}\n\n"
            f"Chay /walkforward_analog {symbol} de validate OOS voi config moi."
        )
    except Exception as e:
        await query.edit_message_text(f"Loi luu DB: {e}")


async def analog_configs_cmd(update, context):
    """/analog_configs — Xem tất cả config đang active."""
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    lines = ["⚙️ ANALOG CONFIGS ACTIVE:", "─" * 35]
    for sym, cfg in sorted(_WF_SYMBOL_CONFIG.items()):
        src = "DB" if sym not in _WF_SYMBOL_CONFIG_DEFAULT else "default"
        lines.append(
            f"  {sym}: [{cfg['combo']}] thr={cfg['threshold']} ({src})"
        )
    await update.message.reply_text("\n".join(lines))


async def analog_remove_cmd(update, context):
    """/analog_remove <MA> — Xóa config khỏi DB."""
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
            "Cú pháp: /analog_remove <MA>\nVí dụ: /analog_remove LPB"
        )
        return

    symbol = args[0].upper()
    try:
        from db import delete_analog_config
        deleted = delete_analog_config(symbol)
    except Exception as e:
        await update.message.reply_text(f"❌ Lỗi xóa DB: {e}")
        return

    if symbol in _WF_SYMBOL_CONFIG_DEFAULT:
        _WF_SYMBOL_CONFIG[symbol] = dict(_WF_SYMBOL_CONFIG_DEFAULT[symbol])
        status = f"Quay về hardcode default: {_WF_SYMBOL_CONFIG[symbol]}"
    else:
        _WF_SYMBOL_CONFIG.pop(symbol, None)
        status = f"Đã xóa hoàn toàn. Cần chạy /analog_pipeline {symbol} lại."

    flag = "✅" if deleted else "⚠️"
    await update.message.reply_text(f"{flag} {symbol}: {status}")


# ── Stubs tương thích bot.py ──────────────────────────────────────────────────

async def backtest_analog_batch_cmd(update, context):
    await update.message.reply_text(
        "Dung /analog_pipeline thay the.\nVi du: /analog_pipeline MWG STB DPM"
    )

async def analog_regime_analysis_cmd(update, context):
    await update.message.reply_text(
        "Da gop vao /walkforward_analog.\nDung: /walkforward_analog <MA>"
    )

async def analog_sim_dist_cmd(update, context):
    await update.message.reply_text(
        "Da loai bo trong V2.\nDung /backtest_analog_detail <MA> de xem phan bo threshold."
    )
