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


# ══════════════════════════════════════════════════════════════════════════════
# CORE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

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

            # Max Drawdown của equity curve
            equity   = np.cumprod([1.0 + r / 100.0 for r in sig_rets])
            peak     = np.maximum.accumulate(equity)
            max_dd   = float(np.min((equity - peak) / peak * 100))

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
            f"MaxDD {baseline['max_dd']:.1f}%  "
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
                f"MaxDD {r['max_dd']:.1f}%  "
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
    /backtest_analog <MA> [days]
    Test 15 combo x 7 nguong = 105 experiments.
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
            "Cu phap: /backtest_analog <MA> [days]\n\n"
            "Vi du:\n"
            "  /backtest_analog HPG\n"
            "  /backtest_analog VCB 1500\n\n"
            "105 experiments | Metrics: WR, Exp, MAE30, MaxDD, Sharpe, PF\n"
            "Thoi gian: 4-10 phut."
        )
        return

    import re as _re
    symbol = args[0].upper().strip()
    if not _re.match(r'^[A-Z0-9]{2,10}$', symbol):
        await update.message.reply_text(f"Ma '{symbol}' khong hop le.")
        return

    try:
        days = max(500, min(int(args[1]), 2500)) if len(args) > 1 else 1800
    except (ValueError, IndexError):
        days = 1800

    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_backtest_analog.get(user_id, 0)
    if since < BACKTEST_ANALOG_COOLDOWN:
        wait = int(BACKTEST_ANALOG_COOLDOWN - since)
        await update.message.reply_text(f"Vui long cho {wait}s truoc khi chay tiep.")
        return
    _last_backtest_analog[user_id] = time.time()

    chat_id = update.effective_chat.id
    msg = await update.message.reply_text(
        f"Backtest Analog T1: {symbol} ({days} ngay)\n"
        f"15 combo x 7 nguong = 105 experiments\n"
        f"Uoc tinh 4-10 phut, vui long doi..."
    )

    async def _bg():
        try:
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


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ANALOG BATCH — nhiều mã một lượt
# ══════════════════════════════════════════════════════════════════════════════

BACKTEST_ANALOG_BATCH_COOLDOWN = 600   # 10 phút per user
_last_backtest_analog_batch: dict[str, float] = {}
BATCH_MAX_SYMBOLS = 10   # giới hạn tối đa để tránh timeout Render


def _run_analog_batch_sync(symbols: list[str], days: int = 1800) -> dict:
    """
    Chạy _run_analog_backtest_sync tuần tự cho từng mã.
    Trả về dict tổng hợp kết quả tất cả mã.
    """
    results = {}
    for symbol in symbols:
        try:
            r = _run_analog_backtest_sync(symbol, days)
            results[symbol] = r
        except Exception as e:
            results[symbol] = {"status": "error", "error": str(e)[:120]}
    return results


def _format_batch_result(batch: dict) -> str:
    """
    Format kết quả batch thành 2 phần:
      1. Bảng tổng quan: mỗi mã 1 dòng
      2. Nhận xét cross-symbol: combo nào nhất quán nhất
    """
    from collections import Counter

    sep  = "=" * 32
    sep2 = "-" * 32

    # ── Gom dữ liệu từng mã ──────────────────────────────────────────────────
    rows        = []   # dữ liệu cho bảng tổng quan
    all_top1    = []   # top combo của mỗi mã (để phân tích nhất quán)
    good_syms   = []   # mã Sharpe >= 1.0
    weak_syms   = []   # mã Sharpe < 1.0 hoặc lỗi

    for symbol, res in batch.items():
        if res.get("status") == "error":
            rows.append({
                "symbol": symbol, "ok": False,
                "error": res.get("error", "?")[:60],
            })
            weak_syms.append(symbol)
            continue

        top     = res.get("top_results", [])
        bl      = res.get("baseline")
        n_valid = res.get("n_valid", 0)

        if not top:
            rows.append({
                "symbol": symbol, "ok": False,
                "error": f"Khong du tin hieu ({n_valid} exp hop le)",
            })
            weak_syms.append(symbol)
            continue

        best = top[0]
        bl_sharpe = bl["sharpe"] if bl else 0.0

        row = {
            "symbol":    symbol,
            "ok":        True,
            "combo":     best["combo"],
            "threshold": best["threshold"],
            "wr":        best["wr"],
            "mean_exp":  best["mean_exp"],
            "mae30":     best["mae30"],
            "max_dd":    best["max_dd"],
            "sharpe":    best["sharpe"],
            "pf":        best["pf"],
            "n":         best["n_signals"],
            "bl_sharpe": bl_sharpe,
            "vs_bl":     round(best["sharpe"] - bl_sharpe, 2),
            "group":     best.get("group", ""),
            "n_valid":   n_valid,
        }
        rows.append(row)
        all_top1.append(best["combo"])

        # Bộ lọc cứng: Exp > 0 và PF >= 1.5 (WR chỉ tham khảo tâm lý)
        if best["mean_exp"] > 0 and best["pf"] >= 1.5:
            good_syms.append(symbol)
        else:
            weak_syms.append(symbol)

    # ── Bảng tổng quan ───────────────────────────────────────────────────────
    lines = [
        f"BACKTEST ANALOG BATCH — {len(batch)} ma",
        f"Days: 1800 | 105 experiments/ma",
        sep,
        "MA      TOP COMBO            WR   Exp   Sharpe  n   vsBase",
        sep2,
    ]

    for r in rows:
        if not r["ok"]:
            lines.append(f"{r['symbol']:<7} LOI: {r['error']}")
            continue

        # Emoji — dựa trên Exp và PF, không dùng WR
        if r["mean_exp"] > 1.0 and r["pf"] >= 2.0:
            em = "✅"
        elif r["mean_exp"] > 0 and r["pf"] >= 1.5:
            em = "🟡"
        else:
            em = "🔴"

        vs = f"{r['vs_bl']:+.2f}"
        combo_short = r["combo"][:18]   # cắt ngắn để vừa dòng

        lines.append(
            f"{em}{r['symbol']:<6} "
            f"{combo_short:<20} "
            f"{r['wr']:.0f}%  "
            f"{r['mean_exp']:+.1f}%  "
            f"{r['sharpe']:.2f}   "
            f"{r['n']:<4} "
            f"{vs}"
        )

    lines.append(sep2)

    # ── Phân tích cross-symbol ────────────────────────────────────────────────
    lines.append("")
    lines.append("PHAN TICH CHUNG:")

    # Mã nên/không nên dùng analog
    if good_syms:
        lines.append(f"  Nen dung analog : {', '.join(good_syms)}")
    if weak_syms:
        lines.append(f"  Nen bo qua      : {', '.join(weak_syms)}")

    lines.append("")

    # Combo nhất quán nhất
    if all_top1:
        combo_counts = Counter(all_top1)
        top_combos   = combo_counts.most_common(3)
        lines.append("  Combo nhat quan tren nhieu ma:")
        for combo, count in top_combos:
            pct = count / len(all_top1) * 100
            lines.append(f"    {combo}: {count}/{len(all_top1)} ma ({pct:.0f}%)")

        # Nhóm thống trị
        ok_rows = [r for r in rows if r.get("ok")]
        if ok_rows:
            groups = Counter(r["group"] for r in ok_rows)
            group_labels = {
                "momentum":  "Momentum", "trend": "Trend Following",
                "reversion": "Mean Reversion", "volume": "Volume",
                "volatility":"Volatility", "crossover": "Combo giao thoa",
                "baseline":  "Baseline",
            }
            top_group = groups.most_common(1)[0]
            lines.append(
                f"  Nhom thong tri: "
                f"{group_labels.get(top_group[0], top_group[0])} "
                f"({top_group[1]}/{len(ok_rows)} ma)"
            )

            # Threshold phổ biến
            thresholds = Counter(r["threshold"] for r in ok_rows)
            top_thresh = thresholds.most_common(1)[0]
            lines.append(
                f"  Nguong pho bien: {top_thresh[0]:.2f} "
                f"({top_thresh[1]}/{len(ok_rows)} ma)"
            )

    lines.append("")

    # Khuyến nghị walk-forward — config riêng cho từng mã
    good_rows = [r for r in rows if r.get("ok") and r["symbol"] in good_syms]
    if good_rows:
        lines.append("  → BUOC TIEP THEO:")
        lines.append("     Chay detail de chon nguong chinh xac:")
        for r in good_rows:
            lines.append(
                f"     /backtest_analog_detail {r['symbol']} "
                f"{r['combo'].split()[0].lower()}"
            )
    else:
        lines.append("  → Khong co ma nao du dieu kien.")
        lines.append("     Thu tang days hoac kiem tra lai data.")

    lines.append("")
    lines.append("Ket qua IN-SAMPLE. Walk-forward 2025 moi la kiem chung thuc su.")
    lines.append(sep)

    return "\n".join(lines)


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
        equity   = np.cumprod([1.0 + r / 100.0 for r in sig_rets])
        peak     = np.maximum.accumulate(equity)
        max_dd   = float(np.min((equity - peak) / peak * 100))

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
        f"{'Nguong':<8} {'WR':<6} {'Exp':<8} {'Med':<8} {'Sharpe':<8} {'PF':<6} {'n':<5} {'MaxDD':<8} {'skip'}",
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
        f"MaxDD {best_exp_row['max_dd']:.1f}%)"
    )
    lines.append(
        f"  Risk-adjusted   : nguong {best_risk_adj['threshold']:.2f} "
        f"(Exp {best_risk_adj['mean_exp']:+.2f}%, PF {best_risk_adj['pf']:.2f}, "
        f"MaxDD {best_risk_adj['max_dd']:.1f}%)"
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
            f"(Exp {best_risk_adj['mean_exp']:+.2f}%, MaxDD {best_risk_adj['max_dd']:.1f}%)"
        )
        exp_diff = best_exp_row["mean_exp"] - best_risk_adj["mean_exp"]
        dd_diff  = abs(best_exp_row["max_dd"] - best_risk_adj["max_dd"])
        lines.append(
            f"  → Goi y: dung {best_risk_adj['threshold']:.2f} cho live trading "
            f"(Exp chi kem {exp_diff:.2f}% nhung MaxDD tot hon {dd_diff:.1f}%)"
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
_WF_SYMBOL_CONFIG = {
    "FPT": {"combo": "Macro Trend",       "threshold": 0.55},
    "MWG": {"combo": "Oversold Bounce",   "threshold": 0.55},
    "STB": {"combo": "Oversold Bounce",   "threshold": 0.60},
    "HPG": {"combo": "Volume Confirmed",  "threshold": 0.80},
}

# Giai đoạn OOS
_WF_START_DATE = "2025-01-01"


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
    MDS_DAYS    = 30
    FWD_DAYS    = 30

    oos_signals   = []   # tín hiệu trong OOS
    train_signals = []   # tín hiệu trong training (để so sánh)
    n_skip_oos    = 0
    n_skip_train  = 0

    for t_idx in range(120, n_bars - FWD_DAYS - 1, 7):
        if t_idx not in vectors:
            continue

        t_date    = pd.Timestamp(dates[t_idx])
        is_oos    = t_date >= wf_start

        target_vec = vectors[t_idx]
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
        # MaxDD equity curve
        equity   = np.cumprod([1.0 + r / 100.0 for r in actuals])
        peak     = np.maximum.accumulate(equity)
        max_dd   = float(np.min((equity - peak) / peak * 100)) if len(equity) > 1 else 0.0
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
        "status":        "ok",
        "symbol":        symbol,
        "combo":         combo_name,
        "threshold":     threshold,
        "wf_start":      _WF_START_DATE,
        "n_oos_bars":    n_oos_bars,
        "n_skip_oos":    n_skip_oos,
        "n_skip_train":  n_skip_train,
        "oos_signals":   oos_signals,
        "oos_metrics":   oos_metrics,
        "train_metrics": train_metrics,
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
            f"  MAE30 {train_m['mae30']:.1f}%  MaxDD {train_m['max_dd']:.1f}%"
        )
    else:
        lines.append("  Khong du tin hieu training.")
    lines.append(sep2)

    # ── OOS metrics ───────────────────────────────────────────────────────────
    lines.append(f"OUT-OF-SAMPLE ({wf_start} → hom nay):")
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
            f"  MAE30 {oos_m['mae30']:.1f}%  MaxDD {oos_m['max_dd']:.1f}%"
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
            f"  MaxDD : OOS {oos_m['max_dd']:.1f}% vs Train {train_m['max_dd']:.1f}%  "
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

    lines.append("")
    lines.append(
        "Config da duoc fix truoc khi chay OOS — ket qua nay co gia tri thong ke."
    )
    lines.append(sep)
    return "\n".join(lines)


async def walkforward_analog_cmd(update, context):
    """
    /walkforward_analog [MA1 MA2 ...]

    Walk-forward OOS test tu 2025-01-01 den hom nay.
    Config co dinh theo ket qua backtest + detail.

    Vi du:
      /walkforward_analog              (chay ca 4 ma mac dinh)
      /walkforward_analog FPT MWG      (chi FPT va MWG)
      /walkforward_analog HPG
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
        # Lọc chỉ mã có config
        unknown = [s for s in symbols if s not in _WF_SYMBOL_CONFIG]
        symbols = [s for s in symbols if s in _WF_SYMBOL_CONFIG]
        if unknown:
            await update.message.reply_text(
                f"Cac ma chua co config: {', '.join(unknown)}\n"
                f"Ma co san: {', '.join(_WF_SYMBOL_CONFIG)}"
            )
            if not symbols:
                return

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_backtest_analog_wf.get(user_id, 0)
    if since < BACKTEST_ANALOG_WF_COOLDOWN:
        wait = int(BACKTEST_ANALOG_WF_COOLDOWN - since)
        await update.message.reply_text(f"Vui long cho {wait}s truoc khi chay tiep.")
        return
    _last_backtest_analog_wf[user_id] = time.time()

    chat_id = update.effective_chat.id
    msg     = await update.message.reply_text(
        f"Walk-forward OOS: {', '.join(symbols)}\n"
        f"Giai doan: {_WF_START_DATE} → hom nay\n"
        f"Uoc tinh ~2-4 phut/ma..."
    )

    async def _bg():
        try:
            all_results = {}
            for i, symbol in enumerate(symbols, 1):
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg.message_id,
                        text=(
                            f"Walk-forward: {symbol} ({i}/{len(symbols)})\n"
                            f"Config: {_WF_SYMBOL_CONFIG[symbol]['combo']} "
                            f"| nguong {_WF_SYMBOL_CONFIG[symbol]['threshold']}\n"
                            f"Dang tinh..."
                        ),
                    )
                except Exception:
                    pass

                res = await asyncio.to_thread(_run_walkforward_sync, symbol)
                all_results[symbol] = res

                # Gửi kết quả từng mã ngay khi xong
                if res["status"] == "error":
                    text = f"❌ {symbol}: {res.get('error','?')[:200]}"
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

            # Nếu nhiều mã → gửi tổng kết
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
                    if res["status"] == "error":
                        fail_syms.append(f"{symbol}(loi)")
                        continue
                    oos_m = res.get("oos_metrics")
                    if not oos_m:
                        fail_syms.append(f"{symbol}(chua du tin hieu)")
                        continue
                    train_m  = res.get("train_metrics", {}) or {}
                    exp_ok   = oos_m["mean_exp"] > 0
                    pf_ok    = oos_m["pf"] >= 1.5
                    exp_deg  = oos_m["mean_exp"] >= (train_m.get("mean_exp", 0) or 0) * 0.60
                    pf_deg   = oos_m["pf"] >= (train_m.get("pf", 0) or 0) * 0.50

                    cfg = _WF_SYMBOL_CONFIG[symbol]
                    line = (
                        f"  {symbol}: "
                        f"Exp {oos_m['mean_exp']:+.2f}%  "
                        f"PF {oos_m['pf']:.2f}  "
                        f"WR {oos_m['wr']:.0f}%  "
                        f"n={oos_m['n']}"
                    )
                    if exp_ok and pf_ok and exp_deg and pf_deg:
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
                    summary_lines.append(f"FAIL    : {', '.join(fail_syms)}")
                if pass_syms:
                    summary_lines.append("")
                    summary_lines.append(
                        f"Ma du dieu kien live trading: {', '.join(pass_syms)}"
                    )

                # Edit message loading thành tổng kết
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
            # Chọn threshold: Exp cao nhất, PF tiebreak, MaxDD tốt nhất trong 90% Exp max
            max_exp    = max(r["mean_exp"] for r in thresh_rows)
            top_thresh = [r for r in thresh_rows if r["mean_exp"] >= max_exp * 0.90]
            best_thresh_row = max(top_thresh, key=lambda x: x["max_dd"])  # MaxDD ít âm nhất
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
    MDS_DAYS    = 30
    FWD_DAYS    = 30

    oos_signals   = []
    train_signals = []

    for t_idx in range(120, n_bars - FWD_DAYS - 1, 7):
        if t_idx not in vectors:
            continue
        t_date = pd.Timestamp(dates[t_idx])
        is_oos = t_date >= wf_start

        target_vec = vectors[t_idx]
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
        equity   = np.cumprod([1.0 + r / 100.0 for r in actuals])
        peak     = np.maximum.accumulate(equity)
        max_dd   = float(np.min((equity - peak) / peak * 100)) if len(equity) > 1 else 0.0
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
