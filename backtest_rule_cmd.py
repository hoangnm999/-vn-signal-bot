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

# ── Hermes notify helper ───────────────────────────────────────────────────────

def _notify_hermes_bg(text: str) -> None:
    """
    Push kết quả sang Hermes /notify để phân tích tự động trong group.
    Chạy trong thread riêng — không block, không crash nếu Hermes offline.
    """
    import os as _os
    import requests as _req
    hermes_url = _os.environ.get(
        "HERMES_NOTIFY_URL",
        "https://hermes-telegram-bot-hnl.onrender.com/notify"
    )
    secret = _os.environ.get("NOTIFY_SECRET", "")
    try:
        payload = {"signal": text[:8000]}
        if secret:
            payload["secret"] = secret
        resp = _req.post(hermes_url, json=payload, timeout=8)
        logger.info(f"[HermesNotify] {resp.status_code} — {len(text)} ký tự")
    except Exception as e:
        logger.debug(f"[HermesNotify] Non-critical: {e}")



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
            import threading as _th
            _th.Thread(target=_notify_hermes_bg, args=(full_text,), daemon=True).start()

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
            import threading as _th
            _th.Thread(target=_notify_hermes_bg, args=(plain_fn(report),), daemon=True).start()

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
            # Notify Hermes phân tích
            import threading as _th
            _th.Thread(target=_notify_hermes_bg, args=(plain(summary),), daemon=True).start()

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
# ANALOG ENGINE V3 — Regime + Trigger (thay thế V2)
# ══════════════════════════════════════════════════════════════════════════════
#
# Thiết kế dựa trên phân tích dữ liệu thực (analyze_features.py):
#
#   Vấn đề V2: cosine similarity trên toàn vector → threshold vô nghĩa
#   vì các chỉ số thay đổi chậm làm mọi ngày "trông giống nhau"
#
#   V3 giải quyết bằng 2 tầng tách biệt:
#
#   Tầng 1 — REGIME FILTER (chỉ số chậm, ổn định):
#     Xác định thị trường đang ở trạng thái nền nào
#     Mỗi mã có regime riêng dựa trên phân tích dữ liệu
#
#   Tầng 2 — TRIGGER DETECTION (chỉ số nhanh, thay đổi hàng ngày):
#     Trong regime đó, hôm nay có tín hiệu đột biến không?
#     Cần >= 2 triggers để phát signal
#
#   Bằng chứng từ data (FWD=10 ngay, training 2019-2024):
#     MWG: Lift +2.73% | STB: Lift +1.32% | DPM: Lift +5.19%
#
# ══════════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────
_FWD_DAYS          = 10    # forward return window (bars) — hold 10 ngay
_WF_START_DATE     = "2025-01-01"
_WF_COOLDOWN_BARS  = 5     # 5 bars ~ 7 calendar days
_MIN_TRIGGERS      = 2     # so luong triggers can de phat signal
_TRIGGER_PCT       = 70    # top 30% cua phan phoi = "dot bien"
_WIN_THRESH        = 1.0   # win = return > 1%

# Cooldown
BACKTEST_ANALOG_COOLDOWN    = 0
BACKTEST_ANALOG_WF_COOLDOWN = 0
ANALOG_PIPELINE_COOLDOWN    = 0
PIPELINE_MAX_SYMBOLS        = 5
_last_backtest_analog:    dict[str, float] = {}
_last_backtest_analog_wf: dict[str, float] = {}
_last_analog_pipeline:    dict[str, float] = {}

# Pipeline filter
_PIPELINE_MIN_EXP = 1.0
_PIPELINE_MIN_WR  = 0.50
_PIPELINE_MIN_PF  = 1.3

# ── Regime Config — dựa trên phân tích dữ liệu thực ─────────────────────────
# regime_indicator: chỉ số xác định trạng thái nền
# regime_condition: "low" = dưới median, "high" = trên median
# trigger_indicators: chỉ số nhanh dùng để detect tín hiệu đột biến
_REGIME_CONFIG_DEFAULT: dict = {
    # MWG: mean reversion — stoch_k oversold WR=76.2% Lift=+3.19%
    "MWG": {
        "regime_indicator":  "atr_ratio",
        "regime_condition":  "low",
        "regime_label":      "Tich luy (ATR thap)",
        "trigger_indicators": ["stoch_k", "momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction": {
            "stoch_k":      "low",
            "momentum_5d":  "high",
            "volume_spike": "high",
            "candle_body":  "high",
        },
    },
    # STB: momentum/breakout — bo stoch_k vi oversold = nguy hiem voi STB
    "STB": {
        "regime_indicator":  "atr_ratio",
        "regime_condition":  "high",
        "regime_label":      "Bien dong cao (ATR cao)",
        "trigger_indicators": ["momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction": {
            "momentum_5d":  "high",
            "volume_spike": "high",
            "candle_body":  "high",
        },
    },
    # DPM: mean reversion co cau truc — stoch_k oversold Lift=+1.03%
    "DPM": {
        "regime_indicator":  "trend_slope",
        "regime_condition":  "low",
        "regime_label":      "Downtrend / Sideways",
        "trigger_indicators": ["momentum_5d", "volume_spike", "stoch_k", "candle_body"],
        "trigger_direction": {
            "momentum_5d":  "high",
            "volume_spike": "high",
            "stoch_k":      "low",
            "candle_body":  "high",
        },
    },
    # HPG: tuong tu MWG — mean reversion
    "HPG": {
        "regime_indicator":  "atr_ratio",
        "regime_condition":  "low",
        "regime_label":      "Tich luy (ATR thap)",
        "trigger_indicators": ["stoch_k", "momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction": {
            "stoch_k":      "low",
            "momentum_5d":  "high",
            "volume_spike": "high",
            "candle_body":  "high",
        },
    },
    # GAS: tuong tu MWG — tich luy
    "GAS": {
        "regime_indicator":  "atr_ratio",
        "regime_condition":  "low",
        "regime_label":      "Tich luy (ATR thap)",
        "trigger_indicators": ["stoch_k", "momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction": {
            "stoch_k":      "low",
            "momentum_5d":  "high",
            "volume_spike": "high",
            "candle_body":  "high",
        },
    },
    # DCM: tuong tu DPM — downtrend/sideways
    "DCM": {
        "regime_indicator":  "trend_slope",
        "regime_condition":  "low",
        "regime_label":      "Downtrend / Sideways",
        "trigger_indicators": ["momentum_5d", "volume_spike", "stoch_k", "candle_body"],
        "trigger_direction": {
            "momentum_5d":  "high",
            "volume_spike": "high",
            "stoch_k":      "low",
            "candle_body":  "high",
        },
    },
    # FPT: tuong tu STB — momentum/breakout
    "FPT": {
        "regime_indicator":  "atr_ratio",
        "regime_condition":  "high",
        "regime_label":      "Bien dong cao (ATR cao)",
        "trigger_indicators": ["momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction": {
            "momentum_5d":  "high",
            "volume_spike": "high",
            "candle_body":  "high",
        },
    },
}
_REGIME_CONFIG: dict = dict(_REGIME_CONFIG_DEFAULT)

# Default cho mã chưa có config riêng
_REGIME_CONFIG_FALLBACK = {
    "regime_indicator":   "atr_ratio",
    "regime_condition":   "low",
    "regime_label":       "Default (ATR thap)",
    "trigger_indicators": ["momentum_5d", "volume_spike", "stoch_k", "candle_body"],
}

# Pending approve state
_pending_approve: dict[str, dict] = {}


# ── Tính indicators V3 ────────────────────────────────────────────────────────

def _ema_v3(c, span):
    return pd.Series(c).ewm(span=span, adjust=False).mean().values

def _sma_v3(c, p):
    return pd.Series(c).rolling(p, min_periods=p).mean().values

def _compute_indicators_v3(df: pd.DataFrame) -> list[dict]:
    """
    Tính indicators cho toàn bộ df.
    Trả về list dict, mỗi dict là 1 ngày với đầy đủ chỉ số.
    """
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    vol    = df["volume"].values.astype(float)
    opn    = df["open"].values.astype(float)
    n      = len(df)

    ema12  = _ema_v3(close, 12)
    ema26  = _ema_v3(close, 26)
    sma20  = _sma_v3(close, 20)
    sma50  = _sma_v3(close, 50)
    vsma5  = _sma_v3(vol, 5)
    vsma20 = _sma_v3(vol, 20)

    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low, np.maximum(
             np.abs(high - h_prev), np.abs(low - h_prev)))
    atr    = _sma_v3(tr, 14)

    delta  = np.diff(close, prepend=close[0])
    gain   = np.where(delta > 0, delta, 0.0)
    loss   = np.where(delta < 0, -delta, 0.0)
    avg_g  = _ema_v3(gain, 14)
    avg_l  = _ema_v3(loss, 14)
    avg_l  = np.where(avg_l == 0, 1e-9, avg_l)
    rsi    = 100 - 100 / (1 + avg_g / avg_l)

    lo14   = pd.Series(low).rolling(14).min().values
    hi14   = pd.Series(high).rolling(14).max().values
    denom  = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch  = 100 * (close - lo14) / denom

    rows = []
    for i in range(60, n):
        px     = close[i]
        atr_v  = atr[i] if np.isfinite(atr[i]) else px * 0.02
        s20    = sma20[i] if np.isfinite(sma20[i]) else px
        s50    = sma50[i] if np.isfinite(sma50[i]) else px
        vs20_v = vsma20[i] if np.isfinite(vsma20[i]) else vol[i]
        vs5_v  = vsma5[i] if np.isfinite(vsma5[i]) else vol[i]
        c5     = close[max(i - 5, 0)]

        body = abs(px - opn[i]) / (atr_v + 1e-9)

        rows.append({
            "idx":          i,
            # Regime indicators (cham)
            "atr_ratio":    float(atr_v / (px + 1e-9) * 100),
            "trend_slope":  float((s20 - s50) / (px + 1e-9) * 100),
            "momentum_20d": float((px / close[max(i - 20, 0)] - 1) * 100),
            # Trigger indicators (nhanh)
            "momentum_5d":  float((px / (c5 + 1e-9) - 1) * 100),
            "volume_spike": float((vol[i] / (vs20_v + 1e-9)) - 1.0),
            "stoch_k":      float(stoch[i]),
            "candle_body":  float(np.clip(body, 0, 3)),
        })
    return rows


# ── Regime & Trigger helpers ──────────────────────────────────────────────────

def _compute_regime_threshold(ind_rows: list[dict], regime_indicator: str) -> float:
    """Tính median của regime indicator trên training data."""
    vals = [r[regime_indicator] for r in ind_rows
            if regime_indicator in r and np.isfinite(r[regime_indicator])]
    return float(np.median(vals)) if vals else 0.0


def _compute_trigger_thresholds(
    ind_rows: list[dict],
    trigger_indicators: list[str],
    trigger_direction: dict[str, str] | None = None,
) -> dict[str, float]:
    """
    Tinh nguong cho tung trigger indicator.
    direction=high: percentile cao (top 30%)
    direction=low:  percentile thap (bottom 30% = oversold)
    """
    thresholds = {}
    for trig in trigger_indicators:
        vals = [r[trig] for r in ind_rows
                if trig in r and np.isfinite(r[trig])]
        if not vals:
            continue
        direction = (trigger_direction or {}).get(trig, "high")
        if direction == "low":
            # Bottom 30% = nguong oversold
            thresholds[trig] = float(np.percentile(vals, 100 - _TRIGGER_PCT))
        else:
            # Top 30%
            thresholds[trig] = float(np.percentile(vals, _TRIGGER_PCT))
    return thresholds


def _check_regime(row: dict, regime_indicator: str,
                  regime_condition: str, threshold: float) -> bool:
    """Kiểm tra ngày T có trong regime không."""
    val = row.get(regime_indicator, np.nan)
    if not np.isfinite(val):
        return False
    if regime_condition == "low":
        return val <= threshold
    else:
        return val > threshold


def _count_triggers(row: dict, trigger_indicators: list[str],
                    thresholds: dict[str, float],
                    trigger_direction: dict[str, str] | None = None) -> int:
    """
    Dem so triggers dang active tai ngay T.
    trigger_direction: "high" = val >= thresh, "low" = val <= thresh
    Mac dinh la "high" neu khong co config.
    """
    count = 0
    for trig in trigger_indicators:
        val    = row.get(trig, np.nan)
        thresh = thresholds.get(trig, np.nan)
        if not (np.isfinite(val) and np.isfinite(thresh)):
            continue
        direction = (trigger_direction or {}).get(trig, "high")
        if direction == "low":
            if val <= thresh:
                count += 1
        else:
            if val >= thresh:
                count += 1
    return count


# ══════════════════════════════════════════════════════════════════════════════
# TẦNG 1: BACKTEST V3
# ══════════════════════════════════════════════════════════════════════════════

def _run_analog_backtest_sync(symbol: str, days: int = 1800) -> dict:
    """
    Backtest V3: kiểm tra regime + trigger trên training data.

    Logic:
      Với mỗi ngày T (step=7 bars, simulate weekly review):
        1. Kiểm tra regime → skip nếu không đúng
        2. Đếm triggers → skip nếu < MIN_TRIGGERS
        3. Tính actual return sau FWD_DAYS

    Skip breakdown:
      n_skip_regime  : không đúng regime
      n_skip_trigger : đúng regime nhưng < 2 triggers
      n_signals      : thực sự tạo signal
    """
    symbol = symbol.upper()

    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=days, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu"}

    df["date"] = pd.to_datetime(df["date"])
    wf_start   = pd.Timestamp(_WF_START_DATE)
    oos_mask   = df["date"] >= wf_start
    oos_start  = int(oos_mask.idxmax()) if oos_mask.any() else len(df)

    # Chỉ dùng training data
    train_df  = df.iloc[:oos_start].reset_index(drop=True)
    n_bars    = len(train_df)
    close_arr = train_df["close"].values.astype(float)
    dates     = train_df["date"].values

    if n_bars < 200:
        return {"status": "error", "error": "Khong du training data"}

    # Tính indicators trên training
    ind_rows = _compute_indicators_v3(train_df)
    ind_map  = {r["idx"]: r for r in ind_rows}

    # Lấy config regime
    cfg = _REGIME_CONFIG.get(symbol, _REGIME_CONFIG_FALLBACK)
    reg_ind   = cfg["regime_indicator"]
    reg_cond  = cfg["regime_condition"]
    trig_inds  = cfg["trigger_indicators"]
    trig_dir   = cfg.get("trigger_direction", {})

    # Tính ngưỡng từ training data
    reg_thresh  = _compute_regime_threshold(ind_rows, reg_ind)
    trig_thresh = _compute_trigger_thresholds(ind_rows, trig_inds, trig_dir)

    # Backtest loop
    n_skip_regime  = 0
    n_skip_trigger = 0
    signals        = []

    for t_idx in range(120, n_bars - _FWD_DAYS - 1, 3):
        row = ind_map.get(t_idx)
        if row is None:
            continue

        # Tầng 1: regime filter
        if not _check_regime(row, reg_ind, reg_cond, reg_thresh):
            n_skip_regime += 1
            continue

        # Tầng 2: trigger
        n_trig = _count_triggers(row, trig_inds, trig_thresh, trig_dir)
        if n_trig < _MIN_TRIGGERS:
            n_skip_trigger += 1
            continue

        # Tính actual return
        fwd_idx = t_idx + _FWD_DAYS
        if fwd_idx >= n_bars:
            continue

        actual_ret = (close_arr[fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100

        signals.append({
            "t_idx":      t_idx,
            "date":       str(pd.Timestamp(dates[t_idx]))[:10],
            "actual_ret": actual_ret,
            "n_trig":     n_trig,
            "regime_val": row.get(reg_ind, 0),
        })

    n_sig = len(signals)
    skip_info = {
        "n_skip_regime":  n_skip_regime,
        "n_skip_trigger": n_skip_trigger,
        "n_signals":      n_sig,
    }

    if n_sig < 3:
        return {
            "status": "error",
            "error":  f"Qua it signals ({n_sig}). "
                      f"skip_regime={n_skip_regime} skip_trigger={n_skip_trigger}",
            **skip_info,
        }

    rets  = [s["actual_ret"] for s in signals]
    wins  = [r for r in rets if r >= _WIN_THRESH]
    loss  = [r for r in rets if r < _WIN_THRESH]
    wr    = len(wins) / n_sig
    mean_r = float(np.mean(rets))
    med_r  = float(np.median(rets))
    std_r  = float(np.std(rets)) if n_sig > 1 else 1e-9
    pf     = sum(wins) / abs(sum(loss)) if loss else 99.0
    max_dd = float(np.min(rets))

    return {
        "status":         "ok",
        "symbol":         symbol,
        "regime":         cfg["regime_label"],
        "regime_thresh":  round(reg_thresh, 3),
        "trig_thresh":    {k: round(v, 3) for k, v in trig_thresh.items()},
        "n_bars":         n_bars,
        "wr":             round(wr * 100, 1),
        "mean_exp":       round(mean_r, 2),
        "med_exp":        round(med_r, 2),
        "pf":             round(pf, 2),
        "max_dd":         round(max_dd, 1),
        "signals":        signals,
        **skip_info,
    }


def _format_analog_backtest_result(res: dict) -> str:
    if res.get("status") == "error":
        return f"Loi backtest: {res['error']}"

    sym = res["symbol"]
    lines = [
        f"ANALOG BACKTEST V3 — {sym}",
        f"Regime: {res['regime']} (nguong={res['regime_thresh']:.2f})",
        f"FWD={_FWD_DAYS}d | MIN_TRIGGERS={_MIN_TRIGGERS}",
        f"{'─'*38}",
        f"Training: {res['n_bars']} bars",
        f"Skip: regime={res['n_skip_regime']} trigger={res['n_skip_trigger']}",
        f"Signals: n={res['n_signals']} WR={res['wr']}% "
        f"Exp={res['mean_exp']:+.2f}% PF={res['pf']:.2f} MaxDD={res['max_dd']:.1f}%",
        f"{'─'*38}",
        "Recent signals:",
    ]
    for s in res.get("signals", [])[-5:]:
        lines.append(
            f"  {s['date']} trig={s['n_trig']} → {s['actual_ret']:+.1f}%"
        )
    lines.append(f"\nDung /walkforward_analog {sym} de validate OOS")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# TẦNG 2: WALK-FORWARD V3
# ══════════════════════════════════════════════════════════════════════════════

def _run_walkforward_sync(symbol: str) -> dict:
    """
    Walk-forward OOS V3:
      - Ngưỡng regime và trigger được tính từ TRAINING (2019-2024)
        → không dùng OOS data để tính ngưỡng (tránh leakage)
      - OOS loop: step=1 bar, cooldown=5 bars
      - Signal khi: đúng regime + >= MIN_TRIGGERS triggers
    """
    symbol = symbol.upper()
    cfg    = _REGIME_CONFIG.get(symbol, _REGIME_CONFIG_FALLBACK)

    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=200)
    except Exception as e:
        return {"status": "error", "error": f"Khong lay duoc data: {e}"}

    if df is None or len(df) < 200:
        return {"status": "error", "error": "Khong du du lieu"}

    df["date"] = pd.to_datetime(df["date"])
    wf_start   = pd.Timestamp(_WF_START_DATE)
    oos_mask   = df["date"] >= wf_start
    oos_start  = int(oos_mask.idxmax()) if oos_mask.any() else len(df)

    if oos_start >= len(df):
        return {"status": "error", "error": f"Khong co data sau {_WF_START_DATE}"}

    # Tính ngưỡng CHỈ từ training
    train_df  = df.iloc[:oos_start].reset_index(drop=True)
    train_rows = _compute_indicators_v3(train_df)

    reg_ind    = cfg["regime_indicator"]
    reg_cond   = cfg["regime_condition"]
    trig_inds  = cfg["trigger_indicators"]
    trig_dir   = cfg.get("trigger_direction", {})
    reg_thresh  = _compute_regime_threshold(train_rows, reg_ind)
    trig_thresh = _compute_trigger_thresholds(train_rows, trig_inds, trig_dir)

    # Tính indicators toàn bộ df (cả OOS)
    n_bars    = len(df)
    close_arr = df["close"].values.astype(float)
    dates     = df["date"].values
    all_rows  = _compute_indicators_v3(df)
    ind_map   = {r["idx"]: r for r in all_rows}

    # ── Training metrics (để compare) ────────────────────────────────────────
    train_signals = []
    for t_idx in range(120, oos_start - _FWD_DAYS - 1, 3):
        row = ind_map.get(t_idx)
        if row is None: continue
        if not _check_regime(row, reg_ind, reg_cond, reg_thresh): continue
        if _count_triggers(row, trig_inds, trig_thresh, trig_dir) < _MIN_TRIGGERS: continue
        fwd_idx = t_idx + _FWD_DAYS
        if fwd_idx >= n_bars: continue
        ret = (close_arr[fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100
        train_signals.append({"actual": ret})

    # ── OOS loop ──────────────────────────────────────────────────────────────
    oos_signals     = []
    n_skip_cooldown = 0
    n_skip_regime   = 0
    n_skip_trigger  = 0
    last_signal_bar = None

    for t_idx in range(max(oos_start, 120), n_bars - _FWD_DAYS - 1):
        row = ind_map.get(t_idx)
        if row is None: continue

        # Cooldown
        if last_signal_bar and (t_idx - last_signal_bar) < _WF_COOLDOWN_BARS:
            n_skip_cooldown += 1
            continue

        # Tầng 1: regime
        if not _check_regime(row, reg_ind, reg_cond, reg_thresh):
            n_skip_regime += 1
            continue

        # Tầng 2: trigger
        n_trig = _count_triggers(row, trig_inds, trig_thresh, trig_dir)
        if n_trig < _MIN_TRIGGERS:
            n_skip_trigger += 1
            continue

        # Signal
        fwd_idx = t_idx + _FWD_DAYS
        pending = fwd_idx >= n_bars
        actual  = None if pending else (
            (close_arr[fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100
        )

        oos_signals.append({
            "t_idx":   t_idx,
            "date":    str(pd.Timestamp(dates[t_idx]))[:10],
            "n_trig":  n_trig,
            "actual":  actual,
            "pending": pending,
        })
        last_signal_bar = t_idx

    # ── Metrics ───────────────────────────────────────────────────────────────
    def _calc(sigs):
        done = [s for s in sigs if not s.get("pending") and s.get("actual") is not None]
        if not done: return {}
        rets = [s["actual"] for s in done]
        wins = [r for r in rets if r >= _WIN_THRESH]
        loss = [r for r in rets if r < _WIN_THRESH]
        return {
            "n":         len(done),
            "n_pending": len(sigs) - len(done),
            "wr":        round(len(wins) / len(rets) * 100, 1),
            "mean_exp":  round(float(np.mean(rets)), 2),
            "pf":        round(sum(wins) / abs(sum(loss)), 2) if loss else 99.0,
            "max_dd":    round(float(np.min(rets)), 1),
        }

    train_m = _calc([{"actual": s["actual"], "pending": False}
                     for s in train_signals])
    oos_m   = _calc(oos_signals)

    return {
        "status":          "ok",
        "symbol":          symbol,
        "regime":          cfg["regime_label"],
        "regime_thresh":   round(reg_thresh, 3),
        "oos_start":       _WF_START_DATE,
        "train_metrics":   train_m,
        "oos_metrics":     oos_m,
        "oos_signals":     oos_signals,
        "n_skip_cooldown": n_skip_cooldown,
        "n_skip_regime":   n_skip_regime,
        "n_skip_trigger":  n_skip_trigger,
    }


def _format_wf_result(res: dict) -> str:
    if res.get("status") == "error":
        return f"Loi WF: {res['error']}"

    sym = res["symbol"]
    tm  = res.get("train_metrics", {})
    om  = res.get("oos_metrics", {})

    lines = [
        f"WALK-FORWARD V3 — {sym}",
        f"Regime: {res['regime']}",
        f"OOS tu: {res['oos_start']} | FWD={_FWD_DAYS}d | Min triggers={_MIN_TRIGGERS}",
        f"{'─'*38}",
        f"TRAINING: n={tm.get('n','?')} WR={tm.get('wr','?')}% "
        f"Exp={tm.get('mean_exp','?'):+}% PF={tm.get('pf','?')}",
        f"{'─'*38}",
        f"OOS: n={om.get('n','?')} (pending={om.get('n_pending','?')}) "
        f"WR={om.get('wr','?')}% Exp={om.get('mean_exp','?'):+}% PF={om.get('pf','?')}",
        f"MaxDD={om.get('max_dd','?')}%",
        f"{'─'*38}",
        f"OOS Skip: cooldown={res['n_skip_cooldown']} "
        f"regime={res['n_skip_regime']} trigger={res['n_skip_trigger']}",
        f"{'─'*38}",
    ]

    # Verdict
    n_oos = om.get("n", 0)
    if n_oos < 3:
        verdict = "Qua it signal OOS — chua du tin cay"
    elif om.get("mean_exp", 0) > _PIPELINE_MIN_EXP and om.get("pf", 0) >= _PIPELINE_MIN_PF:
        verdict = "OOS PASS — du dieu kien live trading"
    elif om.get("mean_exp", 0) > 0:
        verdict = "OOS duong nhung yeu — theo doi them"
    else:
        verdict = "OOS am — xem lai regime config"

    lines.append(verdict)
    lines.append("")
    lines.append("Recent OOS signals:")

    for s in res.get("oos_signals", [])[-5:]:
        ret_txt = f"{s['actual']:+.1f}%" if s["actual"] is not None else "pending"
        lines.append(f"  {s['date']} trig={s['n_trig']} → {ret_txt}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE V3 — Tự động tìm regime config tốt nhất
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline_wf_sync(symbol: str) -> dict:
    """
    Pipeline cho mã chưa có config:
      Thử tất cả regime combinations → chọn tốt nhất → WF OOS
    """
    symbol = symbol.upper()

    # Thử các regime config khác nhau
    candidates = [
        {"regime_indicator": "atr_ratio",   "regime_condition": "low",
         "regime_label": "ATR thap",
         "trigger_indicators": ["momentum_5d","volume_spike","stoch_k","candle_body"]},
        {"regime_indicator": "atr_ratio",   "regime_condition": "high",
         "regime_label": "ATR cao",
         "trigger_indicators": ["momentum_5d","volume_spike","stoch_k","candle_body"]},
        {"regime_indicator": "trend_slope", "regime_condition": "low",
         "regime_label": "Downtrend/Sideways",
         "trigger_indicators": ["momentum_5d","volume_spike","stoch_k","candle_body"]},
        {"regime_indicator": "trend_slope", "regime_condition": "high",
         "regime_label": "Uptrend",
         "trigger_indicators": ["momentum_5d","volume_spike","stoch_k","candle_body"]},
        {"regime_indicator": "momentum_20d","regime_condition": "low",
         "regime_label": "Momentum yeu",
         "trigger_indicators": ["momentum_5d","volume_spike","stoch_k","candle_body"]},
    ]

    best_cfg   = None
    best_score = -999

    for cand in candidates:
        _REGIME_CONFIG[symbol] = cand
        res = _run_analog_backtest_sync(symbol)
        if res.get("status") != "ok": continue
        score = res.get("mean_exp", 0) * (res.get("pf", 0) ** 0.5)
        if score > best_score and res.get("n_signals", 0) >= 5:
            best_score = score
            best_cfg   = cand

    if best_cfg is None:
        _REGIME_CONFIG.pop(symbol, None)
        return {"status": "skip", "reason": "Khong tim duoc regime phu hop"}

    _REGIME_CONFIG[symbol] = best_cfg
    result = _run_walkforward_sync(symbol)

    oos_pass = False
    if result.get("status") == "ok":
        om = result.get("oos_metrics") or {}
        oos_pass = (
            om.get("mean_exp", 0) > _PIPELINE_MIN_EXP
            and om.get("pf", 0) >= _PIPELINE_MIN_PF
        )

    if not oos_pass:
        _REGIME_CONFIG.pop(symbol, None)

    if result.get("status") == "ok":
        result["auto_config"] = True
        result["found_regime"] = best_cfg["regime_label"]
        result["oos_pass"]    = oos_pass

    return result


def _format_pipeline_summary(results: dict) -> str:
    lines = ["ANALOG PIPELINE V3 — Ket qua", "─" * 38]
    for sym, res in results.items():
        status = res.get("status")
        if status == "error":
            lines.append(f"{sym}: {res['error']}")
        elif status == "skip":
            lines.append(f"{sym}: {res['reason']}")
        else:
            om   = res.get("oos_metrics", {})
            flag = "PASS" if res.get("oos_pass") else "WEAK"
            lines.append(
                f"{flag} {sym}: {res.get('found_regime','?')}\n"
                f"  OOS n={om.get('n','?')} WR={om.get('wr','?')}% "
                f"Exp={om.get('mean_exp','?'):+}% PF={om.get('pf','?')}"
            )
    return "\n".join(lines)


def _load_wf_config_from_db():
    """Load regime config từ DB."""
    try:
        from db import load_analog_configs
        db_configs = load_analog_configs()
        if db_configs:
            for sym, cfg in db_configs.items():
                if sym not in _REGIME_CONFIG:
                    _REGIME_CONFIG[sym] = cfg
            logger.info(f"[RegimeConfig] Loaded {len(db_configs)} from DB")
    except Exception as e:
        logger.warning(f"[RegimeConfig] load error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

async def backtest_analog_cmd(update, context):
    """/backtest_analog <MA> — Backtest V3 regime+trigger."""
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    user_id = str(update.effective_user.id)
    if time.time() - _last_backtest_analog.get(user_id, 0) < BACKTEST_ANALOG_COOLDOWN:
        await update.message.reply_text("Vui long doi 5 phut.")
        return
    _last_backtest_analog[user_id] = time.time()

    args   = context.args or []
    symbol = args[0].upper() if args else None
    if not symbol:
        await update.message.reply_text("Cu phap: /backtest_analog <MA>")
        return

    msg = await update.message.reply_text(f"Dang chay backtest V3 cho {symbol}...")

    async def _bg():
        try:
            logger.info(f"[BacktestV3] Start {symbol}")
            res  = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: _run_analog_backtest_sync(symbol)
                ), timeout=300,
            )
            logger.info(f"[BacktestV3] Done {symbol}: {res.get('status')}")
            _analog_result_text = _plain(_format_analog_backtest_result(res))
            await msg.edit_text(_analog_result_text)
            import threading as _th
            _th.Thread(target=_notify_hermes_bg, args=(_analog_result_text,), daemon=True).start()
        except asyncio.TimeoutError:
            await msg.edit_text("Timeout — thu lai sau.")
        except Exception as e:
            logger.exception(f"[BacktestV3] Error {symbol}: {e}")
            await msg.edit_text(f"Loi: {e}")

    asyncio.create_task(_bg())


async def walkforward_analog_cmd(update, context):
    """/walkforward_analog [MA1 MA2 ...] — Walk-forward OOS V3."""
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    user_id = str(update.effective_user.id)
    if time.time() - _last_backtest_analog_wf.get(user_id, 0) < BACKTEST_ANALOG_WF_COOLDOWN:
        await update.message.reply_text("Vui long doi 5 phut.")
        return
    _last_backtest_analog_wf[user_id] = time.time()

    args    = context.args or []
    symbols = [a.upper() for a in args] if args else list(_REGIME_CONFIG.keys())

    msg = await update.message.reply_text(f"Walk-forward V3: {', '.join(symbols)}...")

    async def _bg():
        try:
            lines = []
            for sym in symbols:
                res = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sym: _run_walkforward_sync(s)
                )
                lines.append(_format_wf_result(res))
                lines.append("")
            _wf_text = _plain("\n".join(lines))[:4000]
            await msg.edit_text(_wf_text)
            import threading as _th
            _th.Thread(target=_notify_hermes_bg, args=(_wf_text,), daemon=True).start()
        except Exception as e:
            logger.exception(f"[WFV3] Error: {e}")
            await msg.edit_text(f"Loi: {e}")

    asyncio.create_task(_bg())


async def analog_pipeline_cmd(update, context):
    """/analog_pipeline <MA1> [MA2 ...] — Pipeline tu dong V3."""
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    user_id = str(update.effective_user.id)
    if time.time() - _last_analog_pipeline.get(user_id, 0) < ANALOG_PIPELINE_COOLDOWN:
        await update.message.reply_text("Vui long doi 10 phut.")
        return
    _last_analog_pipeline[user_id] = time.time()

    args    = context.args or []
    symbols = [a.upper() for a in args[:PIPELINE_MAX_SYMBOLS]]
    if not symbols:
        await update.message.reply_text(
            f"Cu phap: /analog_pipeline <MA1> [MA2 ...] (toi da {PIPELINE_MAX_SYMBOLS} ma)"
        )
        return

    msg = await update.message.reply_text(f"Pipeline V3: {', '.join(symbols)}...")

    async def _bg():
        try:
            results = {}
            for sym in symbols:
                results[sym] = await asyncio.get_event_loop().run_in_executor(
                    None, lambda s=sym: _run_pipeline_wf_sync(s)
                )
            _pipeline_text = _plain(_format_pipeline_summary(results))
            await msg.edit_text(_pipeline_text)
            import threading as _th
            _th.Thread(target=_notify_hermes_bg, args=(_pipeline_text,), daemon=True).start()
        except Exception as e:
            await msg.edit_text(f"Loi: {e}")

    asyncio.create_task(_bg())


async def analog_approve_cmd(update, context):
    """
    /analog_approve <MA>              — tu dong tim config tot nhat tu backtest
    /analog_approve <MA> <regime> <condition> — chi dinh tay
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
            "Cu phap: /analog_approve <MA>\n"
            "Vi du: /analog_approve MWG"
        )
        return

    symbol = args[0].upper()

    # Tu dong chay pipeline de tim config
    msg = await update.message.reply_text(
        f"Dang tim regime config cho {symbol}..."
    )
    try:
        loop   = asyncio.get_event_loop()
        res    = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _run_analog_backtest_sync(symbol)),
            timeout=300,
        )
    except asyncio.TimeoutError:
        await msg.edit_text("Timeout.")
        return
    except Exception as e:
        await msg.edit_text(f"Loi: {e}")
        return

    if res.get("status") == "error":
        await msg.edit_text(f"Loi: {res['error']}")
        return

    cfg = _REGIME_CONFIG.get(symbol, _REGIME_CONFIG_FALLBACK)

    # Luu pending
    _pending_approve[user_id] = {
        "symbol":  symbol,
        "regime":  cfg["regime_label"],
        "regime_indicator": cfg["regime_indicator"],
        "regime_condition": cfg["regime_condition"],
        "trigger_indicators": cfg["trigger_indicators"],
    }

    old_cfg = _REGIME_CONFIG_DEFAULT.get(symbol)
    old_txt = (
        f"Config cu: {old_cfg['regime_label']}\n" if old_cfg
        else "Config cu: (chua co)\n"
    )

    text = (
        f"XAC NHAN LUU CONFIG — {symbol}\n"
        f"{'─'*32}\n"
        f"{old_txt}"
        f"Config moi: {cfg['regime_label']}\n"
        f"  n_signals={res['n_signals']} WR={res['wr']}% "
        f"Exp={res['mean_exp']:+.2f}%\n"
        f"{'─'*32}\n"
        "Luu config nay vao DB khong?"
    )

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("Yes", callback_data=f"approve_yes_{user_id}"),
        InlineKeyboardButton("No",  callback_data=f"approve_no_{user_id}"),
    ]])
    await msg.delete()
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
        await query.edit_message_text("Phien het han. Chay lai /analog_approve.")
        return

    if data.startswith("approve_no_"):
        await query.edit_message_text(f"Da huy — config {pending['symbol']} khong duoc luu.")
        return

    symbol = pending["symbol"]
    cfg    = {
        "regime_indicator":   pending["regime_indicator"],
        "regime_condition":   pending["regime_condition"],
        "regime_label":       pending["regime"],
        "trigger_indicators": pending["trigger_indicators"],
    }
    try:
        from db import save_analog_config
        save_analog_config(symbol, cfg)
        _REGIME_CONFIG[symbol] = cfg
        await query.edit_message_text(
            f"Da luu config {symbol}: {cfg['regime_label']}\n\n"
            f"Chay /walkforward_analog {symbol} de validate OOS."
        )
    except Exception as e:
        await query.edit_message_text(f"Loi luu DB: {e}")


async def analog_configs_cmd(update, context):
    """/analog_configs — Xem tat ca regime config dang active."""
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    lines = ["ANALOG REGIME CONFIGS:", "─" * 35]
    for sym, cfg in sorted(_REGIME_CONFIG.items()):
        src = "default" if sym in _REGIME_CONFIG_DEFAULT else "custom"
        lines.append(f"  {sym}: {cfg['regime_label']} ({src})")
    await update.message.reply_text("\n".join(lines))


async def analog_remove_cmd(update, context):
    """/analog_remove <MA> — Xoa config khoi DB."""
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
        await update.message.reply_text("Cu phap: /analog_remove <MA>")
        return

    symbol = args[0].upper()
    try:
        from db import delete_analog_config
        delete_analog_config(symbol)
    except Exception as e:
        await update.message.reply_text(f"Loi xoa DB: {e}")
        return

    if symbol in _REGIME_CONFIG_DEFAULT:
        _REGIME_CONFIG[symbol] = dict(_REGIME_CONFIG_DEFAULT[symbol])
        status = f"Quay ve default: {_REGIME_CONFIG[symbol]['regime_label']}"
    else:
        _REGIME_CONFIG.pop(symbol, None)
        status = "Da xoa. Chay /analog_pipeline de tim config moi."

    await update.message.reply_text(f"{symbol}: {status}")


# ── Stubs tuong thich bot.py ──────────────────────────────────────────────────

async def backtest_analog_batch_cmd(update, context):
    await update.message.reply_text(
        "Dung /analog_pipeline thay the.\nVi du: /analog_pipeline MWG STB DPM"
    )

async def backtest_analog_detail_cmd(update, context):
    await update.message.reply_text(
        "V3 khong con dung combo/threshold.\n"
        "Dung /backtest_analog <MA> de xem ket qua regime+trigger."
    )

async def analog_regime_analysis_cmd(update, context):
    await update.message.reply_text(
        "Da gop vao /walkforward_analog.\n"
        "Dung: /walkforward_analog <MA>"
    )

async def analog_sim_dist_cmd(update, context):
    await update.message.reply_text(
        "Da loai bo trong V3.\n"
        "Dung /backtest_analog <MA> de xem phan tich regime+trigger."
    )
