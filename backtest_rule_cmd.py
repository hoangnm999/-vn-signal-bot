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
    if not symbol.isalnum() or len(symbol) > 10:
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
) -> str:
    """Tạo text output đầy đủ cho /backtest_rule."""
    ret   = metrics.get("total_return_pct", 0)
    cagr  = metrics.get("cagr_pct", 0)
    sh    = metrics.get("sharpe_ratio", 0)
    mdd   = metrics.get("max_drawdown_pct", 0)
    wr    = metrics.get("win_rate_pct", 0)
    pf    = metrics.get("profit_factor", 0)
    vol   = metrics.get("volatility_ann_pct", 0)
    cal   = metrics.get("calmar_ratio", 0)
    cap   = metrics.get("initial_capital_vnd", 100_000_000)
    fin   = metrics.get("final_equity_vnd", cap)
    aw    = metrics.get("avg_win_pct", 0)
    al    = metrics.get("avg_loss_pct", 0)
    pnl   = fin - cap

    grade = "A" if sh > 1.5 and mdd > -20 else \
            "B" if sh > 0.8 else \
            "C" if sh > 0 else "D"

    em_r  = "🟢" if ret > 0 else "🔴"

    # Dynamic exit summary
    dyn_parts = []
    if exit_ec_exit.trailing_stop_pct:
        dyn_parts.append(f"TrailingStop {exit_ec_exit.trailing_stop_pct:.1f}%")
    if exit_ec_exit.take_profit_pct:
        dyn_parts.append(f"TakeProfit {exit_ec_exit.take_profit_pct:.1f}%")
    if exit_ec_exit.stop_loss_pct:
        dyn_parts.append(f"StopLoss {exit_ec_exit.stop_loss_pct:.1f}%")
    if exit_ec_exit.hold_bars:
        dyn_parts.append(f"MaxHold {exit_ec_exit.hold_bars}bar")
    dyn_str = " + ".join(dyn_parts) if dyn_parts else "Khong"

    lines = [
        f"BACKTEST RULE: {symbol} | {days}D ({bars} bar)",
        "═" * 40,
        f"ENTRY: {entry_rule[:60]}",
        f"EXIT : {exit_rule[:60]}",
        f"Dynamic exit: {dyn_str}",
        "─" * 40,
        f"Xep hang    : [{grade}]",
        "",
        "HIEU SUAT:",
        f"  Loi nhuan  : {em_r} {ret:+.2f}%  ({pnl:+,.0f} VND)",
        f"  CAGR/nam   : {cagr:+.2f}%",
        f"  Sharpe     : {sh:.3f}",
        f"  Calmar     : {cal:.3f}",
        f"  Vol/nam    : {vol:.1f}%",
        "",
        "RUI RO:",
        f"  Max DD     : {mdd:.2f}%",
        "",
        "LENH GIAO DICH:",
        f"  Tin hieu   : {n_buy_signals} MUA / {n_sell_signals} BAN",
        f"  Thuc hien  : {n_trades} lenh",
        f"  Win Rate   : {wr:.1f}%",
        f"  TB thang   : {aw:+.2f}%",
        f"  TB thua    : {al:+.2f}%",
        f"  ProfitFac  : {pf:.2f}",
        "",
        f"Von ban dau : {cap:,.0f} VND",
        f"Von cuoi    : {fin:,.0f} VND",
        "═" * 40,
    ]

    # Exit reasons breakdown (nếu có dynamic exit)
    if dyn_parts and trades_sample:
        sells = [t for t in trades_sample if t.get("type") == "SELL"]
        reasons: dict[str, int] = {}
        for t in sells:
            r = t.get("exit_by", "Signal")
            reasons[r] = reasons.get(r, 0) + 1
        if reasons:
            lines.append("LY DO THOAT LENH:")
            for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {r}: {cnt} lan")
            lines.append("─" * 40)

    lines.append("Bieu do equity curve da duoc gui kem.")
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
    from vn_loader import load_vn_ohlcv
    from rule_engine import (
        generate_rule_signals, apply_dynamic_exit, parse_rule,
        format_rule_explanation,
    )
    from backtest_engine import _run_backtest_core, _calc_metrics, _draw_chart

    config = {
        "symbol":           symbol,
        "initial_capital":  100_000_000,
        "commission_pct":   0.0015,
        "slippage_pct":     0.001,
        "position_size":    1.0,
        "allow_short":      False,
        "days":             days,
    }

    # 1. Load data
    df = load_vn_ohlcv(symbol, days=days)

    # 2. Parse & compile rules → signals
    signals, entry_ec, exit_ec, ctx = generate_rule_signals(df, entry_rule, exit_rule)

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
    }


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def backtest_rule_cmd(update, context):
    """
    /backtest_rule <MA> "<entry_rule>" "<exit_rule>" [ngay]

    Vi du:
      /backtest_rule VCB "rsi < 30" "rsi > 70"
      /backtest_rule HPG "rsi < 30 and close > sma20" "rsi > 70 or trailing_stop(5%)"
      /backtest_rule STB "close < 62100 and rsi < 30" "rsi > 70 or close > 68000" 500
    """
    from telegram import Update
    from telegram.ext import ContextTypes

    # Authorization (gọi từ bot.py context)
    try:
        from bot import is_allowed, _deny, plain
    except ImportError:
        # Fallback nếu import cycle
        def is_allowed(_): return True
        async def _deny(_): pass
        plain = _plain

    if not is_allowed(update):
        await _deny(update); return

    args = context.args or []

    # Không có args → help
    if not args:
        await update.message.reply_text(_HELP_TEXT[:4096])
        return

    # Help explicit
    if args[0].lower() in ("help", "--help", "-h", "?"):
        await update.message.reply_text(_HELP_TEXT[:4096])
        return

    # Parse args
    parsed = _parse_args(args)
    if isinstance(parsed, str):
        # Error
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
            summary = _format_rule_result(
                metrics        = result["metrics"],
                symbol         = symbol,
                entry_rule     = entry_rule,
                exit_rule      = exit_rule,
                days           = days,
                n_trades       = result["n_trades"],
                exit_ec_entry  = result["entry_ec"],
                exit_ec_exit   = result["exit_ec"],
                trades_sample  = result["trades"],
                n_buy_signals  = result["n_buy"],
                n_sell_signals = result["n_sell"],
                bars           = result["bars"],
            )

            # Gửi metrics text
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text=plain(summary)[:4096],
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
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg.message_id,
                    text=f"Loi xu ly /backtest_rule: {str(e)[:300]}",
                )
            except Exception:
                pass

    asyncio.create_task(_bg())
