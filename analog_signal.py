"""
analog_signal.py — Tầng 3: Daily Signal Scanner cho Analog System.

Cron 15:30 VN (08:30 UTC) hàng ngày:
  - Scan MWG, STB, HPG với config đã verify qua Tầng 1+2
  - Cooldown 7 ngày/mã để tránh echo signal
  - Push Telegram khi có signal chất lượng
  - Lưu DB để track kết quả thực tế

Lệnh:
  /signal_status   — xem signal đang active
  /signal_history  — lịch sử signal + kết quả thực
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config cố định — đã verify qua Tầng 1 + Tầng 2 ──────────────────────────
SIGNAL_SYMBOLS = {
    "MWG": {
        "combo":     "Oversold Bounce",
        "dims":      ["rsi_norm", "stoch_k_norm", "bb_position", "high_low_pos", "momentum_5d"],
        "threshold": 0.55,
        "t1_exp":    2.51,
        "t1_pf":     11.30,
        "oos_exp":   3.62,
        "oos_pf":    2.96,
        "mae30":     -6.2,
        "sizing":    1.0,   # full size
    },
    "STB": {
        "combo":     "Oversold Bounce",
        "dims":      ["rsi_norm", "stoch_k_norm", "bb_position", "high_low_pos", "momentum_5d"],
        "threshold": 0.60,
        "t1_exp":    1.53,
        "t1_pf":     5.76,
        "oos_exp":   6.25,
        "oos_pf":    3.93,
        "mae30":     -6.1,
        "sizing":    1.0,
    },
    "HPG": {
        "combo":     "Volume Confirmed",
        "dims":      ["volume_spike", "vol_trend", "momentum_5d", "momentum_20d", "rsi_norm"],
        "threshold": 0.80,
        "t1_exp":    1.42,
        "t1_pf":     3.80,
        "oos_exp":   2.46,
        "oos_pf":    2.15,
        "mae30":     -4.9,
        "sizing":    1.0,
    },
    "GAS": {
        "combo":     "No Volume",
        "dims":      [
            "rsi_norm", "macd_hist_norm", "bb_position", "trend_slope",
            "price_vs_sma20", "price_vs_sma50", "stoch_k_norm", "momentum_20d",
        ],
        "threshold": 0.55,
        "t1_exp":    1.50,
        "t1_pf":     2.45,
        "oos_exp":   4.49,
        "oos_pf":    1.79,
        "mae30":     -6.0,   # ước tính từ OOS, cần verify sau live trading
        "sizing":    1.0,
        "note":      "WR ~45% — chap nhan chuoi thua, Exp duong nho PF",
    },
    "DPM": {
        "combo":     "Volatility Aware",
        "dims":      ["atr_ratio", "bb_position", "candle_body", "rsi_norm", "momentum_20d"],
        "threshold": 0.55,
        "t1_exp":    3.36,
        "t1_pf":     8.79,
        "oos_exp":   4.71,
        "oos_pf":    2.36,
        "mae30":     -5.5,   # ước tính, cần verify
        "sizing":    1.0,
        # DPM ưu tiên hơn DCM khi cùng signal — xem CONFLICT_RULES
    },
    "DCM": {
        "combo":     "Volatility Aware",
        "dims":      ["atr_ratio", "bb_position", "candle_body", "rsi_norm", "momentum_20d"],
        "threshold": 0.55,
        "t1_exp":    4.59,
        "t1_pf":     6.41,
        "oos_exp":   3.95,
        "oos_pf":    2.10,
        "mae30":     -7.0,   # ước tính, MaxDD cao hơn DPM
        "sizing":    0.5,    # half size vì MaxDD -53% cao hơn DPM
        "note":      "Sizing 50% vi MaxDD cao. Bi bo qua neu DPM signal cung ngay",
    },
}

# Rule xử lý conflict khi 2 mã cùng signal cùng ngày
# format: {mã bị skip: mã ưu tiên}
# Nếu mã ưu tiên có signal → bỏ qua mã bị skip
CONFLICT_RULES = {
    "DCM": "DPM",   # DPM ưu tiên hơn DCM — cùng combo, MaxDD DPM tốt hơn
}


def _load_signal_symbols_from_db():
    """
    Load analog config từ DB và merge vào SIGNAL_SYMBOLS.

    Logic merge:
    - Hardcode SIGNAL_SYMBOLS là baseline đã verify kỹ (t1_exp, oos_exp, ...)
    - DB config (từ /analog_approve) override hoặc thêm mã mới
    - Mã DB cần có đủ dims để được add vào cron scan
    - Mã DB thiếu dims → bỏ qua, log warning

    Gọi 1 lần khi bot start (từ _start_signal_cron và signal_status_cmd).
    """
    try:
        from db import load_analog_configs
        db_configs = load_analog_configs()
        if not db_configs:
            logger.info("[AnalogSignal] No DB configs found, using hardcoded SIGNAL_SYMBOLS.")
            return

        added   = []
        skipped = []
        for symbol, cfg in db_configs.items():
            dims = cfg.get("dims")
            if not dims or len(dims) == 0:
                skipped.append(f"{symbol}(no dims)")
                continue

            combo     = cfg.get("combo", "")
            threshold = cfg.get("threshold", 0.60)
            mae30     = cfg.get("mae30", 0.0)
            sizing    = cfg.get("sizing", 1.0)

            if symbol in SIGNAL_SYMBOLS:
                # Override các field từ DB, giữ lại t1_exp/oos_exp nếu DB không có
                existing = SIGNAL_SYMBOLS[symbol]
                SIGNAL_SYMBOLS[symbol] = {
                    **existing,           # giữ t1_exp, oos_exp, note từ hardcode
                    "combo":     combo,
                    "dims":      dims,
                    "threshold": threshold,
                    "mae30":     mae30,
                    "sizing":    sizing,
                }
                added.append(f"{symbol}(updated)")
            else:
                # Mã mới hoàn toàn từ DB
                SIGNAL_SYMBOLS[symbol] = {
                    "combo":     combo,
                    "dims":      dims,
                    "threshold": threshold,
                    "mae30":     mae30,
                    "sizing":    sizing,
                    "t1_exp":    cfg.get("t1_exp", 0.0),
                    "t1_pf":     cfg.get("t1_pf",  0.0),
                    "oos_exp":   cfg.get("oos_exp", 0.0),
                    "oos_pf":    cfg.get("oos_pf",  0.0),
                    "note":      cfg.get("note", "Config tu DB — da verify qua walkforward"),
                }
                added.append(f"{symbol}(new)")

        if added:
            logger.info(
                f"[AnalogSignal] DB merge: {len(added)} symbols added/updated: {added}. "
                f"Total SIGNAL_SYMBOLS: {len(SIGNAL_SYMBOLS)}: {list(SIGNAL_SYMBOLS.keys())}"
            )
        if skipped:
            logger.warning(
                f"[AnalogSignal] DB merge skipped (missing dims): {skipped}. "
                f"Chay /analog_approve <MA> voi dims day du de fix."
            )

    except Exception as e:
        logger.warning(f"[AnalogSignal] _load_signal_symbols_from_db error (non-critical): {e}")

SIGNAL_COOLDOWN_DAYS = 7      # tối đa 1 signal/mã mỗi 7 ngày
MIN_SAMPLES          = 5      # tối thiểu 5 analog độc lập
MDS_DAYS             = 30     # minimum distance sampling
FWD_DAYS             = 30     # forward return window
WIN_THRESH           = 1.0    # +1% tối thiểu để tính "thắng"

# Cron: 08:30 UTC = 15:30 VN
SIGNAL_CRON_HOUR   = 8
SIGNAL_CRON_MINUTE = 30

# DB table
SIGNAL_TABLE = "analog_signals"


# ══════════════════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _init_signal_table():
    """Tạo table analog_signals nếu chưa có."""
    try:
        from db import get_conn
        with get_conn() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {SIGNAL_TABLE} (
                    id          SERIAL PRIMARY KEY,
                    symbol      TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    combo       TEXT,
                    threshold   REAL,
                    entry_price REAL,
                    sl_price    REAL,
                    exp_pct     REAL,
                    pf          REAL,
                    n_analogs   INTEGER,
                    avg_sim     REAL,
                    status      TEXT DEFAULT 'active',
                    close_30d   REAL,
                    actual_ret  REAL,
                    closed_date TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_analog_signals_symbol_date
                ON {SIGNAL_TABLE} (symbol, signal_date)
            """)
        logger.info(f"[AnalogSignal] Table {SIGNAL_TABLE} ready")
    except Exception as e:
        logger.warning(f"[AnalogSignal] init_signal_table error: {e}")


def _save_signal(symbol: str, signal_date: str, combo: str, threshold: float,
                 entry_price: float, sl_price: float, exp_pct: float,
                 pf: float, n_analogs: int, avg_sim: float):
    """Lưu signal mới vào DB."""
    try:
        from db import get_conn
        with get_conn() as conn:
            conn.execute(f"""
                INSERT INTO {SIGNAL_TABLE}
                  (symbol, signal_date, combo, threshold, entry_price,
                   sl_price, exp_pct, pf, n_analogs, avg_sim)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (symbol, signal_date, combo, threshold, entry_price,
                  sl_price, exp_pct, pf, n_analogs, avg_sim))
        logger.info(f"[AnalogSignal] Saved signal {symbol} {signal_date}")
    except Exception as e:
        logger.warning(f"[AnalogSignal] save_signal error: {e}")


def _get_last_signal_date(symbol: str) -> Optional[str]:
    """Lấy ngày signal gần nhất của mã để kiểm tra cooldown."""
    try:
        from db import get_conn
        with get_conn() as conn:
            row = conn.execute(f"""
                SELECT signal_date FROM {SIGNAL_TABLE}
                WHERE symbol = %s
                ORDER BY signal_date DESC LIMIT 1
            """, (symbol,)).fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.warning(f"[AnalogSignal] get_last_signal_date error: {e}")
        return None


def _get_active_signals() -> list[dict]:
    """Lấy tất cả signal đang active (chưa đóng)."""
    try:
        from db import get_conn
        with get_conn() as conn:
            rows = conn.execute(f"""
                SELECT symbol, signal_date, combo, threshold,
                       entry_price, sl_price, exp_pct, pf, n_analogs,
                       avg_sim, status, actual_ret, closed_date
                FROM {SIGNAL_TABLE}
                WHERE status = 'active'
                ORDER BY signal_date DESC
            """).fetchall()
            cols = ["symbol","signal_date","combo","threshold",
                    "entry_price","sl_price","exp_pct","pf","n_analogs",
                    "avg_sim","status","actual_ret","closed_date"]
            return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        logger.warning(f"[AnalogSignal] get_active_signals error: {e}")
        return []


def _get_signal_history(limit: int = 20) -> list[dict]:
    """Lấy lịch sử signal gần nhất."""
    try:
        from db import get_conn
        with get_conn() as conn:
            rows = conn.execute(f"""
                SELECT symbol, signal_date, combo, entry_price,
                       sl_price, exp_pct, pf, n_analogs,
                       status, actual_ret, closed_date
                FROM {SIGNAL_TABLE}
                ORDER BY signal_date DESC
                LIMIT %s
            """, (limit,)).fetchall()
            cols = ["symbol","signal_date","combo","entry_price",
                    "sl_price","exp_pct","pf","n_analogs",
                    "status","actual_ret","closed_date"]
            return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        logger.warning(f"[AnalogSignal] get_signal_history error: {e}")
        return []


def _close_expired_signals():
    """
    Đóng signal đã quá 30 ngày — điền actual_ret từ giá thực tế.
    Chạy mỗi ngày trước khi scan signal mới.
    """
    try:
        from db import get_conn
        from vn_loader import load_vn_ohlcv
        with get_conn() as conn:
            rows = conn.execute(f"""
                SELECT id, symbol, signal_date, entry_price
                FROM {SIGNAL_TABLE}
                WHERE status = 'active'
                  AND signal_date <= %s
            """, (
                (datetime.now() - timedelta(days=FWD_DAYS)).strftime("%Y-%m-%d"),
            )).fetchall()

        for row in rows:
            sig_id, symbol, signal_date, entry_price = row
            try:
                df = load_vn_ohlcv(symbol, days=60, min_bars=10)
                if df is None or len(df) < 2:
                    continue
                # Tìm giá đóng cửa ~30 ngày sau signal
                sig_dt    = pd.Timestamp(signal_date)
                close_dt  = sig_dt + timedelta(days=FWD_DAYS)
                df["date"]= pd.to_datetime(df["date"])
                future    = df[df["date"] >= close_dt]
                if future.empty:
                    # Chưa đủ 30 ngày trading → dùng giá mới nhất
                    close_price = float(df["close"].iloc[-1])
                    closed_date = str(df["date"].iloc[-1])[:10]
                else:
                    close_price = float(future["close"].iloc[0])
                    closed_date = str(future["date"].iloc[0])[:10]

                actual_ret = (close_price - entry_price) / entry_price * 100 if entry_price > 0 else None

                from db import get_conn as _gc
                with _gc() as conn2:
                    conn2.execute(f"""
                        UPDATE {SIGNAL_TABLE}
                        SET status='closed', close_30d=%s,
                            actual_ret=%s, closed_date=%s
                        WHERE id=%s
                    """, (close_price, actual_ret, closed_date, sig_id))
                logger.info(f"[AnalogSignal] Closed {symbol} {signal_date} → ret={actual_ret:.1f}%")
            except Exception as e:
                logger.warning(f"[AnalogSignal] close signal {sig_id} error: {e}")
    except Exception as e:
        logger.warning(f"[AnalogSignal] _close_expired_signals error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CORE SCANNER
# ══════════════════════════════════════════════════════════════════════════════

def _scan_symbol(symbol: str) -> Optional[dict]:
    """
    Scan 1 mã — tìm analog ngày hôm nay.
    Returns signal dict nếu có, None nếu không đủ điều kiện.
    """
    cfg       = SIGNAL_SYMBOLS[symbol]
    dims      = cfg["dims"]
    threshold = cfg["threshold"]
    mae30     = cfg["mae30"]

    # Load data
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=1800, min_bars=200)
    except Exception as e:
        logger.warning(f"[AnalogSignal] {symbol} load fail: {e}")
        return None

    if df is None or len(df) < 200:
        return None

    n_bars    = len(df)
    dates     = df["date"].values
    close_arr = df["close"].values.astype(float)

    # Tính vector ngày hôm nay (bar cuối)
    try:
        from state_vector import compute_state_vector_from_df
        target_vec = compute_state_vector_from_df(df)
    except Exception as e:
        logger.warning(f"[AnalogSignal] {symbol} vector fail: {e}")
        return None

    if target_vec is None:
        return None

    # Vector target chỉ trên dims của combo
    target_arr = np.array([target_vec.get(d, 0.0) for d in dims], dtype=float)
    t_norm     = np.linalg.norm(target_arr)
    if t_norm < 1e-9:
        return None

    # Tính vector cho tất cả bars lịch sử (exclude 90 ngày gần nhất)
    try:
        from state_vector import compute_state_vector_for_date
    except ImportError:
        return None

    exclude_cutoff = n_bars - 90   # tránh lookahead
    vectors        = {}
    for i in range(59, exclude_cutoff):
        vec = compute_state_vector_for_date(df, i)
        if vec is not None:
            vectors[i] = vec

    if len(vectors) < 50:
        return None

    # Cosine similarity
    sim_list = []
    for c_idx, c_vec in vectors.items():
        c_arr  = np.array([c_vec.get(d, 0.0) for d in dims], dtype=float)
        c_norm = np.linalg.norm(c_arr)
        if c_norm < 1e-9:
            continue
        sim = float(np.dot(target_arr, c_arr) / (t_norm * c_norm))
        if sim >= threshold:
            sim_list.append((c_idx, sim))

    if not sim_list:
        return None

    # MDS 30D
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
        return None

    # Forward returns + MFE + MAE của từng analog
    fwd_rets  = []
    mfe_vals  = []   # Maximum Favorable Excursion: đỉnh cao nhất trong 30D
    mae_vals  = []   # Maximum Adverse Excursion: đáy thấp nhất trong 30D

    high_arr = df["high"].values.astype(float) if "high" in df.columns else close_arr
    low_arr  = df["low"].values.astype(float)  if "low"  in df.columns else close_arr

    for c_idx, _ in kept:
        fwd_idx = c_idx + FWD_DAYS
        if fwd_idx >= n_bars:
            continue
        entry = close_arr[c_idx]
        if entry <= 0:
            continue

        # Forward return: close[T+30] vs close[T]
        fwd_rets.append((close_arr[fwd_idx] - entry) / entry * 100)

        # MFE: đỉnh cao nhất trong window [T+1, T+30]
        window_high = high_arr[c_idx + 1: fwd_idx + 1]
        if len(window_high) > 0:
            mfe_vals.append((np.max(window_high) - entry) / entry * 100)

        # MAE: đáy thấp nhất trong window [T+1, T+30]
        window_low = low_arr[c_idx + 1: fwd_idx + 1]
        if len(window_low) > 0:
            mae_vals.append((np.min(window_low) - entry) / entry * 100)

    if len(fwd_rets) < MIN_SAMPLES:
        return None

    # Metrics — median(fwd_rets) làm Exp, nhất quán với Tầng 1/2
    exp      = float(np.median(fwd_rets))
    mfe_med  = float(np.median(mfe_vals))  if mfe_vals else exp
    mae_med  = float(np.median(mae_vals))  if mae_vals else mae30

    wins     = [x for x in fwd_rets if x >= WIN_THRESH]
    losses   = [x for x in fwd_rets if x < WIN_THRESH]
    wr       = len(wins) / len(fwd_rets)
    pos_sum  = sum(wins)
    neg_sum  = abs(sum(losses)) if losses else 1e-9
    pf       = pos_sum / neg_sum if neg_sum > 0 else 99.0
    avg_sim  = float(np.mean([s for _, s in kept]))

    # Giá hiện tại
    entry_price = float(close_arr[-1])
    sl_pct      = mae_med - 2.0         # MAE median từ analogs - buffer 2%
    sl_price    = entry_price * (1 + sl_pct / 100)
    tp_pct      = mfe_med               # MFE median làm TP — đỉnh kỳ vọng
    tp_price    = entry_price * (1 + tp_pct / 100)

    # R:R thực tế
    rr = abs(tp_pct / sl_pct) if sl_pct < 0 else 0.0

    return {
        "symbol":      symbol,
        "combo":       cfg["combo"],
        "threshold":   threshold,
        "entry_price": entry_price,
        "sl_price":    sl_price,
        "sl_pct":      round(sl_pct, 1),
        "tp_price":    tp_price,
        "tp_pct":      round(tp_pct, 1),
        "exp_pct":     round(exp, 2),
        "mfe_med":     round(mfe_med, 2),
        "mae_med":     round(mae_med, 2),
        "rr":          round(rr, 2),
        "pf":          round(pf, 2),
        "wr":          round(wr * 100, 1),
        "n_analogs":   len(fwd_rets),
        "avg_sim":     round(avg_sim, 3),
        # So sánh với OOS walk-forward
        "oos_exp":     cfg.get("oos_exp", 0.0),
        "oos_pf":      cfg.get("oos_pf",  0.0),
        "mae30":       mae30,
    }


def _format_signal_msg(sig: dict, signal_date: str) -> str:
    """Format Telegram message cho 1 signal."""
    sym    = sig["symbol"]
    cfg    = SIGNAL_SYMBOLS[sym]
    sep    = "─" * 28

    # So sánh signal hôm nay vs OOS benchmark
    exp_vs = sig["exp_pct"] - cfg["oos_exp"]
    pf_vs  = sig["pf"] - cfg["oos_pf"]
    exp_em = "✅" if sig["exp_pct"] >= cfg["oos_exp"] * 0.7 else "⚠️"
    pf_em  = "✅" if sig["pf"] >= 1.5 else "⚠️"

    sizing  = cfg.get("sizing", 1.0)
    note    = cfg.get("note", "")
    sizing_str = "" if sizing == 1.0 else f"\n  ⚠️  Sizing khuyen nghi: {int(sizing*100)}% (MaxDD cao hon)"

    # MFE / MAE / R:R từ analogs hôm nay
    mfe_med = sig.get("mfe_med", sig["exp_pct"])
    mae_med = sig.get("mae_med", sig["mae30"])
    rr      = sig.get("rr", 0.0)
    rr_str  = f"{rr:.2f}" if rr > 0 else "N/A"
    rr_em   = "✅" if rr >= 1.0 else "⚠️"

    lines = [
        f"📊 ANALOG SIGNAL — {sym}",
        f"Ngay: {signal_date} | {cfg['combo']} | sim>={cfg['threshold']}",
        sep,
        "Tin hieu hom nay:",
        (f"  WR {sig['wr']:.0f}%  "
         f"Exp {sig['exp_pct']:+.2f}%  "
         f"PF {sig['pf']:.2f}  "
         f"n={sig['n_analogs']} analogs"),
        f"  Do tuong dong TB: {sig['avg_sim']:.3f}",
        sep,
        f"Phan tich 30D tu {sig['n_analogs']} analog tuong tu:",
        f"  MFE median : {mfe_med:+.1f}%  (dinh cao nhat ky vong)",
        f"  MAE median : {mae_med:+.1f}%  (day thap nhat ky vong)",
        f"  {rr_em} R:R      : {rr_str}  (MFE / |MAE|)",
        sep,
        "So voi OOS benchmark (2025):",
        f"  {exp_em} Exp: {sig['exp_pct']:+.2f}% vs OOS {cfg['oos_exp']:+.2f}% ({exp_vs:+.2f}%)",
        f"  {pf_em} PF : {sig['pf']:.2f} vs OOS {cfg['oos_pf']:.2f} ({pf_vs:+.2f})",
        sep,
        f"Ke hoach hanh dong (goi y):{sizing_str}",
        f"  Entry : {sig['entry_price']:,.0f}",
        f"  SL    : {sig['sl_price']:,.0f}  (MAE {mae_med:.1f}% - 2% buffer)",
        f"  TP    : {sig['tp_price']:,.0f}  (MFE median {mfe_med:+.1f}%)",
        f"  → Hold toi da 30 ngay, thoat neu cham SL",
    ]
    if note:
        lines.append(sep)
        lines.append(f"📝 {note}")
    lines += [
        sep,
        "⚠️  Day la goi y, khong phai khuyen nghi dau tu.",
        "   Ket qua qua khu khong dam bao tuong lai.",
    ]
    return "\n".join(lines)



# ══════════════════════════════════════════════════════════════════════════════
# CRON
# ══════════════════════════════════════════════════════════════════════════════

async def _start_signal_cron(bot, chat_ids: list):
    """
    Cron 15:30 VN (08:30 UTC) hàng ngày:
      1. Đóng signal đã quá 30 ngày
      2. Scan từng mã trong SIGNAL_SYMBOLS
      3. Push nếu có signal mới (cooldown 7 ngày/mã)
    """
    _init_signal_table()
    _load_signal_symbols_from_db()   # merge DB config vào SIGNAL_SYMBOLS
    logger.info(
        f"[AnalogSignal] Cron started: {len(chat_ids)} chat_ids, "
        f"{len(SIGNAL_SYMBOLS)} symbols: {list(SIGNAL_SYMBOLS.keys())}"
    )

    while True:
        try:
            now    = datetime.now(timezone.utc)
            # Tính giây đến lần chạy tiếp theo
            target = now.replace(
                hour=SIGNAL_CRON_HOUR,
                minute=SIGNAL_CRON_MINUTE,
                second=0, microsecond=0,
            )
            if now >= target:
                target += timedelta(days=1)
            wait_secs = (target - now).total_seconds()
            logger.info(f"[AnalogSignal] Next run in {wait_secs/3600:.1f}h at {target.strftime('%H:%M UTC')}")
            await asyncio.sleep(wait_secs)
        except Exception as e:
            logger.error(f"[AnalogSignal] sleep error: {e}")
            await asyncio.sleep(3600)
            continue

        # ── Chạy scan ─────────────────────────────────────────────────────────
        logger.info("[AnalogSignal] Running daily scan...")
        today = datetime.now().strftime("%Y-%m-%d")

        try:
            # Đóng signal cũ trước
            await asyncio.to_thread(_close_expired_signals)
        except Exception as e:
            logger.warning(f"[AnalogSignal] close_expired error: {e}")

        signals_found = []
        # Track mã đã có signal hôm nay để xử lý conflict
        signaled_today = set()

        for symbol in SIGNAL_SYMBOLS:
            try:
                # Kiểm tra conflict rule — bỏ qua nếu mã ưu tiên đã signal
                if symbol in CONFLICT_RULES:
                    priority_sym = CONFLICT_RULES[symbol]
                    if priority_sym in signaled_today:
                        logger.info(
                            f"[AnalogSignal] {symbol} skipped — "
                            f"{priority_sym} da co signal hom nay (conflict rule)"
                        )
                        continue

                # Kiểm tra cooldown
                last_date = await asyncio.to_thread(_get_last_signal_date, symbol)
                if last_date:
                    last_dt    = datetime.strptime(last_date, "%Y-%m-%d")
                    days_since = (datetime.now() - last_dt).days
                    if days_since < SIGNAL_COOLDOWN_DAYS:
                        logger.info(
                            f"[AnalogSignal] {symbol} cooldown: "
                            f"{days_since}d < {SIGNAL_COOLDOWN_DAYS}d, skip"
                        )
                        continue

                # Scan
                sig = await asyncio.to_thread(_scan_symbol, symbol)
                if sig is None:
                    logger.info(f"[AnalogSignal] {symbol}: no signal today")
                    continue

                signaled_today.add(symbol)

                # Lưu DB
                await asyncio.to_thread(
                    _save_signal,
                    symbol, today,
                    sig["combo"], sig["threshold"],
                    sig["entry_price"], sig["sl_price"],
                    sig["exp_pct"], sig["pf"],
                    sig["n_analogs"], sig["avg_sim"],
                )

                signals_found.append(sig)
                logger.info(
                    f"[AnalogSignal] {symbol} SIGNAL: "
                    f"Exp={sig['exp_pct']:+.2f}% PF={sig['pf']:.2f}"
                )

            except Exception as e:
                logger.error(f"[AnalogSignal] scan {symbol} error: {e}")

        # Push Telegram
        if signals_found and chat_ids:
            for sig in signals_found:
                msg_text = _format_signal_msg(sig, today)
                for chat_id in chat_ids:
                    try:
                        await bot.send_message(chat_id=chat_id, text=msg_text)
                    except Exception as e:
                        logger.warning(f"[AnalogSignal] send to {chat_id} error: {e}")
        else:
            logger.info(f"[AnalogSignal] No signals today ({today})")


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

async def signal_status_cmd(update, context):
    """
    /signal_status
    Xem các analog signal đang active (chưa đủ 30 ngày).
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    try:
        _init_signal_table()
        _load_signal_symbols_from_db()   # đảm bảo SIGNAL_SYMBOLS up-to-date
        signals = await asyncio.to_thread(_get_active_signals)
    except Exception as e:
        await update.message.reply_text(f"Loi doc DB: {e}")
        return

    if not signals:
        await update.message.reply_text(
            "Khong co analog signal nao dang active.\n"
            "Cron chay luc 15:30 VN moi ngay."
        )
        return

    sep   = "─" * 32
    lines = [f"ANALOG SIGNALS ACTIVE ({len(signals)} tin hieu)", sep]

    for s in signals:
        # Tính ngày còn lại
        try:
            sig_dt    = datetime.strptime(s["signal_date"], "%Y-%m-%d")
            close_dt  = sig_dt + timedelta(days=FWD_DAYS)
            days_left = (close_dt - datetime.now()).days
            days_left = max(0, days_left)
        except Exception:
            days_left = "?"

        entry = s.get("entry_price", 0)
        sl    = s.get("sl_price", 0)
        exp   = s.get("exp_pct", 0)

        lines.append(
            f"📊 {s['symbol']} | {s['signal_date']} | {s['combo']}"
        )
        lines.append(
            f"   Entry {entry:,.0f}  SL {sl:,.0f}  Exp {exp:+.2f}%"
        )
        lines.append(
            f"   PF {s['pf']:.2f}  n={s['n_analogs']}  "
            f"Con {days_left} ngay den ket qua"
        )
        lines.append(sep)

    await update.message.reply_text("\n".join(lines)[:4096])


async def signal_history_cmd(update, context):
    """
    /signal_history
    Lịch sử signal + kết quả thực tế (closed signals).
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    try:
        _init_signal_table()
        signals = await asyncio.to_thread(_get_signal_history, 20)
    except Exception as e:
        await update.message.reply_text(f"Loi doc DB: {e}")
        return

    if not signals:
        await update.message.reply_text(
            "Chua co signal nao trong lich su.\n"
            "Cron bat dau push signal tu 15:30 VN hom nay."
        )
        return

    # Tính thống kê tổng thể
    closed = [s for s in signals if s["status"] == "closed" and s["actual_ret"] is not None]
    active = [s for s in signals if s["status"] == "active"]

    sep   = "─" * 32
    lines = [
        f"ANALOG SIGNAL HISTORY ({len(signals)} tin hieu)",
        sep,
    ]

    # Summary nếu đủ data
    if closed:
        wins    = [s["actual_ret"] for s in closed if s["actual_ret"] >= WIN_THRESH]
        losses  = [s["actual_ret"] for s in closed if s["actual_ret"] < WIN_THRESH]
        wr_real = len(wins) / len(closed) * 100
        exp_real= np.mean([s["actual_ret"] for s in closed])
        pos_sum = sum(wins)
        neg_sum = abs(sum(losses)) if losses else 1e-9
        pf_real = pos_sum / neg_sum if neg_sum > 0 else 99.0

        lines.append(f"THONG KE THUC TE ({len(closed)} tin hieu da dong):")
        lines.append(
            f"  WR {wr_real:.0f}%  Exp {exp_real:+.2f}%  PF {pf_real:.2f}"
        )
        lines.append(sep)

    # Chi tiết từng signal
    lines.append("CHI TIET:")
    for s in signals[:15]:
        status_em = "✅" if s["status"] == "active" else (
            "🟢" if (s["actual_ret"] or 0) >= WIN_THRESH else "🔴"
        )
        actual_str = (
            f"→ Thuc te: {s['actual_ret']:+.2f}%"
            if s["actual_ret"] is not None
            else "→ Dang cho ket qua"
        )
        lines.append(
            f"{status_em} {s['symbol']} {s['signal_date']} | {s['combo']}"
        )
        lines.append(
            f"   Exp {s['exp_pct']:+.2f}%  PF {s['pf']:.2f}  "
            f"Entry {s['entry_price']:,.0f}  {actual_str}"
        )

    if active:
        lines.append(sep)
        lines.append(f"Dang active: {len(active)} tin hieu chua co ket qua")

    await update.message.reply_text("\n".join(lines)[:4096])
