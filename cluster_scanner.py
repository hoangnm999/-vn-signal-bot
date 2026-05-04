"""
cluster_scanner.py — Cluster-Based Daily Signal Scanner
VN Trader Bot V6 — Session 30

Thay thế batch_scanner.py với 2-cluster approach từ Session 30.

Cron schedule (UTC, server Render):
  01:30 UTC = 08:30 VN  ← trước giờ mở cửa HOSE 09:00
  05:30 UTC = 12:30 VN  ← giữa phiên, lấy giá mới nhất

Logic mỗi lần scan:
  8:30:  Full scan → phát signal mới nếu có
  12:30: B+C combo:
         B. Update P&L của signals đã phát buổi sáng
         C. Scan lại với giá mới → alert nếu có signal mới

Clusters:
  Mean Reversion (FWD=20d): DCM, NKG, DPM, HAH, HCM, HSG, DGC, GAS
                             + NLG, HDB, BMP (S31 expand)
  Momentum      (FWD=10d):  VCB, BID, MBB, MWG, CTG, FRT, REE, FPT, GMD, STB, PNJ, TCB
                             + SSI, VND, VIX, CTS, VCI, HAG, BCM, ORS, BSR, VSC,
                               DIG, LPB, FTS, APG, VDS (S31 expand Tier 1)

VNI Filter (MR only): vni_atr_ratio >= median training (soft info, shown in signal)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Watchlist & Config ────────────────────────────────────────────────────────
MR_SYMBOLS  = [
    # Original (S30)
    "DCM", "NKG", "DPM", "HAH", "HCM", "HSG", "DGC", "GAS",
    # Tier 1 expand (S31) — Score >= 5, WFE validated
    "NLG", "HDB", "BMP",
]
MOM_SYMBOLS = [
    # Original (S30)
    "VCB", "BID", "MBB", "MWG", "CTG", "FRT", "REE", "FPT",
    "GMD", "STB", "PNJ", "TCB",
    # Tier 1 expand (S31) — Score >= 5, WFE validated
    "SSI", "VND", "VIX", "CTS", "VCI", "HAG", "BCM",
    "ORS", "BSR", "VSC", "DIG", "LPB", "FTS", "APG", "VDS",
]

# Breakout cluster (S31 — validated từ discover_clusters_v2 + walk forward)
# Score >= 5, chưa có trong MR/MOM watchlist
BREAKOUT_SYMBOLS = [
    # Mã chỉ thuộc Breakout
    "VIB", "KBC", "KSB", "HT1", "SHB", "GVR", "SIP",
    # Mã thuộc cả MR + Breakout (nhận signal từ 2 cluster)
    "DGC", "DCM", "NKG", "HSG",
    # Mã thuộc cả MOM + Breakout (nhận signal từ 2 cluster)
    "VIX", "FPT", "TCB", "BSR", "VND", "GMD", "LPB",
]

FWD_DAYS = {"Mean Reversion": 20, "Momentum": 10, "Breakout": 15}

# Cron times (UTC)
MORNING_HOUR,   MORNING_MINUTE   = 1, 30   # 08:30 VN
AFTERNOON_HOUR, AFTERNOON_MINUTE = 5, 30   # 12:30 VN

# Signal logic config (nhất quán với walk_forward_cluster.py)
SIGNAL_CONFIG = {
    "Mean Reversion": {
        "regime_indicator":  "price_vs_sma50",
        "regime_condition":  "low",
        "trigger_indicators":["stoch_k", "volume_spike", "momentum_5d"],
        "trigger_direction": {"stoch_k": "low", "volume_spike": "high",
                              "momentum_5d": "high"},
        "description": "Mua khi giá dưới SMA50 + stoch oversold + volume spike",
    },
    "Momentum": {
        "regime_indicator":  "ema_cross",
        "regime_condition":  "high",
        "trigger_indicators":["momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction": {"momentum_5d": "high", "volume_spike": "high",
                              "candle_body": "high"},
        "description": "Mua khi EMA12>EMA26 + momentum mạnh + volume xác nhận",
    },
    "Breakout": {
        "regime_indicator":  "bb_squeeze",
        "regime_condition":  "high",
        "trigger_indicators":["consolidation", "vol_dry_up"],
        "trigger_direction": {"consolidation": "low", "vol_dry_up": "high"},
        "description": "Mua khi BB rộng + giá sideways + volume khô → bứt phá",
    },
}

TRIGGER_PCT  = 70
MIN_TRIGGERS = 2

# Per-symbol WF stats (từ walk forward results Session 30)
# Format: symbol → {wr, exp, wfe, n_wf}
SYMBOL_STATS = {
    # Mean Reversion                                              pf = estimated từ WR+Exp (S31)
    "DCM": {"wr": 65, "exp": 7.0,  "wfe": 0.73, "n": 31,  "pf": 2.17, "cluster": "Mean Reversion"},
    "NKG": {"wr": 60, "exp": 6.0,  "wfe": 0.80, "n": 39,  "pf": 1.75, "cluster": "Mean Reversion"},
    "DPM": {"wr": 56, "exp": 4.1,  "wfe": 0.47, "n": 33,  "pf": 1.49, "cluster": "Mean Reversion"},
    "HAH": {"wr": 62, "exp": 3.2,  "wfe": 0.97, "n": 40,  "pf": 1.91, "cluster": "Mean Reversion"},
    "HCM": {"wr": 55, "exp": 3.2,  "wfe": 0.51, "n": 41,  "pf": 1.43, "cluster": "Mean Reversion"},
    "HSG": {"wr": 52, "exp": 2.5,  "wfe": 0.30, "n": 46,  "pf": 1.27, "cluster": "Mean Reversion"},
    "DGC": {"wr": 50, "exp": 1.6,  "wfe": 0.43, "n": 46,  "pf": 1.17, "cluster": "Mean Reversion"},
    "GAS": {"wr": 48, "exp": 0.9,  "wfe": 0.30, "n": 28,  "pf": 1.08, "cluster": "Mean Reversion"},
    # Momentum
    "VCB": {"wr": 54, "exp": 1.0,  "wfe": 4.64, "n": 48,  "pf": 1.63, "cluster": "Momentum"},
    "BID": {"wr": 53, "exp": 2.9,  "wfe": 1.59, "n": 53,  "pf": 1.57, "cluster": "Momentum"},
    "MBB": {"wr": 50, "exp": 1.4,  "wfe": 1.11, "n": 67,  "pf": 1.39, "cluster": "Momentum"},
    "MWG": {"wr": 43, "exp": 1.3,  "wfe": 1.39, "n": 61,  "pf": 1.05, "cluster": "Momentum"},
    "CTG": {"wr": 56, "exp": 0.9,  "wfe": 0.49, "n": 51,  "pf": 1.77, "cluster": "Momentum"},
    "FRT": {"wr": 60, "exp": 0.6,  "wfe": 0.39, "n": 59,  "pf": 2.08, "cluster": "Momentum"},
    "REE": {"wr": 53, "exp": 1.2,  "wfe": 0.44, "n": 54,  "pf": 1.57, "cluster": "Momentum"},
    "FPT": {"wr": 25, "exp": 0.9,  "wfe": 0.43, "n": 49,  "pf": 1.00, "cluster": "Momentum"},
    "GMD": {"wr": 44, "exp": 0.4,  "wfe": 0.66, "n": 50,  "pf": 1.09, "cluster": "Momentum"},
    "STB": {"wr": 43, "exp": 0.9,  "wfe": 0.00, "n": 73,  "pf": 1.05, "cluster": "Momentum"},
    "PNJ": {"wr": 47, "exp": 0.4,  "wfe": 3.27, "n": 57,  "pf": 1.23, "cluster": "Momentum"},
    "TCB": {"wr": 54, "exp": 0.3,  "wfe": 0.70, "n": 68,  "pf": 1.63, "cluster": "Momentum"},
    # ── Tier 1 expand S31 — Mean Reversion ───────────────────────────────────
    "NLG": {"wr": 61, "exp": 4.0,  "wfe": 1.11, "n": 136, "pf": 1.89, "cluster": "Mean Reversion"},
    "HDB": {"wr": 62, "exp": 2.5,  "wfe": 1.25, "n": 135, "pf": 1.93, "cluster": "Mean Reversion"},
    "BMP": {"wr": 55, "exp": 2.4,  "wfe": 1.34, "n": 171, "pf": 1.43, "cluster": "Mean Reversion"},
    # ── Tier 1 expand S31 — Momentum ─────────────────────────────────────────
    "SSI": {"wr": 63, "exp": 3.8,  "wfe": 35.73,"n": 236, "pf": 2.43, "cluster": "Momentum"},
    "VND": {"wr": 62, "exp": 4.7,  "wfe": 4.75, "n": 257, "pf": 2.30, "cluster": "Momentum"},
    "VIX": {"wr": 57, "exp": 4.3,  "wfe": 3.97, "n": 231, "pf": 1.85, "cluster": "Momentum"},
    "CTS": {"wr": 63, "exp": 6.3,  "wfe": 1.59, "n": 222, "pf": 2.38, "cluster": "Momentum"},
    "VCI": {"wr": 71, "exp": 5.6,  "wfe": 1.05, "n": 194, "pf": 3.42, "cluster": "Momentum"},
    "HAG": {"wr": 60, "exp": 4.1,  "wfe": 7.21, "n": 246, "pf": 2.10, "cluster": "Momentum"},
    "BCM": {"wr": 51, "exp": 2.8,  "wfe": 12.08,"n": 162, "pf": 1.50, "cluster": "Momentum"},
    "ORS": {"wr": 54, "exp": 3.7,  "wfe": 2.49, "n": 214, "pf": 1.64, "cluster": "Momentum"},
    "BSR": {"wr": 57, "exp": 2.4,  "wfe": 3.92, "n": 250, "pf": 1.86, "cluster": "Momentum"},
    "VSC": {"wr": 60, "exp": 1.6,  "wfe": 4.82, "n": 223, "pf": 2.09, "cluster": "Momentum"},
    "DIG": {"wr": 59, "exp": 3.6,  "wfe": 1.35, "n": 244, "pf": 2.00, "cluster": "Momentum"},
    "LPB": {"wr": 56, "exp": 2.0,  "wfe": 2.67, "n": 229, "pf": 1.79, "cluster": "Momentum"},
    "FTS": {"wr": 62, "exp": 4.7,  "wfe": 1.11, "n": 268, "pf": 2.34, "cluster": "Momentum"},
    "APG": {"wr": 55, "exp": 4.3,  "wfe": 1.75, "n": 219, "pf": 1.72, "cluster": "Momentum"},
    "VDS": {"wr": 59, "exp": 4.2,  "wfe": 1.40, "n": 222, "pf": 2.08, "cluster": "Momentum"},
    # Breakout cluster S31 (validated)
    "VIB": {"wr": 67, "exp": 4.75, "wfe": 1.11, "n": 247, "pf": 3.79, "cluster": "Breakout"},
    "KBC": {"wr": 56, "exp": 5.04, "wfe": 4.66, "n": 238, "pf": 2.06, "cluster": "Breakout"},
    "KSB": {"wr": 64, "exp": 4.89, "wfe": 7.01, "n": 251, "pf": 1.96, "cluster": "Breakout"},
    "HT1": {"wr": 71, "exp": 5.05, "wfe": 1.72, "n": 236, "pf": 3.07, "cluster": "Breakout"},
    "SHB": {"wr": 60, "exp": 7.99, "wfe": 2.54, "n": 193, "pf": 3.08, "cluster": "Breakout"},
    "GVR": {"wr": 67, "exp": 6.07, "wfe": 1.99, "n": 220, "pf": 2.53, "cluster": "Breakout"},
    "SIP": {"wr": 60, "exp": 3.00, "wfe": 1.38, "n": 219, "pf": 1.83, "cluster": "Breakout"},
}

# SL từ MAE p25 analysis
SL_CONFIG = {
    "Mean Reversion": -13.5,  # p25 MAE
    "Momentum":       -6.4,   # p25 MAE
    "Breakout":       -6.4,   # dùng MOM SL (quyết định S31, MAE riêng TBD)
}

# Trailing Stop config — validated từ backtest + walk forward (S31)
# activation_pct: % gain tối thiểu để kích hoạt trailing
# mult: SL trail = peak_price - mult × ATR14
TRAIL_CONFIG = {
    # S30 original (20 ma)
    "GAS": {"mult": 3.0, "activation_pct": 9.25},   # MR  | WFE=7.42  consistency=80%
    "CTG": {"mult": 2.5, "activation_pct": 3.97},   # MOM | WFE=1.53  consistency=100%
    "REE": {"mult": 1.5, "activation_pct": 4.19},   # MOM | WFE=0.85  consistency=60%
    "GMD": {"mult": 2.0, "activation_pct": 6.15},   # MOM | WFE=3.63  consistency=100%
    # S31 expand (18 ma moi)
    "BMP": {"mult": 2.0, "activation_pct": 5.01},   # MR  | WFE=1.51  consistency=80%
    "CTS": {"mult": 2.5, "activation_pct": 5.26},   # MOM | WFE=2.89  consistency=80%
    "HAG": {"mult": 2.0, "activation_pct": 4.60},   # MOM | WFE=357.5 consistency=80%
    "BCM": {"mult": 1.0, "activation_pct": 5.75},   # MOM | WFE=3.52  consistency=80%
    "APG": {"mult": 1.5, "activation_pct": 5.90},   # MOM | WFE=2.69  consistency=80%
}

# Account size để tính position sizing (VND)
# Chỉnh theo vốn thực tế của bạn
ACCOUNT_SIZE = 300_000_000  # 300 triệu

# Continuous position sizing config
BASE_RISK_PCT  = 1.0    # % account cho mã có Score = MEDIAN_SCORE
MIN_RISK_PCT   = 0.4    # % tối thiểu (mã yếu nhất)
MAX_RISK_PCT   = 2.0    # % tối đa (mã mạnh nhất, 2x base)
MEDIAN_SCORE   = 8.4    # median score của toàn watchlist (tính từ data)

# Max concurrent positions & max exposure
MAX_POSITIONS  = 6
MAX_EXPOSURE   = 0.40   # tối đa 40% vốn deployed

# In-memory signal cache (tồn tại trong session)
_morning_signals: dict = {}   # symbol → signal_info (từ 8:30 scan)


# ── Position sizing helper ───────────────────────────────────────────────────

def _calc_position_size(entry_price: float, sl_pct: float,
                        sizing_score: float) -> dict:
    """
    Continuous position sizing — tỷ lệ trực tiếp với Score (log scale).

    Formula:
        risk_pct = BASE_RISK_PCT × log(1 + score) / log(1 + MEDIAN_SCORE)
        Clamped: [MIN_RISK_PCT, MAX_RISK_PCT]

    Dùng log scale để tránh outlier score (SSI=329) chiếm quá nhiều size.
    Mã có Score = MEDIAN (8.4) → risk = BASE_RISK_PCT (1%)
    Mã có Score > median       → risk tăng dần, max 2x
    Mã có Score < median       → risk giảm dần, min 0.4x
    """
    import math

    # Log-scaled risk
    score       = max(sizing_score, 0.1)   # tránh log(0)
    log_score   = math.log(1 + score)
    log_median  = math.log(1 + MEDIAN_SCORE)
    raw_risk    = BASE_RISK_PCT * (log_score / log_median)
    risk_pct    = round(max(MIN_RISK_PCT, min(MAX_RISK_PCT, raw_risk)), 2)

    risk_amount = ACCOUNT_SIZE * risk_pct / 100
    sl_value    = entry_price * abs(sl_pct) / 100
    if sl_value <= 0:
        return {}

    raw_qty  = risk_amount / sl_value
    qty      = max(100, int(raw_qty / 100) * 100)
    value    = qty * entry_price
    exposure = value / ACCOUNT_SIZE * 100

    # Cap tại max per trade
    max_per_trade = ACCOUNT_SIZE * MAX_EXPOSURE / MAX_POSITIONS
    if value > max_per_trade:
        qty      = max(100, int(max_per_trade / entry_price / 100) * 100)
        value    = qty * entry_price
        exposure = value / ACCOUNT_SIZE * 100

    # Label để hiển thị
    if risk_pct >= BASE_RISK_PCT * 1.5:
        size_label = "⬆️ TĂNG SIZE"
    elif risk_pct >= BASE_RISK_PCT * 0.8:
        size_label = "➡️ BÌNH THƯỜNG"
    else:
        size_label = "⬇️ GIẢM SIZE"

    return {
        "qty":         qty,
        "value":       value,
        "risk_amount": round(risk_amount),
        "risk_pct":    risk_pct,
        "size_label":  size_label,
        "exposure":    round(exposure, 1),
    }


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(c, span):
    return pd.Series(c).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c).rolling(p, min_periods=p).mean().values


def _compute_indicators(df: pd.DataFrame) -> dict | None:
    """Tính indicators cho ROW CUỐI của df (ngày hôm nay/mới nhất)."""
    if len(df) < 60:
        return None

    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    vol   = df["volume"].values.astype(float)
    opn   = df["open"].values.astype(float)
    n     = len(df)
    i     = n - 1   # index của ngày mới nhất

    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    sma20  = _sma(close, 20)
    sma50  = _sma(close, 50)
    vsma20 = _sma(vol, 20)

    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr    = _sma(tr, 14)

    lo14   = pd.Series(low).rolling(14).min().values
    hi14   = pd.Series(high).rolling(14).max().values
    denom  = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch  = 100 * (close - lo14) / denom

    px    = close[i]
    atr_v = atr[i]   if np.isfinite(atr[i])   else px * 0.02
    s20   = sma20[i]  if np.isfinite(sma20[i])  else px
    s50   = sma50[i]  if np.isfinite(sma50[i])  else px
    vs20v = vsma20[i] if np.isfinite(vsma20[i]) else vol[i]
    c5    = close[max(i - 5, 0)]

    # Breakout indicators
    vsma60   = _sma(vol, 60)
    vsma60_v = vsma60[i] if np.isfinite(vsma60[i]) else vs20v
    bb_std_v = float(pd.Series(close[:i+1]).rolling(20).std().iloc[-1]) if i >= 20 else atr_v
    bb_width = float(4 * bb_std_v / (s20 + 1e-9) * 100)

    def _consol_val(c_arr, idx):
        if idx < 15: return 0.5
        window = c_arr[idx-14:idx+1]
        mid    = c_arr[idx]
        return float(np.sum(np.abs(window - mid) / (mid + 1e-9) < 0.03)) / len(window)

    return {
        "close":          px,
        "price_vs_sma50": float((px - s50) / (px + 1e-9) * 100),
        "price_vs_sma20": float((px - s20) / (px + 1e-9) * 100),
        "ema_cross":      float((ema12[i] - ema26[i]) / (px + 1e-9) * 100),
        "momentum_5d":    float((px / (c5 + 1e-9) - 1.0) * 100),
        "volume_spike":   float((vol[i] / (vs20v + 1e-9)) - 1.0),
        "stoch_k":        float(stoch[i]),
        "candle_body":    float(np.clip(abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
        "atr_ratio":      float(atr_v / (px + 1e-9) * 100),
        "sma20":          float(s20),
        "sma50":          float(s50),
        "ema12":          float(ema12[i]),
        "ema26":          float(ema26[i]),
        "last_date":      str(df["date"].iloc[i])[:10],
        "volume":         float(vol[i]),
        "vol_sma20":      float(vs20v),
        # Breakout cluster indicators
        "bb_squeeze":    bb_width,
        "consolidation": _consol_val(close, i),
        "vol_dry_up":    float((vs20v / (vsma60_v + 1e-9)) - 1.0),
    }


def _compute_thresholds_from_training(df: pd.DataFrame,
                                       cluster: str) -> dict | None:
    """
    Tính thresholds từ toàn bộ training data 2019-2024.
    Dùng cho signal detection (nhất quán với walk forward).
    """
    cfg = SIGNAL_CONFIG[cluster]
    train = df[
        (df["date"] >= "2019-01-01") &
        (df["date"] <= "2024-12-31")
    ].reset_index(drop=True)

    if len(train) < 200:
        return None

    close = train["close"].values.astype(float)
    high  = train["high"].values.astype(float)
    low   = train["low"].values.astype(float)
    vol   = train["volume"].values.astype(float)
    opn   = train["open"].values.astype(float)
    n     = len(train)

    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    sma50  = _sma(close, 50)
    vsma20 = _sma(vol, 20)
    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr    = _sma(tr, 14)
    lo14   = pd.Series(low).rolling(14).min().values
    hi14   = pd.Series(high).rolling(14).max().values
    denom  = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch  = 100 * (close - lo14) / denom

    rows = []
    for i in range(60, n):
        px    = close[i]
        atr_v = atr[i]   if np.isfinite(atr[i])   else px * 0.02
        s50   = sma50[i]  if np.isfinite(sma50[i])  else px
        vs20v = vsma20[i] if np.isfinite(vsma20[i]) else vol[i]
        c5    = close[max(i - 5, 0)]
        # Breakout indicators
        vsma60_   = _sma(vol, 60)
        vsma60_v_ = vsma60_[i] if np.isfinite(vsma60_[i]) else vs20v
        bb_std_   = pd.Series(close[:i+1]).rolling(20).std().iloc[-1] if i >= 20 else atr_v
        bb_width_ = float(4 * float(bb_std_) / (px + 1e-9) * 100)
        window_   = close[max(0,i-14):i+1]
        consol_   = float(np.sum(np.abs(window_ - px) / (px + 1e-9) < 0.03)) / max(len(window_), 1)

        rows.append({
            "price_vs_sma50": float((px - s50) / (px + 1e-9) * 100),
            "ema_cross":      float((ema12[i] - ema26[i]) / (px + 1e-9) * 100),
            "momentum_5d":    float((px / (c5 + 1e-9) - 1.0) * 100),
            "volume_spike":   float((vol[i] / (vs20v + 1e-9)) - 1.0),
            "stoch_k":        float(stoch[i]),
            "candle_body":    float(np.clip(abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
            "bb_squeeze":     bb_width_,
            "consolidation":  consol_,
            "vol_dry_up":     float((vs20v / (vsma60_v_ + 1e-9)) - 1.0),
        })

    reg_ind  = cfg["regime_indicator"]
    trig_ind = cfg["trigger_indicators"]
    trig_dir = cfg["trigger_direction"]

    reg_vals   = [r[reg_ind] for r in rows if np.isfinite(r.get(reg_ind, float("nan")))]
    reg_thresh = float(np.median(reg_vals)) if reg_vals else 0.0

    trig_thresh = {}
    for t in trig_ind:
        vals = [r[t] for r in rows if np.isfinite(r.get(t, float("nan")))]
        if not vals:
            continue
        if trig_dir.get(t, "high") == "low":
            trig_thresh[t] = float(np.percentile(vals, 100 - TRIGGER_PCT))
        else:
            trig_thresh[t] = float(np.percentile(vals, TRIGGER_PCT))

    return {"reg_thresh": reg_thresh, "trig_thresh": trig_thresh}


# ── VNI ATR ratio ─────────────────────────────────────────────────────────────

_vni_thresh_cache: float | None = None

def _get_vni_atr_info() -> dict:
    """Load VNI, tính ATR ratio hiện tại và so với threshold training."""
    global _vni_thresh_cache
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv("VNINDEX", days=300, min_bars=100)
        df["date"] = pd.to_datetime(df["date"])
        close = df["close"].values.astype(float) * 1000
        h_prev= np.concatenate([[close[0]], close[:-1]])
        tr    = np.abs(close - h_prev)
        atr14 = _sma(tr, 14)

        current_atr = float(atr14[-1] / close[-1] * 100) if np.isfinite(atr14[-1]) else 0.0

        # Tính threshold từ training nếu chưa có
        if _vni_thresh_cache is None:
            train_df = df[df["date"] <= "2024-12-31"]
            if len(train_df) >= 100:
                tc = train_df["close"].values.astype(float) * 1000
                th_prev = np.concatenate([[tc[0]], tc[:-1]])
                t_tr    = np.abs(tc - th_prev)
                t_atr   = _sma(t_tr, 14)
                vals    = [float(t_atr[j] / tc[j] * 100)
                           for j in range(len(tc))
                           if np.isfinite(t_atr[j]) and tc[j] > 0]
                _vni_thresh_cache = float(np.median(vals)) if vals else 0.863
            else:
                _vni_thresh_cache = 0.863  # fallback từ analysis

        thresh   = _vni_thresh_cache
        is_high  = current_atr >= thresh
        last_date= str(df["date"].iloc[-1])[:10]

        return {
            "atr_ratio":  round(current_atr, 3),
            "threshold":  round(thresh, 3),
            "is_high":    is_high,
            "last_date":  last_date,
            "status":     "✅ ATR cao — MR signals mạnh hơn" if is_high
                          else "⚠️ ATR thấp — MR signals yếu hơn",
        }
    except Exception as e:
        logger.warning(f"[VNI] Error: {e}")
        return {"atr_ratio": 0, "threshold": 0.863, "is_high": None,
                "last_date": "?", "status": "⚠️ Không load được VNI"}


# ── Signal detection cho 1 mã ─────────────────────────────────────────────────

def _scan_symbol(symbol: str, cluster: str) -> dict | None:
    """
    Scan 1 mã. Trả về signal dict nếu có signal, None nếu không.
    """
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=400, min_bars=200)
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        logger.debug(f"[Scanner] {symbol} load fail: {e}")
        return None

    # Tính indicators ngày mới nhất
    ind = _compute_indicators(df)
    if ind is None:
        return None

    # Tính thresholds từ training
    thresh = _compute_thresholds_from_training(df, cluster)
    if thresh is None:
        return None

    cfg        = SIGNAL_CONFIG[cluster]
    reg_ind    = cfg["regime_indicator"]
    reg_cond   = cfg["regime_condition"]
    trig_ind   = cfg["trigger_indicators"]
    trig_dir   = cfg["trigger_direction"]
    reg_thresh  = thresh["reg_thresh"]
    trig_thresh = thresh["trig_thresh"]

    # Tầng 1: Regime
    val = ind.get(reg_ind, float("nan"))
    if not np.isfinite(val):
        return None
    in_regime = (val <= reg_thresh) if reg_cond == "low" else (val > reg_thresh)
    if not in_regime:
        return None

    # Tầng 2: Triggers
    triggered = []
    not_triggered = []
    for t in trig_ind:
        v  = ind.get(t, float("nan"))
        th = trig_thresh.get(t, float("nan"))
        if not (np.isfinite(v) and np.isfinite(th)):
            continue
        hit = (v <= th) if trig_dir.get(t, "high") == "low" else (v >= th)
        if hit:
            triggered.append(t)
        else:
            not_triggered.append(t)

    if len(triggered) < MIN_TRIGGERS:
        return None

    # Signal confirmed
    stats     = SYMBOL_STATS.get(symbol, {})
    fwd       = FWD_DAYS[cluster]
    entry     = ind["close"]
    sl_pct    = SL_CONFIG[cluster]
    sl_price  = round(entry * (1 + sl_pct / 100), 1)
    tp_date   = (date.today() + timedelta(days=int(fwd * 1.4))).strftime("%d/%m/%Y")

    # Regime detail string
    if cluster == "Mean Reversion":
        regime_detail = (f"Giá dưới SMA50 ({val:+.1f}%) | "
                         f"SMA50={ind['sma50']:.1f}")
    elif cluster == "Breakout":
        consol_pct = ind.get("consolidation", 0) * 100
        regime_detail = (f"BB rộng (width={val:.1f}%) | "
                         f"Sideways {consol_pct:.0f}% ngày | "
                         f"Vol dry-up={ind.get('vol_dry_up', 0):+.2f}x")
    else:
        regime_detail = (f"EMA12 > EMA26 ({val:+.2f}%) | "
                         f"EMA12={ind['ema12']:.1f} EMA26={ind['ema26']:.1f}")

    # Trigger detail
    trigger_labels = {
        "stoch_k":     f"Stoch oversold ({ind['stoch_k']:.1f})",
        "momentum_5d": f"Momentum 5d ({ind['momentum_5d']:+.1f}%)",
        "volume_spike":f"Volume spike ({ind['volume_spike']:+.1f}x)",
        "candle_body": f"Nến thân lớn ({ind['candle_body']:.2f})",
    }
    trigger_str = " + ".join(trigger_labels.get(t, t) for t in triggered)

    return {
        "symbol":        symbol,
        "cluster":       cluster,
        "entry_price":   round(entry, 2),
        "sl_price":      sl_price,
        "sl_pct":        sl_pct,
        "tp_date":       tp_date,
        "fwd_days":      fwd,
        "regime_detail": regime_detail,
        "trigger_str":   trigger_str,
        "triggered":     triggered,
        "n_triggers":    len(triggered),
        "last_date":     ind["last_date"],
        "stats":         stats,
        "ind":           ind,
        "scan_time":     datetime.now().strftime("%H:%M"),
    }


# ── Format Telegram messages ──────────────────────────────────────────────────

def _format_signal(sig: dict, vni_info: dict,
                   extra_tag: str = "") -> str:
    """Format 1 signal thành Telegram message."""
    sym     = sig["symbol"]
    cluster = sig["cluster"]
    stats   = sig["stats"]
    fwd     = sig["fwd_days"]

    # Cluster emoji + short
    if cluster == "Mean Reversion":
        emoji, cluster_short = "🔄", "MR"
    elif cluster == "Momentum":
        emoji, cluster_short = "🚀", "MOM"
    else:
        emoji, cluster_short = "💥", "BO"

    # WFE badge
    wfe = stats.get("wfe", 0)
    wfe_badge = ("⭐⭐⭐" if wfe >= 1.0 else
                 "⭐⭐"  if wfe >= 0.7 else
                 "⭐"   if wfe >= 0.5 else "")

    lines = [
        f"{emoji} *{sym}* [{cluster_short}]{extra_tag} {wfe_badge}",
        f"",
        f"📅 Data: {sig['last_date']} | {sig['n_triggers']}/{len(SIGNAL_CONFIG[cluster]['trigger_indicators'])} triggers",
        f"",
        f"*Regime:* {sig['regime_detail']}",
        f"*Triggers:* {sig['trigger_str']}",
    ]

    # VNI info cho MR
    if cluster == "Mean Reversion":
        lines.append(f"*VNI ATR:* {vni_info['status']}")

    # Profit Factor
    pf  = stats.get("pf", 0)
    pf_str = f"{pf:.2f}" if pf else "?"

    # Continuous sizing score = Exp × PF × WFE
    sizing_score = (stats.get("exp", 0) * pf * wfe) if (pf and wfe) else 0

    lines += [
        f"",
        f"*📊 Walk Forward OOS (2022→nay):*",
        f"  WR={stats.get('wr', '?')}% | Exp={stats.get('exp', '?'):+.1f}% | "
        f"PF={pf_str} | WFE={wfe:.2f} | n={stats.get('n', '?')}",
        f"  Score={sizing_score:.1f} (median={MEDIAN_SCORE})",
        f"",
        f"*🎯 Trade Plan:*",
        f"  Entry: Close hôm nay ~{sig['entry_price']:,.0f}",
        f"  SL: {sig['sl_price']:,.0f} ({sig['sl_pct']:+.1f}%) — Catastrophic stop",
    ]

    # Trailing stop nếu có config cho mã này
    trail_cfg = TRAIL_CONFIG.get(sym)
    if trail_cfg:
        atr_val  = sig.get("ind", {}).get("atr", 0)
        trail_sl = round(sig["entry_price"] - trail_cfg["mult"] * atr_val, 0)
        lines += [
            f"  Exit: Time Stop T+{fwd}d (~{sig['tp_date']})",
            f"  *🔔 Trailing Stop:* Kích hoạt khi lãi ≥{trail_cfg['activation_pct']}%",
            f"    → SL trail = đỉnh - {trail_cfg['mult']}×ATR "
            f"(≈{trail_sl:,.0f} từ entry)",
        ]
    else:
        lines.append(f"  Exit: Time Stop T+{fwd}d (~{sig['tp_date']})")

    # Position sizing cụ thể
    ps = _calc_position_size(sig["entry_price"], sig["sl_pct"], sizing_score)
    if ps:
        lines += [
            f"",
            f"*💰 Position Sizing ({ACCOUNT_SIZE/1e6:.0f}M account):*",
            f"  {ps['size_label']} — risk {ps['risk_pct']}% "
            f"= {ps['risk_amount']/1e6:.1f}M",
            f"  → Mua: *{ps['qty']:,} cổ* (~{ps['value']/1e6:.1f}M, "
            f"chiếm {ps['exposure']}% vốn)",
            f"  → Max loss nếu chạm SL: "
            f"~{ps['risk_amount']/1e6:.1f}M ({ps['risk_pct']}% account)",
        ]
    else:
        lines.append(f"  Size: Risk {BASE_RISK_PCT}% account")

    return "\n".join(lines)


def _format_morning_scan(
    mr_signals: list[dict],
    mom_signals: list[dict],
    mr_no_signal: list[str],
    mom_no_signal: list[str],
    vni_info: dict,
    scan_label: str = "08:30",
    bo_signals: list[dict] | None = None,
    bo_no_signal: list[str] | None = None,
) -> list[str]:
    """Format full morning scan report."""
    vn_now = datetime.utcnow() + timedelta(hours=7)
    header = (
        f"🔍 *CLUSTER SCAN — {scan_label} VN*\n"
        f"📅 {vn_now.strftime('%d/%m/%Y %H:%M')} VN\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    messages = []
    current  = header + "\n"

    total_signals = len(mr_signals) + len(mom_signals)

    bo_signals    = bo_signals    or []
    bo_no_signal  = bo_no_signal  or []
    total_signals = len(mr_signals) + len(mom_signals) + len(bo_signals)

    if total_signals == 0:
        current += (
            f"\n✅ Không có signal hôm nay\n\n"
            f"*Mean Reversion ({len(MR_SYMBOLS)} mã):* Không đủ điều kiện\n"
            f"*Momentum ({len(MOM_SYMBOLS)} mã):* Không đủ điều kiện\n"
            f"*Breakout ({len(BREAKOUT_SYMBOLS)} mã):* Không đủ điều kiện\n\n"
            f"*VNI:* {vni_info['status']}\n"
            f"_(ATR={vni_info['atr_ratio']:.3f} vs threshold={vni_info['threshold']:.3f})_"
        )
        return [current]

    # MR signals
    if mr_signals:
        current += f"\n\n━━ 🔄 MEAN REVERSION (FWD=20d) ━━\n"
        for sig in mr_signals:
            sig_text = "\n" + _format_signal(sig, vni_info) + "\n"
            if len(current) + len(sig_text) > 3800:
                messages.append(current)
                current = sig_text
            else:
                current += sig_text
    else:
        current += f"\n\n🔄 *MR:* Không có signal"

    # MOM signals
    if mom_signals:
        current += f"\n━━ 🚀 MOMENTUM (FWD=10d) ━━\n"
        for sig in mom_signals:
            sig_text = "\n" + _format_signal(sig, vni_info) + "\n"
            if len(current) + len(sig_text) > 3800:
                messages.append(current)
                current = sig_text
            else:
                current += sig_text
    else:
        current += f"\n\n🚀 *MOM:* Không có signal"

    # Breakout signals
    if bo_signals:
        current += f"\n━━ 💥 BREAKOUT (FWD=15d) ━━\n"
        for sig in bo_signals:
            # Ghi chú nếu mã này cũng thuộc cluster khác
            dual_tag = ""
            if sig["symbol"] in MR_SYMBOLS:
                dual_tag = " _(+MR)_"
            elif sig["symbol"] in MOM_SYMBOLS:
                dual_tag = " _(+MOM)_"
            sig_text = "\n" + _format_signal(sig, vni_info, extra_tag=dual_tag) + "\n"
            if len(current) + len(sig_text) > 3800:
                messages.append(current)
                current = sig_text
            else:
                current += sig_text
    else:
        current += f"\n\n💥 *BO:* Không có signal"

    # Footer
    total_symbols = len(MR_SYMBOLS) + len(MOM_SYMBOLS) + len(BREAKOUT_SYMBOLS)
    footer = (
        f"\n━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_Scan: {total_symbols} mã (MR={len(MR_SYMBOLS)}, MOM={len(MOM_SYMBOLS)}, BO={len(BREAKOUT_SYMBOLS)})_\n"
        f"*VNI ATR:* {vni_info['atr_ratio']:.3f} "
        f"({'cao ✅' if vni_info['is_high'] else 'thấp ⚠️'} "
        f"vs threshold {vni_info['threshold']:.3f})\n"
    )
    if mr_no_signal:
        footer += f"MR không signal: {' '.join(mr_no_signal)}\n"
    if mom_no_signal:
        footer += f"MOM không signal: {' '.join(mom_no_signal)}\n"
    footer += f"⏰ Update tiếp: 12:30 VN"

    if len(current) + len(footer) > 3800:
        messages.append(current)
        messages.append(footer)
    else:
        messages.append(current + footer)

    return messages


def _format_afternoon_update(
    new_signals: list[dict],
    morning_updates: list[dict],
    vni_info: dict,
) -> list[str] | None:
    """
    Format 12:30 update.
    Trả về None nếu không có gì đáng gửi.
    """
    vn_now = datetime.utcnow() + timedelta(hours=7)
    has_new    = len(new_signals) > 0
    has_update = any(u["changed"] for u in morning_updates)

    # Không có gì mới → không gửi
    if not has_new and not has_update:
        logger.info("[Scanner] 12:30: No updates to send")
        return None

    header = (
        f"🔄 *UPDATE 12:30 VN*\n"
        f"📅 {vn_now.strftime('%d/%m/%Y %H:%M')} VN\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    messages = []
    current  = header

    # New signals
    if has_new:
        current += f"\n\n🆕 *SIGNAL MỚI (giá cập nhật):*\n"
        for sig in new_signals:
            current += "\n" + _format_signal(sig, vni_info) + "\n"

    # Morning signal updates (chỉ nếu có thay đổi đáng kể)
    changed = [u for u in morning_updates if u["changed"]]
    if changed:
        current += f"\n\n📊 *CẬP NHẬT SIGNAL 8:30:*\n"
        for u in changed:
            pnl_emoji = "🟢" if u["pnl"] >= 0 else "🔴"
            current += (
                f"\n{pnl_emoji} *{u['symbol']}* [{u['cluster_short']}]: "
                f"{u['pnl']:+.1f}% từ entry {u['entry']:,.0f} "
                f"→ giá hiện tại {u['current']:,.0f}"
            )
            if u.get("note"):
                current += f"\n  ⚠️ {u['note']}"

    current += f"\n\n_VNI ATR: {vni_info['atr_ratio']:.3f}_"

    if len(current) > 3800:
        messages.append(current[:3800])
    else:
        messages.append(current)

    return messages


# ── Main scan functions ───────────────────────────────────────────────────────


# ── Journal auto-logging (S31) ────────────────────────────────────────────────

def _journal_log_signals(signals: list, vni_info: dict) -> None:
    """
    Ghi danh sách signals vào cluster_journal.
    Bỏ qua nếu symbol đã có PENDING entry hôm nay (tránh duplicate từ cron 08:30 + 12:30).
    """
    if not signals:
        return
    try:
        from db import journal_add_signal, journal_get_active
        from datetime import date as _date
        today = _date.today()

        # Lấy set symbols đang PENDING để tránh duplicate
        active = journal_get_active()
        pending_today = {
            r["symbol"] for r in active
            if r["entry_date"] == today
        }

        vni_strong = vni_info.get("is_high", None)
        vni_soft   = "STRONG" if vni_strong else ("WEAK" if vni_strong is False else None)

        for sig in signals:
            sym = sig["symbol"]
            if sym in pending_today:
                logger.info(f"[Journal] {sym} da co PENDING hom nay, bo qua")
                continue
            jid = journal_add_signal(
                symbol       = sym,
                cluster      = sig["cluster"],
                entry_date   = today,
                entry_price  = sig["entry_price"],
                fwd_days     = sig["fwd_days"],
                sl_price     = sig.get("sl_price"),
                vni_atr_soft = vni_soft if sig["cluster"] == "Mean Reversion" else None,
                trigger_str  = sig.get("trigger_str"),
            )
            if jid > 0:
                logger.info(f"[Journal] Logged #{jid} {sym} {sig['cluster']}")
            else:
                logger.warning(f"[Journal] Failed to log {sym}")
    except Exception as e:
        logger.warning(f"[Journal] Auto-log failed (non-critical): {e}")


def run_morning_scan() -> tuple[list[str], dict]:
    """
    Chạy full scan cho cả 2 cluster.
    Trả về (messages, signals_dict).
    """
    logger.info("[Scanner] Starting morning scan...")
    vni_info = _get_vni_atr_info()

    mr_signals, mr_no_signal   = [], []
    mom_signals, mom_no_signal = [], []
    bo_signals,  bo_no_signal  = [], []

    for sym in MR_SYMBOLS:
        sig = _scan_symbol(sym, "Mean Reversion")
        if sig:
            mr_signals.append(sig)
            logger.info(f"[Scanner] {sym} MR SIGNAL: {sig['trigger_str']}")
        else:
            mr_no_signal.append(sym)

    for sym in MOM_SYMBOLS:
        sig = _scan_symbol(sym, "Momentum")
        if sig:
            mom_signals.append(sig)
            logger.info(f"[Scanner] {sym} MOM SIGNAL: {sig['trigger_str']}")
        else:
            mom_no_signal.append(sym)

    for sym in BREAKOUT_SYMBOLS:
        sig = _scan_symbol(sym, "Breakout")
        if sig:
            bo_signals.append(sig)
            logger.info(f"[Scanner] {sym} BO SIGNAL: {sig['trigger_str']}")
        else:
            bo_no_signal.append(sym)

    # Lưu vào memory để 12:30 update
    global _morning_signals
    _morning_signals = {}
    for sig in mr_signals + mom_signals + bo_signals:
        cluster_short = ("MR"  if sig["cluster"] == "Mean Reversion" else
                         "MOM" if sig["cluster"] == "Momentum" else "BO")
        _morning_signals[sig["symbol"]] = {
            "entry":         sig["entry_price"],
            "cluster":       sig["cluster"],
            "cluster_short": cluster_short,
            "scan_time":     sig["scan_time"],
            "last_date":     sig["last_date"],
        }

    # Ghi vao cluster_journal (S31)
    _journal_log_signals(mr_signals + mom_signals + bo_signals, vni_info)

    total = len(mr_signals) + len(mom_signals) + len(bo_signals)
    logger.info(f"[Scanner] Morning scan done: {total} signals "
                f"(MR={len(mr_signals)}, MOM={len(mom_signals)}, BO={len(bo_signals)})")

    messages = _format_morning_scan(
        mr_signals, mom_signals,
        mr_no_signal, mom_no_signal,
        vni_info, "08:30",
        bo_signals=bo_signals, bo_no_signal=bo_no_signal,
    )
    return messages, _morning_signals


def run_afternoon_update() -> list[str] | None:
    """
    Chạy 12:30 update:
    B. Cập nhật P&L của signals buổi sáng với giá mới nhất
    C. Scan lại xem có signal mới không
    """
    logger.info("[Scanner] Starting afternoon update...")
    vni_info = _get_vni_atr_info()

    # B. Update morning signals
    morning_updates = []
    for sym, info in _morning_signals.items():
        try:
            from vn_loader import load_vn_ohlcv
            df  = load_vn_ohlcv(sym, days=100, min_bars=60)
            cur = float(df["close"].iloc[-1])
            pnl = (cur - info["entry"]) / info["entry"] * 100

            # Chỉ báo nếu P&L đáng chú ý (> +3% hoặc < -3%)
            changed = abs(pnl) >= 3.0
            note    = None
            sl_pct  = SL_CONFIG.get(info["cluster"], -10)
            if pnl <= sl_pct * 0.8:
                note    = f"Tiếp cận SL ({sl_pct:+.1f}%)"
                changed = True

            morning_updates.append({
                "symbol":        sym,
                "cluster_short": info["cluster_short"],
                "entry":         info["entry"],
                "current":       round(cur, 2),
                "pnl":           round(pnl, 2),
                "changed":       changed,
                "note":          note,
            })
        except Exception as e:
            logger.debug(f"[Scanner] Update {sym}: {e}")

    # C. Scan lại với giá mới
    new_signals = []
    morning_syms = set(_morning_signals.keys())

    for sym in MR_SYMBOLS:
        if sym in morning_syms:
            continue   # đã có signal buổi sáng
        sig = _scan_symbol(sym, "Mean Reversion")
        if sig and sig["last_date"] != _morning_signals.get(sym, {}).get("last_date"):
            new_signals.append(sig)

    for sym in MOM_SYMBOLS:
        if sym in morning_syms:
            continue
        sig = _scan_symbol(sym, "Momentum")
        if sig:
            new_signals.append(sig)

    if new_signals:
        logger.info(f"[Scanner] Afternoon: {len(new_signals)} new signals")
        # Ghi signals moi buoi chieu vao journal (S31)
        _journal_log_signals(new_signals, vni_info)

    return _format_afternoon_update(new_signals, morning_updates, vni_info)


# ── Telegram command handler ──────────────────────────────────────────────────

async def cluster_scan_cmd(update, context):
    """
    /cluster_scan — chạy manual scan ngay lập tức.
    """
    await update.message.reply_text("🔍 Đang scan cluster signals...")
    try:
        messages, _ = await asyncio.to_thread(run_morning_scan)
        for m in messages:
            await update.message.reply_text(
                m, parse_mode="Markdown"
            )
            await asyncio.sleep(0.3)
    except Exception as e:
        await update.message.reply_text(f"❌ Scan lỗi: {str(e)[:200]}")


# ── Cron loops ────────────────────────────────────────────────────────────────

async def _start_cluster_scan_cron(bot, chat_ids: list[int]):
    """
    Khởi động cả 2 cron tasks:
      - Morning scan: 08:30 VN (01:30 UTC)
      - Afternoon update: 12:30 VN (05:30 UTC)
    """
    asyncio.create_task(_morning_cron(bot, chat_ids))
    asyncio.create_task(_afternoon_cron(bot, chat_ids))
    logger.info(f"[ClusterCron] Started: morning=08:30 VN, afternoon=12:30 VN | "
                f"{len(chat_ids)} chat_ids")


async def _morning_cron(bot, chat_ids: list[int]):
    """Cron 08:30 VN — full scan."""
    import datetime as _dt

    while True:
        now    = _dt.datetime.utcnow()
        target = now.replace(
            hour=MORNING_HOUR, minute=MORNING_MINUTE,
            second=0, microsecond=0
        )
        if now >= target:
            target += _dt.timedelta(days=1)

        wait   = (target - now).total_seconds()
        vn_t   = target + _dt.timedelta(hours=7)
        logger.info(
            f"[MorningCron] Next: {wait/3600:.1f}h "
            f"(UTC {target.strftime('%H:%M')} = VN {vn_t.strftime('%H:%M')})"
        )
        await asyncio.sleep(wait)

        logger.info("[MorningCron] Running morning scan...")
        try:
            messages, _ = await asyncio.to_thread(run_morning_scan)
            for cid in chat_ids:
                for m in messages:
                    try:
                        await bot.send_message(
                            chat_id=cid, text=m[:4000],
                            parse_mode="Markdown"
                        )
                        await asyncio.sleep(0.3)
                    except Exception as se:
                        logger.warning(f"[MorningCron] send {cid}: {se}")
        except Exception as e:
            import traceback
            logger.error(f"[MorningCron] ERROR: {e}\n{traceback.format_exc()}")
            err = f"❌ Cluster scan 8:30 lỗi: {str(e)[:200]}"
            for cid in chat_ids:
                try:
                    await bot.send_message(chat_id=cid, text=err)
                except Exception:
                    pass


async def _afternoon_cron(bot, chat_ids: list[int]):
    """Cron 12:30 VN — update + re-scan."""
    import datetime as _dt

    while True:
        now    = _dt.datetime.utcnow()
        target = now.replace(
            hour=AFTERNOON_HOUR, minute=AFTERNOON_MINUTE,
            second=0, microsecond=0
        )
        if now >= target:
            target += _dt.timedelta(days=1)

        wait   = (target - now).total_seconds()
        vn_t   = target + _dt.timedelta(hours=7)
        logger.info(
            f"[AfternoonCron] Next: {wait/3600:.1f}h "
            f"(UTC {target.strftime('%H:%M')} = VN {vn_t.strftime('%H:%M')})"
        )
        await asyncio.sleep(wait)

        logger.info("[AfternoonCron] Running afternoon update...")
        try:
            messages = await asyncio.to_thread(run_afternoon_update)
            if messages is None:
                logger.info("[AfternoonCron] No updates — skip send")
                continue
            for cid in chat_ids:
                for m in messages:
                    try:
                        await bot.send_message(
                            chat_id=cid, text=m[:4000],
                            parse_mode="Markdown"
                        )
                        await asyncio.sleep(0.3)
                    except Exception as se:
                        logger.warning(f"[AfternoonCron] send {cid}: {se}")
        except Exception as e:
            import traceback
            logger.error(f"[AfternoonCron] ERROR: {e}\n{traceback.format_exc()}")
            # Afternoon errors không cần alert Telegram (không critical)
