"""
cluster_scanner.py — Cluster-Based Daily Signal Scanner
VN Trader Bot V6 — Session 35

Thay thế batch_scanner.py với 2-cluster approach từ Session 30.

Cron schedule (UTC, server Render):
  01:30 UTC = 08:30 VN  ← trước giờ mở cửa HOSE 09:00
  05:30 UTC = 12:30 VN  ← giữa phiên, lấy giá mới nhất

Logic mỗi lần scan:
  8:30:  Full scan → phát signal mới nếu có
  12:30: B+C combo:
         B. Update P&L của signals đã phát buổi sáng
         C. Scan lại với giá mới → alert nếu có signal mới

Clusters (S35 validated):
  Mean Reversion (FWD=20d): NAB[A], BMP[A], LPB[B*], HDB[B*], SSI[B*], FRT[B*],
                             AGR[B*], BSR[B*], VCB[B*], NLG[B*], IJC[B*], PC1[B*],
                             CTI[B*], REE[B*], TLG[B*], KDH[B*], PVP[B*], BWE[B*], HPG[B*]
                             (* = partial pass, chỉ Option B)
  Momentum      (FWD=10d):  VIX[A], SSI[A], VDS[A], LPB[A], VTP[A], BSI[B], SHB[B],
                             NVL[B], QCG[B], FTS[B], SIP[B], CTS[B], DCM[B], BSR[B],
                             MCH[B], DPM[B], HAH[B] + partial: ANV, GEX, DXS, MBB, CTG
  Breakout      (FWD=15d):  BFC[A], VSC[A], BMP[B], FRT[B] + partial: TCH, LPB, HDB, MCH, TCB, VTP

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
    # ── S35 scan v4.1 — Kết quả chính thức sau fix bug regime p70 ────────────
    # Tier A — pass cả 2 options:
    "NAB",   # [A] score=5.52 OOS=+5.52% WFE=4.38* consist=100% ⭐ NEW (*WFE inflate)
    "BMP",   # [A] score=4.40 OOS=+5.28% WFE=3.65  consist=83%  ⭐ NEW (cũng BO)
    # Tier B — chỉ pass Option B (partial pass):
    "LPB",   # [B] score=3.60 OOS=+4.50% WFE=1.77  consist=80%  ⭐ NEW (cũng MOM A, BO A) | PARTIAL
    "HDB",   # [B] score=3.31 OOS=+4.41% WFE=3.93* consist=75%  ✅ GIỮ từ S33 (cũng BO A) | PARTIAL (*WFE inflate)
    "SSI",   # [B] score=2.42 OOS=+3.03% WFE=2.89  consist=80%  ⭐ NEW (cũng MOM B) | PARTIAL
    "FRT",   # [B] score=2.32 OOS=+3.47% WFE=0.85  consist=67%  ⭐ NEW (cũng BO B) | PARTIAL
    "AGR",   # [B] score=2.12 OOS=+2.55% WFE=1.53  consist=83%  ⭐ NEW | PARTIAL
    "BSR",   # [B] score=2.11 OOS=+2.64% WFE=1.33  consist=80%  ⭐ NEW (cũng MOM B) | PARTIAL
    "VCB",   # [B] score=2.02 OOS=+2.53% WFE=1.52  consist=80%  ⭐ NEW | PARTIAL
    "NLG",   # [B] score=2.02 OOS=+3.02% WFE=1.92  consist=67%  ⭐ NEW | PARTIAL
    "IJC",   # [B] score=1.86 OOS=+2.33% WFE=5.02* consist=80%  ⭐ NEW | PARTIAL (*WFE inflate)
    "PC1",   # [B] score=1.63 OOS=+2.44% WFE=2.91  consist=67%  ⭐ NEW | PARTIAL
    "CTI",   # [B] score=1.61 OOS=+2.15% WFE=4.01* consist=75%  ⭐ NEW | PARTIAL (*WFE inflate)
    "REE",   # [B] score=1.60 OOS=+2.39% WFE=1.32  consist=67%  ⭐ NEW | PARTIAL
    "TLG",   # [B] score=1.50 OOS=+2.24% WFE=1.36  consist=67%  ⭐ NEW | PARTIAL
    "KDH",   # [B] score=1.46 OOS=+2.44% WFE=4.48* consist=60%  ⭐ NEW | PARTIAL (*WFE inflate)
    "PVP",   # [B] score=1.40 OOS=+2.33% WFE=1.13  consist=60%  ⭐ NEW | PARTIAL
    "BWE",   # [B] score=1.14 OOS=+1.36% WFE=1.06  consist=83%  ⭐ NEW | PARTIAL
    "HPG",   # [B] score=1.12 OOS=+1.67% WFE=1.84  consist=67%  ⭐ NEW | PARTIAL
    # ── Loại so với S33 ───────────────────────────────────────────────────────
    # DCM: fail MR S35 → giữ trong MOM B
    # NKG: fail MR S35 → loại
    # DPM: fail MR S35 → giữ trong MOM B
    # HAH: fail MR S35 → giữ trong MOM B
    # HCM: fail MR S35 → loại
    # HSG: fail MR S35 → loại
    # DGC: fail MR S35 → loại
    # VGC: score=0.75 < 1.0 → loại
]
MOM_SYMBOLS = [
    # ── S34 scan v4 — Pass CẢ 2 options (Option A + B) ───────────────────────
    # Tier A (score >= 4, consist 100%):
    "VIX",   # [A] OOS=+6.26% WFE=2.06 consist=100% ← current ✅
    "SSI",   # [A] OOS=+4.84% WFE=7.58* consist=100% ← current ✅ (*WFE inflate)
    "VDS",   # [A] OOS=+4.74% WFE=2.92 consist=100% ← current ✅
    "LPB",   # [A] OOS=+4.45% WFE=1.85 consist=100% ← current ✅
    "VTP",   # [A] OOS=+4.17% WFE=2.43 consist=100% ⭐ NEW
    # Tier B (score 1-4, consist >= 60%):
    "BSI",   # [B] OOS=+3.80% WFE=2.17 consist=100% ⭐ NEW
    "SHB",   # [B] OOS=+3.55% WFE=6.12* consist=100% ⭐ NEW (*WFE inflate)
    "NVL",   # [B] OOS=+3.28% WFE=3.77 consist=100% ⭐ NEW
    "QCG",   # [B] OOS=+4.59% WFE=0.94 consist=67%  ⭐ NEW
    "FTS",   # [B] OOS=+2.94% WFE=1.27 consist=100% ← current ✅
    "SIP",   # [B] OOS=+2.94% WFE=1.24 consist=100% ⭐ NEW
    "CTS",   # [B] OOS=+2.55% WFE=2.66 consist=100% ← current ✅
    "DCM",   # [B] OOS=+2.46% WFE=2.51 consist=100% (từ MR S33)
    "BSR",   # [B] OOS=+3.24% WFE=2.00 consist=75%  ⭐ NEW
    "MCH",   # [B] OOS=+3.59% WFE=3.57 consist=60%  ⭐ NEW (cũng pass BO)
    "DPM",   # [B] OOS=+2.72% WFE=2.92 consist=67%  (từ MR S33)
    "HAH",   # [B] OOS=+1.43% WFE=1.80 consist=100% (từ MR S33)
    # ── Partial pass — chỉ 1 option (note khi phát tín hiệu) ─────────────────
    "ANV",   # [B] OOS=+2.75% — CHỈ Option A | ⭐ NEW
    "GEX",   # [B] OOS=+3.11% — CHỈ Option A | ⭐ NEW
    "DXS",   # [B] OOS=+3.14% — CHỈ Option B | ⭐ NEW
    "MBB",   # [B] OOS=+2.62% — CHỈ Option B | WFE inflate | ⭐ NEW
    "CTG",   # [B] OOS=+1.23% — CHỈ Option B | score biên
    # ── Loại so với S33 ───────────────────────────────────────────────────────
    # VND: fail cả 2 options → loại
    # HAG: fail cả 2 options → loại
]

# Breakout cluster (S34 scan v4 results)
BREAKOUT_SYMBOLS = [
    # ── Pass cả 2 options ─────────────────────────────────────────────────────
    "BFC",   # [A] OOS=+4.62% WFE=2.61 consist=100% ⭐ NEW
    "VSC",   # [A] OOS=+6.01% WFE=3.87 consist=67%  ⭐ NEW
    "BMP",   # [B] OOS=+2.84% WFE=1.53 consist=100% ⭐ NEW
    "FRT",   # [B] OOS=+3.01% WFE=2.23 consist=67%  ⭐ NEW
    # ── Partial pass — chỉ Option A (note khi phát tín hiệu) ─────────────────
    "TCH",   # [A] OOS=+6.23% WFE=2.01 consist=100% — CHỈ Option A | ⭐ NEW
    "LPB",   # [A] OOS=+8.04% WFE=3.54 consist=75%  — CHỈ Option A (cũng MOM [A])
    "HDB",   # [B] OOS=+1.57% WFE=4.05* consist=75% — CHỈ Option A | score biên 1.18
    # ── Partial pass — chỉ Option B ──────────────────────────────────────────
    "MCH",   # [B] OOS=+4.63% WFE=1.61 consist=60%  — CHỈ Option B (cũng MOM [B])
    "TCB",   # [B] OOS=+3.62% WFE=1.48 consist=75%  — CHỈ Option B ← current S33
    "VTP",   # [B] OOS=+3.57% WFE=2.72 consist=75%  — CHỈ Option B (cũng MOM [A])
    # ── Loại so với S33 ───────────────────────────────────────────────────────
    # HT1: fail cả 2 options → loại
    # GMD: fail cả 2 options → loại
    # DGC: fail cả 2 options → loại
    # DCM: fail BO → chuyển sang MOM
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
        "description": "Mua khi EMA12>EMA26 + momentum mạnh + volume xác nhận + nến thân lớn",
    },
    "Breakout": {
        "regime_indicator":  "bb_squeeze",
        "regime_condition":  "high",
        # FIX S36: consolidation direction = "high" (nhiều ngày tích lũy = tốt)
        # công thức % ngày sideways đã đúng, chỉ cần flip direction low→high
        "trigger_indicators":["consolidation", "vol_dry_up"],
        "trigger_direction": {"consolidation": "high", "vol_dry_up": "high"},
        "description": "Mua khi BB rộng + giá sideways + volume khô → bứt phá",
    },
}

TRIGGER_PCT  = 70
MIN_TRIGGERS = 2

# ── Per-symbol WF stats (S34 scan v4) ────────────────────────────────────────
# Format: {wr, exp, wfe, n, pf, cluster}
# Score = OOS_exp × (consistency/100) — từ v4 pipeline
# partial_pass: True = chỉ pass 1 option → note trong signal
# wfe_inflate: True = WFE > 5 do IS_exp thấp → không tin WFE, dùng OOS_exp
SYMBOL_STATS = {
    # ── Mean Reversion — S35 scan v4.1 (chính thức sau fix bug regime p70) ────
    # Tier A — pass cả 2 options
    "NAB":    {"wr": 63, "exp": 5.52, "wfe": 4.38, "n": 19, "pf": 2.49, "cluster": "Mean Reversion",
               "score": 5.52, "consist": 100, "partial_pass": False, "wfe_inflate": True},
    "BMP_MR": {"wr": 60, "exp": 5.28, "wfe": 3.65, "n": 21, "pf": 2.35, "cluster": "Mean Reversion",
               "score": 4.40, "consist": 83,  "partial_pass": False},
    # Tier B — chỉ pass Option B (partial pass)
    "LPB_MR": {"wr": 58, "exp": 4.50, "wfe": 1.77, "n": 18, "pf": 1.32, "cluster": "Mean Reversion",
               "score": 3.60, "consist": 80,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "HDB_MR": {"wr": 56, "exp": 4.41, "wfe": 3.93, "n": 17, "pf": 1.39, "cluster": "Mean Reversion",
               "score": 3.31, "consist": 75,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B", "wfe_inflate": True},
    "SSI_MR": {"wr": 57, "exp": 3.03, "wfe": 2.89, "n": 21, "pf": 2.06, "cluster": "Mean Reversion",
               "score": 2.42, "consist": 80,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "FRT_MR": {"wr": 55, "exp": 3.47, "wfe": 0.85, "n": 28, "pf": 1.62, "cluster": "Mean Reversion",
               "score": 2.32, "consist": 67,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "AGR":    {"wr": 56, "exp": 2.55, "wfe": 1.53, "n": 25, "pf": 1.61, "cluster": "Mean Reversion",
               "score": 2.12, "consist": 83,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "BSR_MR": {"wr": 55, "exp": 2.64, "wfe": 1.33, "n": 21, "pf": 1.22, "cluster": "Mean Reversion",
               "score": 2.11, "consist": 80,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "VCB":    {"wr": 55, "exp": 2.53, "wfe": 1.52, "n": 24, "pf": 1.70, "cluster": "Mean Reversion",
               "score": 2.02, "consist": 80,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "NLG":    {"wr": 54, "exp": 3.02, "wfe": 1.92, "n": 25, "pf": 1.44, "cluster": "Mean Reversion",
               "score": 2.02, "consist": 67,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "IJC":    {"wr": 56, "exp": 2.33, "wfe": 5.02, "n": 22, "pf": 1.76, "cluster": "Mean Reversion",
               "score": 1.86, "consist": 80,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B", "wfe_inflate": True},
    "PC1":    {"wr": 54, "exp": 2.44, "wfe": 2.91, "n": 25, "pf": 1.39, "cluster": "Mean Reversion",
               "score": 1.63, "consist": 67,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "CTI":    {"wr": 55, "exp": 2.15, "wfe": 4.01, "n": 14, "pf": 1.36, "cluster": "Mean Reversion",
               "score": 1.61, "consist": 75,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B", "wfe_inflate": True},
    "REE":    {"wr": 54, "exp": 2.39, "wfe": 1.32, "n": 23, "pf": 2.21, "cluster": "Mean Reversion",
               "score": 1.60, "consist": 67,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "TLG":    {"wr": 54, "exp": 2.24, "wfe": 1.36, "n": 26, "pf": 1.62, "cluster": "Mean Reversion",
               "score": 1.50, "consist": 67,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "KDH":    {"wr": 53, "exp": 2.44, "wfe": 4.48, "n": 22, "pf": 1.65, "cluster": "Mean Reversion",
               "score": 1.46, "consist": 60,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B", "wfe_inflate": True},
    "PVP":    {"wr": 53, "exp": 2.33, "wfe": 1.13, "n": 20, "pf": 1.84, "cluster": "Mean Reversion",
               "score": 1.40, "consist": 60,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "BWE":    {"wr": 54, "exp": 1.36, "wfe": 1.06, "n": 26, "pf": 1.47, "cluster": "Mean Reversion",
               "score": 1.14, "consist": 83,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    "HPG":    {"wr": 53, "exp": 1.67, "wfe": 1.84, "n": 22, "pf": 1.70, "cluster": "Mean Reversion",
               "score": 1.12, "consist": 67,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B"},
    # ── Momentum — S34 scan v4 ────────────────────────────────────────────────
    # Tier A — pass cả 2 options
    "VIX":  {"wr": 61, "exp": 6.26, "wfe": 2.06, "n": 24, "pf": 2.50, "cluster": "Momentum",
             "score": 6.26, "consist": 100, "partial_pass": False},
    "SSI":  {"wr": 60, "exp": 4.84, "wfe": 7.58, "n": 20, "pf": 1.52, "cluster": "Momentum",
             "score": 4.84, "consist": 100, "partial_pass": False, "wfe_inflate": True},
    "VDS":  {"wr": 72, "exp": 4.74, "wfe": 2.92, "n": 26, "pf": 2.21, "cluster": "Momentum",
             "score": 4.74, "consist": 100, "partial_pass": False},
    "LPB":  {"wr": 74, "exp": 4.45, "wfe": 1.85, "n": 22, "pf": 2.02, "cluster": "Momentum",
             "score": 4.45, "consist": 100, "partial_pass": False},
    "VTP":  {"wr": 60, "exp": 4.17, "wfe": 2.43, "n": 29, "pf": 1.81, "cluster": "Momentum",
             "score": 4.17, "consist": 100, "partial_pass": False},
    # Tier B — pass cả 2 options
    "BSI":  {"wr": 55, "exp": 3.80, "wfe": 2.17, "n": 20, "pf": 2.06, "cluster": "Momentum",
             "score": 3.80, "consist": 100, "partial_pass": False},
    "SHB":  {"wr": 55, "exp": 3.55, "wfe": 6.12, "n": 22, "pf": 2.21, "cluster": "Momentum",
             "score": 3.55, "consist": 100, "partial_pass": False, "wfe_inflate": True},
    "NVL":  {"wr": 53, "exp": 3.28, "wfe": 3.77, "n": 22, "pf": 2.61, "cluster": "Momentum",
             "score": 3.28, "consist": 100, "partial_pass": False},
    "QCG":  {"wr": 52, "exp": 4.59, "wfe": 0.94, "n": 29, "pf": 1.67, "cluster": "Momentum",
             "score": 3.06, "consist": 67,  "partial_pass": False},
    "FTS":  {"wr": 58, "exp": 2.94, "wfe": 1.27, "n": 15, "pf": 3.09, "cluster": "Momentum",
             "score": 2.94, "consist": 100, "partial_pass": False},
    "SIP":  {"wr": 57, "exp": 2.94, "wfe": 1.24, "n": 21, "pf": 2.17, "cluster": "Momentum",
             "score": 2.94, "consist": 100, "partial_pass": False},
    "CTS":  {"wr": 56, "exp": 2.55, "wfe": 2.66, "n": 23, "pf": 2.48, "cluster": "Momentum",
             "score": 2.55, "consist": 100, "partial_pass": False},
    "DCM":  {"wr": 57, "exp": 2.46, "wfe": 2.51, "n": 17, "pf": 1.65, "cluster": "Momentum",
             "score": 2.46, "consist": 100, "partial_pass": False},
    "BSR":  {"wr": 56, "exp": 3.24, "wfe": 2.00, "n": 18, "pf": 2.79, "cluster": "Momentum",
             "score": 2.43, "consist": 75,  "partial_pass": False},
    "MCH":  {"wr": 59, "exp": 3.59, "wfe": 3.57, "n": 32, "pf": 2.14, "cluster": "Momentum",
             "score": 2.15, "consist": 60,  "partial_pass": False},
    "DPM":  {"wr": 55, "exp": 2.72, "wfe": 2.92, "n": 12, "pf": 1.64, "cluster": "Momentum",
             "score": 1.81, "consist": 67,  "partial_pass": False},
    "HAH":  {"wr": 58, "exp": 1.43, "wfe": 1.80, "n": 13, "pf": 1.76, "cluster": "Momentum",
             "score": 1.43, "consist": 100, "partial_pass": False},
    # Partial pass — chỉ 1 option (note trong signal)
    "ANV":  {"wr": 55, "exp": 2.75, "wfe": 2.33, "n": 41, "pf": 1.55, "cluster": "Momentum",
             "score": 2.75, "consist": 100, "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option A"},
    "GEX":  {"wr": 55, "exp": 3.11, "wfe": 1.93, "n": 33, "pf": 2.32, "cluster": "Momentum",
             "score": 2.33, "consist": 75,  "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option A"},
    "DXS":  {"wr": 50, "exp": 3.14, "wfe": 2.25, "n": 25, "pf": 1.32, "cluster": "Momentum",
             "score": 3.14, "consist": 100, "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option B"},
    "MBB":  {"wr": 47, "exp": 2.62, "wfe": 9.78, "n": 20, "pf": 1.74, "cluster": "Momentum",
             "score": 1.97, "consist": 75,  "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option B",
             "wfe_inflate": True},
    "CTG":  {"wr": 50, "exp": 1.23, "wfe": 1.11, "n": 25, "pf": 1.96, "cluster": "Momentum",
             "score": 1.02, "consist": 83,  "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option B · score biên"},
    # ── Breakout — S34 scan v4 ────────────────────────────────────────────────
    # Tier A — pass cả 2 options
    "BFC":  {"wr": 55, "exp": 4.62, "wfe": 2.61, "n": 12, "pf": 1.25, "cluster": "Breakout",
             "score": 4.62, "consist": 100, "partial_pass": False},
    "VSC":  {"wr": 60, "exp": 6.01, "wfe": 3.87, "n": 11, "pf": 1.55, "cluster": "Breakout",
             "score": 4.01, "consist": 67,  "partial_pass": False},
    # Tier B — pass cả 2 options
    "BMP":  {"wr": 58, "exp": 2.84, "wfe": 1.53, "n": 19, "pf": 1.81, "cluster": "Breakout",
             "score": 2.84, "consist": 100, "partial_pass": False},
    "FRT_BO": {"wr": 55, "exp": 3.01, "wfe": 2.23, "n": 12, "pf": 1.91, "cluster": "Breakout",
               "score": 2.01, "consist": 67,  "partial_pass": False},
    # Partial pass — chỉ Option A
    "TCH":  {"wr": 60, "exp": 6.23, "wfe": 2.01, "n": 15, "pf": 2.35, "cluster": "Breakout",
             "score": 6.23, "consist": 100, "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option A"},
    "LPB_BO": {"wr": 60, "exp": 8.04, "wfe": 3.54, "n": 24, "pf": 3.61, "cluster": "Breakout",
               "score": 6.03, "consist": 75,  "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option A (cũng MOM Tier A)"},
    "HDB":  {"wr": 55, "exp": 1.57, "wfe": 4.05, "n": 23, "pf": 1.73, "cluster": "Breakout",
             "score": 1.18, "consist": 75,  "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option A · score biên · WFE inflate",
             "wfe_inflate": True},
    # Partial pass — chỉ Option B
    "TCB":  {"wr": 90, "exp": 3.62, "wfe": 1.48, "n": 17, "pf": 1.46, "cluster": "Breakout",
             "score": 2.72, "consist": 75,  "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option B"},
    "VTP_BO": {"wr": 60, "exp": 3.57, "wfe": 2.72, "n": 18, "pf": 2.02, "cluster": "Breakout",
               "score": 2.67, "consist": 75,  "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option B (cũng MOM Tier A)"},
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
    # S31 expand (18 ma moi)
    "CTS": {"mult": 2.5, "activation_pct": 5.26},   # MOM | WFE=2.89  consistency=80%
    "HAG": {"mult": 2.0, "activation_pct": 4.60},   # MOM | WFE=357.5 consistency=80%
}

# ── Vibe Filter Config (S33) ─────────────────────────────────────────────────
# Kết quả từ backtest + walk forward per-symbol (Session 33)
# HARD_FILTER: chỉ vào lệnh khi engine đồng ý (signal = +1)
# BONUS:       vào lệnh bình thường, tự tin hơn khi engine đồng ý
#
# Cách áp dụng trong cluster_scanner:
#   1. Khi phát hiện cluster signal cho symbol S:
#   2. Kiểm tra VIBE_FILTER_CONFIG[cluster][symbol]["hard"]
#      → Nếu có engine trong list: chạy engine đó, chỉ forward signal khi engine = +1
#   3. Kiểm tra VIBE_FILTER_CONFIG[cluster][symbol]["bonus"]
#      → Nếu có engine trong list: chạy engine đó, thêm note vào signal output
#
# Metrics OOS (avg across WF folds):
#   MR:  Exp baseline ~3.2%, sau HARD filter ~3.0-4.4% (trừ DGC/HDB/BMP bỏ filter)
#   MOM: Exp baseline ~2.6%, sau HARD filter ~3.9% (+47%)
#   BO:  Không có HARD filter — chỉ BONUS

VIBE_FILTER_CONFIG = {
    "Mean Reversion": {
        # S35 vibe backtest validated (Session 36)
        # Logic: HARD = decay<=25% AND tot_filt>=tot_base | BONUS = decay<=25% AND exp_filt>exp_base
        # HARD: SMC/NAB, SMC/BMP (cov 100%) | MultiFactor/SSI (tot_filt>tot_base)
        # BONUS: exp_filt > exp_base, decay <= 25% — per engine/symbol pair
        "NAB":    {"hard": ["SMC"],         "bonus": ["Volatility", "CrossMarket", "MultiFactor"]},
        "BMP":    {"hard": ["SMC"],         "bonus": []},
        "LPB":    {"hard": [],              "bonus": []},
        "HDB":    {"hard": [],              "bonus": []},
        "SSI":    {"hard": ["MultiFactor"], "bonus": []},
        "FRT":    {"hard": [],              "bonus": ["Candlestick"]},
        "AGR":    {"hard": [],              "bonus": ["SMC", "CrossMarket"]},
        "BSR":    {"hard": [],              "bonus": []},
        "VCB":    {"hard": [],              "bonus": ["Volatility"]},
        "NLG":    {"hard": [],              "bonus": []},
        "IJC":    {"hard": [],              "bonus": ["CrossMarket"]},
        "PC1":    {"hard": [],              "bonus": []},
        "CTI":    {"hard": [],              "bonus": ["MultiFactor"]},
        "REE":    {"hard": [],              "bonus": ["SMC", "Chanlun"]},
        "TLG":    {"hard": [],              "bonus": ["Volatility", "MultiFactor"]},
        "KDH":    {"hard": [],              "bonus": ["CrossMarket"]},
        "PVP":    {"hard": [],              "bonus": []},
        "BWE":    {"hard": [],              "bonus": ["Chanlun"]},
        "HPG":    {"hard": [],              "bonus": []},
    },
    "Momentum": {
        # S35 vibe backtest validated (Session 36)
        # Logic: HARD = decay<=25% AND tot_filt>=tot_base | BONUS = decay<=25% AND exp_filt>exp_base
        # NULL filter (cov=100%): SMC/HAH, SMC/FTS → LOAI
        # HARD: SMC/SSI | CrossMarket/LPB,BSR,DXS | MultiFactor/ANV,GEX
        # BONUS: TechnicalBasic/LPB | Volatility/MBB
        "SSI": {"hard": ["SMC"],         "bonus": []},
        "LPB": {"hard": ["CrossMarket"], "bonus": ["TechnicalBasic"]},
        "BSR": {"hard": ["CrossMarket"], "bonus": []},
        "DXS": {"hard": ["CrossMarket"], "bonus": []},
        "ANV": {"hard": ["MultiFactor"], "bonus": []},
        "GEX": {"hard": ["MultiFactor"], "bonus": []},
        "MBB": {"hard": [],              "bonus": ["Volatility"]},
        "VTP": {"hard": [],  "bonus": []},
        "BSI": {"hard": [],  "bonus": []},
        "SHB": {"hard": [],  "bonus": []},
        "NVL": {"hard": [],  "bonus": []},
        "QCG": {"hard": [],  "bonus": []},
        "SIP": {"hard": [],  "bonus": []},
        "DCM": {"hard": [],  "bonus": []},
        "MCH": {"hard": [],  "bonus": []},
        "DPM": {"hard": [],  "bonus": []},
        "HAH": {"hard": [],  "bonus": []},
        "FTS": {"hard": [],  "bonus": []},
        "CTG": {"hard": [],  "bonus": []},
        "VND": {"hard": [],  "bonus": []},
        "VIX": {"hard": [],  "bonus": []},
        "CTS": {"hard": [],  "bonus": []},
        "HAG": {"hard": [],  "bonus": []},
        "VDS": {"hard": [],  "bonus": []},
    },
    "Breakout": {
        # S35 watchlist (Session 36) — 10 mã
        # Vibe backtest S35: 0 HARD, 0 BONUS
        #   SMC/BMP: null filter (cov=100%) + decay 40% → LOAI
        #   SMC/TCH: null filter (cov=100%) → LOAI (theo precedent HAH/FTS MOM)
        #   CrossMarket/FRT: exp_filt < exp_base → LOAI
        #   CrossMarket/TCH: exp_filt < exp_base → LOAI
        # → Tất cả mã reset về hard=[], bonus=[]
        "BFC": {"hard": [], "bonus": []},
        "VSC": {"hard": [], "bonus": []},
        "BMP": {"hard": [], "bonus": []},
        "FRT": {"hard": [], "bonus": []},
        "TCH": {"hard": [], "bonus": []},
        "LPB": {"hard": [], "bonus": []},
        "HDB": {"hard": [], "bonus": []},
        "MCH": {"hard": [], "bonus": []},
        "TCB": {"hard": [], "bonus": []},
        "VTP": {"hard": [], "bonus": []},
    },
}

# Agree/Disagree bonus khi tích hợp vibe score vào signal score
VIBE_AGREE_BONUS    =  0.20   # +20% score khi engine đồng ý
VIBE_DISAGREE_BONUS = -0.20   # -20% score khi engine phủ nhận (chỉ áp dụng BONUS engines)

# Account size để tính position sizing (VND)
# Chỉnh theo vốn thực tế của bạn
ACCOUNT_SIZE = 300_000_000  # 300 triệu

# Continuous position sizing config
BASE_RISK_PCT  = 1.0    # % account cho mã có Score = MEDIAN_SCORE
MIN_RISK_PCT   = 0.4    # % tối thiểu (mã yếu nhất)
MAX_RISK_PCT   = 2.0    # % tối đa (mã mạnh nhất, 2x base)
MEDIAN_SCORE   = 3.0    # median score S34 (OOS_exp × consist/100)

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
        # FIX S36: momentum_3d cho MOM guard — đà ngắn hạn 3 ngày
        "momentum_3d":    float((px / (close[max(0, i-3)] + 1e-9) - 1.0) * 100),
        "volume_spike":   float((vol[i] / (vs20v + 1e-9)) - 1.0),
        "stoch_k":        float(stoch[i]),
        "candle_body":    float(np.clip(abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
        # FIX S36: candle_bull + volume_spike_bull cho MOM trigger (có hướng nến)
        "candle_bull":    float(np.clip((px - opn[i]) / (atr_v + 1e-9), -3, 3)),
        "volume_spike_bull": float((vol[i] / (vs20v + 1e-9)) - 1.0) if px >= opn[i] else -1.0,
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

    # FIX S33 Bug 3: tính vsma60 1 lần ngoài loop, không tính lại mỗi vòng
    vsma60 = _sma(vol, 60)
    # FIX S33: tính bb_std toàn series 1 lần bằng rolling (nhanh hơn nhiều)
    bb_std_series = pd.Series(close).rolling(20).std().values

    rows = []
    for i in range(60, n):
        px    = close[i]
        atr_v = atr[i]    if np.isfinite(atr[i])    else px * 0.02
        s50   = sma50[i]  if np.isfinite(sma50[i])  else px
        vs20v = vsma20[i] if np.isfinite(vsma20[i]) else vol[i]
        c5    = close[max(i - 5, 0)]
        # Breakout indicators — dùng series đã tính sẵn
        vsma60_v_ = vsma60[i] if np.isfinite(vsma60[i]) else vs20v
        bb_std_   = bb_std_series[i] if np.isfinite(bb_std_series[i]) else atr_v
        bb_width_ = float(4 * float(bb_std_) / (px + 1e-9) * 100)
        window_   = close[max(0, i - 14):i + 1]
        consol_   = float(np.sum(np.abs(window_ - px) / (px + 1e-9) < 0.03)) / max(len(window_), 1)

        rows.append({
            "price_vs_sma50": float((px - s50) / (px + 1e-9) * 100),
            "ema_cross":      float((ema12[i] - ema26[i]) / (px + 1e-9) * 100),
            "momentum_5d":    float((px / (c5 + 1e-9) - 1.0) * 100),
            "volume_spike":   float((vol[i] / (vs20v + 1e-9)) - 1.0),
            "stoch_k":        float(stoch[i]),
            "candle_body":    float(np.clip(abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
            # FIX S36: candle_bull (có hướng) cho MOM — không fire khi nến đỏ to
            "candle_bull":    float(np.clip((px - opn[i]) / (atr_v + 1e-9), -3, 3)),
            # FIX S36: volume_spike_bull = -1 khi nến đỏ — không fire khi bán tháo
            "volume_spike_bull": float((vol[i] / (vs20v + 1e-9)) - 1.0) if px >= opn[i] else -1.0,
            "bb_squeeze":     bb_width_,
            "consolidation":  consol_,
            "vol_dry_up":     float((vs20v / (vsma60_v_ + 1e-9)) - 1.0),
        })

    reg_ind  = cfg["regime_indicator"]
    trig_ind = cfg["trigger_indicators"]
    trig_dir = cfg["trigger_direction"]
    reg_cond = cfg["regime_condition"]

    reg_vals = [r[reg_ind] for r in rows if np.isfinite(r.get(reg_ind, float("nan")))]

    # FIX S33 Bug 1: dùng percentile nhất quán với backtest (TRIGGER_PCT=70)
    # regime "low"  → threshold = p70 (chỉ 30% ngày thấp nhất mới pass)
    # regime "high" → threshold = p30 (chỉ 30% ngày cao nhất mới pass)
    reg_pct    = TRIGGER_PCT if reg_cond == "low" else (100 - TRIGGER_PCT)
    reg_thresh = float(np.percentile(reg_vals, reg_pct)) if reg_vals else 0.0

    trig_thresh = {}
    for t in trig_ind:
        vals = [r[t] for r in rows if np.isfinite(r.get(t, float("nan")))]
        if not vals:
            continue
        # trigger "low"  → signal khi giá trị thấp → threshold = p30
        # trigger "high" → signal khi giá trị cao  → threshold = p70
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
    Timeout 30s trên bước load data để tránh hang vô hạn nếu vnstock API chậm.
    """
    import concurrent.futures
    try:
        from vn_loader import load_vn_ohlcv
        # FIX: timeout 30s trên network call — tránh freeze toàn bộ scan nếu 1 symbol chậm
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(load_vn_ohlcv, symbol, 2000, 200)
            try:
                df = fut.result(timeout=30)
            except concurrent.futures.TimeoutError:
                logger.warning(f"[Scanner] {symbol} load TIMEOUT (>30s) — SKIP")
                return None
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        logger.warning(f"[Scanner] {symbol} load fail: {e}")
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
        "stoch_k":           f"Stoch oversold ({ind['stoch_k']:.1f})",
        "momentum_5d":       f"Momentum 5d ({ind['momentum_5d']:+.1f}%)",
        "volume_spike":      f"Volume spike ({ind['volume_spike']:+.1f}x)",
        "candle_body":       f"Nến thân lớn ({ind['candle_body']:.2f})",
        "consolidation":     f"Sideways ({ind.get('consolidation', 0)*100:.0f}%)",
        "vol_dry_up":        f"Vol kho ({ind.get('vol_dry_up', 0):+.2f}x)",
    }
    trigger_str = " + ".join(trigger_labels.get(t, t) for t in triggered)

    # Partial pass note
    partial_note = stats.get("partial_note", "") if stats.get("partial_pass") else ""
    wfe_inflate_note = "⚠️ WFE inflate (dùng OOS exp)" if stats.get("wfe_inflate") else ""

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
        "partial_note":  partial_note,
        "wfe_inflate_note": wfe_inflate_note,
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

    # Partial pass warning
    if sig.get("partial_note"):
        lines.append(f"*Lưu ý:* {sig['partial_note']}")
    if sig.get("wfe_inflate_note"):
        lines.append(f"*Lưu ý:* {sig['wfe_inflate_note']}")
    if cluster == "Mean Reversion":
        lines.append(f"*VNI ATR:* {vni_info['status']}")

    # Profit Factor
    pf  = stats.get("pf", 0)
    pf_str = f"{pf:.2f}" if pf else "?"

    # Sizing score = score S34 (OOS_exp × consist/100) — đã có trong SYMBOL_STATS
    # Fallback về exp nếu mã chưa có score field (mã S33 cũ)
    sizing_score = stats.get("score") or stats.get("exp", 0)

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
            f"  → Mua: {ps['qty']:,} cổ (~{ps['value']/1e6:.1f}M, "
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
                dual_tag = " (+MR)"
            elif sig["symbol"] in MOM_SYMBOLS:
                dual_tag = " (+MOM)"
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
    import time as _time
    t_start = _time.time()
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
            logger.debug(f"[Scanner] {sym} MR: no signal")

    for sym in MOM_SYMBOLS:
        sig = _scan_symbol(sym, "Momentum")
        if sig:
            mom_signals.append(sig)
            logger.info(f"[Scanner] {sym} MOM SIGNAL: {sig['trigger_str']}")
        else:
            mom_no_signal.append(sym)
            logger.debug(f"[Scanner] {sym} MOM: no signal")

    for sym in BREAKOUT_SYMBOLS:
        sig = _scan_symbol(sym, "Breakout")
        if sig:
            bo_signals.append(sig)
            logger.info(f"[Scanner] {sym} BO SIGNAL: {sig['trigger_str']}")
        else:
            bo_no_signal.append(sym)
            logger.debug(f"[Scanner] {sym} BO: no signal")

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
    elapsed = round(_time.time() - t_start, 1)
    logger.info(f"[Scanner] Morning scan done: {total} signals "
                f"(MR={len(mr_signals)}, MOM={len(mom_signals)}, BO={len(bo_signals)}) "
                f"in {elapsed}s")

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
