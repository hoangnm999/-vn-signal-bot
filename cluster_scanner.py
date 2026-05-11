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
# ── MR2 cluster — Mean Reversion với Smooth Stoch (SMA3) ─────────────────────
# S37: cluster mới, independent universe (không overlap MR)
# Universe = W2_ONLY: pass WF với smooth stoch nhưng KHÔNG pass baseline raw stoch
# Rule vớt n<15: phải có consist=100% VÀ WFE≥2.0
MR2_SYMBOLS = [
    # S37 vibe filter results:
    # BONUS  : DPR (MultiFactor)
    # NO_FILTER: VPI, CTR, TDP, VSC, HCM, CTG — không đủ engine data, chờ live
    # LOAI   : MIG (exp thấp + Candlestick+CrossMarket loại)
    #           PET (exp thấp + Volatility+MultiFactor loại)
    #           TRC (exp thấp + CrossMarket loại)
    # GIỮ với NO_FILTER: CTS và DXG — exp tốt, vibe engines không đủ conclusive
    #   CTS: SMC+CrossMarket loại nhưng tot margin nhỏ, OOS_exp=+2.13% vẫn pass WF
    #   DXG: CrossMarket loại do 1 fold outlier (-20%), OOS_exp=+4.14% rất tốt
    # ── Active watchlist ──────────────────────────────────────────────────────
    "VPI",   # [B] OOS=+3.06% WFE=5.37 consist=83%  n=26 ✅ NO_FILTER
    "CTR",   # [B] OOS=+3.13% WFE=2.65 consist=60%  n=19    NO_FILTER
    "DPR",   # [B] OOS=+3.50% WFE=1.64 consist=60%  n=18    BONUS:MultiFactor
    "DXG",   # [B] OOS=+4.14% WFE=1.02 consist=60%  n=19    NO_FILTER (CrossMarket 1 fold outlier)
    "TDP",   # [B] OOS=+2.57% WFE=1.40 consist=60%  n=21    NO_FILTER
    "VSC",   # [B] OOS=+2.17% WFE=2.56 consist=60%  n=19    NO_FILTER
    "CTS",   # [B] OOS=+2.13% WFE=1.12 consist=60%  n=17    NO_FILTER (engines inconclusive)
    "HCM",   # [A] OOS=+4.40% WFE=2.23 consist=100% n=14 ⚠️ NO_FILTER (n_cap=True)
    "CTG",   # [B] OOS=+3.49% WFE=6.05 consist=100% n=14 ⚠️ NO_FILTER (n_cap=True)
    # ── Loại theo vibe filter S37 ─────────────────────────────────────────────
    # MIG: exp thấp (+1.85%) + Candlestick=LOAI + CrossMarket=LOAI
    # PET: exp thấp (+3.95% nhưng consist=60%) + Volatility=LOAI + MultiFactor=LOAI
    # TRC: CrossMarket=LOAI (n=12, n_cap đã vớt nhưng engine loại)
    # ── Loại từ trước (score/WFE/n filter) ───────────────────────────────────
    # HAH, NKG, GMD, MSB: WFE < 0.8
    # EVF: n=14 + consist=67%
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

# Breakout cluster — S37: redesign với BB Squeeze (W3) + Consol V2 (S6)
# Filter: score ≥ 1.5, WFE ≥ 0.8, consist ≥ 80%, Option B
# BB hẹp (squeeze) → vol dry-up → breakout (khác baseline BB rộng)
BREAKOUT_SYMBOLS = [
    # ── Tier A (score ≥ 4.0) ──────────────────────────────────────────────────
    "CTG",   # [A] score=4.75 OOS=+4.75% WFE=2.256 consist=100% n=33 ✅
    "LPB",   # [A] score=4.36 OOS=+4.36% WFE=0.901 consist=100% n=28 ⚠️WFE (cũng MOM A)
    # ── Tier B (score 1.5–3.99) ───────────────────────────────────────────────
    "CTS",   # [B] score=3.66 OOS=+4.41% WFE=1.726 consist=83%  n=31 ✅ HARD:SMC
    "DXG",   # [B] score=3.49 OOS=+3.49% WFE=2.573 consist=100% n=30 ✅
    "VTP",   # [B] score=3.46 OOS=+4.17% WFE=2.408 consist=83%  n=31 ✅ HARD:MultiFactor
    "BMP",   # [B] score=3.35 OOS=+4.19% WFE=2.346 consist=80%  n=21 ✅ (cũng MR A)
    "TCB",   # [B] score=3.28 OOS=+3.28% WFE=2.694 consist=100% n=33 ✅ BONUS:MultiFactor
    "ACB",   # [B] score=2.58 OOS=+2.58% WFE=3.430 consist=100% n=32 ✅
    "MBB",   # [B] score=2.21 OOS=+2.21% WFE=1.903 consist=100% n=35 ✅ (SMC NULL filter)
    "BID",   # [B] score=2.20 OOS=+2.20% WFE=0.968 consist=100% n=38 ⚠️WFE
    "FPT",   # [B] score=1.77 OOS=+2.13% WFE=0.878 consist=83%  n=29 ⚠️WFE
    "VHC",   # [B] score=1.58 OOS=+1.91% WFE=1.702 consist=83%  n=30 ✅
    "CTD",   # [B] score=2.76 OOS=+3.32% WFE=0.844 consist=83%  n=32 ⚠️WFE
    # ── Loại theo vibe filter S37 ─────────────────────────────────────────────
    # BSI : CrossMarket=LOAI
    # ELC : SMC=LOAI + CrossMarket=LOAI
    # NAB : TechBasic=LOAI + CrossMarket=LOAI (vẫn giữ trong MR cluster)
    # TRC : CrossMarket=LOAI + Chanlun=LOAI
    # TV2 : SMC=LOAI
    # MBB : giữ nhưng không có HARD (SMC NULL filter)
]

FWD_DAYS = {"Mean Reversion": 20, "Mean Reversion 2": 20, "Momentum": 10, "Breakout": 15}

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
    # S37: MR2 — Mean Reversion với Smooth Stoch (SMA3 của raw %K)
    # Universe riêng (MR2_SYMBOLS), không overlap MR
    # stoch_k_smooth = SMA3(raw %K) — bắt oversold muộn hơn, phù hợp mã bounce chậm
    "Mean Reversion 2": {
        "regime_indicator":  "price_vs_sma50",
        "regime_condition":  "low",
        "trigger_indicators":["stoch_k_smooth", "volume_spike", "momentum_5d"],
        "trigger_direction": {"stoch_k_smooth": "low", "volume_spike": "high",
                              "momentum_5d": "high"},
        "description": "MR với smooth stoch (SMA3) — bounce chậm hơn MR baseline",
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
        # S37 W3: 'low' = BB hẹp (squeeze) thay vì 'high' (BB rộng baseline)
        # Squeeze setup: BB hẹp → vol dry-up → consolidation → chờ bứt phá
        "regime_condition":  "low",
        "trigger_indicators":["consolidation", "vol_dry_up", "momentum_5d"],
        "trigger_direction": {"consolidation": "high", "vol_dry_up": "high",
                              "momentum_5d": "high"},
        # S37 S6: consolidation dùng % ngày sideways vs mean(window) (consol_v2)
        # nhất quán với _consol_val trong _compute_indicators
        "description": "BB squeeze + vol dry-up + consolidation → bứt phá",
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
               "score": 1.61, "consist": 75,  "partial_pass": True,  "partial_note": "⚠️ Chỉ pass Option B",
               "wfe_inflate": True},
               # NOTE S37 W8: n=14 dưới ngưỡng per-symbol n≥15 — monitor khi có thêm folds
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
    "LPB_MOM": {"wr": 74, "exp": 4.45, "wfe": 1.85, "n": 22, "pf": 2.02, "cluster": "Momentum",
             "score": 4.45, "consist": 100, "partial_pass": False},
    "VTP_MOM": {"wr": 60, "exp": 4.17, "wfe": 2.43, "n": 29, "pf": 1.81, "cluster": "Momentum",
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
    "CTS_MOM": {"wr": 56, "exp": 2.55, "wfe": 2.66, "n": 23, "pf": 2.48, "cluster": "Momentum",
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
    "MBB_MOM": {"wr": 47, "exp": 2.62, "wfe": 9.78, "n": 20, "pf": 1.74, "cluster": "Momentum",
             "score": 1.97, "consist": 75,  "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option B",
             "wfe_inflate": True},
    "CTG_MOM": {"wr": 50, "exp": 1.23, "wfe": 1.11, "n": 25, "pf": 1.96, "cluster": "Momentum",
             "score": 1.02, "consist": 83,  "partial_pass": True, "partial_note": "⚠️ Chỉ pass Option B · score biên"},
    # ── Breakout — S37 W3+S6 (BB Squeeze + Consol V2) ───────────────────────────
    # Score = OOS_exp × consist/100 | Filter: score≥1.5, WFE≥0.8, consist≥80%
    # ⚠️WFE = WFE 0.80–0.99 (IS > OOS, monitor live)
    # Tier A
    "CTG_BO":  {"wr": 71, "exp": 4.75, "wfe": 2.256, "n": 33, "pf": 2.11, "cluster": "Breakout",
               "score": 4.75, "consist": 100},
    "LPB_BO": {"wr": 68, "exp": 4.36, "wfe": 0.901, "n": 28, "pf": 2.20, "cluster": "Breakout",
               "score": 4.36, "consist": 100,
               "partial_note": "⚠️ WFE borderline 0.90 (cũng MOM Tier A)"},
    # Tier B — WFE ≥ 2.0
    "TRC_BO": {"wr": 75, "exp": 3.68, "wfe": 2.453, "n": 32, "pf": 1.85, "cluster": "Breakout",
               "score": 3.68, "consist": 100},
    "CTS_BO": {"wr": 65, "exp": 4.41, "wfe": 1.726, "n": 31, "pf": 2.83, "cluster": "Breakout",
               "score": 3.66, "consist": 83},
    "BSI_BO": {"wr": 64, "exp": 4.30, "wfe": 2.750, "n": 28, "pf": 3.62, "cluster": "Breakout",
               "score": 3.57, "consist": 83},
    "DXG_BO": {"wr": 68, "exp": 3.49, "wfe": 2.573, "n": 30, "pf": 2.19, "cluster": "Breakout",
               "score": 3.49, "consist": 100},
    "VTP_BO": {"wr": 65, "exp": 4.17, "wfe": 2.408, "n": 31, "pf": 1.37, "cluster": "Breakout",
               "score": 3.46, "consist": 83,
               "partial_note": "ℹ️ Cũng MOM Tier A"},
    "BMP_BO": {"wr": 62, "exp": 4.19, "wfe": 2.346, "n": 21, "pf": 1.57, "cluster": "Breakout",
               "score": 3.35, "consist": 80,
               "partial_note": "ℹ️ Cũng MR Tier A"},
    "TCB":    {"wr": 68, "exp": 3.28, "wfe": 2.694, "n": 33, "pf": 1.60, "cluster": "Breakout",
               "score": 3.28, "consist": 100},
    "ACB":    {"wr": 68, "exp": 2.58, "wfe": 3.430, "n": 32, "pf": 2.13, "cluster": "Breakout",
               "score": 2.58, "consist": 100},
    "TV2":    {"wr": 63, "exp": 3.00, "wfe": 2.525, "n": 30, "pf": 1.47, "cluster": "Breakout",
               "score": 2.49, "consist": 83},
    "MBB_BO": {"wr": 60, "exp": 2.21, "wfe": 1.903, "n": 35, "pf": 1.59, "cluster": "Breakout",
               "score": 2.21, "consist": 100,
               "partial_note": "ℹ️ Cũng MOM Tier B"},
    "ELC":    {"wr": 62, "exp": 2.52, "wfe": 1.376, "n": 37, "pf": 1.99, "cluster": "Breakout",
               "score": 2.09, "consist": 83},
    "VHC":    {"wr": 60, "exp": 1.91, "wfe": 1.702, "n": 30, "pf": 1.34, "cluster": "Breakout",
               "score": 1.58, "consist": 83},
    # Tier B — WFE ⚠️ (0.80–0.99)
    "NAB_BO": {"wr": 63, "exp": 3.60, "wfe": 0.824, "n": 35, "pf": 2.43, "cluster": "Breakout",
               "score": 2.99, "consist": 83,
               "partial_note": "⚠️ WFE 0.82 (cũng MR Tier A)"},
    "CTD":    {"wr": 64, "exp": 3.32, "wfe": 0.844, "n": 32, "pf": 2.15, "cluster": "Breakout",
               "score": 2.76, "consist": 83,
               "partial_note": "⚠️ WFE borderline 0.84"},
    "BID":    {"wr": 65, "exp": 2.20, "wfe": 0.968, "n": 38, "pf": 2.95, "cluster": "Breakout",
               "score": 2.20, "consist": 100,
               "partial_note": "⚠️ WFE borderline 0.97"},
    "FPT":    {"wr": 62, "exp": 2.13, "wfe": 0.878, "n": 29, "pf": 2.33, "cluster": "Breakout",
               "score": 1.77, "consist": 83,
               "partial_note": "⚠️ WFE borderline 0.88"},
    # ── MR2 cluster — S37: Mean Reversion với Smooth Stoch ───────────────────
    # Vibe filter S37: LOAI → MIG, PET, TRC (xóa khỏi watchlist)
    # CTS, DXG: engines loại nhưng margin/outlier → giữ NO_FILTER per quyết định S37
    # n_cap=True: position sizing × 0.7 cho mã OOS_n < 15
    "VPI":  {"wr": 58, "exp": 3.06, "wfe": 5.37, "n": 26, "pf": 1.85, "cluster": "Mean Reversion 2",
             "score": 2.55, "consist": 83},
    "CTR":  {"wr": 63, "exp": 3.13, "wfe": 2.65, "n": 19, "pf": 1.96, "cluster": "Mean Reversion 2",
             "score": 1.88, "consist": 60},
    "DPR":  {"wr": 61, "exp": 3.50, "wfe": 1.64, "n": 18, "pf": 1.82, "cluster": "Mean Reversion 2",
             "score": 2.10, "consist": 60},
    "DXG_MR2": {"wr": 58, "exp": 4.14, "wfe": 1.02, "n": 19, "pf": 1.74, "cluster": "Mean Reversion 2",
             "score": 2.48, "consist": 60,
             "partial_note": "ℹ️ MR2 (CrossMarket 1 fold outlier, NO_FILTER)"},
    "TDP":  {"wr": 57, "exp": 2.57, "wfe": 1.40, "n": 21, "pf": 1.63, "cluster": "Mean Reversion 2",
             "score": 1.54, "consist": 60},
    "VSC":  {"wr": 58, "exp": 2.17, "wfe": 2.56, "n": 19, "pf": 1.65, "cluster": "Mean Reversion 2",
             "score": 1.30, "consist": 60},
    "CTS_MR2": {"wr": 59, "exp": 2.13, "wfe": 1.12, "n": 17, "pf": 1.58, "cluster": "Mean Reversion 2",
             "score": 1.28, "consist": 60,
             "partial_note": "ℹ️ MR2 (SMC+CrossMarket margin nhỏ, NO_FILTER)"},
    "HCM":  {"wr": 71, "exp": 4.40, "wfe": 2.23, "n": 14, "pf": 2.34, "cluster": "Mean Reversion 2",
             "score": 4.40, "consist": 100, "n_cap": True,
             "partial_note": "⚠️ n=14 (<15) — sizing giảm 30%"},
    "CTG_MR2": {"wr": 71, "exp": 3.49, "wfe": 6.05, "n": 14, "pf": 2.20, "cluster": "Mean Reversion 2",
             "score": 3.49, "consist": 100, "n_cap": True,
             "partial_note": "⚠️ n=14 (<15) — sizing giảm 30%"},
}

# SL từ MAE p25 analysis
SL_CONFIG = {
    "Mean Reversion":   -13.5,  # p25 MAE
    "Mean Reversion 2": -13.5,  # S37: dùng cùng SL với MR (chưa có MAE riêng)
    "Momentum":         -6.4,   # p25 MAE
    "Breakout":         -6.4,   # dùng MOM SL (quyết định S31, MAE riêng TBD)
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
        # S38: SMC/NAB và SMC/BMP đã bỏ HARD — cov=100% → NULL filter, không tác dụng
        # BONUS: exp_filt > exp_base, decay <= 25% — per engine/symbol pair
        "NAB":    {"hard": [],              "bonus": ["Volatility", "CrossMarket", "MultiFactor"]},  # S38: SMC cov=100% → NULL filter → bỏ HARD
        "BMP":    {"hard": [],              "bonus": []},                                              # S38: SMC cov=100% → NULL filter → bỏ HARD
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
        # VND: removed S37 — đã loại khỏi MOM_SYMBOLS (fail cả 2 options)
        # HAG: removed S37 — đã loại khỏi MOM_SYMBOLS
        "VIX": {"hard": [],  "bonus": []},
        "CTS": {"hard": [],  "bonus": []},
        "VDS": {"hard": [],  "bonus": []},
    },
    "Breakout": {
        # S37 W3+S6 watchlist — 13 mã còn lại sau vibe filter
        # Loại: BSI(CrossMarket), ELC(SMC+CrossMarket), NAB(TechBasic+CrossMarket),
        #        TRC(CrossMarket+Chanlun), TV2(SMC)
        # MBB: SMC cov=100% → NULL filter → LOAI theo precedent S35
        #
        # HARD_FILTER: signal chỉ fire khi engine đồng ý
        "CTS": {"hard": ["SMC"],         "bonus": []},
        "VTP": {"hard": ["MultiFactor"],  "bonus": []},   # conflict SMC=LOAI, MultiFactor=HARD → HARD wins
        "MBB": {"hard": [],              "bonus": []},    # SMC NULL filter (cov=100%) → NO_FILTER
        # BONUS: engine đồng ý thì tăng điểm score
        "TCB": {"hard": [],              "bonus": ["MultiFactor"]},
        # NO_FILTER: không có engine đủ data (giữ nguyên, chờ live data)
        "CTG": {"hard": [], "bonus": []},
        "LPB": {"hard": [], "bonus": []},
        "DXG": {"hard": [], "bonus": []},
        "BMP": {"hard": [], "bonus": []},   # tất cả engines INSUF
        "ACB": {"hard": [], "bonus": []},
        "CTD": {"hard": [], "bonus": []},
        "BID": {"hard": [], "bonus": []},
        "FPT": {"hard": [], "bonus": []},
        "VHC": {"hard": [], "bonus": []},
    },
    # S37: MR2 cluster — vibe filter kết quả S37
    # LOAI: MIG, PET, TRC (đã xóa khỏi MR2_SYMBOLS)
    # LOAI: CTS (SMC+CrossMarket) và DXG (CrossMarket) → giữ với NO_FILTER per quyết định S37
    "Mean Reversion 2": {
        "VPI": {"hard": [], "bonus": []},
        "CTR": {"hard": [], "bonus": []},
        "DPR": {"hard": [], "bonus": ["MultiFactor"]},
        "DXG": {"hard": [], "bonus": ["MultiFactor"]},  # S38: +2.16% (5.30→7.46%), cov 64%
        "TDP": {"hard": [], "bonus": []},
        "VSC": {"hard": [], "bonus": []},
        "CTS": {"hard": [], "bonus": []},   # SMC+CrossMarket loại nhưng margin nhỏ, giữ NO_FILTER
        "HCM": {"hard": [], "bonus": []},
        "CTG": {"hard": [], "bonus": []},
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
MEDIAN_SCORE   = 2.50   # FIX S37 W6: median thực tế từ toàn bộ 50 mã trong SYMBOL_STATS
                        # (tính lại: sorted scores → median = 2.50, thay vì 3.0 cũ)

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
    # S37 MR2: smooth stoch = SMA3(raw %K) — dùng cho MR2 cluster
    stoch_smooth = _sma(stoch, 3)

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
        # S37 S4: momentum_3d đã xóa — MOM_GUARD_3D experiment chưa validated (S36)
        #         Nếu cần, sẽ thêm lại sau khi có đủ _compute_thresholds + backtest
        "volume_spike":   float((vol[i] / (vs20v + 1e-9)) - 1.0),
        "stoch_k":        float(stoch[i]),
        # S37 MR2: smooth stoch (SMA3) — nhất quán với _compute_thresholds
        "stoch_k_smooth": float(stoch_smooth[i]) if np.isfinite(stoch_smooth[i]) else float(stoch[i]),
        "candle_body":    float(np.clip(abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
        "atr_ratio":      float(atr_v / (px + 1e-9) * 100),
        "atr":            float(atr_v),   # FIX S37 C4: ATR tuyệt đối (VND) — dùng cho trailing stop
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
    # S37 MR2: smooth stoch series — nhất quán với _compute_indicators
    stoch_smooth = _sma(stoch, 3)

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
            # S37 MR2: smooth stoch — nhất quán với _compute_indicators
            "stoch_k_smooth": float(stoch_smooth[i]) if np.isfinite(stoch_smooth[i]) else float(stoch[i]),
            "candle_body":    float(np.clip(abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
            # S37 S3: candle_bull + volume_spike_bull đã xóa — revert về MOM logic gốc (S36)
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
    except SystemExit:
        logger.warning("[VNI] vnstock rate limit (sys.exit) — dùng fallback")
        return {"atr_ratio": 0, "threshold": 0.863, "is_high": None,
                "last_date": "?", "status": "⚠️ Rate limit — dùng fallback VNI"}
    except Exception as e:
        logger.warning(f"[VNI] Error: {e}")
        return {"atr_ratio": 0, "threshold": 0.863, "is_high": None,
                "last_date": "?", "status": "⚠️ Không load được VNI"}


# ── Vibe Filter Application ───────────────────────────────────────────────────

def _run_vibe_engines(engines: list[str], symbol: str,
                      df: pd.DataFrame) -> dict[str, int]:
    """
    Chạy các vibe engines được chỉ định, trả về {engine_name: signal}.
    signal: +1 bull / -1 bear / 0 neutral.
    Lỗi từng engine → 0 (không crash toàn bộ filter).
    """
    try:
        from vibe_skills_1_ import _ENGINES, _prep
    except ImportError:
        logger.warning("[VibeFilter] Không import được vibe_skills_1_ — skip engines")
        return {e: 0 for e in engines}

    try:
        df_prep  = _prep(df)
        data_map = {symbol: df_prep}
    except Exception as ex:
        logger.warning(f"[VibeFilter] _prep lỗi cho {symbol}: {ex}")
        return {e: 0 for e in engines}

    results = {}
    for name in engines:
        engine = _ENGINES.get(name)
        if engine is None:
            logger.warning(f"[VibeFilter] Engine '{name}' không tìm thấy trong _ENGINES")
            results[name] = 0
            continue
        try:
            sigs, _ = engine.generate(data_map)
            results[name] = int(sigs.get(symbol, 0))
            logger.debug(f"[VibeFilter] {symbol}/{name}: signal={results[name]}")
        except Exception as ex:
            logger.warning(f"[VibeFilter] {symbol}/{name} lỗi: {ex}")
            results[name] = 0

    return results


def _apply_vibe_filter(symbol: str, cluster: str,
                       df: pd.DataFrame | None = None) -> dict:
    """
    Áp dụng VIBE_FILTER_CONFIG cho symbol/cluster.

    Logic:
      HARD filter  — TẤT CẢ hard engines phải đồng ý (+1).
                     Nếu bất kỳ engine nào trả -1 → block_signal = True.
                     Engine trả 0 (neutral) → không block (benefit of doubt).
      BONUS filter — Chạy để ghi note lên Telegram; không block signal.
                     +1 → "✅ ENGINE đồng ý" / -1 → "❌ ENGINE phủ nhận" / 0 → neutral

    Args:
        symbol  : mã cổ phiếu
        cluster : "Mean Reversion" | "Momentum" | "Breakout"
        df      : DataFrame OHLCV (từ load_vn_ohlcv) — cần có để chạy engines.
                  Nếu None → không chạy được, hard_status = "NO_DATA", không block.

    Trả về dict:
        hard_engines   : list[str]
        hard_status    : "PASS" | "FAIL" | "NO_DATA" | "N/A"
        hard_signals   : {engine: int}
        bonus_engines  : list[str]
        bonus_signals  : {engine: int}
        note           : str — hiển thị trên Telegram
        block_signal   : bool
    """
    vibe_cfg      = VIBE_FILTER_CONFIG.get(cluster, {}).get(symbol, {})
    hard_engines  = vibe_cfg.get("hard", [])
    bonus_engines = vibe_cfg.get("bonus", [])

    # Trường hợp không có filter nào → trả về nhanh
    if not hard_engines and not bonus_engines:
        return {
            "hard_engines":  [],
            "hard_status":   "N/A",
            "hard_signals":  {},
            "bonus_engines": [],
            "bonus_signals": {},
            "note":          "",
            "block_signal":  False,
        }

    # Không có df → không thể chạy engine
    if df is None:
        logger.warning(f"[VibeFilter] {symbol}/{cluster}: df=None, bỏ qua vibe filter")
        hard_note = (f"🔶 HARD filter ({', '.join(hard_engines)}) không chạy được (no data)"
                     if hard_engines else "")
        return {
            "hard_engines":  hard_engines,
            "hard_status":   "NO_DATA",
            "hard_signals":  {},
            "bonus_engines": bonus_engines,
            "bonus_signals": {},
            "note":          hard_note,
            "block_signal":  False,   # benefit of doubt
        }

    # ── Chạy HARD engines ─────────────────────────────────────────────────────
    hard_signals  = {}
    hard_status   = "N/A"
    block_signal  = False

    if hard_engines:
        hard_signals = _run_vibe_engines(hard_engines, symbol, df)
        # Block nếu bất kỳ engine nào trả -1 (bear)
        failed = [e for e, s in hard_signals.items() if s == -1]
        if failed:
            block_signal = True
            hard_status  = "FAIL"
            logger.info(
                f"[VibeFilter] {symbol}/{cluster}: HARD FAIL — "
                f"engines bearish: {failed}"
            )
        else:
            hard_status = "PASS"
            logger.debug(
                f"[VibeFilter] {symbol}/{cluster}: HARD PASS — "
                f"signals: {hard_signals}"
            )

    # ── Chạy BONUS engines (chỉ nếu signal chưa bị block) ────────────────────
    bonus_signals = {}
    if bonus_engines and not block_signal:
        bonus_signals = _run_vibe_engines(bonus_engines, symbol, df)

    # ── Build Telegram note ───────────────────────────────────────────────────
    note_parts = []

    if hard_engines:
        if hard_status == "PASS":
            h_detail = " | ".join(
                f"{'✅' if s == 1 else '⬜'} {e}"
                for e, s in hard_signals.items()
            )
            note_parts.append(f"🛡️ HARD ({h_detail}): PASS")
        elif hard_status == "FAIL":
            h_detail = " | ".join(
                f"{'❌' if s == -1 else '✅' if s == 1 else '⬜'} {e}"
                for e, s in hard_signals.items()
            )
            note_parts.append(f"🛡️ HARD ({h_detail}): ❌ BLOCK")
        else:
            note_parts.append(f"🔶 HARD ({', '.join(hard_engines)}): NO DATA")

    if bonus_engines and not block_signal:
        b_parts = []
        for e, s in bonus_signals.items():
            icon = "✅" if s == 1 else "❌" if s == -1 else "⬜"
            b_parts.append(f"{icon} {e}")
        note_parts.append(f"💡 Bonus: {' | '.join(b_parts)}")

    note = "\n".join(note_parts)

    return {
        "hard_engines":  hard_engines,
        "hard_status":   hard_status,
        "hard_signals":  hard_signals,
        "bonus_engines": bonus_engines,
        "bonus_signals": bonus_signals,
        "note":          note,
        "block_signal":  block_signal,
    }


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
    except SystemExit as e:
        # FIX S37: vnstock gọi sys.exit() khi rate limit — KHÔNG được crash server
        # Bắt SystemExit, log warning, chờ rồi skip symbol này
        logger.warning(f"[Scanner] {symbol} vnstock rate limit (sys.exit) — chờ 65s rồi skip")
        import time as _time_rl
        _time_rl.sleep(65)
        return None
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

    # Tầng 3: Vibe Filter — HARD block nếu engine bearish, BONUS ghi note
    vibe = _apply_vibe_filter(symbol, cluster, df=df)
    if vibe["block_signal"]:
        logger.info(f"[Scanner] {symbol} {cluster}: blocked by HARD vibe filter "
                    f"({vibe['hard_signals']})")
        return None

    # Signal confirmed
    # Lookup SYMBOL_STATS: thử key có suffix cluster trước (FRT_MR, BMP_MR...)
    # rồi fallback về plain key (FRT, BMP...)
    _cluster_suffix = {"Mean Reversion": "MR", "Mean Reversion 2": "MR2",
                       "Momentum": "MOM", "Breakout": "BO"}
    _suffix = _cluster_suffix.get(cluster, "")
    stats = (SYMBOL_STATS.get(f"{symbol}_{_suffix}")
             or SYMBOL_STATS.get(symbol)
             or {})
    fwd       = FWD_DAYS[cluster]
    entry     = ind["close"]
    sl_pct    = SL_CONFIG[cluster]
    sl_price  = round(entry * (1 + sl_pct / 100), 1)
    tp_date   = (date.today() + timedelta(days=int(fwd * 1.4))).strftime("%d/%m/%Y")

    # Regime detail string
    if cluster in ("Mean Reversion", "Mean Reversion 2"):
        mr2_tag = " [Smooth Stoch]" if cluster == "Mean Reversion 2" else ""
        regime_detail = (f"Giá dưới SMA50 ({val:+.1f}%){mr2_tag} | "
                         f"SMA50={ind['sma50']:.1f}")
    elif cluster == "Breakout":
        consol_pct = ind.get("consolidation", 0) * 100
        regime_detail = (f"BB hẹp/squeeze (width={val:.1f}%) | "
                         f"Sideways {consol_pct:.0f}% ngày | "
                         f"Vol dry-up={ind.get('vol_dry_up', 0):+.2f}x")
    else:
        regime_detail = (f"EMA12 > EMA26 ({val:+.2f}%) | "
                         f"EMA12={ind['ema12']:.1f} EMA26={ind['ema26']:.1f}")

    # Trigger detail
    trigger_labels = {
        "stoch_k":           f"Stoch oversold ({ind['stoch_k']:.1f})",
        # S37 MR2: smooth stoch label
        "stoch_k_smooth":    f"Smooth Stoch oversold ({ind.get('stoch_k_smooth', ind['stoch_k']):.1f})",
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
        "vibe_note":     vibe["note"],       # FIX S37 C2: HARD/BONUS filter status
        "vibe_detail":   vibe,               # full vibe result cho debug
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
    elif cluster == "Mean Reversion 2":
        emoji, cluster_short = "🔄", "MR2"
    elif cluster == "Momentum":
        emoji, cluster_short = "🚀", "MOM"
    else:
        emoji, cluster_short = "💥", "BO"

    # WFE badge — FIX S37 S5: ngưỡng 1.5/2.0/3.0 thay vì 0.5/0.7/1.0 cũ
    # WFE < 1.0: IS tốt hơn OOS → overfit, không badge
    # WFE ≥ 1.5: OOS đáng tin, WFE ≥ 2.0: tốt, WFE ≥ 3.0: xuất sắc
    wfe = stats.get("wfe", 0)
    wfe_badge = ("⭐⭐⭐" if wfe >= 3.0 else
                 "⭐⭐"  if wfe >= 2.0 else
                 "⭐"   if wfe >= 1.5 else "")

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
    if sig.get("vibe_note"):                          # FIX S37 C2
        lines.append(f"*Vibe Filter:* {sig['vibe_note']}")
    if cluster in ("Mean Reversion", "Mean Reversion 2"):
        lines.append(f"*VNI ATR:* {vni_info['status']}")

    # Profit Factor
    pf  = stats.get("pf", 0)
    pf_str = f"{pf:.2f}" if pf else "?"

    # Sizing score = score S34 (OOS_exp × consist/100) — đã có trong SYMBOL_STATS
    # Fallback về exp nếu mã chưa có score field (mã S33 cũ)
    sizing_score = stats.get("score") or stats.get("exp", 0)

    # S37 MR2: n_cap — giảm 30% sizing cho mã OOS_n < 15
    if stats.get("n_cap"):
        sizing_score = sizing_score * 0.7

    lines += [
        f"",
        f"*📊 Walk Forward OOS (2022→nay):*",
        f"  WR={stats.get('wr', '?')}% | Exp={stats.get('exp', 0):+.1f}% | "
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
    mr2_signals: list[dict] | None = None,
    mr2_no_signal: list[str] | None = None,
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

    bo_signals    = bo_signals    or []
    bo_no_signal  = bo_no_signal  or []
    mr2_signals   = mr2_signals   or []
    mr2_no_signal = mr2_no_signal or []
    total_signals = len(mr_signals) + len(mr2_signals) + len(mom_signals) + len(bo_signals)

    if total_signals == 0:
        current += (
            f"\n✅ Không có signal hôm nay\n\n"
            f"*Mean Reversion ({len(MR_SYMBOLS)} mã):* Không đủ điều kiện\n"
            f"*Mean Reversion 2 ({len(MR2_SYMBOLS)} mã):* Không đủ điều kiện\n"
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

    # MR2 signals
    if mr2_signals:
        current += f"\n━━ 🔄 MEAN REVERSION 2 / Smooth Stoch (FWD=20d) ━━\n"
        for sig in mr2_signals:
            sig_text = "\n" + _format_signal(sig, vni_info) + "\n"
            if len(current) + len(sig_text) > 3800:
                messages.append(current)
                current = sig_text
            else:
                current += sig_text
    else:
        current += f"\n\n🔄 *MR2:* Không có signal"

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
            dual_tag = ""
            if sig["symbol"] in MR_SYMBOLS:
                dual_tag = " (+MR)"
            elif sig["symbol"] in MR2_SYMBOLS:
                dual_tag = " (+MR2)"
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
    total_symbols = len(MR_SYMBOLS) + len(MR2_SYMBOLS) + len(MOM_SYMBOLS) + len(BREAKOUT_SYMBOLS)
    footer = (
        f"\n━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_Scan: {total_symbols} mã (MR={len(MR_SYMBOLS)}, MR2={len(MR2_SYMBOLS)}, "
        f"MOM={len(MOM_SYMBOLS)}, BO={len(BREAKOUT_SYMBOLS)})_\n"
        f"*VNI ATR:* {vni_info['atr_ratio']:.3f} "
        f"({'cao ✅' if vni_info['is_high'] else 'thấp ⚠️'} "
        f"vs threshold {vni_info['threshold']:.3f})\n"
    )
    if mr_no_signal:
        footer += f"MR không signal: {' '.join(mr_no_signal)}\n"
    if mr2_no_signal:
        footer += f"MR2 không signal: {' '.join(mr2_no_signal)}\n"
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
    mr2_signals, mr2_no_signal = [], []

    for sym in MR_SYMBOLS:
        try:
            sig = _scan_symbol(sym, "Mean Reversion")
        except SystemExit:
            logger.warning(f"[Scanner] {sym} MR: rate limit crash — skip")
            sig = None
        if sig:
            mr_signals.append(sig)
            logger.info(f"[Scanner] {sym} MR SIGNAL: {sig['trigger_str']}")
        else:
            mr_no_signal.append(sym)
            logger.debug(f"[Scanner] {sym} MR: no signal")

    for sym in MR2_SYMBOLS:
        sig = _scan_symbol(sym, "Mean Reversion 2")
        if sig:
            mr2_signals.append(sig)
            logger.info(f"[Scanner] {sym} MR2 SIGNAL: {sig['trigger_str']}")
        else:
            mr2_no_signal.append(sym)
            logger.debug(f"[Scanner] {sym} MR2: no signal")

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
    for sig in mr_signals + mr2_signals + mom_signals + bo_signals:
        cluster_short = {"Mean Reversion": "MR", "Mean Reversion 2": "MR2",
                         "Momentum": "MOM", "Breakout": "BO"}.get(sig["cluster"], "?")
        _morning_signals[sig["symbol"]] = {
            "entry":         sig["entry_price"],
            "cluster":       sig["cluster"],
            "cluster_short": cluster_short,
            "scan_time":     sig["scan_time"],
            "last_date":     sig["last_date"],
        }

    # Ghi vao cluster_journal (S31)
    _journal_log_signals(mr_signals + mr2_signals + mom_signals + bo_signals, vni_info)

    total = len(mr_signals) + len(mr2_signals) + len(mom_signals) + len(bo_signals)
    elapsed = round(_time.time() - t_start, 1)
    logger.info(f"[Scanner] Morning scan done: {total} signals "
                f"(MR={len(mr_signals)}, MR2={len(mr2_signals)}, "
                f"MOM={len(mom_signals)}, BO={len(bo_signals)}) "
                f"in {elapsed}s")

    messages = _format_morning_scan(
        mr_signals, mom_signals,
        mr_no_signal, mom_no_signal,
        vni_info, "08:30",
        bo_signals=bo_signals, bo_no_signal=bo_no_signal,
        mr2_signals=mr2_signals, mr2_no_signal=mr2_no_signal,
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
        except SystemExit:
            logger.warning(f"[Scanner] Update {sym}: vnstock rate limit — skip")
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

    # S37: MR2 cluster — scan afternoon nhất quán với morning
    for sym in MR2_SYMBOLS:
        if sym in morning_syms:
            continue
        sig = _scan_symbol(sym, "Mean Reversion 2")
        if sig:
            new_signals.append(sig)

    for sym in MOM_SYMBOLS:
        if sym in morning_syms:
            continue
        sig = _scan_symbol(sym, "Momentum")
        if sig:
            new_signals.append(sig)

    # FIX S37 S2: thêm BO scan — nhất quán với morning scan
    for sym in BREAKOUT_SYMBOLS:
        if sym in morning_syms:
            continue
        sig = _scan_symbol(sym, "Breakout")
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
