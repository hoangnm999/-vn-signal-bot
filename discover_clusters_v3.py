"""
discover_clusters_v3.py — Multi-Timeframe Pattern Discovery
VN Trader Bot V6 — Session 31

Cải tiến so với V2:
  1. Multi-timeframe lookback:
     - Fast  indicators: T-1, T-3, T-5   (trigger ngay trước breakout)
     - Medium indicators: T-5, T-10, T-15 (setup đang hình thành)
     - Slow  indicators: T-10, T-20, T-30 (tích lũy dài hạn)
  2. OBV dùng SMA40 thay SMA10 (cần thời gian tích lũy)
  3. Higher lows dùng swing-based (30 ngày) thay rolling (10 ngày)
  4. 2 threshold: SHORT +8%/10d và LONG +15%/20d
  5. Cluster hợp lệ cần ≥1 Slow + ≥1 Fast indicator dominant
     (context tích lũy + trigger rõ ràng)
  6. Toàn bộ 15 features — Fast/Medium/Slow chỉ là lookback khác nhau,
     không loại bỏ feature nào

Chạy: python discover_clusters_v3.py 2>&1 | tee discover_v3_results.txt

Requirements: pip install scikit-learn scipy
"""

from __future__ import annotations

import logging
import sys
import time
import json
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MIN_VOL_BILLION = 3.0

EXISTING_MR  = {"DCM","NKG","DPM","HAH","HCM","HSG","DGC","GAS","NLG","HDB","BMP"}
EXISTING_MOM = {"VCB","BID","MBB","MWG","CTG","FRT","REE","FPT","GMD","STB","PNJ","TCB",
                "SSI","VND","VIX","CTS","VCI","HAG","BCM","ORS","BSR","VSC","DIG","LPB",
                "FTS","APG","VDS"}
EXISTING_BO  = {"VIB","KBC","KSB","HT1","SHB","GVR","SIP",
                "DGC","DCM","NKG","HSG","VIX","FPT","TCB","BSR","VND","GMD","LPB"}
EXISTING_ALL = EXISTING_MR | EXISTING_MOM | EXISTING_BO

# Thresholds — chạy cả 2
SHORT_FWD   = 10
SHORT_UP    = 8.0
SHORT_DOWN  = -6.0

LONG_FWD    = 20
LONG_UP     = 15.0
LONG_DOWN   = -10.0

# ── Multi-timeframe lookback config ──────────────────────────────────────────
# Key insight: mỗi nhóm indicator dùng lookback phù hợp với "tốc độ" của nó
INDICATOR_GROUPS = {
    "fast": {
        "features":  ["momentum_5d", "volume_spike", "candle_body",
                      "stoch_k", "ema_cross", "price_vs_sma20"],
        "lookbacks": [1, 3, 5],     # T-1, T-3, T-5
        "description": "Trigger ngay trước breakout",
    },
    "medium": {
        "features":  ["bb_squeeze", "consolidation", "atr_compression",
                      "momentum_10d", "price_vs_sma50"],
        "lookbacks": [5, 10, 15],   # T-5, T-10, T-15
        "description": "Setup đang hình thành",
    },
    "slow": {
        "features":  ["obv_trend", "higher_lows", "vol_dry_up", "dist_52w_high"],
        "lookbacks": [10, 20, 30],  # T-10, T-20, T-30
        "description": "Tích lũy dài hạn / context",
    },
}

ALL_FEATURES = (
    INDICATOR_GROUPS["fast"]["features"] +
    INDICATOR_GROUPS["medium"]["features"] +
    INDICATOR_GROUPS["slow"]["features"]
)  # = 15 features, đủ cả 3 nhóm

# Build feature → group mapping
FEATURE_GROUP = {}
for grp, cfg in INDICATOR_GROUPS.items():
    for feat in cfg["features"]:
        FEATURE_GROUP[feat] = grp

MIN_SIGNAL_DAYS = 15

# Clustering
K_MIN = 2
K_MAX = 7
DBSCAN_EPS_RANGE   = [0.3, 0.5, 0.8, 1.0, 1.5]
DBSCAN_MIN_SAMPLES = 4
MIN_SYMBOLS_PER_CLUSTER = 6

# Cluster validity rule: cần cả slow + fast
MIN_SLOW_DOMINANT = 1
MIN_FAST_DOMINANT = 1
COHEN_D_MIN       = 0.25

TOP_N_SIGNATURE = 5  # tăng từ 4 lên 5 để capture mixed signals

# MR/MOM/BO features — để identify overlap
MR_MOM_FEATURES = {"price_vs_sma50","ema_cross","stoch_k","momentum_5d",
                   "volume_spike","candle_body"}
BO_FEATURES     = {"bb_squeeze","consolidation","vol_dry_up"}

# Backtest / WF
TRAIN_START     = date(2019, 1, 1)
TRAIN_END       = date(2023, 12, 31)
WF_START        = date(2022, 1, 1)
WF_TRAIN_MONTHS = 18
WF_TEST_MONTHS  = 6
WF_MIN_FOLDS    = 3
MIN_TRADES      = 10
MIN_EXP         = 0.3
MIN_WR          = 45.0
MIN_PF          = 1.1
MIN_WFE         = 0.3
MIN_CONSISTENCY = 60.0


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION — tất cả 15 features
# ══════════════════════════════════════════════════════════════════════════════

def _ema(c, span):
    return pd.Series(c.astype(float)).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c.astype(float)).rolling(p, min_periods=p).mean().values


def _swing_higher_lows(low: np.ndarray, window: int = 30) -> np.ndarray:
    """
    Tính higher_lows dựa trên SWING LOWS thực sự (không phải rolling).
    Swing low = local minimum: low[i] < low[i-1] và low[i] < low[i+1]
    Score = 1 nếu 2 swing lows gần nhất đều higher, 0 nếu không.
    """
    n      = len(low)
    result = np.zeros(n)

    for i in range(window, n - 1):
        # Tìm swing lows trong [i-window, i]
        swing_lows = []
        for j in range(i - window + 1, i):
            if j > 0 and low[j] < low[j-1] and low[j] < low[j+1]:
                swing_lows.append((j, low[j]))

        if len(swing_lows) >= 2:
            # Lấy 2 swing lows gần nhất
            sl1_val = swing_lows[-2][1]
            sl2_val = swing_lows[-1][1]
            # Higher low: swing low sau > swing low trước
            if sl2_val > sl1_val:
                result[i] = 1.0
            # Thêm: consecutive higher lows score cao hơn
            if len(swing_lows) >= 3 and swing_lows[-3][1] < sl1_val < sl2_val:
                result[i] = 1.5   # 3 consecutive higher lows

    return result


def compute_all_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Tính toàn bộ 15 features với các cải tiến V3."""
    if len(df) < 120:
        return None

    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    vol   = df["volume"].values.astype(float)
    opn   = df["open"].values.astype(float)

    # ── Base ─────────────────────────────────────────────────────────────────
    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    sma20  = _sma(close, 20)
    sma50  = _sma(close, 50)
    vsma20 = _sma(vol,   20)
    vsma60 = _sma(vol,   60)

    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr14  = _sma(tr, 14)
    atr60  = _sma(tr, 60)

    lo14   = pd.Series(low).rolling(14).min().values
    hi14   = pd.Series(high).rolling(14).max().values
    denom  = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch  = 100 * (close - lo14) / denom

    c5  = np.concatenate([[close[0]]*5,  close[:-5]])
    c10 = np.concatenate([[close[0]]*10, close[:-10]])

    # ── Fast features ─────────────────────────────────────────────────────────
    momentum_5d  = (close / (c5  + 1e-9) - 1.0) * 100
    momentum_10d = (close / (c10 + 1e-9) - 1.0) * 100
    volume_spike = (vol / (vsma20 + 1e-9)) - 1.0
    candle_body  = np.clip(np.abs(close - opn) / (atr14 + 1e-9), 0, 3)
    ema_cross    = (ema12 - ema26) / (close + 1e-9) * 100
    p_vs_sma20   = (close - sma20) / (close + 1e-9) * 100

    # ── Medium features ───────────────────────────────────────────────────────
    bb_std   = pd.Series(close).rolling(20).std().values
    bb_width = (4 * bb_std) / (sma20 + 1e-9) * 100   # bb_squeeze

    def _consol(x):
        if len(x) < 5: return 0.0
        mid = x[-1]
        return float(np.sum(np.abs(x - mid) / (mid + 1e-9) < 0.03)) / len(x)
    consolidation = pd.Series(close).rolling(15).apply(_consol, raw=True).values

    atr_compression = atr14 / (atr60 + 1e-9)
    p_vs_sma50      = (close - sma50) / (close + 1e-9) * 100

    # ── Slow features — cải tiến V3 ───────────────────────────────────────────
    # OBV với SMA40 (V2 dùng SMA10 → quá ngắn)
    price_chg = np.concatenate([[0], np.diff(close)])
    obv       = np.cumsum(np.sign(price_chg) * vol)
    obv_sma40 = _sma(obv, 40)   # ← V3: SMA40 thay SMA10
    obv_trend = (obv - obv_sma40) / (np.abs(obv_sma40) + 1e-9) * 100

    # Higher lows với swing-based (V2 dùng rolling 10 ngày → noise)
    higher_lows = _swing_higher_lows(low, window=30)  # ← V3: swing lows 30 ngày

    # Vol dry up: vsma20 vs vsma60
    vol_dry_up = (vsma20 / (vsma60 + 1e-9)) - 1.0

    # Distance from 52w high
    hi252      = pd.Series(high).rolling(252, min_periods=60).max().values
    dist_52w   = (close / (hi252 + 1e-9) - 1.0) * 100

    # ── Build result ──────────────────────────────────────────────────────────
    result = df.copy()
    # Fast
    result["momentum_5d"]   = momentum_5d
    result["volume_spike"]  = volume_spike
    result["candle_body"]   = candle_body
    result["stoch_k"]       = stoch
    result["ema_cross"]     = ema_cross
    result["price_vs_sma20"]= p_vs_sma20
    # Medium
    result["bb_squeeze"]    = bb_width
    result["consolidation"] = consolidation
    result["atr_compression"]= atr_compression
    result["momentum_10d"]  = momentum_10d
    result["price_vs_sma50"]= p_vs_sma50
    # Slow
    result["obv_trend"]     = obv_trend
    result["higher_lows"]   = higher_lows
    result["vol_dry_up"]    = vol_dry_up
    result["dist_52w_high"] = dist_52w

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL PATTERN DISCOVERY — Multi-timeframe Cohen's d
# ══════════════════════════════════════════════════════════════════════════════

def find_pattern_signature(df_feat: pd.DataFrame,
                           symbol: str) -> Optional[dict]:
    """
    Tìm pattern signature với multi-timeframe lookback.

    Key difference vs V2:
    - Fast features:   lookback T-1, T-3, T-5
    - Medium features: lookback T-5, T-10, T-15
    - Slow features:   lookback T-10, T-20, T-30

    Cohen's d = signal_days vs counter_days (lần giảm) — giữ từ V2
    """
    close     = df_feat["close"].values.astype(float)
    n         = len(close)

    # Tìm signal days (trước khi tăng mạnh) và counter days (trước khi giảm)
    up_idx, down_idx = [], []

    for i in range(n - max(SHORT_FWD, LONG_FWD)):
        fwd10 = (close[i + SHORT_FWD] - close[i]) / (close[i] + 1e-9) * 100
        fwd20 = (close[i + LONG_FWD]  - close[i]) / (close[i] + 1e-9) * 100

        if fwd10 >= SHORT_UP or fwd20 >= LONG_UP:
            up_idx.append(i)
        if fwd10 <= SHORT_DOWN or fwd20 <= LONG_DOWN:
            down_idx.append(i)

    if len(up_idx) < MIN_SIGNAL_DAYS or len(down_idx) < MIN_SIGNAL_DAYS:
        return None

    # Tính Cohen's d per feature, dùng lookback đúng nhóm
    cohen_d       = {}
    cohen_d_group = {}   # feat → group name

    for grp_name, grp_cfg in INDICATOR_GROUPS.items():
        lookbacks = grp_cfg["lookbacks"]
        for feat in grp_cfg["features"]:
            if feat not in df_feat.columns:
                continue
            feat_vals = df_feat[feat].values.astype(float)

            sig_vals, ctr_vals = [], []
            for lb in lookbacks:
                for i in up_idx:
                    if i >= lb and np.isfinite(feat_vals[i - lb]):
                        sig_vals.append(feat_vals[i - lb])
                for i in down_idx:
                    if i >= lb and np.isfinite(feat_vals[i - lb]):
                        ctr_vals.append(feat_vals[i - lb])

            if len(sig_vals) < 8 or len(ctr_vals) < 8:
                continue

            sig_arr = np.array(sig_vals)
            ctr_arr = np.array(ctr_vals)
            pooled_std = np.sqrt((np.var(sig_arr) + np.var(ctr_arr)) / 2) + 1e-9
            d = (np.mean(sig_arr) - np.mean(ctr_arr)) / pooled_std

            cohen_d[feat]       = round(float(d), 3)
            cohen_d_group[feat] = grp_name

    if not cohen_d:
        return None

    # Sort by |d|
    sorted_feats = sorted(cohen_d.items(), key=lambda x: abs(x[1]), reverse=True)
    dominant     = [(f, d) for f, d in sorted_feats if abs(d) >= COHEN_D_MIN]

    if len(dominant) < 2:
        return None

    # Check có đủ slow + fast không
    dominant_groups = [cohen_d_group.get(f, "fast") for f, _ in dominant[:TOP_N_SIGNATURE]]
    has_slow = "slow"   in dominant_groups
    has_fast = "fast"   in dominant_groups

    signature = {f: "UP" if d > 0 else "DOWN"
                 for f, d in dominant[:TOP_N_SIGNATURE]}

    return {
        "symbol":        symbol,
        "n_up":          len(up_idx),
        "n_down":        len(down_idx),
        "cohen_d":       cohen_d,
        "cohen_d_group": cohen_d_group,
        "dominant":      dominant[:TOP_N_SIGNATURE],
        "signature":     signature,
        "has_slow":      has_slow,
        "has_fast":      has_fast,
        "top_feature":   dominant[0][0] if dominant else None,
        "top_d":         dominant[0][1] if dominant else 0,
        "top_group":     cohen_d_group.get(dominant[0][0], "?") if dominant else "?",
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING — K-Means + DBSCAN (giữ từ V2)
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering(signatures: list[dict]) -> dict:
    """K-Means với silhouette optimization."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        from sklearn.impute import SimpleImputer
    except ImportError:
        print("pip install scikit-learn"); sys.exit(1)

    symbols = [s["symbol"] for s in signatures]
    matrix  = [[s["cohen_d"].get(f, 0.0) for f in ALL_FEATURES]
               for s in signatures]
    X = np.array(matrix)
    X = SimpleImputer(strategy="mean").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X)

    best_k, best_score = 2, -1
    scores = {}
    for k in range(K_MIN, min(K_MAX+1, len(symbols)//MIN_SYMBOLS_PER_CLUSTER+1)):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
            if len(set(labels)) < 2: continue
            score = silhouette_score(X_scaled, labels)
            scores[k] = round(score, 3)
            if score > best_score:
                best_score, best_k = score, k
        except Exception:
            continue

    labels   = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_scaled)
    clusters = {}
    for sym, lbl in zip(symbols, labels):
        clusters.setdefault(int(lbl), []).append(sym)
    centroids = {
        cid: dict(zip(ALL_FEATURES, np.mean(X[[symbols.index(s) for s in syms]], axis=0)))
        for cid, syms in clusters.items()
    }
    return {
        "best_k": best_k, "best_score": best_score, "all_scores": scores,
        "clusters": clusters, "centroids": centroids,
        "X_scaled": X_scaled, "symbols": symbols, "labels": labels.tolist(),
    }


def run_dbscan(signatures: list[dict], X_scaled: np.ndarray) -> dict:
    """DBSCAN để cross-validate."""
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
    except ImportError:
        return {}

    symbols     = [s["symbol"] for s in signatures]
    best_result = None
    best_score  = -1

    for eps in DBSCAN_EPS_RANGE:
        db     = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
        labels = db.fit_predict(X_scaled)
        n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
        if n_cl < 2 or n_cl > K_MAX:
            continue
        mask = labels != -1
        if mask.sum() < 4: continue
        try:
            score = silhouette_score(X_scaled[mask], labels[mask])
        except Exception:
            continue
        if score > best_score:
            best_score  = score
            best_result = {
                "eps": eps, "labels": labels.tolist(),
                "n_clusters": n_cl, "n_outliers": list(labels).count(-1),
                "score": round(score, 3),
            }

    if not best_result:
        return {"status": "NO_VALID_CLUSTERING"}

    clusters = {}
    outliers = []
    for sym, lbl in zip(symbols, best_result["labels"]):
        if lbl == -1:
            outliers.append(sym)
        else:
            clusters.setdefault(int(lbl), []).append(sym)

    return {
        "method": "DBSCAN", **best_result,
        "clusters": clusters, "outliers": outliers, "symbols": symbols,
    }


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRET CLUSTER — với slow/fast validity check
# ══════════════════════════════════════════════════════════════════════════════

def interpret_cluster(cluster_id: int, symbols: list[str],
                      centroid: dict, all_signatures: dict,
                      db_overlap: float = 0.0) -> dict:
    """
    Phân tích cluster — thêm rule V3:
    Cluster hợp lệ cần ≥1 slow + ≥1 fast indicator dominant.
    """
    # Aggregate signatures
    feat_votes = {}
    grp_votes  = {"fast": 0, "medium": 0, "slow": 0}

    for sym in symbols:
        sig = all_signatures.get(sym)
        if not sig: continue
        for feat, direction in sig["signature"].items():
            feat_votes.setdefault(feat, {"UP": 0, "DOWN": 0})[direction] += 1
        for feat, _ in sig["dominant"]:
            grp = FEATURE_GROUP.get(feat, "fast")
            grp_votes[grp] += 1

    # Dominant features (≥40% mã trong cluster có feature này)
    dominant = []
    for feat, votes in sorted(feat_votes.items(),
                               key=lambda x: max(x[1].values()), reverse=True):
        total     = votes["UP"] + votes["DOWN"]
        direction = "UP" if votes["UP"] >= votes["DOWN"] else "DOWN"
        consensus = max(votes["UP"], votes["DOWN"]) / max(total, 1)
        if total >= len(symbols) * 0.4:
            dominant.append((feat, direction, round(consensus, 2),
                             FEATURE_GROUP.get(feat, "fast")))

    # Feature sets
    dominant_names = {f for f, _, _, _ in dominant}
    overlap_existing = dominant_names & (MR_MOM_FEATURES | BO_FEATURES)
    new_features     = dominant_names - MR_MOM_FEATURES - BO_FEATURES
    overlap_ratio    = len(overlap_existing) / max(len(dominant_names), 1)
    existing_sym_ratio = len(set(symbols) & EXISTING_ALL) / max(len(symbols), 1)

    # Slow/Fast check
    slow_dominant = [f for f, _, _, g in dominant if g == "slow"]
    fast_dominant = [f for f, _, _, g in dominant if g == "fast"]
    has_slow = len(slow_dominant) >= MIN_SLOW_DOMINANT
    has_fast = len(fast_dominant) >= MIN_FAST_DOMINANT

    # Phân loại
    if overlap_ratio >= 0.6 or existing_sym_ratio >= 0.5:
        cluster_type = "MR_MOM_BO_DUPLICATE"
    elif not dominant:
        cluster_type = "NOISE"
    elif not has_slow and not has_fast:
        cluster_type = "NOISE"
    elif not has_slow:
        cluster_type = "FAST_ONLY"    # chỉ trigger, không có context tích lũy
    elif not has_fast:
        cluster_type = "SLOW_ONLY"    # có context nhưng không có trigger rõ
    else:
        cluster_type = "NEW_CANDIDATE"  # có cả slow context + fast trigger ✅

    return {
        "cluster_id":      cluster_id,
        "type":            cluster_type,
        "symbols":         symbols,
        "n_symbols":       len(symbols),
        "dominant":        dominant[:8],
        "slow_dominant":   slow_dominant,
        "fast_dominant":   fast_dominant,
        "new_features":    list(new_features),
        "overlap_existing":list(overlap_existing),
        "overlap_ratio":   round(overlap_ratio, 2),
        "db_overlap":      db_overlap,
        "has_slow":        has_slow,
        "has_fast":        has_fast,
        "grp_votes":       grp_votes,
        "suggested_name":  _suggest_name(dominant, new_features, slow_dominant, fast_dominant),
        "suggested_logic": _suggest_logic(dominant, new_features),
    }


def _suggest_name(dominant, new_features, slow_dom, fast_dom) -> str:
    feat_set = {f for f, _, _, _ in dominant}
    if "obv_trend" in new_features and "higher_lows" in new_features:
        return "Smart Money Accumulation (OBV + Higher Lows)"
    if "higher_lows" in new_features and "volume_spike" in feat_set:
        return "Accumulation Breakout (Higher Lows + Volume)"
    if "obv_trend" in new_features and "momentum_5d" in feat_set:
        return "OBV Divergence Momentum"
    if "dist_52w_high" in new_features and "volume_spike" in feat_set:
        return "Deep Value Breakout (52W Low + Volume)"
    if "atr_compression" in new_features and "volume_spike" in feat_set:
        return "Volatility Squeeze Breakout"
    slow_str = "+".join(slow_dom[:2]) if slow_dom else "?"
    fast_str = "+".join(fast_dom[:2]) if fast_dom else "?"
    return f"Slow({slow_str}) + Fast({fast_str})"


def _suggest_logic(dominant, new_features) -> dict:
    new_dom = [(f, d, g) for f, d, _, g in dominant if f in new_features]
    if not new_dom:
        return {}
    # Regime = slow feature có d cao nhất
    slow_dom = [(f, d, g) for f, d, g in new_dom if g == "slow"]
    fast_dom = [(f, d, g) for f, d, g in new_dom if g == "fast"]
    if not slow_dom:
        slow_dom = new_dom[:1]

    regime_feat, regime_dir, _ = slow_dom[0]
    triggers = {f: d.lower() for f, d, _ in fast_dom[:3]}

    # FWD dựa vào dominant group
    if slow_dom:
        fwd = 20   # slow context → hold lâu hơn
    else:
        fwd = 15

    return {
        "regime":   {regime_feat: regime_dir.lower()},
        "triggers": triggers,
        "fwd_days": fwd,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST + WALK FORWARD — template từ backtest_trailing_stop_v2.py
# ══════════════════════════════════════════════════════════════════════════════

def collect_trades_for_logic(df_feat: pd.DataFrame,
                              logic: dict,
                              window_start: date,
                              window_end: date,
                              train_end: date) -> list[float]:
    """Collect trades với proper train_end để tránh lookahead."""
    if not logic or not logic.get("regime"):
        return []

    regime_feat, regime_dir = list(logic["regime"].items())[0]
    trigger_cfg = logic.get("triggers", {})
    fwd         = logic.get("fwd_days", 15)

    df_feat   = df_feat.reset_index(drop=True)
    dates_arr = pd.to_datetime(df_feat["date"]).dt.date.values
    close_arr = df_feat["close"].values.astype(float)
    n         = len(df_feat)

    # Train end index
    te_idx = next((i for i, d in enumerate(dates_arr) if d > train_end), n)

    # Thresholds từ training window
    def _pct(feat, p):
        vals = df_feat[feat].iloc[:te_idx].dropna().values
        return float(np.percentile(vals, p)) if len(vals) >= 20 else None

    reg_pct    = 30 if regime_dir in ("down","low") else 70
    reg_thresh = _pct(regime_feat, reg_pct)
    if reg_thresh is None:
        return []

    trig_thresh = {}
    for t_feat, t_dir in trigger_cfg.items():
        if t_feat not in df_feat.columns: continue
        t_pct = 70 if t_dir in ("up","high") else 30
        t = _pct(t_feat, t_pct)
        if t is not None:
            trig_thresh[t_feat] = (t, t_dir)

    pnls = []
    for i in range(100, n - fwd):
        d = dates_arr[i]
        if d < window_start or d > window_end:
            continue

        # Regime check
        val = df_feat[regime_feat].iloc[i] if regime_feat in df_feat.columns else np.nan
        if not np.isfinite(val): continue
        in_regime = (val <= reg_thresh if regime_dir in ("down","low")
                     else val >= reg_thresh)
        if not in_regime: continue

        # Triggers
        triggered = 0
        for t_feat, (t_val, t_dir) in trig_thresh.items():
            v = df_feat[t_feat].iloc[i] if t_feat in df_feat.columns else np.nan
            if not np.isfinite(v): continue
            if (t_dir in ("up","high") and v >= t_val) or \
               (t_dir in ("down","low") and v <= t_val):
                triggered += 1

        if triggered < max(1, len(trig_thresh) // 2):
            continue

        entry = close_arr[i]
        if i + fwd < n:
            pnls.append((close_arr[i+fwd] - entry) / entry * 100)

    return pnls


def _metrics(pnls):
    if not pnls: return {"n": 0, "exp": 0.0, "wr": 0.0, "pf": 0.0}
    arr    = np.array(pnls)
    wins   = arr[arr > 0]; losses = np.abs(arr[arr <= 0])
    gw     = wins.sum() if len(wins) > 0 else 0.0
    gl     = losses.sum() if len(losses) > 0 else 1e-9
    return {
        "n":   len(pnls),
        "exp": round(float(np.mean(arr)), 3),
        "wr":  round(float(len(wins)/len(arr)*100), 1),
        "pf":  round(float(gw/gl), 2),
    }


def walk_forward_symbol(sym: str, df_feat: pd.DataFrame,
                        logic: dict) -> dict:
    """Walk Forward với expanding window — correct template."""
    dates_arr  = pd.to_datetime(df_feat["date"]).dt.date.values
    folds      = []
    fold_start = WF_START

    while True:
        train_end = fold_start + timedelta(days=WF_TRAIN_MONTHS * 30)
        test_end  = train_end  + timedelta(days=WF_TEST_MONTHS  * 30)
        if test_end > dates_arr[-1]: break

        is_pnls  = collect_trades_for_logic(df_feat, logic,
                                            fold_start, train_end, train_end)
        oos_pnls = collect_trades_for_logic(df_feat, logic,
                                            train_end, test_end, train_end)

        is_m  = _metrics(is_pnls)
        oos_m = _metrics(oos_pnls)

        if is_m["n"] < 3 or oos_m["n"] < 3:
            fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)
            continue

        wfe = (oos_m["exp"] / is_m["exp"]) if is_m["exp"] > 0.001 else 0.0
        folds.append({
            "period":  f"{train_end.strftime('%Y-%m')}→{test_end.strftime('%Y-%m')}",
            "is_exp":  is_m["exp"], "oos_exp": oos_m["exp"],
            "oos_wr":  oos_m["wr"], "oos_n":   oos_m["n"],
            "wfe":     round(wfe, 2),
        })
        fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)

    if len(folds) < WF_MIN_FOLDS:
        return {"status": "INSUFFICIENT_FOLDS", "folds": folds}

    oos_exps    = [f["oos_exp"] for f in folds]
    wfes        = [f["wfe"]     for f in folds]
    pos_folds   = sum(1 for x in oos_exps if x > 0)
    consistency = round(pos_folds / len(folds) * 100, 1)
    valid_wfes  = [w for w in wfes if np.isfinite(w) and w != 0]
    avg_wfe     = round(float(np.mean(valid_wfes)), 2) if valid_wfes else 0.0
    avg_oos     = round(float(np.mean(oos_exps)), 3)

    status = ("PASS" if avg_wfe >= MIN_WFE
              and consistency >= MIN_CONSISTENCY
              and avg_oos > 0.0
              else "FAIL")

    return {
        "status": status, "avg_wfe": avg_wfe,
        "avg_oos_exp": avg_oos, "consistency": consistency,
        "n_folds": len(folds), "folds": folds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def get_universe(load_fn) -> list[str]:
    all_symbols = []
    for source in ["VCI", "KBS"]:
        try:
            from vnstock import Vnstock
            listing = (Vnstock().stock(symbol="VCB", source=source)
                       .listing.symbols_by_exchange())
            hose = listing[listing["exchange"].str.upper() == "HOSE"]
            all_symbols = hose["symbol"].str.upper().tolist()
            logger.info(f"  {source}: {len(all_symbols)} HOSE symbols")
            break
        except Exception as e:
            logger.warning(f"  {source} fail: {e}")

    if not all_symbols: return []
    all_symbols = [s for s in all_symbols
                   if not (len(s) > 3 and s[0]=='C' and s[-1].isdigit())]

    vol_map = {}
    logger.info(f"  Checking volume {len(all_symbols)} symbols...")
    for i, sym in enumerate(all_symbols):
        if i % 50 == 0:
            logger.info(f"  {i}/{len(all_symbols)}...")
        for attempt in range(3):
            try:
                df = load_fn(sym, days=40, min_bars=20)
                if df is None or len(df) < 20: break
                close   = df["close"].values[-20:].astype(float)
                vol     = df["volume"].values[-20:].astype(float)
                avg_vnd = float((vol * close).mean()) * 1000
                if avg_vnd >= MIN_VOL_BILLION * 1e9:
                    vol_map[sym] = avg_vnd
                break
            except Exception as e:
                err = str(e).lower()
                if "rate limit" in err or "60" in err:
                    wait = 65 * (attempt + 1)
                    logger.warning(f"  Rate limit {sym}, wait {wait}s...")
                    time.sleep(wait)
                else: break
        time.sleep(1.1)

    return sorted(vol_map, key=vol_map.get, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    try:
        from vn_loader import load_vn_ohlcv
    except ImportError:
        print("ERROR: Khong import duoc vn_loader."); sys.exit(1)

    print("=" * 65)
    print("DISCOVER CLUSTERS V3 — Multi-Timeframe Pattern Discovery")
    print(f"Fast lookback:   {INDICATOR_GROUPS['fast']['lookbacks']}")
    print(f"Medium lookback: {INDICATOR_GROUPS['medium']['lookbacks']}")
    print(f"Slow lookback:   {INDICATOR_GROUPS['slow']['lookbacks']}")
    print(f"Thresholds: UP={SHORT_UP}%/{SHORT_FWD}d & {LONG_UP}%/{LONG_FWD}d")
    print(f"OBV: SMA40 | Higher Lows: swing-based 30d")
    print(f"Cluster validity: need ≥1 slow + ≥1 fast dominant feature")
    print("=" * 65)

    # ── Bước 1: Universe ─────────────────────────────────────────────────────
    print(f"\nBước 1: Load universe HOSE (vol > {MIN_VOL_BILLION} tỷ)...")
    universe = get_universe(load_vn_ohlcv)
    print(f"  {len(universe)} mã đủ volume")

    # ── Bước 2: Per-symbol pattern discovery ─────────────────────────────────
    print(f"\nBước 2: Multi-timeframe pattern discovery ({len(universe)} mã)...")
    signatures = {}
    df_cache   = {}

    for idx, sym in enumerate(universe, 1):
        print(f"  [{idx:>3}/{len(universe)}] {sym}...", end=" ", flush=True)
        df = None
        for attempt in range(3):
            try:
                df = load_vn_ohlcv(sym, days=2500, min_bars=400)
                break
            except Exception as e:
                err = str(e).lower()
                if "rate limit" in err or "60" in err:
                    wait = 65 * (attempt+1)
                    print(f"wait {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else: break

        if df is None or len(df) < 400:
            print("skip (data)"); continue

        df_feat = compute_all_features(df)
        if df_feat is None:
            print("skip (features)"); continue

        sig = find_pattern_signature(df_feat, sym)
        if sig is None:
            print(f"skip (n_up/down insufficient)"); continue

        df_cache[sym]  = df_feat
        signatures[sym] = sig

        slow_flag = "✅slow" if sig["has_slow"] else "❌slow"
        fast_flag = "✅fast" if sig["has_fast"] else "❌fast"
        print(f"OK (up={sig['n_up']} dn={sig['n_down']} "
              f"top={sig['top_feature']}[{sig['top_group']}]:{sig['top_d']:+.2f} "
              f"{slow_flag} {fast_flag})")
        time.sleep(1.1)

    print(f"\n  → {len(signatures)} mã có signature")
    slow_fast_both = sum(1 for s in signatures.values() if s["has_slow"] and s["has_fast"])
    print(f"  → {slow_fast_both} mã có cả slow + fast signals")

    if len(signatures) < K_MIN * MIN_SYMBOLS_PER_CLUSTER:
        print("ERROR: Không đủ mã."); sys.exit(1)

    # ── Bước 3: Clustering ────────────────────────────────────────────────────
    print(f"\nBước 3: Clustering...")
    sigs_list = [signatures[sym] for sym in signatures]

    km = run_clustering(sigs_list)
    print(f"  K-Means scores: {km['all_scores']}")
    print(f"  → Optimal K={km['best_k']} (score={km['best_score']:.3f})")
    print(f"  (V1=0.242, V2=0.177 — {'BETTER' if km['best_score'] > 0.242 else 'SIMILAR/WORSE'})")

    db = run_dbscan(sigs_list, km["X_scaled"])
    if db.get("clusters"):
        print(f"  DBSCAN: {db['n_clusters']} clusters "
              f"(eps={db['best_eps']}, score={db['best_score']:.3f}, "
              f"outliers={db['n_outliers']})")
    else:
        print(f"  DBSCAN: {db.get('status','no result')}")

    # ── Bước 4: Interpret ─────────────────────────────────────────────────────
    print(f"\nBước 4: Interpret {km['best_k']} clusters...")
    interpretations = []

    for cid, syms in km["clusters"].items():
        # Cross-validate với DBSCAN
        db_overlap = 0.0
        if db.get("clusters"):
            sym_set = set(syms)
            for db_syms in db["clusters"].values():
                ov = len(sym_set & set(db_syms)) / max(len(sym_set), 1)
                db_overlap = max(db_overlap, ov)

        interp = interpret_cluster(cid, syms, km["centroids"][cid],
                                   signatures, db_overlap)
        interpretations.append(interp)

        icon = ("🟢" if interp["type"] == "NEW_CANDIDATE" else
                "🟡" if "ONLY" in interp["type"] else "🔴")
        print(f"\n  {'='*55}")
        print(f"  Cluster {cid}: {interp['suggested_name']} {icon}")
        print(f"  Type={interp['type']} | {len(syms)} mã | "
              f"DB_overlap={db_overlap:.0%}")
        print(f"  Slow features: {interp['slow_dominant']}")
        print(f"  Fast features: {interp['fast_dominant']}")
        print(f"  Dominant (feat, dir, consensus, group):")
        for feat, direction, consensus, grp in interp["dominant"][:6]:
            new_tag = " ★NEW" if feat not in MR_MOM_FEATURES and feat not in BO_FEATURES else ""
            print(f"    [{grp:6}] {feat:<22}: {direction} ({consensus:.0%}){new_tag}")
        print(f"  → {interp['type']}: {interp['suggested_name']}")

    # ── Bước 5: Backtest + WF ─────────────────────────────────────────────────
    candidates = [i for i in interpretations
                  if i["type"] == "NEW_CANDIDATE"
                  and i["n_symbols"] >= MIN_SYMBOLS_PER_CLUSTER]

    print(f"\n\nBước 5: Backtest + WF cho {len(candidates)} NEW_CANDIDATE clusters...")

    validated = []
    for interp in candidates:
        name   = interp["suggested_name"]
        logic  = interp["suggested_logic"]
        syms   = interp["symbols"]

        print(f"\n  [{name}] — {len(syms)} mã")
        print(f"  Logic: {json.dumps(logic, ensure_ascii=False)}")

        # Quick BT filter
        bt_pass = []
        for sym in syms:
            if sym not in df_cache: continue
            pnls = collect_trades_for_logic(df_cache[sym], logic,
                                            TRAIN_START, TRAIN_END, TRAIN_END)
            m = _metrics(pnls)
            if m["n"] < MIN_TRADES: continue
            if m["exp"] >= MIN_EXP and m["wr"] >= MIN_WR and m["pf"] >= MIN_PF:
                bt_pass.append({"sym": sym, "bt": m})
            else:
                print(f"    {sym} BT FAIL (exp={m['exp']:+.2f}% wr={m['wr']:.0f}%)")

        print(f"  BT pass: {len(bt_pass)}/{len(syms)} mã")
        if len(bt_pass) < MIN_SYMBOLS_PER_CLUSTER // 2:
            print(f"  → Skip WF"); continue

        # Walk Forward
        wf_pass = []
        for r in bt_pass:
            sym = r["sym"]
            print(f"    {sym} WF...", end=" ", flush=True)
            wf = walk_forward_symbol(sym, df_cache[sym], logic)
            icon = "✅" if wf["status"] == "PASS" else "❌"
            print(f"{icon} {wf['status']} "
                  f"WFE={wf.get('avg_wfe','?')} "
                  f"consistency={wf.get('consistency','?')}% "
                  f"OOS={wf.get('avg_oos_exp',0):+.3f}%")
            if wf["status"] == "PASS":
                wf_pass.append({"sym": sym, "bt": r["bt"], "wf": wf})

        if len(wf_pass) >= MIN_SYMBOLS_PER_CLUSTER // 2:
            validated.append({
                "name": name, "logic": logic,
                "symbols": [r["sym"] for r in wf_pass],
                "results": wf_pass,
            })

    # ── Final Report ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("FINAL REPORT — V3")
    print(f"{'='*65}")
    print(f"Universe: {len(universe)} mã | Signatures: {len(signatures)} mã")
    print(f"K-Means score: {km['best_score']:.3f} "
          f"(V1=0.242, V2=0.177 → "
          f"{'BETTER ✅' if km['best_score'] > 0.242 else 'SIMILAR/WORSE ⚠️'})")
    print(f"New candidates: {len(candidates)}")
    print(f"Validated clusters: {len(validated)}")

    if validated:
        print(f"\n✅ CLUSTER MỚI VALIDATED:")
        for vc in validated:
            print(f"\n  [{vc['name']}]")
            print(f"  Logic: {json.dumps(vc['logic'], ensure_ascii=False)}")
            print(f"  Symbols ({len(vc['symbols'])}): {vc['symbols']}")
            print(f"\n  {'Sym':<10} {'BT_Exp':>7} {'BT_WR':>6} {'BT_PF':>6} "
                  f"{'WFE':>6} {'OOS_Exp':>8} {'Consist':>8} {'Score':>7}")
            print(f"  {'─'*60}")
            for r in sorted(vc["results"],
                            key=lambda x: x["bt"]["exp"]*x["bt"]["pf"]*x["wf"]["avg_wfe"],
                            reverse=True):
                bt = r["bt"]; wf = r["wf"]
                score = round(bt["exp"] * bt["pf"] * wf["avg_wfe"], 1)
                print(f"  {r['sym']:<10} {bt['exp']:>+6.2f}% "
                      f"{bt['wr']:>5.0f}% {bt['pf']:>6.2f} "
                      f"{wf['avg_wfe']:>6.2f} "
                      f"{wf['avg_oos_exp']:>+7.3f}% "
                      f"{wf['consistency']:>7.1f}% "
                      f"{score:>7.1f}")
    else:
        print(f"\n  Không tìm được cluster mới validated.")
        print(f"  → 3 clusters (MR, MOM, Breakout) đã capture phần lớn edge của HOSE.")
        print(f"  → Tập trung vào position sizing và trailing stop optimization.")


if __name__ == "__main__":
    run()
