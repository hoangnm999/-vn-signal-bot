"""
discover_clusters_v2.py — Per-Symbol Pattern Discovery + Clustering
VN Trader Bot V6 — Session 31

Cải tiến so với V1:
  1. Per-symbol pattern discovery trước, cluster sau
     → Mỗi mã tìm "signature" riêng (top indicators) trước
     → So sánh signatures giữa các mã để gom cluster
  2. Review cả tăng VÀ giảm
     → Cohen's d: signal_days (tăng mạnh) vs counter_days (giảm mạnh)
     → Sắc nét hơn V1 (V1 dùng signal vs all days)
  3. Loại features correlated cao (> 0.7) để tránh double-count
  4. Threshold mạnh hơn: SHORT +8%/10d, LONG +15%/20d
  5. DBSCAN song song K-Means → cross-validate
  6. Rate limit handling: 1.1s/request, retry 3 lần

Chạy: python discover_clusters_v2.py 2>&1 | tee discover_v2_results.txt

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

EXISTING_MR  = {"DCM", "NKG", "DPM", "HAH", "HCM", "HSG", "DGC", "GAS",
                "NLG", "HDB", "BMP"}
EXISTING_MOM = {"VCB", "BID", "MBB", "MWG", "CTG", "FRT", "REE", "FPT",
                "GMD", "STB", "PNJ", "TCB", "SSI", "VND", "VIX", "CTS",
                "VCI", "HAG", "BCM", "ORS", "BSR", "VSC", "DIG", "LPB",
                "FTS", "APG", "VDS"}
EXISTING_ALL = EXISTING_MR | EXISTING_MOM

# Threshold mạnh hơn V1
SHORT_FWD    = 10
SHORT_UP     = 8.0    # +8% trong 10 ngày → big short move up
SHORT_DOWN   = -6.0   # -6% trong 10 ngày → significant down move

LONG_FWD     = 20
LONG_UP      = 15.0   # +15% trong 20 ngày → big long move up
LONG_DOWN    = -10.0  # -10% trong 20 ngày → significant down move

MIN_SIGNAL_DAYS = 20  # cần ít nhất 20 lần tăng/giảm để tính Cohen's d

# Feature correlation threshold — loại 1 trong 2 nếu corr > này
MAX_CORR = 0.70

# Cohen's d threshold để feature được coi là "dominant" trong signature
COHEN_D_MIN = 0.25    # thấp hơn V1 (0.3) vì dùng signal vs counter (sắc hơn)

# Top N features cho signature của mỗi mã
TOP_N_SIGNATURE = 4

# Clustering
K_MIN = 2
K_MAX = 7
MIN_SYMBOLS_PER_CLUSTER = 6   # thấp hơn V1 (8) vì cluster nhỏ hơn sau per-symbol

# DBSCAN params
DBSCAN_EPS_RANGE   = [0.3, 0.5, 0.8, 1.0, 1.5]
DBSCAN_MIN_SAMPLES = 4

# Backtest filter
TRAIN_START  = date(2019, 1, 1)
TRAIN_END    = date(2023, 12, 31)
MIN_TRADES   = 12
MIN_EXP      = 0.3
MIN_WR       = 45.0
MIN_PF       = 1.1

# Walk Forward
WF_START        = date(2022, 1, 1)
WF_TRAIN_MONTHS = 18
WF_TEST_MONTHS  = 6
MIN_WFE         = 0.3
MIN_CONSISTENCY = 60.0

# MR/MOM features (để identify nếu cluster mới trùng)
MR_MOM_FEATURES = {
    "price_vs_sma50", "ema_cross", "stoch_k",
    "momentum_5d", "volume_spike", "candle_body"
}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _ema(c, span):
    return pd.Series(c.astype(float)).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c.astype(float)).rolling(p, min_periods=p).mean().values

def compute_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Tính toàn bộ features. Trả về df với cột indicators."""
    if len(df) < 120:
        return None

    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    vol   = df["volume"].values.astype(float)
    opn   = df["open"].values.astype(float)

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

    bb_std   = pd.Series(close).rolling(20).std().values
    bb_width = (4 * bb_std) / (sma20 + 1e-9) * 100

    price_chg = np.concatenate([[0], np.diff(close)])
    obv       = np.cumsum(np.sign(price_chg) * vol)
    obv_sma10 = _sma(obv, 10)

    hi252 = pd.Series(high).rolling(252, min_periods=60).max().values

    higher_lows = pd.Series(low).rolling(10).apply(
        lambda x: float(np.sum(np.diff(x) > 0)) / max(len(x)-1, 1),
        raw=True
    ).values

    def _consol(x):
        if len(x) < 5: return 0.0
        mid = x[-1]
        return float(np.sum(np.abs(x - mid) / (mid + 1e-9) < 0.03)) / len(x)
    consolidation = pd.Series(close).rolling(15).apply(_consol, raw=True).values

    result = df.copy()

    # MR/MOM features
    result["price_vs_sma50"] = (close - sma50) / (close + 1e-9) * 100
    result["ema_cross"]      = (ema12 - ema26) / (close + 1e-9) * 100
    result["momentum_5d"]    = (close / (c5  + 1e-9) - 1.0) * 100
    result["volume_spike"]   = (vol / (vsma20 + 1e-9)) - 1.0
    result["stoch_k"]        = stoch
    result["candle_body"]    = np.clip(np.abs(close - opn) / (atr14 + 1e-9), 0, 3)

    # New features
    result["bb_squeeze"]      = bb_width
    result["atr_compression"] = atr14 / (atr60 + 1e-9)
    result["obv_trend"]       = (obv - obv_sma10) / (np.abs(obv_sma10) + 1e-9) * 100
    result["vol_dry_up"]      = (vsma20 / (vsma60 + 1e-9)) - 1.0
    result["dist_52w_high"]   = (close / (hi252 + 1e-9) - 1.0) * 100
    result["consolidation"]   = consolidation
    result["higher_lows"]     = higher_lows
    result["momentum_10d"]    = (close / (c10 + 1e-9) - 1.0) * 100

    return result


ALL_FEATURES = [
    "price_vs_sma50", "ema_cross", "momentum_5d", "volume_spike",
    "stoch_k", "candle_body",
    "bb_squeeze", "atr_compression",
    "obv_trend", "vol_dry_up",
    "dist_52w_high", "consolidation", "higher_lows",
    "momentum_10d",
]


def remove_correlated_features(df_feat: pd.DataFrame,
                                features: list[str]) -> list[str]:
    """
    Loại bỏ features có correlation > MAX_CORR với nhau.
    Giữ feature có variance cao hơn khi 2 cái correlated.
    """
    data = df_feat[features].dropna()
    if len(data) < 50:
        return features

    corr_matrix = data.corr().abs()
    to_remove   = set()

    for i in range(len(features)):
        if features[i] in to_remove:
            continue
        for j in range(i + 1, len(features)):
            if features[j] in to_remove:
                continue
            if corr_matrix.iloc[i, j] > MAX_CORR:
                # Giữ feature có variance cao hơn
                var_i = data[features[i]].var()
                var_j = data[features[j]].var()
                drop  = features[j] if var_i >= var_j else features[i]
                to_remove.add(drop)

    kept = [f for f in features if f not in to_remove]
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# PER-SYMBOL PATTERN DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def find_pattern_signature(df_feat: pd.DataFrame,
                           symbol: str,
                           active_features: list[str]) -> Optional[dict]:
    """
    Tìm pattern signature của 1 mã:
    1. Tìm signal_days (trước khi tăng mạnh)
    2. Tìm counter_days (trước khi giảm mạnh)
    3. Tính Cohen's d: signal vs counter (sắc hơn V1)
    4. Trả về top features theo |Cohen's d|

    Đây là điểm khác biệt chính so với V1:
    - V1: signal vs ALL days (baseline)
    - V2: signal vs counter_days (lần giảm) → signal sắc nét hơn
    """
    close     = df_feat["close"].values.astype(float)
    dates_arr = pd.to_datetime(df_feat["date"]).dt.date.values
    n         = len(close)

    # Tìm signal days (T trước khi giá tăng mạnh)
    up_short_idx  = []
    up_long_idx   = []
    down_short_idx = []
    down_long_idx  = []

    for i in range(n - max(SHORT_FWD, LONG_FWD)):
        fwd10 = (close[i + SHORT_FWD] - close[i]) / (close[i] + 1e-9) * 100
        fwd20 = (close[i + LONG_FWD]  - close[i]) / (close[i] + 1e-9) * 100

        if fwd10 >= SHORT_UP:
            up_short_idx.append(i)
        if fwd10 <= SHORT_DOWN:
            down_short_idx.append(i)
        if fwd20 >= LONG_UP:
            up_long_idx.append(i)
        if fwd20 <= LONG_DOWN:
            down_long_idx.append(i)

    # Kết hợp up và down signals
    signal_idx  = list(set(up_short_idx + up_long_idx))
    counter_idx = list(set(down_short_idx + down_long_idx))

    if len(signal_idx) < MIN_SIGNAL_DAYS or len(counter_idx) < MIN_SIGNAL_DAYS:
        return None

    # Tính Cohen's d cho từng feature: signal vs counter
    cohen_d = {}
    for feat in active_features:
        if feat not in df_feat.columns:
            continue
        feat_vals = df_feat[feat].values.astype(float)

        # Lookback T-1, T-3, T-5
        sig_vals = []
        ctr_vals = []
        for lb in [1, 3, 5]:
            for i in signal_idx:
                if i >= lb and np.isfinite(feat_vals[i - lb]):
                    sig_vals.append(feat_vals[i - lb])
            for i in counter_idx:
                if i >= lb and np.isfinite(feat_vals[i - lb]):
                    ctr_vals.append(feat_vals[i - lb])

        if len(sig_vals) < 10 or len(ctr_vals) < 10:
            continue

        sig_arr = np.array(sig_vals)
        ctr_arr = np.array(ctr_vals)

        # Cohen's d = (mean_signal - mean_counter) / pooled_std
        pooled_std = np.sqrt((np.var(sig_arr) + np.var(ctr_arr)) / 2) + 1e-9
        d = (np.mean(sig_arr) - np.mean(ctr_arr)) / pooled_std
        cohen_d[feat] = round(float(d), 3)

    if not cohen_d:
        return None

    # Top features theo |d|
    sorted_feats = sorted(cohen_d.items(), key=lambda x: abs(x[1]), reverse=True)
    dominant     = [(f, d) for f, d in sorted_feats if abs(d) >= COHEN_D_MIN]

    if len(dominant) < 2:
        return None

    # Signature = top N features với hướng (positive/negative)
    signature = {f: ("UP" if d > 0 else "DOWN")
                 for f, d in dominant[:TOP_N_SIGNATURE]}

    return {
        "symbol":      symbol,
        "n_up":        len(signal_idx),
        "n_down":      len(counter_idx),
        "cohen_d":     cohen_d,
        "dominant":    dominant[:TOP_N_SIGNATURE],
        "signature":   signature,
        "top_feature": dominant[0][0] if dominant else None,
        "top_d":       dominant[0][1] if dominant else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING — K-Means + DBSCAN
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans(signatures: list[dict],
               active_features: list[str]) -> dict:
    """K-Means clustering trên Cohen's d matrix."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        from sklearn.impute import SimpleImputer
    except ImportError:
        print("ERROR: pip install scikit-learn")
        sys.exit(1)

    symbols = [s["symbol"] for s in signatures]
    matrix  = [[s["cohen_d"].get(f, 0.0) for f in active_features]
               for s in signatures]
    X = np.array(matrix)

    imputer  = SimpleImputer(strategy="mean")
    X        = imputer.fit_transform(X)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k     = 2
    best_score = -1
    scores     = {}

    for k in range(K_MIN, min(K_MAX + 1, len(symbols) // MIN_SYMBOLS_PER_CLUSTER + 1)):
        try:
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            score  = silhouette_score(X_scaled, labels)
            scores[k] = round(score, 3)
            if score > best_score:
                best_score = score
                best_k     = k
        except Exception:
            continue

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels   = km_final.fit_predict(X_scaled)

    clusters = {}
    for sym, label in zip(symbols, labels):
        clusters.setdefault(int(label), []).append(sym)

    centroids = {}
    for label_id, syms in clusters.items():
        idxs = [symbols.index(s) for s in syms]
        centroids[label_id] = dict(zip(active_features,
                                       np.mean(X[idxs], axis=0)))

    return {
        "method":     "KMeans",
        "best_k":     best_k,
        "best_score": best_score,
        "all_scores": scores,
        "clusters":   clusters,
        "centroids":  centroids,
        "X_scaled":   X_scaled,
        "symbols":    symbols,
        "labels":     labels.tolist(),
    }


def run_dbscan(signatures: list[dict],
               active_features: list[str],
               X_scaled: np.ndarray) -> dict:
    """DBSCAN clustering — tự tìm K, xử lý outliers tốt hơn K-Means."""
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
    except ImportError:
        return {}

    symbols    = [s["symbol"] for s in signatures]
    best_result = None
    best_score  = -1

    for eps in DBSCAN_EPS_RANGE:
        db     = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES)
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers  = list(labels).count(-1)

        if n_clusters < 2 or n_clusters > K_MAX:
            continue

        # Chỉ tính silhouette cho non-outlier points
        mask = labels != -1
        if mask.sum() < 4:
            continue

        try:
            score = silhouette_score(X_scaled[mask], labels[mask])
        except Exception:
            continue

        if score > best_score:
            best_score  = score
            best_result = {
                "eps":       eps,
                "labels":    labels.tolist(),
                "n_clusters": n_clusters,
                "n_outliers": n_outliers,
                "score":      round(score, 3),
            }

    if best_result is None:
        return {"method": "DBSCAN", "status": "NO_VALID_CLUSTERING"}

    # Build clusters
    clusters  = {}
    outliers  = []
    for sym, label in zip(symbols, best_result["labels"]):
        if label == -1:
            outliers.append(sym)
        else:
            clusters.setdefault(int(label), []).append(sym)

    return {
        "method":     "DBSCAN",
        "best_eps":   best_result["eps"],
        "best_score": best_result["score"],
        "n_clusters": best_result["n_clusters"],
        "n_outliers": best_result["n_outliers"],
        "clusters":   clusters,
        "outliers":   outliers,
        "labels":     best_result["labels"],
        "symbols":    symbols,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATE K-Means vs DBSCAN
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate_clusters(kmeans_result: dict,
                             dbscan_result: dict) -> list[dict]:
    """
    Tìm clusters được confirm bởi cả 2 method.
    Nếu cùng nhóm mã xuất hiện ở cả K-Means và DBSCAN → rất đáng tin.
    """
    confirmed = []

    if not dbscan_result.get("clusters"):
        # DBSCAN không tìm được → dùng K-Means
        for cid, syms in kmeans_result["clusters"].items():
            confirmed.append({
                "symbols":    syms,
                "confidence": "KMEANS_ONLY",
                "centroid":   kmeans_result["centroids"].get(cid, {}),
            })
        return confirmed

    # Tìm overlap giữa K-Means clusters và DBSCAN clusters
    for km_cid, km_syms in kmeans_result["clusters"].items():
        km_set = set(km_syms)
        best_overlap = 0
        best_db_syms = []

        for db_cid, db_syms in dbscan_result["clusters"].items():
            overlap = len(km_set & set(db_syms)) / max(len(km_set), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_db_syms = db_syms

        confidence = ("HIGH"   if best_overlap >= 0.6 else
                      "MEDIUM" if best_overlap >= 0.3 else
                      "LOW")

        confirmed.append({
            "symbols":      km_syms,
            "db_overlap":   round(best_overlap, 2),
            "confidence":   confidence,
            "centroid":     kmeans_result["centroids"].get(km_cid, {}),
        })

    return confirmed


# ══════════════════════════════════════════════════════════════════════════════
# INTERPRET CLUSTER
# ══════════════════════════════════════════════════════════════════════════════

def interpret_cluster(cluster_info: dict,
                      all_signatures: dict,
                      active_features: list[str]) -> dict:
    """Phân tích cluster: dominant features, overlap MR/MOM, đề xuất tên."""
    syms     = cluster_info["symbols"]
    centroid = cluster_info.get("centroid", {})

    # Aggregate signatures của các mã trong cluster
    feat_votes = {}  # feature → {UP: count, DOWN: count}
    for sym in syms:
        sig = all_signatures.get(sym)
        if not sig:
            continue
        for feat, direction in sig["signature"].items():
            if feat not in feat_votes:
                feat_votes[feat] = {"UP": 0, "DOWN": 0}
            feat_votes[feat][direction] += 1

    # Features được vote nhiều nhất
    dominant_feats = []
    for feat, votes in sorted(feat_votes.items(),
                               key=lambda x: max(x[1].values()),
                               reverse=True):
        total     = votes["UP"] + votes["DOWN"]
        direction = "UP" if votes["UP"] >= votes["DOWN"] else "DOWN"
        consensus = max(votes["UP"], votes["DOWN"]) / max(total, 1)
        if total >= len(syms) * 0.4:  # ít nhất 40% mã trong cluster có feature này
            dominant_feats.append((feat, direction, round(consensus, 2)))

    # Check overlap với MR/MOM
    dominant_names  = {f for f, _, _ in dominant_feats}
    overlap_mr_mom  = dominant_names & MR_MOM_FEATURES
    new_features    = dominant_names - MR_MOM_FEATURES
    overlap_ratio   = len(overlap_mr_mom) / max(len(dominant_names), 1)

    # Check overlap với existing watchlist
    existing_overlap = len(set(syms) & EXISTING_ALL) / max(len(syms), 1)

    # Phân loại
    if overlap_ratio >= 0.6 or existing_overlap >= 0.5:
        cluster_type = "MR_MOM_DUPLICATE"
    elif len(dominant_feats) < 2:
        cluster_type = "NOISE"
    else:
        cluster_type = "NEW_CANDIDATE"

    suggested_name  = _suggest_name(dominant_feats, new_features)
    suggested_logic = _suggest_signal_logic(dominant_feats, new_features)

    return {
        "symbols":          syms,
        "n_symbols":        len(syms),
        "type":             cluster_type,
        "dominant_feats":   dominant_feats[:6],
        "new_features":     list(new_features),
        "overlap_mr_mom":   list(overlap_mr_mom),
        "overlap_ratio":    round(overlap_ratio, 2),
        "existing_overlap": round(existing_overlap, 2),
        "confidence":       cluster_info.get("confidence", "?"),
        "db_overlap":       cluster_info.get("db_overlap", 0),
        "suggested_name":   suggested_name,
        "suggested_logic":  suggested_logic,
    }


def _suggest_name(dominant: list, new_features: set) -> str:
    feat_names = {f for f, _, _ in dominant}
    if "bb_squeeze" in new_features and "consolidation" in new_features:
        return "Breakout (BB Squeeze + Consolidation)"
    if "atr_compression" in new_features and "vol_dry_up" in new_features:
        return "Volatility Compression Breakout"
    if "higher_lows" in new_features and "obv_trend" in new_features:
        return "Accumulation (Higher Lows + OBV)"
    if "dist_52w_high" in new_features:
        d_vals = {f: dir_ for f, dir_, _ in dominant}
        return ("Near 52W High Breakout" if d_vals.get("dist_52w_high") == "UP"
                else "Deep Value Recovery")
    if not new_features and dominant:
        return "MR/MOM Variant (check overlap)"
    return f"Pattern: {', '.join(list(new_features)[:3]) or 'mixed'}"


def _suggest_signal_logic(dominant: list, new_features: set) -> dict:
    new_dom = [(f, d, c) for f, d, c in dominant if f in new_features]
    if not new_dom:
        return {}
    regime_feat, regime_dir, _ = new_dom[0]
    triggers = {f: d for f, d, _ in new_dom[1:4]}
    fwd = (15 if "bb_squeeze" in new_features or "consolidation" in new_features
           else 20 if "higher_lows" in new_features else 10)
    return {
        "regime":   {regime_feat: regime_dir.lower()},
        "triggers": triggers,
        "fwd_days": fwd,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST + WALK FORWARD
# ══════════════════════════════════════════════════════════════════════════════

def backtest_cluster(symbols: list[str], logic: dict,
                     df_cache: dict) -> list[dict]:
    """Backtest signal logic đề xuất cho cluster mới."""
    if not logic or not logic.get("regime"):
        return []

    results = []
    regime_feat, regime_cond = list(logic["regime"].items())[0]
    regime_cond = regime_cond.lower()
    trigger_cfg = logic.get("triggers", {})
    fwd         = logic.get("fwd_days", 15)

    for sym in symbols:
        if sym not in df_cache:
            continue
        df      = df_cache[sym]
        df_feat = compute_features(df)
        if df_feat is None:
            continue

        dates_arr = pd.to_datetime(df_feat["date"]).dt.date.values
        close_arr = df_feat["close"].values.astype(float)
        n         = len(df_feat)
        pnls      = []

        for i in range(100, n - fwd):
            d = dates_arr[i]
            if d < TRAIN_START or d > TRAIN_END:
                continue

            sub = df_feat.iloc[:i]

            # Regime check
            if regime_feat not in df_feat.columns:
                continue
            reg_vals = sub[regime_feat].dropna().values
            if len(reg_vals) < 30:
                continue
            pct       = 30 if regime_cond in ("down", "low") else 70
            reg_thresh = np.percentile(reg_vals, pct)
            val        = df_feat[regime_feat].iloc[i]
            if not np.isfinite(val):
                continue
            in_regime = (val <= reg_thresh if regime_cond in ("down", "low")
                         else val >= reg_thresh)
            if not in_regime:
                continue

            # Triggers
            triggered = 0
            for t_feat, t_dir in trigger_cfg.items():
                if t_feat not in df_feat.columns:
                    continue
                tv     = df_feat[t_feat].iloc[i]
                t_vals = sub[t_feat].dropna().values
                if len(t_vals) < 20 or not np.isfinite(tv):
                    continue
                t_pct   = 70 if t_dir.lower() in ("up", "high") else 30
                t_thresh = np.percentile(t_vals, t_pct)
                if ((t_dir.lower() in ("up", "high") and tv >= t_thresh) or
                    (t_dir.lower() in ("down", "low") and tv <= t_thresh)):
                    triggered += 1

            if triggered < 1:
                continue

            entry = close_arr[i]
            if i + fwd < n:
                pnls.append((close_arr[i + fwd] - entry) / entry * 100)

        if len(pnls) < MIN_TRADES:
            continue

        pnls   = np.array(pnls)
        wins   = pnls[pnls > 0]
        losses = np.abs(pnls[pnls <= 0])
        gw     = wins.sum()   if len(wins)   > 0 else 0.0
        gl     = losses.sum() if len(losses) > 0 else 1e-9

        results.append({
            "sym": sym,
            "n":   len(pnls),
            "exp": round(float(np.mean(pnls)), 3),
            "wr":  round(float(len(wins) / len(pnls) * 100), 1),
            "pf":  round(float(gw / gl), 2),
        })

    return [r for r in results
            if r["exp"] >= MIN_EXP and r["wr"] >= MIN_WR and r["pf"] >= MIN_PF]


def walk_forward_cluster(sym: str, logic: dict, df: pd.DataFrame) -> dict:
    """Walk Forward cho 1 mã với logic đề xuất."""
    if not logic:
        return {"status": "NO_LOGIC"}

    df_feat   = compute_features(df)
    if df_feat is None:
        return {"status": "NO_DATA"}

    dates_arr  = pd.to_datetime(df_feat["date"]).dt.date.values
    folds      = []
    fold_start = WF_START

    while True:
        train_end = fold_start + timedelta(days=WF_TRAIN_MONTHS * 30)
        test_end  = train_end  + timedelta(days=WF_TEST_MONTHS  * 30)
        if test_end > dates_arr[-1]:
            break

        def _bt(start, end):
            df_tmp        = df.copy()
            df_tmp["_dt"] = pd.to_datetime(df_tmp["date"]).dt.date
            # Reuse backtest_cluster but for 1 symbol
            r = backtest_cluster([sym], logic, {sym: df_tmp})
            return r[0] if r else {"exp": 0, "n": 0}

        is_r  = _bt(fold_start, train_end)
        oos_r = _bt(train_end,  test_end)

        if is_r["n"] < 5 or oos_r["n"] < 3:
            fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)
            continue

        wfe = (oos_r["exp"] / is_r["exp"]) if is_r["exp"] > 0 else 0.0
        folds.append({
            "period":  f"{train_end.strftime('%Y-%m')}→{test_end.strftime('%Y-%m')}",
            "is_exp":  is_r["exp"],
            "oos_exp": oos_r["exp"],
            "wfe":     round(wfe, 2),
        })
        fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)

    if len(folds) < 3:
        return {"status": "INSUFFICIENT_FOLDS"}

    oos_exps    = [f["oos_exp"] for f in folds]
    consistency = round(sum(1 for x in oos_exps if x > 0) / len(folds) * 100, 1)
    avg_wfe     = round(float(np.mean([f["wfe"] for f in folds
                                       if np.isfinite(f["wfe"])])), 2)
    avg_oos_exp = round(float(np.mean(oos_exps)), 3)

    status = ("PASS" if avg_wfe >= MIN_WFE
              and consistency >= MIN_CONSISTENCY
              and avg_oos_exp > 0
              else "FAIL")

    return {
        "status":      status,
        "avg_wfe":     avg_wfe,
        "avg_oos_exp": avg_oos_exp,
        "consistency": consistency,
        "n_folds":     len(folds),
        "folds":       folds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def get_universe(load_vn_ohlcv) -> list[str]:
    """Lấy HOSE universe vol > 3 tỷ, avg 20 phiên."""
    all_symbols = []
    for source in ["VCI", "KBS"]:
        try:
            from vnstock import Vnstock
            listing = (Vnstock().stock(symbol="VCB", source=source)
                       .listing.symbols_by_exchange())
            hose_df = listing[listing["exchange"].str.upper() == "HOSE"]
            all_symbols = hose_df["symbol"].str.upper().tolist()
            logger.info(f"  {source} listing: {len(all_symbols)} HOSE symbols")
            break
        except Exception as e:
            logger.warning(f"  {source} fail: {e}")

    if not all_symbols:
        return []

    # Filter chứng quyền
    all_symbols = [s for s in all_symbols
                   if not (len(s) > 3 and s[0] == 'C' and s[-1].isdigit())]

    vol_map = {}
    logger.info(f"  Checking volume for {len(all_symbols)} symbols...")
    for i, sym in enumerate(all_symbols):
        if i % 50 == 0:
            logger.info(f"  {i}/{len(all_symbols)}...")
        for attempt in range(3):
            try:
                df = load_vn_ohlcv(sym, days=40, min_bars=20)
                if df is None or len(df) < 20:
                    break
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
                    logger.warning(f"  Rate limit at {sym}, wait {wait}s...")
                    time.sleep(wait)
                else:
                    break
        time.sleep(1.1)

    return sorted(vol_map, key=vol_map.get, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    try:
        from vn_loader import load_vn_ohlcv
    except ImportError:
        print("ERROR: Khong import duoc vn_loader.")
        sys.exit(1)

    print("=" * 65)
    print("DISCOVER CLUSTERS V2 — VN Trader Bot V6")
    print(f"Thresholds: UP={SHORT_UP}%/{SHORT_FWD}d & {LONG_UP}%/{LONG_FWD}d")
    print(f"Counter:    DOWN={SHORT_DOWN}%/{SHORT_FWD}d & {LONG_DOWN}%/{LONG_FWD}d")
    print("=" * 65)

    # ── Bước 1: Universe ─────────────────────────────────────────────────────
    print(f"\nBước 1: Load universe HOSE (vol > {MIN_VOL_BILLION} tỷ)...")
    universe = get_universe(load_vn_ohlcv)
    print(f"  {len(universe)} mã đủ volume")

    # ── Bước 2: Load data + tính features ────────────────────────────────────
    print(f"\nBước 2: Load data & tính per-symbol pattern...")
    df_cache    = {}
    signatures  = {}   # sym → signature dict
    active_feats_per_sym = {}

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
                    wait = 65 * (attempt + 1)
                    print(f"wait {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    break

        if df is None or len(df) < 400:
            print("skip (data)")
            continue

        df_feat = compute_features(df)
        if df_feat is None:
            print("skip (features)")
            continue

        # Loại features correlated cao cho mã này
        active = remove_correlated_features(df_feat, ALL_FEATURES)

        sig = find_pattern_signature(df_feat, sym, active)
        if sig is None:
            print(f"skip (n_up={len([i for i in range(len(df_feat)-LONG_FWD) if (df_feat['close'].values[i+LONG_FWD]-df_feat['close'].values[i])/df_feat['close'].values[i]*100>=LONG_UP])} < {MIN_SIGNAL_DAYS})")
            continue

        df_cache[sym]              = df
        signatures[sym]            = sig
        active_feats_per_sym[sym]  = active

        print(f"OK (up={sig['n_up']} down={sig['n_down']} "
              f"top={sig['top_feature']}:{sig['top_d']:+.2f})")
        time.sleep(1.1)

    print(f"\n  → {len(signatures)} mã có pattern signature")

    if len(signatures) < K_MIN * MIN_SYMBOLS_PER_CLUSTER:
        print("ERROR: Không đủ mã để clustering.")
        sys.exit(1)

    # ── Bước 3: Chọn active features chung ───────────────────────────────────
    # Lấy features xuất hiện nhiều nhất across tất cả mã
    feat_count = {}
    for feats in active_feats_per_sym.values():
        for f in feats:
            feat_count[f] = feat_count.get(f, 0) + 1
    active_features = [f for f, cnt in sorted(feat_count.items(),
                                               key=lambda x: x[1], reverse=True)
                       if cnt >= len(signatures) * 0.5]  # ít nhất 50% mã có feature này

    print(f"\nBước 3: Active features ({len(active_features)}): {active_features}")

    # ── Bước 4: Clustering ────────────────────────────────────────────────────
    sigs_list = [signatures[sym] for sym in signatures]

    print(f"\nBước 4: K-Means clustering...")
    km_result = run_kmeans(sigs_list, active_features)
    print(f"  Silhouette scores: {km_result['all_scores']}")
    print(f"  → K-Means optimal K={km_result['best_k']} "
          f"(score={km_result['best_score']:.3f})")

    print(f"\n  DBSCAN clustering...")
    db_result = run_dbscan(sigs_list, active_features, km_result["X_scaled"])
    if db_result.get("clusters"):
        print(f"  → DBSCAN: {db_result['n_clusters']} clusters "
              f"(eps={db_result['best_eps']}, score={db_result['best_score']:.3f}, "
              f"outliers={db_result['n_outliers']})")
    else:
        print(f"  → DBSCAN: {db_result.get('status', 'no result')}")

    # ── Bước 5: Cross-validate ────────────────────────────────────────────────
    print(f"\nBước 5: Cross-validate K-Means vs DBSCAN...")
    confirmed = cross_validate_clusters(km_result, db_result)

    # ── Bước 6: Interpret ─────────────────────────────────────────────────────
    print(f"\nBước 6: Interpret {len(confirmed)} clusters...")
    interpretations = []

    for i, cl in enumerate(confirmed):
        interp = interpret_cluster(cl, signatures, active_features)
        interpretations.append(interp)

        conf_icon = ("🟢" if cl.get("confidence") == "HIGH" else
                     "🟡" if cl.get("confidence") == "MEDIUM" else "🔴")
        print(f"\n  {'='*55}")
        print(f"  Cluster {i}: {interp['suggested_name']} {conf_icon}")
        print(f"  Type={interp['type']} | {len(interp['symbols'])} mã "
              f"| KM/DB overlap={interp['db_overlap']:.0%}")
        print(f"  Symbols: {interp['symbols'][:15]}"
              f"{'...' if len(interp['symbols']) > 15 else ''}")
        print(f"  Dominant features (vote %):")
        for feat, direction, consensus in interp["dominant_feats"]:
            new_tag = " ★NEW" if feat not in MR_MOM_FEATURES else ""
            print(f"    {feat:<22}: {direction} ({consensus:.0%}){new_tag}")
        print(f"  → {interp['type']}")

    # ── Bước 7: Backtest + WF cho NEW_CANDIDATE ───────────────────────────────
    new_candidates = [i for i in interpretations
                      if i["type"] == "NEW_CANDIDATE"
                      and i["n_symbols"] >= MIN_SYMBOLS_PER_CLUSTER]

    print(f"\n\nBước 7: Backtest + Walk Forward "
          f"cho {len(new_candidates)} NEW_CANDIDATE clusters...")

    validated = []
    for interp in new_candidates:
        name   = interp["suggested_name"]
        logic  = interp["suggested_logic"]
        syms   = interp["symbols"]

        print(f"\n  [{name}] — {len(syms)} mã")
        print(f"  Logic: {json.dumps(logic, ensure_ascii=False)}")

        bt_pass = backtest_cluster(syms, logic, df_cache)
        print(f"  BT pass: {len(bt_pass)}/{len(syms)} mã "
              f"({[r['sym'] for r in bt_pass]})")

        if len(bt_pass) < MIN_SYMBOLS_PER_CLUSTER // 2:
            print(f"  → Skip WF (không đủ mã pass BT)")
            continue

        wf_pass = []
        for r in bt_pass:
            sym = r["sym"]
            print(f"    {sym} WF...", end=" ", flush=True)
            wf = walk_forward_cluster(sym, logic, df_cache[sym])
            icon = "✅" if wf["status"] == "PASS" else "❌"
            print(f"{icon} {wf['status']} "
                  f"WFE={wf.get('avg_wfe','?')} "
                  f"consistency={wf.get('consistency','?')}%")
            if wf["status"] == "PASS":
                wf_pass.append({"sym": sym, "bt": r, "wf": wf})

        if len(wf_pass) >= MIN_SYMBOLS_PER_CLUSTER // 2:
            validated.append({
                "name":    name,
                "logic":   logic,
                "symbols": [r["sym"] for r in wf_pass],
                "results": wf_pass,
            })

    # ── Final Report ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"FINAL REPORT — V2")
    print(f"{'='*65}")
    print(f"Universe: {len(universe)} mã | Signatures: {len(signatures)} mã")
    print(f"K-Means score: {km_result['best_score']:.3f} "
          f"(V1 was 0.242 — {'better' if km_result['best_score'] > 0.242 else 'similar/worse'})")
    print(f"New candidates: {len(new_candidates)}")
    print(f"Validated clusters: {len(validated)}")

    if validated:
        print(f"\n✅ CLUSTER MỚI VALIDATED:")
        for vc in validated:
            print(f"\n  [{vc['name']}]")
            print(f"  Logic: {json.dumps(vc['logic'], ensure_ascii=False)}")
            print(f"  Symbols: {vc['symbols']}")
            print(f"\n  {'Sym':<7} {'Exp':>6} {'WR':>5} {'PF':>5} "
                  f"{'WFE':>5} {'Consist':>8}")
            print(f"  {'-'*40}")
            for r in vc["results"]:
                bt = r["bt"]
                wf = r["wf"]
                print(f"  {r['sym']:<7} {bt['exp']:>+5.2f}% "
                      f"{bt['wr']:>4.0f}% {bt['pf']:>5.2f} "
                      f"{wf['avg_wfe']:>5.2f} {wf['consistency']:>7.1f}%")
    else:
        print(f"\n  Không tìm được cluster mới validated.")
        print(f"  → Confirm: MR và MOM đã capture phần lớn edge của HOSE.")


if __name__ == "__main__":
    run()
