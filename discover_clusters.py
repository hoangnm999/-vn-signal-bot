"""
discover_clusters.py — Data-First Cluster Discovery
VN Trader Bot V6 — Session 31

Approach (giống cách tìm ra MR và MOM):
  1. Với mỗi mã, tìm tất cả "lần tăng giá đáng kể"
     - Short move: forward 10d > +5%
     - Long  move: forward 20d > +8%
  2. Nhìn lại T-1, T-3, T-5: indicator nào thay đổi bất thường nhất?
     → Tính Cohen's d (signal days vs random days)
  3. Tổng hợp across tất cả mã → feature signature matrix
  4. Clustering (K-Means, tự tìm K tối ưu)
  5. Validate: cluster nào dominated bởi MR/MOM features → discard
     Cluster mới → interpret + đặt tên nếu giải thích được
  6. Backtest signal logic cho cluster mới → Walk Forward nếu pass

Chạy: python discover_clusters.py 2>&1 | tee discover_results.txt

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

# Existing clusters — để identify và discard nếu cluster mới trùng
EXISTING_MR  = {"DCM", "NKG", "DPM", "HAH", "HCM", "HSG", "DGC", "GAS"}
EXISTING_MOM = {"VCB", "BID", "MBB", "MWG", "CTG", "FRT", "REE", "FPT",
                "GMD", "STB", "PNJ", "TCB"}
EXISTING_ALL = EXISTING_MR | EXISTING_MOM

# "Lần tăng giá đáng kể" thresholds
SHORT_FWD     = 10   # ngày
SHORT_THRESH  = 5.0  # % gain trong 10 ngày

LONG_FWD      = 20   # ngày
LONG_THRESH   = 8.0  # % gain trong 20 ngày

# Feature lookback windows
LOOKBACK_DAYS = [1, 3, 5]  # T-1, T-3, T-5 trước khi tăng

# Clustering
K_MIN = 2
K_MAX = 8   # tự tìm K tối ưu bằng silhouette score
MIN_SYMBOLS_PER_CLUSTER = 8

# Backtest filter
TRAIN_START  = date(2019, 1, 1)
TRAIN_END    = date(2023, 12, 31)
MIN_TRADES   = 15
MIN_EXP      = 0.3
MIN_WR       = 45.0
MIN_PF       = 1.1

# Walk Forward filter
WF_START        = date(2022, 1, 1)
WF_TRAIN_MONTHS = 18
WF_TEST_MONTHS  = 6
MIN_WFE         = 0.3
MIN_CONSISTENCY = 60.0

# MR/MOM features — nếu cluster mới dominated bởi những này → discard
MR_MOM_FEATURES = {
    "price_vs_sma50", "ema_cross", "stoch_k",
    "momentum_5d", "volume_spike", "candle_body"
}

# Mức Cohen's d để một feature được coi là "dominant" trong cluster
COHEN_D_THRESHOLD = 0.3


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _ema(c, span):
    return pd.Series(c.astype(float)).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c.astype(float)).rolling(p, min_periods=p).mean().values

def compute_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Tính toàn bộ 15 features cho mỗi ngày.
    Bao gồm cả MR/MOM features (để identify) và NEW features (để discover).
    """
    if len(df) < 120:
        return None

    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    vol   = df["volume"].values.astype(float)
    opn   = df["open"].values.astype(float)
    n     = len(close)

    # ── Base indicators ───────────────────────────────────────────────────────
    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    sma20  = _sma(close, 20)
    sma50  = _sma(close, 50)
    sma120 = _sma(close, 120)
    vsma20 = _sma(vol, 20)
    vsma60 = _sma(vol, 60)

    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr14  = _sma(tr, 14)
    atr60  = _sma(tr, 60)

    lo14 = pd.Series(low).rolling(14).min().values
    hi14 = pd.Series(high).rolling(14).max().values
    denom = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch = 100 * (close - lo14) / denom

    c5  = np.concatenate([[close[0]]*5,  close[:-5]])
    c10 = np.concatenate([[close[0]]*10, close[:-10]])

    # Bollinger Bands
    bb_mid = sma20
    bb_std = pd.Series(close).rolling(20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9) * 100

    # OBV
    price_chg = np.concatenate([[0], np.diff(close)])
    obv_sign  = np.sign(price_chg)
    obv       = np.cumsum(obv_sign * vol)
    obv_sma10 = _sma(obv, 10)

    # 52-week high
    hi252 = pd.Series(high).rolling(252, min_periods=60).max().values

    # Higher lows: count higher lows trong 10 ngày (giá low tăng dần)
    lows_ser = pd.Series(low)
    higher_lows = lows_ser.rolling(10).apply(
        lambda x: float(np.sum(np.diff(x) > 0)) / max(len(x)-1, 1),
        raw=True
    ).values

    # Consolidation: số ngày giá nằm trong range ±3% của close hiện tại
    def _consol(x):
        if len(x) < 5: return 0.0
        mid = x[-1]
        return float(np.sum(np.abs(x - mid) / (mid + 1e-9) < 0.03)) / len(x)
    consolidation = pd.Series(close).rolling(15).apply(_consol, raw=True).values

    result = df.copy()

    # ── MR/MOM features (giữ để identify) ────────────────────────────────────
    result["price_vs_sma50"] = (close - sma50) / (close + 1e-9) * 100
    result["ema_cross"]      = (ema12 - ema26) / (close + 1e-9) * 100
    result["momentum_5d"]    = (close / (c5 + 1e-9) - 1.0) * 100
    result["volume_spike"]   = (vol / (vsma20 + 1e-9)) - 1.0
    result["stoch_k"]        = stoch
    result["candle_body"]    = np.clip(np.abs(close - opn) / (atr14 + 1e-9), 0, 3)

    # ── NEW features (để discover cluster mới) ────────────────────────────────

    # Volatility compression
    result["bb_squeeze"]       = bb_width  # nhỏ = đang tích lũy
    result["atr_compression"]  = atr14 / (atr60 + 1e-9)  # < 1 = vol đang co

    # Volume structure
    result["obv_trend"]    = (obv - obv_sma10) / (np.abs(obv_sma10) + 1e-9) * 100
    result["vol_dry_up"]   = (vsma20 / (vsma60 + 1e-9)) - 1.0  # âm = vol đang giảm
    result["vol_price_div"] = result["momentum_5d"] - (
        (vsma20 / (vsma60 + 1e-9) - 1.0) * 100
    )  # giá tăng nhưng vol giảm → positive

    # Price structure
    result["dist_52w_high"]    = (close / (hi252 + 1e-9) - 1.0) * 100  # âm = xa đỉnh
    result["consolidation"]    = consolidation  # cao = đang sideways
    result["higher_lows"]      = higher_lows    # cao = đáy đang tăng

    # Momentum variants
    result["momentum_10d"]  = (close / (c10 + 1e-9) - 1.0) * 100
    result["price_vs_sma20"] = (close - sma20) / (close + 1e-9) * 100

    return result


ALL_FEATURES = [
    # MR/MOM
    "price_vs_sma50", "ema_cross", "momentum_5d", "volume_spike",
    "stoch_k", "candle_body",
    # NEW
    "bb_squeeze", "atr_compression",
    "obv_trend", "vol_dry_up", "vol_price_div",
    "dist_52w_high", "consolidation", "higher_lows",
    "momentum_10d", "price_vs_sma20",
]


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: TÌM "LẦN TĂNG GIÁ ĐÁNG KỂ" VÀ TÍNH COHEN'S D
# ══════════════════════════════════════════════════════════════════════════════

def compute_cohens_d_signature(df_feat: pd.DataFrame,
                                symbol: str) -> Optional[dict]:
    """
    Với mỗi mã, tìm các lần tăng đáng kể và tính Cohen's d
    cho từng feature tại T-1, T-3, T-5 trước khi tăng.

    Trả về dict: feature → cohen_d (trung bình across lookbacks)
    """
    close     = df_feat["close"].values.astype(float)
    dates_arr = pd.to_datetime(df_feat["date"]).dt.date.values
    n         = len(close)

    # Tìm signal days (trước khi tăng)
    short_signal_idx = []
    long_signal_idx  = []

    for i in range(n - LONG_FWD):
        # Short move: +5% trong 10 ngày
        fwd10 = (close[i + SHORT_FWD] - close[i]) / (close[i] + 1e-9) * 100
        if fwd10 >= SHORT_THRESH:
            short_signal_idx.append(i)

        # Long move: +8% trong 20 ngày
        fwd20 = (close[i + LONG_FWD] - close[i]) / (close[i] + 1e-9) * 100
        if fwd20 >= LONG_THRESH:
            long_signal_idx.append(i)

    if len(short_signal_idx) < 10 and len(long_signal_idx) < 10:
        return None

    # Tính Cohen's d cho từng feature và từng lookback
    cohen_d_matrix = {}  # feature → {lookback → d}

    for feat in ALL_FEATURES:
        if feat not in df_feat.columns:
            continue
        feat_vals = df_feat[feat].values.astype(float)
        all_valid = feat_vals[np.isfinite(feat_vals)]
        if len(all_valid) < 50:
            continue

        base_mean = np.mean(all_valid)
        base_std  = np.std(all_valid) + 1e-9

        ds_by_lookback = []
        for lb in LOOKBACK_DAYS:
            # Short signals
            short_vals = [feat_vals[i - lb]
                          for i in short_signal_idx
                          if i >= lb and np.isfinite(feat_vals[i - lb])]
            # Long signals
            long_vals  = [feat_vals[i - lb]
                          for i in long_signal_idx
                          if i >= lb and np.isfinite(feat_vals[i - lb])]

            combined = short_vals + long_vals
            if len(combined) < 5:
                continue

            signal_mean = np.mean(combined)
            d = (signal_mean - base_mean) / base_std
            ds_by_lookback.append(d)

        if ds_by_lookback:
            # Lấy max abs Cohen's d across lookbacks (T-1, T-3, T-5)
            cohen_d_matrix[feat] = round(float(np.mean(ds_by_lookback)), 3)

    if not cohen_d_matrix:
        return None

    return {
        "symbol":        symbol,
        "n_short":       len(short_signal_idx),
        "n_long":        len(long_signal_idx),
        "cohen_d":       cohen_d_matrix,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering(signatures: list[dict]) -> dict:
    """
    Cluster symbols dựa trên Cohen's d signature.
    Tự tìm K tối ưu bằng silhouette score.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        from sklearn.impute import SimpleImputer
    except ImportError:
        print("ERROR: pip install scikit-learn")
        sys.exit(1)

    symbols = [s["symbol"] for s in signatures]

    # Build feature matrix: rows=symbols, cols=features
    matrix = []
    for sig in signatures:
        row = [sig["cohen_d"].get(f, 0.0) for f in ALL_FEATURES]
        matrix.append(row)

    X = np.array(matrix)

    # Impute NaN nếu có
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tìm K tối ưu
    best_k     = 2
    best_score = -1
    scores     = {}

    for k in range(K_MIN, min(K_MAX + 1, len(symbols) // MIN_SYMBOLS_PER_CLUSTER + 1)):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X_scaled, labels)
            scores[k] = round(score, 3)
            if score > best_score:
                best_score = score
                best_k     = k
        except Exception:
            continue

    print(f"\n  Silhouette scores: {scores}")
    print(f"  → Optimal K = {best_k} (score={best_score:.3f})")

    # Final clustering với best K
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels   = km_final.fit_predict(X_scaled)

    # Group symbols by cluster
    clusters = {}
    for sym, label in zip(symbols, labels):
        clusters.setdefault(int(label), []).append(sym)

    # Tính centroid Cohen's d cho từng cluster
    cluster_centroids = {}
    for label_id, syms in clusters.items():
        idxs = [symbols.index(s) for s in syms]
        centroid = np.mean(X[idxs], axis=0)
        cluster_centroids[label_id] = dict(zip(ALL_FEATURES, centroid))

    return {
        "best_k":      best_k,
        "best_score":  best_score,
        "all_scores":  scores,
        "clusters":    clusters,
        "centroids":   cluster_centroids,
        "symbols":     symbols,
        "labels":      labels.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: INTERPRET CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════

def interpret_cluster(cluster_id: int, symbols: list[str],
                      centroid: dict) -> dict:
    """
    Phân tích cluster:
    - Dominant features (Cohen's d > threshold)
    - Overlap với MR/MOM
    - Đề xuất tên và signal logic
    """
    # Sort features by abs Cohen's d
    sorted_features = sorted(centroid.items(),
                             key=lambda x: abs(x[1]), reverse=True)

    # Top features (|d| > threshold)
    dominant = [(f, d) for f, d in sorted_features if abs(d) >= COHEN_D_THRESHOLD]

    # Check overlap với MR/MOM
    dominant_names   = {f for f, _ in dominant}
    overlap_mr_mom   = dominant_names & MR_MOM_FEATURES
    new_features     = dominant_names - MR_MOM_FEATURES
    overlap_ratio    = len(overlap_mr_mom) / max(len(dominant_names), 1)

    # Check overlap với existing watchlist symbols
    existing_overlap = len(set(symbols) & EXISTING_ALL)
    existing_ratio   = existing_overlap / max(len(symbols), 1)

    # Phán đoán cluster type
    if overlap_ratio >= 0.6 or existing_ratio >= 0.5:
        cluster_type = "MR_MOM_DUPLICATE"  # trùng với existing
    elif len(dominant) < 2:
        cluster_type = "NOISE"             # không có feature nào nổi bật
    else:
        cluster_type = "NEW_CANDIDATE"     # tiềm năng cluster mới

    # Đề xuất tên dựa vào dominant new features
    suggested_name = _suggest_name(dominant, new_features)

    # Đề xuất signal logic
    suggested_logic = _suggest_signal_logic(dominant, new_features)

    return {
        "cluster_id":      cluster_id,
        "type":            cluster_type,
        "symbols":         symbols,
        "n_symbols":       len(symbols),
        "dominant_feats":  dominant[:8],     # top 8
        "new_features":    list(new_features),
        "overlap_mr_mom":  list(overlap_mr_mom),
        "overlap_ratio":   round(overlap_ratio, 2),
        "suggested_name":  suggested_name,
        "suggested_logic": suggested_logic,
    }


def _suggest_name(dominant: list, new_features: set) -> str:
    """Đề xuất tên cluster dựa vào dominant features."""
    feat_names = {f for f, _ in dominant}

    if "bb_squeeze" in new_features and "consolidation" in new_features:
        return "Breakout (BB Squeeze + Consolidation)"
    if "atr_compression" in new_features and "vol_dry_up" in new_features:
        return "Volatility Compression Breakout"
    if "higher_lows" in new_features and "obv_trend" in new_features:
        return "Accumulation (Higher Lows + OBV)"
    if "dist_52w_high" in new_features:
        d = dict(dominant)
        if d.get("dist_52w_high", 0) > 0:
            return "Near 52W High Breakout"
        else:
            return "Deep Value Recovery"
    if "vol_price_div" in new_features:
        return "Volume/Price Divergence"
    if not new_features:
        return "MR/MOM Overlap (discard)"
    return f"Unknown ({', '.join(list(new_features)[:3])})"


def _suggest_signal_logic(dominant: list, new_features: set) -> dict:
    """Đề xuất regime + triggers dựa vào dominant features."""
    feat_dir = {f: ("HIGH" if d > 0 else "LOW") for f, d in dominant}

    # Chọn regime indicator (feature mạnh nhất trong new_features)
    new_dominant = [(f, d) for f, d in dominant if f in new_features]
    if not new_dominant:
        return {}

    regime_feat, regime_d = new_dominant[0]
    regime_cond = "high" if regime_d > 0 else "low"

    # Triggers: top 2-3 features tiếp theo
    triggers = {}
    for f, d in new_dominant[1:4]:
        triggers[f] = "high" if d > 0 else "low"

    # FWD days: dựa vào pattern type
    if "bb_squeeze" in new_features or "consolidation" in new_features:
        fwd = 15   # breakout thường nhanh hơn MR
    elif "higher_lows" in new_features:
        fwd = 20   # accumulation pattern dài hơn
    else:
        fwd = 10   # default

    return {
        "regime":   {regime_feat: regime_cond},
        "triggers": triggers,
        "fwd_days": fwd,
        "note":     "Cần validate bằng backtest trước khi deploy",
    }


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: BACKTEST cluster mới
# ══════════════════════════════════════════════════════════════════════════════

def backtest_new_cluster(symbol: str, df: pd.DataFrame,
                         logic: dict) -> dict:
    """Backtest signal logic đề xuất cho cluster mới."""
    if not logic or not logic.get("regime") or not logic.get("triggers"):
        return {"n": 0}

    df_feat = compute_features(df)
    if df_feat is None:
        return {"n": 0}

    dates_arr = pd.to_datetime(df_feat["date"]).dt.date.values
    close_arr = df_feat["close"].values.astype(float)
    fwd       = logic["fwd_days"]
    n         = len(df_feat)

    regime_feat, regime_cond = list(logic["regime"].items())[0]
    trigger_config = logic["triggers"]

    pnls = []

    for i in range(100, n - fwd):
        d = dates_arr[i]
        if d < TRAIN_START or d > TRAIN_END:
            continue

        # Expanding threshold
        sub = df_feat.iloc[:i]

        # Check regime
        reg_vals = sub[regime_feat].dropna().values if regime_feat in sub.columns else []
        if len(reg_vals) < 30:
            continue
        pct = 30 if regime_cond == "low" else 70
        reg_thresh = np.percentile(reg_vals, pct)
        val = df_feat[regime_feat].iloc[i]
        if not np.isfinite(val):
            continue
        in_regime = (val <= reg_thresh) if regime_cond == "low" else (val >= reg_thresh)
        if not in_regime:
            continue

        # Check triggers
        triggered = 0
        for t_feat, t_dir in trigger_config.items():
            if t_feat not in df_feat.columns:
                continue
            tv = df_feat[t_feat].iloc[i]
            t_vals = sub[t_feat].dropna().values
            if len(t_vals) < 20 or not np.isfinite(tv):
                continue
            t_pct   = 70 if t_dir == "high" else 30
            t_thresh = np.percentile(t_vals, t_pct)
            if (t_dir == "low" and tv <= t_thresh) or \
               (t_dir == "high" and tv >= t_thresh):
                triggered += 1

        if triggered < 1:  # cluster mới chỉ cần 1 trigger (ít hơn MR/MOM)
            continue

        entry = close_arr[i]
        if i + fwd < n:
            exit_p = close_arr[i + fwd]
            pnls.append((exit_p - entry) / entry * 100)

    if len(pnls) < MIN_TRADES:
        return {"n": len(pnls)}

    pnls   = np.array(pnls)
    wins   = pnls[pnls > 0]
    losses = np.abs(pnls[pnls <= 0])
    gw     = wins.sum()   if len(wins)   > 0 else 0.0
    gl     = losses.sum() if len(losses) > 0 else 1e-9

    return {
        "n":   len(pnls),
        "exp": round(float(np.mean(pnls)), 3),
        "wr":  round(float(len(wins) / len(pnls) * 100), 1),
        "pf":  round(float(gw / gl), 2),
    }


def walk_forward_new_cluster(symbol: str, df: pd.DataFrame,
                              logic: dict) -> dict:
    """Walk Forward cho cluster mới."""
    if not logic:
        return {"status": "NO_LOGIC"}

    df_feat   = compute_features(df)
    if df_feat is None:
        return {"status": "NO_DATA"}

    dates_arr = pd.to_datetime(df_feat["date"]).dt.date.values
    folds     = []
    fold_start = WF_START

    while True:
        train_end = fold_start + timedelta(days=WF_TRAIN_MONTHS * 30)
        test_end  = train_end  + timedelta(days=WF_TEST_MONTHS  * 30)
        if test_end > dates_arr[-1]:
            break

        def _bt(s, e):
            df_tmp = df.copy()
            df_tmp["_date"] = pd.to_datetime(df_tmp["date"]).dt.date
            return backtest_new_cluster(symbol, df_tmp, logic)

        is_m  = _bt(fold_start, train_end)
        oos_m = _bt(train_end,  test_end)

        if is_m["n"] < 5 or oos_m["n"] < 3:
            fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)
            continue

        is_exp  = is_m.get("exp", 0)
        oos_exp = oos_m.get("exp", 0)
        wfe     = (oos_exp / is_exp) if is_exp > 0 else 0.0

        folds.append({
            "period":  f"{train_end.strftime('%Y-%m')}→{test_end.strftime('%Y-%m')}",
            "is_exp":  is_exp,
            "oos_exp": oos_exp,
            "wfe":     round(wfe, 2),
        })
        fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)

    if len(folds) < 3:
        return {"status": "INSUFFICIENT_FOLDS", "folds": folds}

    oos_exps    = [f["oos_exp"] for f in folds]
    wfes        = [f["wfe"]     for f in folds]
    pos_folds   = sum(1 for x in oos_exps if x > 0)
    consistency = round(pos_folds / len(folds) * 100, 1)
    avg_wfe     = round(float(np.mean([w for w in wfes if np.isfinite(w)])), 2)
    avg_oos_exp = round(float(np.mean(oos_exps)), 3)

    status = "PASS" if (avg_wfe >= MIN_WFE and
                        consistency >= MIN_CONSISTENCY and
                        avg_oos_exp > 0) else "FAIL"

    return {
        "status":       status,
        "avg_wfe":      avg_wfe,
        "avg_oos_exp":  avg_oos_exp,
        "consistency":  consistency,
        "n_folds":      len(folds),
        "folds":        folds,
    }


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
    print("DISCOVER CLUSTERS — VN Trader Bot V6")
    print("=" * 65)

    # ── Load universe ─────────────────────────────────────────────────────────
    print(f"\nBước 1: Load universe HOSE (vol > {MIN_VOL_BILLION} tỷ)...")
    universe = _get_universe(load_vn_ohlcv)
    print(f"  {len(universe)} mã đủ volume")

    # ── Tính Cohen's d signature cho từng mã ─────────────────────────────────
    print(f"\nBước 2: Tính feature signature (Cohen's d) cho {len(universe)} mã...")
    signatures = []
    df_cache   = {}

    for idx, sym in enumerate(universe, 1):
        print(f"  [{idx:>3}/{len(universe)}] {sym}...", end=" ", flush=True)
        try:
            df = load_vn_ohlcv(sym, days=2500, min_bars=400)
            if df is None or len(df) < 400:
                print("skip")
                continue
            df_cache[sym] = df

            df_feat = compute_features(df)
            if df_feat is None:
                print("skip (features)")
                continue

            sig = compute_cohens_d_signature(df_feat, sym)
            if sig is None:
                print("skip (no signal days)")
                continue

            signatures.append(sig)
            print(f"OK (short={sig['n_short']}, long={sig['n_long']})")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)

    print(f"\n  → {len(signatures)} mã có đủ signature data")

    if len(signatures) < K_MIN * MIN_SYMBOLS_PER_CLUSTER:
        print("ERROR: Không đủ mã để clustering.")
        sys.exit(1)

    # ── Clustering ────────────────────────────────────────────────────────────
    print(f"\nBước 3: Clustering (K={K_MIN}→{K_MAX})...")
    cluster_result = run_clustering(signatures)

    # ── Interpret từng cluster ────────────────────────────────────────────────
    print(f"\nBước 4: Interpret {cluster_result['best_k']} clusters...")
    interpretations = []

    for cid, syms in cluster_result["clusters"].items():
        centroid = cluster_result["centroids"][cid]
        interp   = interpret_cluster(cid, syms, centroid)
        interpretations.append(interp)

        print(f"\n  {'='*55}")
        print(f"  Cluster {cid}: {interp['suggested_name']}")
        print(f"  Type: {interp['type']} | {len(syms)} mã")
        print(f"  Symbols: {syms}")
        print(f"  Dominant features (Cohen's d):")
        for f, d in interp["dominant_feats"][:6]:
            new_tag = " ★NEW" if f not in MR_MOM_FEATURES else ""
            print(f"    {f:<22}: {d:+.3f}{new_tag}")
        if interp["overlap_mr_mom"]:
            print(f"  Overlap MR/MOM: {interp['overlap_mr_mom']}")
        print(f"  → {interp['type']}")

    # ── Backtest + WF cho NEW_CANDIDATE clusters ──────────────────────────────
    new_candidates = [i for i in interpretations
                      if i["type"] == "NEW_CANDIDATE"
                      and i["n_symbols"] >= MIN_SYMBOLS_PER_CLUSTER]

    print(f"\n\nBước 5: Backtest + Walk Forward cho "
          f"{len(new_candidates)} NEW_CANDIDATE clusters...")

    validated_clusters = []

    for interp in new_candidates:
        cid     = interp["cluster_id"]
        name    = interp["suggested_name"]
        logic   = interp["suggested_logic"]
        symbols = interp["symbols"]

        print(f"\n  [{name}] — {len(symbols)} mã")
        print(f"  Logic đề xuất: {json.dumps(logic, ensure_ascii=False)}")

        sym_results = []
        for sym in symbols:
            if sym not in df_cache:
                continue
            print(f"    {sym}...", end=" ", flush=True)
            bt = backtest_new_cluster(sym, df_cache[sym], logic)
            if bt["n"] < MIN_TRADES:
                print(f"skip (n={bt['n']})")
                continue
            exp = bt.get("exp", 0)
            wr  = bt.get("wr",  0)
            pf  = bt.get("pf",  0)
            if exp >= MIN_EXP and wr >= MIN_WR and pf >= MIN_PF:
                print(f"BT OK (exp={exp:+.2f}% wr={wr:.0f}% pf={pf:.2f})")
                sym_results.append({"sym": sym, "bt": bt})
            else:
                print(f"BT FAIL (exp={exp:+.2f}% wr={wr:.0f}% pf={pf:.2f})")

        if len(sym_results) < MIN_SYMBOLS_PER_CLUSTER // 2:
            print(f"  → Không đủ mã pass backtest, skip WF")
            continue

        # Walk Forward
        print(f"\n  Walk Forward cho {len(sym_results)} mã pass backtest...")
        wf_pass = []
        for sr in sym_results:
            sym = sr["sym"]
            print(f"    {sym} WF...", end=" ", flush=True)
            wf = walk_forward_new_cluster(sym, df_cache[sym], logic)
            status_icon = "✅" if wf["status"] == "PASS" else "❌"
            print(f"{status_icon} {wf['status']} "
                  f"(WFE={wf.get('avg_wfe','?')} "
                  f"consistency={wf.get('consistency','?')}%)")
            if wf["status"] == "PASS":
                wf_pass.append({"sym": sym, "bt": sr["bt"], "wf": wf})

        if len(wf_pass) >= MIN_SYMBOLS_PER_CLUSTER // 2:
            validated_clusters.append({
                "name":       name,
                "cluster_id": cid,
                "logic":      logic,
                "symbols":    [r["sym"] for r in wf_pass],
                "results":    wf_pass,
            })

    # ── Final Report ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"FINAL REPORT")
    print(f"{'='*65}")
    print(f"Universe: {len(universe)} mã | Signatures: {len(signatures)} mã")
    print(f"Clusters found: {cluster_result['best_k']}")
    print(f"New candidates: {len(new_candidates)}")
    print(f"Validated new clusters: {len(validated_clusters)}")

    if validated_clusters:
        print(f"\n✅ CLUSTER MỚI VALIDATED:")
        for vc in validated_clusters:
            print(f"\n  [{vc['name']}]")
            print(f"  Logic: {json.dumps(vc['logic'], ensure_ascii=False)}")
            print(f"  Symbols ({len(vc['symbols'])}): {vc['symbols']}")
            print(f"\n  {'Sym':<7} {'Exp':>6} {'WR':>5} {'PF':>5} "
                  f"{'WFE':>5} {'Consist':>8}")
            print(f"  {'-'*40}")
            for r in vc["results"]:
                bt = r["bt"]
                wf = r["wf"]
                print(f"  {r['sym']:<7} {bt.get('exp',0):>+5.2f}% "
                      f"{bt.get('wr',0):>4.0f}% {bt.get('pf',0):>5.2f} "
                      f"{wf.get('avg_wfe',0):>5.2f} "
                      f"{wf.get('consistency',0):>7.1f}%")

        print(f"\n\n# Thêm vào cluster_scanner.py nếu muốn deploy:")
        for vc in validated_clusters:
            name_clean = vc["name"].upper().replace(" ", "_").replace("(","").replace(")","")
            syms = vc["symbols"]
            print(f'\n{name_clean}_SYMBOLS = {syms}')
            print(f'FWD_DAYS["{vc["name"]}"] = {vc["logic"].get("fwd_days", 15)}')
    else:
        print(f"\n  Không tìm được cluster mới validated.")
        print(f"  Kết luận: MR và MOM đã capture phần lớn edge của thị trường VN.")
        print(f"  → Tập trung mở rộng watchlist trong 2 cluster hiện tại.")


def _get_universe(load_vn_ohlcv) -> list[str]:
    """Lấy HOSE universe có volume > 3 tỷ/ngày (avg 20 phiên)."""
    all_symbols = []
    try:
        from vnstock import Vnstock
        listing = Vnstock().stock(symbol="VCB", source="VCI").listing.symbols_by_exchange()
        hose_df = listing[listing["exchange"].str.upper() == "HOSE"]
        all_symbols = hose_df["symbol"].str.upper().tolist()
    except Exception:
        try:
            from vnstock import Vnstock
            listing = Vnstock().stock(symbol="VCB", source="KBS").listing.symbols_by_exchange()
            hose_df = listing[listing["exchange"].str.upper() == "HOSE"]
            all_symbols = hose_df["symbol"].str.upper().tolist()
        except Exception as e:
            print(f"ERROR listing: {e}")
            return []

    # Filter chứng quyền
    all_symbols = [s for s in all_symbols
                   if not (len(s) > 3 and s[0] == 'C' and s[-1].isdigit())]

    vol_map = {}
    logger.info(f"  Checking volume {len(all_symbols)} symbols...")
    for i, sym in enumerate(all_symbols):
        if i % 50 == 0:
            logger.info(f"  {i}/{len(all_symbols)}...")
        try:
            df = load_vn_ohlcv(sym, days=40, min_bars=20)
            if df is None or len(df) < 20:
                continue
            close = df["close"].values[-20:].astype(float)
            vol   = df["volume"].values[-20:].astype(float)
            avg_vnd = float((vol * close).mean()) * 1000
            if avg_vnd >= MIN_VOL_BILLION * 1e9:
                vol_map[sym] = avg_vnd
        except Exception:
            continue
        time.sleep(0.1)

    return sorted(vol_map, key=vol_map.get, reverse=True)


if __name__ == "__main__":
    run()
