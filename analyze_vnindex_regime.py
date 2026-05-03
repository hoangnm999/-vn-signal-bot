"""
analyze_vnindex_regime.py — VN-Index Regime Filter
Session 30 — VN Trader Bot V6

2 phần:
  A. Entry Regime Filter:
     Tại ngày T, VN-Index features nào phân biệt winning vs losing?
     → Derive entry filter từ distribution (không grid search)

  B. Regime Change Exit:
     Trong holding [T+1..T+N], nếu VN-Index 5d momentum đảo chiều
     → Exit sớm có cải thiện return không?
     "Đảo chiều" = VN-Index 5d return chuyển từ dương sang âm
                   (hoặc âm sang dương tùy cluster)

VN-Index data: symbol "VNINDEX", giá đơn vị nghìn điểm (1.85 = 1850đ)

Chạy:
  python analyze_vnindex_regime.py
  python analyze_vnindex_regime.py MR
  python analyze_vnindex_regime.py MOM
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CLUSTER_ASSIGNMENTS = {
    # Clean watchlist — chỉ mã đã pass WF
    "Mean Reversion": ["DCM", "NKG", "DPM", "HAH", "HCM", "HSG",
                       "DGC", "GAS", "SSI", "VHM"],
    "Momentum":       ["VCB", "BID", "MBB", "MWG", "CTG", "FRT",
                       "REE", "FPT", "GMD", "STB", "PNJ", "TCB"],
}

FWD_CONFIG = {
    "Mean Reversion": 20,
    "Momentum":       10,
}

SIGNAL_CONFIG = {
    "Mean Reversion": {
        "regime_indicator":  "price_vs_sma50",
        "regime_condition":  "low",
        "trigger_indicators":["stoch_k", "volume_spike", "momentum_5d"],
        "trigger_direction": {"stoch_k": "low", "volume_spike": "high",
                              "momentum_5d": "high"},
    },
    "Momentum": {
        "regime_indicator":  "ema_cross",
        "regime_condition":  "high",
        "trigger_indicators":["momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction": {"momentum_5d": "high", "volume_spike": "high",
                              "candle_body": "high"},
    },
}

TRAIN_START  = "2019-01-01"
TRAIN_END    = "2024-12-31"
TRIGGER_PCT  = 70
MIN_TRIGGERS = 2
WIN_THRESH   = 1.0
WF_COOLDOWN  = 5

# VN-Index momentum window cho regime change detection
VNI_MOM_WINDOW = 5   # 5d momentum

# Cohen's d threshold để xác định feature có discriminative power
COHEN_D_MIN = 0.2


# ── VN-Index indicators ───────────────────────────────────────────────────────

def _sma(c, p):
    return pd.Series(c).rolling(p, min_periods=p).mean().values

def _ema(c, span):
    return pd.Series(c).ewm(span=span, adjust=False).mean().values


def compute_vnindex_features(df_vni: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Tính các features của VN-Index theo ngày.
    Trả về dict: feature_name → array (indexed by df_vni row)
    """
    close = df_vni["close"].values.astype(float) * 1000  # convert sang điểm thật
    n     = len(close)

    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    h_prev= np.concatenate([[close[0]], close[:-1]])
    tr    = np.abs(close - h_prev)
    atr14 = _sma(tr, 14)

    # 52-week high (252 ngày)
    high_252 = pd.Series(close).rolling(252, min_periods=50).max().values

    features = {
        # 1. Trend: SMA20 vs SMA50
        "vni_trend":       np.where(sma20 > sma50, 1.0, -1.0),
        "vni_trend_slope": np.array([
            float((sma20[i] - sma50[i]) / (close[i] + 1e-9) * 100)
            if np.isfinite(sma20[i]) and np.isfinite(sma50[i]) else 0.0
            for i in range(n)
        ]),

        # 2. Momentum
        "vni_mom_5d": np.array([
            float((close[i] / close[max(i-5,0)] - 1) * 100)
            for i in range(n)
        ]),
        "vni_mom_20d": np.array([
            float((close[i] / close[max(i-20,0)] - 1) * 100)
            for i in range(n)
        ]),

        # 3. Volatility
        "vni_atr_ratio": np.array([
            float(atr14[i] / (close[i] + 1e-9) * 100)
            if np.isfinite(atr14[i]) else 0.0
            for i in range(n)
        ]),

        # 4. Distance from 52w high
        "vni_dist_high": np.array([
            float((close[i] / high_252[i] - 1) * 100)
            if np.isfinite(high_252[i]) and high_252[i] > 0 else 0.0
            for i in range(n)
        ]),
    }

    return features


# ── Indicator computation cho individual stocks ───────────────────────────────

def _sma_s(c, p):
    return pd.Series(c).rolling(p, min_periods=p).mean().values


def compute_stock_indicators(df: pd.DataFrame) -> list[dict]:
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    vol    = df["volume"].values.astype(float)
    opn    = df["open"].values.astype(float)
    n      = len(df)

    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    sma50  = _sma_s(close, 50)
    vsma20 = _sma_s(vol, 20)

    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr    = _sma_s(tr, 14)

    lo14   = pd.Series(low).rolling(14).min().values
    hi14   = pd.Series(high).rolling(14).max().values
    denom  = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch  = 100 * (close - lo14) / denom

    rows = []
    for i in range(60, n):
        px    = close[i]
        atr_v = atr[i]  if np.isfinite(atr[i])  else px * 0.02
        s50   = sma50[i] if np.isfinite(sma50[i]) else px
        vs20v = vsma20[i] if np.isfinite(vsma20[i]) else vol[i]
        c5    = close[max(i - 5, 0)]

        rows.append({
            "idx":            i,
            "price_vs_sma50": float((px - s50) / (px + 1e-9) * 100),
            "ema_cross":      float((ema12[i] - ema26[i]) / (px + 1e-9) * 100),
            "momentum_5d":    float((px / (c5 + 1e-9) - 1.0) * 100),
            "volume_spike":   float((vol[i] / (vs20v + 1e-9)) - 1.0),
            "stoch_k":        float(stoch[i]),
            "candle_body":    float(np.clip(
                abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
        })
    return rows


# ── Signal detection ──────────────────────────────────────────────────────────

def get_signals(
    cluster: str,
    df_full: pd.DataFrame,
    ind_map: dict,
    close_arr: np.ndarray,
    n_bars: int,
    vni_features: dict,          # aligned với df_full theo date
    vni_mom5_arr: np.ndarray,    # VN-Index 5d momentum array (aligned)
) -> list[dict]:
    """
    Lấy signals + gắn VN-Index features tại ngày T
    và track VN-Index regime trong holding period.
    """
    cfg        = SIGNAL_CONFIG[cluster]
    fwd        = FWD_CONFIG[cluster]
    train_idx  = df_full[
        (df_full["date"] >= TRAIN_START) &
        (df_full["date"] <= TRAIN_END)
    ].index.tolist()

    train_rows = [ind_map[i] for i in train_idx if i in ind_map]
    reg_ind    = cfg["regime_indicator"]
    reg_cond   = cfg["regime_condition"]
    trig_ind   = cfg["trigger_indicators"]
    trig_dir   = cfg["trigger_direction"]

    reg_vals   = [r[reg_ind] for r in train_rows
                  if np.isfinite(r.get(reg_ind, np.nan))]
    reg_thresh = float(np.median(reg_vals)) if reg_vals else 0.0

    trig_thresh = {}
    for t in trig_ind:
        vals = [r[t] for r in train_rows if np.isfinite(r.get(t, np.nan))]
        if not vals:
            continue
        trig_thresh[t] = float(
            np.percentile(vals, 100 - TRIGGER_PCT)
            if trig_dir.get(t) == "low"
            else np.percentile(vals, TRIGGER_PCT)
        )

    signals     = []
    last_signal = None

    for t_idx in train_idx[::3]:
        row = ind_map.get(t_idx)
        if row is None:
            continue
        if last_signal and (t_idx - last_signal) < WF_COOLDOWN:
            continue

        val = row.get(reg_ind, np.nan)
        if not np.isfinite(val):
            continue
        in_regime = (val <= reg_thresh) if reg_cond == "low" else (val > reg_thresh)
        if not in_regime:
            continue

        n_trig = 0
        for t in trig_ind:
            v  = row.get(t, np.nan)
            th = trig_thresh.get(t, np.nan)
            if not (np.isfinite(v) and np.isfinite(th)):
                continue
            if trig_dir.get(t) == "low":
                if v <= th: n_trig += 1
            else:
                if v >= th: n_trig += 1
        if n_trig < MIN_TRIGGERS:
            continue
        if t_idx + fwd >= n_bars:
            continue

        fwd_return = float(
            (close_arr[t_idx + fwd] - close_arr[t_idx]) /
            close_arr[t_idx] * 100
        )

        # VN-Index features tại ngày T
        vni_at_T = {k: float(v[t_idx]) if t_idx < len(v) else 0.0
                    for k, v in vni_features.items()}

        # Track VN-Index 5d momentum trong holding period
        # Detect ngày đầu tiên momentum đảo chiều
        mom_at_T     = vni_at_T.get("vni_mom_5d", 0.0)
        mom_positive = mom_at_T >= 0  # chiều ban đầu
        regime_flip_day = None        # ngày momentum đảo chiều (1-indexed)

        for k in range(1, fwd + 1):
            idx = t_idx + k
            if idx >= len(vni_mom5_arr):
                break
            mom_now = float(vni_mom5_arr[idx])
            now_positive = mom_now >= 0
            if now_positive != mom_positive:
                regime_flip_day = k
                break

        # Return nếu exit tại ngày regime flip
        flip_return = None
        if regime_flip_day is not None:
            fi = t_idx + regime_flip_day
            if fi < n_bars:
                flip_return = float(
                    (close_arr[fi] - close_arr[t_idx]) /
                    close_arr[t_idx] * 100
                )

        date_str = str(df_full["date"].iloc[t_idx])[:10]
        signals.append({
            "t_idx":           t_idx,
            "date":            date_str,
            "half":            "H1" if int(date_str[5:7]) <= 6 else "H2",
            "fwd_return":      round(fwd_return, 2),
            "vni":             {k: round(v, 3) for k, v in vni_at_T.items()},
            "regime_flip_day": regime_flip_day,
            "flip_return":     round(flip_return, 2) if flip_return is not None else None,
        })
        last_signal = t_idx

    return signals


# ── Cohen's d ─────────────────────────────────────────────────────────────────

def cohen_d(a, b):
    if len(a) < 3 or len(b) < 3:
        return 0.0
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2) / (na+nb-2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 1e-9 else 0.0


def calc_metrics(rets):
    if not rets:
        return {"n": 0, "wr": 0, "mean": 0, "pf": 0, "sharpe": 0}
    wins = [r for r in rets if r >= WIN_THRESH]
    loss = [r for r in rets if r < WIN_THRESH]
    pf   = round(sum(wins) / abs(sum(loss)), 2) if loss and sum(loss) != 0 else 99.0
    sh   = round(float(np.mean(rets)) / (float(np.std(rets)) + 1e-9), 3)
    return {
        "n": len(rets), "wr": round(len(wins)/len(rets)*100, 1),
        "mean": round(float(np.mean(rets)), 2), "pf": pf, "sharpe": sh,
    }


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_cluster(cluster: str, all_signals: list[dict]):
    fwd = FWD_CONFIG[cluster]
    n   = len(all_signals)

    print(f"\n{'═'*65}")
    print(f"VN-INDEX REGIME FILTER — {cluster.upper()} (n={n})")
    print(f"{'═'*65}")

    wins = [s for s in all_signals if s["fwd_return"] >= WIN_THRESH]
    loss = [s for s in all_signals if s["fwd_return"] < WIN_THRESH]

    # ══════════════════════════════════════════════════════════════
    # PHẦN A: ENTRY REGIME FILTER
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"PHẦN A — ENTRY REGIME: VN-Index features tại ngày T")
    print(f"{'─'*65}")
    print(f"  Winning signals: {len(wins)} | Losing signals: {len(loss)}")

    vni_feature_names = [
        "vni_trend_slope", "vni_mom_5d", "vni_mom_20d",
        "vni_atr_ratio",   "vni_dist_high",
    ]

    feature_analysis = []
    for feat in vni_feature_names:
        w_vals = [s["vni"].get(feat, 0) for s in wins]
        l_vals = [s["vni"].get(feat, 0) for s in loss]
        if not w_vals or not l_vals:
            continue

        d = cohen_d(np.array(w_vals), np.array(l_vals))
        feature_analysis.append({
            "feature":   feat,
            "cohen_d":   round(d, 3),
            "abs_d":     round(abs(d), 3),
            "mean_win":  round(float(np.mean(w_vals)), 3),
            "mean_loss": round(float(np.mean(l_vals)), 3),
            "direction": "Win>Loss" if d > 0 else "Win<Loss",
        })

    feature_analysis.sort(key=lambda x: -x["abs_d"])

    print(f"\n  {'Feature':<20} {'Cohen_d':>8} {'Mean_Win':>10} "
          f"{'Mean_Loss':>10}  Direction  Strength")
    print(f"  {'─'*65}")
    for r in feature_analysis:
        strength = ("★★ Manh" if r["abs_d"] >= 0.5 else
                    "★  Trung binh" if r["abs_d"] >= 0.3 else
                    "·  Yeu" if r["abs_d"] >= 0.2 else
                    "   Noise")
        print(f"  {r['feature']:<20} {r['cohen_d']:>+8.3f} "
              f"{r['mean_win']:>+10.3f} {r['mean_loss']:>+10.3f}  "
              f"{r['direction']:<10} {strength}")

    # Identify strong features
    strong_features = [r for r in feature_analysis if r["abs_d"] >= 0.2]

    if not strong_features:
        print(f"\n  ❌ Không có VN-Index feature nào phân biệt được Win vs Loss")
        print(f"  → Entry regime filter không có giá trị cho {cluster}")
    else:
        print(f"\n  → {len(strong_features)} features có discriminative power")

        # Derive filter từ strongest feature
        best = strong_features[0]
        feat = best["feature"]

        all_vals = [s["vni"].get(feat, 0) for s in all_signals]
        threshold = float(np.median(all_vals))

        # Test filter
        if best["direction"] == "Win>Loss":
            # Win có giá trị cao hơn → filter: chỉ trade khi feature >= threshold
            pass_sigs  = [s for s in all_signals if s["vni"].get(feat, 0) >= threshold]
            block_sigs = [s for s in all_signals if s["vni"].get(feat, 0) < threshold]
            filter_dir = f">= {threshold:.3f}"
        else:
            # Win có giá trị thấp hơn → filter: chỉ trade khi feature <= threshold
            pass_sigs  = [s for s in all_signals if s["vni"].get(feat, 0) <= threshold]
            block_sigs = [s for s in all_signals if s["vni"].get(feat, 0) > threshold]
            filter_dir = f"<= {threshold:.3f}"

        print(f"\n  DERIVE ENTRY FILTER từ strongest feature:")
        print(f"  Feature: {feat}  |  Filter: {filter_dir}")
        print(f"  (Threshold = median của training distribution)")

        base_m  = calc_metrics([s["fwd_return"] for s in all_signals])
        pass_m  = calc_metrics([s["fwd_return"] for s in pass_sigs])
        block_m = calc_metrics([s["fwd_return"] for s in block_sigs])

        print(f"\n  {'Group':<15} {'n':>5} {'WR':>7} {'MeanExp':>9} "
              f"{'PF':>6} {'Sharpe':>7}")
        print(f"  {'─'*50}")
        for label, m in [("Baseline", base_m),
                          ("Pass filter", pass_m),
                          ("Block filter", block_m)]:
            print(f"  {label:<15} {m['n']:>5} {m['wr']:>6.1f}% "
                  f"{m['mean']:>+8.2f}% {m['pf']:>6.2f} "
                  f"{m['sharpe']:>+6.3f}")

        improvement = pass_m["mean"] - base_m["mean"]
        recall      = pass_m["n"] / base_m["n"] * 100

        print(f"\n  Filter impact:")
        print(f"  Exp:  {base_m['mean']:+.2f}% → {pass_m['mean']:+.2f}% "
              f"({improvement:+.2f}%)")
        print(f"  Recall: {recall:.1f}% signals được giữ lại")

        if improvement > 0.3 and recall >= 50:
            print(f"  ✅ Filter có giá trị: cải thiện Exp và giữ ≥50% signals")
        elif improvement > 0.1:
            print(f"  · Filter cải thiện nhẹ nhưng recall đủ")
        else:
            print(f"  ❌ Filter không cải thiện đủ — không đáng dùng")

        # Test tất cả features strong cùng lúc (AND combination)
        if len(strong_features) >= 2:
            print(f"\n  COMBINE ALL STRONG FEATURES (AND logic):")
            combo_pass = all_signals.copy()
            for r in strong_features:
                f2    = r["feature"]
                vals2 = [s["vni"].get(f2, 0) for s in all_signals]
                th2   = float(np.median(vals2))
                if r["direction"] == "Win>Loss":
                    combo_pass = [s for s in combo_pass
                                  if s["vni"].get(f2, 0) >= th2]
                else:
                    combo_pass = [s for s in combo_pass
                                  if s["vni"].get(f2, 0) <= th2]

            combo_m  = calc_metrics([s["fwd_return"] for s in combo_pass])
            combo_recall = combo_m["n"] / base_m["n"] * 100 if base_m["n"] else 0

            print(f"  n={combo_m['n']} ({combo_recall:.1f}% recall) | "
                  f"WR={combo_m['wr']:.1f}% | "
                  f"Exp={combo_m['mean']:+.2f}% | "
                  f"PF={combo_m['pf']:.2f}")

            if combo_m["mean"] > pass_m["mean"] and combo_recall >= 30:
                print(f"  ✅ Combination tốt hơn single feature")
            elif combo_recall < 30:
                print(f"  ⚠️  Recall quá thấp ({combo_recall:.0f}%) — too restrictive")
            else:
                print(f"  · Single feature đủ tốt")

    # ══════════════════════════════════════════════════════════════
    # PHẦN B: REGIME CHANGE EXIT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"PHẦN B — REGIME CHANGE EXIT")
    print(f"  Định nghĩa: VN-Index 5d momentum đảo chiều trong holding")
    print(f"{'─'*65}")

    has_flip = [s for s in all_signals if s["regime_flip_day"] is not None]
    no_flip  = [s for s in all_signals if s["regime_flip_day"] is None]

    print(f"\n  Signals có regime flip:  {len(has_flip)} ({len(has_flip)/n*100:.1f}%)")
    print(f"  Signals không có flip:   {len(no_flip)}  ({len(no_flip)/n*100:.1f}%)")

    if has_flip:
        flip_days = [s["regime_flip_day"] for s in has_flip]
        print(f"\n  Flip day distribution:")
        print(f"  Mean={np.mean(flip_days):.1f}d | "
              f"Median={np.median(flip_days):.1f}d | "
              f"p25={np.percentile(flip_days,25):.1f}d | "
              f"p75={np.percentile(flip_days,75):.1f}d")

    # So sánh return: exit sớm khi flip vs giữ đến T+N
    print(f"\n  SO SÁNH RETURN (chỉ nhóm có regime flip):")
    print(f"  {'Strategy':<30} {'n':>5} {'WR':>7} {'MeanExp':>9} "
          f"{'PF':>6} {'Sharpe':>7}")
    print(f"  {'─'*60}")

    # Baseline: giữ đến T+N (tất cả signals)
    base_all = calc_metrics([s["fwd_return"] for s in all_signals])
    print(f"  {'ALL — Hold to T+N':<30} {base_all['n']:>5} "
          f"{base_all['wr']:>6.1f}% {base_all['mean']:>+8.2f}% "
          f"{base_all['pf']:>6.2f} {base_all['sharpe']:>+6.3f}")

    if has_flip:
        # Nhóm có flip: giữ T+N
        flip_hold = calc_metrics([s["fwd_return"] for s in has_flip])
        print(f"  {'Has flip — Hold to T+N':<30} {flip_hold['n']:>5} "
              f"{flip_hold['wr']:>6.1f}% {flip_hold['mean']:>+8.2f}% "
              f"{flip_hold['pf']:>6.2f} {flip_hold['sharpe']:>+6.3f}")

        # Nhóm có flip: exit khi flip
        flip_exit_rets = [s["flip_return"] for s in has_flip
                          if s["flip_return"] is not None]
        if flip_exit_rets:
            flip_exit = calc_metrics(flip_exit_rets)
            print(f"  {'Has flip — Exit on flip':<30} {flip_exit['n']:>5} "
                  f"{flip_exit['wr']:>6.1f}% {flip_exit['mean']:>+8.2f}% "
                  f"{flip_exit['pf']:>6.2f} {flip_exit['sharpe']:>+6.3f}")

            # Nhóm không có flip
            no_flip_m = calc_metrics([s["fwd_return"] for s in no_flip])
            if no_flip:
                print(f"  {'No flip — Hold to T+N':<30} {no_flip_m['n']:>5} "
                      f"{no_flip_m['wr']:>6.1f}% {no_flip_m['mean']:>+8.2f}% "
                      f"{no_flip_m['pf']:>6.2f} {no_flip_m['sharpe']:>+6.3f}")

            # Mixed: exit on flip nếu có, else hold
            mixed_rets = []
            for s in all_signals:
                if s["regime_flip_day"] and s["flip_return"] is not None:
                    mixed_rets.append(s["flip_return"])
                else:
                    mixed_rets.append(s["fwd_return"])
            mixed_m = calc_metrics(mixed_rets)
            print(f"  {'MIXED — Exit if flip, else hold':<30} {mixed_m['n']:>5} "
                  f"{mixed_m['wr']:>6.1f}% {mixed_m['mean']:>+8.2f}% "
                  f"{mixed_m['pf']:>6.2f} {mixed_m['sharpe']:>+6.3f}")

            # Verdict
            print(f"\n  VERDICT Regime Change Exit:")
            delta = mixed_m["mean"] - base_all["mean"]
            if delta > 0.2:
                print(f"  ✅ Exit on regime flip cải thiện: "
                      f"{base_all['mean']:+.2f}% → {mixed_m['mean']:+.2f}% "
                      f"({delta:+.2f}%)")
                print(f"  → Tích hợp vào exit logic: exit sớm khi VNI mom5 đảo chiều")
            elif delta > 0:
                print(f"  · Cải thiện nhẹ ({delta:+.2f}%) — có thể tích hợp")
            else:
                print(f"  ❌ Exit on regime flip không cải thiện ({delta:+.2f}%)")
                print(f"  → Giữ nguyên Time Stop T+{fwd}")

        # Breakdown flip by timing (sớm vs muộn)
        print(f"\n  FLIP TIMING BREAKDOWN:")
        mid = fwd // 2
        early_flip = [s for s in has_flip
                      if s["regime_flip_day"] and s["regime_flip_day"] <= mid]
        late_flip  = [s for s in has_flip
                      if s["regime_flip_day"] and s["regime_flip_day"] > mid]

        for label, grp in [(f"Early flip (day≤{mid})", early_flip),
                           (f"Late flip (day>{mid})",  late_flip)]:
            if not grp:
                continue
            # Return nếu exit
            exit_rets = [s["flip_return"] for s in grp
                         if s["flip_return"] is not None]
            hold_rets = [s["fwd_return"] for s in grp]
            if exit_rets and hold_rets:
                em = calc_metrics(exit_rets)
                hm = calc_metrics(hold_rets)
                print(f"  {label:<25}: Exit={em['mean']:+.2f}% vs "
                      f"Hold={hm['mean']:+.2f}% → "
                      + ("Exit better ✅" if em["mean"] > hm["mean"]
                         else "Hold better"))

    # ── Tóm tắt ──────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"TÓM TẮT — {cluster}")
    print(f"{'═'*65}")

    # Entry filter verdict
    if strong_features:
        best_feat = strong_features[0]
        print(f"\n  Entry Filter:")
        print(f"  Feature tốt nhất: {best_feat['feature']} "
              f"(Cohen's d={best_feat['cohen_d']:+.3f})")
        if best_feat["abs_d"] >= 0.3:
            print(f"  ✅ Đáng tích hợp vào entry condition")
        else:
            print(f"  · Weak signal — theo dõi thêm trên OOS")
    else:
        print(f"\n  Entry Filter: ❌ Không có feature đủ mạnh")

    # Exit flip verdict
    if has_flip:
        flip_pct = len(has_flip) / n * 100
        print(f"\n  Regime Change Exit:")
        print(f"  {flip_pct:.0f}% signals có regime flip trong holding")
        flip_exit_r = [s["flip_return"] for s in has_flip
                       if s["flip_return"] is not None]
        flip_hold_r = [s["fwd_return"] for s in has_flip]
        if flip_exit_r and flip_hold_r:
            exit_exp = np.mean(flip_exit_r)
            hold_exp = np.mean(flip_hold_r)
            if exit_exp > hold_exp + 0.2:
                print(f"  ✅ Exit early khi flip tốt hơn hold "
                      f"({exit_exp:+.2f}% vs {hold_exp:+.2f}%)")
            else:
                print(f"  · Flip exit không cải thiện đáng kể — giữ Time Stop")

    return {
        "cluster":        cluster,
        "n":              n,
        "strong_features":strong_features,
        "has_flip_pct":   round(len(has_flip) / n * 100, 1) if n else 0,
    }


# ── Load data ─────────────────────────────────────────────────────────────────

def load_vnindex() -> tuple[pd.DataFrame, dict, np.ndarray] | None:
    """Load VN-Index và tính features. Trả về (df, features_dict, mom5_arr)."""
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv("VNINDEX", days=2500, min_bars=300)
        df["date"] = pd.to_datetime(df["date"])
        df = df.reset_index(drop=True)
        print(f"  VN-Index loaded: {len(df)} bars "
              f"({str(df['date'].iloc[0])[:10]} → "
              f"{str(df['date'].iloc[-1])[:10]})")
        features = compute_vnindex_features(df)
        mom5_arr = features["vni_mom_5d"]
        return df, features, mom5_arr
    except Exception as e:
        print(f"  ERROR load VNINDEX: {e}")
        return None


def align_vni_to_stock(
    df_stock: pd.DataFrame,
    df_vni: pd.DataFrame,
    vni_features: dict,
    vni_mom5_arr: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """
    Align VN-Index features theo date của df_stock.
    Với mỗi ngày trong df_stock, lấy VN-Index value của ngày đó
    (hoặc ngày trading gần nhất trước đó nếu không khớp).
    """
    # Build date → vni_row_idx map
    vni_date_map = {}
    for i, row in df_vni.iterrows():
        d = str(row["date"])[:10]
        vni_date_map[d] = i

    n_stock = len(df_stock)

    # Aligned arrays — same length as df_stock
    aligned = {k: np.zeros(n_stock) for k in vni_features}
    aligned_mom5 = np.zeros(n_stock)

    last_vni_idx = 0
    for i, row in df_stock.iterrows():
        d = str(row["date"])[:10]
        if d in vni_date_map:
            last_vni_idx = vni_date_map[d]
        # Dùng last available (forward fill)
        for k, arr in vni_features.items():
            if last_vni_idx < len(arr):
                aligned[k][i] = arr[last_vni_idx]
        if last_vni_idx < len(vni_mom5_arr):
            aligned_mom5[i] = vni_mom5_arr[last_vni_idx]

    return aligned, aligned_mom5


def load_symbol(symbol: str, cluster: str,
                df_vni: pd.DataFrame,
                vni_features: dict,
                vni_mom5_arr: np.ndarray) -> list[dict]:
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=300)
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"  ERROR {symbol}: {e}")
        return []

    df_full   = df.reset_index(drop=True)
    n_bars    = len(df_full)
    close_arr = df_full["close"].values.astype(float)

    # Align VN-Index to stock dates
    aligned_feat, aligned_mom5 = align_vni_to_stock(
        df_full, df_vni, vni_features, vni_mom5_arr
    )

    all_rows = compute_stock_indicators(df_full)
    ind_map  = {r["idx"]: r for r in all_rows}

    signals = get_signals(
        cluster, df_full, ind_map, close_arr, n_bars,
        aligned_feat, aligned_mom5
    )
    return signals


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = [a.upper() for a in sys.argv[1:]]
    run_mr  = not args or "MR"  in args
    run_mom = not args or "MOM" in args

    print(f"{'='*65}")
    print(f"VN-INDEX REGIME FILTER ANALYSIS — VN Trader Bot V6")
    print(f"Entry filter + Regime Change Exit (VNI mom5 flip)")
    print(f"{'='*65}")

    # Load VN-Index một lần dùng chung
    print(f"\nLoading VN-Index...")
    result = load_vnindex()
    if result is None:
        print("Cannot load VNINDEX — exit")
        return
    df_vni, vni_features, vni_mom5_arr = result

    all_output = {}

    for cluster, run_flag in [
        ("Mean Reversion", run_mr),
        ("Momentum",       run_mom),
    ]:
        if not run_flag:
            continue

        symbols = CLUSTER_ASSIGNMENTS[cluster]
        fwd     = FWD_CONFIG[cluster]
        print(f"\nLoading {cluster} ({len(symbols)} ma, FWD={fwd}d)...")

        all_signals = []
        for sym in symbols:
            print(f"  {sym}...", end=" ", flush=True)
            sigs = load_symbol(sym, cluster, df_vni, vni_features, vni_mom5_arr)
            print(f"{len(sigs)} signals")
            all_signals.extend(sigs)

        if not all_signals:
            print(f"  Khong co signal")
            continue

        print(f"  Total: {len(all_signals)} signals")
        result_cl = analyze_cluster(cluster, all_signals)
        all_output[cluster] = result_cl

    # Save
    Path("vnindex_regime_results.json").write_text(
        json.dumps(all_output, indent=2, ensure_ascii=False, default=str)
    )
    print(f"\n  Da luu: vnindex_regime_results.json")

    # Cross-cluster summary
    if len(all_output) == 2:
        print(f"\n{'═'*65}")
        print(f"CROSS-CLUSTER SUMMARY")
        print(f"{'═'*65}")
        for cl, r in all_output.items():
            sf = r.get("strong_features", [])
            print(f"\n  {cl}:")
            if sf:
                print(f"  Entry filter: {sf[0]['feature']} "
                      f"d={sf[0]['cohen_d']:+.3f} "
                      f"({'tích hợp' if sf[0]['abs_d'] >= 0.3 else 'theo dõi'})")
            else:
                print(f"  Entry filter: không có feature đủ mạnh")
            print(f"  Regime flip: {r['has_flip_pct']:.0f}% signals")


if __name__ == "__main__":
    main()
