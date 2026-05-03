"""
analyze_mae_mfe.py — MAE/MFE Analysis + SL/TP Derivation
Session 30 — VN Trader Bot V6

Logic:
  Với mỗi signal tại ngày T:
    MAE = min(low[T+1..T+N] - entry) / entry × 100  ← worst drawdown
    MFE = max(high[T+1..T+N] - entry) / entry × 100 ← best run-up

  SL = median MAE của cluster (per H1/H2)
  TP = median MFE của cluster (per H1/H2)

  Sau đó simulate lại với SL/TP:
    - Nếu giá hit SL trước TP → loss = SL
    - Nếu giá hit TP trước SL → win = TP
    - Nếu không hit cả hai → dùng forward return ngày T+N

  So sánh:
    A. Forward return đơn giản (hiện tại)
    B. Với SL/TP từ median MAE/MFE
    → Cái nào có EV và Sharpe tốt hơn?

Chạy:
  python analyze_mae_mfe.py
  python analyze_mae_mfe.py MOM
  python analyze_mae_mfe.py MR
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Cluster config ────────────────────────────────────────────────────────────
CLUSTER_ASSIGNMENTS = {
    # Clean watchlist — chỉ mã đã pass WF (WF_Exp > 0 và WFE > 0)
    # Loại: VND, NVL (MR WF_Exp âm) | PC1, HPG, KDH (MOM WFE âm)
    "Mean Reversion": ['DCM', 'NKG', 'DPM', 'HAH', 'HCM', 'HSG', 'DGC', 'GAS', 'SSI', 'VHM'],
    "Momentum":       ['VCB', 'BID', 'MBB', 'MWG', 'CTG', 'FRT', 'REE', 'FPT', 'GMD', 'STB', 'PNJ', 'TCB'],
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


# ── Indicator computation ─────────────────────────────────────────────────────

def _ema(c, span):
    return pd.Series(c).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c).rolling(p, min_periods=p).mean().values


def compute_indicators(df: pd.DataFrame) -> list[dict]:
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    vol    = df["volume"].values.astype(float)
    opn    = df["open"].values.astype(float)
    n      = len(df)

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

    rows = []
    for i in range(60, n):
        px    = close[i]
        atr_v = atr[i]  if np.isfinite(atr[i])  else px * 0.02
        s20   = sma20[i] if np.isfinite(sma20[i]) else px
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
            "atr14_pct":      float(atr_v / (px + 1e-9) * 100),
        })
    return rows


# ── Signal detection ──────────────────────────────────────────────────────────

def get_signals(
    symbol: str,
    cluster: str,
    df_full: pd.DataFrame,
    ind_map: dict,
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    n_bars: int,
) -> list[dict]:
    """
    Lấy tất cả signals trên training + tính MAE/MFE cho từng signal.
    Entry = close ngày T (giả định mua cuối phiên T).
    MAE/MFE dùng high/low của [T+1..T+fwd].
    """
    cfg      = SIGNAL_CONFIG[cluster]
    fwd      = FWD_CONFIG[cluster]
    train_idx= df_full[
        (df_full["date"] >= TRAIN_START) &
        (df_full["date"] <= TRAIN_END)
    ].index.tolist()

    train_rows  = [ind_map[i] for i in train_idx if i in ind_map]
    reg_ind     = cfg["regime_indicator"]
    reg_cond    = cfg["regime_condition"]
    trig_ind    = cfg["trigger_indicators"]
    trig_dir    = cfg["trigger_direction"]

    # Thresholds từ training
    reg_vals   = [r[reg_ind] for r in train_rows
                  if np.isfinite(r.get(reg_ind, np.nan))]
    reg_thresh = float(np.median(reg_vals)) if reg_vals else 0.0

    trig_thresh = {}
    for t in trig_ind:
        vals = [r[t] for r in train_rows if np.isfinite(r.get(t, np.nan))]
        if not vals:
            continue
        if trig_dir.get(t, "high") == "low":
            trig_thresh[t] = float(np.percentile(vals, 100 - TRIGGER_PCT))
        else:
            trig_thresh[t] = float(np.percentile(vals, TRIGGER_PCT))

    signals     = []
    last_signal = None

    for t_idx in train_idx[::3]:
        row = ind_map.get(t_idx)
        if row is None:
            continue

        if last_signal and (t_idx - last_signal) < WF_COOLDOWN:
            continue

        # Regime
        val = row.get(reg_ind, np.nan)
        if not np.isfinite(val):
            continue
        in_regime = (val <= reg_thresh) if reg_cond == "low" else (val > reg_thresh)
        if not in_regime:
            continue

        # Triggers
        n_trig = 0
        for t in trig_ind:
            v  = row.get(t, np.nan)
            th = trig_thresh.get(t, np.nan)
            if not (np.isfinite(v) and np.isfinite(th)):
                continue
            if trig_dir.get(t, "high") == "low":
                if v <= th: n_trig += 1
            else:
                if v >= th: n_trig += 1

        if n_trig < MIN_TRIGGERS:
            continue

        if t_idx + fwd >= n_bars:
            continue

        entry = close_arr[t_idx]
        if entry <= 0:
            continue

        # MAE/MFE từ high/low của [T+1..T+fwd]
        window_high = high_arr[t_idx + 1: t_idx + fwd + 1]
        window_low  = low_arr[t_idx + 1:  t_idx + fwd + 1]

        if len(window_high) == 0:
            continue

        mae = float((np.min(window_low)  - entry) / entry * 100)
        mfe = float((np.max(window_high) - entry) / entry * 100)
        fwd_return = float((close_arr[t_idx + fwd] - entry) / entry * 100)

        # Date info cho H1/H2
        date_str = str(df_full["date"].iloc[t_idx])[:10]
        month    = int(date_str[5:7])
        half     = "H1" if month <= 6 else "H2"
        year     = date_str[:4]

        signals.append({
            "t_idx":      t_idx,
            "date":       date_str,
            "half":       half,
            "year":       year,
            "entry":      round(entry, 2),
            "mae":        round(mae, 2),
            "mfe":        round(mfe, 2),
            "fwd_return": round(fwd_return, 2),
            "atr14_pct":  round(row.get("atr14_pct", 0), 2),
        })
        last_signal = t_idx

    return signals


# ── Simulate với SL/TP ────────────────────────────────────────────────────────

def simulate_sl_tp(
    signals: list[dict],
    sl_pct: float,   # âm, ví dụ -4.5
    tp_pct: float,   # dương, ví dụ +7.2
    high_arr: np.ndarray,
    low_arr:  np.ndarray,
    close_arr: np.ndarray,
    n_bars: int,
    fwd: int,
) -> list[dict]:
    """
    Simulate từng signal với SL/TP cụ thể.
    Dùng daily high/low để xác định ngày nào hit SL hoặc TP.
    Giả định: nếu cùng ngày hit cả SL và TP → SL (conservative).
    """
    results = []
    for s in signals:
        t_idx = s["t_idx"]
        entry = s["entry"]
        sl_price = entry * (1 + sl_pct / 100)
        tp_price = entry * (1 + tp_pct / 100)

        hit_sl = None
        hit_tp = None
        actual = s["fwd_return"]

        for k in range(1, fwd + 1):
            idx = t_idx + k
            if idx >= n_bars:
                break
            day_low  = low_arr[idx]
            day_high = high_arr[idx]

            # Check hit
            sl_hit = day_low  <= sl_price
            tp_hit = day_high >= tp_price

            if sl_hit and tp_hit:
                # Cùng ngày → SL (conservative)
                hit_sl = k
                break
            elif sl_hit:
                hit_sl = k
                break
            elif tp_hit:
                hit_tp = k
                break

        if hit_sl is not None:
            ret = sl_pct
            outcome = "SL"
        elif hit_tp is not None:
            ret = tp_pct
            outcome = "TP"
        else:
            ret = actual
            outcome = "HOLD"

        results.append({
            **s,
            "sl_pct":    sl_pct,
            "tp_pct":    tp_pct,
            "sim_return":round(ret, 2),
            "outcome":   outcome,
        })
    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def calc_metrics(rets: list[float], label: str = "") -> dict:
    if not rets:
        return {"n": 0, "wr": 0, "mean_exp": 0, "pf": 0,
                "sharpe": 0, "max_dd": 0, "label": label}
    wins = [r for r in rets if r >= WIN_THRESH]
    loss = [r for r in rets if r < WIN_THRESH]
    pf   = round(sum(wins) / abs(sum(loss)), 2) if loss and sum(loss) != 0 else 99.0
    sharpe = round(float(np.mean(rets)) / (float(np.std(rets)) + 1e-9), 3)
    return {
        "label":    label,
        "n":        len(rets),
        "wr":       round(len(wins) / len(rets) * 100, 1),
        "mean_exp": round(float(np.mean(rets)), 2),
        "pf":       round(pf, 2),
        "sharpe":   round(sharpe, 3),
        "max_dd":   round(float(np.min(rets)), 1),
    }


def percentile_str(vals: list[float], label: str) -> str:
    if not vals:
        return f"{label}: (no data)"
    p = [np.percentile(vals, q) for q in [10, 25, 50, 75, 90]]
    return (f"{label}: "
            f"p10={p[0]:+.1f}% p25={p[1]:+.1f}% "
            f"p50={p[2]:+.1f}% p75={p[3]:+.1f}% p90={p[4]:+.1f}%")


# ── Per-symbol analysis ───────────────────────────────────────────────────────

def analyze_symbol(symbol: str, cluster: str) -> dict | None:
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=300)
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"    [{symbol}] ERROR: {e}")
        return None

    df_full   = df.reset_index(drop=True)
    n_bars    = len(df_full)
    close_arr = df_full["close"].values.astype(float)
    high_arr  = df_full["high"].values.astype(float)
    low_arr   = df_full["low"].values.astype(float)

    all_rows = compute_indicators(df_full)
    ind_map  = {r["idx"]: r for r in all_rows}

    signals = get_signals(
        symbol, cluster, df_full, ind_map,
        close_arr, high_arr, low_arr, n_bars
    )

    if not signals:
        return None

    return {
        "symbol":  symbol,
        "cluster": cluster,
        "signals": signals,
    }


# ── Cluster-level MAE/MFE analysis ───────────────────────────────────────────

def analyze_cluster(cluster: str, all_signals: list[dict]):
    fwd = FWD_CONFIG[cluster]

    print(f"\n{'═'*65}")
    print(f"MAE/MFE ANALYSIS — {cluster.upper()} (FWD={fwd}d)")
    print(f"{'═'*65}")

    # ── 1. Overall distribution ───────────────────────────────────────────────
    mae_all = [s["mae"] for s in all_signals]
    mfe_all = [s["mfe"] for s in all_signals]
    fwd_all = [s["fwd_return"] for s in all_signals]

    print(f"\nOVERALL (n={len(all_signals)}):")
    print(f"  {percentile_str(mae_all, 'MAE')}")
    print(f"  {percentile_str(mfe_all, 'MFE')}")
    print(f"  {percentile_str(fwd_all, 'FWD')}")

    # ── 2. H1 vs H2 MAE/MFE breakdown ────────────────────────────────────────
    h1_sigs = [s for s in all_signals if s["half"] == "H1"]
    h2_sigs = [s for s in all_signals if s["half"] == "H2"]

    print(f"\nH1 vs H2 MAE/MFE BREAKDOWN:")
    print(f"  {'':6} {'n':>5} {'MAE_p50':>9} {'MAE_p75':>9} "
          f"{'MFE_p50':>9} {'MFE_p75':>9} {'FWD_mean':>9}")
    print(f"  {'─'*60}")

    for label, sigs in [("H1", h1_sigs), ("H2", h2_sigs), ("ALL", all_signals)]:
        if not sigs:
            continue
        maes = [s["mae"] for s in sigs]
        mfes = [s["mfe"] for s in sigs]
        fwds = [s["fwd_return"] for s in sigs]
        print(
            f"  {label:<6} {len(sigs):>5} "
            f"{np.percentile(maes, 50):>+8.2f}% "
            f"{np.percentile(maes, 75):>+8.2f}% "
            f"{np.percentile(mfes, 50):>+8.2f}% "
            f"{np.percentile(mfes, 75):>+8.2f}% "
            f"{np.mean(fwds):>+8.2f}%"
        )

    # ── 3. Derive SL/TP từ median MAE/MFE ────────────────────────────────────
    sl_tp = {}
    print(f"\nDERIVED SL/TP (từ median MAE/MFE):")
    print(f"  {'':6} {'SL (median MAE)':>16} {'TP (median MFE)':>16}  R:R ratio")
    print(f"  {'─'*55}")

    for label, sigs in [("H1", h1_sigs), ("H2", h2_sigs), ("ALL", all_signals)]:
        if not sigs:
            continue
        maes = [s["mae"] for s in sigs]
        mfes = [s["mfe"] for s in sigs]
        sl   = round(float(np.percentile(maes, 50)), 2)
        tp   = round(float(np.percentile(mfes, 50)), 2)
        rr   = round(tp / abs(sl), 2) if sl != 0 else 0.0
        sl_tp[label] = {"sl": sl, "tp": tp}
        print(f"  {label:<6} {sl:>+14.2f}%  {tp:>+14.2f}%  {rr:.2f}x")

    # ── 4. Simulate với SL/TP ─────────────────────────────────────────────────
    print(f"\nSIMULATION — Forward Return vs SL/TP:")
    print(f"  {'Scenario':<25} {'n':>5} {'WR':>7} {'MeanExp':>9} "
          f"{'PF':>6} {'Sharpe':>7} {'MaxDD':>8}")
    print(f"  {'─'*65}")

    # Cần high/low arrays — load lại per-symbol trong simulate
    # Ở đây simulate đơn giản dùng MAE/MFE đã tính
    # Nếu MAE <= SL → hit SL, nếu MFE >= TP → hit TP
    # Nếu cả hai → SL (conservative, vì không biết thứ tự)

    # Tính thêm p25 MAE làm Catastrophic SL
    mae_all_vals = [s["mae"] for s in all_signals]
    sl_catastrophic = round(float(np.percentile(mae_all_vals, 25)), 2)
    tp_median       = sl_tp.get("ALL", {}).get("tp", +7.0)

    print(f"\n  Catastrophic SL (p25 MAE) = {sl_catastrophic:+.2f}%")
    print(f"  TP median MFE             = {tp_median:+.2f}%")

    sim_results = {}
    for scenario_label, sigs in [
        ("ALL signals", all_signals),
        ("H1 signals",  h1_sigs),
        ("H2 signals",  h2_sigs),
    ]:
        if not sigs:
            continue

        # 4a. Baseline (forward return — Time Stop only)
        base_rets = [s["fwd_return"] for s in sigs]
        base_m    = calc_metrics(base_rets, f"Baseline (Time Stop)")

        # 4b. Catastrophic SL p50 + TP median (cũ — để so sánh)
        sl_p50 = sl_tp.get("ALL", {}).get("sl", -5.0)
        tp_p50 = sl_tp.get("ALL", {}).get("tp", +7.0)

        sim_rets_p50 = []
        oc_p50 = {"SL": 0, "TP": 0, "HOLD": 0}
        for s in sigs:
            if s["mae"] <= sl_p50 and s["mfe"] >= tp_p50:
                ret = sl_p50; oc_p50["SL"] += 1
            elif s["mae"] <= sl_p50:
                ret = sl_p50; oc_p50["SL"] += 1
            elif s["mfe"] >= tp_p50:
                ret = tp_p50; oc_p50["TP"] += 1
            else:
                ret = s["fwd_return"]; oc_p50["HOLD"] += 1
            sim_rets_p50.append(ret)
        sim_p50 = calc_metrics(sim_rets_p50, f"SL=p50({sl_p50:+.1f}%) TP=p50")

        # 4c. Catastrophic SL p25 + Time Stop (không có TP cứng)
        sim_rets_cat = []
        oc_cat = {"SL": 0, "HOLD": 0}
        for s in sigs:
            if s["mae"] <= sl_catastrophic:
                ret = sl_catastrophic; oc_cat["SL"] += 1
            else:
                ret = s["fwd_return"]; oc_cat["HOLD"] += 1
            sim_rets_cat.append(ret)
        sim_cat = calc_metrics(sim_rets_cat,
                               f"Catastrophic SL=p25({sl_catastrophic:+.1f}%)")

        # 4d. Catastrophic SL p25 + TP median
        sim_rets_cat_tp = []
        oc_cat_tp = {"SL": 0, "TP": 0, "HOLD": 0}
        for s in sigs:
            if s["mae"] <= sl_catastrophic and s["mfe"] >= tp_median:
                ret = sl_catastrophic; oc_cat_tp["SL"] += 1
            elif s["mae"] <= sl_catastrophic:
                ret = sl_catastrophic; oc_cat_tp["SL"] += 1
            elif s["mfe"] >= tp_median:
                ret = tp_median; oc_cat_tp["TP"] += 1
            else:
                ret = s["fwd_return"]; oc_cat_tp["HOLD"] += 1
            sim_rets_cat_tp.append(ret)
        sim_cat_tp = calc_metrics(sim_rets_cat_tp,
                                  f"p25 SL + p50 TP")

        # Print
        print(f"\n  [{scenario_label}]")
        print(f"  {'Scenario':<35} {'n':>5} {'WR':>7} {'MeanExp':>9} "
              f"{'PF':>6} {'Sharpe':>7} {'MaxDD':>8}")
        print(f"  {'─'*75}")

        for m, oc in [
            (base_m,       None),
            (sim_p50,      oc_p50),
            (sim_cat,      oc_cat),
            (sim_cat_tp,   oc_cat_tp),
        ]:
            marker = "→" if m == base_m else "  "
            print(
                f"  {marker} {m['label']:<33} {m['n']:>5} "
                f"{m['wr']:>6.1f}% {m['mean_exp']:>+8.2f}% "
                f"{m['pf']:>6.2f} {m['sharpe']:>+6.3f} "
                f"{m['max_dd']:>+7.1f}%"
            )
            if oc:
                total = sum(oc.values())
                parts = " ".join(f"{k}={v/total*100:.0f}%"
                                 for k, v in oc.items() if total > 0)
                print(f"       Outcomes: {parts}")

        sim_results[scenario_label] = {
            "baseline":       base_m,
            "sl_p50_tp_p50":  sim_p50,
            "catastrophic_sl":sim_cat,
            "cat_sl_tp":      sim_cat_tp,
        }

    # ── 5. Key insight summary ────────────────────────────────────────────────
    print(f"{'─'*65}")
    print(f"KEY INSIGHTS — {cluster}:")

    # MAE của H1 vs H2
    if h1_sigs and h2_sigs:
        h1_mae_med = np.percentile([s["mae"] for s in h1_sigs], 50)
        h2_mae_med = np.percentile([s["mae"] for s in h2_sigs], 50)
        h1_mfe_med = np.percentile([s["mfe"] for s in h1_sigs], 50)
        h2_mfe_med = np.percentile([s["mfe"] for s in h2_sigs], 50)

        print(f"\n  H1: MAE_med={h1_mae_med:+.2f}%  MFE_med={h1_mfe_med:+.2f}%  "
              f"R:R={h1_mfe_med/abs(h1_mae_med):.2f}x")
        print(f"  H2: MAE_med={h2_mae_med:+.2f}%  MFE_med={h2_mfe_med:+.2f}%  "
              f"R:R={h2_mfe_med/abs(h2_mae_med):.2f}x")

        if abs(h2_mae_med) > abs(h1_mae_med) * 1.2:
            print(f"\n  ⚠️  H2 MAE rộng hơn H1 {abs(h2_mae_med/h1_mae_med):.1f}x")
            print(f"  → H2 không chỉ win rate thấp hơn mà còn risk cao hơn")
            print(f"  → Đề xuất: reduce size 50% hoặc skip H2 hoàn toàn")
        elif h2_mfe_med < h1_mfe_med * 0.8:
            print(f"\n  ⚠️  H2 MFE thấp hơn H1 đáng kể")
            print(f"  → H2 moves nhỏ hơn → khó cover phí giao dịch")
            print(f"  → Đề xuất: raise TP bar hoặc reduce size H2")
        else:
            print(f"\n  ✅ H1/H2 MAE/MFE không khác biệt nhiều")
            print(f"  → H2 kém hơn chủ yếu vì win rate, không phải risk size")

    # SL/TP recommendation
    sl_all = sl_tp.get("ALL", {}).get("sl", -5.0)
    tp_all = sl_tp.get("ALL", {}).get("tp", +7.0)
    sl_h1  = sl_tp.get("H1", {}).get("sl", sl_all)
    tp_h1  = sl_tp.get("H1", {}).get("tp", tp_all)
    sl_h2  = sl_tp.get("H2", {}).get("sl", sl_all)
    tp_h2  = sl_tp.get("H2", {}).get("tp", tp_all)

    print(f"\n  RECOMMENDED SL/TP (từ median MAE/MFE):")
    print(f"  H1: SL={sl_h1:+.2f}%  TP={tp_h1:+.2f}%")
    print(f"  H2: SL={sl_h2:+.2f}%  TP={tp_h2:+.2f}%")
    print(f"  ALL: SL={sl_all:+.2f}%  TP={tp_all:+.2f}%")

    return {
        "cluster":    cluster,
        "n_signals":  len(all_signals),
        "sl_tp":      sl_tp,
        "sim_results": {
            k: {ph: {mk: mv for mk, mv in m.items()}
                for ph, m in v.items()}
            for k, v in sim_results.items()
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = [a.upper() for a in sys.argv[1:]]
    run_mr  = not args or "MR"  in args
    run_mom = not args or "MOM" in args

    print(f"{'='*65}")
    print(f"MAE/MFE ANALYSIS — VN Trader Bot V6")
    print(f"Entry=Close T | MAE/MFE từ High/Low daily")
    print(f"SL = median MAE | TP = median MFE (per H1/H2)")
    print(f"{'='*65}")

    all_output = {}

    for cluster, run_flag in [
        ("Mean Reversion", run_mr),
        ("Momentum",       run_mom),
    ]:
        if not run_flag:
            continue

        symbols = CLUSTER_ASSIGNMENTS[cluster]
        print(f"\nLoading {cluster} ({len(symbols)} ma)...")

        all_signals = []
        for sym in symbols:
            print(f"  {sym}...", end=" ", flush=True)
            r = analyze_symbol(sym, cluster)
            if r:
                n = len(r["signals"])
                print(f"{n} signals")
                all_signals.extend(r["signals"])
            else:
                print("FAIL/no signals")

        if not all_signals:
            print(f"  Khong co signal cho {cluster}")
            continue

        print(f"\nTotal signals: {len(all_signals)}")
        result = analyze_cluster(cluster, all_signals)
        all_output[cluster] = result

    # ── Cross-cluster comparison ──────────────────────────────────────────────
    if len(all_output) == 2:
        print(f"\n{'═'*65}")
        print(f"CROSS-CLUSTER: Baseline vs SL/TP")
        print(f"{'═'*65}")
        print(f"  {'Cluster':<20} {'Scenario':<15} {'WR':>7} "
              f"{'MeanExp':>9} {'PF':>6} {'Sharpe':>7}")
        print(f"  {'─'*65}")

        for cl, out in all_output.items():
            for scenario in ["ALL signals"]:
                sr = out["sim_results"].get(scenario, {})
                for phase, label in [
                    ("baseline",        "Baseline"),
                    ("sl_p50_tp_p50",   "SL=p50/TP=p50"),
                    ("catastrophic_sl", "Catastrophic SL p25"),
                    ("cat_sl_tp",       "p25 SL + p50 TP"),
                ]:
                    m = sr.get(phase, {})
                    if not m:
                        continue
                    print(
                        f"  {cl:<20} {label:<15} "
                        f"{m.get('wr',0):>6.1f}% "
                        f"{m.get('mean_exp',0):>+8.2f}% "
                        f"{m.get('pf',0):>6.2f} "
                        f"{m.get('sharpe',0):>+6.3f}"
                    )
            print()

    # Save
    out_path = Path("mae_mfe_results.json")
    out_path.write_text(json.dumps(all_output, indent=2,
                                   ensure_ascii=False, default=str))
    print(f"  Da luu: {out_path}")
    print(f"\n  BUOC TIEP THEO:")
    print(f"  1. Review SL/TP recommendations per cluster")
    print(f"  2. Integrate SL/TP vao backtest_rule_cmd.py")
    print(f"  3. Build daily scan pipeline voi SL/TP moi")


if __name__ == "__main__":
    main()
