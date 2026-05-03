"""
analyze_timing_recovery.py — Timing & Recovery Analysis
Session 30 — VN Trader Bot V6

Mục tiêu:
  ① Timing Analysis:
     - MAE xảy ra ngày thứ mấy trong FWD window?
     - MFE xảy ra ngày thứ mấy?
     - Case A (MAE trước MFE) chiếm bao nhiêu %?
     → Derive optimal entry/exit timing từ distribution

  ② Recovery Rate Analysis:
     - Signals có MAE sâu: bao nhiêu % hồi về dương?
     - Phân nhóm theo MAE depth: Shallow/Medium/Deep
     - Validate "MR signal thật" vs "falling knife"

  Nguyên tắc: OBSERVE distribution → DERIVE rule
              KHÔNG optimize/grid search trên training

Chạy:
  python analyze_timing_recovery.py
  python analyze_timing_recovery.py MR    ← chỉ Mean Reversion
  python analyze_timing_recovery.py MOM   ← chỉ Momentum
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CLUSTER_ASSIGNMENTS = {
    "Mean Reversion": ["DCM", "NKG", "DPM", "HAH",
                       "HCM", "HSG", "DGC", "GAS",
                       "SSI", "VHM", "VND", "NVL"],
    "Momentum":       ["VCB", "BID", "MBB", "MWG",
                       "CTG", "FRT", "REE", "FPT",
                       "GMD", "STB", "PNJ", "TCB",
                       "HPG", "KDH", "PC1"],
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

# MAE depth buckets
MAE_BUCKETS = {
    "Shallow": (0,    -3.0),   # MAE > -3%
    "Medium":  (-3.0, -8.0),   # -8% < MAE <= -3%
    "Deep":    (-8.0, -999),   # MAE <= -8%
}


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
        })
    return rows


# ── Signal detection với full path ───────────────────────────────────────────

def get_signals_with_path(
    cluster: str,
    df_full: pd.DataFrame,
    ind_map: dict,
    close_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    n_bars: int,
) -> list[dict]:
    """
    Lấy signals trên training + tính full daily path trong FWD window.
    Trả về danh sách signal với:
      - mae_day: ngày thứ mấy MAE xảy ra (1-indexed)
      - mfe_day: ngày thứ mấy MFE xảy ra
      - ordering: "A" (MAE trước MFE), "B" (MFE trước MAE), "tie"
      - daily_path: list return mỗi ngày trong FWD window
      - recovery_path: list return từ MAE day trở đi
    """
    cfg      = SIGNAL_CONFIG[cluster]
    fwd      = FWD_CONFIG[cluster]
    train_idx= df_full[
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

        # Full daily path (return từ entry mỗi ngày)
        daily_path = []
        for k in range(1, fwd + 1):
            idx = t_idx + k
            if idx >= n_bars:
                break
            # Dùng close để tính path
            ret = (close_arr[idx] - entry) / entry * 100
            daily_path.append(round(ret, 2))

        if not daily_path:
            continue

        # MAE/MFE từ high/low
        window_low  = low_arr[t_idx + 1: t_idx + fwd + 1]
        window_high = high_arr[t_idx + 1: t_idx + fwd + 1]

        mae     = float((np.min(window_low)  - entry) / entry * 100)
        mfe     = float((np.max(window_high) - entry) / entry * 100)
        mae_day = int(np.argmin(window_low))  + 1   # 1-indexed
        mfe_day = int(np.argmax(window_high)) + 1

        # Ordering
        if mae_day < mfe_day:
            ordering = "A"   # MAE trước MFE → bounce pattern
        elif mfe_day < mae_day:
            ordering = "B"   # MFE trước MAE → reverse pattern
        else:
            ordering = "tie"

        # Recovery path: return từ MAE day đến cuối FWD
        recovery_path = daily_path[mae_day:]  # từ sau MAE day
        recovery_from_mae = [
            round((close_arr[t_idx + mae_day + k] -
                   low_arr[t_idx + mae_day]) /
                  (low_arr[t_idx + mae_day] + 1e-9) * 100, 2)
            for k in range(1, fwd - mae_day + 1)
            if t_idx + mae_day + k < n_bars
        ]

        fwd_return = daily_path[-1] if daily_path else 0.0

        # Date info
        date_str = str(df_full["date"].iloc[t_idx])[:10]
        half     = "H1" if int(date_str[5:7]) <= 6 else "H2"

        signals.append({
            "t_idx":      t_idx,
            "date":       date_str,
            "half":       half,
            "entry":      round(entry, 2),
            "mae":        round(mae, 2),
            "mfe":        round(mfe, 2),
            "mae_day":    mae_day,
            "mfe_day":    mfe_day,
            "ordering":   ordering,
            "fwd_return": round(fwd_return, 2),
            "daily_path": daily_path,
            "recovery_from_mae": recovery_from_mae,
        })
        last_signal = t_idx

    return signals


def get_mae_bucket(mae: float) -> str:
    if mae > -3.0:
        return "Shallow"
    elif mae > -8.0:
        return "Medium"
    else:
        return "Deep"


def calc_metrics(rets: list[float]) -> dict:
    if not rets:
        return {"n": 0, "wr": 0, "mean": 0, "pf": 0, "sharpe": 0}
    wins = [r for r in rets if r >= WIN_THRESH]
    loss = [r for r in rets if r < WIN_THRESH]
    pf   = round(sum(wins) / abs(sum(loss)), 2) if loss and sum(loss) != 0 else 99.0
    sharpe = round(float(np.mean(rets)) / (float(np.std(rets)) + 1e-9), 3)
    return {
        "n":      len(rets),
        "wr":     round(len(wins) / len(rets) * 100, 1),
        "mean":   round(float(np.mean(rets)), 2),
        "pf":     round(pf, 2),
        "sharpe": sharpe,
    }


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_cluster(cluster: str, all_signals: list[dict]):
    fwd = FWD_CONFIG[cluster]
    n   = len(all_signals)

    print(f"\n{'═'*65}")
    print(f"TIMING & RECOVERY — {cluster.upper()} (FWD={fwd}d, n={n})")
    print(f"{'═'*65}")

    # ══════════════════════════════════════════════════════════════
    # PHẦN I: TIMING ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"PHẦN I — TIMING ANALYSIS")
    print(f"{'─'*65}")

    mae_days = [s["mae_day"] for s in all_signals]
    mfe_days = [s["mfe_day"] for s in all_signals]

    # Distribution MAE day
    print(f"\n1. MAE DAY distribution (ngày thứ mấy giá chạm đáy):")
    for p, q in [(10,10),(25,25),(50,50),(75,75),(90,90)]:
        val = np.percentile(mae_days, q)
        print(f"   p{p:2d} = day {val:.1f}")

    mae_day_med = float(np.percentile(mae_days, 50))
    mae_day_p25 = float(np.percentile(mae_days, 25))
    mae_day_p75 = float(np.percentile(mae_days, 75))

    # Histogram MAE day
    print(f"\n   Histogram MAE day (ngày 1-{fwd}):")
    day_counts = {}
    for d in mae_days:
        day_counts[d] = day_counts.get(d, 0) + 1
    max_count = max(day_counts.values()) if day_counts else 1
    for day in sorted(day_counts.keys()):
        cnt    = day_counts[day]
        bar    = "█" * int(cnt / max_count * 30)
        pct    = cnt / n * 100
        marker = " ◄ median" if day == round(mae_day_med) else ""
        print(f"   Day {day:2d}: {bar:<30} {cnt:3d} ({pct:.1f}%){marker}")

    # Distribution MFE day
    print(f"\n2. MFE DAY distribution (ngày thứ mấy giá chạm đỉnh):")
    for p, q in [(10,10),(25,25),(50,50),(75,75),(90,90)]:
        val = np.percentile(mfe_days, q)
        print(f"   p{p:2d} = day {val:.1f}")

    mfe_day_med = float(np.percentile(mfe_days, 50))

    # Histogram MFE day
    print(f"\n   Histogram MFE day (ngày 1-{fwd}):")
    mfe_day_counts = {}
    for d in mfe_days:
        mfe_day_counts[d] = mfe_day_counts.get(d, 0) + 1
    max_count_mfe = max(mfe_day_counts.values()) if mfe_day_counts else 1
    for day in sorted(mfe_day_counts.keys()):
        cnt    = mfe_day_counts[day]
        bar    = "█" * int(cnt / max_count_mfe * 30)
        pct    = cnt / n * 100
        marker = " ◄ median" if day == round(mfe_day_med) else ""
        print(f"   Day {day:2d}: {bar:<30} {cnt:3d} ({pct:.1f}%){marker}")

    # Ordering analysis
    print(f"\n3. ORDERING ANALYSIS (MAE trước hay MFE trước?):")
    n_A   = sum(1 for s in all_signals if s["ordering"] == "A")
    n_B   = sum(1 for s in all_signals if s["ordering"] == "B")
    n_tie = sum(1 for s in all_signals if s["ordering"] == "tie")
    pct_A = n_A / n * 100
    pct_B = n_B / n * 100

    print(f"   Case A (MAE trước MFE — bounce pattern): {n_A:3d} ({pct_A:.1f}%)")
    print(f"   Case B (MFE trước MAE — reverse pattern): {n_B:3d} ({pct_B:.1f}%)")
    print(f"   Tie (cùng ngày):                          {n_tie:3d} ({n_tie/n*100:.1f}%)")

    if pct_A >= 65:
        print(f"\n   ✅ BOUNCE PATTERN DOMINANT ({pct_A:.0f}%)")
        print(f"   → Entry muộn (sau MAE day) có cơ sở")
        print(f"   → Exit muộn (gần MFE day) có cơ sở")
    elif pct_A >= 50:
        print(f"\n   · Bounce pattern moderate ({pct_A:.0f}%)")
        print(f"   → Entry muộn có thể giúp nhưng không chắc")
    else:
        print(f"\n   ❌ Bounce pattern yếu ({pct_A:.0f}%)")
        print(f"   → Entry muộn KHÔNG có cơ sở cho cluster này")

    # FWD return theo ordering
    print(f"\n4. FWD RETURN theo ordering:")
    for ordering, label in [("A","Case A (bounce)"),("B","Case B (reverse)")]:
        sigs = [s for s in all_signals if s["ordering"] == ordering]
        if not sigs:
            continue
        rets = [s["fwd_return"] for s in sigs]
        m    = calc_metrics(rets)
        print(f"   {label:<25}: n={m['n']:3d} WR={m['wr']:.1f}% "
              f"Exp={m['mean']:+.2f}% PF={m['pf']:.2f}")

    # ── Simulate entry muộn ───────────────────────────────────────
    print(f"\n5. SIMULATE ENTRY MUỘN:")
    print(f"   (Derive từ MAE day median = {mae_day_med:.1f}d)")
    print(f"   {'Entry':<20} {'n':>5} {'WR':>7} {'MeanExp':>9} "
          f"{'PF':>6} {'Sharpe':>7}")
    print(f"   {'─'*55}")

    # Entry T (baseline)
    base_rets = [s["fwd_return"] for s in all_signals]
    bm = calc_metrics(base_rets)
    print(f"   {'Entry T (baseline)':<20} {bm['n']:>5} {bm['wr']:>6.1f}% "
          f"{bm['mean']:>+8.2f}% {bm['pf']:>6.2f} {bm['sharpe']:>+6.3f}")

    # Entry T+k cho k = p25, p50, p75 của MAE day
    for entry_day, label in [
        (int(mae_day_p25), f"Entry T+{int(mae_day_p25)} (p25 MAE)"),
        (int(mae_day_med), f"Entry T+{int(mae_day_med)} (p50 MAE)"),
        (int(mae_day_p75), f"Entry T+{int(mae_day_p75)} (p75 MAE)"),
    ]:
        if entry_day <= 0:
            continue
        late_rets = []
        for s in all_signals:
            path = s["daily_path"]
            # Return từ T+entry_day đến T+fwd
            if len(path) <= entry_day:
                continue
            # Giá tại T+entry_day
            entry_ret_from_T = path[entry_day - 1]  # return từ T đến T+entry_day
            # Return từ T+entry_day đến T+fwd
            final_ret_from_T  = path[-1]
            ret_from_entry    = final_ret_from_T - entry_ret_from_T
            late_rets.append(round(ret_from_entry, 2))

        if not late_rets:
            continue
        m = calc_metrics(late_rets)
        print(f"   {label:<20} {m['n']:>5} {m['wr']:>6.1f}% "
              f"{m['mean']:>+8.2f}% {m['pf']:>6.2f} {m['sharpe']:>+6.3f}")

    # ── Simulate exit timing ──────────────────────────────────────
    print(f"\n6. SIMULATE EXIT TIMING:")
    print(f"   (Derive từ MFE day median = {mfe_day_med:.1f}d)")
    print(f"   {'Exit':<20} {'n':>5} {'WR':>7} {'MeanExp':>9} "
          f"{'PF':>6} {'Sharpe':>7}")
    print(f"   {'─'*55}")

    mfe_day_p25 = float(np.percentile(mfe_days, 25))
    mfe_day_p75 = float(np.percentile(mfe_days, 75))

    for exit_day, label in [
        (int(mfe_day_p25), f"Exit T+{int(mfe_day_p25)} (p25 MFE)"),
        (int(mfe_day_med), f"Exit T+{int(mfe_day_med)} (p50 MFE)"),
        (fwd,              f"Exit T+{fwd} (hiện tại)"),
        (int(mfe_day_p75), f"Exit T+{int(mfe_day_p75)} (p75 MFE)"),
        (min(fwd + 5, 30), f"Exit T+{min(fwd+5,30)} (+5 ngày)"),
    ]:
        if exit_day <= 0 or exit_day > 30:
            continue
        exit_rets = []
        for s in all_signals:
            path = s["daily_path"]
            idx  = min(exit_day, len(path)) - 1
            if idx < 0:
                continue
            exit_rets.append(path[idx])

        if not exit_rets:
            continue
        m = calc_metrics(exit_rets)
        marker = " ◄ baseline" if exit_day == fwd else ""
        print(f"   {label:<20} {m['n']:>5} {m['wr']:>6.1f}% "
              f"{m['mean']:>+8.2f}% {m['pf']:>6.2f} "
              f"{m['sharpe']:>+6.3f}{marker}")

    # ── Best combination ──────────────────────────────────────────
    print(f"\n7. BEST COMBINATION (Entry muộn + Exit muộn):")
    print(f"   Entry = T+{int(mae_day_med)} (p50 MAE day)")
    print(f"   Exit  = T+{int(mfe_day_med)} (p50 MFE day)")

    combo_rets = []
    for s in all_signals:
        path      = s["daily_path"]
        ed        = int(mae_day_med)
        xd        = int(mfe_day_med)
        if len(path) <= ed or xd <= ed:
            continue
        entry_ret = path[ed - 1]
        exit_idx  = min(xd, len(path)) - 1
        exit_ret  = path[exit_idx]
        combo_rets.append(round(exit_ret - entry_ret, 2))

    if combo_rets:
        m = calc_metrics(combo_rets)
        print(f"   Result: n={m['n']} WR={m['wr']:.1f}% "
              f"Exp={m['mean']:+.2f}% PF={m['pf']:.2f} Sharpe={m['sharpe']:+.3f}")
        bm2 = calc_metrics([s["fwd_return"] for s in all_signals])
        if m['mean'] > bm2['mean']:
            print(f"   ✅ Cải thiện so với baseline: "
                  f"{bm2['mean']:+.2f}% → {m['mean']:+.2f}%")
        else:
            print(f"   · Không cải thiện: "
                  f"{bm2['mean']:+.2f}% vs {m['mean']:+.2f}%")

    # ══════════════════════════════════════════════════════════════
    # PHẦN II: RECOVERY RATE ANALYSIS
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"PHẦN II — RECOVERY RATE ANALYSIS")
    print(f"{'─'*65}")

    # Phân nhóm theo MAE depth
    print(f"\n1. PHÂN NHÓM THEO MAE DEPTH:")
    print(f"   {'Bucket':<10} {'n':>5} {'WR':>7} {'MeanExp':>9} "
          f"{'PF':>6} {'Recovery%':>11}  Verdict")
    print(f"   {'─'*60}")

    bucket_results = {}
    for bucket in ["Shallow", "Medium", "Deep"]:
        sigs = [s for s in all_signals if get_mae_bucket(s["mae"]) == bucket]
        if not sigs:
            continue
        rets     = [s["fwd_return"] for s in sigs]
        m        = calc_metrics(rets)
        # Recovery: % signals hồi về > 0 sau MAE
        n_recover = sum(
            1 for s in sigs
            if s["recovery_from_mae"] and s["recovery_from_mae"][-1] > 0
        )
        recover_pct = n_recover / len(sigs) * 100 if sigs else 0

        thresholds = MAE_BUCKETS[bucket]
        t_str = (f"MAE > {thresholds[0]:.0f}%" if thresholds[0] == 0
                 else f"{thresholds[1]:.0f}% < MAE ≤ {thresholds[0]:.0f}%"
                 if thresholds[1] > -100
                 else f"MAE ≤ {thresholds[0]:.0f}%")

        verdict = ("✅ Strong" if m['mean'] > 2 and recover_pct > 60
                   else "· OK"    if m['mean'] > 0 and recover_pct > 50
                   else "❌ Weak")

        print(f"   {bucket:<10} {m['n']:>5} {m['wr']:>6.1f}% "
              f"{m['mean']:>+8.2f}% {m['pf']:>6.2f} "
              f"{recover_pct:>9.1f}%  {verdict}")
        bucket_results[bucket] = {
            "metrics": m, "recover_pct": round(recover_pct, 1)
        }

    # Recovery rate từ MAE về dương
    print(f"\n2. RECOVERY RATE — từ MAE day về dương:")
    print(f"   (% signals hồi về > 0% từ điểm MAE trước khi FWD kết thúc)")

    for bucket in ["Shallow", "Medium", "Deep", "ALL"]:
        if bucket == "ALL":
            sigs = all_signals
        else:
            sigs = [s for s in all_signals if get_mae_bucket(s["mae"]) == bucket]
        if not sigs:
            continue

        # Recovery về 0% từ MAE
        n_recover_0  = sum(
            1 for s in sigs
            if s["recovery_from_mae"] and
            any(r >= 0 for r in s["recovery_from_mae"])
        )
        # Recovery về +2%
        n_recover_2  = sum(
            1 for s in sigs
            if s["recovery_from_mae"] and
            any(r >= 2 for r in s["recovery_from_mae"])
        )
        # Recovery về +5%
        n_recover_5  = sum(
            1 for s in sigs
            if s["recovery_from_mae"] and
            any(r >= 5 for r in s["recovery_from_mae"])
        )
        nn = len(sigs)

        print(f"   {bucket:<10}: n={nn:3d} | "
              f"Hoi ve >0%: {n_recover_0/nn*100:5.1f}% | "
              f">+2%: {n_recover_2/nn*100:5.1f}% | "
              f">+5%: {n_recover_5/nn*100:5.1f}%")

    # Recovery speed — trung bình mất bao nhiêu ngày để recover
    print(f"\n3. RECOVERY SPEED (từ MAE về 0%):")
    recovery_days_list = []
    for s in all_signals:
        for k, r in enumerate(s["recovery_from_mae"]):
            if r >= 0:
                recovery_days_list.append(k + 1)
                break

    if recovery_days_list:
        print(f"   Mean  = {np.mean(recovery_days_list):.1f} ngày")
        print(f"   Median= {np.median(recovery_days_list):.1f} ngày")
        print(f"   p75   = {np.percentile(recovery_days_list, 75):.1f} ngày")

    # Falling knife detection
    print(f"\n4. FALLING KNIFE DETECTION:")
    n_no_recovery = sum(
        1 for s in all_signals
        if not s["recovery_from_mae"] or
        not any(r >= 0 for r in s["recovery_from_mae"])
    )
    knife_pct = n_no_recovery / n * 100

    print(f"   Signals không hồi về 0% trong FWD window: "
          f"{n_no_recovery} ({knife_pct:.1f}%)")

    if knife_pct <= 25:
        print(f"   ✅ Falling knife risk thấp ({knife_pct:.1f}%)")
        print(f"   → Signal là MR thật, không phải catching falling knife")
    elif knife_pct <= 40:
        print(f"   ⚠️  Falling knife risk moderate ({knife_pct:.1f}%)")
        print(f"   → Cần thêm filter để loại falling knife signals")
    else:
        print(f"   ❌ Falling knife risk cao ({knife_pct:.1f}%)")
        print(f"   → Signal quality cần review lại")

    # ── Tóm tắt insights ─────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"TÓM TẮT INSIGHTS — {cluster}")
    print(f"{'═'*65}")

    print(f"\n  MAE thường xảy ra: Day {mae_day_med:.0f} "
          f"(range p25={mae_day_p25:.0f} → p75={mae_day_p75:.0f})")
    print(f"  MFE thường xảy ra: Day {mfe_day_med:.0f} "
          f"(range p25={mfe_day_p25:.0f} → p75={mfe_day_p75:.0f})")
    print(f"  Bounce pattern (Case A): {pct_A:.0f}%")
    print(f"  Falling knife risk: {knife_pct:.0f}%")

    if pct_A >= 65 and knife_pct <= 30:
        print(f"\n  ✅ CONCLUSION: Signal có bounce pattern rõ ràng")
        print(f"  → Entry muộn T+{int(mae_day_med)} có thể cải thiện entry price")
        print(f"  → Exit T+{int(mfe_day_med)} thay vì T+{fwd} đáng test trên OOS")
    elif pct_A >= 50:
        print(f"\n  · CONCLUSION: Bounce pattern moderate")
        print(f"  → Entry muộn có thể giúp nhưng không chắc")
        print(f"  → Giữ nguyên T+{fwd} exit, observe thêm")
    else:
        print(f"\n  ❌ CONCLUSION: Không có bounce pattern đủ mạnh")
        print(f"  → Entry muộn và exit timing thay đổi không có cơ sở")
        print(f"  → Giữ nguyên logic hiện tại")

    return {
        "cluster":      cluster,
        "n":            n,
        "mae_day_med":  mae_day_med,
        "mfe_day_med":  mfe_day_med,
        "pct_case_A":   round(pct_A, 1),
        "knife_pct":    round(knife_pct, 1),
        "bucket_results": bucket_results,
    }


# ── Per-symbol load ───────────────────────────────────────────────────────────

def load_symbol(symbol: str, cluster: str) -> list[dict]:
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
    high_arr  = df_full["high"].values.astype(float)
    low_arr   = df_full["low"].values.astype(float)

    all_rows = compute_indicators(df_full)
    ind_map  = {r["idx"]: r for r in all_rows}

    signals = get_signals_with_path(
        cluster, df_full, ind_map,
        close_arr, high_arr, low_arr, n_bars
    )
    return signals


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = [a.upper() for a in sys.argv[1:]]
    run_mr  = not args or "MR"  in args
    run_mom = not args or "MOM" in args

    print(f"{'='*65}")
    print(f"TIMING & RECOVERY ANALYSIS — VN Trader Bot V6")
    print(f"Training: {TRAIN_START} → {TRAIN_END}")
    print(f"Nguyên tắc: OBSERVE distribution → DERIVE rule")
    print(f"{'='*65}")

    all_output = {}

    for cluster, run_flag in [
        ("Mean Reversion", run_mr),
        ("Momentum",       run_mom),
    ]:
        if not run_flag:
            continue

        symbols = CLUSTER_ASSIGNMENTS[cluster]
        fwd     = FWD_CONFIG[cluster]
        print(f"\nLoading {cluster} ({len(symbols)} ma)...")

        all_signals = []
        for sym in symbols:
            print(f"  {sym}...", end=" ", flush=True)
            sigs = load_symbol(sym, cluster)
            print(f"{len(sigs)} signals")
            all_signals.extend(sigs)

        if not all_signals:
            print(f"  Khong co signal")
            continue

        print(f"  Total: {len(all_signals)} signals")
        result = analyze_cluster(cluster, all_signals)
        all_output[cluster] = result

    # Save
    save_data = {
        k: {mk: mv for mk, mv in v.items() if mk != "bucket_results"}
        for k, v in all_output.items()
    }
    Path("timing_recovery_results.json").write_text(
        json.dumps(save_data, indent=2, ensure_ascii=False)
    )
    print(f"\n  Da luu: timing_recovery_results.json")


if __name__ == "__main__":
    main()
