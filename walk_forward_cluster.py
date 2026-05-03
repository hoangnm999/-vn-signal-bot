"""
walk_forward_cluster.py — True Walk Forward với Expanding Window
Session 30 — VN Trader Bot V6

Thiết kế:
  Expanding window, fold size = 6 tháng:
    Fold 1: Train 2019-01 → 2021-12 | Test 2022-01 → 2022-06
    Fold 2: Train 2019-01 → 2022-06 | Test 2022-07 → 2022-12
    Fold 3: Train 2019-01 → 2022-12 | Test 2023-01 → 2023-06
    Fold 4: Train 2019-01 → 2023-06 | Test 2023-07 → 2023-12
    Fold 5: Train 2019-01 → 2023-12 | Test 2024-01 → 2024-06
    Fold 6: Train 2019-01 → 2024-06 | Test 2024-07 → 2024-12
    Fold 7: Train 2019-01 → 2024-12 | Test 2025-01 → nay (OOS thật)

  Mỗi fold:
    - Tính threshold từ toàn bộ training data trước test period
    - Chạy signal detection trên test period
    - Không leakage

  Metrics:
    - Per-fold: n, WR, Exp, PF
    - Aggregate: tổng hợp tất cả fold
    - WFE = OOS_Exp_aggregate / Train_Exp_aggregate
    - Consistency: % fold có Exp > 0

Chạy:
  python walk_forward_cluster.py
  python walk_forward_cluster.py MOM   ← chỉ Momentum cluster
  python walk_forward_cluster.py MR    ← chỉ Mean Reversion cluster
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Cluster assignments (fallback từ test_cluster_signal.py) ──────────────────
CLUSTER_ASSIGNMENTS = {
    "Mean Reversion": ["SSI", "VND", "HCM", "DGC", "DCM", "DPM",
                       "NKG", "HSG", "VHM", "NVL", "GAS", "HAH"],
    "Momentum":       ["VCB", "BID", "CTG", "MBB", "TCB", "STB",
                       "HPG", "KDH", "MWG", "FRT", "PNJ",
                       "REE", "PC1", "FPT", "GMD"],
}

# ── Walk Forward config ───────────────────────────────────────────────────────
WF_TRAIN_START  = "2019-01-01"   # training bắt đầu (cố định)
WF_FIRST_TEST   = "2022-01-01"   # test period đầu tiên
FOLD_MONTHS     = 6              # mỗi fold = 6 tháng
MIN_TRAIN_BARS  = 400            # tối thiểu bars training
MIN_TEST_SIGS   = 3              # tối thiểu signals để tính metrics 1 fold

# ── Signal config ─────────────────────────────────────────────────────────────
TRIGGER_PCT   = 70
MIN_TRIGGERS  = 2
WIN_THRESH    = 1.0
WF_COOLDOWN   = 5

FWD_CONFIG = {
    "Mean Reversion": 20,   # FWD=20 (kết quả tốt nhất từ Tầng 2)
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


# ── Threshold computation ─────────────────────────────────────────────────────

def compute_thresholds(train_rows: list[dict], cfg: dict) -> dict:
    reg_ind  = cfg["regime_indicator"]
    trig_ind = cfg["trigger_indicators"]
    trig_dir = cfg["trigger_direction"]

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

    return {"reg_thresh": reg_thresh, "trig_thresh": trig_thresh}


# ── Signal detection trên một period ─────────────────────────────────────────

def run_period(
    ind_map: dict,
    close_arr: np.ndarray,
    test_indices: list[int],
    n_bars: int,
    cfg: dict,
    thresholds: dict,
    fwd_days: int,
) -> list[dict]:
    """Chạy signal detection trên test_indices với threshold cho trước."""
    reg_ind    = cfg["regime_indicator"]
    reg_cond   = cfg["regime_condition"]
    trig_ind   = cfg["trigger_indicators"]
    trig_dir   = cfg["trigger_direction"]
    reg_thresh  = thresholds["reg_thresh"]
    trig_thresh = thresholds["trig_thresh"]

    last_signal = None
    signals     = []

    for t_idx in test_indices:
        row = ind_map.get(t_idx)
        if row is None:
            continue

        # Cooldown
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

        fwd_idx = t_idx + fwd_days
        if fwd_idx >= n_bars:
            continue

        actual = float(
            (close_arr[fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100
        )
        signals.append({
            "t_idx":  t_idx,
            "actual": actual,
            "n_trig": n_trig,
        })
        last_signal = t_idx

    return signals


def calc_metrics(signals: list[dict]) -> dict:
    if not signals:
        return {"n": 0, "wr": 0.0, "mean_exp": 0.0, "pf": 0.0, "max_dd": 0.0}
    rets = [s["actual"] for s in signals]
    wins = [r for r in rets if r >= WIN_THRESH]
    loss = [r for r in rets if r < WIN_THRESH]
    pf   = round(sum(wins) / abs(sum(loss)), 2) if loss and sum(loss) != 0 else 99.0
    return {
        "n":        len(signals),
        "wr":       round(len(wins) / len(rets) * 100, 1),
        "mean_exp": round(float(np.mean(rets)), 2),
        "pf":       round(pf, 2),
        "max_dd":   round(float(np.min(rets)), 1),
    }


# ── Walk Forward folds ────────────────────────────────────────────────────────

def build_folds(df: pd.DataFrame) -> list[dict]:
    """
    Tạo danh sách folds với expanding window.
    Mỗi fold: train_end, test_start, test_end.
    """
    folds     = []
    test_start = pd.Timestamp(WF_FIRST_TEST)
    today      = pd.Timestamp(datetime.today().date())

    while test_start < today:
        test_end = test_start + pd.DateOffset(months=FOLD_MONTHS) - pd.Timedelta(days=1)
        if test_end > today:
            test_end = today

        train_end = test_start - pd.Timedelta(days=1)

        folds.append({
            "train_start": WF_TRAIN_START,
            "train_end":   str(train_end.date()),
            "test_start":  str(test_start.date()),
            "test_end":    str(test_end.date()),
            "label":       f"{str(test_start.date())[:7]}→{str(test_end.date())[:7]}",
        })

        test_start = test_start + pd.DateOffset(months=FOLD_MONTHS)

    return folds


# ── Per-symbol walk forward ───────────────────────────────────────────────────

def walk_forward_symbol(symbol: str, cluster: str) -> dict | None:
    cfg     = SIGNAL_CONFIG[cluster]
    fwd     = FWD_CONFIG[cluster]

    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=300)
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

    df_full   = df.reset_index(drop=True)
    n_bars    = len(df_full)
    close_arr = df_full["close"].values.astype(float)

    all_rows = compute_indicators(df_full)
    ind_map  = {r["idx"]: r for r in all_rows}

    # Tạo date → index map
    date_to_idx = {str(row["date"])[:10]: i
                   for i, row in df_full.iterrows()}

    folds = build_folds(df_full)

    # Training baseline (toàn bộ 2019-2024, step=3)
    train_mask  = (df_full["date"] >= WF_TRAIN_START) & \
                  (df_full["date"] <= "2024-12-31")
    train_idx   = df_full[train_mask].index.tolist()
    train_rows  = [ind_map[i] for i in train_idx if i in ind_map]

    if len(train_rows) < MIN_TRAIN_BARS:
        return {"symbol": symbol, "error": "Khong du training data"}

    train_thresh   = compute_thresholds(train_rows, cfg)
    train_signals  = run_period(
        ind_map, close_arr, train_idx[::3], n_bars, cfg, train_thresh, fwd
    )
    train_metrics  = calc_metrics(train_signals)

    # Walk forward folds
    fold_results = []

    for fold in folds:
        # Training indices cho fold này
        fold_train_mask = (df_full["date"] >= fold["train_start"]) & \
                          (df_full["date"] <= fold["train_end"])
        fold_test_mask  = (df_full["date"] >= fold["test_start"]) & \
                          (df_full["date"] <= fold["test_end"])

        fold_train_idx = df_full[fold_train_mask].index.tolist()
        fold_test_idx  = df_full[fold_test_mask].index.tolist()

        if len(fold_train_idx) < MIN_TRAIN_BARS or not fold_test_idx:
            continue

        # Tính threshold từ training của fold này (expanding)
        fold_train_rows = [ind_map[i] for i in fold_train_idx if i in ind_map]
        if not fold_train_rows:
            continue

        fold_thresh  = compute_thresholds(fold_train_rows, cfg)
        fold_signals = run_period(
            ind_map, close_arr, fold_test_idx, n_bars, cfg, fold_thresh, fwd
        )
        fold_metrics = calc_metrics(fold_signals)

        fold_results.append({
            "label":       fold["label"],
            "train_end":   fold["train_end"],
            "test_start":  fold["test_start"],
            "test_end":    fold["test_end"],
            "metrics":     fold_metrics,
            "signals":     fold_signals,   # raw signals cho PF + H1/H2
            "reg_thresh":  round(fold_thresh["reg_thresh"], 3),
            "is_oos_2025": fold["test_start"] >= "2025-01-01",
            "half":        "H1" if int(fold["test_start"][5:7]) <= 6 else "H2",
        })

    # Aggregate WF metrics (tất cả folds)
    all_wf_signals = []
    for f in fold_results:
        m = f["metrics"]
        # Reconstruct rets từ metrics (đơn giản hóa)
        all_wf_signals.append(m)

    return {
        "symbol":       symbol,
        "cluster":      cluster,
        "fwd_days":     fwd,
        "train_metrics":train_metrics,
        "fold_results": fold_results,
    }


# ── Cluster aggregation ───────────────────────────────────────────────────────

def aggregate_wf(sym_results: list[dict]) -> dict:
    """
    Tổng hợp walk forward metrics cross-symbol.
    Dùng weighted average theo n signals của từng fold.
    """
    # Per-fold aggregate
    fold_labels = []
    for r in sym_results:
        for f in r.get("fold_results", []):
            if f["label"] not in fold_labels:
                fold_labels.append(f["label"])
    fold_labels.sort()

    fold_agg = {}
    for label in fold_labels:
        all_rets = []

        for r in sym_results:
            for f in r.get("fold_results", []):
                if f["label"] == label:
                    for s in f.get("signals", []):
                        all_rets.append(s["actual"])

        if not all_rets:
            continue

        wins_r = [r for r in all_rets if r >= WIN_THRESH]
        loss_r = [r for r in all_rets if r < WIN_THRESH]
        pf     = round(sum(wins_r) / abs(sum(loss_r)), 2)                  if loss_r and sum(loss_r) != 0 else 99.0

        fold_agg[label] = {
            "label":    label,
            "n":        len(all_rets),
            "mean_exp": round(float(np.mean(all_rets)), 2),
            "wr":       round(len(wins_r) / len(all_rets) * 100, 1),
            "pf":       pf,
            "max_dd":   round(float(np.min(all_rets)), 1),
        }

    # Overall WF aggregate
    total_n   = sum(v["n"] for v in fold_agg.values())
    total_exp = sum(v["mean_exp"] * v["n"] for v in fold_agg.values())
    wf_exp    = round(total_exp / total_n, 2) if total_n else 0.0

    # Train aggregate
    train_n   = sum(r.get("train_metrics", {}).get("n", 0) for r in sym_results)
    train_sum = sum(
        r.get("train_metrics", {}).get("mean_exp", 0) *
        r.get("train_metrics", {}).get("n", 0)
        for r in sym_results
    )
    train_exp = round(train_sum / train_n, 2) if train_n else 0.0

    # WFE
    wfe = round(wf_exp / train_exp, 2) if train_exp > 0 else 0.0

    # Consistency: % folds có mean_exp > 0
    n_positive = sum(1 for v in fold_agg.values() if v["mean_exp"] > 0)
    consistency = round(n_positive / len(fold_agg) * 100, 1) if fold_agg else 0.0

    return {
        "fold_agg":    fold_agg,
        "wf_exp":      wf_exp,
        "train_exp":   train_exp,
        "wfe":         wfe,
        "consistency": consistency,
        "n_folds":     len(fold_agg),
        "n_positive":  n_positive,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_wf_results(cluster: str, sym_results: list[dict], agg: dict):
    fwd = FWD_CONFIG[cluster]

    print(f"\n{'═'*70}")
    print(f"WALK FORWARD — {cluster.upper()} (FWD={fwd}d)")
    print(f"{'═'*70}")

    # Per-fold aggregate table
    print(f"\nPER-FOLD AGGREGATE (cross-symbol):")
    print(f"  {'Fold':<20} {'n':>5} {'WR':>7} {'MeanExp':>9} {'PF':>8}  Bar")
    print(f"  {'─'*50}")

    for label, m in sorted(agg["fold_agg"].items()):
        is_2025 = label >= "2025"
        bar_len = max(0, min(20, int((m["mean_exp"] + 3) * 3)))
        bar     = "█" * bar_len if m["mean_exp"] > 0 else "░" * abs(bar_len)
        marker  = " ◄ OOS 2025" if is_2025 else ""
        pf_str  = f"PF={m.get('pf', 0):.2f}"
        print(
            f"  {label:<20} {m['n']:>5} {m['wr']:>6.1f}% "
            f"{m['mean_exp']:>+8.2f}% {pf_str:>8}  {bar}{marker}"
        )

    # Per-symbol summary
    print(f"\nPER-SYMBOL SUMMARY:")
    print(f"  {'Symbol':<7} {'Train_n':>7} {'Train_Exp':>10} "
          f"{'WF_n':>6} {'WF_Exp':>8} {'WFE':>6}  {'Folds+':>7}  Verdict")
    print(f"  {'─'*72}")

    sym_rows = []
    for r in sorted(sym_results,
                    key=lambda x: -(x.get("train_metrics", {}).get("mean_exp", -99))):
        if r.get("error"):
            continue
        tm = r.get("train_metrics", {})
        folds = r.get("fold_results", [])
        if not folds:
            continue

        wf_n    = sum(f["metrics"]["n"] for f in folds)
        wf_sigs = sum(f["metrics"]["mean_exp"] * f["metrics"]["n"]
                      for f in folds if f["metrics"]["n"] > 0)
        wf_exp  = round(wf_sigs / wf_n, 2) if wf_n else 0.0
        wfe     = round(wf_exp / tm["mean_exp"], 2) if tm.get("mean_exp", 0) > 0 else 0.0
        n_pos   = sum(1 for f in folds if f["metrics"]["mean_exp"] > 0)
        pct_pos = round(n_pos / len(folds) * 100) if folds else 0

        if wfe >= 0.7 and wf_exp > 0.5:
            verdict = "✅ Strong"
        elif wf_exp > 0.3:
            verdict = "· Pass"
        elif wf_exp > 0:
            verdict = "· Duong yeu"
        else:
            verdict = "❌ Am"

        sym_rows.append((r["symbol"], tm.get("n", 0), tm.get("mean_exp", 0),
                         wf_n, wf_exp, wfe, pct_pos, verdict))
        print(
            f"  {r['symbol']:<7} {tm.get('n',0):>7} "
            f"{tm.get('mean_exp',0):>+9.2f}% "
            f"{wf_n:>6} {wf_exp:>+7.2f}% {wfe:>6.2f}  "
            f"{pct_pos:>5}%  {verdict}"
        )

    # ── H1 vs H2 seasonality analysis ───────────────────────────────────────
    print(f"\nH1 vs H2 SEASONALITY (per-symbol):")
    print(f"  {'Symbol':<7} " + "  ".join(
        f"{'H1':>6}/{'H2':>6}" for y in ["2022","2023","2024","2025"]
    ) + "  H1_avg  H2_avg  Pattern")
    print(f"  {'─'*75}")

    h1h2_h1_all, h1h2_h2_all = [], []

    for r in sorted(sym_results,
                    key=lambda x: -(x.get("train_metrics",{}).get("mean_exp",-99))):
        if r.get("error"):
            continue
        sym    = r["symbol"]
        folds  = r.get("fold_results", [])

        # Group by year+half
        yh = {}
        for f in folds:
            yr   = f["test_start"][:4]
            half = f["half"]
            key  = f"{yr}{half}"
            sigs = f.get("signals", [])
            if sigs:
                yh[key] = float(np.mean([s["actual"] for s in sigs]))

        row = f"  {sym:<7}"
        h1_vals, h2_vals = [], []
        for yr in ["2022","2023","2024","2025"]:
            h1 = yh.get(f"{yr}H1")
            h2 = yh.get(f"{yr}H2")
            h1_str = f"{h1:>+5.1f}" if h1 is not None else "  n/a"
            h2_str = f"{h2:>+5.1f}" if h2 is not None else "  n/a"
            row += f"  {h1_str}/{h2_str}"
            if h1 is not None: h1_vals.append(h1)
            if h2 is not None: h2_vals.append(h2)

        h1_avg = float(np.mean(h1_vals)) if h1_vals else 0.0
        h2_avg = float(np.mean(h2_vals)) if h2_vals else 0.0
        h1h2_h1_all.extend(h1_vals)
        h1h2_h2_all.extend(h2_vals)

        n_h1_better = sum(1 for h1, h2 in zip(h1_vals, h2_vals) if h1 > h2)
        n_pairs     = min(len(h1_vals), len(h2_vals))
        pct_h1      = n_h1_better / n_pairs * 100 if n_pairs else 0

        pattern = (f"H1>{pct_h1:.0f}%" if pct_h1 >= 60
                   else f"H2>{100-pct_h1:.0f}%" if pct_h1 <= 40
                   else "Mixed")
        row += f"  {h1_avg:>+6.2f}  {h2_avg:>+6.2f}  {pattern}"
        print(row)

    # Cross-symbol H1 vs H2 summary
    if h1h2_h1_all and h1h2_h2_all:
        avg_h1 = float(np.mean(h1h2_h1_all))
        avg_h2 = float(np.mean(h1h2_h2_all))
        n_pairs_total = min(len(h1h2_h1_all), len(h1h2_h2_all))
        # Count per-year cross-symbol
        n_h1_better_total = sum(1 for h1, h2 in zip(h1h2_h1_all, h1h2_h2_all)
                                if h1 > h2)
        pct_total = n_h1_better_total / n_pairs_total * 100 if n_pairs_total else 0
        print(f"  {'─'*75}")
        print(f"  {'AGGREGATE':<7}  {'':>33}  {avg_h1:>+6.2f}  {avg_h2:>+6.2f}  "
              f"H1 wins {pct_total:.0f}% of time")
        if pct_total >= 65:
            print(f"  → PATTERN NHAT QUAN: H1 tot hon H2 trong {pct_total:.0f}% truong hop")
            print(f"  → Goi y: tang position size H1, giam H2")
        elif pct_total <= 35:
            print(f"  → PATTERN NHAT QUAN: H2 tot hon H1 trong {100-pct_total:.0f}% truong hop")
        else:
            print(f"  → KHONG CO PATTERN RO RANG (H1 wins {pct_total:.0f}%) — co the la nhieu")

    # Overall verdict
    print(f"\n{'─'*70}")
    wfe     = agg["wfe"]
    wf_exp  = agg["wf_exp"]
    consist = agg["consistency"]

    # Tính WF PF từ fold_agg
    all_wf_rets  = []
    for r in sym_results:
        for f in r.get("fold_results", []):
            for s in f.get("signals", []):
                all_wf_rets.append(s["actual"])
    wf_wins = [r for r in all_wf_rets if r >= WIN_THRESH]
    wf_loss = [r for r in all_wf_rets if r < WIN_THRESH]
    wf_pf   = round(sum(wf_wins) / abs(sum(wf_loss)), 2)               if wf_loss and sum(wf_loss) != 0 else 99.0

    print(f"  CLUSTER AGGREGATE:")
    print(f"    Training Exp : {agg['train_exp']:+.2f}%")
    print(f"    WF Exp       : {wf_exp:+.2f}%  |  WF PF: {wf_pf:.2f}")
    print(f"    WFE          : {wfe:.2f}  "
          + ("✅ >0.7 Tốt" if wfe >= 0.7 else
             "· 0.5-0.7 Chấp nhận" if wfe >= 0.5 else
             "❌ <0.5 Overfit"))
    print(f"    Consistency  : {consist:.1f}% folds có Exp > 0  "
          + ("✅" if consist >= 60 else "⚠️"))
    print(f"    Folds total  : {agg['n_folds']} ({agg['n_positive']} dương)")

    # Watchlist đề xuất
    strong = [row[0] for row in sym_rows if row[6] == "✅ Strong"]
    passing = [row[0] for row in sym_rows if row[6] == "· Pass"]
    if strong or passing:
        print(f"\n  WATCHLIST ĐỀ XUẤT (từ WF):")
        if strong:
            print(f"    ★ Strong  : {strong}")
        if passing:
            print(f"    · Passing : {passing}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Parse args
    args = [a.upper() for a in sys.argv[1:]]
    run_mr  = not args or "MR" in args
    run_mom = not args or "MOM" in args

    print(f"{'='*70}")
    print(f"TRUE WALK FORWARD — VN Trader Bot V6")
    print(f"Expanding window | Fold={FOLD_MONTHS}M | "
          f"First test: {WF_FIRST_TEST}")
    print(f"MR FWD={FWD_CONFIG['Mean Reversion']}d | "
          f"MOM FWD={FWD_CONFIG['Momentum']}d")
    print(f"WFE target: >0.70")
    print(f"{'='*70}")

    all_output = {}

    for cluster, run_flag in [("Mean Reversion", run_mr), ("Momentum", run_mom)]:
        if not run_flag:
            continue

        symbols = CLUSTER_ASSIGNMENTS[cluster]
        fwd     = FWD_CONFIG[cluster]
        print(f"\nRunning {cluster} ({len(symbols)} ma, FWD={fwd}d)...")

        sym_results = []
        for sym in symbols:
            print(f"  {sym}...", end=" ", flush=True)
            r = walk_forward_symbol(sym, cluster)
            if r and not r.get("error"):
                folds_done = len(r.get("fold_results", []))
                wf_n = sum(f["metrics"]["n"] for f in r.get("fold_results", []))
                tm   = r.get("train_metrics", {})
                print(f"train_exp={tm.get('mean_exp',0):+.2f}% "
                      f"folds={folds_done} wf_signals={wf_n}")
            else:
                print(f"ERROR: {r.get('error','?') if r else 'None'}")
            sym_results.append(r)

        valid = [r for r in sym_results if r and not r.get("error")]
        agg   = aggregate_wf(valid)

        print_wf_results(cluster, valid, agg)
        # Lưu fold_results nhưng strip raw signals để giảm file size
        def strip_fold_signals(results):
            out = []
            for r in results:
                if not r: continue
                rc = dict(r)
                rc["fold_results"] = [
                    {k: v for k, v in f.items() if k != "signals"}
                    for f in r.get("fold_results", [])
                ]
                out.append(rc)
            return out

        all_output[cluster] = {
            "agg": agg,
            "symbols": strip_fold_signals(valid),
        }

    # Cross-cluster WFE comparison
    if run_mr and run_mom and len(all_output) == 2:
        print(f"\n{'═'*70}")
        print(f"CROSS-CLUSTER WFE COMPARISON")
        print(f"{'═'*70}")
        print(f"  {'Cluster':<20} {'TrainExp':>9} {'WF_Exp':>8} "
              f"{'WFE':>6} {'Consist':>8}  Verdict")
        print(f"  {'─'*60}")
        for cl, out in all_output.items():
            a = out["agg"]
            wfe_verdict = ("✅ Tốt" if a["wfe"] >= 0.7 else
                           "· OK"   if a["wfe"] >= 0.5 else "❌ Overfit")
            print(f"  {cl:<20} {a['train_exp']:>+8.2f}% {a['wf_exp']:>+7.2f}% "
                  f"{a['wfe']:>6.2f} {a['consistency']:>7.1f}%  {wfe_verdict}")

        print(f"\n  KET LUAN:")
        for cl, out in all_output.items():
            a = out["agg"]
            if a["wfe"] >= 0.7 and a["wf_exp"] > 0.5:
                print(f"  ✅ {cl}: WFE={a['wfe']:.2f} — "
                      f"Signal co edge thuc su, du tin cay de deploy")
            elif a["wfe"] >= 0.5 and a["wf_exp"] > 0:
                print(f"  · {cl}: WFE={a['wfe']:.2f} — "
                      f"Co edge nhung can theo doi them truoc khi deploy")
            else:
                print(f"  ❌ {cl}: WFE={a['wfe']:.2f} — "
                      f"Overfit hoac khong co edge, khong nen deploy")

    # Save
    out_path = Path("walk_forward_results.json")
    out_path.write_text(json.dumps(all_output, indent=2, ensure_ascii=False))
    print(f"\n  Da luu: {out_path}")


if __name__ == "__main__":
    main()
