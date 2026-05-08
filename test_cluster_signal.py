"""
test_cluster_signal.py — Signal Logic Test cho 2 Cluster
Session 30 — VN Trader Bot V6

Flow:
  1. Load cross_symbol_analysis.json → auto-assign cluster cho 27 mã
     Dựa trên Cohen's d của momentum_20d và ema_cross:
       d(momentum_20d) < -0.2 → Mean Reversion (mua khi yếu)
       d(momentum_20d) > +0.2 → Momentum (mua khi mạnh)
       Không rõ → assign theo ngành fallback

  2. Test signal logic cho từng cluster:
     Mean Reversion:
       Regime:  price_vs_sma50 < 0  (giá dưới SMA50)
       Trigger: stoch_k oversold + volume_spike cao + momentum_5d > 0
     Momentum:
       Regime:  ema_cross > 0  (EMA12 > EMA26)
       Trigger: momentum_5d cao + volume_spike cao + candle_body lớn

  3. Tầng 1: FWD=10 cho cả 2 cluster → comparable metrics
     Tầng 2: Mean Reversion test thêm FWD=15, FWD=20

  4. Output: per-symbol + per-cluster aggregated metrics

Chạy:
  python test_cluster_signal.py
  python test_cluster_signal.py --fwd 10    ← chỉ tầng 1
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Universe & Cluster config ─────────────────────────────────────────────────
UNIVERSE = {
    "Ngan hang":          ["VCB", "BID", "CTG", "MBB", "TCB", "STB"],
    "Chung khoan":        ["SSI", "VND", "HCM"],
    "Hoa chat/Phan bon":  ["DGC", "DCM", "DPM"],
    "Thep":               ["HPG", "NKG", "HSG"],
    "Bat dong san":       ["VHM", "NVL", "KDH"],
    "Tieu dung/Ban le":   ["MWG", "FRT", "PNJ"],
    "Nang luong":         ["REE", "PC1", "GAS"],
    "Cong nghe/Logistics":["FPT", "GMD", "HAH"],
}
ALL_SYMBOLS = [s for g in UNIVERSE.values() for s in g]

# Fallback assignment nếu không đủ data trong JSON
SECTOR_FALLBACK = {
    "Mean Reversion": ["SSI", "VND", "HCM", "DGC", "DCM", "DPM",
                       "NKG", "HSG", "VHM", "NVL", "GAS", "HAH"],
    "Momentum":       ["VCB", "BID", "CTG", "MBB", "TCB", "STB",
                       "HPG", "KDH", "MWG", "FRT", "PNJ",
                       "REE", "PC1", "FPT", "GMD"],
}

# ── Signal config ─────────────────────────────────────────────────────────────
TRAIN_START   = "2019-01-01"
TRAIN_END     = "2024-12-31"
OOS_START     = "2025-01-01"
TRIGGER_PCT   = 70       # top 30% = đột biến
MIN_TRIGGERS  = 2
WIN_THRESH    = 1.0      # return > 1% = win
WF_COOLDOWN   = 5        # bars

# Tầng 1: comparable
FWD_TIER1     = 10
# Tầng 2: Mean Reversion optimize
FWD_MR_LIST   = [10, 15, 20]

# Cluster assignment threshold
CLUSTER_D_THRESH = 0.15  # |d| >= 0.15 để assign, else fallback

# Signal definitions — từ data observation
SIGNAL_CONFIG = {
    "Mean Reversion": {
        "regime_indicator":  "price_vs_sma50",
        "regime_condition":  "low",
        "trigger_indicators":["stoch_k", "volume_spike", "momentum_5d"],
        "trigger_direction": {
            "stoch_k":      "low",
            "volume_spike": "high",
            "momentum_5d":  "high",
        },
        "description": "Mua khi gia duoi SMA50 + stoch oversold + volume + momentum hoi",
    },
    "Momentum": {
        "regime_indicator":  "ema_cross",
        "regime_condition":  "high",
        "trigger_indicators":["momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction": {
            "momentum_5d":  "high",
            "volume_spike": "high",
            "candle_body":  "high",
        },
        "description": "ORIGINAL S34: momentum_5d + volume_spike + candle_body",
    },
}

# S36 variant: dùng candle_bull + volume_spike_bull (có hướng nến)
# Chạy song song để so sánh số lệnh và OOS exp
SIGNAL_CONFIG_S36 = {
    "Momentum": {
        "regime_indicator":  "ema_cross",
        "regime_condition":  "high",
        "trigger_indicators":["momentum_5d", "volume_spike_bull", "candle_bull"],
        "trigger_direction": {
            "momentum_5d":     "high",
            "volume_spike_bull":"high",
            "candle_bull":     "high",
        },
        "description": "S36 FIX: momentum_5d + volume_spike_bull + candle_bull (co huong nen)",
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
            "price_vs_sma20": float((px - s20) / (px + 1e-9) * 100),
            "ema_cross":      float((ema12[i] - ema26[i]) / (px + 1e-9) * 100),
            "momentum_5d":    float((px / (c5  + 1e-9) - 1.0) * 100),
            "momentum_20d":   float((px / close[max(i - 20, 0)] - 1.0) * 100),
            "volume_spike":   float((vol[i] / (vs20v + 1e-9)) - 1.0),
            "stoch_k":        float(stoch[i]),
            "candle_body":    float(np.clip(
                abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
            "atr_ratio":      float(atr_v / (px + 1e-9) * 100),
            # S36: indicators có hướng để validate
            "candle_bull":       float(np.clip((px - opn[i]) / (atr_v + 1e-9), -3, 3)),
            "volume_spike_bull": float((vol[i] / (vs20v + 1e-9)) - 1.0) if px >= opn[i] else -1.0,
            "momentum_3d":       float((px / (close[max(i-3, 0)] + 1e-9) - 1.0) * 100),
        })
    return rows


# ── Cluster assignment ────────────────────────────────────────────────────────

def assign_clusters(json_path: str = "cross_symbol_analysis.json") -> dict[str, str]:
    """
    Đọc cross_symbol_analysis.json, assign cluster cho từng mã.
    Logic:
      Với mỗi mã, lấy Cohen's d của momentum_20d và ema_cross.
      d(momentum_20d) âm + d(ema_cross) âm → Mean Reversion
      d(momentum_20d) dương + d(ema_cross) dương → Momentum
      Không nhất quán hoặc |d| nhỏ → fallback theo sector
    """
    assignments = {}

    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        per_symbol = data.get("per_symbol", {})
    except Exception as e:
        print(f"  Khong doc duoc {json_path}: {e} → dung fallback")
        per_symbol = {}

    # Fallback map
    fallback_map = {}
    for cluster, syms in SECTOR_FALLBACK.items():
        for s in syms:
            fallback_map[s] = cluster

    for sym in ALL_SYMBOLS:
        sym_data = per_symbol.get(sym, {})
        top5     = sym_data.get("top5", [])

        d_map = {r["indicator"]: r["cohen_d"] for r in top5}

        # Lấy thêm từ full results nếu có
        # (top5 chỉ có 5 indicators, cần momentum_20d và ema_cross)
        d_mom20  = d_map.get("momentum_20d", None)
        d_ema    = d_map.get("ema_cross", None)

        # Đọc từ cross_results nếu không có trong top5
        if d_mom20 is None or d_ema is None:
            cross = data.get("cross_results", []) if per_symbol else []
            for r in cross:
                if r["indicator"] == "momentum_20d":
                    d_mom20 = r.get("per_symbol_d", {}).get(sym)
                if r["indicator"] == "ema_cross":
                    d_ema = r.get("per_symbol_d", {}).get(sym)

        # Assign logic
        if d_mom20 is not None and d_ema is not None:
            # Cả 2 chỉ số cùng chiều và đủ mạnh
            both_mr  = (d_mom20 <= -CLUSTER_D_THRESH and
                        d_ema   <= -CLUSTER_D_THRESH)
            both_mom = (d_mom20 >= CLUSTER_D_THRESH and
                        d_ema   >= CLUSTER_D_THRESH)
            # Chỉ 1 chỉ số
            one_mr   = (d_mom20 <= -CLUSTER_D_THRESH or
                        d_ema   <= -CLUSTER_D_THRESH)
            one_mom  = (d_mom20 >= CLUSTER_D_THRESH or
                        d_ema   >= CLUSTER_D_THRESH)

            if both_mr:
                assignments[sym] = "Mean Reversion"
            elif both_mom:
                assignments[sym] = "Momentum"
            elif one_mr and not one_mom:
                assignments[sym] = "Mean Reversion"
            elif one_mom and not one_mr:
                assignments[sym] = "Momentum"
            else:
                assignments[sym] = fallback_map.get(sym, "Momentum")
        else:
            assignments[sym] = fallback_map.get(sym, "Momentum")

    return assignments


# ── Threshold computation ─────────────────────────────────────────────────────

def compute_thresholds(ind_rows: list[dict], cfg: dict) -> dict:
    reg_ind  = cfg["regime_indicator"]
    trig_ind = cfg["trigger_indicators"]
    trig_dir = cfg["trigger_direction"]

    reg_vals   = [r[reg_ind] for r in ind_rows
                  if np.isfinite(r.get(reg_ind, np.nan))]
    reg_thresh = float(np.median(reg_vals)) if reg_vals else 0.0

    trig_thresh = {}
    for t in trig_ind:
        vals = [r[t] for r in ind_rows if np.isfinite(r.get(t, np.nan))]
        if not vals:
            continue
        if trig_dir.get(t, "high") == "low":
            trig_thresh[t] = float(np.percentile(vals, 100 - TRIGGER_PCT))
        else:
            trig_thresh[t] = float(np.percentile(vals, TRIGGER_PCT))

    return {"reg_thresh": reg_thresh, "trig_thresh": trig_thresh}


# ── Backtest core ─────────────────────────────────────────────────────────────

def run_backtest(
    symbol: str,
    fwd_days: int,
    phase: str,        # "train" hoặc "oos"
    ind_map: dict,
    close_arr: np.ndarray,
    indices: list[int],
    n_bars: int,
    cfg: dict,
    thresholds: dict,
) -> dict:
    """Chạy backtest/OOS cho 1 mã với 1 FWD_DAYS."""
    reg_ind   = cfg["regime_indicator"]
    reg_cond  = cfg["regime_condition"]
    trig_ind  = cfg["trigger_indicators"]
    trig_dir  = cfg["trigger_direction"]
    reg_thresh  = thresholds["reg_thresh"]
    trig_thresh = thresholds["trig_thresh"]

    step         = 3 if phase == "train" else 1
    last_signal  = None
    n_skip_reg   = 0
    n_skip_trig  = 0
    signals      = []

    for t_idx in indices[::step]:
        row = ind_map.get(t_idx)
        if row is None:
            continue

        # Cooldown (chỉ OOS)
        if phase == "oos" and last_signal and (t_idx - last_signal) < WF_COOLDOWN:
            continue

        # Regime
        val = row.get(reg_ind, np.nan)
        if not np.isfinite(val):
            continue
        in_regime = (val <= reg_thresh) if reg_cond == "low" else (val > reg_thresh)
        if not in_regime:
            n_skip_reg += 1
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
            n_skip_trig += 1
            continue

        fwd_idx = t_idx + fwd_days
        if fwd_idx >= n_bars:
            continue

        actual = float(
            (close_arr[fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100
        )
        signals.append({"t_idx": t_idx, "actual": actual, "n_trig": n_trig})
        last_signal = t_idx

    # Metrics
    if not signals:
        return {"n": 0, "wr": 0, "mean_exp": 0, "pf": 0,
                "max_dd": 0, "n_skip_reg": n_skip_reg,
                "n_skip_trig": n_skip_trig, "signals": []}

    rets  = [s["actual"] for s in signals]
    wins  = [r for r in rets if r >= WIN_THRESH]
    loss  = [r for r in rets if r < WIN_THRESH]
    pf    = round(sum(wins) / abs(sum(loss)), 2) if loss and sum(loss) != 0 else 99.0

    return {
        "n":           len(signals),
        "wr":          round(len(wins) / len(rets) * 100, 1),
        "mean_exp":    round(float(np.mean(rets)), 2),
        "pf":          round(pf, 2),
        "max_dd":      round(float(np.min(rets)), 1),
        "n_skip_reg":  n_skip_reg,
        "n_skip_trig": n_skip_trig,
        "signals":     signals,
    }


# ── Per-symbol full analysis ──────────────────────────────────────────────────

def analyze_symbol(symbol: str, cluster: str) -> dict | None:
    cfg = SIGNAL_CONFIG[cluster]

    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=300)
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"    [{symbol}] ERROR load: {e}")
        return None

    df_full   = df.reset_index(drop=True)
    n_bars    = len(df_full)
    close_arr = df_full["close"].values.astype(float)

    mask_train = (df_full["date"] >= TRAIN_START) & (df_full["date"] <= TRAIN_END)
    mask_oos   = df_full["date"] >= OOS_START
    train_idx  = df_full[mask_train].index.tolist()
    oos_idx    = df_full[mask_oos].index.tolist()

    if len(train_idx) < 200:
        print(f"    [{symbol}] Khong du training data")
        return None

    all_rows  = compute_indicators(df_full)
    ind_map   = {r["idx"]: r for r in all_rows}
    train_rows = [ind_map[i] for i in train_idx if i in ind_map]

    thresholds = compute_thresholds(train_rows, cfg)

    result = {
        "symbol":    symbol,
        "cluster":   cluster,
        "train":     {},
        "oos":       {},
        "mr_fwd":    {},   # chỉ Mean Reversion
    }

    # ── Tầng 1: FWD=10 cho cả 2 cluster ─────────────────────────────────────
    result["train"] = run_backtest(
        symbol, FWD_TIER1, "train",
        ind_map, close_arr, train_idx, n_bars, cfg, thresholds
    )
    if oos_idx:
        result["oos"] = run_backtest(
            symbol, FWD_TIER1, "oos",
            ind_map, close_arr, oos_idx, n_bars, cfg, thresholds
        )

    # ── Tầng 2: Mean Reversion test FWD=15, 20 ───────────────────────────────
    if cluster == "Mean Reversion":
        for fwd in FWD_MR_LIST:
            tr = run_backtest(
                symbol, fwd, "train",
                ind_map, close_arr, train_idx, n_bars, cfg, thresholds
            )
            oos = run_backtest(
                symbol, fwd, "oos",
                ind_map, close_arr, oos_idx, n_bars, cfg, thresholds
            ) if oos_idx else {}
            result["mr_fwd"][f"fwd{fwd}"] = {"train": tr, "oos": oos}

    return result


# ── Cluster aggregation ───────────────────────────────────────────────────────

def aggregate_cluster(sym_results: list[dict], fwd: int, phase: str) -> dict:
    """Tổng hợp metrics của tất cả mã trong 1 cluster."""
    all_rets = []
    n_total  = 0
    wins     = 0

    key = "train" if phase == "train" else "oos"

    for r in sym_results:
        if not r:
            continue
        metrics = r.get(key, {})
        sigs    = metrics.get("signals", [])
        for s in sigs:
            all_rets.append(s["actual"])
            n_total += 1
            if s["actual"] >= WIN_THRESH:
                wins += 1

    if not all_rets:
        return {"n": 0, "wr": 0, "mean_exp": 0, "pf": 0, "max_dd": 0}

    w_rets = [r for r in all_rets if r >= WIN_THRESH]
    l_rets = [r for r in all_rets if r < WIN_THRESH]
    pf     = round(sum(w_rets) / abs(sum(l_rets)), 2) if l_rets and sum(l_rets) != 0 else 99.0

    return {
        "n":        n_total,
        "wr":       round(wins / n_total * 100, 1),
        "mean_exp": round(float(np.mean(all_rets)), 2),
        "pf":       round(pf, 2),
        "max_dd":   round(float(np.min(all_rets)), 1),
    }


# ── Print helpers ─────────────────────────────────────────────────────────────

def print_cluster_results(cluster: str, sym_results: list[dict]):
    cfg = SIGNAL_CONFIG[cluster]
    print(f"\n{'═'*65}")
    print(f"CLUSTER: {cluster.upper()}")
    print(f"Signal: {cfg['description']}")
    print(f"Regime: {cfg['regime_indicator']} ({cfg['regime_condition']})")
    print(f"Triggers: {cfg['trigger_indicators']}")
    print(f"{'═'*65}")

    # Per-symbol table — Tầng 1 FWD=10
    print(f"\nPER-SYMBOL (FWD={FWD_TIER1}d):")
    print(f"  {'Symbol':<6} {'Train_n':>7} {'Train_WR':>9} {'Train_Exp':>10} "
          f"{'OOS_n':>6} {'OOS_WR':>7} {'OOS_Exp':>8}  Verdict")
    print(f"  {'─'*70}")

    for r in sorted(sym_results, key=lambda x: -(x or {}).get("train", {}).get("mean_exp", -99)):
        if not r:
            continue
        tr  = r.get("train", {})
        oos = r.get("oos",   {})
        sym = r["symbol"]

        oos_exp = oos.get("mean_exp", 0)
        tr_exp  = tr.get("mean_exp",  0)
        gap     = oos_exp - tr_exp

        if oos.get("n", 0) < 3:
            verdict = "? Chua du OOS"
        elif oos_exp > 0.5 and oos.get("wr", 0) >= 50:
            verdict = "✅ Pass"
        elif oos_exp > 0:
            verdict = "· Duong yeu"
        else:
            verdict = "❌ Am"

        print(
            f"  {sym:<6} {tr.get('n',0):>7} {tr.get('wr',0):>8.1f}% "
            f"{tr_exp:>+9.2f}% "
            f"{oos.get('n',0):>6} {oos.get('wr',0):>6.1f}% "
            f"{oos_exp:>+7.2f}%  {verdict}"
        )

    # Cluster aggregate
    agg_tr  = aggregate_cluster(sym_results, FWD_TIER1, "train")
    agg_oos = aggregate_cluster(sym_results, FWD_TIER1, "oos")

    print(f"\n  AGGREGATE FWD={FWD_TIER1}d:")
    print(f"  {'':6} {'n':>7} {'WR':>9} {'MeanExp':>10} {'PF':>5} {'MaxDD':>7}")
    print(f"  {'─'*45}")
    print(f"  {'TRAIN':<6} {agg_tr['n']:>7} {agg_tr['wr']:>8.1f}% "
          f"{agg_tr['mean_exp']:>+9.2f}% {agg_tr['pf']:>5.2f} "
          f"{agg_tr['max_dd']:>+6.1f}%")
    print(f"  {'OOS':<6} {agg_oos['n']:>7} {agg_oos['wr']:>8.1f}% "
          f"{agg_oos['mean_exp']:>+9.2f}% {agg_oos['pf']:>5.2f} "
          f"{agg_oos['max_dd']:>+6.1f}%")

    # Tầng 2: Mean Reversion FWD comparison
    if cluster == "Mean Reversion":
        print(f"\n  MEAN REVERSION — FWD COMPARISON (Training):")
        print(f"  {'FWD':<6} {'n':>7} {'WR':>9} {'MeanExp':>10} {'PF':>5}  Verdict")
        print(f"  {'─'*45}")

        for fwd in FWD_MR_LIST:
            key  = f"fwd{fwd}"
            rets_all = []
            wins_all = 0
            for r in sym_results:
                if not r:
                    continue
                sigs = r.get("mr_fwd", {}).get(key, {}).get("train", {}).get("signals", [])
                for s in sigs:
                    rets_all.append(s["actual"])
                    if s["actual"] >= WIN_THRESH:
                        wins_all += 1

            if not rets_all:
                print(f"  FWD={fwd:<3}  (no signals)")
                continue

            w = [r for r in rets_all if r >= WIN_THRESH]
            l = [r for r in rets_all if r < WIN_THRESH]
            wr   = wins_all / len(rets_all) * 100
            exp  = float(np.mean(rets_all))
            pf   = round(sum(w) / abs(sum(l)), 2) if l and sum(l) != 0 else 99.0
            best = "← BEST" if exp == max(
                float(np.mean([s["actual"] for r in sym_results if r
                               for s in r.get("mr_fwd", {}).get(f"fwd{f}", {})
                               .get("train", {}).get("signals", [])]))
                if any(r.get("mr_fwd", {}).get(f"fwd{f}", {}).get("train", {}).get("signals")
                       for r in sym_results if r) else [0]
                for f in FWD_MR_LIST
            ) else ""
            print(f"  FWD={fwd:<3} {len(rets_all):>7} {wr:>8.1f}% "
                  f"{exp:>+9.2f}% {pf:>5.2f}  {best}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"{'='*65}")
    print(f"CLUSTER SIGNAL TEST — VN Trader Bot V6")
    print(f"Training: {TRAIN_START}→{TRAIN_END} | OOS: {OOS_START}+")
    print(f"FWD Tier1={FWD_TIER1}d | MR Tier2={FWD_MR_LIST}d")
    print(f"{'='*65}")

    # ── Step 1: Assign clusters ───────────────────────────────────────────────
    print(f"\nASSIGN CLUSTERS:")
    assignments = assign_clusters()

    mr_syms  = [s for s, c in assignments.items() if c == "Mean Reversion"]
    mom_syms = [s for s, c in assignments.items() if c == "Momentum"]

    print(f"  Mean Reversion ({len(mr_syms)}): {mr_syms}")
    print(f"  Momentum       ({len(mom_syms)}): {mom_syms}")

    # Per-sector view
    print(f"\n  Per sector:")
    for sector, syms in UNIVERSE.items():
        assignments_str = " ".join(
            f"{s}({'MR' if assignments.get(s)=='Mean Reversion' else 'MOM'})"
            for s in syms
        )
        print(f"    {sector:<25} {assignments_str}")

    # ── Step 2: Run analysis per symbol ──────────────────────────────────────
    print(f"\nRUNNING ANALYSIS...")
    mr_results  = []
    mom_results = []

    for sym in mr_syms:
        print(f"  [MR ] {sym}...", end=" ", flush=True)
        r = analyze_symbol(sym, "Mean Reversion")
        mr_results.append(r)
        if r:
            tr  = r.get("train", {})
            oos = r.get("oos",   {})
            print(f"train n={tr.get('n',0)} exp={tr.get('mean_exp',0):+.2f}% | "
                  f"oos n={oos.get('n',0)} exp={oos.get('mean_exp',0):+.2f}%")
        else:
            print("FAIL")

    for sym in mom_syms:
        print(f"  [MOM] {sym}...", end=" ", flush=True)
        r = analyze_symbol(sym, "Momentum")
        mom_results.append(r)
        if r:
            tr  = r.get("train", {})
            oos = r.get("oos",   {})
            print(f"train n={tr.get('n',0)} exp={tr.get('mean_exp',0):+.2f}% | "
                  f"oos n={oos.get('n',0)} exp={oos.get('mean_exp',0):+.2f}%")
        else:
            print("FAIL")

    # ── Step 3: Print results ─────────────────────────────────────────────────
    mr_valid  = [r for r in mr_results  if r]
    mom_valid = [r for r in mom_results if r]

    print_cluster_results("Mean Reversion", mr_valid)
    print_cluster_results("Momentum",       mom_valid)

    # ── Step 4: Cross-cluster comparison (FWD=10) ─────────────────────────────
    print(f"\n{'═'*65}")
    print(f"CROSS-CLUSTER COMPARISON (FWD={FWD_TIER1}d — comparable)")
    print(f"{'═'*65}")

    for cluster, results in [("Mean Reversion", mr_valid), ("Momentum", mom_valid)]:
        agg_tr  = aggregate_cluster(results, FWD_TIER1, "train")
        agg_oos = aggregate_cluster(results, FWD_TIER1, "oos")
        gap     = agg_oos["mean_exp"] - agg_tr["mean_exp"]

        print(f"\n  {cluster}:")
        print(f"    Train: n={agg_tr['n']:>4} WR={agg_tr['wr']:.1f}% "
              f"Exp={agg_tr['mean_exp']:+.2f}% PF={agg_tr['pf']:.2f}")
        print(f"    OOS:   n={agg_oos['n']:>4} WR={agg_oos['wr']:.1f}% "
              f"Exp={agg_oos['mean_exp']:+.2f}% PF={agg_oos['pf']:.2f}")
        print(f"    Gap:   {gap:+.2f}% | "
              + ("✅ OOS duong" if agg_oos["mean_exp"] > 0.3 else
                 "⚠️  OOS yeu"  if agg_oos["mean_exp"] > 0 else
                 "❌ OOS am"))

    # ── Step 5: Save JSON ─────────────────────────────────────────────────────
    def strip_signals(results):
        out = []
        for r in results:
            if not r:
                continue
            rc = dict(r)
            for phase in ["train", "oos"]:
                if phase in rc:
                    rc[phase] = {k: v for k, v in rc[phase].items() if k != "signals"}
            if "mr_fwd" in rc:
                rc["mr_fwd"] = {
                    fk: {ph: {k: v for k, v in pv.items() if k != "signals"}
                         for ph, pv in fv.items()}
                    for fk, fv in rc["mr_fwd"].items()
                }
            out.append(rc)
        return out

    output = {
        "assignments":  assignments,
        "mr_symbols":   mr_syms,
        "mom_symbols":  mom_syms,
        "signal_config":SIGNAL_CONFIG,
        "mr_results":   strip_signals(mr_valid),
        "mom_results":  strip_signals(mom_valid),
    }
    out_path = Path("cluster_signal_results.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Da luu: {out_path}")
    print(f"  Buoc tiep: Review per-symbol OOS → chon ma vao watchlist")

    # ── So sánh S36 vs Original cho MOM ──────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"SO SÁNH MOM: Original (candle_body) vs S36 (candle_bull)")
    print(f"{'═'*65}")

    mom_results_s36 = []
    for sym in mom_syms:
        print(f"  [S36] {sym}...", end=" ", flush=True)
        # Temporarily swap config
        orig_cfg = SIGNAL_CONFIG["Momentum"]
        SIGNAL_CONFIG["Momentum"] = SIGNAL_CONFIG_S36["Momentum"]
        r = analyze_symbol(sym, "Momentum")
        SIGNAL_CONFIG["Momentum"] = orig_cfg
        mom_results_s36.append(r)
        if r:
            tr  = r.get("train", {})
            oos = r.get("oos",   {})
            print(f"train n={tr.get('n',0)} exp={tr.get('mean_exp',0):+.2f}% | "
                  f"oos n={oos.get('n',0)} exp={oos.get('mean_exp',0):+.2f}%")
        else:
            print("FAIL")

    mom_valid_s36 = [r for r in mom_results_s36 if r]
    agg_orig = aggregate_cluster(mom_valid,     FWD_TIER1, "train")
    agg_s36  = aggregate_cluster(mom_valid_s36, FWD_TIER1, "train")
    agg_orig_oos = aggregate_cluster(mom_valid,     FWD_TIER1, "oos")
    agg_s36_oos  = aggregate_cluster(mom_valid_s36, FWD_TIER1, "oos")

    print(f"\n  {'Version':<20} {'Train_n':>8} {'Train_exp':>10} {'OOS_n':>7} {'OOS_exp':>8}")
    print(f"  {'─'*60}")
    print(f"  {'Original (S34)':<20} {agg_orig['n']:>8} {agg_orig['mean_exp']:>+9.2f}% "
          f"{agg_orig_oos['n']:>7} {agg_orig_oos['mean_exp']:>+7.2f}%")
    print(f"  {'S36 (candle_bull)':<20} {agg_s36['n']:>8} {agg_s36['mean_exp']:>+9.2f}% "
          f"{agg_s36_oos['n']:>7} {agg_s36_oos['mean_exp']:>+7.2f}%")
    print(f"\n  Kết luận:")
    if agg_s36_oos['mean_exp'] > agg_orig_oos['mean_exp']:
        print(f"  → S36 OOS exp CAO hơn Original (+{agg_s36_oos['mean_exp']-agg_orig_oos['mean_exp']:.2f}%)")
        print(f"  → candle_bull discriminative hơn candle_body ✅ — giữ S36 fix")
    else:
        print(f"  → Original OOS exp CAO hơn S36 (+{agg_orig_oos['mean_exp']-agg_s36_oos['mean_exp']:.2f}%)")
        print(f"  → candle_body tốt hơn candle_bull ← quay về logic gốc")


if __name__ == "__main__":
    main()
