"""
analyze_cross_symbol.py — Cross-Symbol Feature Analysis
Session 30 — VN Trader Bot V6

Mục tiêu:
  Chạy feature analysis trên 27 mã universe, tổng hợp để tìm:
  1. Indicator nào có Cohen's d cao nhất CROSS-SYMBOL
     → đây là feature thật sự discriminative trên thị trường VN
  2. Mã nào có behavior tương đồng nhau
     → cơ sở để cluster

Logic:
  - Với mỗi mã: tính Cohen's d của 15 indicators (GOOD vs BAD)
  - GOOD = return > GOOD_THRESH sau FWD_DAYS
  - BAD  = return < BAD_THRESH sau FWD_DAYS
  - Tổng hợp cross-symbol:
      * Indicator nào có |d| >= 0.3 trên >= 60% số mã → pattern thật
      * Indicator nào chỉ mạnh ở 1-2 mã → mã-specific, không universal
  - Output: bảng ranked + heatmap dạng text + JSON

Chạy:
  python analyze_cross_symbol.py
  python analyze_cross_symbol.py VCB BID MBB   ← chỉ một số mã

Output:
  - cross_symbol_analysis.json
  - Console: bảng tổng hợp + per-symbol detail
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Universe 27 mã ────────────────────────────────────────────────────────────
UNIVERSE = {
    "Ngan hang":        ["VCB", "BID", "CTG", "MBB", "TCB", "STB"],
    "Chung khoan":      ["SSI", "VND", "HCM"],
    "Hoa chat/Phan bon":["DGC", "DCM", "DPM"],
    "Thep":             ["HPG", "NKG", "HSG"],
    "Bat dong san":     ["VHM", "NVL", "KDH"],
    "Tieu dung/Ban le": ["MWG", "FRT", "PNJ"],
    "Nang luong":       ["REE", "PC1", "GAS"],
    "Cong nghe/Logistics":["FPT", "GMD", "HAH"],
}

ALL_SYMBOLS = [s for group in UNIVERSE.values() for s in group]

# ── Config ────────────────────────────────────────────────────────────────────
FWD_DAYS    = 10
TRAIN_START = "2019-01-01"
TRAIN_END   = "2024-12-31"

# Relative threshold — tính riêng cho từng mã
# GOOD = return > mean + GOOD_STD_MULT * std của mã đó
# BAD  = return < mean - BAD_STD_MULT  * std của mã đó
GOOD_STD_MULT = 0.75   # top ~22% của distribution
BAD_STD_MULT  = 0.75   # bottom ~22% của distribution

# Ngưỡng để xác định "cross-symbol pattern thật"
CROSS_MIN_D    = 0.3   # Cohen's d tối thiểu để tính là "có signal"
CROSS_MIN_PCT  = 0.50  # >= 50% số mã có signal → pattern cross-symbol

# 15 indicators
ALL_INDICATORS = [
    "rsi_14", "macd_hist", "bb_position", "volume_spike",
    "trend_slope", "price_vs_sma20", "price_vs_sma50", "atr_ratio",
    "stoch_k", "ema_cross", "momentum_5d", "momentum_20d",
    "high_low_pos", "vol_trend", "candle_body",
]


# ── Indicator computation ─────────────────────────────────────────────────────

def _ema(c, span):
    return pd.Series(c).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c).rolling(p, min_periods=p).mean().values


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    vol    = df["volume"].values.astype(float)
    opn    = df["open"].values.astype(float)
    n      = len(df)

    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    macd   = ema12 - ema26
    sig    = _ema(macd, 9)
    hist   = macd - sig

    sma20  = _sma(close, 20)
    sma50  = _sma(close, 50)
    vsma5  = _sma(vol, 5)
    vsma20 = _sma(vol, 20)

    delta  = np.diff(close, prepend=close[0])
    gain   = np.where(delta > 0, delta, 0.0)
    loss   = np.where(delta < 0, -delta, 0.0)
    avg_g  = _ema(gain, 14)
    avg_l  = _ema(loss, 14)
    avg_l  = np.where(avg_l == 0, 1e-9, avg_l)
    rsi    = 100 - 100 / (1 + avg_g / avg_l)

    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr    = _sma(tr, 14)

    std20    = pd.Series(close).rolling(20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    lo14   = pd.Series(low).rolling(14).min().values
    hi14   = pd.Series(high).rolling(14).max().values
    denom  = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch  = 100 * (close - lo14) / denom

    rows = []
    for i in range(60, n):
        px     = close[i]
        atr_v  = atr[i]  if np.isfinite(atr[i])  else px * 0.02
        s20    = sma20[i] if np.isfinite(sma20[i]) else px
        s50    = sma50[i] if np.isfinite(sma50[i]) else px
        bb_u   = bb_upper[i] if np.isfinite(bb_upper[i]) else px * 1.04
        bb_l   = bb_lower[i] if np.isfinite(bb_lower[i]) else px * 0.96
        vs5v   = vsma5[i]  if np.isfinite(vsma5[i])  else vol[i]
        vs20v  = vsma20[i] if np.isfinite(vsma20[i]) else vol[i]
        c5     = close[max(i - 5,  0)]
        c20    = close[max(i - 20, 0)]
        hi20   = float(np.nanmax(high[max(i - 19, 0):i + 1]))
        lo20   = float(np.nanmin(low[max(i - 19,  0):i + 1]))

        bb_range = bb_u - bb_l
        bb_pos   = float(np.clip((px - bb_l) / bb_range, 0, 1)) if bb_range > 0 else 0.5
        hlp      = float((px - lo20) / (hi20 - lo20)) if hi20 > lo20 else 0.5
        body     = abs(px - opn[i]) / (atr_v + 1e-9)

        rows.append({
            "idx":            i,
            "date":           df["date"].iloc[i],
            "close":          px,
            "rsi_14":         float(rsi[i]),
            "macd_hist":      float(hist[i]),
            "bb_position":    bb_pos,
            "volume_spike":   float((vol[i] / (vs20v + 1e-9)) - 1.0),
            "trend_slope":    float((s20 - s50) / (px + 1e-9) * 100),
            "price_vs_sma20": float((px - s20) / (px + 1e-9) * 100),
            "price_vs_sma50": float((px - s50) / (px + 1e-9) * 100),
            "atr_ratio":      float(atr_v / (px + 1e-9) * 100),
            "stoch_k":        float(stoch[i]),
            "ema_cross":      float((ema12[i] - ema26[i]) / (px + 1e-9) * 100),
            "momentum_5d":    float((px / (c5  + 1e-9) - 1.0) * 100),
            "momentum_20d":   float((px / (c20 + 1e-9) - 1.0) * 100),
            "high_low_pos":   float(hlp),
            "vol_trend":      float((vs5v / (vs20v + 1e-9)) - 1.0),
            "candle_body":    float(np.clip(body, 0, 3)),
        })

    return pd.DataFrame(rows)


# ── Label + Cohen's d ─────────────────────────────────────────────────────────

def label_and_analyze(ind_df: pd.DataFrame, close_arr: np.ndarray, n_full: int) -> list[dict]:
    """
    Gán nhãn GOOD/BAD theo relative threshold của từng mã.
    GOOD = return > mean + 0.75*std  (top ~22%)
    BAD  = return < mean - 0.75*std  (bottom ~22%)
    → Mỗi mã tự so sánh với chính nó, không dùng ngưỡng tuyệt đối.
    """
    fwd_rets = []
    for i in ind_df["idx"].values:
        fi = i + FWD_DAYS
        fwd_rets.append(
            float((close_arr[fi] - close_arr[i]) / close_arr[i] * 100)
            if fi < n_full else np.nan
        )
    ind_df = ind_df.copy()
    ind_df["fwd_return"] = fwd_rets
    ind_df = ind_df.dropna(subset=["fwd_return"])

    # Tính ngưỡng relative từ distribution của mã này
    ret_mean = float(ind_df["fwd_return"].mean())
    ret_std  = float(ind_df["fwd_return"].std())
    good_thresh = ret_mean + GOOD_STD_MULT * ret_std
    bad_thresh  = ret_mean - BAD_STD_MULT  * ret_std

    ind_df["label"] = ind_df["fwd_return"].apply(
        lambda r: "GOOD" if r >= good_thresh else ("BAD" if r <= bad_thresh else "NEUTRAL")
    )

    # Ghi lại ngưỡng để debug
    ind_df.attrs["good_thresh"] = round(good_thresh, 2)
    ind_df.attrs["bad_thresh"]  = round(bad_thresh, 2)
    ind_df.attrs["ret_mean"]    = round(ret_mean, 2)
    ind_df.attrs["ret_std"]     = round(ret_std, 2)

    good = ind_df[ind_df["label"] == "GOOD"]
    bad  = ind_df[ind_df["label"] == "BAD"]

    results = []
    for ind in ALL_INDICATORS:
        if ind not in ind_df.columns:
            continue
        gv = good[ind].dropna().values
        bv = bad[ind].dropna().values
        if len(gv) < 5 or len(bv) < 5:
            continue

        # Cohen's d
        ng, nb = len(gv), len(bv)
        sg, sb = np.std(gv, ddof=1), np.std(bv, ddof=1)
        pooled = np.sqrt(((ng-1)*sg**2 + (nb-1)*sb**2) / (ng+nb-2))
        d = float((np.mean(gv) - np.mean(bv)) / pooled) if pooled > 1e-9 else 0.0

        results.append({
            "indicator":   ind,
            "cohen_d":     round(d, 3),
            "abs_d":       round(abs(d), 3),
            "good_median": round(float(np.median(gv)), 3),
            "bad_median":  round(float(np.median(bv)), 3),
            "n_good":      ng,
            "n_bad":       nb,
            "direction":   "GOOD>BAD" if d > 0 else "GOOD<BAD",
        })

    results.sort(key=lambda x: -x["abs_d"])
    return results


# ── Per-symbol analysis ───────────────────────────────────────────────────────

def analyze_symbol(symbol: str) -> dict | None:
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2200, min_bars=200)
        df["date"] = pd.to_datetime(df["date"])
        train = df[
            (df["date"] >= TRAIN_START) &
            (df["date"] <= TRAIN_END)
        ].reset_index(drop=True)
    except Exception as e:
        print(f"  [{symbol}] ERROR: {e}")
        return None

    if len(train) < 200:
        print(f"  [{symbol}] Khong du data ({len(train)} bars)")
        return None

    ind_df    = compute_indicators(train)
    close_arr = train["close"].values.astype(float)
    n_full    = len(train)

    results = label_and_analyze(ind_df, close_arr, n_full)

    # Tính n_good/n_bad với relative threshold nhất quán
    fwd_rets = []
    for i in ind_df["idx"].values:
        fi = i + FWD_DAYS
        fwd_rets.append(
            float((close_arr[fi] - close_arr[i]) / close_arr[i] * 100)
            if fi < n_full else np.nan
        )
    ind_df2 = ind_df.copy()
    ind_df2["fwd_return"] = fwd_rets
    ind_df2 = ind_df2.dropna(subset=["fwd_return"])
    n_total     = len(ind_df2)
    ret_mean    = float(ind_df2["fwd_return"].mean())
    ret_std     = float(ind_df2["fwd_return"].std())
    good_thresh = ret_mean + GOOD_STD_MULT * ret_std
    bad_thresh  = ret_mean - BAD_STD_MULT  * ret_std
    n_good      = int((ind_df2["fwd_return"] >= good_thresh).sum())
    n_bad       = int((ind_df2["fwd_return"] <= bad_thresh).sum())

    return {
        "symbol":      symbol,
        "n_bars":      len(train),
        "n_total":     n_total,
        "n_good":      n_good,
        "n_bad":       n_bad,
        "pct_good":    round(n_good / n_total * 100, 1) if n_total else 0,
        "pct_bad":     round(n_bad  / n_total * 100, 1) if n_total else 0,
        "good_thresh": round(good_thresh, 2),
        "bad_thresh":  round(bad_thresh, 2),
        "ret_mean":    round(ret_mean, 2),
        "ret_std":     round(ret_std, 2),
        "results":     results,
    }


# ── Cross-symbol aggregation ──────────────────────────────────────────────────

def aggregate_cross_symbol(per_symbol: dict[str, dict]) -> list[dict]:
    """
    Với mỗi indicator:
      - Đếm số mã có |d| >= CROSS_MIN_D
      - Tính mean/median Cohen's d cross-symbol
      - Xác định direction nhất quán không
      - Phân loại: Universal / Sector-specific / Noise
    """
    agg = {}
    for ind in ALL_INDICATORS:
        agg[ind] = {
            "indicator": ind,
            "d_values":  [],   # (symbol, d)
            "n_positive": 0,   # d > 0
            "n_negative": 0,   # d < 0
        }

    for sym, data in per_symbol.items():
        for r in data.get("results", []):
            ind = r["indicator"]
            if ind not in agg:
                continue
            agg[ind]["d_values"].append((sym, r["cohen_d"]))
            if r["cohen_d"] > 0:
                agg[ind]["n_positive"] += 1
            else:
                agg[ind]["n_negative"] += 1

    results = []
    n_syms  = len(per_symbol)

    for ind, info in agg.items():
        dvs = info["d_values"]
        if not dvs:
            continue

        ds      = [d for _, d in dvs]
        abs_ds  = [abs(d) for d in ds]
        n_signal= sum(1 for d in ds if abs(d) >= CROSS_MIN_D)
        pct_sig = n_signal / n_syms if n_syms else 0

        # Direction consistency: % mã cùng chiều
        n_pos = info["n_positive"]
        n_neg = info["n_negative"]
        n_nonzero = n_pos + n_neg
        dir_consistency = max(n_pos, n_neg) / n_nonzero if n_nonzero else 0
        dominant_dir    = "GOOD>BAD" if n_pos >= n_neg else "GOOD<BAD"

        # Phân loại
        if pct_sig >= CROSS_MIN_PCT and dir_consistency >= 0.7:
            category = "Universal"
        elif pct_sig >= 0.3 and dir_consistency >= 0.6:
            category = "Partial"
        else:
            category = "Noise/Specific"

        results.append({
            "indicator":       ind,
            "mean_abs_d":      round(float(np.mean(abs_ds)), 3),
            "median_abs_d":    round(float(np.median(abs_ds)), 3),
            "mean_d":          round(float(np.mean(ds)), 3),
            "n_signal":        n_signal,
            "pct_signal":      round(pct_sig * 100, 1),
            "dir_consistency": round(dir_consistency * 100, 1),
            "dominant_dir":    dominant_dir,
            "category":        category,
            "per_symbol_d":    {sym: round(d, 3) for sym, d in dvs},
        })

    results.sort(key=lambda x: -x["mean_abs_d"])
    return results


# ── Behavior profile cho clustering ──────────────────────────────────────────

def compute_behavior_profile(per_symbol: dict[str, dict],
                              cross_results: list[dict]) -> dict[str, dict]:
    """
    Từ kết quả cross-symbol, tạo behavior profile cho từng mã.
    Profile gồm Cohen's d của các Universal indicators.
    Đây là input cho clustering ở bước sau.
    """
    universal = [r["indicator"] for r in cross_results
                 if r["category"] == "Universal"]
    partial   = [r["indicator"] for r in cross_results
                 if r["category"] == "Partial"]
    cluster_features = (universal + partial)[:8]  # top 8 features

    profiles = {}
    for sym, data in per_symbol.items():
        d_map = {r["indicator"]: r["cohen_d"] for r in data.get("results", [])}
        profile = {feat: d_map.get(feat, 0.0) for feat in cluster_features}
        profiles[sym] = {
            "symbol":   sym,
            "profile":  profile,
            "n_good":   data["n_good"],
            "n_bad":    data["n_bad"],
            "pct_good": data["pct_good"],
        }

    return profiles, cluster_features


# ── Text heatmap ──────────────────────────────────────────────────────────────

def print_heatmap(cross_results: list[dict], per_symbol: dict[str, dict]):
    """
    In heatmap dạng text: rows = indicators, cols = symbols.
    Ký hiệu: ++ (d>0.5), + (d>0.3), · (d>0.1), _ (|d|<=0.1), - (d<-0.3), -- (d<-0.5)
    """
    # Chỉ lấy top 10 indicators
    top_inds = [r["indicator"] for r in cross_results[:10]]
    symbols  = sorted(per_symbol.keys())

    def cell(d):
        if d >= 0.5:   return "++"
        if d >= 0.3:   return " +"
        if d >= 0.1:   return " ·"
        if d >= -0.1:  return "  "
        if d >= -0.3:  return " -"
        return "--"

    # Header
    sym_short = [s[:4] for s in symbols]
    print(f"\n{'HEATMAP Cohen_d (++ strong positive, -- strong negative)'}")
    print(f"{'Indicator':<18} " + " ".join(f"{s:<4}" for s in sym_short))
    print("─" * (18 + len(symbols) * 5))

    for ind in top_inds:
        d_map = cross_results[next(
            i for i, r in enumerate(cross_results) if r["indicator"] == ind
        )]["per_symbol_d"]
        row = f"{ind:<18} "
        for sym in symbols:
            d = d_map.get(sym, 0.0)
            row += f"{cell(d):<5}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ALL_SYMBOLS

    print(f"{'='*65}")
    print(f"CROSS-SYMBOL FEATURE ANALYSIS (Relative Threshold)")
    print(f"Universe: {len(symbols)} ma | FWD={FWD_DAYS}d | Training {TRAIN_START}→{TRAIN_END}")
    print(f"GOOD = mean + {GOOD_STD_MULT}*std | BAD = mean - {BAD_STD_MULT}*std (per-symbol)")
    print(f"{'='*65}")

    # ── Per-symbol analysis ───────────────────────────────────────────────────
    per_symbol = {}
    failed     = []

    for sym in symbols:
        print(f"  Analyzing {sym}...", end=" ", flush=True)
        result = analyze_symbol(sym)
        if result:
            per_symbol[sym] = result
            print(f"OK (good={result['n_good']}, bad={result['n_bad']})")
        else:
            failed.append(sym)

    if failed:
        print(f"\n  FAILED: {failed}")

    if len(per_symbol) < 3:
        print("Qua it ma thanh cong. Dung lai.")
        return

    # ── Cross-symbol aggregation ──────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"CROSS-SYMBOL AGGREGATION ({len(per_symbol)} ma thanh cong)")
    print(f"{'─'*65}")

    cross_results = aggregate_cross_symbol(per_symbol)

    # Print bảng tổng hợp
    universal = [r for r in cross_results if r["category"] == "Universal"]
    partial   = [r for r in cross_results if r["category"] == "Partial"]
    noise     = [r for r in cross_results if r["category"] == "Noise/Specific"]

    print(f"\n  {'Indicator':<18} {'MeanAbsD':>9} {'%Signal':>8} "
          f"{'DirConsis':>10} {'Category':<16} Direction")
    print(f"  {'─'*75}")

    for r in cross_results:
        em = "★★" if r["category"] == "Universal" else \
             "★ " if r["category"] == "Partial"   else "  "
        print(
            f"  {em}{r['indicator']:<16} {r['mean_abs_d']:>9.3f} "
            f"{r['pct_signal']:>7.1f}% {r['dir_consistency']:>9.1f}% "
            f"{r['category']:<16} {r['dominant_dir']}"
        )

    # Summary
    print(f"\n  PHAN LOAI:")
    print(f"  ★★ Universal  ({len(universal)} indicators): "
          f"{[r['indicator'] for r in universal]}")
    print(f"  ★  Partial    ({len(partial)} indicators): "
          f"{[r['indicator'] for r in partial]}")
    print(f"     Noise/Spec ({len(noise)} indicators): "
          f"{[r['indicator'] for r in noise]}")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    print_heatmap(cross_results, per_symbol)

    # ── Behavior profiles cho clustering ─────────────────────────────────────
    profiles, cluster_features = compute_behavior_profile(per_symbol, cross_results)

    print(f"\n{'─'*65}")
    print(f"BEHAVIOR PROFILES (input cho clustering)")
    print(f"Features su dung: {cluster_features}")
    print(f"{'─'*65}")

    # In profile từng mã theo nhóm ngành — kèm threshold để verify
    for sector, syms in UNIVERSE.items():
        print(f"\n  [{sector}]")
        for sym in syms:
            if sym not in profiles:
                continue
            p        = profiles[sym]
            sym_data = per_symbol.get(sym, {})
            thresh_str = (f"good>{sym_data.get('good_thresh','?')}% "
                          f"bad<{sym_data.get('bad_thresh','?')}% "
                          f"mean={sym_data.get('ret_mean','?')} "
                          f"std={sym_data.get('ret_std','?')}")
            vals = " | ".join(
                f"{feat.split('_')[0][:4]}={p['profile'].get(feat, 0):+.2f}"
                for feat in cluster_features[:5]
            ) if cluster_features else "(no universal features yet)"
            print(f"    {sym}: {vals}")
            print(f"         [{thresh_str}] n_good={p['n_good']} n_bad={p['n_bad']}")

    # ── Kết luận ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"KET LUAN")
    print(f"{'='*65}")

    if universal:
        print(f"\n  INDICATORS UNIVERSAL (dung lam cluster features):")
        for r in universal:
            interp = ""
            if r["indicator"] == "momentum_5d" and r["dominant_dir"] == "GOOD>BAD":
                interp = "→ Mua khi momentum ngan han duong"
            elif r["indicator"] == "stoch_k" and r["dominant_dir"] == "GOOD<BAD":
                interp = "→ Mua khi oversold"
            elif r["indicator"] == "volume_spike" and r["dominant_dir"] == "GOOD>BAD":
                interp = "→ Volume dot bien xac nhan tang"
            elif r["indicator"] == "atr_ratio":
                interp = "→ Volatility regime indicator"
            elif r["indicator"] == "candle_body" and r["dominant_dir"] == "GOOD>BAD":
                interp = "→ Nen than lon = mua manh"
            print(f"  ★★ {r['indicator']:<18} MeanD={r['mean_abs_d']:.3f} "
                  f"Dir={r['dominant_dir']} {interp}")

    print(f"\n  BUOC TIEP THEO:")
    print(f"  1. Review heatmap — xem ma nao co profile tuong dong nhau")
    print(f"  2. Dung {len(cluster_features)} features tren de cluster 27 ma")
    print(f"  3. Validate cluster bang ly luan kinh te")

    # ── Lưu JSON ──────────────────────────────────────────────────────────────
    output = {
        "universe":        symbols,
        "n_success":       len(per_symbol),
        "failed":          failed,
        "fwd_days":        FWD_DAYS,
        "good_thresh":     GOOD_THRESH,
        "bad_thresh":      BAD_THRESH,
        "cross_results":   cross_results,
        "cluster_features":cluster_features,
        "per_symbol":      {
            sym: {
                "n_bars":   d["n_bars"],
                "n_good":   d["n_good"],
                "n_bad":    d["n_bad"],
                "pct_good": d["pct_good"],
                "top5": d["results"][:5],
            }
            for sym, d in per_symbol.items()
        },
        "behavior_profiles": {
            sym: p["profile"]
            for sym, p in profiles.items()
        },
    }

    out_path = Path("cross_symbol_analysis.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Da luu: {out_path}")
    print(f"  Buoc tiep: python cluster_symbols.py  (dung output nay lam input)")


if __name__ == "__main__":
    main()
