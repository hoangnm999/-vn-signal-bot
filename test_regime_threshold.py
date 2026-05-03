"""
test_regime_threshold.py — So sánh Regime Threshold A vs B
Session 30 — VN Trader Bot V6

Hypothesis:
  Ngưỡng median tính từ 2019-2024 (Cách A) có thể không còn phù hợp
  với market 2025. Tính lại từ 2023-2024 (Cách B) sẽ cải thiện OOS.

So sánh:
  Cách A: median(2019-2024) — hiện tại V3
  Cách B: median(2023-2024) — 2 năm gần nhất trước OOS

Metrics so sánh trên OOS 2025:
  1. n_pass_regime  : số ngày lọt qua regime filter
  2. n_signals      : số ngày lọt qua cả regime + trigger
  3. WR, Exp, PF    : chất lượng signals

Chạy:
  python test_regime_threshold.py MWG
  python test_regime_threshold.py MWG STB DPM
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FWD_DAYS       = 10
WIN_THRESH     = 1.0
TRIGGER_PCT    = 70
MIN_TRIGGERS   = 2
WF_COOLDOWN    = 5

TRAIN_START    = "2019-01-01"
THRESH_A_END   = "2024-12-31"   # Cách A: toàn bộ training
THRESH_B_START = "2023-01-01"   # Cách B: 2 năm gần nhất
THRESH_B_END   = "2024-12-31"
OOS_START      = "2025-01-01"

# Regime + Trigger config — nhất quán với V3
REGIME_CONFIG = {
    "MWG": {"regime_indicator": "atr_ratio",   "regime_condition": "low"},
    "STB": {"regime_indicator": "atr_ratio",   "regime_condition": "high"},
    "DPM": {"regime_indicator": "trend_slope", "regime_condition": "low"},
}
REGIME_CONFIG_FALLBACK = {"regime_indicator": "atr_ratio", "regime_condition": "low"}

TRIGGER_CONFIG = {
    "MWG": {
        "indicators": ["stoch_k", "momentum_5d", "volume_spike", "candle_body"],
        "direction":  {"stoch_k": "low", "momentum_5d": "high",
                       "volume_spike": "high", "candle_body": "high"},
    },
    "STB": {
        "indicators": ["momentum_5d", "volume_spike", "candle_body"],
        "direction":  {"momentum_5d": "high", "volume_spike": "high",
                       "candle_body": "high"},
    },
    "DPM": {
        "indicators": ["momentum_5d", "volume_spike", "stoch_k", "candle_body"],
        "direction":  {"momentum_5d": "high", "volume_spike": "high",
                       "stoch_k": "low", "candle_body": "high"},
    },
}
TRIGGER_CONFIG_FALLBACK = {
    "indicators": ["momentum_5d", "volume_spike", "stoch_k", "candle_body"],
    "direction":  {"momentum_5d": "high", "volume_spike": "high",
                   "stoch_k": "low", "candle_body": "high"},
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
        atr_v = atr[i] if np.isfinite(atr[i]) else px * 0.02
        s20   = sma20[i] if np.isfinite(sma20[i]) else px
        s50   = sma50[i] if np.isfinite(sma50[i]) else px
        vs20v = vsma20[i] if np.isfinite(vsma20[i]) else vol[i]
        c5    = close[max(i - 5, 0)]

        rows.append({
            "idx":          i,
            "atr_ratio":    float(atr_v / (px + 1e-9) * 100),
            "trend_slope":  float((s20 - s50) / (px + 1e-9) * 100),
            "momentum_20d": float((px / close[max(i - 20, 0)] - 1) * 100),
            "momentum_5d":  float((px / (c5 + 1e-9) - 1) * 100),
            "volume_spike": float((vol[i] / (vs20v + 1e-9)) - 1.0),
            "stoch_k":      float(stoch[i]),
            "candle_body":  float(np.clip(
                abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
        })
    return rows


# ── Threshold computation ─────────────────────────────────────────────────────

def compute_thresholds(ind_rows: list[dict], symbol: str) -> dict:
    """
    Tính regime threshold + trigger thresholds từ một tập rows cho trước.
    Trả về dict đầy đủ để chạy OOS.
    """
    cfg_r = REGIME_CONFIG.get(symbol, REGIME_CONFIG_FALLBACK)
    cfg_t = TRIGGER_CONFIG.get(symbol, TRIGGER_CONFIG_FALLBACK)

    reg_ind  = cfg_r["regime_indicator"]
    trig_ind = cfg_t["indicators"]
    trig_dir = cfg_t["direction"]

    # Regime threshold
    reg_vals   = [r[reg_ind] for r in ind_rows if np.isfinite(r.get(reg_ind, np.nan))]
    reg_thresh = float(np.median(reg_vals)) if reg_vals else 0.0

    # Trigger thresholds
    trig_thresh = {}
    for t in trig_ind:
        vals = [r[t] for r in ind_rows if np.isfinite(r.get(t, np.nan))]
        if not vals:
            continue
        if trig_dir.get(t, "high") == "low":
            trig_thresh[t] = float(np.percentile(vals, 100 - TRIGGER_PCT))
        else:
            trig_thresh[t] = float(np.percentile(vals, TRIGGER_PCT))

    return {
        "reg_thresh":  reg_thresh,
        "trig_thresh": trig_thresh,
        "reg_median":  reg_thresh,  # alias để in
        "reg_vals_mean": float(np.mean(reg_vals)) if reg_vals else 0.0,
        "reg_vals_std":  float(np.std(reg_vals)) if reg_vals else 0.0,
    }


# ── OOS evaluation ────────────────────────────────────────────────────────────

def run_oos(
    symbol: str,
    ind_map: dict,
    close_arr: np.ndarray,
    oos_indices: list[int],
    n_bars: int,
    thresholds: dict,
    label: str,
) -> dict:
    """
    Chạy OOS loop với threshold cho trước.
    Trả về dict metrics + signal list.
    """
    cfg_r = REGIME_CONFIG.get(symbol, REGIME_CONFIG_FALLBACK)
    cfg_t = TRIGGER_CONFIG.get(symbol, TRIGGER_CONFIG_FALLBACK)

    reg_ind   = cfg_r["regime_indicator"]
    reg_cond  = cfg_r["regime_condition"]
    trig_ind  = cfg_t["indicators"]
    trig_dir  = cfg_t["direction"]

    reg_thresh  = thresholds["reg_thresh"]
    trig_thresh = thresholds["trig_thresh"]

    n_skip_regime  = 0
    n_skip_trigger = 0
    n_pass_regime  = 0
    last_signal    = None
    signals        = []

    for t_idx in oos_indices:
        row = ind_map.get(t_idx)
        if row is None:
            continue

        # Cooldown
        if last_signal and (t_idx - last_signal) < WF_COOLDOWN:
            continue

        # Tầng 1: regime
        val = row.get(reg_ind, np.nan)
        if not np.isfinite(val):
            continue
        in_regime = (val <= reg_thresh) if reg_cond == "low" else (val > reg_thresh)
        if not in_regime:
            n_skip_regime += 1
            continue
        n_pass_regime += 1

        # Tầng 2: trigger
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
            n_skip_trigger += 1
            continue

        # Signal
        fwd_idx = t_idx + FWD_DAYS
        pending = fwd_idx >= n_bars
        actual  = None if pending else float(
            (close_arr[fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100
        )

        signals.append({
            "t_idx":   t_idx,
            "actual":  actual,
            "pending": pending,
            "n_trig":  n_trig,
            "reg_val": round(val, 3),
        })
        last_signal = t_idx

    # Metrics
    done = [s for s in signals if not s["pending"] and s["actual"] is not None]
    if done:
        rets  = [s["actual"] for s in done]
        wins  = [r for r in rets if r >= WIN_THRESH]
        loss  = [r for r in rets if r < WIN_THRESH]
        wr    = len(wins) / len(rets) * 100
        mean_r = float(np.mean(rets))
        pf    = round(sum(wins) / abs(sum(loss)), 2) if loss and sum(loss) != 0 else 99.0
        max_dd = float(np.min(rets))
    else:
        wr = mean_r = pf = max_dd = 0.0
        rets = []

    total_oos = len(oos_indices)
    pct_pass  = n_pass_regime / total_oos * 100 if total_oos else 0

    return {
        "label":          label,
        "reg_thresh":     round(reg_thresh, 3),
        "n_oos_bars":     total_oos,
        "n_pass_regime":  n_pass_regime,
        "pct_pass_regime":round(pct_pass, 1),
        "n_skip_regime":  n_skip_regime,
        "n_skip_trigger": n_skip_trigger,
        "n_signals":      len(done),
        "n_pending":      len(signals) - len(done),
        "wr":             round(wr, 1),
        "mean_exp":       round(mean_r, 2),
        "pf":             round(pf, 2),
        "max_dd":         round(max_dd, 1) if rets else 0.0,
        "signals":        signals,
    }


# ── Distribution comparison ───────────────────────────────────────────────────

def compare_distributions(
    train_rows: list[dict],
    thresh_b_rows: list[dict],
    oos_rows: list[dict],
    reg_ind: str,
) -> dict:
    """
    So sánh phân phối của regime indicator qua 3 giai đoạn.
    Giúp hiểu tại sao threshold cũ không còn phù hợp.
    """
    def stats(rows, label):
        vals = [r[reg_ind] for r in rows if np.isfinite(r.get(reg_ind, np.nan))]
        if not vals:
            return {}
        return {
            "label":  label,
            "n":      len(vals),
            "mean":   round(float(np.mean(vals)), 3),
            "median": round(float(np.median(vals)), 3),
            "std":    round(float(np.std(vals)), 3),
            "p25":    round(float(np.percentile(vals, 25)), 3),
            "p75":    round(float(np.percentile(vals, 75)), 3),
        }

    return {
        "2019_2024": stats(train_rows,   "Training (2019-2024)"),
        "2023_2024": stats(thresh_b_rows,"Thresh B  (2023-2024)"),
        "2025_oos":  stats(oos_rows,     "OOS       (2025+)   "),
    }


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_threshold(symbol: str):
    print(f"\n{'='*65}")
    print(f"REGIME THRESHOLD ANALYSIS — {symbol}")
    print(f"Cach A: median({TRAIN_START}→{THRESH_A_END})")
    print(f"Cach B: median({THRESH_B_START}→{THRESH_B_END})")
    print(f"OOS: {OOS_START}+  |  FWD={FWD_DAYS}d")
    print(f"{'='*65}")

    # Load data
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2500, min_bars=300)
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        print(f"ERROR load data: {e}")
        return None

    # Tách các giai đoạn
    mask_train   = (df["date"] >= TRAIN_START)   & (df["date"] <= THRESH_A_END)
    mask_thresh_b= (df["date"] >= THRESH_B_START) & (df["date"] <= THRESH_B_END)
    mask_oos     = df["date"] >= OOS_START

    df_full  = df.reset_index(drop=True)
    n_bars   = len(df_full)
    close_arr= df_full["close"].values.astype(float)

    # Tính indicators trên toàn bộ df (cần context đủ dài cho warmup)
    all_rows = compute_indicators(df_full)
    ind_map  = {r["idx"]: r for r in all_rows}

    # Lấy index ranges cho từng giai đoạn
    train_indices   = df_full[mask_train].index.tolist()
    thresh_b_indices= df_full[mask_thresh_b].index.tolist()
    oos_indices     = df_full[mask_oos].index.tolist()

    train_rows   = [ind_map[i] for i in train_indices   if i in ind_map]
    thresh_b_rows= [ind_map[i] for i in thresh_b_indices if i in ind_map]
    oos_rows     = [ind_map[i] for i in oos_indices      if i in ind_map]

    print(f"\nData bars: train={len(train_rows)} | thresh_b={len(thresh_b_rows)} | oos={len(oos_rows)}")

    if len(thresh_b_rows) < 100:
        print(f"CANH BAO: Qua it data cho Cach B ({len(thresh_b_rows)} bars)")
        return None

    # ── Phân phối regime indicator ────────────────────────────────────────────
    reg_ind = REGIME_CONFIG.get(symbol, REGIME_CONFIG_FALLBACK)["regime_indicator"]
    dist    = compare_distributions(train_rows, thresh_b_rows, oos_rows, reg_ind)

    print(f"\nPHAN PHOI '{reg_ind}' QUA CAC GIAI DOAN:")
    print(f"  {'Giai doan':<25} {'N':>5} {'Mean':>7} {'Median':>8} "
          f"{'Std':>7} {'P25':>7} {'P75':>7}")
    print(f"  {'─'*65}")
    for key, s in dist.items():
        if not s:
            continue
        print(f"  {s['label']:<25} {s['n']:>5} {s['mean']:>7.3f} "
              f"{s['median']:>8.3f} {s['std']:>7.3f} "
              f"{s['p25']:>7.3f} {s['p75']:>7.3f}")

    # Nhận xét về distribution shift
    med_a   = dist["2019_2024"].get("median", 0)
    med_b   = dist["2023_2024"].get("median", 0)
    med_oos = dist["2025_oos"].get("median", 0)
    shift_pct = (med_oos - med_a) / med_a * 100 if med_a else 0

    print(f"\n  Regime shift: median {TRAIN_START[:4]}-{THRESH_A_END[:4]}={med_a:.3f} "
          f"→ OOS 2025={med_oos:.3f} ({shift_pct:+.1f}%)")
    if abs(shift_pct) >= 10:
        print(f"  ⚠️  SHIFT DANG KE ({shift_pct:+.1f}%) — Cach B co the giup")
    elif abs(shift_pct) >= 5:
        print(f"  · Shift nhe ({shift_pct:+.1f}%) — Cach B co the giup mot phan")
    else:
        print(f"  ✓ Shift nho ({shift_pct:+.1f}%) — Cach A va B ket qua tuong duong")

    # ── Tính thresholds ───────────────────────────────────────────────────────
    thresh_a = compute_thresholds(train_rows,    symbol)
    thresh_b = compute_thresholds(thresh_b_rows, symbol)

    print(f"\nNGUONG REGIME ({reg_ind}):")
    print(f"  Cach A (2019-2024): {thresh_a['reg_thresh']:.3f}")
    print(f"  Cach B (2023-2024): {thresh_b['reg_thresh']:.3f}")
    print(f"  OOS median:         {med_oos:.3f}")

    # ── Chạy OOS với cả 2 threshold ───────────────────────────────────────────
    if not oos_indices:
        print(f"\nKhong co OOS data sau {OOS_START}")
        return None

    res_a = run_oos(symbol, ind_map, close_arr, oos_indices, n_bars, thresh_a, "A")
    res_b = run_oos(symbol, ind_map, close_arr, oos_indices, n_bars, thresh_b, "B")

    # ── Print kết quả so sánh ─────────────────────────────────────────────────
    print(f"\nKET QUA OOS (2025+):")
    print(f"  {'Metric':<22} {'Cach A':>10} {'Cach B':>10}  {'Delta':>8}")
    print(f"  {'─'*55}")

    metrics = [
        ("Regime thresh",    f"{res_a['reg_thresh']:.3f}",  f"{res_b['reg_thresh']:.3f}", ""),
        ("Pass regime (n)",  res_a['n_pass_regime'],         res_b['n_pass_regime'],
         res_b['n_pass_regime'] - res_a['n_pass_regime']),
        ("Pass regime (%)",  f"{res_a['pct_pass_regime']}%", f"{res_b['pct_pass_regime']}%", ""),
        ("Signals (n)",      res_a['n_signals'],             res_b['n_signals'],
         res_b['n_signals'] - res_a['n_signals']),
        ("WR (%)",           f"{res_a['wr']}%",              f"{res_b['wr']}%",
         f"{res_b['wr'] - res_a['wr']:+.1f}%"),
        ("Mean Exp (%)",     f"{res_a['mean_exp']:+.2f}%",   f"{res_b['mean_exp']:+.2f}%",
         f"{res_b['mean_exp'] - res_a['mean_exp']:+.2f}%"),
        ("PF",               res_a['pf'],                    res_b['pf'],
         f"{res_b['pf'] - res_a['pf']:+.2f}"),
        ("Max DD (%)",       f"{res_a['max_dd']:.1f}%",      f"{res_b['max_dd']:.1f}%",
         f"{res_b['max_dd'] - res_a['max_dd']:+.1f}%"),
    ]

    for name, va, vb, delta in metrics:
        delta_str = str(delta) if delta != "" else "—"
        print(f"  {name:<22} {str(va):>10} {str(vb):>10}  {delta_str:>8}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n  {'─'*55}")
    exp_a = res_a['mean_exp']
    exp_b = res_b['mean_exp']
    sig_a = res_a['n_signals']
    sig_b = res_b['n_signals']

    if exp_b > exp_a and sig_b >= sig_a * 0.7:
        verdict = f"✅ Cach B TOT HON: Exp {exp_a:+.2f}% → {exp_b:+.2f}% (n={sig_a}→{sig_b})"
    elif exp_b > exp_a and sig_b < sig_a * 0.7:
        verdict = (f"⚠️  Cach B Exp cao hon ({exp_b:+.2f}% vs {exp_a:+.2f}%) "
                   f"NHUNG it signal hon nhieu ({sig_a}→{sig_b}) — can xem xet")
    elif abs(exp_b - exp_a) < 0.1:
        verdict = f"· Cach A va B TUONG DUONG — regime shift khong phai nguyen nhan chinh"
    else:
        verdict = f"❌ Cach B KEM HON: Exp {exp_a:+.2f}% → {exp_b:+.2f}%"

    print(f"  VERDICT: {verdict}")

    # ── Recent signals comparison ─────────────────────────────────────────────
    print(f"\n  RECENT OOS SIGNALS:")
    sigs_a = [s for s in res_a["signals"] if not s["pending"]][-5:]
    sigs_b = [s for s in res_b["signals"] if not s["pending"]][-5:]

    # Map t_idx → date
    idx2date = {i: str(df_full["date"].iloc[i])[:10]
                for i in range(n_bars) if i < n_bars}

    print(f"  Cach A (n={res_a['n_signals']}):")
    if sigs_a:
        for s in sigs_a:
            date = idx2date.get(s["t_idx"], "?")
            print(f"    {date} reg={s['reg_val']} trig={s['n_trig']} "
                  f"→ {s['actual']:+.1f}%")
    else:
        print(f"    (khong co signal)")

    print(f"  Cach B (n={res_b['n_signals']}):")
    if sigs_b:
        for s in sigs_b:
            date = idx2date.get(s["t_idx"], "?")
            print(f"    {date} reg={s['reg_val']} trig={s['n_trig']} "
                  f"→ {s['actual']:+.1f}%")
    else:
        print(f"    (khong co signal)")

    # ── Lưu JSON ──────────────────────────────────────────────────────────────
    output = {
        "symbol":      symbol,
        "reg_ind":     reg_ind,
        "distribution": dist,
        "thresh_a":    thresh_a["reg_thresh"],
        "thresh_b":    thresh_b["reg_thresh"],
        "oos_median":  med_oos,
        "shift_pct":   round(shift_pct, 1),
        "result_a":    {k: v for k, v in res_a.items() if k != "signals"},
        "result_b":    {k: v for k, v in res_b.items() if k != "signals"},
        "verdict":     verdict,
    }
    out_path = Path(f"regime_threshold_{symbol}.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Da luu: {out_path}")
    return output


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["MWG"]

    results = {}
    for sym in symbols:
        r = analyze_threshold(sym.upper())
        if r:
            results[sym.upper()] = r

    # Cross-symbol summary
    if len(results) > 1:
        print(f"\n{'#'*65}")
        print(f"CROSS-SYMBOL SUMMARY")
        print(f"{'#'*65}")
        print(f"  {'Symbol':<6} {'Shift%':>7} {'ExpA':>7} {'ExpB':>7} "
              f"{'Delta':>7} {'nA':>4} {'nB':>4}  Verdict")
        print(f"  {'─'*65}")
        for sym, r in results.items():
            ra = r["result_a"]
            rb = r["result_b"]
            delta = rb["mean_exp"] - ra["mean_exp"]
            em = "✅" if delta > 0.1 else ("❌" if delta < -0.1 else "·")
            print(
                f"  {sym:<6} {r['shift_pct']:>+7.1f}% "
                f"{ra['mean_exp']:>+7.2f}% {rb['mean_exp']:>+7.2f}% "
                f"{delta:>+7.2f}% {ra['n_signals']:>4} {rb['n_signals']:>4}  {em}"
            )

        # Kết luận tổng
        print(f"\n  KET LUAN TONG:")
        improved = [s for s, r in results.items()
                    if r["result_b"]["mean_exp"] > r["result_a"]["mean_exp"] + 0.1]
        neutral  = [s for s, r in results.items()
                    if abs(r["result_b"]["mean_exp"] - r["result_a"]["mean_exp"]) <= 0.1]
        worse    = [s for s, r in results.items()
                    if r["result_b"]["mean_exp"] < r["result_a"]["mean_exp"] - 0.1]

        if improved:
            print(f"  ✅ Cach B giup ich: {improved}")
        if neutral:
            print(f"  ·  Khong doi: {neutral} — tim nguyen nhan khac")
        if worse:
            print(f"  ❌ Cach B kem hon: {worse} — giu nguyen Cach A")
