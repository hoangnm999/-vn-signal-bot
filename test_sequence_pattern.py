"""
test_sequence_pattern.py — Exploratory Sequence Pattern Analysis
Session 30 — VN Trader Bot V6

Mục tiêu:
  Không assume pattern trước. Để data cho thấy N ngày trước T
  của B_good và B_bad trông khác nhau như thế nào.

Logic:
  1. Lấy tập V3 signal (regime đúng + trigger >= 2) từ training
  2. Chia thành B_good (actual_return >= WIN_THRESH) và B_bad
  3. Với mỗi ngày T trong 2 nhóm, nhìn lại N=3 và N=5 ngày
  4. Tính 3 loại sequence feature:
       a. mean_N     : giá trị trung bình N ngày
       b. trend_N    : hệ số hồi quy tuyến tính (dương = tăng dần)
       c. days_above : số ngày vượt ngưỡng percentile 70 trong N ngày
  5. So sánh phân phối B_good vs B_bad → Cohen's d
  6. In bảng ranked theo |Cohen's d| để thấy feature nào phân biệt nhất

Chạy:
  python test_sequence_pattern.py MWG
  python test_sequence_pattern.py MWG STB DPM

Output:
  - In bảng phân tích ra console
  - Lưu JSON: seq_analysis_<SYMBOL>.json
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FWD_DAYS    = 10
WIN_THRESH  = 1.0      # return > 1% = win
TRAIN_START = "2019-01-01"
TRAIN_END   = "2024-12-31"
SEQ_WINDOWS = [3, 5]   # N ngày nhìn lại
TRIGGER_PCT = 70       # top 30% = "đột biến" (nhất quán với V3)

# Chỉ số để phân tích sequence (giống V3)
SEQ_INDICATORS = [
    "momentum_5d",
    "volume_spike",
    "stoch_k",
    "candle_body",
    "atr_ratio",
    "trend_slope",
]

# Regime config — nhất quán với V3
REGIME_CONFIG = {
    "MWG": {"regime_indicator": "atr_ratio",   "regime_condition": "low"},
    "STB": {"regime_indicator": "atr_ratio",   "regime_condition": "high"},
    "DPM": {"regime_indicator": "trend_slope", "regime_condition": "low"},
}
REGIME_CONFIG_FALLBACK = {"regime_indicator": "atr_ratio", "regime_condition": "low"}

# Trigger config — nhất quán với V3
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
MIN_TRIGGERS = 2


# ── Helpers tính indicator ────────────────────────────────────────────────────

def _ema(c, span):
    return pd.Series(c).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c).rolling(p, min_periods=p).mean().values


def compute_indicators(df: pd.DataFrame) -> list[dict]:
    """
    Tính đầy đủ indicators cho từng ngày.
    Trả về list dict (giống _compute_indicators_v3 trong V3).
    """
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
            "candle_body":  float(np.clip(abs(px - opn[i]) / (atr_v + 1e-9), 0, 3)),
        })
    return rows


# ── V3 signal detection ───────────────────────────────────────────────────────

def get_v3_signals(symbol: str, ind_rows: list[dict], close_arr, n_bars: int) -> list[dict]:
    """
    Tái tạo V3 signal detection trên training data.
    Trả về list dict gồm: t_idx, actual_return, win/loss.
    """
    cfg_r  = REGIME_CONFIG.get(symbol, REGIME_CONFIG_FALLBACK)
    cfg_t  = TRIGGER_CONFIG.get(symbol, TRIGGER_CONFIG_FALLBACK)

    reg_ind  = cfg_r["regime_indicator"]
    reg_cond = cfg_r["regime_condition"]
    trig_ind = cfg_t["indicators"]
    trig_dir = cfg_t["direction"]

    # Tính ngưỡng từ training
    reg_vals  = [r[reg_ind] for r in ind_rows if np.isfinite(r[reg_ind])]
    reg_thresh = float(np.median(reg_vals))

    trig_thresh = {}
    for t in trig_ind:
        vals = [r[t] for r in ind_rows if np.isfinite(r[t])]
        if not vals:
            continue
        if trig_dir.get(t, "high") == "low":
            trig_thresh[t] = float(np.percentile(vals, 100 - TRIGGER_PCT))
        else:
            trig_thresh[t] = float(np.percentile(vals, TRIGGER_PCT))

    ind_map = {r["idx"]: r for r in ind_rows}
    signals = []

    for t_idx in range(120, n_bars - FWD_DAYS - 1, 3):
        row = ind_map.get(t_idx)
        if row is None:
            continue

        # Tầng 1: regime
        val = row.get(reg_ind, np.nan)
        if not np.isfinite(val):
            continue
        in_regime = (val <= reg_thresh) if reg_cond == "low" else (val > reg_thresh)
        if not in_regime:
            continue

        # Tầng 2: trigger
        n_trig = 0
        for t in trig_ind:
            v = row.get(t, np.nan)
            th = trig_thresh.get(t, np.nan)
            if not (np.isfinite(v) and np.isfinite(th)):
                continue
            if trig_dir.get(t, "high") == "low":
                if v <= th: n_trig += 1
            else:
                if v >= th: n_trig += 1

        if n_trig < MIN_TRIGGERS:
            continue

        fwd_idx = t_idx + FWD_DAYS
        if fwd_idx >= n_bars:
            continue

        actual = (close_arr[fwd_idx] - close_arr[t_idx]) / close_arr[t_idx] * 100
        signals.append({
            "t_idx":  t_idx,
            "actual": actual,
            "win":    actual >= WIN_THRESH,
            "n_trig": n_trig,
        })

    return signals


# ── Sequence feature extraction ───────────────────────────────────────────────

def extract_sequence_features(
    t_idx: int,
    ind_map: dict,
    n_window: int,
    thresholds: dict[str, float],
) -> dict | None:
    """
    Với ngày T = t_idx, nhìn lại n_window ngày trước T (KHÔNG bao gồm T).
    Tính 3 loại feature cho mỗi indicator:
      - mean_N     : giá trị trung bình
      - trend_N    : slope tuyến tính (dương = đang tăng dần về phía T)
      - days_above : số ngày vượt ngưỡng trong window

    Trả về None nếu không đủ data.
    """
    # Lấy rows cho [T-n_window, ..., T-1]
    window_rows = []
    for k in range(n_window, 0, -1):  # T-n, T-n+1, ..., T-1
        row = ind_map.get(t_idx - k)
        if row is None:
            return None
        window_rows.append(row)

    if len(window_rows) < n_window:
        return None

    features = {}
    for ind in SEQ_INDICATORS:
        vals = [r.get(ind, np.nan) for r in window_rows]
        if any(not np.isfinite(v) for v in vals):
            continue

        vals_arr = np.array(vals)

        # a. Mean
        features[f"{ind}__mean_{n_window}"] = float(np.mean(vals_arr))

        # b. Trend (slope của linear regression qua n_window điểm)
        x = np.arange(n_window, dtype=float)
        if n_window >= 2:
            slope = float(np.polyfit(x, vals_arr, 1)[0])
        else:
            slope = 0.0
        features[f"{ind}__trend_{n_window}"] = slope

        # c. Days above/below threshold
        thresh = thresholds.get(ind)
        if thresh is not None:
            # Với stoch_k: đếm ngày DƯỚI ngưỡng (oversold)
            # Với các chỉ số khác: đếm ngày TRÊN ngưỡng
            if ind == "stoch_k":
                days = int(np.sum(vals_arr <= thresh))
            else:
                days = int(np.sum(vals_arr >= thresh))
            features[f"{ind}__days_above_{n_window}"] = days

    return features if features else None


# ── Cohen's d ────────────────────────────────────────────────────────────────

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: (mean_a - mean_b) / pooled_std"""
    if len(a) < 3 or len(b) < 3:
        return 0.0
    na, nb = len(a), len(b)
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled < 1e-9:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def interpret_d(d: float) -> str:
    ad = abs(d)
    if ad >= 0.8:  return "Manh"
    if ad >= 0.5:  return "Trung binh"
    if ad >= 0.2:  return "Yeu"
    return "Khong ro"


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_sequence(symbol: str):
    print(f"\n{'='*65}")
    print(f"SEQUENCE PATTERN ANALYSIS — {symbol}")
    print(f"FWD={FWD_DAYS}d | WIN_THRESH={WIN_THRESH}% | Training {TRAIN_START}→{TRAIN_END}")
    print(f"{'='*65}")

    # Load data
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2000, min_bars=200)
        df["date"] = pd.to_datetime(df["date"])
        train = df[
            (df["date"] >= TRAIN_START) &
            (df["date"] <= TRAIN_END)
        ].reset_index(drop=True)
    except Exception as e:
        print(f"ERROR load data: {e}")
        return None

    print(f"Training bars: {len(train)}")

    close_arr = train["close"].values.astype(float)
    n_bars    = len(train)

    # Tính indicators
    ind_rows = compute_indicators(train)
    ind_map  = {r["idx"]: r for r in ind_rows}

    # Lấy V3 signals
    signals = get_v3_signals(symbol, ind_rows, close_arr, n_bars)
    b_good  = [s for s in signals if s["win"]]
    b_bad   = [s for s in signals if not s["win"]]

    print(f"\nV3 Signals: total={len(signals)} | B_good={len(b_good)} | B_bad={len(b_bad)}")
    if len(signals) > 0:
        wr = len(b_good) / len(signals) * 100
        mean_ret = np.mean([s["actual"] for s in signals])
        print(f"WR={wr:.1f}% | Mean return={mean_ret:+.2f}%")

    if len(b_good) < 5 or len(b_bad) < 5:
        print(f"CANH BAO: Qua it signal ({len(b_good)} good, {len(b_bad)} bad) — ket qua khong tin cay")
        return None

    # Tính ngưỡng toàn bộ training cho days_above
    all_thresholds = {}
    for ind in SEQ_INDICATORS:
        vals = [r[ind] for r in ind_rows if np.isfinite(r.get(ind, np.nan))]
        if not vals:
            continue
        if ind == "stoch_k":
            all_thresholds[ind] = float(np.percentile(vals, 30))  # oversold threshold
        else:
            all_thresholds[ind] = float(np.percentile(vals, TRIGGER_PCT))

    # Phân tích từng window
    all_results = {}

    for n_window in SEQ_WINDOWS:
        print(f"\n{'─'*65}")
        print(f"WINDOW N={n_window} ngay truoc T")
        print(f"{'─'*65}")

        # Extract features cho B_good và B_bad
        feats_good, feats_bad = [], []

        for s in b_good:
            f = extract_sequence_features(s["t_idx"], ind_map, n_window, all_thresholds)
            if f is not None:
                feats_good.append(f)

        for s in b_bad:
            f = extract_sequence_features(s["t_idx"], ind_map, n_window, all_thresholds)
            if f is not None:
                feats_bad.append(f)

        if not feats_good or not feats_bad:
            print("  Khong du data cho window nay")
            continue

        # Convert to DataFrame để dễ xử lý
        df_good = pd.DataFrame(feats_good)
        df_bad  = pd.DataFrame(feats_bad)

        print(f"  B_good: {len(df_good)} ngay | B_bad: {len(df_bad)} ngay")

        # Tính Cohen's d cho từng feature
        feature_results = []
        for col in df_good.columns:
            if col not in df_bad.columns:
                continue
            g_vals = df_good[col].dropna().values
            b_vals = df_bad[col].dropna().values
            if len(g_vals) < 3 or len(b_vals) < 3:
                continue

            d = cohen_d(g_vals, b_vals)
            feature_results.append({
                "feature":    col,
                "cohen_d":    round(d, 3),
                "abs_d":      round(abs(d), 3),
                "strength":   interpret_d(d),
                "mean_good":  round(float(np.mean(g_vals)), 3),
                "mean_bad":   round(float(np.mean(b_vals)), 3),
                "direction":  "good > bad" if d > 0 else "good < bad",
            })

        # Sort theo |Cohen's d| giảm dần
        feature_results.sort(key=lambda x: -x["abs_d"])

        # In bảng
        print(f"\n  {'Feature':<30} {'Cohen_d':>8} {'Strength':<12} "
              f"{'Mean_good':>10} {'Mean_bad':>9}  Direction")
        print(f"  {'─'*80}")

        for r in feature_results:
            marker = "★" if r["abs_d"] >= 0.5 else ("·" if r["abs_d"] >= 0.2 else " ")
            print(
                f"  {marker} {r['feature']:<28} {r['cohen_d']:>+8.3f} "
                f"{r['strength']:<12} {r['mean_good']:>+10.3f} "
                f"{r['mean_bad']:>+9.3f}  {r['direction']}"
            )

        # Tóm tắt top signals
        strong   = [r for r in feature_results if r["abs_d"] >= 0.5]
        moderate = [r for r in feature_results if 0.2 <= r["abs_d"] < 0.5]

        print(f"\n  TOP FINDINGS (N={n_window}):")
        if strong:
            print(f"  ★ MANH ({len(strong)} features):")
            for r in strong[:5]:
                print(f"    {r['feature']}: d={r['cohen_d']:+.3f} "
                      f"(good={r['mean_good']:+.3f} vs bad={r['mean_bad']:+.3f})")
        if moderate:
            print(f"  · TRUNG BINH ({len(moderate)} features):")
            for r in moderate[:5]:
                print(f"    {r['feature']}: d={r['cohen_d']:+.3f} "
                      f"(good={r['mean_good']:+.3f} vs bad={r['mean_bad']:+.3f})")
        if not strong and not moderate:
            print(f"  Khong co feature nao phan biet duoc B_good vs B_bad")

        all_results[f"N{n_window}"] = feature_results

    # ── So sánh N=3 vs N=5: feature nào ổn định qua cả 2 window ──────────────
    if "N3" in all_results and "N5" in all_results:
        print(f"\n{'─'*65}")
        print(f"ON DINH QUA CA 2 WINDOW (N=3 va N=5)")
        print(f"{'─'*65}")

        r3 = {r["feature"]: r for r in all_results["N3"]}
        r5 = {r["feature"]: r for r in all_results["N5"]}

        # Chỉ lấy features xuất hiện ở cả 2
        common = set(r3.keys()) & set(r5.keys())

        stable = []
        for feat in common:
            d3 = r3[feat]["cohen_d"]
            d5 = r5[feat]["cohen_d"]
            # Ổn định = cùng chiều và cả 2 đều >= 0.2
            if abs(d3) >= 0.2 and abs(d5) >= 0.2 and (d3 * d5 > 0):
                avg = (abs(d3) + abs(d5)) / 2
                stable.append({
                    "feature": feat,
                    "d_N3":    d3,
                    "d_N5":    d5,
                    "avg_abs": round(avg, 3),
                    "strength": interpret_d((abs(d3) + abs(d5)) / 2),
                })

        stable.sort(key=lambda x: -x["avg_abs"])

        if stable:
            print(f"\n  {'Feature':<30} {'d(N=3)':>8} {'d(N=5)':>8} {'Avg':>7}  Strength")
            print(f"  {'─'*60}")
            for r in stable:
                marker = "★" if r["avg_abs"] >= 0.5 else "·"
                print(f"  {marker} {r['feature']:<28} {r['d_N3']:>+8.3f} "
                      f"{r['d_N5']:>+8.3f} {r['avg_abs']:>7.3f}  {r['strength']}")

            print(f"\n  → {len(stable)} features on dinh qua ca 2 window")
            print(f"  → Day la ung vien tot nhat de dinh nghia sequence condition")
        else:
            print(f"  Khong co feature nao on dinh qua ca 2 window")
            print(f"  → Sequence co the khong co nhieu gia tri bo sung cho V3")

    # ── Kết luận tổng ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"KET LUAN — {symbol}")
    print(f"{'='*65}")

    has_finding = any(
        any(r["abs_d"] >= 0.3 for r in res)
        for res in all_results.values()
    )

    if has_finding:
        print(f"  Co pattern phan biet B_good vs B_bad.")
        print(f"  Buoc tiep theo: dinh nghia sequence condition tu cac feature manh nhat")
        print(f"  roi test: B_good ∩ sequence vs B_bad ∩ sequence")
    else:
        print(f"  Khong tim thay pattern ro rang trong {SEQ_WINDOWS} ngay truoc T.")
        print(f"  Ket luan: Sequence khong bo sung them thong tin cho V3 voi {symbol}.")
        print(f"  Goi y: Xem xet cach tiep can khac (regime shift, exit layer).")

    # ── Lưu JSON ──────────────────────────────────────────────────────────────
    output = {
        "symbol":         symbol,
        "fwd_days":       FWD_DAYS,
        "win_thresh":     WIN_THRESH,
        "train_start":    TRAIN_START,
        "train_end":      TRAIN_END,
        "n_signals":      len(signals),
        "n_good":         len(b_good),
        "n_bad":          len(b_bad),
        "win_rate":       round(len(b_good) / len(signals) * 100, 1) if signals else 0,
        "results_by_window": {
            k: v for k, v in all_results.items()
        },
    }

    out_path = Path(f"seq_analysis_{symbol}.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Da luu: {out_path}")
    return output


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["MWG"]

    all_outputs = {}
    for sym in symbols:
        result = analyze_sequence(sym.upper())
        if result:
            all_outputs[sym.upper()] = result

    # Cross-symbol summary nếu chạy nhiều mã
    if len(all_outputs) > 1:
        print(f"\n{'#'*65}")
        print(f"CROSS-SYMBOL SUMMARY")
        print(f"{'#'*65}")
        for sym, out in all_outputs.items():
            print(f"\n  {sym}: n_signals={out['n_signals']} "
                  f"WR={out['win_rate']}% "
                  f"(good={out['n_good']}, bad={out['n_bad']})")

            for win_key, res in out["results_by_window"].items():
                strong = [r for r in res if r["abs_d"] >= 0.5]
                if strong:
                    print(f"    {win_key} — Strong features: "
                          f"{[r['feature'] for r in strong[:3]]}")
