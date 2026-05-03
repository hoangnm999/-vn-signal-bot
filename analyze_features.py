"""
analyze_features.py — Phân tích chỉ số nào thực sự dự báo được giá tăng

Chạy: python analyze_features.py MWG
Output: analyze_MWG.json — dùng để chọn chỉ số cho analog vector

Logic:
  - Với mỗi ngày T trong training (2019-2024):
      tính 15 chỉ số tại ngày T
      tính actual return sau FWD_DAYS=15 bars
      gán nhãn GOOD (>2%) / BAD (<-2%) / NEUTRAL
  - Với từng chỉ số:
      so sánh phân phối giữa GOOD vs BAD
      tính separation score (khả năng phân biệt)
  - Rank 15 chỉ số theo separation score
  - Xuất kết quả JSON để dùng lại
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FWD_DAYS     = 10     # hold 10 ngày
GOOD_THRESH  = 5.0    # % tăng để gán nhãn GOOD
BAD_THRESH   = -3.0   # % giảm để gán nhãn BAD
TRAIN_END    = "2024-12-31"
TRAIN_START  = "2019-01-01"

# 15 chỉ số cần phân tích
ALL_INDICATORS = [
    "rsi_14", "macd_hist", "bb_position", "volume_spike",
    "trend_slope", "price_vs_sma20", "price_vs_sma50", "atr_ratio",
    "stoch_k", "ema_cross", "momentum_5d", "momentum_20d",
    "high_low_pos", "vol_trend", "candle_body",
]


# ── Tính 15 chỉ số tại mỗi ngày ──────────────────────────────────────────────

def _ema(c, span):
    return pd.Series(c).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c).rolling(p, min_periods=p).mean().values

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tính 15 chỉ số cho toàn bộ df.
    Trả về DataFrame cùng index với df, thêm 15 cột chỉ số.
    """
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)
    vol    = df["volume"].values.astype(float)
    opn    = df["open"].values.astype(float)
    n      = len(df)

    # RSI(14)
    delta  = np.diff(close, prepend=close[0])
    gain   = np.where(delta > 0, delta, 0.0)
    loss   = np.where(delta < 0, -delta, 0.0)
    avg_g  = _ema(gain, 14)
    avg_l  = _ema(loss, 14)
    avg_l  = np.where(avg_l == 0, 1e-9, avg_l)
    rsi    = 100 - 100 / (1 + avg_g / avg_l)

    # MACD
    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    macd   = ema12 - ema26
    sig    = _ema(macd, 9)
    hist   = macd - sig

    # SMA
    sma20  = _sma(close, 20)
    sma50  = _sma(close, 50)
    vsma5  = _sma(vol, 5)
    vsma20 = _sma(vol, 20)

    # ATR(14)
    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr    = _sma(tr, 14)

    # Bollinger
    std20    = pd.Series(close).rolling(20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20

    # Stochastic K(14)
    lo14   = pd.Series(low).rolling(14).min().values
    hi14   = pd.Series(high).rolling(14).max().values
    denom  = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch  = 100 * (close - lo14) / denom

    # Tính từng chỉ số (raw, chưa normalize) tại từng ngày
    rows = []
    for i in range(60, n):
        px       = close[i]
        atr_v    = atr[i] if np.isfinite(atr[i]) else px * 0.02
        s20      = sma20[i] if np.isfinite(sma20[i]) else px
        s50      = sma50[i] if np.isfinite(sma50[i]) else px
        bb_u     = bb_upper[i] if np.isfinite(bb_upper[i]) else px * 1.04
        bb_l     = bb_lower[i] if np.isfinite(bb_lower[i]) else px * 0.96
        vs5_v    = vsma5[i] if np.isfinite(vsma5[i]) else vol[i]
        vs20_v   = vsma20[i] if np.isfinite(vsma20[i]) else vol[i]

        c5  = close[max(i-5, 0)]
        c20 = close[max(i-20, 0)]
        hi20 = float(np.nanmax(high[max(i-19,0):i+1]))
        lo20 = float(np.nanmin(low[max(i-19,0):i+1]))

        # BB position [0,1]
        bb_range  = bb_u - bb_l
        bb_pos    = (px - bb_l) / bb_range if bb_range > 0 else 0.5
        bb_pos    = float(np.clip(bb_pos, 0, 1))

        # High-low position trong 20 ngày [0,1]
        hlp = (px - lo20) / (hi20 - lo20) if hi20 > lo20 else 0.5

        # Candle body normalized by ATR
        body = abs(px - opn[i]) / (atr_v + 1e-9)

        rows.append({
            "idx":           i,
            "date":          df["date"].iloc[i],
            "close":         px,
            # 15 chỉ số — raw values (không normalize) để dễ interpret
            "rsi_14":        float(rsi[i]),
            "macd_hist":     float(hist[i]),
            "bb_position":   bb_pos,
            "volume_spike":  float((vol[i] / (vs20_v + 1e-9)) - 1.0),
            "trend_slope":   float((s20 - s50) / (px + 1e-9) * 100),   # %
            "price_vs_sma20":float((px - s20) / (px + 1e-9) * 100),    # %
            "price_vs_sma50":float((px - s50) / (px + 1e-9) * 100),    # %
            "atr_ratio":     float(atr_v / (px + 1e-9) * 100),         # %
            "stoch_k":       float(stoch[i]),
            "ema_cross":     float((ema12[i] - ema26[i]) / (px + 1e-9) * 100),  # %
            "momentum_5d":   float((px / (c5 + 1e-9) - 1.0) * 100),    # %
            "momentum_20d":  float((px / (c20 + 1e-9) - 1.0) * 100),   # %
            "high_low_pos":  float(hlp),
            "vol_trend":     float((vs5_v / (vs20_v + 1e-9)) - 1.0),
            "candle_body":   float(np.clip(body, 0, 3)),
        })

    return pd.DataFrame(rows)


# ── Gán nhãn GOOD / BAD / NEUTRAL ────────────────────────────────────────────

def label_days(ind_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột fwd_return và label vào ind_df."""
    close_arr = full_df["close"].values.astype(float)
    n_full    = len(full_df)

    fwd_rets = []
    for row in ind_df.itertuples():
        i       = row.idx
        fwd_idx = i + FWD_DAYS
        if fwd_idx >= n_full:
            fwd_rets.append(np.nan)
        else:
            ret = (close_arr[fwd_idx] - close_arr[i]) / close_arr[i] * 100
            fwd_rets.append(ret)

    ind_df = ind_df.copy()
    ind_df["fwd_return"] = fwd_rets
    ind_df = ind_df.dropna(subset=["fwd_return"])

    def label(r):
        if r >= GOOD_THRESH: return "GOOD"
        if r <= BAD_THRESH:  return "BAD"
        return "NEUTRAL"

    ind_df["label"] = ind_df["fwd_return"].apply(label)
    return ind_df


# ── Tính separation score cho từng chỉ số ────────────────────────────────────

def separation_score(good_vals, bad_vals):
    """
    Đo mức độ tách biệt giữa phân phối GOOD và BAD.
    Dùng Cohen's d: (mean_good - mean_bad) / pooled_std
    Càng cao → chỉ số càng phân biệt được GOOD vs BAD.
    """
    if len(good_vals) < 5 or len(bad_vals) < 5:
        return 0.0
    mg = np.mean(good_vals)
    mb = np.mean(bad_vals)
    sg = np.std(good_vals, ddof=1)
    sb = np.std(bad_vals, ddof=1)
    ng = len(good_vals)
    nb = len(bad_vals)
    pooled = np.sqrt(((ng-1)*sg**2 + (nb-1)*sb**2) / (ng+nb-2))
    if pooled < 1e-9:
        return 0.0
    return float((mg - mb) / pooled)


def analyze_indicators(labeled_df: pd.DataFrame) -> list:
    """
    Phân tích từng chỉ số, tính separation score và thống kê.
    Trả về list kết quả đã sort theo |score| giảm dần.
    """
    good = labeled_df[labeled_df["label"] == "GOOD"]
    bad  = labeled_df[labeled_df["label"] == "BAD"]

    results = []
    for ind in ALL_INDICATORS:
        if ind not in labeled_df.columns:
            continue

        gv = good[ind].dropna().values
        bv = bad[ind].dropna().values

        score = separation_score(gv, bv)

        results.append({
            "indicator":    ind,
            "cohen_d":      round(score, 3),
            "abs_cohen_d":  round(abs(score), 3),
            # Thống kê GOOD
            "good_mean":    round(float(np.mean(gv)), 3) if len(gv) else None,
            "good_median":  round(float(np.median(gv)), 3) if len(gv) else None,
            "good_std":     round(float(np.std(gv)), 3) if len(gv) else None,
            "good_p25":     round(float(np.percentile(gv, 25)), 3) if len(gv) else None,
            "good_p75":     round(float(np.percentile(gv, 75)), 3) if len(gv) else None,
            "good_n":       len(gv),
            # Thống kê BAD
            "bad_mean":     round(float(np.mean(bv)), 3) if len(bv) else None,
            "bad_median":   round(float(np.median(bv)), 3) if len(bv) else None,
            "bad_std":      round(float(np.std(bv)), 3) if len(bv) else None,
            "bad_p25":      round(float(np.percentile(bv, 25)), 3) if len(bv) else None,
            "bad_p75":      round(float(np.percentile(bv, 75)), 3) if len(bv) else None,
            "bad_n":        len(bv),
            # Interpretation
            "direction":    "GOOD_higher" if score > 0 else "BAD_higher",
            "strength":     (
                "strong"   if abs(score) >= 0.5 else
                "moderate" if abs(score) >= 0.3 else
                "weak"
            ),
        })

    results.sort(key=lambda x: -x["abs_cohen_d"])
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis(symbol: str) -> dict:
    print(f"\n{'='*50}")
    print(f"FEATURE ANALYSIS — {symbol}")
    print(f"FWD_DAYS={FWD_DAYS}, GOOD>{GOOD_THRESH}%, BAD<{BAD_THRESH}%")
    print(f"Training: {TRAIN_START} → {TRAIN_END}")
    print('='*50)

    # Load data
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2000, min_bars=200)
        print(f"Data: {len(df)} bars ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return {}

    # Lọc training period
    df["date"] = pd.to_datetime(df["date"])
    train_df   = df[
        (df["date"] >= TRAIN_START) &
        (df["date"] <= TRAIN_END)
    ].reset_index(drop=True)
    print(f"Training bars: {len(train_df)}")

    # Tính chỉ số
    print("Tinh chi so...")
    ind_df = compute_all_indicators(train_df)

    # Gán nhãn (dùng toàn bộ df để tính fwd return chính xác)
    # Map lại index từ train_df sang df gốc
    full_close = train_df["close"].values.astype(float)
    fwd_rets   = []
    for i in ind_df["idx"].values:
        fwd_idx = i + FWD_DAYS
        if fwd_idx >= len(train_df):
            fwd_rets.append(np.nan)
        else:
            ret = (full_close[fwd_idx] - full_close[i]) / full_close[i] * 100
            fwd_rets.append(ret)

    ind_df["fwd_return"] = fwd_rets
    ind_df = ind_df.dropna(subset=["fwd_return"])
    ind_df["label"] = ind_df["fwd_return"].apply(
        lambda r: "GOOD" if r >= GOOD_THRESH else ("BAD" if r <= BAD_THRESH else "NEUTRAL")
    )

    n_good    = (ind_df["label"] == "GOOD").sum()
    n_bad     = (ind_df["label"] == "BAD").sum()
    n_neutral = (ind_df["label"] == "NEUTRAL").sum()
    print(f"Labels: GOOD={n_good} ({n_good/len(ind_df)*100:.1f}%) | "
          f"BAD={n_bad} ({n_bad/len(ind_df)*100:.1f}%) | "
          f"NEUTRAL={n_neutral}")

    # Phân tích
    print("Phan tich separation score...")
    analysis = analyze_indicators(ind_df)

    # In kết quả
    print(f"\n{'─'*65}")
    print(f"{'Rank':<4} {'Indicator':<16} {'Cohen_d':>8} {'Strength':<10} "
          f"{'GOOD_med':>9} {'BAD_med':>9} {'Direction'}")
    print(f"{'─'*65}")
    for i, r in enumerate(analysis, 1):
        print(
            f"{i:<4} {r['indicator']:<16} {r['cohen_d']:>+8.3f} "
            f"{r['strength']:<10} "
            f"{r['good_median']:>+9.2f} {r['bad_median']:>+9.2f}  "
            f"{r['direction']}"
        )

    # Đề xuất top indicators
    strong   = [r for r in analysis if r["strength"] == "strong"]
    moderate = [r for r in analysis if r["strength"] == "moderate"]
    top      = (strong + moderate)[:7]

    print(f"\n{'='*50}")
    print(f"DE XUAT vector cho {symbol}:")
    print(f"  Strong  ({len(strong)}): {[r['indicator'] for r in strong]}")
    print(f"  Moderate({len(moderate)}): {[r['indicator'] for r in moderate]}")
    print(f"  Top 7   : {[r['indicator'] for r in top]}")
    print('='*50)

    # Lưu JSON
    output = {
        "symbol":      symbol,
        "fwd_days":    FWD_DAYS,
        "good_thresh": GOOD_THRESH,
        "bad_thresh":  BAD_THRESH,
        "train_start": TRAIN_START,
        "train_end":   TRAIN_END,
        "n_bars":      len(train_df),
        "n_good":      int(n_good),
        "n_bad":       int(n_bad),
        "n_neutral":   int(n_neutral),
        "analysis":    analysis,
        "recommended_indicators": [r["indicator"] for r in top],
    }

    out_path = Path(f"analyze_{symbol}.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nDa luu: {out_path}")
    return output


if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["MWG"]

    # Bước 0: phân tích tốc độ thay đổi trước
    for sym in symbols:
        analyze_variability(sym.upper())


    # Chạy tất cả window để so sánh
    windows = [2, 5, 10, 15]

    for sym in symbols:
        sym = sym.upper()
        print(f"\n{'#'*60}")
        print(f"# SO SANH FWD_DAYS — {sym}")
        print(f"{'#'*60}")

        # Summary table
        summary = {}  # fwd_days -> list of (indicator, cohen_d)

        for fwd in windows:
            FWD_DAYS = fwd

            # Load data 1 lần
            try:
                from vn_loader import load_vn_ohlcv
                import pandas as pd
                df = load_vn_ohlcv(sym, days=2000, min_bars=200)
                df["date"] = pd.to_datetime(df["date"])
                train_df = df[
                    (df["date"] >= TRAIN_START) &
                    (df["date"] <= TRAIN_END)
                ].reset_index(drop=True)
            except Exception as e:
                print(f"ERROR: {e}")
                break

            ind_df = compute_all_indicators(train_df)
            full_close = train_df["close"].values.astype(float)
            fwd_rets = []
            for i in ind_df["idx"].values:
                fwd_idx = i + fwd
                if fwd_idx >= len(train_df):
                    fwd_rets.append(float("nan"))
                else:
                    ret = (full_close[fwd_idx] - full_close[i]) / full_close[i] * 100
                    fwd_rets.append(ret)

            ind_df["fwd_return"] = fwd_rets
            ind_df = ind_df.dropna(subset=["fwd_return"])
            ind_df["label"] = ind_df["fwd_return"].apply(
                lambda r: "GOOD" if r >= GOOD_THRESH else ("BAD" if r <= BAD_THRESH else "NEUTRAL")
            )

            n_good = (ind_df["label"] == "GOOD").sum()
            n_bad  = (ind_df["label"] == "BAD").sum()
            analysis = analyze_indicators(ind_df)
            summary[fwd] = analysis

            # In top 5 cho window này
            print(f"\nFWD={fwd:2d}d | GOOD={n_good}({n_good/len(ind_df)*100:.0f}%) BAD={n_bad}({n_bad/len(ind_df)*100:.0f}%)")
            print(f"  {'Indicator':<16} {'Cohen_d':>8}  {'Strength'}")
            print(f"  {'-'*38}")
            for r in analysis[:5]:
                print(f"  {r['indicator']:<16} {r['cohen_d']:>+8.3f}  {r['strength']}")

        # So sánh chỉ số nào ổn định nhất qua các window
        print(f"\n{'='*55}")
        print(f"ON DINH QUA CAC WINDOW — {sym}")
        print(f"{'Indicator':<16} " + " ".join(f"FWD={w:2d}" for w in windows) + "  AVG_ABS")
        print("-" * 55)
        for ind in ALL_INDICATORS:
            scores = []
            for fwd in windows:
                s = next((r["cohen_d"] for r in summary[fwd] if r["indicator"] == ind), 0)
                scores.append(s)
            avg_abs = sum(abs(s) for s in scores) / len(scores)
            row = f"{ind:<16} " + " ".join(f"{s:>+7.3f}" for s in scores) + f"  {avg_abs:.3f}"
            print(row)
        print("="*55)


# ══════════════════════════════════════════════════════════════════════════════
# PHÂN TÍCH TỐC ĐỘ THAY ĐỔI — chỉ số nào thay đổi đủ nhanh để phân biệt ngày
# ══════════════════════════════════════════════════════════════════════════════

def analyze_variability(symbol: str):
    """
    Với mỗi chỉ số, đo:
      1. std_daily   : độ lệch chuẩn ngày qua ngày → thay đổi nhanh hay chậm
      2. autocorr_1  : tự tương quan lag-1 → nếu gần 1.0 = thay đổi rất chậm
      3. pct_days_diff: % ngày mà chỉ số thay đổi > 5% so với ngày trước

    Chỉ số tốt để phân biệt ngày: autocorr thấp + std_daily cao
    """
    import pandas as pd

    print(f"\n{'='*60}")
    print(f"TOC DO THAY DOI CHI SO — {symbol}")
    print(f"(autocorr gần 1.0 = thay đổi chậm = khó phân biệt ngày)")
    print(f"{'='*60}")

    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv(symbol, days=2000, min_bars=200)
        df["date"] = pd.to_datetime(df["date"])
        train_df = df[
            (df["date"] >= TRAIN_START) &
            (df["date"] <= TRAIN_END)
        ].reset_index(drop=True)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    ind_df = compute_all_indicators(train_df)

    print(f"\n{'Indicator':<16} {'AutoCorr':>9} {'Std_daily':>10} {'%Ngay_diff5%':>13}  Nhan_xet")
    print("-" * 70)

    results = []
    for ind in ALL_INDICATORS:
        if ind not in ind_df.columns:
            continue
        vals = ind_df[ind].dropna().values

        # Tự tương quan lag-1
        if len(vals) > 10:
            autocorr = float(pd.Series(vals).autocorr(lag=1))
        else:
            autocorr = 0.0

        # Std của thay đổi ngày qua ngày
        daily_changes = np.diff(vals)
        std_daily = float(np.std(daily_changes))

        # % ngày thay đổi > 5% range của chỉ số
        val_range = float(np.nanmax(vals) - np.nanmin(vals))
        if val_range > 0:
            pct_diff = float(np.mean(np.abs(daily_changes) > 0.05 * val_range) * 100)
        else:
            pct_diff = 0.0

        # Nhận xét
        if autocorr > 0.95:
            nhan_xet = "⛔ Qua cham"
        elif autocorr > 0.85:
            nhan_xet = "⚠️  Cham"
        elif autocorr > 0.70:
            nhan_xet = "🟡 Trung binh"
        else:
            nhan_xet = "✅ Nhanh"

        results.append({
            "indicator": ind,
            "autocorr":  autocorr,
            "std_daily": std_daily,
            "pct_diff":  pct_diff,
            "nhan_xet":  nhan_xet,
        })

    # Sort theo autocorr tăng dần (nhanh nhất lên đầu)
    results.sort(key=lambda x: x["autocorr"])

    for r in results:
        print(
            f"{r['indicator']:<16} {r['autocorr']:>+9.3f} {r['std_daily']:>10.4f} "
            f"{r['pct_diff']:>12.1f}%  {r['nhan_xet']}"
        )

    print(f"\n→ Chỉ dùng chỉ số có autocorr < 0.85 để phân biệt ngày hiệu quả")
    return results
