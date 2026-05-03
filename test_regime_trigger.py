"""
test_regime_trigger.py — Kiểm tra suy luận 2 tầng

Suy luận cần kiểm tra:
  "Khi thị trường đang ở cùng một trạng thái nền (regime),
   nếu xuất hiện tín hiệu đột biến ngắn hạn (volume tăng, nến lớn...),
   thì xác suất giá tăng trong 10 ngày tới cao hơn bình thường"

Cách kiểm tra:
  Với mỗi ngày T trong training:
    1. Xác định regime của T (dựa vào chỉ số chậm)
    2. Xác định trigger của T (dựa vào chỉ số nhanh)
    3. Đo actual return sau FWD_DAYS

  So sánh 4 nhóm:
    A. Cùng regime + có trigger  → kỳ vọng tốt nhất
    B. Cùng regime + không trigger
    C. Khác regime + có trigger
    D. Khác regime + không trigger → baseline

  Nếu nhóm A tốt hơn D rõ rệt → suy luận được ủng hộ

Chạy: python test_regime_trigger.py MWG
"""
import sys
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
FWD_DAYS    = 10
TRAIN_START = "2019-01-01"
TRAIN_END   = "2024-12-31"

# ── Định nghĩa Regime (chỉ số chậm, ổn định) ─────────────────────────────────
# Mỗi mã có regime riêng dựa trên phân tích trước

REGIME_CONFIG = {
    # MWG: tăng khi ATR thấp + momentum yếu → regime "tích lũy yên tĩnh"
    "MWG": {
        "regime_indicator": "atr_ratio",
        "regime_condition": "low",      # low = dưới median → tích lũy
        "regime_label":     "Tich luy (ATR thap)",
    },
    # STB: tăng khi ATR cao → regime "biến động cao / breakout"
    "STB": {
        "regime_indicator": "atr_ratio",
        "regime_condition": "high",     # high = trên median → biến động
        "regime_label":     "Bien dong cao (ATR cao)",
    },
    # DPM: tăng khi trend yếu/âm → regime "downtrend / sideways"
    "DPM": {
        "regime_indicator": "trend_slope",
        "regime_condition": "low",      # low = SMA20 < SMA50 → downtrend
        "regime_label":     "Downtrend / Sideways (trend yeu)",
    },
}

# Trigger: chỉ số nhanh báo hiệu đột biến
TRIGGER_INDICATORS = ["volume_spike", "candle_body", "momentum_5d", "stoch_k"]

# Ngưỡng trigger: top 30% của phân phối = "đột biến"
TRIGGER_PERCENTILE = 70


# ── Tính chỉ số ───────────────────────────────────────────────────────────────

def _ema(c, span):
    return pd.Series(c).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c).rolling(p, min_periods=p).mean().values

def compute_indicators(df):
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    vol   = df["volume"].values.astype(float)
    opn   = df["open"].values.astype(float)
    n     = len(df)

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    vsma5 = _sma(vol, 5)
    vsma20= _sma(vol, 20)

    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high-low, np.maximum(
             np.abs(high-h_prev), np.abs(low-h_prev)))
    atr    = _sma(tr, 14)

    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta>0, delta, 0.0)
    loss  = np.where(delta<0, -delta, 0.0)
    avg_g = _ema(gain, 14)
    avg_l = _ema(loss, 14)
    avg_l = np.where(avg_l==0, 1e-9, avg_l)
    rsi   = 100 - 100/(1 + avg_g/avg_l)

    lo14  = pd.Series(low).rolling(14).min().values
    hi14  = pd.Series(high).rolling(14).max().values
    denom = np.where(hi14-lo14==0, 1e-9, hi14-lo14)
    stoch = 100*(close-lo14)/denom

    rows = []
    for i in range(60, n):
        px    = close[i]
        atr_v = atr[i] if np.isfinite(atr[i]) else px*0.02
        s20   = sma20[i] if np.isfinite(sma20[i]) else px
        s50   = sma50[i] if np.isfinite(sma50[i]) else px
        vs20v = vsma20[i] if np.isfinite(vsma20[i]) else vol[i]
        vs5v  = vsma5[i] if np.isfinite(vsma5[i]) else vol[i]
        c5    = close[max(i-5,0)]

        body = abs(px - opn[i]) / (atr_v + 1e-9)

        rows.append({
            "idx":          i,
            "date":         df["date"].iloc[i],
            "close":        px,
            # Regime indicators (chậm)
            "atr_ratio":    float(atr_v/(px+1e-9)*100),
            "trend_slope":  float((s20-s50)/(px+1e-9)*100),
            "momentum_20d": float((px/close[max(i-20,0)]-1)*100),
            # Trigger indicators (nhanh)
            "volume_spike": float((vol[i]/(vs20v+1e-9))-1),
            "candle_body":  float(np.clip(body, 0, 3)),
            "momentum_5d":  float((px/(c5+1e-9)-1)*100),
            "stoch_k":      float(stoch[i]),
        })

    return pd.DataFrame(rows)


# ── Phân tích 2 tầng ──────────────────────────────────────────────────────────

def analyze_two_layer(symbol: str):
    print(f"\n{'='*60}")
    print(f"KIEM TRA SUY LUAN 2 TANG — {symbol}")
    print(f"FWD={FWD_DAYS} ngay | Training: {TRAIN_START} → {TRAIN_END}")
    print('='*60)

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
        print(f"ERROR: {e}")
        return

    print(f"Training bars: {len(train)}")

    # Tính chỉ số
    ind = compute_indicators(train)
    close_arr = train["close"].values.astype(float)

    # Tính forward return
    fwd_rets = []
    for i in ind["idx"].values:
        fi = i + FWD_DAYS
        if fi >= len(train):
            fwd_rets.append(np.nan)
        else:
            fwd_rets.append((close_arr[fi] - close_arr[i]) / close_arr[i] * 100)
    ind["fwd_return"] = fwd_rets
    ind = ind.dropna(subset=["fwd_return"])

    # ── Tầng 1: Xác định regime ───────────────────────────────────────────────
    cfg       = REGIME_CONFIG.get(symbol, REGIME_CONFIG["MWG"])
    reg_col   = cfg["regime_indicator"]
    reg_cond  = cfg["regime_condition"]
    reg_label = cfg["regime_label"]

    median_val = ind[reg_col].median()
    if reg_cond == "low":
        in_regime = ind[reg_col] <= median_val
    else:
        in_regime = ind[reg_col] > median_val

    n_regime = in_regime.sum()
    print(f"\nTang 1 — Regime: {reg_label}")
    print(f"  Nguong {reg_col} median = {median_val:.3f}")
    print(f"  So ngay trong regime: {n_regime}/{len(ind)} ({n_regime/len(ind)*100:.0f}%)")

    # ── Tầng 2: Xác định trigger ──────────────────────────────────────────────
    print(f"\nTang 2 — Trigger (top {100-TRIGGER_PERCENTILE}% dot bien):")

    best_combo = None
    best_lift  = -999

    results = []

    # Test từng trigger indicator
    for trig_col in TRIGGER_INDICATORS:
        thresh = np.percentile(ind[trig_col].values, TRIGGER_PERCENTILE)
        has_trigger = ind[trig_col] >= thresh

        # 4 nhóm
        grp_A = ind[in_regime  & has_trigger]["fwd_return"]   # regime + trigger
        grp_B = ind[in_regime  & ~has_trigger]["fwd_return"]  # regime, no trigger
        grp_C = ind[~in_regime & has_trigger]["fwd_return"]   # no regime, trigger
        grp_D = ind[~in_regime & ~has_trigger]["fwd_return"]  # baseline

        if len(grp_A) < 5 or len(grp_D) < 5:
            continue

        mean_A = grp_A.mean()
        mean_B = grp_B.mean()
        mean_C = grp_C.mean()
        mean_D = grp_D.mean()
        lift   = mean_A - mean_D  # A tốt hơn baseline bao nhiêu

        # Win rate (return > 0)
        wr_A = (grp_A > 0).mean() * 100
        wr_D = (grp_D > 0).mean() * 100

        results.append({
            "trigger":  trig_col,
            "thresh":   thresh,
            "n_A":      len(grp_A),
            "mean_A":   mean_A,
            "wr_A":     wr_A,
            "mean_B":   mean_B,
            "mean_C":   mean_C,
            "mean_D":   mean_D,
            "wr_D":     wr_D,
            "lift":     lift,
        })

        if lift > best_lift:
            best_lift  = lift
            best_combo = trig_col

    # In kết quả từng trigger
    print(f"\n  {'Trigger':<14} {'n_A':>5} {'Mean_A':>8} {'WR_A':>7} "
          f"{'Mean_D':>8} {'WR_D':>7} {'Lift':>8}  Ket luan")
    print(f"  {'-'*75}")

    for r in sorted(results, key=lambda x: -x["lift"]):
        ket_luan = (
            "✅ Ung ho" if r["lift"] > 0.5 and r["mean_A"] > 0
            else "⚠️  Yeu"   if r["lift"] > 0
            else "❌ Phan bac"
        )
        print(
            f"  {r['trigger']:<14} {r['n_A']:>5} {r['mean_A']:>+7.2f}% {r['wr_A']:>6.1f}% "
            f"{r['mean_D']:>+7.2f}% {r['wr_D']:>6.1f}% {r['lift']:>+7.2f}%  {ket_luan}"
        )

    # ── Test kết hợp nhiều trigger ────────────────────────────────────────────
    print(f"\n  Test ket hop nhieu trigger cung luc:")

    for n_trig in [2, 3]:
        # Yêu cầu ít nhất n_trig triggers cùng lúc
        trigger_cols = TRIGGER_INDICATORS
        trigger_flags = pd.DataFrame({
            col: ind[col] >= np.percentile(ind[col].values, TRIGGER_PERCENTILE)
            for col in trigger_cols
        })
        n_triggers_active = trigger_flags.sum(axis=1)
        has_multi = n_triggers_active >= n_trig

        grp_A2 = ind[in_regime  & has_multi]["fwd_return"]
        grp_D2 = ind[~in_regime & ~has_multi]["fwd_return"]

        if len(grp_A2) < 3:
            print(f"  {n_trig} triggers: qua it ngay ({len(grp_A2)})")
            continue

        mean_A2 = grp_A2.mean()
        wr_A2   = (grp_A2 > 0).mean() * 100
        mean_D2 = grp_D2.mean() if len(grp_D2) > 0 else 0
        lift2   = mean_A2 - mean_D2

        ket_luan2 = (
            "✅ Ung ho manh" if lift2 > 1.0 and mean_A2 > 1.0
            else "⚠️  Yeu"    if lift2 > 0
            else "❌ Phan bac"
        )
        print(
            f"  >={n_trig} triggers: n={len(grp_A2):3d} "
            f"Mean={mean_A2:+.2f}% WR={wr_A2:.1f}% "
            f"Lift={lift2:+.2f}%  {ket_luan2}"
        )

    # ── Kết luận tổng ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    supported = [r for r in results if r["lift"] > 0.5 and r["mean_A"] > 0]
    if supported:
        best = max(supported, key=lambda x: x["lift"])
        print(f"KET LUAN {symbol}: Suy luan DUOC ung ho")
        print(f"  Trigger tot nhat: {best['trigger']}")
        print(f"  Regime + trigger: Mean={best['mean_A']:+.2f}% WR={best['wr_A']:.1f}%")
        print(f"  Baseline:         Mean={best['mean_D']:+.2f}% WR={best['wr_D']:.1f}%")
        print(f"  Lift:             {best['lift']:+.2f}%")
    else:
        print(f"KET LUAN {symbol}: Suy luan CHUA DUOC ung ho du")
        print(f"  Can xem xet lai dinh nghia regime hoac trigger")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["MWG"]
    for sym in symbols:
        analyze_two_layer(sym.upper())
