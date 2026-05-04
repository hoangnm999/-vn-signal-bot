"""
validate_new_clusters.py — Walk Forward validation cho 2 cluster mới
VN Trader Bot V6 — Session 31

2 clusters từ discover_clusters_v2.py:
  1. Breakout (BB Squeeze + Consolidation) — FWD=15d
  2. Deep Value Recovery — FWD=10d

Dùng WF template từ backtest_trailing_stop_v2.py (đã validated):
  - Expanding window: train 18 tháng, test 6 tháng
  - Threshold tính từ data trước train_end (không lookahead)
  - WFE = OOS_exp / IS_exp

Chạy: python validate_new_clusters.py 2>&1 | tee validate_clusters_results.txt

Filter để add vào watchlist:
  WFE >= 0.3, Consistency >= 60%, OOS_exp > 0, n_trades >= 10
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Cluster definitions từ discover_clusters_v2 ───────────────────────────────

BREAKOUT_SYMBOLS = [
    "SHB", "HPG", "VIX", "FPT", "TCB", "VPB", "BSR", "DXG", "VJC", "PC1",
    "DGC", "DCM", "VND", "PDR", "MSB", "PLX", "GMD", "VIB", "KBC", "KDH",
    "GVR", "LPB", "NKG", "CTD", "HSG", "HDG", "VHC", "DCL", "MCH", "FUEVFVND",
    "NAF", "TCM", "KDC", "KSB", "LCG", "HVN", "E1VFVN30", "SIP", "BFC",
    "CDC", "HT1", "AGR", "SHI",
]

DEEP_VALUE_SYMBOLS = [
    "BID", "EIB", "VPI", "DPM", "TIX", "PNJ", "BAF", "REE", "BVH", "SAB",
    "PET", "BCM", "NT2", "DPR", "KOS", "PHR", "TTA", "CTF", "PPC", "IDI",
]

CLUSTER_CONFIG = {
    "Breakout": {
        "symbols":   BREAKOUT_SYMBOLS,
        "fwd_days":  15,
        "regime":    {"bb_squeeze": "high"},
        "triggers":  {"consolidation": "low", "vol_dry_up": "high"},
        "min_triggers": 1,
    },
    "Deep Value": {
        "symbols":   DEEP_VALUE_SYMBOLS,
        "fwd_days":  10,
        "regime":    {"vol_dry_up": "low"},
        "triggers":  {"dist_52w_high": "low"},
        "min_triggers": 1,
    },
}

# WF Config (giống backtest_trailing_stop_v2.py)
WF_START        = date(2022, 1, 1)
WF_TRAIN_MONTHS = 18
WF_TEST_MONTHS  = 6
WF_MIN_FOLDS    = 3

# Filter để accept
MIN_WFE         = 0.3
MIN_CONSISTENCY = 60.0
MIN_OOS_EXP     = 0.0
MIN_TRADES_FOLD = 3

# Backtest training window
TRAIN_START = date(2019, 1, 1)
TRAIN_END   = date(2023, 12, 31)
MIN_TRADES  = 10
MIN_EXP     = 0.3
MIN_WR      = 45.0
MIN_PF      = 1.1


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sma(c, p):
    return pd.Series(c.astype(float)).rolling(p, min_periods=p).mean().values

def _ema(c, span):
    return pd.Series(c.astype(float)).ewm(span=span, adjust=False).mean().values

def compute_features_full(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Tính tất cả features cần thiết cho 2 cluster mới."""
    if len(df) < 120:
        return None

    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    vol   = df["volume"].values.astype(float)
    opn   = df["open"].values.astype(float)

    sma20  = _sma(close, 20)
    vsma20 = _sma(vol,   20)
    vsma60 = _sma(vol,   60)

    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr14  = _sma(tr, 14)

    # BB squeeze (bb_width)
    bb_std   = pd.Series(close).rolling(20).std().values
    bb_width = (4 * bb_std) / (sma20 + 1e-9) * 100

    # OBV trend
    price_chg = np.concatenate([[0], np.diff(close)])
    obv       = np.cumsum(np.sign(price_chg) * vol)
    obv_sma10 = _sma(obv, 10)

    # 52w high distance
    hi252 = pd.Series(high).rolling(252, min_periods=60).max().values

    # Consolidation: % ngày trong 15 phiên giá nằm trong ±3%
    def _consol(x):
        if len(x) < 5: return 0.0
        mid = x[-1]
        return float(np.sum(np.abs(x - mid) / (mid + 1e-9) < 0.03)) / len(x)
    consolidation = pd.Series(close).rolling(15).apply(_consol, raw=True).values

    # Volume dry up: vsma20 vs vsma60
    vol_dry_up = (vsma20 / (vsma60 + 1e-9)) - 1.0

    result = df.copy()
    result["bb_squeeze"]    = bb_width
    result["consolidation"] = consolidation
    result["vol_dry_up"]    = vol_dry_up
    result["dist_52w_high"] = (close / (hi252 + 1e-9) - 1.0) * 100
    result["obv_trend"]     = (obv - obv_sma10) / (np.abs(obv_sma10) + 1e-9) * 100
    result["close_val"]     = close

    return result


# ══════════════════════════════════════════════════════════════════════════════
# COLLECT TRADES — giống backtest_trailing_stop_v2.py template
# ══════════════════════════════════════════════════════════════════════════════

def collect_trades(df_feat: pd.DataFrame,
                   cluster_name: str,
                   window_start: date,
                   window_end: date,
                   train_end: date) -> list[float]:
    """
    Collect PnL list trong [window_start, window_end].
    Threshold tính từ data TRƯỚC train_end (không lookahead).
    Key fix vs discover_clusters_v2: train_end được pass đúng vào đây.
    """
    cfg         = CLUSTER_CONFIG[cluster_name]
    fwd         = cfg["fwd_days"]
    regime_cfg  = cfg["regime"]
    trigger_cfg = cfg["triggers"]
    min_trig    = cfg["min_triggers"]

    df_feat     = df_feat.reset_index(drop=True)
    dates_arr   = pd.to_datetime(df_feat["date"]).dt.date.values
    close_arr   = df_feat["close_val"].values.astype(float)
    n           = len(df_feat)

    # Train end index — threshold chỉ tính từ data trước đây
    train_end_idx = next(
        (i for i, d in enumerate(dates_arr) if d > train_end),
        len(dates_arr)
    )

    # Tính thresholds từ training window
    thresholds = {}
    for feat in list(regime_cfg.keys()) + list(trigger_cfg.keys()):
        if feat not in df_feat.columns:
            continue
        vals = df_feat[feat].iloc[:train_end_idx].dropna().values
        if len(vals) < 30:
            continue
        thresholds[f"{feat}_p30"] = np.percentile(vals, 30)
        thresholds[f"{feat}_p70"] = np.percentile(vals, 70)

    if len(thresholds) == 0:
        return []

    pnls = []

    for i in range(100, n - fwd):
        d = dates_arr[i]
        if d < window_start or d > window_end:
            continue

        # Regime check
        regime_ok = False
        for feat, direction in regime_cfg.items():
            if feat not in df_feat.columns:
                continue
            val    = df_feat[feat].iloc[i]
            thresh = thresholds.get(f"{feat}_p{'30' if direction == 'low' else '70'}")
            if thresh is None or not np.isfinite(val):
                continue
            regime_ok = (val <= thresh if direction == "low" else val >= thresh)
            break

        if not regime_ok:
            continue

        # Trigger check
        triggered = 0
        for feat, direction in trigger_cfg.items():
            if feat not in df_feat.columns:
                continue
            val    = df_feat[feat].iloc[i]
            thresh = thresholds.get(f"{feat}_p{'30' if direction == 'low' else '70'}")
            if thresh is None or not np.isfinite(val):
                continue
            if (direction == "low"  and val <= thresh) or \
               (direction == "high" and val >= thresh):
                triggered += 1

        if triggered < min_trig:
            continue

        # Trade: Time Stop FWD days
        entry = close_arr[i]
        if i + fwd < n:
            exit_p = close_arr[i + fwd]
            pnls.append((exit_p - entry) / entry * 100)

    return pnls


def metrics_from_pnls(pnls: list[float]) -> dict:
    if not pnls:
        return {"n": 0, "exp": 0.0, "wr": 0.0, "pf": 0.0, "ppd": 0.0}
    arr    = np.array(pnls)
    wins   = arr[arr > 0]
    losses = np.abs(arr[arr <= 0])
    gw     = wins.sum()   if len(wins)   > 0 else 0.0
    gl     = losses.sum() if len(losses) > 0 else 1e-9
    fwd    = 1  # placeholder — ppd tính riêng khi cần
    return {
        "n":   len(pnls),
        "exp": round(float(np.mean(arr)), 3),
        "wr":  round(float(len(wins) / len(arr) * 100), 1),
        "pf":  round(float(gw / gl), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# WALK FORWARD — template từ backtest_trailing_stop_v2.py
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward(symbol: str, cluster_name: str,
                 df_feat: pd.DataFrame) -> dict:
    """
    Walk Forward với expanding window — đúng template từ backtest_trailing_stop_v2.
    IS: train_window, OOS: test_window, threshold tính từ IS data.
    WFE = OOS_exp / IS_exp
    """
    dates_arr  = pd.to_datetime(df_feat["date"]).dt.date.values
    folds      = []
    fold_start = WF_START

    while True:
        train_end = fold_start + timedelta(days=WF_TRAIN_MONTHS * 30)
        test_end  = train_end  + timedelta(days=WF_TEST_MONTHS  * 30)
        if test_end > dates_arr[-1]:
            break

        # IS trades: trong [fold_start, train_end], threshold từ fold_start→train_end
        is_pnls  = collect_trades(df_feat, cluster_name,
                                  fold_start, train_end, train_end)

        # OOS trades: trong [train_end, test_end], threshold từ data trước train_end
        oos_pnls = collect_trades(df_feat, cluster_name,
                                  train_end, test_end, train_end)

        is_m  = metrics_from_pnls(is_pnls)
        oos_m = metrics_from_pnls(oos_pnls)

        if is_m["n"] < MIN_TRADES_FOLD or oos_m["n"] < MIN_TRADES_FOLD:
            fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)
            continue

        wfe = (oos_m["exp"] / is_m["exp"]) if is_m["exp"] > 0.001 else 0.0

        folds.append({
            "period":  f"{train_end.strftime('%Y-%m')}→{test_end.strftime('%Y-%m')}",
            "is_n":    is_m["n"],
            "oos_n":   oos_m["n"],
            "is_exp":  is_m["exp"],
            "oos_exp": oos_m["exp"],
            "oos_wr":  oos_m["wr"],
            "oos_pf":  oos_m["pf"],
            "wfe":     round(wfe, 2),
        })
        fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)

    if len(folds) < WF_MIN_FOLDS:
        return {"status": "INSUFFICIENT_FOLDS", "n_folds": len(folds), "folds": folds}

    oos_exps     = [f["oos_exp"] for f in folds]
    wfes         = [f["wfe"]     for f in folds]
    pos_folds    = sum(1 for x in oos_exps if x > 0)
    consistency  = round(pos_folds / len(folds) * 100, 1)
    valid_wfes   = [w for w in wfes if np.isfinite(w) and w != 0]
    avg_wfe      = round(float(np.mean(valid_wfes)), 2) if valid_wfes else 0.0
    avg_oos_exp  = round(float(np.mean(oos_exps)), 3)

    status = ("PASS" if avg_wfe >= MIN_WFE
              and consistency >= MIN_CONSISTENCY
              and avg_oos_exp > MIN_OOS_EXP
              else "FAIL")

    return {
        "status":       status,
        "avg_wfe":      avg_wfe,
        "avg_oos_exp":  avg_oos_exp,
        "consistency":  consistency,
        "n_folds":      len(folds),
        "folds":        folds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    try:
        from vn_loader import load_vn_ohlcv
    except ImportError:
        print("ERROR: Khong import duoc vn_loader.")
        sys.exit(1)

    print("=" * 65)
    print("VALIDATE NEW CLUSTERS — VN Trader Bot V6")
    print(f"WF: train={WF_TRAIN_MONTHS}m test={WF_TEST_MONTHS}m "
          f"min_folds={WF_MIN_FOLDS}")
    print(f"Filter: WFE>={MIN_WFE} consistency>={MIN_CONSISTENCY}% "
          f"OOS_exp>{MIN_OOS_EXP}")
    print("=" * 65)

    all_results = {}

    for cluster_name, cfg in CLUSTER_CONFIG.items():
        symbols = cfg["symbols"]
        fwd     = cfg["fwd_days"]

        print(f"\n{'='*65}")
        print(f"CLUSTER: {cluster_name} (FWD={fwd}d) — {len(symbols)} mã")
        print(f"Logic: regime={cfg['regime']} triggers={cfg['triggers']}")
        print(f"{'='*65}")

        passed = []
        failed = []

        for idx, sym in enumerate(symbols, 1):
            print(f"\n  [{idx:>2}/{len(symbols)}] {sym}...", end=" ", flush=True)

            # Load data
            df = None
            for attempt in range(3):
                try:
                    df = load_vn_ohlcv(sym, days=2500, min_bars=400)
                    break
                except Exception as e:
                    err = str(e).lower()
                    if "rate limit" in err or "60" in err:
                        wait = 65 * (attempt + 1)
                        print(f"wait {wait}s...", end=" ", flush=True)
                        time.sleep(wait)
                    else:
                        break

            if df is None or len(df) < 400:
                print("skip (data)")
                failed.append({"sym": sym, "reason": "data"})
                continue

            df_feat = compute_features_full(df)
            if df_feat is None:
                print("skip (features)")
                failed.append({"sym": sym, "reason": "features"})
                continue

            # Step 1: Quick backtest trên training data
            bt_pnls = collect_trades(df_feat, cluster_name,
                                     TRAIN_START, TRAIN_END, TRAIN_END)
            bt = metrics_from_pnls(bt_pnls)

            if bt["n"] < MIN_TRADES:
                print(f"BT FAIL (n={bt['n']} < {MIN_TRADES})")
                failed.append({"sym": sym, "reason": f"n={bt['n']}"})
                continue

            if bt["exp"] < MIN_EXP or bt["wr"] < MIN_WR or bt["pf"] < MIN_PF:
                print(f"BT FAIL (exp={bt['exp']:+.2f}% wr={bt['wr']:.0f}% pf={bt['pf']:.2f})")
                failed.append({"sym": sym, "reason": f"BT: exp={bt['exp']:+.2f}%"})
                continue

            print(f"BT OK (exp={bt['exp']:+.2f}% wr={bt['wr']:.0f}% "
                  f"pf={bt['pf']:.2f} n={bt['n']}) → WF...", end=" ", flush=True)

            # Step 2: Walk Forward
            wf = walk_forward(sym, cluster_name, df_feat)

            status_icon = "✅" if wf["status"] == "PASS" else "❌"
            print(f"{status_icon} {wf['status']} "
                  f"WFE={wf.get('avg_wfe','?')} "
                  f"consistency={wf.get('consistency','?')}% "
                  f"OOS_exp={wf.get('avg_oos_exp','?'):+}%")

            # Print fold details
            for fold in wf.get("folds", []):
                ok = "✅" if fold["oos_exp"] > 0 else "❌"
                print(f"      {fold['period']}: IS={fold['is_exp']:+.3f}% "
                      f"OOS={fold['oos_exp']:+.3f}% "
                      f"WFE={fold['wfe']} "
                      f"(n_oos={fold['oos_n']}) {ok}")

            result = {
                "sym":     sym,
                "cluster": cluster_name,
                "bt":      bt,
                "wf":      wf,
            }

            if wf["status"] == "PASS":
                passed.append(result)
            else:
                failed.append({"sym": sym, "reason": f"WF {wf['status']}"})

            time.sleep(1.1)

        all_results[cluster_name] = {"passed": passed, "failed": failed}

    # ── Final Report ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("FINAL REPORT")
    print(f"{'='*65}")

    for cluster_name, res in all_results.items():
        passed = res["passed"]
        failed = res["failed"]
        cfg    = CLUSTER_CONFIG[cluster_name]

        print(f"\n{'─'*65}")
        print(f"[{cluster_name}] — {len(passed)} pass / "
              f"{len(passed)+len(failed)} total")

        if passed:
            passed.sort(key=lambda x: (
                x["wf"]["avg_wfe"] * x["bt"]["exp"] * x["bt"]["pf"]
            ), reverse=True)

            print(f"\n  ✅ PASS ({len(passed)} mã):")
            print(f"  {'Sym':<10} {'BT_Exp':>7} {'BT_WR':>6} {'BT_PF':>6} "
                  f"{'WFE':>6} {'OOS_Exp':>8} {'Consist':>8} {'Score':>7}")
            print(f"  {'─'*65}")
            for r in passed:
                bt    = r["bt"]
                wf    = r["wf"]
                score = round(bt["exp"] * bt["pf"] * wf["avg_wfe"], 1)
                print(f"  {r['sym']:<10} {bt['exp']:>+6.2f}% "
                      f"{bt['wr']:>5.0f}% {bt['pf']:>6.2f} "
                      f"{wf['avg_wfe']:>6.2f} "
                      f"{wf['avg_oos_exp']:>+7.3f}% "
                      f"{wf['consistency']:>7.1f}% "
                      f"{score:>7.1f}")

            print(f"\n  # Thêm vào cluster_scanner.py:")
            print(f"  # {cluster_name.upper().replace(' ','_')}_SYMBOLS = "
                  f"{[r['sym'] for r in passed]}")
            print(f"  # FWD_DAYS[\"{cluster_name}\"] = {cfg['fwd_days']}")

        if failed:
            print(f"\n  ❌ FAIL ({len(failed)} mã):")
            for r in failed[:10]:
                print(f"    {r['sym']:<10} — {r['reason']}")
            if len(failed) > 10:
                print(f"    ... và {len(failed)-10} mã khác")


if __name__ == "__main__":
    run()
