"""
expand_watchlist.py — Mở rộng watchlist từ HOSE universe
VN Trader Bot V6 — Session 31

Pipeline:
  Bước 1: Lấy ~50 mã HOSE volume > 3 tỷ/ngày (loại trừ watchlist hiện tại)
  Bước 2: Cluster assignment — MR hay MOM dựa vào Cohen's d analysis
  Bước 3: Backtest signal logic trên training data (2019-2023)
           → Lọc: Exp > 0, WR > 45%, PF > 1.1, n_trades >= 15
  Bước 4: Walk Forward validation (2022→nay)
           → Lọc: WFE > 0.3, consistency >= 60%, OOS Exp > 0
  Bước 5: Output danh sách mã đủ điều kiện + config

Chạy: python expand_watchlist.py
      python expand_watchlist.py 2>&1 | tee expand_results.txt
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Watchlist hiện tại — loại trừ khỏi scan
CURRENT_MR  = {"DCM", "NKG", "DPM", "HAH", "HCM", "HSG", "DGC", "GAS"}
CURRENT_MOM = {"VCB", "BID", "MBB", "MWG", "CTG", "FRT", "REE", "FPT",
               "GMD", "STB", "PNJ", "TCB"}
CURRENT_ALL = CURRENT_MR | CURRENT_MOM

# Volume filter
MIN_VOL_BILLION = 3.0   # tỷ VND/ngày
# Không giới hạn TOP_N — lấy tất cả mã đủ volume > MIN_VOL_BILLION

# Backtest filter (training 2019-2023)
TRAIN_START   = date(2019, 1, 1)
TRAIN_END     = date(2023, 12, 31)
MIN_TRADES    = 15
MIN_EXP       = 0.3     # % per trade
MIN_WR        = 45.0    # %
MIN_PF        = 1.1

# Walk Forward filter
WF_START         = date(2022, 1, 1)
WF_TRAIN_MONTHS  = 18
WF_TEST_MONTHS   = 6
MIN_WFE          = 0.3
MIN_CONSISTENCY  = 60.0  # %
MIN_OOS_EXP      = 0.0   # OOS avg exp > 0

# Signal logic (copy từ cluster_scanner.py)
FWD_DAYS = {"Mean Reversion": 20, "Momentum": 10}
MIN_TRIGGERS = 2
TRIGGER_PCT  = 70

SIGNAL_CONFIG = {
    "Mean Reversion": {
        "regime_indicator":   "price_vs_sma50",
        "regime_condition":   "low",
        "trigger_indicators": ["stoch_k", "volume_spike", "momentum_5d"],
        "trigger_direction":  {"stoch_k": "low", "volume_spike": "high",
                               "momentum_5d": "high"},
    },
    "Momentum": {
        "regime_indicator":   "ema_cross",
        "regime_condition":   "high",
        "trigger_indicators": ["momentum_5d", "volume_spike", "candle_body"],
        "trigger_direction":  {"momentum_5d": "high", "volume_spike": "high",
                               "candle_body": "high"},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ema(c, span):
    return pd.Series(c.astype(float)).ewm(span=span, adjust=False).mean().values

def _sma(c, p):
    return pd.Series(c.astype(float)).rolling(p, min_periods=p).mean().values

def _compute_all_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Tính indicators cho toàn bộ df (vectorized). Trả về df với cột indicators."""
    if len(df) < 100:
        return None
    close = df["close"].values.astype(float)
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    vol   = df["volume"].values.astype(float)
    opn   = df["open"].values.astype(float)
    n     = len(close)

    ema12  = _ema(close, 12)
    ema26  = _ema(close, 26)
    sma20  = _sma(close, 20)
    sma50  = _sma(close, 50)
    vsma20 = _sma(vol,   20)

    h_prev = np.concatenate([[close[0]], close[:-1]])
    tr     = np.maximum(high - low,
             np.maximum(np.abs(high - h_prev), np.abs(low - h_prev)))
    atr14  = _sma(tr, 14)

    lo14 = pd.Series(low).rolling(14).min().values
    hi14 = pd.Series(high).rolling(14).max().values
    denom = np.where(hi14 - lo14 == 0, 1e-9, hi14 - lo14)
    stoch = 100 * (close - lo14) / denom

    c5 = np.concatenate([[close[0]]*5, close[:-5]])

    result = df.copy()
    result["price_vs_sma50"] = (close - sma50) / (close + 1e-9) * 100
    result["ema_cross"]      = (ema12 - ema26) / (close + 1e-9) * 100
    result["momentum_5d"]    = (close / (c5 + 1e-9) - 1.0) * 100
    result["volume_spike"]   = (vol / (vsma20 + 1e-9)) - 1.0
    result["stoch_k"]        = stoch
    result["candle_body"]    = np.clip(np.abs(close - opn) / (atr14 + 1e-9), 0, 3)
    result["atr"]            = atr14
    result["close_val"]      = close
    result["sma50"]          = sma50

    return result


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: CLUSTER ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def assign_cluster(df: pd.DataFrame) -> str:
    """
    Quyết định mã thuộc MR hay MOM cluster dựa vào Cohen's d analysis.

    Logic từ Session 30:
    - MR: khi giá dưới SMA50, các trigger (stoch oversold, volume, momentum dương)
          có Cohen's d cao hơn về phía MR
    - MOM: khi EMA12>EMA26, momentum cao hơn

    Simplified: so sánh predictive power của regime indicator
    cho MR signal logic vs MOM signal logic.
    """
    if len(df) < 200:
        return "Momentum"  # default

    close = df["close"].values.astype(float)
    sma50 = _sma(close, 50)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)

    # Tính forward return 10 ngày tại mỗi điểm
    fwd_ret = np.full(len(close), np.nan)
    for i in range(len(close) - 10):
        fwd_ret[i] = (close[i+10] - close[i]) / close[i] * 100

    valid = np.isfinite(sma50) & np.isfinite(fwd_ret) & (close > 0)

    # MR regime: giá dưới SMA50
    mr_regime = valid & (close < sma50)
    # MOM regime: EMA12 > EMA26
    mom_regime = valid & (ema12 > ema26)

    if mr_regime.sum() < 20 or mom_regime.sum() < 20:
        return "Momentum"

    mr_ret  = fwd_ret[mr_regime]
    mom_ret = fwd_ret[mom_regime]
    base_ret = fwd_ret[valid]

    # Cohen's d so với baseline
    base_std = np.std(base_ret) + 1e-9
    mr_d  = (np.mean(mr_ret)  - np.mean(base_ret)) / base_std
    mom_d = (np.mean(mom_ret) - np.mean(base_ret)) / base_std

    return "Mean Reversion" if mr_d > mom_d else "Momentum"


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def backtest_symbol(df: pd.DataFrame, cluster: str,
                    start: date, end: date) -> dict:
    """
    Backtest signal logic trên window [start, end].
    Dùng expanding threshold từ đầu data đến ngày entry.
    """
    ind_df = _compute_all_indicators(df)
    if ind_df is None:
        return {"n": 0}

    dates_arr = pd.to_datetime(ind_df["date"]).dt.date.values
    close_arr = ind_df["close_val"].values
    fwd       = FWD_DAYS[cluster]
    cfg       = SIGNAL_CONFIG[cluster]
    reg_ind   = cfg["regime_indicator"]
    reg_cond  = cfg["regime_condition"]
    trig_inds = cfg["trigger_indicators"]
    trig_dirs = cfg["trigger_direction"]
    n         = len(ind_df)

    pnls = []

    for i in range(100, n - fwd):
        d = dates_arr[i]
        if d < start or d > end:
            continue

        # Expanding threshold
        sub = ind_df.iloc[:i]
        reg_vals = sub[reg_ind].dropna().values
        if len(reg_vals) < 30:
            continue

        reg_thresh = np.percentile(reg_vals,
                                   TRIGGER_PCT if reg_cond == "low"
                                   else 100 - TRIGGER_PCT)

        # Check regime
        val = ind_df[reg_ind].iloc[i]
        if not np.isfinite(val):
            continue
        in_regime = (val <= reg_thresh) if reg_cond == "low" else (val > reg_thresh)
        if not in_regime:
            continue

        # Check triggers
        triggered = 0
        for t in trig_inds:
            tv   = ind_df[t].iloc[i]
            th_v = sub[t].dropna().values
            if len(th_v) < 20 or not np.isfinite(tv):
                continue
            direction = trig_dirs.get(t, "high")
            th = np.percentile(th_v, TRIGGER_PCT if direction == "high"
                               else 100 - TRIGGER_PCT)
            if (direction == "low" and tv <= th) or (direction == "high" and tv >= th):
                triggered += 1

        if triggered < MIN_TRIGGERS:
            continue

        # Trade: entry T+1, exit T+1+fwd
        entry = close_arr[i]
        if i + fwd < n:
            exit_p = close_arr[i + fwd]
            pnls.append((exit_p - entry) / entry * 100)

    if len(pnls) < MIN_TRADES:
        return {"n": len(pnls)}

    pnls   = np.array(pnls)
    wins   = pnls[pnls > 0]
    losses = np.abs(pnls[pnls <= 0])
    gw     = wins.sum()   if len(wins)   > 0 else 0.0
    gl     = losses.sum() if len(losses) > 0 else 1e-9

    return {
        "n":   len(pnls),
        "exp": round(float(np.mean(pnls)), 3),
        "wr":  round(float(len(wins) / len(pnls) * 100), 1),
        "pf":  round(float(gw / gl), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: WALK FORWARD
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward(df: pd.DataFrame, cluster: str) -> dict:
    """Walk Forward validation với expanding window."""
    folds      = []
    fold_start = WF_START

    dates_arr = pd.to_datetime(df["date"]).dt.date.values

    while True:
        train_end = fold_start + timedelta(days=WF_TRAIN_MONTHS * 30)
        test_end  = train_end  + timedelta(days=WF_TEST_MONTHS  * 30)
        if test_end > dates_arr[-1]:
            break

        # IS
        is_m = backtest_symbol(df, cluster, fold_start, train_end)
        # OOS
        oos_m = backtest_symbol(df, cluster, train_end, test_end)

        if is_m["n"] < 5 or oos_m["n"] < 3:
            fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)
            continue

        is_exp  = is_m.get("exp",  0)
        oos_exp = oos_m.get("exp", 0)
        wfe     = (oos_exp / is_exp) if is_exp > 0 else 0.0

        folds.append({
            "period":  f"{train_end.strftime('%Y-%m')}→{test_end.strftime('%Y-%m')}",
            "is_exp":  is_exp,
            "oos_exp": oos_exp,
            "wfe":     round(wfe, 2),
            "oos_wr":  oos_m.get("wr", 0),
            "oos_n":   oos_m["n"],
        })
        fold_start = fold_start + timedelta(days=WF_TEST_MONTHS * 30)

    if len(folds) < 3:
        return {"status": "INSUFFICIENT_FOLDS", "folds": folds}

    oos_exps     = [f["oos_exp"] for f in folds]
    wfes         = [f["wfe"]     for f in folds]
    pos_folds    = sum(1 for x in oos_exps if x > 0)
    consistency  = round(pos_folds / len(folds) * 100, 1)
    avg_wfe      = round(float(np.mean([w for w in wfes if np.isfinite(w)])), 2)
    avg_oos_exp  = round(float(np.mean(oos_exps)), 3)

    if avg_wfe < MIN_WFE or consistency < MIN_CONSISTENCY or avg_oos_exp <= MIN_OOS_EXP:
        status = "FAIL"
    else:
        status = "PASS"

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
        print("ERROR: Khong import duoc vn_loader. Chay trong project dir.")
        sys.exit(1)

    # ── Bước 1: Lấy universe từ HOSE ─────────────────────────────────────────
    print("=" * 65)
    print("EXPAND WATCHLIST — VN Trader Bot V6")
    print("=" * 65)
    print(f"\nBước 1: Lấy universe HOSE (vol > {MIN_VOL_BILLION} tỷ/ngày)...")

    universe = _get_hose_universe(load_vn_ohlcv)
    if not universe:
        print("ERROR: Không lấy được universe. Kiểm tra kết nối.")
        sys.exit(1)

    # Loại trừ watchlist hiện tại
    candidates = [s for s in universe if s not in CURRENT_ALL]
    print(f"  Universe đủ volume: {len(universe)} mã")
    print(f"  Trừ {len(CURRENT_ALL)} mã hiện tại → {len(candidates)} candidates mới")
    print(f"  {candidates}\n")

    # ── Bước 2-4: Pipeline từng mã ───────────────────────────────────────────
    passed    = []
    failed    = []
    skipped   = []

    total = len(candidates)
    for idx, sym in enumerate(candidates, 1):
        print(f"[{idx:>3}/{total}] {sym}", end=" ... ", flush=True)

        # Load data (với retry nếu rate limit)
        df = None
        for attempt in range(3):
            try:
                df = load_vn_ohlcv(sym, days=2500, min_bars=400)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "60" in err_str:
                    wait = 65 * (attempt + 1)
                    print(f"rate limit, wait {wait}s...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    break
        if df is None or len(df) < 400:
            print("skip (data)")
            skipped.append(sym)
            continue

        # Bước 2: Cluster assignment
        cluster = assign_cluster(df)

        # Bước 3: Backtest training
        bt = backtest_symbol(df, cluster, TRAIN_START, TRAIN_END)
        if bt["n"] < MIN_TRADES:
            print(f"FAIL (n={bt['n']} < {MIN_TRADES})")
            failed.append({"sym": sym, "cluster": cluster,
                           "reason": f"n={bt['n']} < {MIN_TRADES}", "bt": bt})
            continue

        exp = bt.get("exp", 0)
        wr  = bt.get("wr",  0)
        pf  = bt.get("pf",  0)

        if exp < MIN_EXP or wr < MIN_WR or pf < MIN_PF:
            reason = f"exp={exp:+.2f}% wr={wr:.0f}% pf={pf:.2f}"
            print(f"FAIL ({reason})")
            failed.append({"sym": sym, "cluster": cluster,
                           "reason": reason, "bt": bt})
            continue

        print(f"BT OK (exp={exp:+.2f}% wr={wr:.0f}% pf={pf:.2f} n={bt['n']}) → WF...",
              end=" ", flush=True)

        # Bước 4: Walk Forward
        wf = walk_forward(df, cluster)
        if wf["status"] == "PASS":
            sizing_score = exp * pf * wf["avg_wfe"]
            print(f"✅ PASS (WFE={wf['avg_wfe']} consistency={wf['consistency']}% "
                  f"oos_exp={wf['avg_oos_exp']:+.2f}% score={sizing_score:.1f})")
            passed.append({
                "sym":          sym,
                "cluster":      cluster,
                "bt_exp":       exp,
                "bt_wr":        wr,
                "bt_pf":        pf,
                "bt_n":         bt["n"],
                "wf_wfe":       wf["avg_wfe"],
                "wf_oos_exp":   wf["avg_oos_exp"],
                "wf_consist":   wf["consistency"],
                "sizing_score": round(sizing_score, 1),
                "wf_folds":     wf["folds"],
            })
        else:
            print(f"❌ WF FAIL ({wf['status']} WFE={wf.get('avg_wfe','?')} "
                  f"consistency={wf.get('consistency','?')}%)")
            failed.append({"sym": sym, "cluster": cluster,
                           "reason": f"WF {wf['status']}", "bt": bt, "wf": wf})

        time.sleep(1.1)  # rate limit: 60 req/phút

    # ── Final Report ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"FINAL RESULTS — {len(passed)} mã pass / {total} candidates")
    print(f"{'='*65}")

    if passed:
        # Sort by sizing score desc
        passed.sort(key=lambda x: x["sizing_score"], reverse=True)

        print(f"\n✅ MÃ ĐỦ ĐIỀU KIỆN THÊM VÀO WATCHLIST ({len(passed)}):")
        print(f"{'Sym':<7} {'Cluster':<18} {'Exp':>6} {'WR':>5} {'PF':>5} "
              f"{'WFE':>5} {'OOS_Exp':>8} {'Consist':>8} {'Score':>6}")
        print("-" * 70)
        for r in passed:
            print(f"{r['sym']:<7} {r['cluster']:<18} "
                  f"{r['bt_exp']:>+5.2f}% {r['bt_wr']:>4.0f}% "
                  f"{r['bt_pf']:>5.2f} {r['wf_wfe']:>5.2f} "
                  f"{r['wf_oos_exp']:>+7.2f}% {r['wf_consist']:>7.1f}% "
                  f"{r['sizing_score']:>6.1f}")

        print(f"\n\n# THÊM VÀO cluster_scanner.py:")
        mr_new  = [r for r in passed if r["cluster"] == "Mean Reversion"]
        mom_new = [r for r in passed if r["cluster"] == "Momentum"]

        if mr_new:
            syms = [r["sym"] for r in mr_new]
            print(f'\n# Mean Reversion — thêm vào MR_SYMBOLS:')
            print(f'# {syms}')

        if mom_new:
            syms = [r["sym"] for r in mom_new]
            print(f'\n# Momentum — thêm vào MOM_SYMBOLS:')
            print(f'# {syms}')

        print(f"\n# SYMBOL_STATS entries mới:")
        for r in passed:
            # Estimate PF dùng công thức S31
            rr     = 1.17 if r["cluster"] == "Mean Reversion" else 1.39
            wr     = r["bt_wr"] / 100
            exp    = r["bt_exp"]
            denom  = wr*rr - (1-wr)
            if abs(denom) > 0.001:
                avg_loss = exp / denom
                avg_win  = rr * avg_loss
                gw       = wr * avg_win
                gl       = (1-wr) * avg_loss
                pf_est   = round(gw/gl, 2) if gl > 0 else 1.0
            else:
                pf_est = 1.0
            # WFE từ WF
            wfe = r["wf_wfe"]
            # n estimate: dùng bt_n scaled
            print(f'    "{r["sym"]}": {{"wr": {int(r["bt_wr"])}, '
                  f'"exp": {r["bt_exp"]:.1f}, '
                  f'"wfe": {wfe:.2f}, '
                  f'"n": {r["bt_n"]}, '
                  f'"pf": {pf_est}, '
                  f'"cluster": "{r["cluster"]}"}},')

    if failed:
        print(f"\n❌ MÃ KHÔNG ĐẠT ({len(failed)}):")
        for r in failed[:20]:  # chỉ in 20 đầu
            print(f"  {r['sym']:<7} {r['cluster']:<18} — {r['reason']}")
        if len(failed) > 20:
            print(f"  ... và {len(failed)-20} mã khác")

    if skipped:
        print(f"\n⚠️  BỎ QUA (data issues): {skipped}")


def _get_hose_universe(load_vn_ohlcv) -> list[str]:
    """Lấy danh sách mã HOSE có volume > MIN_VOL_BILLION."""
    # Thử lấy listing từ vnstock
    all_symbols = []
    try:
        from vnstock import Vnstock
        listing = Vnstock().stock(symbol="VCB", source="VCI").listing.symbols_by_exchange()
        hose_df = listing[listing["exchange"].str.upper() == "HOSE"]
        all_symbols = hose_df["symbol"].str.upper().tolist()
        logger.info(f"  VCI listing: {len(all_symbols)} HOSE symbols")
    except Exception as e:
        logger.warning(f"  VCI fail: {e}")
        try:
            from vnstock import Vnstock
            listing = Vnstock().stock(symbol="VCB", source="KBS").listing.symbols_by_exchange()
            hose_df = listing[listing["exchange"].str.upper() == "HOSE"]
            all_symbols = hose_df["symbol"].str.upper().tolist()
            logger.info(f"  KBS listing: {len(all_symbols)} HOSE symbols")
        except Exception as e2:
            logger.error(f"  KBS fail: {e2}")
            return []

    # Filter chứng quyền
    all_symbols = [s for s in all_symbols
                   if not (len(s) > 3 and s[0] == 'C' and s[-1].isdigit())]

    # Tính volume từng mã, lấy top TOP_N_UNIVERSE
    logger.info(f"  Checking volume for {len(all_symbols)} symbols...")
    vol_map = {}
    for i, sym in enumerate(all_symbols):
        if i % 50 == 0:
            logger.info(f"  Progress: {i}/{len(all_symbols)}...")
        for attempt in range(3):  # retry tối đa 3 lần
            try:
                df = load_vn_ohlcv(sym, days=40, min_bars=20)
                if df is None or len(df) < 20:
                    break
                close = df["close"].values[-20:].astype(float)
                vol   = df["volume"].values[-20:].astype(float)
                avg_vnd = float((vol * close).mean()) * 1000
                if avg_vnd >= MIN_VOL_BILLION * 1e9:
                    vol_map[sym] = avg_vnd
                break  # thành công → thoát retry loop
            except Exception as e:
                err_str = str(e).lower()
                if "rate limit" in err_str or "60" in err_str:
                    wait = 65 * (attempt + 1)  # 65s, 130s, 195s
                    logger.warning(f"  Rate limit hit at {sym}, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    break  # lỗi khác → bỏ qua
        time.sleep(1.1)  # rate limit: 60 req/phút → 1.1s/req

    # Sort và lấy top N
    sorted_syms = sorted(vol_map, key=vol_map.get, reverse=True)  # lấy tất cả đủ volume
    logger.info(f"  {len(sorted_syms)} symbols with vol > {MIN_VOL_BILLION}B VND")
    return sorted_syms


if __name__ == "__main__":
    run()
