"""
historical_analog.py — VN Signal Bot
Cosine similarity + MDS + price journey analytics
Session: 2026-04-27 — Applied 5 changes per user request
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLDS    = [0.80, 0.75, 0.70]
MIN_RESULTS              = 3
MIN_SAMPLE_WARNING       = 5
MIN_SAMPLE_DISTANCE_DAYS = 30   # Minimum Distance Sampling
MIN_SAMPLE_DISTANCE_FB   = 20   # fallback nếu < MIN_RESULTS


# ──────────────────────────────────────────────────────────────────────────────
# PRICE JOURNEY CALCULATION
# ──────────────────────────────────────────────────────────────────────────────
def _calc_price_journey(df, idx, window=90):
    """
    Tính hành trình giá trong `window` ngày tiếp theo kể từ idx.
    Returns dict với các metrics hoặc None nếu không đủ data.
    """
    if idx + window >= len(df):
        window = len(df) - idx - 1
    if window < 5:
        return None

    entry_price = df["close"].iloc[idx]
    future_slice = df.iloc[idx + 1: idx + 1 + window]

    if future_slice.empty:
        return None

    high_vals  = future_slice["high"].values
    low_vals   = future_slice["low"].values
    close_vals = future_slice["close"].values

    # MFE — max gain từ entry
    max_gain     = (high_vals.max() - entry_price) / entry_price
    max_gain_day = int(np.argmax(high_vals)) + 1  # 1-indexed

    # MAE — max drawdown từ entry
    max_dd      = (low_vals.min() - entry_price) / entry_price
    max_dd_day  = int(np.argmin(low_vals)) + 1

    # Forward returns
    n = len(close_vals)
    fwd_30 = (close_vals[min(29, n-1)] - entry_price) / entry_price if n >= 1 else None
    fwd_60 = (close_vals[min(59, n-1)] - entry_price) / entry_price if n >= 60 else None
    fwd_90 = (close_vals[min(89, n-1)] - entry_price) / entry_price if n >= 90 else None

    # ── CHANGE 4: Tính ngày hồi phục về hòa vốn sau MAE ──────────────────────
    # Tìm ngày đầu tiên sau max_dd_day mà giá >= entry_price
    recovery_days = None
    if max_dd < 0:  # chỉ tính khi có drawdown thực sự
        dd_idx = max_dd_day  # 1-indexed, tức slice index = max_dd_day
        post_dd_closes = close_vals[dd_idx:]  # từ ngày sau đáy
        for k, c in enumerate(post_dd_closes):
            if c >= entry_price:
                recovery_days = k + 1  # số ngày từ đáy đến hòa vốn
                break
        # Nếu không hồi phục trong window → None (caller handle "chưa hồi phục")

    return {
        "max_gain":      max_gain,
        "max_gain_day":  max_gain_day,
        "max_drawdown":  max_dd,
        "max_dd_day":    max_dd_day,
        "fwd_30":        fwd_30,
        "fwd_60":        fwd_60,
        "fwd_90":        fwd_90,
        "recovery_days": recovery_days,  # CHANGE 4: ngày hồi về hòa vốn (None = chưa hồi)
        "entry_price":   entry_price,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MINIMUM DISTANCE SAMPLING
# ──────────────────────────────────────────────────────────────────────────────
def _apply_mds(matches: list, min_distance: int) -> list:
    """
    Loại bỏ các ngày quá gần nhau (< min_distance bars).
    Input: list of (date_idx, similarity, journey_dict) sorted by date.
    Output: filtered list.
    """
    if not matches:
        return []
    selected = [matches[0]]
    for m in matches[1:]:
        if m[0] - selected[-1][0] >= min_distance:
            selected.append(m)
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# FIND SIMILAR
# ──────────────────────────────────────────────────────────────────────────────
def find_similar(current_vector: np.ndarray, cache: dict,
                 df=None, window=90) -> Optional[dict]:
    """
    Main similarity search với MDS.
    cache: {date_str: {"vector": np.ndarray, "idx": int}}
    df: OHLCV DataFrame để tính price journey
    Returns dict với "analogs" list và "_meta".
    """
    if not cache or current_vector is None:
        return None

    dates   = sorted(cache.keys())
    vectors = np.array([cache[d]["vector"] for d in dates])

    # Cosine similarity vectorized
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(current_vector)
    norms[norms == 0] = 1e-9
    similarities = (vectors @ current_vector) / norms

    threshold_used  = None
    raw_matches     = []

    for thresh in SIMILARITY_THRESHOLDS:
        idxs = np.where(similarities >= thresh)[0]
        if len(idxs) == 0:
            continue

        raw_matches = [
            (cache[dates[i]]["idx"], float(similarities[i]), dates[i])
            for i in idxs
        ]
        raw_matches.sort(key=lambda x: x[0])

        # MDS
        filtered = _apply_mds(raw_matches, MIN_SAMPLE_DISTANCE_DAYS)
        if len(filtered) >= MIN_RESULTS:
            threshold_used = thresh
            break
        # fallback MDS
        filtered_fb = _apply_mds(raw_matches, MIN_SAMPLE_DISTANCE_FB)
        if len(filtered_fb) >= MIN_RESULTS:
            threshold_used = thresh
            filtered = filtered_fb
            break

    if not raw_matches or threshold_used is None:
        return None

    # Build analogs
    analogs = []
    for (bar_idx, sim, date_str) in filtered:
        if df is not None:
            journey = _calc_price_journey(df, bar_idx, window)
        else:
            journey = None
        analogs.append({
            "date":       date_str,
            "similarity": sim,
            "bar_idx":    bar_idx,
            "journey":    journey,
        })

    return {
        "analogs": analogs,
        "_meta": {
            "total_matches":     len(raw_matches),
            "independent_n":     len(filtered),
            "search_bars":       len(dates),
            "avg_similarity":    float(np.mean([a["similarity"] for a in analogs])),
            "threshold_used":    threshold_used,
            "min_distance_used": MIN_SAMPLE_DISTANCE_DAYS,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# COMPUTE BASE STATS
# ──────────────────────────────────────────────────────────────────────────────
def compute_base_stats(analogs: list) -> Optional[dict]:
    """
    Tính toàn bộ metrics từ list analogs (đã qua MDS).
    """
    journeys = [a["journey"] for a in analogs if a.get("journey")]
    if not journeys:
        return None

    fwd30s = [j["fwd_30"] for j in journeys if j.get("fwd_30") is not None]
    fwd60s = [j["fwd_60"] for j in journeys if j.get("fwd_60") is not None]
    fwd90s = [j["fwd_90"] for j in journeys if j.get("fwd_90") is not None]
    maes   = [j["max_drawdown"] for j in journeys]
    mfes   = [j["max_gain"]     for j in journeys]
    holds  = [j["max_gain_day"] for j in journeys]
    recovery_list = [j["recovery_days"] for j in journeys
                     if j.get("recovery_days") is not None]

    if not fwd30s:
        return None

    arr = np.array(fwd30s)
    wins = arr[arr > 0]
    loss = arr[arr < 0]

    pf = 99.0
    if len(loss) > 0 and loss.sum() != 0:
        pf = float(wins.sum() / abs(loss.sum())) if len(wins) > 0 else 0.0

    std_ret = float(np.std(arr)) if len(arr) > 1 else 0.001
    median_ret = float(np.median(arr))

    # Capture rate
    mfe_arr = np.array(mfes)
    capture_rates = []
    for j in journeys:
        if j.get("fwd_30") is not None and j.get("max_gain", 0) > 0:
            capture_rates.append(j["fwd_30"] / j["max_gain"] * 100)

    return {
        "median_ret":       median_ret,
        "std_ret":          std_ret,
        "win_rate":         len(wins) / len(arr),
        "win_count":        len(wins),
        "loss_count":       len(loss),
        "total_n":          len(arr),
        "return_vol_ratio": median_ret / std_ret if std_ret > 0 else 0,
        "expectancy":       float(np.mean(arr)),
        "profit_factor":    pf,
        "p25_ret":          float(np.percentile(arr, 25)),
        "p75_ret":          float(np.percentile(arr, 75)),
        "median_60":        float(np.median(fwd60s)) if fwd60s else None,
        "median_90":        float(np.median(fwd90s)) if fwd90s else None,
        "mae_avg":          float(np.mean(maes)),
        "mae_worst":        float(np.min(maes)),
        "mfe_avg":          float(np.mean(mfes)),
        "mfe_best":         float(np.max(mfes)),
        "hold_avg":         float(np.mean(holds)),
        "hold_median":      float(np.median(holds)),
        "hold_max":         float(np.max(holds)),
        "mfe_capture_rate": float(np.mean(capture_rates)) if capture_rates else None,
        # CHANGE 4 data
        "recovery_days_avg":    float(np.mean(recovery_list)) if recovery_list else None,
        "recovery_days_n":      len(recovery_list),
        "recovery_not_healed":  len(journeys) - len(recovery_list),
        # CHANGE 5 data — tính trong format function để có context
    }


# ──────────────────────────────────────────────────────────────────────────────
# BEST HOLDING PERIOD  (CHANGE 1 applied here)
# ──────────────────────────────────────────────────────────────────────────────
def _best_holding(analogs: list) -> dict:
    """
    CHANGE 1: Dùng median(max_gain_day) làm thời gian nắm giữ khuyến nghị
    thay vì mốc WR cao nhất (tránh bull bias dài hạn).
    Vẫn giữ WR breakdown theo mốc để tham khảo.
    """
    journeys = [a["journey"] for a in analogs if a.get("journey")]
    if not journeys:
        return {"recommended_hold": None, "wr_by_horizon": {}}

    # CHANGE 1: median của max_gain_day
    hold_days = [j["max_gain_day"] for j in journeys]
    median_hold = int(np.median(hold_days))
    # Round về bội số 5 gần nhất để đẹp hơn
    recommended = round(median_hold / 5) * 5
    if recommended == 0:
        recommended = 5

    # WR breakdown các mốc chuẩn (để tham khảo)
    wr_by_horizon = {}
    for horizon in [30, 60, 90]:
        key = f"fwd_{horizon}"
        vals = [j[key] for j in journeys if j.get(key) is not None]
        if vals:
            wins = sum(1 for v in vals if v > 0)
            wr_by_horizon[horizon] = {"wr": wins / len(vals), "n": len(vals)}

    return {
        "recommended_hold": recommended,
        "median_hold_raw":  median_hold,
        "wr_by_horizon":    wr_by_horizon,
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAE SURVIVAL RATE  (CHANGE 5)
# ──────────────────────────────────────────────────────────────────────────────
def _calc_mae_survival_rate(analogs: list) -> dict:
    """
    CHANGE 5: Tỷ lệ các analog có MAE < 0 (tức có sụt giảm thực sự)
    mà sau đó fwd_30 > 0 (vượt qua và về dương trong 30D).
    """
    journeys = [a["journey"] for a in analogs if a.get("journey")]
    had_drawdown = [j for j in journeys
                    if j.get("max_drawdown") is not None and j["max_drawdown"] < -0.01]
    survived = [j for j in had_drawdown
                if j.get("fwd_30") is not None and j["fwd_30"] > 0]
    return {
        "total_with_drawdown": len(had_drawdown),
        "survived_count":      len(survived),
        "survival_rate":       len(survived) / len(had_drawdown) if had_drawdown else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ACTION PLAN  (CHANGE 3 — revised formula)
# ──────────────────────────────────────────────────────────────────────────────
def _build_action_plan(stats: dict, current_price: float) -> dict:
    """
    CHANGE 3: Kế hoạch hành động tự động từ stats.
    Công thức được sửa so với đề xuất ban đầu:
    - Entry: giá hiện tại (không ± vì MAE là drawdown, không phải spread)
    - SL: dùng mae_worst (worst case thực tế từ analogs) thay vì mae_avg+2%
    - TP1: giá × (1 + median_ret)
    - TP2: giá × (1 + p75_ret)
    - Warning nếu MAE TB > 10%
    """
    if not current_price or current_price <= 0:
        return {}

    entry     = current_price
    # SL: dùng mae_avg (thực tế) chứ không dùng mae_worst (quá rộng, vô nghĩa)
    # mae_worst vẫn hiển thị riêng để tham khảo
    sl_pct    = stats.get("mae_avg", 0)         # âm, e.g. -0.147
    sl_worst  = stats.get("mae_worst", 0)       # worst case reference
    tp1_pct   = stats.get("median_ret", 0)
    tp2_pct   = stats.get("p75_ret", 0)
    mae_avg   = stats.get("mae_avg", 0)

    sl_price  = entry * (1 + sl_pct)   if sl_pct  < 0 else None
    tp1_price = entry * (1 + tp1_pct)  if tp1_pct > 0 else None
    tp2_price = entry * (1 + tp2_pct)  if tp2_pct > 0 else None

    high_risk_warning = mae_avg < -0.10  # MAE TB > 10%

    return {
        "entry":            entry,
        "sl_price":         sl_price,
        "sl_pct":           sl_pct,
        "sl_worst_pct":     sl_worst,
        "sl_worst_price":   entry * (1 + sl_worst) if sl_worst < 0 else None,
        "tp1_price":        tp1_price,
        "tp1_pct":          tp1_pct,
        "tp2_price":        tp2_price,
        "tp2_pct":          tp2_pct,
        "high_risk_warning": high_risk_warning,
        "mae_avg_pct":      mae_avg,
    }


# ──────────────────────────────────────────────────────────────────────────────
# FORMAT ANALOG REPORT  (main output function)
# ──────────────────────────────────────────────────────────────────────────────
def format_analog_report(result: dict, symbol: str = "",
                          current_price: float = None) -> str:
    """
    Format báo cáo tương đồng lịch sử — 4 phần chính + 3 phần mới.
    result: output của find_similar()
    current_price: giá hiện tại (cho Action Plan, nếu có)
    """
    if not result:
        return "Không tìm được mẫu tương đồng."

    analogs = result.get("analogs", [])
    meta    = result.get("_meta", {})

    if not analogs:
        return "Không có analog đủ điều kiện sau MDS."

    stats       = compute_base_stats(analogs)
    holding     = _best_holding(analogs)
    survival    = _calc_mae_survival_rate(analogs)
    action_plan = _build_action_plan(stats, current_price) if current_price else {}

    n_total       = meta.get("total_matches", 0)
    n_indep       = meta.get("independent_n", len(analogs))
    history_bars  = meta.get("search_bars", 0)
    avg_sim       = meta.get("avg_similarity", 0)
    threshold     = meta.get("threshold_used", 0.80)
    min_dist      = meta.get("min_distance_used", MIN_SAMPLE_DISTANCE_DAYS)

    lines = []
    hdr = f"PHAN TICH TUONG DONG: {symbol}" if symbol else "PHAN TICH TUONG DONG"
    lines.append("═" * 42)
    lines.append(hdr)
    lines.append("═" * 42)

    # ── PHẦN 1: TÓM TẮT NHANH ────────────────────────────────────────────────
    lines.append(f"Nguong     : {int(threshold*100)}%")
    lines.append(f"Mau doc lap: {n_indep}/{n_total} ngay (loc {min_dist}D)")
    lines.append(f"Lich su    : {history_bars} ngay | Do TD TB: {avg_sim*100:.1f}%")

    if n_indep < MIN_SAMPLE_WARNING:
        lines.append(f"⚠ Mau nho ({n_indep} mau) — ket qua co tinh tham khao cao")

    # ── PHẦN 2: HÀNH TRÌNH GIÁ ───────────────────────────────────────────────
    lines.append("")
    lines.append("HANH TRINH GIA (90D tiep theo):")
    lines.append("─" * 42)

    if stats:
        mfe_avg   = stats["mfe_avg"]   * 100
        mfe_best  = stats["mfe_best"]  * 100
        mae_avg   = stats["mae_avg"]   * 100
        mae_worst = stats["mae_worst"] * 100
        mfe_mae_ratio = abs(mfe_avg / mae_avg) if mae_avg != 0 else 99

        lines.append(f"  MFE (dinh cao nhat): TB {mfe_avg:+.1f}% | Max {mfe_best:+.1f}%")
        lines.append(f"  MAE (day sau nhat) : TB {mae_avg:+.1f}% | Worst {mae_worst:+.1f}%")
        lines.append(f"  MFE/MAE ratio      : {mfe_mae_ratio:.2f}x (ly tuong>2.0x)")

        cap = stats.get("mfe_capture_rate")
        if cap is not None:
            cap_note = "exit som" if cap < 40 else ("exit hieu qua" if cap > 80 else "trung binh")
            lines.append(f"  MFE thu duoc (30D) : {cap:.0f}%  {cap_note}")

        hold_med = stats["hold_median"]
        hold_max = stats["hold_max"]
        lines.append(f"  Hold den dinh TB   : {stats['hold_avg']:.0f}D (median {hold_med:.0f}D, max {hold_max:.0f}D)")

        if stats.get("median_ret") is not None and stats.get("p75_ret") is not None:
            lines.append(f"  Ket qua 30D: tot {stats['p75_ret']*100:+.1f}% | xau {stats['mae_avg']*100:+.1f}%")

    # ── PHẦN 3: THỐNG KÊ ĐẦY ĐỦ ─────────────────────────────────────────────
    lines.append("")
    lines.append(f"THONG KE ({n_indep} MAU DOC LAP — loc {min_dist}D):")
    lines.append("─" * 42)

    if stats:
        wc   = stats["win_count"]
        lc   = stats["loss_count"]
        tot  = stats["total_n"]
        wr   = stats["win_rate"] * 100

        lines.append(f"  WR 30D        : {wc}/{tot} ({wr:.0f}%) | Thua: {100-wr:.0f}%")
        lines.append(f"  Median LN 30D : {stats['median_ret']*100:+.2f}%"
                     f" [P25:{stats['p25_ret']*100:+.1f}% P75:{stats['p75_ret']*100:+.1f}%]")
        lines.append(f"  Expectancy    : {stats['expectancy']*100:+.2f}%")
        lines.append(f"  Profit Factor : {stats['profit_factor']:.2f}")

        rvr = stats["return_vol_ratio"]
        # ── CHANGE 2: ghi chú Return/Vol thấp ──────────────────────────────
        rvr_note = ""
        if rvr < 0.5:
            rvr_note = " (Thap do bien dong manh dau ky – phu hop nguoi chiu rung lac)"
        lines.append(f"  Return/Vol 30D: {rvr:.2f}{rvr_note}")

        if stats.get("median_60") is not None:
            lines.append(f"  Median LN 60D : {stats['median_60']*100:+.2f}%")
        if stats.get("median_90") is not None:
            lines.append(f"  Median LN 90D : {stats['median_90']*100:+.2f}%")

        lines.append(f"  MAE TB (MDD)  : {stats['mae_avg']*100:+.2f}%")
        lines.append(f"  MFE TB (Peak) : {stats['mfe_avg']*100:+.2f}%")

        # CHANGE 1: Thời gian nắm giữ khuyến nghị
        rec_hold = holding.get("recommended_hold")
        med_raw  = holding.get("median_hold_raw")
        if rec_hold:
            lines.append(f"  Thoi gian nam giu KN: {rec_hold}D (median dinh: {med_raw}D)")

        # WR breakdown tham khảo
        wr_bh = holding.get("wr_by_horizon", {})
        if 90 in wr_bh:
            lines.append(f"  Thoi gian TU  : 90D (WR={wr_bh[90]['wr']*100:.0f}%)")

    # ── CHANGE 4: PHỤC HỒI SAU SỤT GIẢM ─────────────────────────────────────
    if stats:
        rec_avg = stats.get("recovery_days_avg")
        rec_n   = stats.get("recovery_days_n", 0)
        not_healed = stats.get("recovery_not_healed", 0)

        lines.append("")
        lines.append("PHUC HOI SAU SUT GIAM:")
        lines.append("─" * 42)
        if rec_avg is not None and rec_n > 0:
            lines.append(f"  Phuc hoi TB sau sut giam : {rec_avg:.0f} ngay (tren {rec_n} mau)")
            if not_healed > 0:
                lines.append(f"  Chua hoi phuc trong 90D  : {not_healed} mau")
        else:
            lines.append("  Khong du du lieu phuc hoi (tat ca mau chua hoi ve hoa von trong 90D)")

    # ── CHANGE 5: MAE SURVIVAL RATE ──────────────────────────────────────────
    lines.append("")
    lines.append("MAE SURVIVAL RATE:")
    lines.append("─" * 42)
    total_dd = survival["total_with_drawdown"]
    survived = survival["survived_count"]
    surv_rate = survival.get("survival_rate")
    if surv_rate is not None:
        lines.append(f"  Ty le vuot qua sut giam : {survived}/{total_dd} ({surv_rate*100:.0f}%)")
        if surv_rate >= 0.70:
            lines.append("  → Cap do phuc hoi TOT: phan lon truong hop sut giam roi phuc hoi")
        elif surv_rate >= 0.50:
            lines.append("  → Cap do phuc hoi TRUNG BINH: can quan sat them")
        else:
            lines.append("  → Cap do phuc hoi THAP: rui ro khong phuc hoi cao")
    else:
        lines.append("  Khong du du lieu (khong co mau nao co drawdown >1%)")

    # ── CHANGE 3: KẾ HOẠCH HÀNH ĐỘNG ────────────────────────────────────────
    if action_plan:
        lines.append("")
        lines.append("KE HOACH HANH DONG (tu dong):")
        lines.append("─" * 42)
        ep = action_plan["entry"]
        lines.append(f"  Entry Zone : {ep:,.0f} (gia hien tai)")

        if action_plan.get("sl_price"):
            sl_p = action_plan["sl_price"]
            sl_pct = action_plan["sl_pct"] * 100
            lines.append(f"  Stop Loss  : {sl_p:,.0f} ({sl_pct:+.1f}%) — MAE TB (tham khao)")
            # worst case reference
            sw_p = action_plan.get("sl_worst_price")
            sw_pct = action_plan.get("sl_worst_pct", 0) * 100
            if sw_p:
                lines.append(f"  SL Worst   : {sw_p:,.0f} ({sw_pct:+.1f}%) — MAE toi te nhat")

        if action_plan.get("tp1_price"):
            tp1_p = action_plan["tp1_price"]
            tp1_pct = action_plan["tp1_pct"] * 100
            lines.append(f"  TP1        : {tp1_p:,.0f} ({tp1_pct:+.1f}%) — Median return")

        if action_plan.get("tp2_price"):
            tp2_p = action_plan["tp2_price"]
            tp2_pct = action_plan["tp2_pct"] * 100
            lines.append(f"  TP2        : {tp2_p:,.0f} ({tp2_pct:+.1f}%) — P75 return")

        if action_plan.get("high_risk_warning"):
            mae_pct = action_plan["mae_avg_pct"] * 100
            lines.append(f"  ⚠ MAE TB {mae_pct:.1f}%: Chi vao lenh neu chap nhan rui ro sut giam manh")

    # ── PHẦN 4: CẢNH BÁO RỦI RO ──────────────────────────────────────────────
    lines.append("")
    lines.append("CANH BAO:")
    lines.append("─" * 42)

    warnings = []
    if stats:
        mae_avg_pct = stats["mae_avg"] * 100
        if mae_avg_pct < -10:
            warnings.append(f" MAE TB {mae_avg_pct:.1f}%: nhip giam manh truoc khi tang.")
        if avg_sim < 0.85:
            warnings.append(f" Do TD TB < 85%, ket qua chi mang tinh tham khao.")
        if stats["win_rate"] < 0.6:
            warnings.append(f" WR thap ({stats['win_rate']*100:.0f}%): xac suat thua kha cao.")

    warnings.append(" Luu y: Phan tich chi mang tinh tham khao. QK khong dam bao TL.")

    for w in warnings:
        lines.append(w)

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# DEMO / TEST với data HTG mô phỏng
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    random.seed(42)
    np.random.seed(42)

    # Simulate HTG-like analogs dựa trên output thực tế đã có:
    # WR 8/11, median 4.82%, P25=-1.0%, P75=9.1%
    # MAE TB -14.7%, MFE TB +20.4%, hold median 64D
    # Tạo 11 mẫu synthetic match các stats trên

    def _make_journey(fwd30, mfe, mae, gain_day, dd_day, rec_days=None):
        return {
            "max_gain":      mfe,
            "max_gain_day":  gain_day,
            "max_drawdown":  mae,
            "max_dd_day":    dd_day,
            "fwd_30":        fwd30,
            "fwd_60":        fwd30 * 1.95 if fwd30 else None,
            "fwd_90":        fwd30 * 3.2  if fwd30 else None,
            "recovery_days": rec_days,
            "entry_price":   28000,
        }

    # 11 analogs: 8 win, 3 loss (WR=73%)
    fake_journeys = [
        # Wins
        _make_journey(0.355, 0.461, -0.082, 88, 10, 15),
        _make_journey(0.091, 0.180, -0.210, 72, 22, 35),
        _make_journey(0.145, 0.220, -0.050, 55, 8,  12),
        _make_journey(0.048, 0.120, -0.195, 60, 30, 48),
        _make_journey(0.082, 0.195, -0.170, 64, 25, 40),
        _make_journey(0.037, 0.095, -0.280, 45, 15, None),   # chưa hồi trong 90D
        _make_journey(0.110, 0.230, -0.120, 70, 18, 28),
        _make_journey(0.067, 0.156, -0.098, 58, 12, 20),
        # Losses
        _make_journey(-0.141, 0.043, -0.376, 20, 55, None),  # chưa hồi
        _make_journey(-0.032, 0.078, -0.142, 35, 40, None),  # chưa hồi
        _make_journey(-0.012, 0.060, -0.188, 40, 50, 85),
    ]

    fake_analogs = [
        {
            "date": f"2023-{i+1:02d}-15",
            "similarity": 0.80 + random.uniform(0, 0.08),
            "bar_idx": i * 60,
            "journey": j,
        }
        for i, j in enumerate(fake_journeys)
    ]

    fake_result = {
        "analogs": fake_analogs,
        "_meta": {
            "total_matches":     35,
            "independent_n":     11,
            "search_bars":       938,
            "avg_similarity":    0.838,
            "threshold_used":    0.80,
            "min_distance_used": 30,
        },
    }

    # Giá HTG hiện tại (giả định)
    current_price = 28000  # VND

    output = format_analog_report(fake_result, symbol="HTG",
                                  current_price=current_price)
    print(output)
