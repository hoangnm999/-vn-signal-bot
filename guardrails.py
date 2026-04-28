"""
guardrails.py — 5-layer Guard Rails cho Historical Analog Scanner.

Layer 1: Data Quality Gates
Layer 2: Statistical Sanity Checks
Layer 3: Score Penalties & Caps (Reference Score)
Layer 4: Output Framing
Layer 5: System-level Safeguards
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

# Layer 1
MIN_VOLUME_BILLION    = 3.0    # tỷ VND / ngày TB 20 phiên
MIN_CACHE_DAYS        = 300    # ngày lịch sử tối thiểu
MIN_SAMPLE_FOR_RANK   = 5      # số ngày tương đồng tối thiểu để rank
MAX_CHECK_AGE_HOURS   = 48     # giờ — quá cũ thì refresh

# Layer 2
OUTLIER_RETURN_THRESH = 15.0   # % — return > này AND sample < 10 → flag
OUTLIER_SAMPLE_THRESH = 10     # số mẫu ngưỡng cho outlier check
HIGH_DISPERSION_STD   = 20.0   # % std — quá phân tán
RECENCY_BIAS_THRESH   = 0.60   # > 60% mẫu trong 12 tháng gần → flag

# Layer 3
WINRATE_FLOOR         = 0.40   # hard gate — dưới này reject luôn (giảm từ 0.50)
WINRATE_SOFT_FLOOR    = 0.50   # dưới này ghi chú warning nhưng không cap
EXPECTANCY_FLOOR      = 0.0    # hard gate — Expectancy <= 0 → reject
PF_FLOOR              = 1.0    # hard gate — Profit Factor < 1 → reject
MAXDD_FLAG_THRESH     = -15.0  # % — flag "Rủi ro cao", giảm score × 0.8
MAXDD_EXCLUDE_THRESH  = -25.0  # % — loại top5, đưa vào mục riêng
SHARPE_FLOOR          = 0.30   # dưới này score × 0.8
SMALL_SAMPLE_PENALTY  = 0.70   # nhân hệ số nếu sample < MIN_SAMPLE_FOR_RANK
SCORE_MAX             = 8.5    # ceiling — không bao giờ = 10
SCORE_CAP_LOW_WR      = 4.0    # cap khi WinRate < 50% (soft warning)

# Layer 5
MARKET_DOWN_THRESH    = -3.0   # % — VN-Index giảm > 3% trong 5 ngày → warning
SCORE_OPPORTUNITY_MIN = 5.0    # dưới này "không có cơ hội đủ tin cậy"


# ══════════════════════════════════════════════════════════════════════
# LAYER 1 — DATA QUALITY GATES
# ══════════════════════════════════════════════════════════════════════

class GateReject(Exception):
    """Raise khi mã không qua Gate — chứa lý do."""
    pass


def gate_data_quality(
    symbol:          str,
    volume_avg_bill: Optional[float],   # tỷ VND, TB 20 phiên
    cache_days:      Optional[int],      # số rows trong CSV cache
    check_age_hours: Optional[float],    # giờ kể từ /check gần nhất
) -> dict:
    """
    Kiểm tra 4 gates chất lượng dữ liệu.
    Raise GateReject nếu không qua.
    Return dict thông tin nếu pass.
    """
    issues = []

    # Gate 1: Thanh khoản
    if volume_avg_bill is not None and volume_avg_bill < MIN_VOLUME_BILLION:
        raise GateReject(
            f"Thanh khoản TB {volume_avg_bill:.1f} tỷ < {MIN_VOLUME_BILLION} tỷ "
            f"— dễ bị thao túng giá, bỏ qua"
        )

    # Gate 2: Lịch sử đủ dài
    if cache_days is not None and cache_days < MIN_CACHE_DAYS:
        raise GateReject(
            f"Cache chỉ có {cache_days} ngày < {MIN_CACHE_DAYS} ngày tối thiểu"
        )

    # Gate 3 & 4 chỉ warning, không reject (xử lý ở caller)
    if check_age_hours is not None and check_age_hours > MAX_CHECK_AGE_HOURS:
        issues.append(f"Dữ liệu /check đã {check_age_hours:.0f}h — cần refresh")

    return {"symbol": symbol, "gate": "PASS", "issues": issues}


# ══════════════════════════════════════════════════════════════════════
# LAYER 2 — STATISTICAL SANITY CHECKS
# ══════════════════════════════════════════════════════════════════════

def compute_base_stats(analogs: list) -> dict:
    """
    Tính toán các chỉ số thống kê từ danh sách analogs.
    Dùng MEDIAN thay vì mean để loại outlier.

    Regime Filter: nếu analogs có trường regime_weight → tính weighted WR,
    weighted Expectancy song song với unweighted để so sánh và hiển thị.
    """
    f30s  = [a["fwd_30"]        for a in analogs if a.get("fwd_30")        is not None]
    mdds  = [a["max_drawdown"]  for a in analogs if a.get("max_drawdown")  is not None]
    close = analogs[0].get("close", 0) if analogs else 0

    n = len(f30s)
    if n == 0:
        return {"valid": False, "n": 0}

    median_ret = float(np.median(f30s))
    std_ret    = float(np.std(f30s, ddof=1)) if n > 1 else 0.0
    win_rate   = sum(1 for x in f30s if x > 0) / n
    median_mdd = float(np.median(mdds)) if mdds else 0.0

    # ── Regime-weighted statistics ────────────────────────────────────────────
    # Nếu analogs có regime_weight → tính weighted WR và weighted Expectancy
    weights = [a.get("regime_weight", 1.0) for a in analogs if a.get("fwd_30") is not None]
    total_w = sum(weights)
    regime_filter_active = any(a.get("regime_weight", 1.0) != 1.0 for a in analogs)

    if regime_filter_active and total_w > 0:
        # Weighted WR: Σ(weight * 1_win) / Σ(weight)
        weighted_win_sum = sum(
            w for a, w in zip(
                [a for a in analogs if a.get("fwd_30") is not None],
                weights
            ) if a.get("fwd_30", 0) > 0
        )
        win_rate_weighted = weighted_win_sum / total_w

        # Weighted Expectancy: Σ(weight * fwd_30) / Σ(weight)
        weighted_exp = sum(
            f * w for f, w in zip(f30s, weights)
        ) / total_w

        weighted_n = round(total_w, 2)
    else:
        win_rate_weighted = win_rate
        weighted_exp      = float(np.mean(f30s))
        weighted_n        = float(n)

    # Regime match stats
    current_regime  = analogs[0].get("_meta", {}).get("current_regime", 0) if analogs else 0
    same_regime_n   = sum(
        1 for a in analogs
        if a.get("sample_regime", 0) == current_regime and current_regime > 0
    )
    regime_match_pct = same_regime_n / len(analogs) if analogs and current_regime > 0 else None

    # Return/Vol ratio 30D
    return_vol_ratio = (median_ret / std_ret) if std_ret > 0 else 0.0

    # Expectancy = weighted nếu có, unweighted nếu không
    expectancy = round(weighted_exp, 2)

    # Profit Factor
    pos_sum = sum(x for x in f30s if x > 0)
    neg_sum = abs(sum(x for x in f30s if x < 0))
    profit_factor = round(pos_sum / neg_sum, 2) if neg_sum > 0 else 99.0

    # P25 / P75
    p25 = float(np.percentile(f30s, 25)) if n >= 4 else None
    p75 = float(np.percentile(f30s, 75)) if n >= 4 else None

    # Recency
    from datetime import date, timedelta
    cutoff_12m    = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    n_recent      = sum(1 for a in analogs if a.get("date", "") >= cutoff_12m)
    recency_ratio = n_recent / len(analogs) if analogs else 0

    return {
        "valid":              True,
        "n":                  n,
        "median_ret":         round(median_ret, 2),
        "std_ret":            round(std_ret, 2),
        # win_rate: dùng weighted nếu có regime filter
        "win_rate":           round(win_rate_weighted, 4),
        "win_rate_raw":       round(win_rate, 4),          # unweighted — để compare
        "weighted_n":         weighted_n,                  # effective sample count
        "regime_filter_active": regime_filter_active,
        "regime_match_pct":   round(regime_match_pct, 2) if regime_match_pct is not None else None,
        "return_vol_ratio":   round(return_vol_ratio, 3),
        "median_mdd":         round(median_mdd, 2),
        "expectancy":         expectancy,                  # weighted expectancy
        "profit_factor":      profit_factor,
        "p25_ret":            round(p25, 2) if p25 is not None else None,
        "p75_ret":            round(p75, 2) if p75 is not None else None,
        "recency_ratio":      round(recency_ratio, 3),
        "close":              close,
        "median_60":          round(float(np.median(
                                  [a["fwd_60"] for a in analogs if a.get("fwd_60") is not None]
                              )), 2) if any(a.get("fwd_60") for a in analogs) else None,
        "median_90":          round(float(np.median(
                                  [a["fwd_90"] for a in analogs if a.get("fwd_90") is not None]
                              )), 2) if any(a.get("fwd_90") for a in analogs) else None,
        "mae_avg":    round(float(np.mean(
                          [a["max_drawdown"] for a in analogs if a.get("max_drawdown") is not None]
                      )), 2) if any(a.get("max_drawdown") for a in analogs) else None,
        "mae_worst":  round(float(np.min(
                          [a["max_drawdown"] for a in analogs if a.get("max_drawdown") is not None]
                      )), 2) if any(a.get("max_drawdown") for a in analogs) else None,
        "mfe_avg":    round(float(np.mean(
                          [a["max_gain"] for a in analogs if a.get("max_gain") is not None]
                      )), 2) if any(a.get("max_gain") for a in analogs) else None,
        "mfe_best":   round(float(np.max(
                          [a["max_gain"] for a in analogs if a.get("max_gain") is not None]
                      )), 2) if any(a.get("max_gain") for a in analogs) else None,
        "hold_avg":   round(float(np.mean(
                          [a["max_gain_day"] for a in analogs if a.get("max_gain_day") is not None]
                      )), 1) if any(a.get("max_gain_day") for a in analogs) else None,
        "hold_median":round(float(np.median(
                          [a["max_gain_day"] for a in analogs if a.get("max_gain_day") is not None]
                      )), 1) if any(a.get("max_gain_day") for a in analogs) else None,
        "mfe_capture_rate": round(float(np.mean(
                          [a["fwd_30"] / a["max_gain"] * 100
                           for a in analogs
                           if a.get("fwd_30") is not None
                           and a.get("max_gain") is not None
                           and a["max_gain"] > 0]
                      )), 1) if any(
                          a.get("max_gain") and a["max_gain"] > 0 for a in analogs
                      ) else None,
    }


def sanity_check(stats: dict) -> list[str]:
    """
    Trả về list flags từ Layer 2.
    """
    flags = []
    n           = stats.get("n", 0)
    median_ret  = stats.get("median_ret", 0)
    std_ret     = stats.get("std_ret", 0)
    recency     = stats.get("recency_ratio", 0)
    median_60   = stats.get("median_60")
    median_90   = stats.get("median_90")

    # Check 1: Outlier risk
    if median_ret > OUTLIER_RETURN_THRESH and n < OUTLIER_SAMPLE_THRESH:
        flags.append(
            f"OUTLIER_RISK: Return median {median_ret:+.1f}% cao bất thường "
            f"với chỉ {n} mẫu — có thể bị ảnh hưởng bởi biến động cá biệt"
        )

    # Check 2: Dispersion quá cao
    if std_ret > HIGH_DISPERSION_STD:
        flags.append(
            f"HIGH_DISPERSION: Std {std_ret:.1f}% > {HIGH_DISPERSION_STD}% "
            f"— kết quả quá phân tán, khó dự đoán"
        )

    # Check 3: Dead cat bounce
    if (median_ret > 0 and
            median_60 is not None and median_60 < 0 and
            median_90 is not None and median_90 < 0):
        flags.append(
            "DEAD_CAT: Tăng ngắn hạn (30D dương) nhưng xu hướng "
            "dài hạn yếu (60D và 90D âm)"
        )

    # Check 4: Recency bias
    if recency > RECENCY_BIAS_THRESH:
        flags.append(
            f"RECENCY_BIAS: {recency:.0%} mẫu trong 12 tháng gần nhất "
            f"— pattern mới, chưa được kiểm chứng đủ lâu"
        )

    return flags


# ══════════════════════════════════════════════════════════════════════
# LAYER 3 — REFERENCE SCORE
# ══════════════════════════════════════════════════════════════════════

def compute_reference_score(
    stats:       dict,
    flags:       list[str],
    all_stats:   list[dict],
    n_override:  Optional[int] = None,
) -> tuple[float, list[str], str]:
    """
    Tính Reference Score (0–8.5).

    FORMULA MỚI — dựa trên Expectancy + Profit Factor:
      raw_score = expectancy_z * 0.40
                + pf_z         * 0.30
                + rvr_z        * 0.20
                + mdd_z        * 0.10

    Hard gates (reject trước khi tính score):
      - Expectancy <= 0      → EXCLUDED
      - Profit Factor < 1.0  → EXCLUDED
      - Win Rate < 40%       → EXCLUDED
      - MaxDD < -25%         → EXCLUDED

    Win Rate chỉ dùng làm soft warning (không ảnh hưởng score công thức).

    Returns:
        (score, penalty_notes, risk_tier)
        risk_tier: "NORMAL" | "HIGH_RISK" | "EXCLUDED"
    """
    if not stats.get("valid"):
        return 0.0, ["Khong du du lieu thong ke"], "EXCLUDED"

    n           = n_override if n_override is not None else stats["n"]
    win_rate    = stats["win_rate"]
    median_ret  = stats["median_ret"]
    expectancy  = stats.get("expectancy", median_ret)  # fallback median_ret nếu chưa có
    pf          = stats.get("profit_factor", 0.0)
    rvr         = stats.get("return_vol_ratio", 0.0)
    median_mdd  = stats["median_mdd"]
    penalties   = []
    risk_tier   = "NORMAL"

    # ══════════════════════════════════════════════════════════════════
    # HARD GATES — reject trước khi tính score
    # ══════════════════════════════════════════════════════════════════

    # Gate 1: Expectancy âm → không có edge
    if expectancy <= EXPECTANCY_FLOOR:
        return 0.0, [
            f"Expectancy {expectancy:+.1f}% <= 0 — khong co edge, reject"
        ], "EXCLUDED"

    # Gate 2: Profit Factor < 1 → thua nhiều hơn thắng về giá trị
    if pf < PF_FLOOR:
        return 0.0, [
            f"Profit Factor {pf:.2f} < 1.0 — tong thua > tong thang, reject"
        ], "EXCLUDED"

    # Gate 3: Win Rate < 40% → quá ít lần thắng
    if win_rate < WINRATE_FLOOR:
        return 0.0, [
            f"Win Rate {win_rate:.0%} < {WINRATE_FLOOR:.0%} — reject"
        ], "EXCLUDED"

    # Gate 4: MaxDD quá lớn
    if median_mdd < MAXDD_EXCLUDE_THRESH:
        return 0.0, [
            f"MaxDD median {median_mdd:.1f}% < {MAXDD_EXCLUDE_THRESH}% — rui ro qua cao, reject"
        ], "EXCLUDED"

    # ══════════════════════════════════════════════════════════════════
    # Z-SCORE NORMALIZATION trên toàn watchlist
    # ══════════════════════════════════════════════════════════════════

    def _z(field: str, val: float, invert: bool = False) -> float:
        vals = [
            s[field] for s in all_stats
            if s.get("valid") and s.get(field) is not None
        ]
        if len(vals) < 2:
            return 0.0
        arr  = np.array(vals, dtype=float)
        std  = float(np.std(arr, ddof=1))
        mean = float(np.mean(arr))
        if std < 1e-9:
            return 0.0
        z = (val - mean) / std
        return float(np.clip(-z if invert else z, -3, 3))

    z_exp = _z("expectancy",        expectancy)
    z_pf  = _z("profit_factor",     pf)
    z_rvr = _z("return_vol_ratio",  rvr)
    z_mdd = _z("median_mdd",        median_mdd, invert=True)

    # ══════════════════════════════════════════════════════════════════
    # FORMULA MỚI — Expectancy + PF làm chủ đạo
    # ══════════════════════════════════════════════════════════════════
    # z-score range [-3, 3] → raw range ~ [-3, 3]
    # Scale: (raw + 3) / 6 * 10 → [0, 10]
    raw_score = (
        z_exp * 0.40 +   # Expectancy — metric quyết định có edge không
        z_pf  * 0.30 +   # Profit Factor — chất lượng W/L ratio
        z_rvr * 0.20 +   # Return/Vol — consistency, tránh high vol
        z_mdd * 0.10     # Drawdown control — risk guard
    )
    score = float(np.clip((raw_score + 3.0) / 6.0 * 10, 0, 10))

    # ══════════════════════════════════════════════════════════════════
    # SOFT WARNINGS — không reject nhưng ghi chú và điều chỉnh score
    # ══════════════════════════════════════════════════════════════════

    # Soft 1: WR 40-50% — warning, nhân 0.85
    if win_rate < WINRATE_SOFT_FLOOR:
        score *= 0.85
        penalties.append(
            f"WR {win_rate:.0%} thap (40-50%) — can than, score x0.85"
        )

    # Soft 2: MaxDD flag
    if median_mdd < MAXDD_FLAG_THRESH:
        score    *= 0.8
        risk_tier = "HIGH_RISK"
        penalties.append(
            f"MaxDD median {median_mdd:.1f}% → Rui ro cao, score x0.8"
        )

    # Soft 3: Return/Vol thấp
    if rvr < SHARPE_FLOOR and risk_tier == "NORMAL":
        score *= 0.8
        penalties.append(f"Return/Vol {rvr:.2f} < {SHARPE_FLOOR} → score x0.8")

    # Soft 4: Mẫu nhỏ
    if n < MIN_SAMPLE_FOR_RANK:
        score *= SMALL_SAMPLE_PENALTY
        penalties.append(
            f"Mau nho ({n} < {MIN_SAMPLE_FOR_RANK}) → score x{SMALL_SAMPLE_PENALTY}"
        )

    # Soft 5: Outlier risk từ Layer 2
    if any("OUTLIER_RISK" in f for f in flags):
        score *= 0.6
        penalties.append("Outlier risk → score x0.6")

    # Soft 6: High dispersion — bỏ z_rvr khỏi tính toán
    if any("HIGH_DISPERSION" in f for f in flags):
        raw2  = z_exp * 0.50 + z_pf * 0.35 + z_mdd * 0.15
        score = float(np.clip((raw2 + 3.0) / 6.0 * 10, 0, 10))
        penalties.append("High dispersion → bo Return/Vol khoi cong thuc")

    # Soft 7: Dead cat — cảnh báo xu hướng dài hạn yếu
    if any("DEAD_CAT" in f for f in flags):
        score *= 0.85
        penalties.append("Dead cat pattern — xu huong dai han yeu, score x0.85")

    # Score ceiling
    score = min(score, SCORE_MAX)
    score = round(score, 2)

    return score, penalties, risk_tier


# ══════════════════════════════════════════════════════════════════════
# LAYER 4 — OUTPUT FRAMING
# ══════════════════════════════════════════════════════════════════════

_FLAG_LABELS = {
    "OUTLIER_RISK":    "⚠️ Kết quả có thể bị ảnh hưởng bởi biến động cá biệt",
    "HIGH_DISPERSION": "⚠️ Độ phân tán cao, khó dự đoán",
    "DEAD_CAT":        "⚠️ Tăng ngắn hạn, xu hướng dài hạn yếu",
    "RECENCY_BIAS":    "⚠️ Pattern mới, chưa được kiểm chứng đủ lâu",
}

_SCORE_DISCLAIMER = (
    "* Reference Score là gợi ý sắp xếp, không phải thước đo chính xác tuyệt đối."
)
_SYSTEM_DISCLAIMER = (
    "Phân tích dựa trên tương đồng lịch sử kỹ thuật. "
    "Quá khứ không đảm bảo tương lai. "
    "Chỉ dùng để sàng lọc — không phải tín hiệu mua/bán."
)


def format_symbol_entry(
    rank:          int,
    symbol:        str,
    stats:         dict,
    score:         float,
    flags:         list[str],
    penalties:     list[str],
    risk_tier:     str,
    n_total:       int,   # ngày thô vượt ngưỡng similarity
    n_independent: int = 0,  # mẫu độc lập sau Minimum Distance Sampling
) -> str:
    """Format một mã trong output top 5."""
    win_rate   = stats["win_rate"]
    median_ret = stats["median_ret"]
    median_mdd = stats["median_mdd"]
    rvr        = stats.get("return_vol_ratio", 0)
    expectancy = stats.get("expectancy", 0)
    pf         = stats.get("profit_factor", 0)
    pf_str     = "99.00" if pf >= 99 else f"{pf:.2f}"
    p25        = stats.get("p25_ret")
    p75        = stats.get("p75_ret")
    close      = stats.get("close", 0)
    n          = stats["n"]

    em = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"

    lines = [f"{em} #{rank} — {symbol} (Giá: {close:,.0f})"]

    # Regime filter info — hiển thị nếu đang active
    regime_filter = stats.get("regime_filter_active", False)
    weighted_n    = stats.get("weighted_n", float(n))
    regime_match  = stats.get("regime_match_pct")
    win_rate_raw  = stats.get("win_rate_raw")

    if regime_filter:
        regime_names = {1: "R1 Bull Quiet", 2: "R2 Bull Volatile",
                        3: "R3 Bear Quiet",  4: "R4 Bear Volatile"}
        cur_r = 0
        # Lấy current_regime từ _meta của analog đầu tiên (nếu có)
        cur_r_name = regime_names.get(cur_r, "")
        match_str  = f" | Khop regime: {regime_match:.0%}" if regime_match is not None else ""
        lines.append(
            f"🎯 Regime Filter: ON | Weighted N: {weighted_n:.1f}{match_str}"
        )
        if win_rate_raw is not None and abs(win_rate_raw - win_rate) > 0.03:
            lines.append(
                f"   WR co trong so: {win_rate:.0%} "
                f"(raw khong loc: {win_rate_raw:.0%})"
            )

    # Sample info — hiển thị mẫu độc lập (sau Minimum Distance Sampling)
    n_ind = n_independent if n_independent > 0 else n
    sample_note = " ⚠️ Mẫu nhỏ" if n_ind < MIN_SAMPLE_FOR_RANK else ""
    lines.append(
        f"📊 Mẫu độc lập: {n_ind}/{n_total} | "
        f"WR 30D: {win_rate:.0%} | Thua: {1-win_rate:.0%}"
        f"{sample_note}"
    )

    # Return với confidence interval
    ci_str = ""
    if p25 is not None and p75 is not None:
        ci_str = f" [P25: {p25:+.1f}%, P75: {p75:+.1f}%]"
    lines.append(
        f"💰 LN kỳ vọng (median): {median_ret:+.1f}%{ci_str} | "
        f"Rủi ro (median MDD): {median_mdd:.1f}%"
    )

    lines.append(
        f"📉 Expectancy: {expectancy:+.1f}% | "
        f"Profit Factor: {pf_str} | "
        f"Return/Vol 30D: {rvr:.2f}"
    )
    lines.append(
        f"🎯 Reference Score: {score:.1f}/10 *"
    )

    # MAE/MFE + Hold Duration từ price journey của các analog
    mae_avg = stats.get("mae_avg")
    mfe_avg = stats.get("mfe_avg")
    hold_avg = stats.get("hold_avg")
    capture = stats.get("mfe_capture_rate")
    if mae_avg is not None and mfe_avg is not None:
        capture_str = ""
        if capture is not None:
            cap_note = " ⚠️ exit sớm" if capture < 40 else " ✅ hiệu quả" if capture > 80 else ""
            capture_str = f" | Thu: {capture:.0f}%{cap_note}"
        lines.append(
            f"📉 MAE TB: {mae_avg:+.1f}% | MFE TB: {mfe_avg:+.1f}%{capture_str}"
        )
    if hold_avg is not None:
        lines.append(f"⏱️  Hold đỉnh TB: {hold_avg:.0f} ngày")

    if risk_tier == "HIGH_RISK":
        lines.append("🔴 Rủi ro cao")

    # Flags từ Layer 2
    for f in flags:
        key = f.split(":")[0]
        if key in _FLAG_LABELS:
            lines.append(_FLAG_LABELS[key])

    return "\n".join(lines)


def format_full_report(
    ranked:         list[dict],   # list {symbol, stats, score, flags, penalties, risk_tier}
    excluded:       list[dict],   # mã bị loại do MaxDD < -25%
    market_warning: Optional[str],
    scan_date:      str,
) -> str:
    """Tạo full text report."""
    parts = []

    # Header
    parts.append(f"🏆 TOP CƠ HỘI THEO LỊCH SỬ ({scan_date})")
    parts.append("═" * 38)
    parts.append(_SYSTEM_DISCLAIMER)
    parts.append("")

    # Market warning (Layer 5)
    if market_warning:
        parts.append(f"⚠️ {market_warning}")
        parts.append("")

    if not ranked:
        parts.append(
            "Không tìm thấy cơ hội nào đủ tin cậy hôm nay. "
            "Hãy kiên nhẫn chờ đợi."
        )
    else:
        top5 = [r for r in ranked if r["risk_tier"] != "EXCLUDED"][:5]
        for i, r in enumerate(top5, 1):
            entry = format_symbol_entry(
                rank          = i,
                symbol        = r["symbol"],
                stats         = r["stats"],
                score         = r["score"],
                flags         = r["flags"],
                penalties     = r["penalties"],
                risk_tier     = r["risk_tier"],
                n_total       = r.get("n_total", r["stats"]["n"]),
                n_independent = r.get("n_independent", 0),
            )
            parts.append(entry)
            parts.append("")

        parts.append(_SCORE_DISCLAIMER)

    # Excluded mã rủi ro rất cao
    if excluded:
        parts.append("")
        parts.append("─" * 38)
        parts.append("CƠ HỘI RỦI RO RẤT CAO (MaxDD < -25%):")
        for r in excluded:
            s   = r["stats"]
            exp    = s.get("expectancy", 0)
            pf     = s.get("profit_factor", 0)
            pf_str = "99.00" if pf >= 99 else f"{pf:.2f}"
            parts.append(
                f"  {r['symbol']}: Score {r['score']:.1f} | "
                f"WR {s['win_rate']:.0%} | "
                f"Median {s['median_ret']:+.1f}% | "
                f"Expectancy {exp:+.1f}% | "
                f"PF {pf_str} | "
                f"MDD {s['median_mdd']:.1f}%"
            )
        parts.append("  ⚠️ Chỉ phù hợp cho nhà đầu tư chấp nhận rủi ro rất cao.")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════
# LAYER 5 — SYSTEM-LEVEL SAFEGUARDS
# ══════════════════════════════════════════════════════════════════════

def check_market_regime(vnindex_5d_change: Optional[float]) -> Optional[str]:
    """
    Trả về chuỗi cảnh báo nếu VN-Index đang trong xu hướng giảm mạnh.
    None nếu bình thường.
    """
    if vnindex_5d_change is None:
        return None
    if vnindex_5d_change < MARKET_DOWN_THRESH:
        return (
            f"Thị trường đang trong xu hướng giảm "
            f"(VN-Index {vnindex_5d_change:+.1f}% trong 5 ngày) — "
            f"mọi tín hiệu tích cực cần thận trọng gấp đôi"
        )
    return None


def check_no_opportunity(ranked: list[dict]) -> bool:
    """True nếu không có mã nào đạt Score > SCORE_OPPORTUNITY_MIN."""
    valid = [r for r in ranked if r["risk_tier"] != "EXCLUDED"]
    return not any(r["score"] > SCORE_OPPORTUNITY_MIN for r in valid)


# ══════════════════════════════════════════════════════════════════════
# PIPELINE — Chạy toàn bộ 5 layers cho 1 mã
# ══════════════════════════════════════════════════════════════════════

def run_guardrails_for_symbol(
    symbol:          str,
    analogs:         list,
    all_stats:       list[dict],   # stats của tất cả mã để z-score
    volume_avg_bill: Optional[float] = None,
    cache_days:      Optional[int]   = None,
    check_age_hours: Optional[float] = None,
) -> dict:
    """
    Chạy đầy đủ 5 layers cho 1 mã.
    Returns dict kết quả với đầy đủ thông tin.
    """
    result = {
        "symbol":        symbol,
        "gate":          "PASS",
        "stats":         {},
        "flags":         [],
        "score":         0.0,
        "penalties":     [],
        "risk_tier":     "EXCLUDED",
        "n_total":       0,
        "n_independent": 0,
    }

    # Layer 1
    try:
        gate_data_quality(symbol, volume_avg_bill, cache_days, check_age_hours)
    except GateReject as e:
        result["gate"]     = "REJECT"
        result["reason"]   = str(e)
        return result

    # Không có analogs
    if not analogs:
        result["gate"]   = "REJECT"
        result["reason"] = "Không tìm được ngày tương đồng"
        return result

    meta          = analogs[0].get("_meta", {})
    n_total       = meta.get("total_matches",   len(analogs))
    n_independent = meta.get("independent_n",   len(analogs))
    result["n_total"]       = n_total
    result["n_independent"] = n_independent

    # Layer 2
    stats = compute_base_stats(analogs)
    result["stats"] = stats
    if not stats["valid"]:
        result["gate"]   = "REJECT"
        result["reason"] = "Không đủ dữ liệu thống kê"
        return result

    flags = sanity_check(stats)
    result["flags"] = flags

    # Layer 3: truyền n_independent để gate mẫu nhỏ check đúng
    score, penalties, risk_tier = compute_reference_score(
        stats, flags, all_stats, n_override=n_independent
    )
    result["score"]     = score
    result["penalties"] = penalties
    result["risk_tier"] = risk_tier

    return result
