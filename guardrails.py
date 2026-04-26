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
WINRATE_FLOOR         = 0.50   # dưới này cap score ở 4.0
MAXDD_FLAG_THRESH     = -15.0  # % — flag "Rủi ro cao", giảm score × 0.8
MAXDD_EXCLUDE_THRESH  = -25.0  # % — loại top5, đưa vào mục riêng
SHARPE_FLOOR          = 0.30   # dưới này score × 0.8
SMALL_SAMPLE_PENALTY  = 0.70   # nhân hệ số nếu sample < MIN_SAMPLE_FOR_RANK
SCORE_MAX             = 8.5    # ceiling — không bao giờ = 10
SCORE_CAP_LOW_WR      = 4.0    # cap khi WinRate < 50%

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
    sharpe     = (median_ret / std_ret) if std_ret > 0 else 0.0
    median_mdd = float(np.median(mdds)) if mdds else 0.0

    # P25 / P75 để hiển thị confidence interval
    p25 = float(np.percentile(f30s, 25)) if n >= 4 else None
    p75 = float(np.percentile(f30s, 75)) if n >= 4 else None

    # Recency: % mẫu trong 12 tháng gần nhất
    from datetime import datetime, timedelta
    cutoff_12m = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    n_recent   = sum(1 for a in analogs if a.get("date", "") >= cutoff_12m)
    recency_ratio = n_recent / len(analogs) if analogs else 0

    return {
        "valid":          True,
        "n":              n,
        "median_ret":     round(median_ret, 2),
        "std_ret":        round(std_ret, 2),
        "win_rate":       round(win_rate, 4),
        "sharpe":         round(sharpe, 3),
        "median_mdd":     round(median_mdd, 2),
        "p25_ret":        round(p25, 2) if p25 is not None else None,
        "p75_ret":        round(p75, 2) if p75 is not None else None,
        "recency_ratio":  round(recency_ratio, 3),
        "close":          close,
        # Consistency: 30D vs 60D vs 90D
        "median_60":      round(float(np.median(
                              [a["fwd_60"] for a in analogs if a.get("fwd_60") is not None]
                          )), 2) if any(a.get("fwd_60") for a in analogs) else None,
        "median_90":      round(float(np.median(
                              [a["fwd_90"] for a in analogs if a.get("fwd_90") is not None]
                          )), 2) if any(a.get("fwd_90") for a in analogs) else None,
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
    stats:      dict,
    flags:      list[str],
    all_stats:  list[dict],   # toàn bộ mã để z-score
) -> tuple[float, list[str], str]:
    """
    Tính Reference Score (0–8.5) với các penalties.

    Returns:
        (score, penalty_notes, risk_tier)
        risk_tier: "NORMAL" | "HIGH_RISK" | "EXCLUDED"
    """
    if not stats.get("valid"):
        return 0.0, ["Không đủ dữ liệu thống kê"], "EXCLUDED"

    n          = stats["n"]
    win_rate   = stats["win_rate"]
    median_ret = stats["median_ret"]
    sharpe     = stats["sharpe"]
    median_mdd = stats["median_mdd"]
    penalties  = []
    risk_tier  = "NORMAL"

    # ── Z-score normalization trên toàn bộ mã ─────────────────────────
    def _zscore_field(field: str, val: float, invert: bool = False) -> float:
        """Tính z-score của val trong toàn bộ all_stats."""
        vals = [s[field] for s in all_stats if s.get("valid") and s.get(field) is not None]
        if len(vals) < 2:
            return 0.0
        arr  = np.array(vals, dtype=float)
        std  = float(np.std(arr, ddof=1))
        mean = float(np.mean(arr))
        if std == 0:
            return 0.0
        z = (val - mean) / std
        return float(np.clip(-z if invert else z, -3, 3))

    z_ret   = _zscore_field("median_ret", median_ret)
    z_sh    = _zscore_field("sharpe",     sharpe)
    z_mdd   = _zscore_field("median_mdd", median_mdd, invert=True)  # mdd âm → invert

    # ── Composite formula ──────────────────────────────────────────────
    raw_score = (
        win_rate          * 0.35 +
        z_ret             * 0.25 +
        z_sh              * 0.25 -
        z_mdd             * 0.15
    )
    # Scale về 0–10 (z-score range ~[-3,3] → raw ~ [-1.35, 1.35])
    score = float(np.clip((raw_score + 1.35) / 2.70 * 10, 0, 10))

    # ── Penalties theo thứ tự ─────────────────────────────────────────

    # Rule 1: WinRate < 50% → cap
    if win_rate < WINRATE_FLOOR:
        score = min(score, SCORE_CAP_LOW_WR)
        penalties.append(f"Cap {SCORE_CAP_LOW_WR}/10 do WinRate {win_rate:.0%} < 50%")

    # Rule 2: MaxDD
    if median_mdd < MAXDD_EXCLUDE_THRESH:
        risk_tier = "EXCLUDED"
        penalties.append(
            f"MaxDD median {median_mdd:.1f}% < {MAXDD_EXCLUDE_THRESH}% "
            f"— loại khỏi Top 5, đưa vào mục 'Rủi ro cao'"
        )
    elif median_mdd < MAXDD_FLAG_THRESH:
        score     *= 0.8
        risk_tier  = "HIGH_RISK"
        penalties.append(
            f"MaxDD median {median_mdd:.1f}% → Rủi ro cao, score × 0.8"
        )

    # Rule 3: Sharpe thấp
    if sharpe < SHARPE_FLOOR and risk_tier == "NORMAL":
        score *= 0.8
        penalties.append(f"Sharpe {sharpe:.2f} < {SHARPE_FLOOR} → score × 0.8")

    # Rule 4: Mẫu nhỏ
    if n < MIN_SAMPLE_FOR_RANK:
        score *= SMALL_SAMPLE_PENALTY
        penalties.append(
            f"Mẫu nhỏ ({n} ngày < {MIN_SAMPLE_FOR_RANK}) → score × {SMALL_SAMPLE_PENALTY}"
        )

    # Outlier risk từ Layer 2
    if any("OUTLIER_RISK" in f for f in flags):
        score *= 0.6
        penalties.append("Outlier risk → score × 0.6")

    # High dispersion → Sharpe = 0 trong score
    if any("HIGH_DISPERSION" in f for f in flags):
        # Tính lại không có z_sh
        raw2  = win_rate * 0.35 + z_ret * 0.25 - z_mdd * 0.15
        score = float(np.clip((raw2 + 1.35) / 2.70 * 10, 0, 10))
        penalties.append("High dispersion → Sharpe = 0 trong công thức")

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
    rank:       int,
    symbol:     str,
    stats:      dict,
    score:      float,
    flags:      list[str],
    penalties:  list[str],
    risk_tier:  str,
    n_total:    int,     # tổng số ngày tương đồng (kể cả mốc khác)
) -> str:
    """Format một mã trong output top 5."""
    win_rate   = stats["win_rate"]
    median_ret = stats["median_ret"]
    median_mdd = stats["median_mdd"]
    sharpe     = stats["sharpe"]
    p25        = stats.get("p25_ret")
    p75        = stats.get("p75_ret")
    close      = stats.get("close", 0)
    n          = stats["n"]

    em = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"

    lines = [f"{em} #{rank} — {symbol} (Giá: {close:,.0f})"]

    # Sample info
    sample_note = f" ⚠️ Mẫu nhỏ" if n < MIN_SAMPLE_FOR_RANK else ""
    lines.append(
        f"📊 Mẫu: {n} ngày (tổng {n_total} khớp) | "
        f"Tỷ lệ thắng 30D: {win_rate:.0%} | Tỷ lệ thua: {1-win_rate:.0%}"
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
        f"📈 Sharpe: {sharpe:.2f} | "
        f"Reference Score: {score:.1f}/10 *"
    )

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
                rank      = i,
                symbol    = r["symbol"],
                stats     = r["stats"],
                score     = r["score"],
                flags     = r["flags"],
                penalties = r["penalties"],
                risk_tier = r["risk_tier"],
                n_total   = r.get("n_total", r["stats"]["n"]),
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
            s = r["stats"]
            parts.append(
                f"  {r['symbol']}: Score {r['score']:.1f} | "
                f"WR {s['win_rate']:.0%} | "
                f"Median return {s['median_ret']:+.1f}% | "
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
        "symbol":    symbol,
        "gate":      "PASS",
        "stats":     {},
        "flags":     [],
        "score":     0.0,
        "penalties": [],
        "risk_tier": "EXCLUDED",
        "n_total":   0,
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

    result["n_total"] = analogs[0].get("_meta", {}).get("total_matches", len(analogs))

    # Layer 2
    stats = compute_base_stats(analogs)
    result["stats"] = stats
    if not stats["valid"]:
        result["gate"]   = "REJECT"
        result["reason"] = "Không đủ dữ liệu thống kê"
        return result

    flags = sanity_check(stats)
    result["flags"] = flags

    # Layer 3
    score, penalties, risk_tier = compute_reference_score(stats, flags, all_stats)
    result["score"]     = score
    result["penalties"] = penalties
    result["risk_tier"] = risk_tier

    return result
