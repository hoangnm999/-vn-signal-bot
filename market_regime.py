"""
market_regime.py — Market Regime Detection (Phase 1)

Phát hiện trạng thái thị trường VNINDEX theo 4 regime:
  R1 — Bull Quiet:    trend tăng + volatility thấp  → tín hiệu mua tin cậy nhất
  R2 — Bull Volatile: trend tăng + volatility cao   → momentum, nhưng rủi ro gap
  R3 — Bear Quiet:    trend giảm + volatility thấp  → tích lũy đáy, mean-reversion
  R4 — Bear Volatile: trend giảm + volatility cao   → cash is king, tránh mọi signal

Approach: Rule-based threshold (transparent, không cần ML dependency)
  trend  = slope tuyến tính của SMA20 trong TREND_WINDOW ngày gần nhất
  vol    = realized volatility VOL_WINDOW ngày (std of log returns, annualized)
  threshold vol = adaptive median lịch sử (không hardcode)

Weight điều chỉnh cho wave/analog signal theo regime:
  R1: wave_weight=1.0, analog_weight=1.0  → tin cậy đầy đủ
  R2: wave_weight=0.7, analog_weight=0.8  → giảm nhẹ do noise cao
  R3: wave_weight=0.8, analog_weight=0.7  → wave còn giá trị, analog cẩn thận
  R4: wave_weight=0.3, analog_weight=0.3  → không tin signal nào, ưu tiên cash

Commands:
  /regime              — xem regime VNINDEX hiện tại
  /regime --history    — xem lịch sử regime 90 ngày gần nhất

Cache:
  data/VNINDEX_regime.json — rebuild mỗi ngày (market open)
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR             = pathlib.Path("data")
REGIME_CACHE_FILE    = DATA_DIR / "VNINDEX_regime.json"
CACHE_MAX_AGE_HOURS  = 6          # rebuild nếu cache > 6 giờ (intraday)

TREND_WINDOW         = 20         # ngày để tính slope SMA20
VOL_WINDOW           = 20         # ngày để tính realized vol
VOL_HISTORY_WINDOW   = 252        # ngày lịch sử để tính median vol (adaptive threshold)
LOAD_DAYS            = 600        # ~2.5 năm cho VNINDEX
MIN_BARS_REQUIRED    = 60         # tối thiểu để tính regime

# Trend threshold: slope SMA20 / SMA20_mean — normalize để so sánh qua thời gian
# +0.0005/ngày ≈ +0.05%/ngày ≈ +1%/tháng → Bull territory
# -0.0005/ngày → Bear territory
# Vùng (-0.0005, +0.0005) → Neutral (chỉ dùng trong 3-regime, không dùng ở 4-regime)
TREND_BULL_THRESH    = +0.0005    # slope > này → Bull
TREND_BEAR_THRESH    = -0.0005    # slope < này → Bear

# ── Regime definitions ────────────────────────────────────────────────────────
REGIME_LABELS = {
    1: "R1 — Bull Quiet",
    2: "R2 — Bull Volatile",
    3: "R3 — Bear Quiet",
    4: "R4 — Bear Volatile",
}

REGIME_EMOJI = {
    1: "🟢",
    2: "🟡",
    3: "🟠",
    4: "🔴",
}

REGIME_DESC = {
    1: "Uptrend on dinh — tin hieu mua co do tin cay cao nhat",
    2: "Uptrend bien dong — momentum plays, chu y risk management",
    3: "Downtrend on dinh — tich luy day, mean-reversion co the work",
    4: "Downtrend bien dong — uu tien cash, tranh mo position moi",
}

# Weight điều chỉnh ĐỊNH HƯỚNG cho wave/analog signal theo regime
#
# Triết lý: regime không scale cả 2 chiều xuống như nhau
# mà TĂNG CƯỜNG chiều phù hợp với regime, GIẢM chiều ngược lại
#
# Format: {
#   "bull": float,   # nhân vào score_up   (Bull signal)
#   "bear": float,   # nhân vào score_down (Bear signal)
#   "engine": float, # overall engine weight (cho /check)
# }
#
# R1 Bull Quiet:    up ×1.2, down ×0.8  → Bull signal được khuếch đại
# R2 Bull Volatile: up ×1.1, down ×0.9  → Bull nhẹ hơn vì vol cao = rủi ro đảo chiều
# R3 Bear Quiet:    up ×0.8, down ×1.2  → Bear signal được khuếch đại
# R4 Bear Volatile: up ×0.6, down ×1.4  → Bear rất mạnh, Bull bị giảm mạnh
#
# Giới hạn: nhân tối đa 1.4 để tránh score vượt 100% sau adjust
REGIME_WEIGHTS = {
    1: {"bull": 1.20, "bear": 0.80, "engine": 1.00},
    2: {"bull": 1.10, "bear": 1.00, "engine": 0.85},  # bear neutral — vol cao, không dám giảm score_down
    3: {"bull": 0.80, "bear": 1.20, "engine": 0.75},
    4: {"bull": 0.60, "bear": 1.40, "engine": 0.40},
}

# Lời khuyên position sizing theo regime
REGIME_POSITION_ADVICE = {
    1: "Full size (100%) — thi truong thuan loi",
    2: "Giam size (70%) — bien dong cao, stop loss chat hon",
    3: "Nua size (50%) — chi entry khi co xac nhan ro rang",
    4: "Khong mo moi (0-20%) — bao toan von la uu tien",
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_trend_slope(close: np.ndarray, window: int = TREND_WINDOW) -> float:
    """
    Tính normalized slope của SMA20 trong `window` ngày gần nhất.

    Normalize = slope_raw / SMA20_mean → loại bỏ ảnh hưởng của mức giá tuyệt đối
    (VNINDEX 1200 vs 600 không ảnh hưởng threshold)

    Returns float (normalized slope per day):
      > 0: uptrend
      < 0: downtrend
      magnitude: ~0.001/ngày = ~0.1%/ngày = ~2%/tháng
    """
    if len(close) < window + 20:
        return 0.0

    # Tính SMA20 cho toàn bộ series
    sma20 = pd.Series(close).rolling(20).mean().values

    # Lấy window ngày gần nhất của SMA20 (bỏ NaN đầu)
    sma_window = sma20[-window:]
    valid_mask = ~np.isnan(sma_window)
    if valid_mask.sum() < window // 2:
        return 0.0

    sma_valid = sma_window[valid_mask]
    x = np.arange(len(sma_valid), dtype=float)

    # Linear regression slope
    slope_raw = float(np.polyfit(x, sma_valid, 1)[0])

    # Normalize bởi mean SMA20 để so sánh được qua thời gian
    mean_price = float(np.nanmean(sma_valid))
    if mean_price < 1e-6:
        return 0.0

    return round(slope_raw / mean_price, 6)


def _compute_realized_vol(close: np.ndarray, window: int = VOL_WINDOW) -> float:
    """
    Tính realized volatility (annualized) trong `window` ngày gần nhất.

    Vol = std(log_returns) * sqrt(252)
    Dùng log returns để đối xứng và scale-invariant.

    Returns float (annualized vol, e.g. 0.20 = 20% per year):
    """
    if len(close) < window + 2:
        return 0.0

    prices = close[-(window + 1):]
    log_ret = np.log(prices[1:] / prices[:-1])
    vol = float(np.std(log_ret, ddof=1)) * np.sqrt(252)
    return round(vol, 4)


def _compute_vol_history(close: np.ndarray, window: int = VOL_WINDOW,
                          history: int = VOL_HISTORY_WINDOW) -> np.ndarray:
    """
    Tính rolling realized vol cho `history` ngày — dùng để lấy median (adaptive threshold).

    Returns array of vol values (length ≈ history).
    """
    if len(close) < window + history:
        # Fallback: tính vol toàn bộ series
        log_ret = np.log(close[1:] / close[:-1])
        return np.array([float(np.std(log_ret, ddof=1)) * np.sqrt(252)])

    vols = []
    for i in range(history):
        end_idx = len(close) - i
        if end_idx < window + 1:
            break
        prices  = close[end_idx - window - 1: end_idx]
        log_ret = np.log(prices[1:] / prices[:-1])
        vols.append(float(np.std(log_ret, ddof=1)) * np.sqrt(252))

    return np.array(vols[::-1])  # chronological order


def _classify_regime(
    trend_slope: float,
    current_vol: float,
    vol_median:  float,
) -> int:
    """
    Phân loại regime dựa trên trend và volatility.

    Logic:
      Bull = trend_slope > TREND_BULL_THRESH
      Bear = trend_slope < TREND_BEAR_THRESH
      Neutral (slope giữa 2 threshold) → gán theo momentum gần nhất
        (tức là giữ regime Bull/Bear, không có R_Neutral ở 4-regime model)

      Quiet    = current_vol <= vol_median
      Volatile = current_vol > vol_median

    Returns: 1, 2, 3, hoặc 4
    """
    is_bull    = trend_slope >= TREND_BULL_THRESH
    is_bear    = trend_slope <= TREND_BEAR_THRESH
    is_quiet   = current_vol <= vol_median

    # Vùng neutral: gán Bear (conservative) khi không rõ ràng
    if not is_bull and not is_bear:
        is_bull = False

    if is_bull and is_quiet:     return 1   # Bull Quiet
    if is_bull and not is_quiet: return 2   # Bull Volatile
    if not is_bull and is_quiet: return 3   # Bear Quiet
    return 4                                 # Bear Volatile


def compute_regime(df: pd.DataFrame) -> dict:
    """
    Tính regime hiện tại và các metrics liên quan từ DataFrame OHLCV.

    Returns dict:
    {
        "regime":       int (1-4),
        "label":        str,
        "emoji":        str,
        "desc":         str,
        "trend_slope":  float,   # normalized slope/day
        "current_vol":  float,   # annualized realized vol
        "vol_median":   float,   # adaptive threshold
        "vol_pct":      float,   # current_vol / vol_median — >1 là Volatile
        "trend_pct":    float,   # trend_slope / TREND_BULL_THRESH — >1 là Bull rõ
        "weights":      dict,    # {"wave": float, "analog": float, "engine": float}
        "position_advice": str,
        "is_bull":      bool,
        "is_quiet":     bool,
        # Lịch sử 90 ngày
        "history_90d":  list of {"date": str, "regime": int, "trend": float, "vol": float}
    }
    """
    close  = df["close"].values.astype(float)
    dates  = df.index if hasattr(df.index, 'strftime') else pd.RangeIndex(len(df))

    trend_slope = _compute_trend_slope(close)
    current_vol = _compute_realized_vol(close)
    vol_history = _compute_vol_history(close)
    vol_median  = float(np.median(vol_history)) if len(vol_history) > 0 else current_vol

    regime = _classify_regime(trend_slope, current_vol, vol_median)

    # Vol percentage vs median — dùng trong report
    vol_pct   = round(current_vol / vol_median, 2) if vol_median > 1e-6 else 1.0
    # Trend percentage vs threshold — hiểu độ mạnh của trend
    trend_ref = abs(TREND_BULL_THRESH)
    trend_pct = round(trend_slope / trend_ref, 1) if trend_ref > 1e-9 else 0.0

    # Lịch sử 90 ngày — tính regime cho mỗi ngày
    history_90d = _compute_regime_history(close, dates, days=90)

    return {
        "regime":          regime,
        "label":           REGIME_LABELS[regime],
        "emoji":           REGIME_EMOJI[regime],
        "desc":            REGIME_DESC[regime],
        "trend_slope":     round(trend_slope, 6),
        "current_vol":     round(current_vol, 4),
        "vol_median":      round(vol_median, 4),
        "vol_pct":         vol_pct,
        "trend_pct":       trend_pct,
        "weights":         REGIME_WEIGHTS[regime],
        "position_advice": REGIME_POSITION_ADVICE[regime],
        "is_bull":         trend_slope >= TREND_BULL_THRESH,
        "is_quiet":        current_vol <= vol_median,
        "history_90d":     history_90d,
    }


def _compute_regime_history(
    close: np.ndarray,
    dates,
    days:  int = 90,
) -> list[dict]:
    """
    Tính regime cho mỗi ngày trong `days` ngày gần nhất.
    Dùng rolling window — mỗi ngày chỉ dùng data đến ngày đó (no lookahead).

    Returns list of dicts sorted chronologically.
    """
    n       = len(close)
    results = []
    min_idx = max(TREND_WINDOW + 20 + 1, n - days)

    # Tính vol_median một lần dùng toàn bộ lịch sử (không lookahead về vol threshold)
    vol_history_all = _compute_vol_history(close)
    vol_median_global = float(np.median(vol_history_all)) if len(vol_history_all) > 0 else 0.20

    for i in range(min_idx, n):
        close_slice = close[:i + 1]
        trend  = _compute_trend_slope(close_slice)
        vol    = _compute_realized_vol(close_slice)
        regime = _classify_regime(trend, vol, vol_median_global)

        try:
            date_str = dates[i].strftime("%Y-%m-%d") if hasattr(dates[i], 'strftime') else str(i)
        except Exception:
            date_str = str(i)

        results.append({
            "date":   date_str,
            "regime": regime,
            "trend":  round(trend, 6),
            "vol":    round(vol, 4),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _load_regime_cache() -> Optional[dict]:
    """Load cache nếu còn mới (< CACHE_MAX_AGE_HOURS giờ)."""
    if not REGIME_CACHE_FILE.exists():
        return None
    try:
        with open(REGIME_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        built = datetime.fromisoformat(data.get("built_at", "2000-01-01"))
        age_hours = (datetime.now() - built).total_seconds() / 3600
        if age_hours > CACHE_MAX_AGE_HOURS:
            logger.info(f"Regime cache: {age_hours:.1f}h → rebuild")
            return None
        return data
    except Exception as e:
        logger.warning(f"Regime cache load fail: {e}")
        return None


def _save_regime_cache(data: dict):
    """Lưu cache."""
    try:
        DATA_DIR.mkdir(exist_ok=True)
        with open(REGIME_CACHE_FILE, "w", encoding="utf-8") as f:
            # Không lưu history_90d vào cache (tốn space, tính lại nhanh)
            cache_data = {k: v for k, v in data.items() if k != "history_90d"}
            cache_data["built_at"] = datetime.now().isoformat()
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Regime cache saved: regime={data['regime']}")
    except Exception as e:
        logger.warning(f"Regime cache save fail: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN API
# ══════════════════════════════════════════════════════════════════════════════

def get_market_regime(force_rebuild: bool = False) -> dict:
    """
    Lấy regime VNINDEX hiện tại.

    Ưu tiên cache, rebuild nếu cần.

    Returns dict từ compute_regime() + "ok" bool + "error" str.
    Luôn trả về dict (không raise exception).
    """
    result_err = {
        "ok": False, "error": None,
        "regime": 3, "label": REGIME_LABELS[3],
        "emoji": REGIME_EMOJI[3], "desc": REGIME_DESC[3],
        "trend_slope": 0.0, "current_vol": 0.0,
        "vol_median": 0.0, "vol_pct": 1.0, "trend_pct": 0.0,
        "weights": REGIME_WEIGHTS[3],
        "position_advice": REGIME_POSITION_ADVICE[3],
        "is_bull": False, "is_quiet": True,
        "history_90d": [],
        "built_at": datetime.now().isoformat(),
    }

    # Load cache
    if not force_rebuild:
        cached = _load_regime_cache()
        if cached:
            cached["ok"]         = True
            cached["history_90d"] = []   # không cache history
            cached["weights"]    = REGIME_WEIGHTS.get(cached.get("regime", 3), REGIME_WEIGHTS[3])
            return cached

    # Load VNINDEX data
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv("VNINDEX", days=LOAD_DAYS, min_bars=MIN_BARS_REQUIRED)
    except Exception as e:
        result_err["error"] = f"Khong load duoc VNINDEX: {e}"
        logger.error(result_err["error"])
        return result_err

    if len(df) < MIN_BARS_REQUIRED:
        result_err["error"] = f"VNINDEX chi co {len(df)} bars (can >= {MIN_BARS_REQUIRED})"
        return result_err

    # Compute regime
    try:
        regime_data = compute_regime(df)
    except Exception as e:
        result_err["error"] = f"Loi tinh regime: {e}"
        logger.error(result_err["error"])
        return result_err

    regime_data["ok"]       = True
    regime_data["built_at"] = datetime.now().isoformat()

    _save_regime_cache(regime_data)
    return regime_data


def get_signal_weights(regime: Optional[int] = None) -> dict:
    """
    Lấy directional weights theo regime hiện tại.

    Returns:
        {"bull": float, "bear": float, "engine": float}
        - bull:   nhân vào score_up   (khuếch đại hay giảm nhẹ tín hiệu tăng)
        - bear:   nhân vào score_down (khuếch đại hay giảm nhẹ tín hiệu giảm)
        - engine: overall weight cho 16 engines (/check)
        Fallback về R3 nếu load thất bại (conservative).
    """
    if regime is not None:
        return REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS[3])

    try:
        r = get_market_regime()
        return r.get("weights", REGIME_WEIGHTS[3])
    except Exception:
        return REGIME_WEIGHTS[3]   # conservative fallback


# ══════════════════════════════════════════════════════════════════════════════
# REGIME HISTORY STATS
# ══════════════════════════════════════════════════════════════════════════════

def regime_history_stats(history: list[dict]) -> dict:
    """
    Tổng hợp thống kê từ lịch sử regime.

    Returns:
    {
        "distribution": {1: pct, 2: pct, 3: pct, 4: pct},
        "current_streak": int,          # số ngày liên tiếp ở regime hiện tại
        "current_streak_regime": int,
        "transitions": list of str,     # các lần đổi regime gần nhất
    }
    """
    if not history:
        return {}

    # Distribution
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for h in history:
        r = h.get("regime", 3)
        counts[r] = counts.get(r, 0) + 1
    total = len(history)
    distribution = {r: round(n / total * 100, 1) for r, n in counts.items()}

    # Current streak
    current_regime = history[-1]["regime"]
    streak = 0
    for h in reversed(history):
        if h["regime"] == current_regime:
            streak += 1
        else:
            break

    # Transitions — các lần đổi regime (tối đa 5 lần gần nhất)
    transitions = []
    prev_regime = history[0]["regime"]
    for h in history[1:]:
        if h["regime"] != prev_regime:
            transitions.append(
                f"{h['date']}: R{prev_regime}→R{h['regime']} "
                f"({REGIME_LABELS[prev_regime].split('—')[1].strip()} → "
                f"{REGIME_LABELS[h['regime']].split('—')[1].strip()})"
            )
            prev_regime = h["regime"]
    transitions = transitions[-5:]  # 5 lần gần nhất

    return {
        "distribution":          distribution,
        "current_streak":        streak,
        "current_streak_regime": current_regime,
        "transitions":           transitions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def format_regime_report(regime_data: dict, show_history: bool = False) -> str:
    """
    Format regime report thành text Telegram-friendly.
    """
    if not regime_data.get("ok"):
        return f"❌ Loi lay regime: {regime_data.get('error', '?')}"

    r          = regime_data["regime"]
    label      = regime_data["label"]
    emoji      = regime_data["emoji"]
    desc       = regime_data["desc"]
    trend      = regime_data["trend_slope"]
    vol        = regime_data["current_vol"]
    vol_med    = regime_data["vol_median"]
    vol_pct    = regime_data["vol_pct"]
    trend_pct  = regime_data["trend_pct"]
    weights    = regime_data["weights"]
    pos_advice = regime_data["position_advice"]
    built      = regime_data.get("built_at", "?")[:16].replace("T", " ")
    is_bull    = regime_data["is_bull"]
    is_quiet   = regime_data["is_quiet"]

    lines = []

    # ── Header ────────────────────────────────────────────────────────────
    lines += [
        "VNINDEX MARKET REGIME",
        "═" * 40,
        f"{emoji} {label}",
        f"   {desc}",
        f"   Cap nhat: {built}",
        "",
    ]

    # ── Metrics ───────────────────────────────────────────────────────────
    # Trend bar: -3x đến +3x threshold
    def _trend_bar(tp: float) -> str:
        # tp = trend_pct, range [-3, +3], bar 11 chars với │ ở giữa
        steps = round(max(-5, min(5, tp)))
        if steps >= 0:
            right = "█" * steps + "░" * (5 - steps)
            left  = "░" * 5
        else:
            left  = "░" * (5 + steps) + "▒" * (-steps)
            right = "░" * 5
        return f"[{left}│{right}]"

    def _vol_bar(vp: float) -> str:
        # vp = vol/median, range [0, 2+], bar 10 chars
        filled = min(10, round(vp * 5))
        return "[" + "█" * filled + "░" * (10 - filled) + "]"

    trend_dir  = "↑ Uptrend" if is_bull else "↓ Downtrend"
    trend_str  = f"{trend_dir} ({trend:+.4f}/ngay, {trend_pct:+.1f}x threshold)"
    vol_status = "Quiet" if is_quiet else "Volatile"
    vol_str    = f"{vol_status} ({vol:.1%} vs median {vol_med:.1%})"

    lines += [
        "CHI SO HIEN TAI:",
        "─" * 40,
        f"  Trend : {_trend_bar(trend_pct)}  {trend_str}",
        f"  Vol   : {_vol_bar(vol_pct)}  {vol_str}",
        "",
    ]

    # ── Weight điều chỉnh ─────────────────────────────────────────────────
    def _w_bar(w: float) -> str:
        filled = round(w * 5)
        return "[" + "█" * filled + "░" * (5 - filled) + "]"

    lines += [
        "DIEU CHINH WEIGHT SIGNAL (directional):",
        "─" * 40,
        f"  Tin hieu TANG (score_up)  : {_w_bar(weights['bull'])} x{weights['bull']:.2f}",
        f"  Tin hieu GIAM (score_down): {_w_bar(weights['bear'])} x{weights['bear']:.2f}",
        f"  16 Engines overall        : {_w_bar(weights['engine'])} x{weights['engine']:.2f}",
        "  * Bull regime: khuech dai score_up, giam score_down",
        "  * Bear regime: khuech dai score_down, giam score_up",
        "",
        f"📊 Position sizing: {pos_advice}",
        "",
    ]

    # ── Lịch sử nếu có ───────────────────────────────────────────────────
    history = regime_data.get("history_90d", [])
    if history:
        stats = regime_history_stats(history)
        dist  = stats.get("distribution", {})
        streak = stats.get("current_streak", 0)
        transitions = stats.get("transitions", [])

        lines += [
            "LICH SU 90 NGAY:",
            "─" * 40,
        ]
        for ri in [1, 2, 3, 4]:
            pct   = dist.get(ri, 0.0)
            rlbl  = REGIME_LABELS[ri].split("—")[1].strip()
            bar   = "[" + "█" * round(pct / 10) + "░" * (10 - round(pct / 10)) + "]"
            lines.append(f"  R{ri} {rlbl:<14}: {bar} {pct:.0f}%")

        lines += [
            "",
            f"  Streak hien tai: {streak} ngay lien tiep o {REGIME_LABELS[r]}",
            "",
        ]

        if transitions:
            lines.append("  Chuyen doi regime gan day:")
            for t in transitions[-3:]:
                lines.append(f"    • {t}")
            lines.append("")

        # Mini chart 30 ngày
        if show_history and len(history) >= 10:
            lines.append("  Chart 30 ngay (moi o = 1 ngay):")
            recent = history[-30:]
            regime_char = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
            chart_line = "  " + "".join(regime_char.get(h["regime"], "⚪") for h in recent)
            lines.append(chart_line)
            lines.append("  (trai=cu, phai=moi nhat)")
            lines.append("")

    # ── Hướng dẫn đọc ─────────────────────────────────────────────────────
    lines += [
        "─" * 40,
        "HUONG DAN DOC WEIGHT:",
        "  Wave/Analog weight: nhan voi score goc de ra score dieu chinh",
        "  Vi du: Wave score 51% x 0.30 (R4) = 15.3% effective score",
        "  → Signal rat yeu khi thi truong dang Bear Volatile",
        "─" * 40,
        "⚠️  Regime chi la context — khong phai lenh mua/ban.",
        "   Ket hop voi /check va /wave de ra quyet dinh.",
    ]

    return "\n".join(lines)


def format_regime_inline(regime_data: dict) -> str:
    """
    Format ngắn gọn để nhúng vào /check output (1-2 dòng).

    Ví dụ: "🟢 R1 Bull Quiet (wave×1.0 / analog×1.0)"
    """
    if not regime_data.get("ok"):
        return ""
    r       = regime_data["regime"]
    emoji   = regime_data["emoji"]
    label   = regime_data["label"].split("—")[1].strip()
    weights = regime_data["weights"]
    return (
        f"{emoji} Regime: R{r} {label} "
        f"| Bull×{weights['bull']:.2f} Bear×{weights['bear']:.2f}"
    )


def apply_regime_weight(
    score_up:    float,
    score_down:  float,
    regime_data: Optional[dict] = None,
) -> tuple[float, float, str, str]:
    """
    Áp dụng directional regime weight vào cặp score (up, down).

    Logic:
      - Regime Bull → khuếch đại score_up, giảm score_down
      - Regime Bear → khuếch đại score_down, giảm score_up
      → Verdict sau điều chỉnh rõ ràng hơn, không phải scale đều cả 2

    Args:
        score_up:    score gốc chiều tăng (0-1)
        score_down:  score gốc chiều giảm (0-1)
        regime_data: dict từ get_market_regime(), nếu None thì tự load

    Returns:
        (adj_up, adj_down, note_up, note_down)
        adj_*: score đã điều chỉnh, clip tại [0, 1]
        note_*: mô tả điều chỉnh
    """
    if regime_data is None:
        try:
            regime_data = get_market_regime()
        except Exception:
            return score_up, score_down, "", ""

    if not regime_data.get("ok"):
        return score_up, score_down, "", ""

    weights    = regime_data.get("weights", REGIME_WEIGHTS[3])
    w_bull     = weights.get("bull", 1.0)
    w_bear     = weights.get("bear", 1.0)
    r          = regime_data.get("regime", 3)
    label      = REGIME_LABELS[r].split("—")[1].strip()

    adj_up   = round(min(1.0, score_up   * w_bull), 3)
    adj_down = round(min(1.0, score_down * w_bear), 3)

    note_up   = f"×{w_bull:.2f} (R{r} {label})"
    note_down = f"×{w_bear:.2f} (R{r} {label})"

    return adj_up, adj_down, note_up, note_down


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM COMMAND HANDLER
# ══════════════════════════════════════════════════════════════════════════════

async def regime_cmd(update, context):
    """
    /regime [--history] [--rebuild]

    Hiển thị Market Regime VNINDEX hiện tại.
    --history: thêm mini chart 30 ngày và transition history
    --rebuild: bỏ cache, tính lại
    """
    try:
        from bot import is_allowed, _deny
    except ImportError:
        def is_allowed(_): return True
        async def _deny(_): pass

    if not is_allowed(update):
        await _deny(update)
        return

    args          = context.args or []
    show_history  = "--history" in args
    force_rebuild = "--rebuild" in args

    msg = await update.message.reply_text(
        "📊 Dang phan tich Market Regime VNINDEX..."
    )

    import asyncio
    try:
        regime_data = await asyncio.to_thread(get_market_regime, force_rebuild)
    except Exception as e:
        import traceback
        logger.error(f"regime_cmd: {e}\n{traceback.format_exc()}")
        await msg.edit_text(f"❌ Loi /regime: {str(e)[:200]}")
        return

    # Load history nếu cần (không cache để tiết kiệm)
    if show_history and regime_data.get("ok"):
        try:
            from vn_loader import load_vn_ohlcv
            df = await asyncio.to_thread(
                load_vn_ohlcv, "VNINDEX", LOAD_DAYS, MIN_BARS_REQUIRED
            )
            full_data = await asyncio.to_thread(compute_regime, df)
            regime_data["history_90d"] = full_data.get("history_90d", [])
        except Exception as e:
            logger.warning(f"regime history load fail: {e}")

    report = format_regime_report(regime_data, show_history=show_history)

    if len(report) <= 4096:
        try:
            await msg.edit_text(report)
        except Exception:
            await update.message.reply_text(report[:4096])
    else:
        split_at = report.rfind("\n\n", 0, 4000)
        if split_at < 0:
            split_at = 4000
        try:
            await msg.edit_text(report[:split_at].strip())
        except Exception:
            await update.message.reply_text(report[:4096])
        if report[split_at:].strip():
            await update.message.reply_text(report[split_at:].strip()[:4096])
