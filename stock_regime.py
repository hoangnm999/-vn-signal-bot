"""
stock_regime.py — Stock Regime Detection (Phase 2)

Phát hiện trạng thái của từng cổ phiếu riêng lẻ bằng GMM clustering
trên 15-dim state vector. Độc lập với Market Regime (Phase 1 / VNINDEX level).

4 Regime (Wyckoff-inspired):
  SR1 — Accumulation : giá sideway đáy, volume tăng, smart money mua vào
  SR2 — Markup       : trend tăng rõ, momentum mạnh, breakout khỏi tích lũy
  SR3 — Distribution : giá cao, volume tăng nhưng momentum yếu dần, smart money thoát
  SR4 — Markdown     : trend giảm rõ, momentum âm, volume xác nhận bán

Approach:
  - Online fit per-symbol: GMM fit trên chính lịch sử của mã đó (rolling ~500 bars)
  - Label assignment: dựa trên đặc tính centroid (kinh tế học), không phụ thuộc
    số thứ tự cluster → nhất quán giữa các mã
  - Confidence score: GMM probability → trader biết "Markup rõ" vs "biên giới"
  - Cache 24h per symbol → chỉ chậm lần đầu

Label assignment logic (centroid-based):
  momentum_score = momentum_5d + momentum_20d (dims 10, 11)
  volume_score   = volume_spike (dim 3)
  trend_score    = trend_slope + price_vs_sma20 (dims 4, 5)

  Markup:       momentum_score cao nhất
  Markdown:     momentum_score thấp nhất (âm)
  Accumulation: momentum trung bình thấp + volume spike cao (tích lũy)
  Distribution: momentum trung bình cao + trend yếu dần (đỉnh)
  → Nếu không phân biệt được rõ: dùng fallback ranking

Integration:
  /check    → format_stock_regime_inline()   — 1 dòng
  /wave     → get_stock_regime()             — dict đầy đủ
  /backtest → get_stock_regime()             — dict đầy đủ

Cache:
  data/stock_regime_{SYMBOL}.json — TTL 24h
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR              = pathlib.Path("data")
CACHE_TTL_HOURS       = 24          # cache per symbol
N_CLUSTERS            = 4           # Accumulation / Markup / Distribution / Markdown
MIN_BARS_FOR_GMM      = 120         # tối thiểu để fit GMM có ý nghĩa
PREFERRED_BARS        = 500         # dùng 500 bars lịch sử để fit
GMM_COVARIANCE_TYPE   = "full"      # full covariance → capture correlation giữa dims
GMM_MAX_ITER          = 200
GMM_N_INIT            = 5           # 5 random init → chọn best log-likelihood
GMM_REG_COVAR         = 1e-4        # regularization tránh singular matrix

# Index của các dimensions quan trọng trong VECTOR_DIMS (state_vector.py)
# VECTOR_DIMS = ["rsi_norm","macd_hist_norm","bb_position","volume_spike",
#                "trend_slope","price_vs_sma20","price_vs_sma50","atr_ratio",
#                "stoch_k_norm","ema_cross","momentum_5d","momentum_20d",
#                "high_low_pos","vol_trend","candle_body"]
_IDX_MOMENTUM_5D  = 10
_IDX_MOMENTUM_20D = 11
_IDX_VOLUME_SPIKE = 3
_IDX_TREND_SLOPE  = 4
_IDX_PRICE_SMA20  = 5
_IDX_RSI          = 0
_IDX_BB_POS       = 2

# ── Regime definitions ─────────────────────────────────────────────────────────
SR_LABELS = {
    1: "SR1 — Accumulation",
    2: "SR2 — Markup",
    3: "SR3 — Distribution",
    4: "SR4 — Markdown",
}

SR_EMOJI = {
    1: "🔵",   # tích lũy — trung tính/chuẩn bị
    2: "🟢",   # markup   — tăng
    3: "🟡",   # distribution — cảnh báo
    4: "🔴",   # markdown — giảm
}

SR_DESC = {
    1: "Tich luy: gia sideway day, volume tang dan, co hoi mua khi co xac nhan",
    2: "Markup: xu huong tang ro rang, momentum manh — setup mua co xac suat cao",
    3: "Phan phoi: gia cao, momentum yeu dan — giam position, dat SL chat",
    4: "Markdown: xu huong giam ro, tranh mua — cho Accumulation hinh thanh",
}

SR_ACTION = {
    1: "Theo doi, dat alert khi breakout volume",
    2: "Co the mua / giu — xac nhan voi /check va /wave",
    3: "Giam position, SL chat, khong mo moi",
    4: "Tranh mua, co the short neu co setup",
}

# Confidence thresholds
CONF_HIGH   = 0.75   # >= 75%: regime rõ ràng
CONF_MEDIUM = 0.50   # 50-75%: khá rõ
# < 50%: đang ở biên giới, không chắc


# ══════════════════════════════════════════════════════════════════════════════
# CORE: GMM FIT + LABEL ASSIGNMENT
# ══════════════════════════════════════════════════════════════════════════════

def _assign_labels(centroids: np.ndarray) -> dict[int, int]:
    """
    Gán label kinh tế cho từng cluster dựa trên đặc tính centroid.
    Đảm bảo nhất quán bất kể thứ tự cluster GMM.

    Args:
        centroids: shape (N_CLUSTERS, 15) — mean vector của mỗi cluster

    Returns:
        dict {cluster_idx: stock_regime (1-4)}

    Logic:
        1. Tính momentum_score = momentum_5d + momentum_20d cho mỗi centroid
        2. Tính volume_score   = volume_spike
        3. Rank theo momentum:
           - Cao nhất  → SR2 Markup
           - Thấp nhất → SR4 Markdown
        4. Trong 2 centroid còn lại:
           - Volume cao hơn → SR1 Accumulation (tích lũy + volume tăng)
           - Volume thấp    → SR3 Distribution  (đỉnh + volume giảm dần)
    """
    n = len(centroids)
    momentum_scores = (centroids[:, _IDX_MOMENTUM_5D] +
                       centroids[:, _IDX_MOMENTUM_20D])
    volume_scores   = centroids[:, _IDX_VOLUME_SPIKE]

    # Sort by momentum
    ranked = np.argsort(momentum_scores)   # ascending: [worst, ..., best]

    mapping = {}
    # Markup = highest momentum
    mapping[int(ranked[-1])] = 2   # SR2
    # Markdown = lowest momentum
    mapping[int(ranked[0])]  = 4   # SR4

    # Middle 2: distinguish by volume
    middle = [int(ranked[1]), int(ranked[2])]
    vol_middle = [(i, volume_scores[i]) for i in middle]
    vol_middle.sort(key=lambda x: x[1], reverse=True)

    # Higher volume in middle → Accumulation (vol spike + low momentum)
    mapping[vol_middle[0][0]] = 1   # SR1 Accumulation
    # Lower volume in middle → Distribution
    mapping[vol_middle[1][0]] = 3   # SR3 Distribution

    return mapping


def _fit_gmm(vectors: np.ndarray) -> Optional[tuple]:
    """
    Fit GMM trên ma trận vectors.

    Returns:
        (gmm_model, label_mapping, centroids) hoặc None nếu fail
    """
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("scikit-learn chưa được cài. pip install scikit-learn")
        return None

    if len(vectors) < MIN_BARS_FOR_GMM:
        logger.warning(f"Không đủ bars cho GMM: {len(vectors)} < {MIN_BARS_FOR_GMM}")
        return None

    # StandardScaler để normalize — GMM nhạy cảm với scale
    scaler = StandardScaler()
    X = scaler.fit_transform(vectors)

    # Fit GMM
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type=GMM_COVARIANCE_TYPE,
        max_iter=GMM_MAX_ITER,
        n_init=GMM_N_INIT,
        reg_covar=GMM_REG_COVAR,
        random_state=42,
    )
    gmm.fit(X)

    # Centroids trong original space (inverse transform)
    centroids_scaled = gmm.means_          # (4, 15) trong scaled space
    centroids_orig   = scaler.inverse_transform(centroids_scaled)

    # Assign labels
    label_mapping = _assign_labels(centroids_orig)

    return gmm, scaler, label_mapping, centroids_orig


def _predict_current(
    gmm,
    scaler,
    label_mapping: dict,
    current_vector: np.ndarray,
) -> tuple[int, float, np.ndarray]:
    """
    Predict regime cho vector hiện tại.

    Returns:
        (stock_regime: int 1-4,
         confidence: float 0-1,
         all_probs: np.ndarray shape (4,) — prob của mỗi SR)
    """
    x = scaler.transform(current_vector.reshape(1, -1))
    probs_raw = gmm.predict_proba(x)[0]   # (N_CLUSTERS,)

    # Map raw cluster probs → SR probs
    sr_probs = np.zeros(4)
    for cluster_idx, sr_idx in label_mapping.items():
        sr_probs[sr_idx - 1] += probs_raw[cluster_idx]

    # Predict = SR với prob cao nhất
    best_sr = int(np.argmax(sr_probs)) + 1   # 1-indexed
    confidence = float(sr_probs[best_sr - 1])

    return best_sr, confidence, sr_probs


# ══════════════════════════════════════════════════════════════════════════════
# BUILD VECTORS FROM HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def _build_vector_matrix(df: pd.DataFrame, max_bars: int = PREFERRED_BARS) -> Optional[np.ndarray]:
    """
    Tính ma trận state vectors từ lịch sử OHLCV.
    Mỗi hàng = vector của 1 ngày.

    Args:
        df: DataFrame OHLCV, cần >= 120 bars
        max_bars: số bars gần nhất để dùng (tránh quá cũ)

    Returns:
        np.ndarray shape (n_valid, 15) hoặc None
    """
    try:
        from state_vector import compute_state_vector_for_date, VECTOR_DIMS
    except ImportError:
        logger.error("Không import được state_vector.py")
        return None

    # Dùng max_bars bars gần nhất
    df_use = df.iloc[-max_bars:].reset_index(drop=True) if len(df) > max_bars else df.reset_index(drop=True)
    n = len(df_use)

    vectors = []
    for i in range(n):
        if i < 59:   # cần ít nhất 60 bars để tính vector
            continue
        vec = compute_state_vector_for_date(df_use, i)
        if vec is None:
            continue
        arr = vec.get("_array")
        if arr is not None and len(arr) == 15 and np.all(np.isfinite(arr)):
            vectors.append(arr.astype(np.float32))

    if len(vectors) < MIN_BARS_FOR_GMM:
        return None

    return np.array(vectors, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(symbol: str) -> pathlib.Path:
    return DATA_DIR / f"stock_regime_{symbol.upper()}.json"


def _load_cache(symbol: str) -> Optional[dict]:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        built = datetime.fromisoformat(data.get("built_at", "2000-01-01"))
        age_h = (datetime.now() - built).total_seconds() / 3600
        if age_h > CACHE_TTL_HOURS:
            logger.info(f"Stock regime cache {symbol}: {age_h:.1f}h → rebuild")
            return None
        return data
    except Exception as e:
        logger.warning(f"Stock regime cache load {symbol}: {e}")
        return None


def _save_cache(symbol: str, data: dict):
    try:
        DATA_DIR.mkdir(exist_ok=True)
        path = _cache_path(symbol)
        # Không lưu history (lớn), chỉ lưu kết quả hiện tại
        save_data = {k: v for k, v in data.items()
                     if k not in ("history_30d",)}
        save_data["built_at"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Stock regime cache save {symbol}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE STOCK REGIME
# ══════════════════════════════════════════════════════════════════════════════

def _compute_stock_regime(symbol: str, df: pd.DataFrame) -> dict:
    """
    Core compute: fit GMM, predict regime hiện tại, tính history 30 ngày.

    Returns dict đầy đủ hoặc dict với ok=False.
    """
    err_result = {
        "ok": False, "symbol": symbol,
        "regime": 1, "label": SR_LABELS[1],
        "emoji": SR_EMOJI[1], "desc": SR_DESC[1],
        "action": SR_ACTION[1],
        "confidence": 0.0, "confidence_label": "Khong xac dinh",
        "sr_probs": [0.25, 0.25, 0.25, 0.25],
        "n_bars_used": 0, "history_30d": [],
        "error": None,
    }

    # Build vector matrix
    vectors = _build_vector_matrix(df, max_bars=PREFERRED_BARS)
    if vectors is None or len(vectors) < MIN_BARS_FOR_GMM:
        err_result["error"] = (
            f"Khong du vectors: can >={MIN_BARS_FOR_GMM}, "
            f"co {len(vectors) if vectors is not None else 0}"
        )
        return err_result

    # Fit GMM
    fit_result = _fit_gmm(vectors)
    if fit_result is None:
        err_result["error"] = "GMM fit that bai"
        return err_result

    gmm, scaler, label_mapping, centroids = fit_result

    # Predict hiện tại (vector cuối)
    current_vec = vectors[-1]
    sr, conf, sr_probs = _predict_current(gmm, scaler, label_mapping, current_vec)

    # Confidence label
    if conf >= CONF_HIGH:
        conf_label = f"Ro rang ({conf:.0%})"
    elif conf >= CONF_MEDIUM:
        conf_label = f"Kha ro ({conf:.0%})"
    else:
        conf_label = f"Bien gioi ({conf:.0%}) — co the chuyen sang {_border_note(sr_probs, sr)}"

    # History 30 ngày gần nhất
    history_30d = _compute_history(vectors, gmm, scaler, label_mapping, df, days=30)

    # Centroid summary — giúp debug
    centroid_summary = {}
    for c_idx, sr_idx in label_mapping.items():
        c = centroids[c_idx]
        centroid_summary[f"SR{sr_idx}"] = {
            "momentum": round(float(c[_IDX_MOMENTUM_5D] + c[_IDX_MOMENTUM_20D]), 3),
            "volume":   round(float(c[_IDX_VOLUME_SPIKE]), 3),
            "trend":    round(float(c[_IDX_TREND_SLOPE]), 3),
        }

    return {
        "ok":               True,
        "symbol":           symbol,
        "regime":           sr,
        "label":            SR_LABELS[sr],
        "emoji":            SR_EMOJI[sr],
        "desc":             SR_DESC[sr],
        "action":           SR_ACTION[sr],
        "confidence":       round(conf, 3),
        "confidence_label": conf_label,
        "sr_probs":         [round(float(p), 3) for p in sr_probs],
        "n_bars_used":      len(vectors),
        "history_30d":      history_30d,
        "centroid_summary": centroid_summary,
        "error":            None,
    }


def _border_note(sr_probs: np.ndarray, current_sr: int) -> str:
    """Tên regime gần thứ 2 khi confidence thấp."""
    probs = list(enumerate(sr_probs, start=1))
    probs.sort(key=lambda x: x[1], reverse=True)
    for sr_idx, _ in probs:
        if sr_idx != current_sr:
            return SR_LABELS[sr_idx].split("—")[1].strip()
    return ""


def _compute_history(
    vectors: np.ndarray,
    gmm,
    scaler,
    label_mapping: dict,
    df: pd.DataFrame,
    days: int = 30,
) -> list[dict]:
    """
    Tính regime cho 30 ngày gần nhất dùng model đã fit.
    Nhanh vì không refit, chỉ predict.
    """
    history = []
    n = len(vectors)
    start = max(0, n - days)

    # Lấy dates từ df nếu có
    df_reset = df.reset_index(drop=True)
    # vectors[i] tương ứng với df_use[i+59] do skip 59 bars đầu
    offset = 59

    for i in range(start, n):
        try:
            sr, conf, _ = _predict_current(gmm, scaler, label_mapping, vectors[i])
            # Map index về df
            df_idx = i + offset
            try:
                date_val = df_reset.iloc[df_idx].get("time", "") if df_idx < len(df_reset) else ""
                if hasattr(date_val, "strftime"):
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_val)[:10]
            except Exception:
                date_str = ""

            history.append({
                "date":       date_str,
                "regime":     sr,
                "label":      SR_LABELS[sr].split("—")[1].strip(),
                "confidence": round(conf, 2),
            })
        except Exception:
            continue

    return history


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_stock_regime(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    force_rebuild: bool = False,
) -> dict:
    """
    Lấy Stock Regime của một mã cổ phiếu.

    Args:
        symbol:        mã cổ phiếu (VD: "HAH", "VCB")
        df:            DataFrame OHLCV (nếu None thì tự load)
        force_rebuild: bỏ qua cache, tính lại

    Returns dict:
        ok:               bool
        symbol:           str
        regime:           int (1-4)
        label:            str "SR1 — Accumulation" ...
        emoji:            str
        desc:             str — mô tả regime
        action:           str — gợi ý hành động
        confidence:       float (0-1)
        confidence_label: str — "Ro rang (82%)" / "Bien gioi (48%)"
        sr_probs:         list[float] — [p_SR1, p_SR2, p_SR3, p_SR4]
        n_bars_used:      int
        history_30d:      list[dict]
        error:            str | None
    """
    # Check cache
    if not force_rebuild:
        cached = _load_cache(symbol)
        if cached:
            cached["ok"] = True
            cached.setdefault("history_30d", [])
            return cached

    # Load data nếu không có
    if df is None:
        try:
            from vn_loader import load_vn_ohlcv
            df = load_vn_ohlcv(symbol, days=600, min_bars=MIN_BARS_FOR_GMM)
        except Exception as e:
            return {
                "ok": False, "symbol": symbol,
                "error": f"Khong load duoc data {symbol}: {e}",
                "regime": 1, "label": SR_LABELS[1],
                "emoji": SR_EMOJI[1], "desc": SR_DESC[1],
                "action": SR_ACTION[1],
                "confidence": 0.0, "confidence_label": "Load data that bai",
                "sr_probs": [0.25, 0.25, 0.25, 0.25],
                "n_bars_used": 0, "history_30d": [],
            }

    result = _compute_stock_regime(symbol, df)
    if result["ok"]:
        _save_cache(symbol, result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT FUNCTIONS — dùng trong /check, /wave, /backtest
# ══════════════════════════════════════════════════════════════════════════════

def format_stock_regime_inline(sr_data: dict) -> str:
    """
    Format 1 dòng để nhúng vào /check output.

    Ví dụ:
        "🟢 SR2 Markup (82% ro rang) | Mua/giu — xac nhan voi wave"
        "🔵 SR1 Accumulation (58% kha ro) | Theo doi breakout"
        "🟡 SR3 Distribution (48% bien gioi→Markup) | Giam position"
    """
    if not sr_data.get("ok"):
        err = sr_data.get("error", "N/A")[:50]
        return f"⚪ Stock Regime: N/A ({err})"

    r     = sr_data["regime"]
    emoji = sr_data["emoji"]
    label = SR_LABELS[r].split("—")[1].strip()
    conf_label = sr_data["confidence_label"]
    action_short = SR_ACTION[r].split("—")[0].strip()

    return f"{emoji} Stock: SR{r} {label} | {conf_label} | {action_short}"


def format_stock_regime_block(sr_data: dict, show_history: bool = False) -> str:
    """
    Format block đầy đủ để dùng trong /wave và /backtest.

    Trả về multi-line string.
    """
    if not sr_data.get("ok"):
        return f"⚪ Stock Regime: Khong xac dinh ({sr_data.get('error', 'N/A')})"

    r      = sr_data["regime"]
    emoji  = sr_data["emoji"]
    label  = SR_LABELS[r]
    conf   = sr_data["confidence"]
    c_lbl  = sr_data["confidence_label"]
    desc   = sr_data["desc"]
    action = sr_data["action"]
    probs  = sr_data.get("sr_probs", [0.25] * 4)
    n_bars = sr_data.get("n_bars_used", 0)

    # Probability bar
    def _pbar(p: float) -> str:
        filled = min(10, round(p * 10))
        return "█" * filled + "░" * (10 - filled)

    lines = [
        f"{emoji} STOCK REGIME: {label}",
        f"   {c_lbl} | {n_bars} bars",
        f"   {desc}",
        f"   → {action}",
        "",
        "   Phan phoi xac suat:",
        f"   SR1 Accumulation [{_pbar(probs[0])}] {probs[0]:.0%}",
        f"   SR2 Markup       [{_pbar(probs[1])}] {probs[1]:.0%}",
        f"   SR3 Distribution [{_pbar(probs[2])}] {probs[2]:.0%}",
        f"   SR4 Markdown     [{_pbar(probs[3])}] {probs[3]:.0%}",
    ]

    # History 30 ngày nếu có
    if show_history and sr_data.get("history_30d"):
        history = sr_data["history_30d"]
        regime_char = {1: "🔵", 2: "🟢", 3: "🟡", 4: "🔴"}
        chart = "".join(regime_char.get(h["regime"], "⚪") for h in history[-20:])
        lines += [
            "",
            f"   20 ngay gan nhat: {chart}",
            "   (trai=cu, phai=moi | 🔵Acc 🟢Mup 🟡Dist 🔴Mkdn)",
        ]

        # Transition detection
        transitions = []
        for i in range(1, len(history)):
            if history[i]["regime"] != history[i-1]["regime"]:
                d    = history[i].get("date", "")[-5:]   # MM-DD
                from_r = history[i-1]["regime"]
                to_r   = history[i]["regime"]
                transitions.append(
                    f"   {d}: SR{from_r}→SR{to_r} "
                    f"({SR_LABELS[from_r].split('—')[1].strip()} "
                    f"→ {SR_LABELS[to_r].split('—')[1].strip()})"
                )
        if transitions:
            lines.append("   Chuyen doi:")
            lines.extend(transitions[-3:])

    return "\n".join(lines)


def format_stock_regime_for_wave(sr_data: dict) -> str:
    """
    Format ngắn gọn để nhúng vào /wave report (1-2 dòng).
    """
    if not sr_data.get("ok"):
        return ""
    r      = sr_data["regime"]
    emoji  = sr_data["emoji"]
    label  = SR_LABELS[r].split("—")[1].strip()
    conf   = sr_data["confidence_label"]

    # Alignment note giữa wave và stock regime
    note = ""
    if r == 2:   # Markup
        note = "→ Stock regime dong thuan voi song tang"
    elif r == 4:  # Markdown
        note = "→ Stock regime dong thuan voi song giam"
    elif r == 1:  # Accumulation
        note = "→ Tich luy: song giam co the la day cuoi"
    elif r == 3:  # Distribution
        note = "→ Phan phoi: song tang co the la dinh"

    return f"{emoji} StockRegime: SR{r} {label} ({conf}) {note}"


def format_stock_regime_for_backtest(sr_data: dict) -> str:
    """
    Format để nhúng vào /backtest report — 2 dòng.
    """
    if not sr_data.get("ok"):
        return ""
    r      = sr_data["regime"]
    emoji  = sr_data["emoji"]
    label  = SR_LABELS[r].split("—")[1].strip()
    conf   = sr_data["confidence_label"]
    action = sr_data["action"]
    return (
        f"{emoji} StockRegime: SR{r} {label} ({conf})\n"
        f"   {action}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-MARKET ANALYSIS — so sánh stock regime vs market regime
# ══════════════════════════════════════════════════════════════════════════════

def get_regime_alignment(
    stock_regime: int,
    market_regime: int,
) -> dict:
    """
    Phân tích sự đồng thuận / mâu thuẫn giữa Stock Regime và Market Regime.

    Returns dict:
        alignment: "DONG_THUAN" | "MAU_THUAN" | "TRUNG_TINH"
        note: str mô tả
        action_modifier: float — nhân vào confidence của action (1.2 đồng thuận, 0.7 mâu thuẫn)
    """
    # Bull market regimes (Phase 1): R1, R2
    # Bear market regimes (Phase 1): R3, R4
    market_bull = market_regime in (1, 2)
    market_bear = market_regime in (3, 4)

    # Bull stock regimes: SR2 Markup
    # Bear stock regimes: SR4 Markdown
    stock_bull = stock_regime == 2
    stock_bear = stock_regime == 4
    stock_neutral = stock_regime in (1, 3)

    if (stock_bull and market_bull):
        return {
            "alignment": "DONG_THUAN",
            "note": "Stock Markup + Market Bull → setup mua co xac suat cao nhat",
            "action_modifier": 1.2,
        }
    elif (stock_bear and market_bear):
        return {
            "alignment": "DONG_THUAN",
            "note": "Stock Markdown + Market Bear → tranh mua, rui ro cao",
            "action_modifier": 1.2,
        }
    elif (stock_bull and market_bear):
        return {
            "alignment": "MAU_THUAN",
            "note": "Stock Markup nhung Market Bear → alpha play, rui ro cao hon binh thuong",
            "action_modifier": 0.7,
        }
    elif (stock_bear and market_bull):
        return {
            "alignment": "MAU_THUAN",
            "note": "Stock Markdown nhung Market Bull → co phieu yeu hon thi truong, tranh mua",
            "action_modifier": 0.7,
        }
    elif stock_neutral:
        return {
            "alignment": "TRUNG_TINH",
            "note": f"Stock dang {'tich luy' if stock_regime == 1 else 'phan phoi'} — cho tin hieu ro hon",
            "action_modifier": 1.0,
        }
    else:
        return {
            "alignment": "TRUNG_TINH",
            "note": "Chua xac dinh ro alignment",
            "action_modifier": 1.0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED REGIME CONTEXT — dùng trong /check
# ══════════════════════════════════════════════════════════════════════════════

def get_combined_regime_context(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Lấy cả Stock Regime + Market Regime + alignment analysis.
    Dùng trong /check để có context đầy đủ.

    Returns dict:
        stock_regime:    dict từ get_stock_regime()
        market_regime:   dict từ get_market_regime()
        alignment:       dict từ get_regime_alignment()
        inline_text:     str — 2 dòng để nhúng vào /check
    """
    # Stock regime
    sr_data = get_stock_regime(symbol, df=df)

    # Market regime (Phase 1 — đã có cache)
    try:
        from market_regime import get_market_regime
        mr_data = get_market_regime()
    except Exception as e:
        mr_data = {"ok": False, "regime": 3, "error": str(e)}

    # Alignment
    sr_val = sr_data.get("regime", 1)
    mr_val = mr_data.get("regime", 3)
    alignment = get_regime_alignment(sr_val, mr_val)

    # Inline text (2 dòng cho /check)
    sr_inline = format_stock_regime_inline(sr_data)
    mr_emoji  = mr_data.get("emoji", "⚪")
    mr_label  = mr_data.get("label", "N/A").split("—")[-1].strip() if mr_data.get("ok") else "N/A"
    mr_inline = f"{mr_emoji} Market: {mr_label}" if mr_data.get("ok") else "⚪ Market Regime: N/A"

    align_emoji = {"DONG_THUAN": "✅", "MAU_THUAN": "⚠️", "TRUNG_TINH": "➡️"}.get(
        alignment["alignment"], "➡️"
    )
    inline_text = (
        f"{sr_inline}\n"
        f"{mr_inline}\n"
        f"{align_emoji} {alignment['note']}"
    )

    return {
        "stock_regime":  sr_data,
        "market_regime": mr_data,
        "alignment":     alignment,
        "inline_text":   inline_text,
    }
