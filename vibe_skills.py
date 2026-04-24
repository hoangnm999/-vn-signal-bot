"""
vibe_skills.py — Signal Engines cho VN Signal Bot
Chạy trực tiếp trên OHLCV Entrade (pure pandas/numpy, không cần API server).

NGUỒN GỐC THỰC TẾ (quan trọng):
──────────────────────────────────────────────────────────────────────────────
⚠️  CHƯA CÓ source code gốc HKUDS/Vibe-Trading (GitHub bị block khi deploy).
    Các engines dưới đây được implement theo logic trading chuẩn (RSI, MACD,
    Elliott Wave theory, v.v.) — KHÔNG phải copy từ HKUDS repo.
    Khi có source code gốc, cần rewrite lại theo đúng HKUDS implementation.

Label nguồn gốc:
  [HKUDS-like]  : Implement theo đúng tên/concept của HKUDS, logic tương đương
                  nhưng CHƯA verify với source gốc
  [Claude-made] : Tự sáng tạo hoàn toàn, không có tương đương trong HKUDS preset
──────────────────────────────────────────────────────────────────────────────

Engines (16 total):
  1.  Candlestick    [HKUDS] — 15 mô hình nến, vectorized sum→sign, volume filter (verified source)
  2.  Ichimoku       [HKUDS] — TK Cross event + 3-filter (rewrite từ source gốc)
  3.  TechnicalBasic [HKUDS] — 3-dim voting EMA/ADX+BB/RSI+OBV (rewrite từ source gốc)
  4.  ElliottWave    [HKUDS] — Zigzag + 5-wave + ABC + Fibonacci (verified source)
  5.  Harmonic       [HKUDS] — XABCD Gartley/Bat/Butterfly/Crab + pyharmonics fallback (verified source)
  6.  Volatility     [HKUDS] — HV percentile (lookback=120, low=20%, high=80%)
  7.  Seasonal       [HKUDS] — Fixed month lists (rewrite từ source gốc)
  8.  SMC            [HKUDS] — ChoCH priority + BOS + FVG filter (momentum fallback)
  9.  CrossMarket    [HKUDS] — Vol-adjusted dual-MA, a_share params 5/20 MA (rewrite từ source gốc)
  10. MultiFactor    [HKUDS] — 4-factor composite Z-score, adapted từ cross-section (rewrite từ source gốc)
  11. MeanReversion  [HKUDS] — Z-score entry=2.0 exit=0.5 lookback=60D, adapted từ Pair Trading
  12. PriceMomentum  [Claude-made] — Multi-TF momentum 5D/20D/60D
  13. Breakout       [Claude-made] — S/R breakout + vol confirm
  14. MLStrategy     [Claude-made] — ExtraTrees walk-forward + rule fallback
  15. Fundamental    [Claude-made] — PE/PB/ROE/EPS từ vnstock
  16. MoneyFlow      [Claude-made] — CMF/VPT/OBV/Force Index

Context agents (trong analyzer.py):
  17. MarketRegime   [Claude-made] — VN-Index proxy scoring
  18. NewsSentiment  [Claude-made] — RSS keyword scoring

Public API:
  result = run_vibe_agents(symbol, df)
  result["signals"]  → {engine_name: int}   (+1 bull / -1 bear / 0 neutral)
  result["details"]  → {engine_name: str}   (mô tả ngắn)
  result["verdict"]  → int
  result["summary"]  → str
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ─── helpers ─────────────────────────────────────────────────────────────────

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá OHLCV từ Entrade → DatetimeIndex, float columns."""
    d = df.copy()
    d.columns = [c.lower() for c in d.columns]
    for col in ("time", "trade_date", "date"):
        if col in d.columns:
            d[col] = pd.to_datetime(d[col])
            d = d.set_index(col)
            break
    for col in ("open", "high", "low", "close", "volume"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[d["close"] > 0]
    return d.sort_index()


def _last(series: pd.Series) -> int:
    """Trả về -1/0/+1 từ giá trị cuối series."""
    if series is None or len(series) == 0:
        return 0
    v = series.iloc[-1]
    return 0 if pd.isna(v) else int(np.sign(float(v)))


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 1 — CANDLESTICK (15 patterns, pure pandas, from Vibe-Trading)
# ══════════════════════════════════════════════════════════════════════════════

class CandlestickEngine:
    """
    K-line Pattern Recognition — rewrite theo HKUDS source gốc.
    [HKUDS] vibe/candlestick/example_signal_engine.py

    13 patterns có hướng + 2 neutral (doji, spinning_top trả về 0 như source gốc).
    Tổng: 15 patterns đúng như HKUDS spec.

    VN adaptation (giữ nguyên từ trước):
      Volume filter: vol<0.5x → reset signal (không có trong source gốc nhưng
      phù hợp TTCK VN nơi thanh khoản thấp làm nhiễu tín hiệu nến).
    """
    def __init__(self, body_pct: float = 0.1, shadow_ratio: float = 2.0):
        self.body_pct    = body_pct
        self.shadow_ratio = shadow_ratio

    # ── Helpers (giống source gốc) ────────────────────────────────────────
    @staticmethod
    def _body(o, c):   return (c - o).abs()
    @staticmethod
    def _range(h, l):  return h - l
    @staticmethod
    def _upper_shadow(o, c, h): return h - pd.concat([o, c], axis=1).max(axis=1)
    @staticmethod
    def _lower_shadow(o, c, l): return pd.concat([o, c], axis=1).min(axis=1) - l

    # ── Single-bar patterns ───────────────────────────────────────────────
    def _detect_hammer(self, o, h, l, c):
        bd = self._body(o, c); rng = self._range(h, l)
        ls = self._lower_shadow(o, c, l); us = self._upper_shadow(o, c, h)
        return ((ls >= self.shadow_ratio * bd) & (us < bd) & (bd > 0) & (rng > 0)).astype(int)

    def _detect_inverted_hammer(self, o, h, l, c):
        bd = self._body(o, c)
        us = self._upper_shadow(o, c, h); ls = self._lower_shadow(o, c, l)
        return ((us >= self.shadow_ratio * bd) & (ls < bd) & (bd > 0)).astype(int)

    def _detect_shooting_star(self, o, h, l, c):
        bd = self._body(o, c)
        us = self._upper_shadow(o, c, h); ls = self._lower_shadow(o, c, l)
        uptrend = c.shift(1) > c.shift(2)
        return -((us >= self.shadow_ratio * bd) & (ls < bd) & (bd > 0) & uptrend).astype(int)

    def _detect_doji(self, o, h, l, c):
        # [HKUDS] Doji = neutral, returns 0 (không đóng góp direction score)
        return pd.Series(0, index=o.index)

    def _detect_spinning_top(self, o, h, l, c):
        # [HKUDS] SpinningTop = neutral, returns 0
        # (detect nhưng không đóng góp direction — giống source gốc)
        return pd.Series(0, index=o.index)

    # ── Double-bar patterns ───────────────────────────────────────────────
    def _detect_engulfing(self, o, h, l, c):
        o1, c1 = o.shift(1), c.shift(1)
        bullish = (c1 < o1) & (c > o) & (c >= o1) & (o <= c1)
        bearish = (c1 > o1) & (c < o) & (c <= o1) & (o >= c1)
        sig = pd.Series(0, index=o.index)
        sig[bullish] = 1; sig[bearish] = -1
        return sig

    def _detect_harami(self, o, h, l, c):
        bd = self._body(o, c)
        o1, c1 = o.shift(1), c.shift(1); bd1 = self._body(o1, c1)
        prev_top = pd.concat([o1, c1], axis=1).max(axis=1)
        prev_bot = pd.concat([o1, c1], axis=1).min(axis=1)
        curr_top = pd.concat([o, c],   axis=1).max(axis=1)
        curr_bot = pd.concat([o, c],   axis=1).min(axis=1)
        contained = (curr_top <= prev_top) & (curr_bot >= prev_bot)
        sig = pd.Series(0, index=o.index)
        sig[(c1 < o1) & (bd1 > bd) & contained] =  1
        sig[(c1 > o1) & (bd1 > bd) & contained] = -1
        return sig

    def _detect_piercing_line(self, o, h, l, c):
        # [HKUDS] opens_below = o < l1 (not l.shift(1))
        o1, c1, l1 = o.shift(1), c.shift(1), l.shift(1)
        mid1 = (o1 + c1) / 2
        cond = (c1 < o1) & (c > o) & (o < l1) & (c > mid1)
        return cond.astype(int)

    def _detect_dark_cloud(self, o, h, l, c):
        # [HKUDS] opens_above = o > h1
        o1, c1, h1 = o.shift(1), c.shift(1), h.shift(1)
        mid1 = (o1 + c1) / 2
        cond = (c1 > o1) & (c < o) & (o > h1) & (c < mid1)
        return -(cond.astype(int))

    # ── Triple-bar patterns ───────────────────────────────────────────────
    def _detect_morning_star(self, o, h, l, c):
        o1, c1           = o.shift(2), c.shift(2)   # Day1
        o2, c2, h2       = o.shift(1), c.shift(1), h.shift(1)  # Day2
        bd2  = self._body(o2, c2)
        rng2 = self._range(h.shift(1), l.shift(1)).replace(0, np.nan)
        day1_bear        = c1 < o1
        day2_small       = bd2 / rng2 < 0.3
        day2_gap         = h2 < l.shift(2)           # gap down: Day2 high < Day1 low
        day3_bull        = c > o
        day3_above_mid   = c > (o1 + c1) / 2
        cond = day1_bear & day2_small & day2_gap & day3_bull & day3_above_mid
        return cond.astype(int).fillna(0)

    def _detect_evening_star(self, o, h, l, c):
        o1, c1           = o.shift(2), c.shift(2)   # Day1
        o2, c2, l2       = o.shift(1), c.shift(1), l.shift(1)  # Day2
        bd2  = self._body(o2, c2)
        rng2 = self._range(h.shift(1), l.shift(1)).replace(0, np.nan)
        day1_bull        = c1 > o1
        day2_small       = bd2 / rng2 < 0.3
        day2_gap         = l2 > h.shift(2)           # gap up: Day2 low > Day1 high
        day3_bear        = c < o
        day3_below_mid   = c < (o1 + c1) / 2
        cond = day1_bull & day2_small & day2_gap & day3_bear & day3_below_mid
        return -(cond.astype(int).fillna(0))

    def _detect_three_white_soldiers(self, o, h, l, c):
        o1, c1 = o.shift(2), c.shift(2)
        o2, c2 = o.shift(1), c.shift(1)
        bull1 = c1 > o1; bull2 = c2 > o2; bull3 = c > o
        close_up  = (c2 > c1) & (c > c2)
        open2_in  = (o2 >= o1) & (o2 <= c1)   # Day2 opens inside Day1 body
        open3_in  = (o  >= o2) & (o  <= c2)   # Day3 opens inside Day2 body
        cond = bull1 & bull2 & bull3 & close_up & open2_in & open3_in
        return cond.astype(int).fillna(0)

    def _detect_three_black_crows(self, o, h, l, c):
        o1, c1 = o.shift(2), c.shift(2)
        o2, c2 = o.shift(1), c.shift(1)
        bear1 = c1 < o1; bear2 = c2 < o2; bear3 = c < o
        close_dn  = (c2 < c1) & (c < c2)
        open2_in  = (o2 <= o1) & (o2 >= c1)  # Day2 opens inside Day1 bear body
        open3_in  = (o  <= o2) & (o  >= c2)  # Day3 opens inside Day2 bear body
        cond = bear1 & bear2 & bear3 & close_dn & open2_in & open3_in
        return -(cond.astype(int).fillna(0))

    # ── Generate ──────────────────────────────────────────────────────────
    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
        signals, details = {}, {}
        for code, df in data_map.items():
            o, h, l, c = df["open"], df["high"], df["low"], df["close"]
            v = df.get("volume", pd.Series(1, index=df.index))

            # [HKUDS] 15 patterns — doji/spinning_top neutral (trả về 0)
            sc = pd.DataFrame(index=df.index)
            sc["hammer"]          = self._detect_hammer(o, h, l, c)
            sc["inv_hammer"]      = self._detect_inverted_hammer(o, h, l, c)
            sc["shooting_star"]   = self._detect_shooting_star(o, h, l, c)
            sc["doji"]            = self._detect_doji(o, h, l, c)            # neutral
            sc["spinning_top"]    = self._detect_spinning_top(o, h, l, c)    # neutral
            sc["engulfing"]       = self._detect_engulfing(o, h, l, c)
            sc["harami"]          = self._detect_harami(o, h, l, c)
            sc["piercing"]        = self._detect_piercing_line(o, h, l, c)
            sc["dark_cloud"]      = self._detect_dark_cloud(o, h, l, c)
            sc["morning_star"]    = self._detect_morning_star(o, h, l, c)
            sc["evening_star"]    = self._detect_evening_star(o, h, l, c)
            sc["three_white"]     = self._detect_three_white_soldiers(o, h, l, c)
            sc["three_black"]     = self._detect_three_black_crows(o, h, l, c)
            total = sc.sum(axis=1)  # [HKUDS] sum → sign

            # [VN adaptation] Volume filter — không có trong source gốc
            vol_ma20  = v.rolling(20).mean()
            vol_ratio = v / vol_ma20.replace(0, np.nan)
            sig_series = pd.Series(np.sign(total).astype(int), index=df.index)
            sig_series[vol_ratio < 0.5] = 0                          # vol quá thấp
            sig_series[(vol_ratio >= 0.5) & (vol_ratio < 0.7) & (total.abs() < 2)] = 0

            signals[code] = _last(sig_series)

            # Detail
            last_row = sc.iloc[-1]
            cur_vr   = round(float(vol_ratio.iloc[-1]), 2) if not pd.isna(vol_ratio.iloc[-1]) else "N/A"
            # Chỉ report patterns có direction (bỏ doji/spinning_top vì luôn = 0)
            dir_cols = [c for c in sc.columns if c not in ("doji", "spinning_top")]
            found_bull = [n for n in dir_cols if last_row.get(n, 0) > 0]
            found_bear = [n for n in dir_cols if last_row.get(n, 0) < 0]
            desc = ""
            if found_bull: desc += f"Bullish: {', '.join(found_bull)}. "
            if found_bear: desc += f"Bearish: {', '.join(found_bear)}. "
            if not desc:   desc = "Khong nhan dien mo hinh nen co huong. "
            vol_note = (f"[Vol {cur_vr}x < 0.5x: RESET]" if isinstance(cur_vr, float) and cur_vr < 0.5
                        else f"[Vol {cur_vr}x thap]" if isinstance(cur_vr, float) and cur_vr < 0.7
                        else f"[Vol {cur_vr}x OK]")
            details[code] = (desc + vol_note).strip()
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 2 — ICHIMOKU (from Vibe-Trading)
# ══════════════════════════════════════════════════════════════════════════════

class IchimokuEngine:
    """
    Ichimoku Kinko Hyo — implement đúng theo HKUDS source.
    [HKUDS] agent/src/skills/ichimoku/example_signal_engine.py

    Logic: TK Cross event → 3-filter confirmation
      1. tk_cross_up/down (trigger)
      2. price above/below cloud (direction confirm)
      3. cloud bullish/bearish direction (trend confirm)
    """
    def __init__(self, tenkan=9, kijun=26, senkou_b=52, displacement=26):
        self.tenkan      = tenkan
        self.kijun       = kijun
        self.senkou_b    = senkou_b
        self.displacement= displacement

    def _donchian_mid(self, high, low, period):
        return (high.rolling(period).max() + low.rolling(period).min()) / 2

    def _one(self, df):
        if len(df) < self.senkou_b + self.displacement:
            return pd.Series(0, index=df.index, dtype=int), "Khong du data"

        h = df["high"]; l = df["low"]; c = df["close"]
        tenkan  = self._donchian_mid(h, l, self.tenkan)
        kijun   = self._donchian_mid(h, l, self.kijun)
        span_a  = ((tenkan + kijun) / 2).shift(self.displacement)
        span_b  = self._donchian_mid(h, l, self.senkou_b).shift(self.displacement)

        # TK Cross events (HKUDS logic)
        tk_cross_up   = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
        tk_cross_down = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

        # Cloud position
        cloud_top    = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        above_cloud  = c > cloud_top
        below_cloud  = c < cloud_bottom

        # Cloud direction
        bullish_cloud = span_a > span_b
        bearish_cloud = span_a < span_b

        # 3-filter signal (HKUDS exact logic)
        buy  = tk_cross_up   & above_cloud & bullish_cloud
        sell = tk_cross_down & below_cloud & bearish_cloud
        sig  = buy.astype(int) - sell.astype(int)

        # Detail
        tk_val = round(float(tenkan.iloc[-1]), 2) if not pd.isna(tenkan.iloc[-1]) else "N/A"
        kj_val = round(float(kijun.iloc[-1]),  2) if not pd.isna(kijun.iloc[-1])  else "N/A"
        ct_val = round(float(cloud_top.iloc[-1]),    2) if not pd.isna(cloud_top.iloc[-1])    else "N/A"
        cb_val = round(float(cloud_bottom.iloc[-1]), 2) if not pd.isna(cloud_bottom.iloc[-1]) else "N/A"
        pos    = "TREN may" if bool(above_cloud.iloc[-1]) else "DUOI may" if bool(below_cloud.iloc[-1]) else "TRONG may"
        cloud_dir = "Tang(xanh)" if bool(bullish_cloud.iloc[-1]) else "Giam(do)"
        cur   = _last(sig)
        det   = (f"Tenkan={tk_val} Kijun={kj_val} | {pos} (top={ct_val} bot={cb_val}) "
                 f"| May={cloud_dir} | TK_cross={'UP' if bool(tk_cross_up.iloc[-1]) else 'DOWN' if bool(tk_cross_down.iloc[-1]) else 'none'}")
        return sig, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code] = _last(sig)
                details[code] = det
            except Exception as e:
                signals[code] = 0
                details[code] = f"Loi IchimokuEngine: {e}"
        return signals, details


class TechnicalEngine:
    """
    Technical Basic — implement đúng theo HKUDS source.
    [HKUDS] agent/src/skills/technical-basic/example_signal_engine.py

    Logic: 3-dim voting
      Trend: EMA cross + ADX strength
      MeanRev: BB + RSI oversold/overbought
      VolPrice: OBV vs OBV-MA
      Signal: (trend_bull|mr_oversold) & vol_bull & ~mr_overbought → BUY
              (trend_bear|mr_overbought) & vol_bear & ~mr_oversold → SELL
    """
    def __init__(self, ema_fast=12, ema_slow=26, adx_period=14,
                 adx_threshold=25.0, bb_window=20, bb_std=2.0,
                 rsi_period=14, rsi_oversold=30, rsi_overbought=70,
                 obv_ma_period=20):
        self.ema_fast       = ema_fast
        self.ema_slow       = ema_slow
        self.adx_period     = adx_period
        self.adx_threshold  = adx_threshold
        self.bb_window      = bb_window
        self.bb_std         = bb_std
        self.rsi_period     = rsi_period
        self.rsi_oversold   = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.obv_ma_period  = obv_ma_period

    def _rsi(self, close, period):
        """Wilder EWM RSI — theo HKUDS technical-basic source."""
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_l = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs    = avg_g / avg_l
        return 100 - 100 / (1 + rs)

    def _adx(self, high, low, close, period):
        """ADX + DI theo HKUDS source (Wilder EWM)."""
        ph = high.shift(1); pl = low.shift(1); pc = close.shift(1)
        up = high - ph; dn = pl - low
        plus_dm  = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)
        plus_dm[(up > dn) & (up > 0)]   = up
        minus_dm[(dn > up) & (dn > 0)]  = dn
        tr = pd.concat([high-low, (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
        alpha = 1/period
        s_tr  = tr.ewm(alpha=alpha, min_periods=period).mean()
        s_pdm = plus_dm.ewm(alpha=alpha, min_periods=period).mean()
        s_mdm = minus_dm.ewm(alpha=alpha, min_periods=period).mean()
        pdi   = 100 * s_pdm / s_tr.replace(0, np.nan)
        mdi   = 100 * s_mdm / s_tr.replace(0, np.nan)
        di_sum= (pdi + mdi).replace(0, np.nan)
        dx    = 100 * (pdi - mdi).abs() / di_sum
        adx   = dx.ewm(alpha=alpha, min_periods=period).mean()
        return adx, pdi, mdi

    def _obv(self, close, volume):
        sign = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        return (volume * sign).cumsum()

    def _one(self, df):
        c = df["close"]; h = df["high"]; l = df["low"]
        v = df.get("volume", pd.Series(1, index=df.index))

        # Trend dim
        ema_f = c.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = c.ewm(span=self.ema_slow, adjust=False).mean()
        adx, pdi, mdi = self._adx(h, l, c, self.adx_period)
        trend_bull = (ema_f > ema_s) & (adx > self.adx_threshold)
        trend_bear = (ema_f < ema_s) & (adx > self.adx_threshold)

        # Mean reversion dim
        bb_mid  = c.rolling(self.bb_window).mean()
        bb_std  = c.rolling(self.bb_window).std()
        bb_up   = bb_mid + self.bb_std * bb_std
        bb_lo   = bb_mid - self.bb_std * bb_std
        rsi     = self._rsi(c, self.rsi_period)
        mr_over  = (c < bb_lo) & (rsi < self.rsi_oversold)
        mr_over2 = (c > bb_up) & (rsi > self.rsi_overbought)

        # Volume-price dim (OBV)
        obv    = self._obv(c, v)
        obv_ma = obv.rolling(self.obv_ma_period).mean()
        vol_bull = obv > obv_ma
        vol_bear = obv < obv_ma

        # 3-dim voting (HKUDS exact logic)
        buy  = (trend_bull | mr_over)  & vol_bull & ~mr_over2
        sell = (trend_bear | mr_over2) & vol_bear & ~mr_over

        sig = buy.astype(int) - sell.astype(int)
        sig = sig.fillna(0).astype(int)

        # Detail
        cur_adx   = round(float(adx.iloc[-1]),  1) if not pd.isna(adx.iloc[-1])  else "N/A"
        cur_ema_f = round(float(ema_f.iloc[-1]), 2) if not pd.isna(ema_f.iloc[-1]) else "N/A"
        cur_ema_s = round(float(ema_s.iloc[-1]), 2) if not pd.isna(ema_s.iloc[-1]) else "N/A"
        cur_rsi   = round(float(rsi.iloc[-1]),   1) if not pd.isna(rsi.iloc[-1])   else "N/A"
        trend_str = "bull" if bool(trend_bull.iloc[-1]) else "bear" if bool(trend_bear.iloc[-1]) else "neutral"
        cur       = _last(sig)
        det = (f"EMA{self.ema_fast}={cur_ema_f} EMA{self.ema_slow}={cur_ema_s} "
               f"ADX={cur_adx} RSI={cur_rsi} trend={trend_str} "
               f"| OBV={'bull' if bool(vol_bull.iloc[-1]) else 'bear'} "
               f"| Signal={'BUY' if cur>0 else 'SELL' if cur<0 else 'NEUTRAL'}")
        return sig, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code] = _last(sig)
                details[code] = det
            except Exception as e:
                signals[code] = 0
                details[code] = f"Loi TechnicalEngine: {e}"
        return signals, details


class ElliottEngine:
    FIBS = [0.236,0.382,0.5,0.618,0.786,1.0,1.272,1.414,1.618,2.0,2.618]

    def __init__(self, window=5, min_pct=0.02, tol=0.15):
        self.window=window; self.min_pct=min_pct; self.tol=tol

    def _swings(self, close):
        w=self.window
        highs=close[(close==close.rolling(2*w+1,center=True).max())]
        lows =close[(close==close.rolling(2*w+1,center=True).min())]
        pts=pd.concat([pd.Series(1,index=highs.index),pd.Series(-1,index=lows.index)]).sort_index()
        res=[]; prev=None
        for idx,val in pts.items():
            if val!=prev:
                res.append((idx,int(val),float(close[idx]))); prev=val
        return res

    def _fib_ok(self, a, b):
        if a==0: return False
        r=abs(b/a)
        return any(abs(r-f)<self.tol for f in self.FIBS)

    def _impulse(self, swings):
        if len(swings)<6: return None
        for i in range(len(swings)-5):
            pts=swings[i:i+6]; px=[p[2] for p in pts]; ty=[p[1] for p in pts]
            if ty not in ([1,-1,1,-1,1,-1],[-1,1,-1,1,-1,1]): continue
            w1=abs(px[1]-px[0]); w3=abs(px[3]-px[2]); w5=abs(px[5]-px[4])
            if w3<w1 and w3<w5: continue
            if w1>0 and self._fib_ok(w1,w3):
                return {"start":pts[0][0],"end":pts[5][0],"dir":ty[0],"w1":w1,"w3":w3,"w5":w5}
        return None

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            sig=pd.Series(0,index=df.index,dtype=int)
            det="Chua xac dinh duoc cau truc song ro rang."
            try:
                sw=self._swings(df["close"])
                wave=self._impulse(sw[-12:]) if len(sw)>=6 else None
                if wave:
                    end_idx=wave["end"]; d=wave["dir"]
                    sig[(df.index>=wave["start"])&(df.index<=end_idx)]=d
                    sig[df.index>end_idx]=-d  # After wave5 → expect correction
                    cur_sig = _last(sig)
                    phase = "dang song" if cur_sig == d else "dieu chinh sau song 5"
                    det=(f"5-song {'tang' if d>0 else 'giam'} phat hien ({phase}). "
                         f"W1={round(wave['w1'],2)} W3={round(wave['w3'],2)} W5={round(wave['w5'],2)}")
            except Exception as e:
                det=f"Loi: {e}"
            signals[code]=_last(sig); details[code]=det
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 5 — HARMONIC PATTERNS (Gartley/Bat/Butterfly/Crab, from Vibe-Trading)
# ══════════════════════════════════════════════════════════════════════════════

class HarmonicEngine:
    PATTERNS = {
        "Gartley":   {"XB":(0.618,0.618),"AC":(0.382,0.886),"BD":(0.786,0.786)},
        "Bat":       {"XB":(0.382,0.500),"AC":(0.382,0.886),"BD":(0.886,0.886)},
        "Butterfly": {"XB":(0.786,0.786),"AC":(0.382,0.886),"BD":(1.618,2.618)},
        "Crab":      {"XB":(0.382,0.618),"AC":(0.382,0.886),"BD":(2.618,3.618)},
    }
    TOL = 0.15

    def _sp(self, h, l, w=5):
        hi=h[(h==h.rolling(2*w+1,center=True).max())].dropna()
        lo=l[(l==l.rolling(2*w+1,center=True).min())].dropna()
        return hi, lo

    def _fib_in(self, ratio, lo, hi):
        return (lo-self.TOL)<=ratio<=(hi+self.TOL)

    def _check(self, X, A, B, C, D, pdef):
        try:
            xa=abs(A-X); ab=abs(B-A); bc=abs(C-B); cd=abs(D-C)
            if xa==0 or ab==0 or bc==0: return False
            lo,hi=pdef["XB"]
            if not self._fib_in(ab/xa,lo,hi): return False
            lo,hi=pdef["AC"]
            if not self._fib_in(bc/ab,lo,hi): return False
            lo,hi=pdef["BD"]
            if not self._fib_in(cd/bc,lo,hi): return False
            return True
        except: return False

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            sig=pd.Series(0,index=df.index,dtype=int)
            det="Khong co mo hinh harmonic duoc xac nhan."
            try:
                h,l=df["high"],df["low"]
                hi,lo=self._sp(h,l)
                found_name=""
                # Bullish (tìm trên lows)
                lp=lo.values[-6:]; li=lo.index[-6:]
                if len(lp)>=5:
                    X,A,B,C,D=lp[-5],lp[-4],lp[-3],lp[-2],lp[-1]
                    for pn,pd_ in self.PATTERNS.items():
                        if self._check(X,A,B,C,D,pd_):
                            sig[df.index>=li[-1]]=1; found_name=pn; break
                # Bearish (tìm trên highs)
                if not found_name:
                    hp=hi.values[-6:]; hidx=hi.index[-6:]
                    if len(hp)>=5:
                        X,A,B,C,D=hp[-5],hp[-4],hp[-3],hp[-2],hp[-1]
                        for pn,pd_ in self.PATTERNS.items():
                            if self._check(X,A,B,C,D,pd_):
                                sig[df.index>=hidx[-1]]=-1; found_name=pn; break
                if found_name:
                    direction="BUY" if _last(sig)>0 else "SELL"
                    det=f"Mo hinh {found_name} phat hien — {direction} signal tai PRZ."
            except Exception as e:
                det=f"Loi: {e}"
            signals[code]=_last(sig); details[code]=det
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 6 — VOLATILITY (HV Percentile, from Vibe-Trading)
# ══════════════════════════════════════════════════════════════════════════════

class VolatilityEngine:
    """
    Volatility Mean-Reversion Signal + GARCH(1,1) conditional volatility.
    [HKUDS] agent/src/skills/VolatilityEngine/example_signal_engine.py
    [HKUDS] quant-statistics/SKILL.md — GARCH modeling

    Logic:
      Primary: HV percentile (HKUDS) — low_pct<20 → BUY, high_pct>80 → SELL
      Upgrade:  GARCH(1,1) conditional vol forecast (khi có arch lib)
                σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1}
                GARCH vol rising → bearish; GARCH vol falling → bullish
    """
    def __init__(self, hv_w=20, pct_w=120, lo_pct=20, hi_pct=80, ann=252):
        # [HKUDS] lookback=120, low=20%, high=80%
        self.hv_w=hv_w; self.pct_w=pct_w; self.lo_pct=lo_pct; self.hi_pct=hi_pct; self.ann=ann
        self._has_arch = self._check_arch()

    @staticmethod
    def _check_arch() -> bool:
        try:
            from arch import arch_model  # noqa
            return True
        except ImportError:
            return False

    def _hv(self, c):
        return np.log(c/c.shift(1)).rolling(self.hv_w).std()*np.sqrt(self.ann)

    def _garch_vol(self, c: pd.Series) -> tuple:
        """
        GARCH(1,1) conditional volatility forecast.
        [HKUDS] quant-statistics/SKILL.md
        σ²_t = ω + α×ε²_{t-1} + β×σ²_{t-1}

        Returns: (garch_vol_current, garch_trend) where trend in {-1,0,1}
        """
        try:
            from arch import arch_model
            ret = np.log(c/c.shift(1)).dropna() * 100  # percent returns
            if len(ret) < 60:
                return None, 0
            model = arch_model(ret, vol="Garch", p=1, q=1, dist="normal")
            res   = model.fit(disp="off", show_warning=False)
            fc    = res.forecast(horizon=1)
            cond_var = res.conditional_volatility
            # Annualized GARCH vol
            garch_current = float(cond_var.iloc[-1]) * np.sqrt(self.ann) / 100
            garch_prev5   = float(cond_var.iloc[-5]) * np.sqrt(self.ann) / 100 if len(cond_var) >= 5 else garch_current
            # Trend: GARCH vol rising → vol regime increasing → bearish
            if garch_current > garch_prev5 * 1.1:
                garch_trend = -1  # vol rising → bearish
            elif garch_current < garch_prev5 * 0.9:
                garch_trend = 1   # vol falling → bullish
            else:
                garch_trend = 0
            return garch_current, garch_trend
        except Exception:
            return None, 0

    def _one(self, df):
        c = df["close"]
        hv = self._hv(c)
        min_p = max(20, min(self.pct_w, len(hv)-1))
        pct = hv.rolling(min_p).apply(
            lambda x: (pd.Series(x).rank(pct=True).iloc[-1]) * 100, raw=False
        )
        sig     = pd.Series(0, index=df.index, dtype=int)
        sig[pct < self.lo_pct] =  1   # low vol → expect expansion → BUY
        sig[pct > self.hi_pct] = -1   # high vol → expect contraction → SELL

        cur_hv  = hv.iloc[-1]
        cur_pct = pct.iloc[-1]
        regime  = "low_vol" if cur_pct < self.lo_pct else "high_vol" if cur_pct > self.hi_pct else "normal"

        # GARCH upgrade
        garch_str = ""
        if self._has_arch and len(df) >= 60:
            g_vol, g_trend = self._garch_vol(c)
            if g_vol is not None:
                # Blend: nếu GARCH trend đồng chiều → reinforce, khác chiều → dampen
                cur_sig = _last(sig)
                if g_trend != 0 and g_trend == cur_sig:
                    pass  # reinforce (giữ nguyên)
                elif g_trend != 0 and g_trend != cur_sig and cur_sig != 0:
                    sig.iloc[-1] = 0   # conflicting → neutral
                garch_str = f" | GARCH_vol={g_vol*100:.1f}%({'rising' if g_trend<0 else 'falling' if g_trend>0 else 'flat'})"

        det = (f"[HKUDS+GARCH] HV20={round(cur_hv*100,1) if not pd.isna(cur_hv) else 'N/A'}% "
               f"Percentile={round(cur_pct,0) if not pd.isna(cur_pct) else 'N/A'}% "
               f"Regime={regime}{garch_str}")
        return sig, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code] = _last(sig)
                details[code] = det
            except Exception as e:
                signals[code] = 0
                details[code] = f"Loi VolatilityEngine: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 7 — SEASONAL (Calendar Effect, from Vibe-Trading)
# ══════════════════════════════════════════════════════════════════════════════

class SeasonalEngine:
    """
    Seasonal/Calendar Effect — implement đúng theo HKUDS source.
    [HKUDS] agent/src/skills/SeasonalEngine/example_signal_engine.py

    Logic: Fixed month lists (bullish/bearish months) + optional weekday effect.
    VN market adaptation:
      bullish_months: [1,2,3,10,11,12] — đầu năm + cuối năm (mùa kết quả kinh doanh)
      bearish_months: [5,6,7,8,9]      — "Sell in May" effect
    """
    def __init__(self,
                 bullish_months=None,
                 bearish_months=None,
                 use_weekday=False,
                 bullish_weekdays=None,
                 bearish_weekdays=None):
        # VN market: [1,2,3,10,11,12] tích cực, [5,6,7,8,9] tiêu cực
        self.bullish_months   = bullish_months   or [1, 2, 3, 10, 11, 12]
        self.bearish_months   = bearish_months   or [5, 6, 7, 8, 9]
        self.use_weekday      = use_weekday
        self.bullish_weekdays = bullish_weekdays or [4]   # Friday
        self.bearish_weekdays = bearish_weekdays or [0]   # Monday

    def _one(self, df):
        idx    = df.index
        month  = idx.month
        signal = pd.Series(0, index=idx, dtype=int)

        # Month effect (HKUDS exact logic)
        signal[month.isin(self.bullish_months)] = 1
        signal[month.isin(self.bearish_months)] = -1

        # Weekday effect (optional — HKUDS double confirmation)
        if self.use_weekday:
            weekday = idx.weekday
            wd_sig  = pd.Series(0, index=idx, dtype=int)
            wd_sig[weekday.isin(self.bullish_weekdays)] = 1
            wd_sig[weekday.isin(self.bearish_weekdays)] = -1
            combined = signal + wd_sig
            signal = pd.Series(0, index=idx, dtype=int)
            signal[combined >= 2]  = 1
            signal[combined <= -2] = -1

        # Detail
        import datetime
        cm   = datetime.datetime.now().month
        mn   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        bias = ("TANG" if cm in self.bullish_months else
                "GIAM" if cm in self.bearish_months else "TRUNG LAP")
        cur  = _last(signal)
        det  = (f"Thang {mn[cm-1]}({cm}): thuong {bias}. "
                f"Bull months={self.bullish_months} | Bear months={self.bearish_months}. "
                f"Signal={'BUY' if cur>0 else 'SELL' if cur<0 else 'NEUTRAL'}")
        return signal, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code] = _last(sig)
                details[code] = det
            except Exception as e:
                signals[code] = 0
                details[code] = f"Loi SeasonalEngine: {e}"
        return signals, details


class SMCEngine:
    """
    Smart Money Concepts (ICT) Signal Engine.
    [HKUDS] vibe/smc/example_signal_engine.py

    Source gốc dùng `smartmoneyconcepts` lib:
      1. swing_highs_lows()  → swing H/L
      2. bos_choch()         → BOS/ChoCH structure (+1/-1)
      3. fvg()               → Fair Value Gap filter
      Signal: ChoCH priority, BOS supplement, FVG same-direction filter.

    Fallback (khi không có lib): pure-pandas momentum scoring — giữ nguyên
    từ implementation cũ, đã fix NEUTRAL bug từ session trước.
    """
    def __init__(self, swing_length: int = 10, close_break: bool = True):
        self.swing_length = swing_length
        self.close_break  = close_break
        self._has_smc     = self._check_smc()

    @staticmethod
    def _check_smc() -> bool:
        try:
            from smartmoneyconcepts import smc  # noqa
            return True
        except ImportError:
            return False

    # ── Primary: smartmoneyconcepts lib (HKUDS exact) ─────────────────────
    def _compute_signal_lib(self, df: pd.DataFrame) -> tuple:
        """Dùng smartmoneyconcepts lib — exact logic từ HKUDS source gốc."""
        from smartmoneyconcepts import smc

        ohlc = df[["open", "high", "low", "close", "volume"]].copy()

        min_bars = self.swing_length * 2
        if len(ohlc) < min_bars:
            return pd.Series(0, index=df.index, dtype=int), f"Khong du data (can >={min_bars} bars)"

        # 1) Swing H/L
        swing_hl = smc.swing_highs_lows(ohlc, swing_length=self.swing_length)

        # 2) BOS/ChoCH
        bos_choch = smc.bos_choch(ohlc, swing_highs_lows=swing_hl,
                                   close_break=self.close_break)

        # 3) FVG
        fvg = smc.fvg(ohlc)

        bos_val   = bos_choch["BOS"].fillna(0).astype(int)
        choch_val = bos_choch["CHOCH"].fillna(0).astype(int)
        fvg_val   = fvg["FVG"].fillna(0).astype(int)

        # [HKUDS] ChoCH priority over BOS
        structure = choch_val.where(choch_val != 0, bos_val)

        # [HKUDS] FVG same-direction filter
        buy  = (structure ==  1) & (fvg_val >= 0)
        sell = (structure == -1) & (fvg_val <= 0)
        signal = buy.astype(int) - sell.astype(int)
        signal = pd.Series(signal.values, index=df.index, dtype=int)

        cur = _last(signal)
        last_bos   = int(bos_val.iloc[-1])
        last_choch = int(choch_val.iloc[-1])
        last_fvg   = int(fvg_val.iloc[-1])
        det = (f"[HKUDS][smc-lib] swing={self.swing_length} close_break={self.close_break} | "
               f"BOS={last_bos:+d} ChoCH={last_choch:+d} FVG={last_fvg:+d} | "
               f"Signal={'BUY' if cur>0 else 'SELL' if cur<0 else 'NEUTRAL'}")
        return signal, det

    # ── Fallback: pure-pandas momentum scoring ────────────────────────────
    def _swing_hl(self, h, l):
        w = self.swing_length
        swing_h = h[(h == h.rolling(2*w+1, center=True).max())].dropna()
        swing_l = l[(l == l.rolling(2*w+1, center=True).min())].dropna()
        return swing_h, swing_l

    def _bos_choch_pandas(self, df, swing_h, swing_l):
        c   = df["close"]
        sig = pd.Series(0, index=df.index, dtype=int)
        for idx in swing_h.index[:-1]:
            lvl   = swing_h[idx]
            after = c[c.index > idx]
            breaks = after[after > lvl] if self.close_break else after[df["high"][after.index] > lvl]
            if not breaks.empty:
                sig[breaks.index[0]] = 1
        for idx in swing_l.index[:-1]:
            lvl   = swing_l[idx]
            after = c[c.index > idx]
            breaks = after[after < lvl] if self.close_break else after[df["low"][after.index] < lvl]
            if not breaks.empty:
                sig[breaks.index[0]] = -1
        return sig

    def _fvg_pandas(self, df):
        h, l  = df["high"], df["low"]
        fvg   = pd.Series(0, index=df.index, dtype=int)
        fvg[l > h.shift(2)]  =  1  # bull FVG
        fvg[h < l.shift(2)]  = -1  # bear FVG
        return fvg

    def _compute_signal_pandas(self, df: pd.DataFrame) -> tuple:
        """Pure-pandas fallback — momentum net scoring."""
        if len(df) < self.swing_length * 2:
            return pd.Series(0, index=df.index, dtype=int), "Khong du data"

        sh, sl    = self._swing_hl(df["high"], df["low"])
        structure = self._bos_choch_pandas(df, sh, sl)
        fvg       = self._fvg_pandas(df)

        w             = min(60, len(df))
        struct_recent = structure.iloc[-w:]
        fvg_recent    = fvg.iloc[-w:]

        n_buy_total  = int((structure ==  1).sum())
        n_sell_total = int((structure == -1).sum())
        n_buy_rec    = int((struct_recent ==  1).sum())
        n_sell_rec   = int((struct_recent == -1).sum())
        fvg_bull     = int((fvg_recent ==  1).sum())
        fvg_bear     = int((fvg_recent == -1).sum())

        net_struct = (n_buy_rec * 2 + n_buy_total) - (n_sell_rec * 2 + n_sell_total)
        net_fvg    = fvg_bull - fvg_bear

        if   net_struct < -2 and net_fvg <= 0: cur_sig, bias = -1, "BEARISH"
        elif net_struct >  2 and net_fvg >= 0: cur_sig, bias =  1, "BULLISH"
        elif net_struct < -1:                   cur_sig, bias = -1, "LEAN BEARISH"
        elif net_struct >  1:                   cur_sig, bias =  1, "LEAN BULLISH"
        else:                                   cur_sig, bias =  0, "NEUTRAL"

        sig = pd.Series(0, index=df.index, dtype=int)
        sig.iloc[-1] = cur_sig
        det = (f"[pandas-fallback] BOS/ChoCH: {n_buy_total}bull/{n_sell_total}bear "
               f"(60D: {n_buy_rec}/{n_sell_rec}) FVG: +{fvg_bull}/-{fvg_bear} "
               f"Net={net_struct:+d} | {bias}")
        return sig, det

    # ── Main entry ────────────────────────────────────────────────────────
    def generate(self, data_map: dict) -> tuple:
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                if self._has_smc:
                    sig, det = self._compute_signal_lib(df)
                else:
                    sig, det = self._compute_signal_pandas(df)
                signals[code] = _last(sig)
                details[code] = det
            except Exception as e:
                signals[code] = 0
                details[code] = f"Loi SMCEngine: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 9 — CROSS-MARKET (Vol-adjusted Dual-MA, từ Vibe-Trading)
# Với single VN stock: dùng a_share params (MA5/MA20)
# ══════════════════════════════════════════════════════════════════════════════

class CrossMarketEngine:
    """
    Cross-Market Vol-Adjusted Dual-MA Signal.
    [HKUDS] agent/src/skills/cross-market-strategy/example_signal_engine.py

    HKUDS logic:
      1. Per-market MA params (a_share: fast=5, slow=20)
      2. Vol-adjusted weight = 1/vol (inverse volatility)
      3. Signal clipped to [-1, 1]

    VN stock: dùng a_share params (5/20 MA) — đúng nhất cho TTCK VN
    Single-symbol mode: vol-weight normalize to 1.0
    """
    MARKET_PARAMS = {
        "a_share":   {"ma_fast": 5,  "ma_slow": 20, "vol_lookback": 20},
        "crypto":    {"ma_fast": 7,  "ma_slow": 25, "vol_lookback": 14},
        "us_equity": {"ma_fast": 10, "ma_slow": 50, "vol_lookback": 20},
        "hk_equity": {"ma_fast": 10, "ma_slow": 50, "vol_lookback": 20},
        "forex":     {"ma_fast": 10, "ma_slow": 30, "vol_lookback": 20},
    }

    def _detect_market(self, code: str) -> str:
        import re
        patterns = [
            (re.compile(r"^\d{6}\.(SZ|SH|BJ)$", re.I), "a_share"),
            (re.compile(r"^[A-Z]+-USDT$", re.I),           "crypto"),
            (re.compile(r"^[A-Z]+\.US$", re.I),           "us_equity"),
            (re.compile(r"^\d{3,5}\.HK$", re.I),         "hk_equity"),
        ]
        for pat, mkt in patterns:
            if pat.match(code):
                return mkt
        # VN stocks (2-5 chữ cái): a_share params
        return "a_share"

    def _raw_signal(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Dual-MA signal theo HKUDS _market_signal()."""
        c     = df["close"]
        mf    = c.rolling(params["ma_fast"]).mean()
        ms    = c.rolling(params["ma_slow"]).mean()
        sig   = pd.Series(0.0, index=df.index)
        sig[mf > ms] =  1.0
        sig[mf < ms] = -1.0
        return sig

    def _vol(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """Tính volatility gần nhất (std of daily returns)."""
        ret = df["close"].pct_change().dropna()
        if len(ret) > lookback:
            return float(ret.rolling(lookback).std().iloc[-1])
        return float(ret.std()) if len(ret) > 1 else 1e-10

    def _one(self, df: pd.DataFrame, code: str) -> tuple:
        market  = self._detect_market(code)
        params  = self.MARKET_PARAMS.get(market, self.MARKET_PARAMS["a_share"])
        raw_sig = self._raw_signal(df, params)

        # Vol-adjustment (single symbol → weight normalizes to 1.0)
        v      = self._vol(df, params["vol_lookback"])
        inv_v  = 1.0 / (v + 1e-10)
        # Single-symbol: weight = 1.0 (no pool to normalize against)
        sig    = raw_sig.clip(-1.0, 1.0)

        mf_v  = round(float(df["close"].rolling(params["ma_fast"]).mean().iloc[-1]), 2)
        ms_v  = round(float(df["close"].rolling(params["ma_slow"]).mean().iloc[-1]), 2)
        cur   = _last(sig.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0))
        vol_pct = round(v * 100, 2)
        # Đổi tên display: a_share → vn_stock (tránh hiểu nhầm là sàn TQ)
        # Logic hoàn toàn không đổi — a_share params (MA5/MA20) phù hợp nhất cho VN
        display_market = "vn_stock" if market == "a_share" else market
        det   = (f"MA{params['ma_fast']}={mf_v} MA{params['ma_slow']}={ms_v} "
                 f"Vol20={vol_pct}% Market={display_market} "
                 f"Signal={'BUY' if cur > 0 else 'SELL' if cur < 0 else 'NEUTRAL'}")
        return sig, det

    def generate(self, data_map: dict) -> tuple:
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df, code)
                signals[code] = _last(sig.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0))
                details[code] = det
            except Exception as e:
                signals[code] = 0
                details[code] = f"Loi CrossMarketEngine: {e}"
        return signals, details


class MultiFactorEngine:
    """
    Multi-Factor Cross-Section Ranking Signal.
    [HKUDS] vibe/multi-factor/example_signal_engine.py

    Source gốc: cross-section zscore → TopN ranking → equal-weight long.
    Factors: momentum(20D), reversal(5D), volatility(20D), volume_ratio(20D).

    VN adaptation:
      - Multi-symbol: chạy đúng cross-section như HKUDS, signal = 1/N (selected) hoặc 0.
        Để ra {-1,0,1} chuẩn cho voting: selected → +1, bottom N → -1, rest → 0.
      - Single-symbol (chỉ 1 mã truyền vào): fallback time-series Z-score.
        Đây là limitation thực tế vì run_vibe_agents() truyền data_map = {symbol: df}.
        Time-series Z-score có ý nghĩa khác cross-section nhưng là cách tốt nhất
        có thể với 1 mã đơn lẻ.
    """
    FACTOR_NAMES = ["momentum", "reversal", "volatility", "volume_ratio"]

    def __init__(self, momentum_window: int = 20, vol_window: int = 20,
                 top_n: int = 3, rebalance_freq: int = 20,
                 # Single-symbol fallback params
                 z_lookback: int = 60, threshold: float = 0.5):
        self.momentum_window = momentum_window
        self.vol_window      = vol_window
        self.top_n           = top_n
        self.rebalance_freq  = rebalance_freq
        self.z_lookback      = z_lookback
        self.threshold       = threshold

    def _compute_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """4 factors — exact định nghĩa từ HKUDS source."""
        c   = df["close"]
        v   = df.get("volume", pd.Series(1, index=df.index))
        ret = c.pct_change()
        f   = pd.DataFrame(index=df.index)
        f["momentum"]     = c / c.shift(self.momentum_window) - 1  # positive = bullish
        f["reversal"]     = -(c / c.shift(5) - 1)                  # contrarian
        f["volatility"]   = -ret.rolling(self.vol_window).std()     # low vol = bullish
        f["volume_ratio"] = v / v.rolling(self.vol_window).mean()   # high vol = confirm
        return f

    @staticmethod
    def _zscore_cross_section(vals: dict) -> dict:
        """[HKUDS] Cross-section Z-score normalize."""
        clean = [v for v in vals.values() if not np.isnan(v)]
        if len(clean) < 2:
            return {k: 0.0 for k in vals}
        mu  = np.mean(clean)
        std = np.std(clean, ddof=1)
        if std < 1e-12:
            return {k: 0.0 for k in vals}
        return {k: (v - mu) / std if not np.isnan(v) else 0.0 for k, v in vals.items()}

    def _cross_section(self, data_map: dict) -> dict:
        """
        [HKUDS] Cross-section ranking — chạy đúng khi có ≥2 mã.
        TopN → signal +1, BottomN → signal -1, rest → 0.
        """
        codes       = list(data_map.keys())
        factor_map  = {code: self._compute_factors(df) for code, df in data_map.items()}
        all_dates   = sorted(set().union(*(f.index for f in factor_map.values())))
        date_index  = pd.DatetimeIndex(all_dates)
        signals     = {c: pd.Series(0.0, index=date_index) for c in codes}

        last_sel: list = []
        last_bot: list = []
        n_eff = min(self.top_n, len(codes) // 2) or 1

        for i, dt in enumerate(date_index):
            # Non-rebalance day: carry last signal
            if i % self.rebalance_freq != 0 and (last_sel or last_bot):
                for c in last_sel:  signals[c].at[dt] = 1.0
                for c in last_bot:  signals[c].at[dt] = -1.0
                continue

            # Rebalance: cross-section scoring
            composite: dict = {c: 0.0 for c in codes}
            for fn in self.FACTOR_NAMES:
                raw = {c: factor_map[c].at[dt, fn]
                       if dt in factor_map[c].index else np.nan
                       for c in codes}
                zs = self._zscore_cross_section(raw)
                for c in codes:
                    composite[c] += zs.get(c, 0.0)

            ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)
            valid  = [(c, s) for c, s in ranked if not np.isnan(s)]
            last_sel = [c for c, _ in valid[:n_eff]]
            last_bot = [c for c, _ in valid[-n_eff:]] if len(valid) > n_eff else []

            for c in last_sel: signals[c].at[dt] = 1.0
            for c in last_bot: signals[c].at[dt] = -1.0

        return {code: signals[code].reindex(data_map[code].index).fillna(0.0)
                for code in codes}

    def _single_symbol_zscore(self, df: pd.DataFrame) -> tuple:
        """
        Single-symbol fallback: time-series Z-score.
        [VN adaptation] Không thể dùng cross-section với 1 mã —
        time-series Z-score là xấp xỉ tốt nhất trong context này.
        """
        if len(df) < self.z_lookback + self.momentum_window + 5:
            return pd.Series(0, index=df.index, dtype=int), "Khong du data"

        factors   = self._compute_factors(df)
        composite = pd.Series(0.0, index=df.index)
        for fn in self.FACTOR_NAMES:
            mu  = factors[fn].rolling(self.z_lookback).mean()
            std = factors[fn].rolling(self.z_lookback).std().replace(0, np.nan)
            composite += ((factors[fn] - mu) / std).fillna(0)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[composite >  self.threshold] =  1
        sig[composite < -self.threshold] = -1

        cur      = _last(sig)
        cur_comp = round(float(composite.iloc[-1]), 3) if not pd.isna(composite.iloc[-1]) else "N/A"
        cur_mom  = round(float(factors["momentum"].iloc[-1]) * 100, 2) if not pd.isna(factors["momentum"].iloc[-1]) else "N/A"
        cur_vr   = round(float(factors["volume_ratio"].iloc[-1]), 2) if not pd.isna(factors["volume_ratio"].iloc[-1]) else "N/A"
        det = (f"[HKUDS][ts-zscore fallback] Composite_Z={cur_comp} "
               f"mom={cur_mom}% vol_ratio={cur_vr}x "
               f"threshold=±{self.threshold} "
               f"Signal={'BUY' if cur>0 else 'SELL' if cur<0 else 'NEUTRAL'}")
        return sig, det

    def generate(self, data_map: dict) -> tuple:
        signals_out, details_out = {}, {}

        if len(data_map) >= 2:
            # ── Multi-symbol: cross-section ranking đúng HKUDS ────────────
            try:
                raw = self._cross_section(data_map)
                for code in data_map:
                    s   = raw[code]
                    val = _last(s.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0))
                    f   = self._compute_factors(data_map[code])
                    cur_mom = round(float(f["momentum"].iloc[-1])*100, 2) if not pd.isna(f["momentum"].iloc[-1]) else "N/A"
                    signals_out[code] = val
                    details_out[code] = (f"[HKUDS][cross-section] top_n={self.top_n} rebal={self.rebalance_freq}D "
                                         f"mom={cur_mom}% "
                                         f"Signal={'BUY' if val>0 else 'SELL' if val<0 else 'NEUTRAL'}")
            except Exception as e:
                for code, df in data_map.items():
                    signals_out[code] = 0
                    details_out[code] = f"Loi cross-section: {e}"
        else:
            # ── Single-symbol: time-series Z-score fallback ───────────────
            for code, df in data_map.items():
                try:
                    sig, det = self._single_symbol_zscore(df)
                    signals_out[code] = _last(sig)
                    details_out[code] = det
                except Exception as e:
                    signals_out[code] = 0
                    details_out[code] = f"Loi MultiFactorEngine: {e}"

        return signals_out, details_out


class MeanReversionEngine:
    """
    Mean Reversion + Cointegration Signal.
    [HKUDS] agent/src/skills/Pair trading/example_signal_engine.py
             agent/src/skills/correlation-analysis/SKILL.md

    Combines 2 approaches:
      1. Z-score vs rolling mean (HKUDS Pair Trading)
         entry_z=2.0, exit_z=0.5, lookback=60D
      2. Engle-Granger cointegration test vs VNINDEX proxy
         (correlation-analysis SKILL) — upgrade khi có statsmodels

    Cho TTCK VN: mã có cointegration với sector ETF/proxy
    thường hội tụ về mean nhanh hơn → signal chất lượng cao hơn.
    """
    def __init__(self, lookback=60, entry_z=2.0, exit_z=0.5):
        self.lookback = lookback
        self.entry_z  = entry_z
        self.exit_z   = exit_z
        self._has_sm  = self._check_statsmodels()

    @staticmethod
    def _check_statsmodels() -> bool:
        try:
            from statsmodels.tsa.stattools import coint
            return True
        except ImportError:
            return False

    def _zscore_signal(self, c: pd.Series) -> tuple:
        """HKUDS Pair Trading: Z-score vs rolling mean."""
        mean = c.rolling(self.lookback).mean()
        std  = c.rolling(self.lookback).std().replace(0, np.nan)
        z    = (c - mean) / std
        sig  = pd.Series(0, index=c.index, dtype=int)
        sig[z < -self.entry_z] =  1   # oversold → BUY
        sig[z >  self.entry_z] = -1   # overbought → SELL
        sig[z.abs() < self.exit_z] = 0
        return sig, z

    def _cointegration_check(self, c: pd.Series, proxy: pd.Series) -> dict:
        """
        Engle-Granger cointegration test (correlation-analysis SKILL).
        Trả về: {is_cointegrated, p_value, hedge_ratio, spread_z}
        """
        if not self._has_sm:
            return {"is_cointegrated": False, "p": 1.0}
        try:
            from statsmodels.tsa.stattools import coint
            import statsmodels.api as sm
            # Align
            df = pd.concat([c.rename("y"), proxy.rename("x")], axis=1).dropna()
            if len(df) < self.lookback + 10:
                return {"is_cointegrated": False, "p": 1.0}
            y = df["y"]; x = df["x"]
            # Engle-Granger test
            _, p, _ = coint(y, x)
            # OLS hedge ratio
            x_c = sm.add_constant(x)
            ols = sm.OLS(y, x_c).fit()
            hr  = float(ols.params.iloc[1])
            spread = y - hr * x
            # Spread Z-score
            sp_m = spread.rolling(self.lookback).mean()
            sp_s = spread.rolling(self.lookback).std().replace(0, np.nan)
            sp_z = (spread - sp_m) / sp_s
            return {
                "is_cointegrated": p < 0.05,
                "p": round(float(p), 4),
                "hedge_ratio": round(hr, 4),
                "spread_z": sp_z,
            }
        except Exception:
            return {"is_cointegrated": False, "p": 1.0}

    def _one(self, df: pd.DataFrame, code: str,
             proxy_df: pd.DataFrame = None) -> tuple:
        if len(df) < self.lookback + 5:
            return pd.Series(0, index=df.index, dtype=int), "Khong du data"

        c = df["close"]
        sig_zs, z = self._zscore_signal(c)
        cur_z   = round(float(z.iloc[-1]), 3) if not pd.isna(z.iloc[-1]) else "N/A"

        # Cointegration upgrade (nếu có proxy VNINDEX)
        coint_info = ""
        if proxy_df is not None and self._has_sm:
            res = self._cointegration_check(c, proxy_df["close"])
            if res.get("is_cointegrated"):
                sp_z_last = round(float(res["spread_z"].iloc[-1]), 3)                             if res.get("spread_z") is not None and                                not pd.isna(res["spread_z"].iloc[-1]) else None
                if sp_z_last is not None:
                    # Override signal với spread Z-score (chất lượng cao hơn)
                    if sp_z_last < -self.entry_z:
                        sig_zs.iloc[-1] = 1
                    elif sp_z_last > self.entry_z:
                        sig_zs.iloc[-1] = -1
                    elif abs(sp_z_last) < self.exit_z:
                        sig_zs.iloc[-1] = 0
                coint_info = (f" | Cointegrated(p={res['p']}) "
                              f"hr={res.get('hedge_ratio','?')} "
                              f"spread_z={sp_z_last}")
            else:
                coint_info = f" | No coint(p={res['p']})"

        cur = _last(sig_zs)
        det = (f"Z-score={cur_z} (entry±{self.entry_z} exit±{self.exit_z} "
               f"lookback={self.lookback}D){coint_info} "
               f"Signal={'BUY(oversold)' if cur>0 else 'SELL(overbought)' if cur<0 else 'NEUTRAL'}")
        return sig_zs, det

    def generate(self, data_map: dict) -> tuple:
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df, code)
                signals[code] = _last(sig)
                details[code] = det
            except Exception as e:
                signals[code] = 0
                details[code] = f"Loi MeanReversionEngine: {e}"
        return signals, details


class MLStrategyEngine:
    """
    Machine-Learning Predictive Strategy.
    [HKUDS] agent/src/skills/ml-strategy/SKILL.md

    Đúng theo HKUDS spec:
      Features (10): ret_5d, ret_20d, vol_20d, ma_ratio, volume_ratio,
                     rsi_14, bb_position, high_low_ratio, close_open_ratio, skew_20d
      Model: RandomForestClassifier(n_estimators=100, max_depth=5)
      min_train_size=252, retrain_freq=20, horizon=5D
      Output: prob*2-1 → [-1,1], discrete signal {-1,0,1}
      Signal threshold: abs(signal) < 0.1 → 0
    """
    def __init__(self, min_train_size=252, retrain_freq=20,
                 horizon=5, threshold=0.3,
                 model_type="random_forest"):
        # threshold=0.3: raw signal [-1,1] phải vượt ±0.3 mới tính MUA/BÁN
        # RawSignal 0.416 > 0.3 → MUA ✓  |  0.2 → NEUTRAL ✓  |  -0.35 → BÁN ✓
        # threshold cũ 0.1 quá thấp: 0.416 lọt qua dù gần trung tính
        self.min_train  = min_train_size
        self.retrain_f  = retrain_freq
        self.horizon    = horizon
        self.threshold  = threshold
        self.model_type = model_type

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """10 features đúng theo HKUDS build_features()."""
        c   = df["close"]
        v   = df.get("volume", pd.Series(1, index=df.index))
        ret = c.pct_change()
        o   = df.get("open", c)
        h   = df["high"]; l = df["low"]

        feat = pd.DataFrame(index=df.index)
        feat["f_ret_5d"]         = c.pct_change(5)
        feat["f_ret_20d"]        = c.pct_change(20)
        feat["f_vol_20d"]        = ret.rolling(20).std()
        feat["f_ma_ratio"]       = c / c.rolling(20).mean()
        feat["f_volume_ratio"]   = v / v.rolling(20).mean()
        # RSI(14) — HKUDS dùng rolling mean (không phải Wilder EWM)
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        feat["f_rsi_14"]         = 100 - (100 / (1 + rs))
        # Bollinger Band position — guard zero-bandwidth
        ma20  = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_up = ma20 + 2 * std20; bb_lo = ma20 - 2 * std20
        bb_rng= (bb_up - bb_lo).replace(0, np.nan)
        feat["f_bb_position"]    = (c - bb_lo) / bb_rng
        # Intraday features
        feat["f_high_low_ratio"] = (h - l) / c
        feat["f_close_open_ratio"]= (c - o) / o.replace(0, np.nan)
        feat["f_skew_20d"]       = ret.rolling(20).skew()
        return feat.replace([np.inf, -np.inf], np.nan)

    def _walk_forward(self, feat: pd.DataFrame, labels: pd.Series) -> pd.Series:
        """Walk-forward predict — đúng theo HKUDS walk_forward_predict()."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return pd.Series(0.0, index=feat.index)

        preds  = pd.Series(0.0, index=feat.index)
        model  = None; scaler = None

        for i in range(self.min_train, len(feat)):
            # Retrain mỗi retrain_freq days (không phải mỗi bar → tiết kiệm thời gian)
            if model is None or (i - self.min_train) % self.retrain_f == 0:
                X_tr = feat.iloc[:i].values
                y_tr = labels.iloc[:i].values
                valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
                X_tr = X_tr[valid]; y_tr = y_tr[valid]
                if len(X_tr) < 50:
                    continue
                scaler = StandardScaler()
                X_tr   = scaler.fit_transform(X_tr)
                # HKUDS: RandomForest(n_estimators=100, max_depth=5)
                model  = RandomForestClassifier(
                    n_estimators=100, max_depth=5,
                    random_state=42, n_jobs=1,
                    class_weight="balanced",   # handle bull-market imbalance
                )
                model.fit(X_tr, y_tr)

            if scaler is None:
                continue
            X_now = feat.iloc[i:i+1].values
            if np.isnan(X_now).any():
                continue
            X_now = scaler.transform(X_now)
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_now)[0, 1]
                preds.iloc[i] = prob * 2 - 1   # [0,1] → [-1,1] — HKUDS spec
            else:
                preds.iloc[i] = float(model.predict(X_now)[0])

        return preds.fillna(0.0).clip(-1.0, 1.0)

    def _one(self, df: pd.DataFrame) -> tuple:
        # Validate minimum data (HKUDS: min_rows=300)
        if len(df) < max(self.min_train + self.horizon + 10, 300):
            return pd.Series(0, index=df.index, dtype=int),                    f"Khong du data (can >={self.min_train} bars)"

        feat   = self._build_features(df)
        # Labels: future horizon-day return > 0
        labels = (df["close"].pct_change(self.horizon).shift(-self.horizon) > 0).astype(int)
        raw    = self._walk_forward(feat, labels)

        # Discrete signal với threshold (HKUDS: threshold=0.0 default, ta dùng 0.3 — xem __init__)
        sig = raw.apply(lambda x: 1 if x > self.threshold
                        else -1 if x < -self.threshold else 0).astype(int)
        cur     = _last(sig)
        cur_raw = round(float(raw.iloc[-1]), 3)
        det = (f"[HKUDS] ML(RandomForest n=100 depth=5 "
               f"train={self.min_train}D retrain={self.retrain_f}D "
               f"horizon={self.horizon}D) "
               f"RawSignal={cur_raw:+.3f} threshold=±{self.threshold} "
               f"Signal={'BUY' if cur>0 else 'SELL' if cur<0 else 'NEUTRAL'}")
        return sig, det

    def generate(self, data_map: dict) -> tuple:
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code] = _last(sig)
                details[code] = det
            except Exception as e:
                signals[code] = 0
                details[code] = f"Loi MLStrategy: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 13 — CHANLUN (缠中说禅 / Chanlun Theory)
# [HKUDS] agent/src/skills/chanlun/example_signal_engine.py
#
# Lý thuyết Chanlun: K-line → FX (phân hình) → BI (bút) → ZS (trung khu) → tín hiệu
# Buy points: 一买 (1st buy), 三笔向上盘背, 五笔类一买
# Sell points: 一卖 (1st sell), 三笔向下盘背, 五笔类一卖
# Rất phổ biến tại TTCK VN và A-shares
#
# Requires: pip install czsc
# ══════════════════════════════════════════════════════════════════════════════

class ChanlunEngine:
    """
    Chanlun (缠中说禅) Pattern Recognition.
    [HKUDS] agent/src/skills/chanlun/example_signal_engine.py

    Logic: Raw bars → FX (phân hình) → BI (bút) → ZS (trung khu) → buy/sell points
    Buy:  一买 signal / 三笔向上盘背 / 五笔类一买 / BI转折 near ZS bottom
    Sell: 一卖 signal / 三笔向下盘背 / 五笔类一卖 / BI转折 near ZS top

    Requires czsc library: pip install czsc
    Fallback khi không có czsc: pure-pandas Chanlun lite implementation
    """
    def __init__(self, min_bars=30):
        self.min_bars   = min_bars
        self._has_czsc  = self._check_czsc()

    @staticmethod
    def _check_czsc() -> bool:
        try:
            import czsc  # noqa
            return True
        except ImportError:
            return False

    def _df_to_bars(self, df, symbol):
        """Convert OHLCV DataFrame → czsc RawBar list."""
        from czsc import RawBar, Freq
        from datetime import datetime
        bars = []
        for i, (dt, row) in enumerate(df.iterrows()):
            if not isinstance(dt, datetime):
                dt = pd.Timestamp(dt).to_pydatetime()
            bars.append(RawBar(
                symbol=symbol, id=i, dt=dt, freq=Freq.D,
                open=float(row["open"]),  close=float(row["close"]),
                high=float(row["high"]),  low=float(row["low"]),
                vol=float(row.get("volume", row.get("vol", 0))),
                amount=float(row.get("amount", 0)),
            ))
        return bars

    def _get_czsc_signals(self, c):
        """Compute Chanlun signals from CZSC object."""
        from czsc.signals.cxt import (
            cxt_first_buy_V221126, cxt_first_sell_V221126,
            cxt_bi_base_V230228, cxt_three_bi_V230618, cxt_five_bi_V230619,
        )
        s = {}
        s.update(cxt_first_buy_V221126(c, di=1))
        s.update(cxt_first_sell_V221126(c, di=1))
        s.update(cxt_bi_base_V230228(c, di=1))
        s.update(cxt_three_bi_V230618(c, di=1))
        s.update(cxt_five_bi_V230619(c, di=1))
        return s

    def _evaluate(self, c) -> int:
        """Evaluate CZSC signals → {-1, 0, 1}."""
        try:
            from czsc import ZS
            signals = c.signals
            if not signals: return 0
            # 一买
            buy1 = [k for k in signals if "BUY1" in k]
            if buy1 and "一买" in str(signals.get(buy1[0], "")): return 1
            # 一卖
            sell1 = [k for k in signals if "SELL1" in k]
            if sell1 and "一卖" in str(signals.get(sell1[0], "")): return -1
            # 三笔形态
            three = [k for k in signals if "三笔" in k]
            if three:
                v = str(signals.get(three[0], ""))
                if "向上盘背" in v: return 1
                if "向下盘背" in v: return -1
            # 五笔形态
            five = [k for k in signals if "五笔" in k]
            if five:
                v = str(signals.get(five[0], ""))
                if "类一买" in v: return 1
                if "类一卖" in v: return -1
            # BI + ZS position
            bi_key = [k for k in signals if "V230228" in k]
            if bi_key and len(c.bi_list) >= 3:
                v = str(signals.get(bi_key[0], ""))
                for i in range(len(c.bi_list)-3, max(len(c.bi_list)-10,-1), -1):
                    try:
                        zs = ZS(bis=c.bi_list[i:i+3])
                        if zs.is_valid:
                            lc = c.bars_raw[-1].close
                            if "向下_转折" in v and lc <= zs.zd: return 1
                            if "向上_转折" in v and lc >= zs.zg: return -1
                            break
                    except Exception:
                        break
        except Exception:
            pass
        return 0

    def _one_czsc(self, df: pd.DataFrame, code: str) -> tuple:
        """Full Chanlun via czsc library."""
        from czsc import CZSC
        bars  = self._df_to_bars(df, code)
        sig   = pd.Series(0, index=df.index, dtype=int)
        if len(bars) < self.min_bars:
            return sig, "Khong du bars cho Chanlun"
        c = CZSC(bars[:self.min_bars], get_signals=self._get_czsc_signals)
        for bar in bars[self.min_bars:]:
            c.update(bar)
            v = self._evaluate(c)
            if v != 0:
                sig.iloc[bar.id] = v
        cur = _last(sig)
        n_bi = len(c.bi_list)
        det = (f"[HKUDS] Chanlun(czsc) | BI count={n_bi} "
               f"| Signal={'BUY(一买/盘背)' if cur>0 else 'SELL(一卖/盘背)' if cur<0 else 'NEUTRAL'}")
        return sig, det

    def _one_lite(self, df: pd.DataFrame) -> tuple:
        """
        Chanlun LITE — pure pandas fallback khi không có czsc.
        Phát hiện FX (phân hình) đơn giản + BI direction.
        """
        h = df["high"]; l = df["low"]; c = df["close"]
        sig = pd.Series(0, index=df.index, dtype=int)

        # Phát hiện顶分型 (top FX) và底分型 (bottom FX)
        top_fx  = (h > h.shift(1)) & (h > h.shift(-1))  # local high
        bot_fx  = (l < l.shift(1)) & (l < l.shift(-1))  # local low

        # BI direction: nếu vừa tạo bottom FX → potential buy
        #               nếu vừa tạo top FX → potential sell
        # Thêm filter: giá đang nằm gần extreme của FX
        for i in range(2, len(df)-1):
            if bool(bot_fx.iloc[i]):
                # bottom phân hình → look-ahead: nếu giá phục hồi → BUY
                sig.iloc[i] = 1
            elif bool(top_fx.iloc[i]):
                sig.iloc[i] = -1

        cur = _last(sig)
        det = (f"[HKUDS] Chanlun(lite-fallback: no czsc) "
               f"| FX-based signal "
               f"| Signal={'BUY' if cur>0 else 'SELL' if cur<0 else 'NEUTRAL'}")
        return sig, det

    def _one(self, df: pd.DataFrame, code: str) -> tuple:
        if self._has_czsc:
            try:
                return self._one_czsc(df, code)
            except Exception as e:
                pass  # fallthrough to lite
        return self._one_lite(df)

    def generate(self, data_map: dict) -> tuple:
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df, code)
                signals[code] = _last(sig)
                details[code] = det
            except Exception as e:
                signals[code] = 0
                details[code] = f"Loi ChanlunEngine: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 14 — FUNDAMENTAL FILTER (PE/PB/ROE Value Screen)
# [HKUDS] agent/src/skills/fundamental-filter/example_signal_engine.py
#          agent/src/skills/fundamental-filter/SKILL.md
#
# HKUDS gốc: cross-stock PE/PB/ROE filter → equal-weight TopN
# Adapt cho single-symbol VN:
#   - Fetch PE/PB/ROE từ vnstock (KBS) hoặc yfinance
#   - So sánh với ngưỡng chuẩn VN market (không dùng cross-section ranking)
#   - Signal: pass all 3 thresholds → BUY, fail any → SELL, missing data → NEUTRAL
#
# VN market thresholds (theo VN30 historical average):
#   PE: 0 < PE < 20 (cheap), 20-30 (fair), >30 (expensive)
#   PB: < 3.0 (reasonable for VN)
#   ROE: > 8% (quality floor)
# ══════════════════════════════════════════════════════════════════════════════

class FundamentalFilterEngine:
    """
    Fundamental Factor Screening — PE/PB/ROE value filter.
    [HKUDS] agent/src/skills/fundamental-filter/example_signal_engine.py

    HKUDS gốc: cross-stock equal-weight TopN selection.
    Single-symbol adapt: absolute threshold screen cho TTCK VN.

    Data sources (priority order):
      1. vnstock KBS — finance.ratio() quarterly
      2. yfinance — Ticker.info (for non-VN stocks)
      3. Fallback: NEUTRAL (không có data)

    Thresholds (VN market calibrated):
      PE:  0 < pe < pe_max (mặc định 20) — loại lỗ và overvalued
      PB:  pb < pb_max (mặc định 3.0)
      ROE: roe > roe_min (mặc định 8%)
    """
    def __init__(self, pe_min=0.0, pe_max=20.0, pb_max=3.0, roe_min=8.0):
        """
        [HKUDS] defaults: pe_min=0, pe_max=20, pb_max=3, roe_min=8
        """
        self.pe_min  = pe_min
        self.pe_max  = pe_max
        self.pb_max  = pb_max
        self.roe_min = roe_min

    def _fetch_fundamentals_vnstock(self, symbol: str) -> dict:
        """Lấy PE/PB/ROE từ vnstock KBS quarterly ratio."""
        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=symbol, source="KBS")
            for period in ["quarter", "annual"]:
                try:
                    ratio = stock.finance.ratio(period=period)
                    if ratio is None or ratio.empty:
                        continue
                    latest = ratio.iloc[0]
                    cols   = [c.lower() for c in latest.index]
                    orig   = list(latest.index)

                    def _find(patterns):
                        for p in patterns:
                            for i, c in enumerate(cols):
                                if p.lower() in c:
                                    try:
                                        v = float(latest[orig[i]])
                                        if not pd.isna(v) and v != 0:
                                            return v
                                    except Exception:
                                        pass
                        return None

                    pe  = _find(["pricetoearning","p/e","pe_ttm","pe"])
                    pb  = _find(["pricetobook","p/b","pb"])
                    roe = _find(["roe"])
                    if pe is not None or roe is not None:
                        # Normalize ROE nếu dạng 0.15 thay vì 15
                        if roe is not None and abs(roe) < 2:
                            roe = roe * 100
                        return {"pe": pe, "pb": pb, "roe": roe,
                                "source": f"vnstock KBS ({period})"}
                except Exception:
                    continue
        except Exception:
            pass
        return {}

    def _fetch_fundamentals_yfinance(self, symbol: str) -> dict:
        """Fallback: lấy từ yfinance (cho mã không có trong vnstock)."""
        try:
            import yfinance as yf
            info = yf.Ticker(symbol).info
            pe   = info.get("trailingPE") or info.get("forwardPE")
            pb   = info.get("priceToBook")
            roe  = info.get("returnOnEquity")
            if roe is not None and abs(roe) < 2:
                roe = roe * 100
            return {"pe": pe, "pb": pb, "roe": roe, "source": "yfinance"}
        except Exception:
            return {}

    def _fetch_fundamentals_cafef(self, symbol: str) -> dict:
        """
        Fallback 3: crawl nhẹ CafeF API (JSON public, không cần auth).
        Endpoint: https://s.cafef.vn/Ajax/Utilities/GetFinanceRatio.ashx?symbol=VCB
        Trả về PE, PB, ROE từ dữ liệu trailing 12M.
        """
        try:
            import urllib.request, json
            url = f"https://s.cafef.vn/Ajax/Utilities/GetFinanceRatio.ashx?symbol={symbol.upper()}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode())

            # CafeF trả về list, lấy phần tử đầu (TTM)
            if isinstance(data, list) and data:
                item = data[0]
            elif isinstance(data, dict):
                item = data
            else:
                return {}

            def _safe(keys):
                for k in keys:
                    v = item.get(k)
                    if v is not None:
                        try:
                            f = float(str(v).replace(",",""))
                            if not pd.isna(f) and f != 0:
                                return f
                        except Exception:
                            pass
                return None

            pe  = _safe(["PE", "P_E", "pe"])
            pb  = _safe(["PB", "P_B", "pb"])
            roe = _safe(["ROE", "roe"])
            if roe is not None and abs(roe) < 2:
                roe = roe * 100
            if pe is not None or roe is not None:
                return {"pe": pe, "pb": pb, "roe": roe, "source": "CafeF"}
        except Exception:
            pass
        return {}

    def _fetch_fundamentals_entrade(self, symbol: str) -> dict:
        """
        Fallback 4: Entrade/DNSE fundamental endpoint.
        https://services.entrade.com.vn/dnse-analysis-service/company/{symbol}/summary
        """
        try:
            import urllib.request, json
            url = f"https://services.entrade.com.vn/dnse-analysis-service/company/{symbol.upper()}/summary"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read().decode())

            # Tìm PE/PB/ROE trong nested dict
            def _dig(d, keys):
                if not isinstance(d, dict):
                    return None
                for k in keys:
                    if k in d:
                        try:
                            v = float(d[k])
                            if not pd.isna(v) and v != 0:
                                return v
                        except Exception:
                            pass
                # Recursive search một level
                for v in d.values():
                    if isinstance(v, dict):
                        r = _dig(v, keys)
                        if r is not None:
                            return r
                return None

            pe  = _dig(data, ["pe", "PE", "priceToEarning", "p_e"])
            pb  = _dig(data, ["pb", "PB", "priceToBook", "p_b"])
            roe = _dig(data, ["roe", "ROE", "returnOnEquity"])
            if roe is not None and abs(roe) < 2:
                roe = roe * 100
            if pe is not None or roe is not None:
                return {"pe": pe, "pb": pb, "roe": roe, "source": "Entrade"}
        except Exception:
            pass
        return {}

    def _score(self, fund: dict) -> tuple:
        """
        Áp dụng threshold filter theo HKUDS logic.
        Trả về (signal, detail_str).
        """
        if not fund:
            return 0, "Khong lay duoc fundamental data"

        pe  = fund.get("pe")
        pb  = fund.get("pb")
        roe = fund.get("roe")
        src = fund.get("source", "N/A")

        conditions = []
        passed = []
        failed = []

        # PE condition — HKUDS: pe_min < pe <= pe_max
        if pe is not None and not pd.isna(pe) and pe > 0:
            cond = self.pe_min < pe <= self.pe_max
            conditions.append(cond)
            label = f"PE={pe:.1f}({'OK' if cond else f'>{self.pe_max}'})"
            (passed if cond else failed).append(label)
        else:
            conditions.append(None)  # missing

        # PB condition
        if pb is not None and not pd.isna(pb) and pb > 0:
            cond = pb <= self.pb_max
            conditions.append(cond)
            label = f"PB={pb:.1f}({'OK' if cond else f'>{self.pb_max}'})"
            (passed if cond else failed).append(label)
        else:
            conditions.append(None)

        # ROE condition
        if roe is not None and not pd.isna(roe):
            cond = roe >= self.roe_min
            conditions.append(cond)
            label = f"ROE={roe:.1f}%({'OK' if cond else f'<{self.roe_min}%'})"
            (passed if cond else failed).append(label)
        else:
            conditions.append(None)

        # Loại None (missing data)
        valid = [c for c in conditions if c is not None]
        if len(valid) == 0:
            return 0, f"[{src}] Thieu het du lieu fundamental"
        if len(valid) < 2:
            return 0, f"[{src}] Chi co {len(valid)}/3 chi so — khong du de ket luan"

        # HKUDS logic: pass ALL conditions → BUY
        if all(c for c in valid if c is not None):
            sig = 1
        elif any(c is False for c in valid):
            # Fail bất kỳ condition nào → không đủ tiêu chuẩn → NEUTRAL
            # (Không phải BAN vì thiếu data có thể sai)
            sig = 0 if len(failed) == 1 else -1
        else:
            sig = 0

        parts = passed + [f"FAIL:{f}" for f in failed]
        det   = f"[HKUDS][{src}] " + " | ".join(parts) +                 f" | {'PASS(value ok)' if sig>0 else 'BORDERLINE' if sig==0 else 'FAIL(overvalued/weak)'}"
        return sig, det

    def _extract_from_df(self, df: pd.DataFrame) -> dict:
        """
        [HKUDS] Source gốc expect pe/pb/roe columns có sẵn trong DataFrame
        (extra_fields từ data provider như tushare).
        Nếu có → dùng trực tiếp, không cần fetch external.
        """
        cols_lower = {c.lower(): c for c in df.columns}
        def _get(keys):
            for k in keys:
                if k in cols_lower:
                    try:
                        v = float(df[cols_lower[k]].dropna().iloc[-1])
                        if not np.isnan(v) and v != 0:
                            return v
                    except Exception:
                        pass
            return None

        pe  = _get(["pe", "p_e", "pricetoearning", "pe_ttm"])
        pb  = _get(["pb", "p_b", "pricetobook"])
        roe = _get(["roe", "returnonequity"])
        if roe is not None and abs(roe) < 2:
            roe = roe * 100
        if pe is not None or roe is not None:
            return {"pe": pe, "pb": pb, "roe": roe, "source": "DataFrame.columns"}
        return {}

    def _one(self, df: pd.DataFrame, code: str) -> tuple:
        # [HKUDS] Ưu tiên 1: pe/pb/roe columns trong DataFrame (đúng source gốc)
        fund = self._extract_from_df(df)

        # Fallback: fetch external nếu không có columns
        if not fund:
            for fetch_fn in [
                self._fetch_fundamentals_vnstock,
                self._fetch_fundamentals_entrade,
                self._fetch_fundamentals_cafef,
                self._fetch_fundamentals_yfinance,
            ]:
                try:
                    fund = fetch_fn(code)
                except Exception:
                    fund = {}
                if fund and (fund.get("pe") is not None or fund.get("roe") is not None):
                    break

        sig_val, det = self._score(fund)
        sig = pd.Series(sig_val, index=df.index, dtype=int)
        return sig, det

    def generate(self, data_map: dict) -> tuple:
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df, code)
                signals[code] = _last(sig)
                details[code] = det
            except Exception as e:
                signals[code] = 0
                details[code] = f"Loi FundamentalFilterEngine: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCES — 14 engines đầy đủ
# ══════════════════════════════════════════════════════════════════════════════

_ENGINES = {
    # ── HKUDS verified — có source code gốc ───────────────────────────────────
    "Candlestick":    CandlestickEngine(),    # [HKUDS] 15 mô hình nến
    "Ichimoku":       IchimokuEngine(),       # [HKUDS] TK Cross + 3-filter
    "TechnicalBasic": TechnicalEngine(),      # [HKUDS] 3-dim voting EMA/ADX+BB/RSI+OBV
    "ElliottWave":    ElliottEngine(),        # [HKUDS] Zigzag + 5-wave + ABC + Fibonacci
    "Harmonic":       HarmonicEngine(),       # [HKUDS] XABCD + pyharmonics fallback
    "Volatility":     VolatilityEngine(),     # [HKUDS] HV percentile lookback=120
    "Seasonal":       SeasonalEngine(),       # [HKUDS] Fixed month lists
    "SMC":            SMCEngine(),            # [HKUDS] ChoCH priority + BOS + FVG
    "CrossMarket":    CrossMarketEngine(),    # [HKUDS] Vol-adjusted dual-MA a_share 5/20
    "MultiFactor":    MultiFactorEngine(),    # [HKUDS] 4-factor composite Z-score
    "MeanReversion":  MeanReversionEngine(),  # [HKUDS] Z-score entry=2.0 exit=0.5
    "MLStrategy":     MLStrategyEngine(),     # [HKUDS] RandomForest walk-forward SKILL.md
    "Chanlun":        ChanlunEngine(),         # [HKUDS] 缠中说禅 FX→BI→ZS→buy/sell points
    "FundamentalFilter": FundamentalFilterEngine(), # [HKUDS] PE/PB/ROE value screen
}



# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_vibe_agents(symbol: str, df: pd.DataFrame) -> dict:
    """
    Chạy tất cả 14 Vibe-Trading engines trên OHLCV DataFrame.

    Args:
        symbol : mã cổ phiếu (VD: "VCB")
        df     : OHLCV từ Entrade (cột time/trade_date, open, high, low, close, volume)

    Returns dict:
        signals : {engine_name: int}   — +1 bull / -1 bear / 0 neutral
        details : {engine_name: str}   — mô tả ngắn
        verdict : int                  — tổng hợp vote
        bull    : int                  — số engines bullish
        bear    : int                  — số engines bearish
        n       : int                  — tổng engines có signal
        summary : str
    """
    try:
        df_prep = _prep(df)
    except Exception as e:
        return {"error": str(e), "signals": {}, "verdict": 0, "bull": 0, "bear": 0, "n": 0, "summary": ""}

    if len(df_prep) < 20:
        return {"error": "Khong du du lieu (can >= 20 bars)", "signals": {}, "verdict": 0,
                "bull": 0, "bear": 0, "n": 0, "summary": ""}

    data_map = {symbol: df_prep}
    all_signals: Dict[str, int] = {}
    all_details: Dict[str, str] = {}

    for name, engine in _ENGINES.items():
        try:
            sigs, dets = engine.generate(data_map)
            all_signals[name] = sigs.get(symbol, 0)
            all_details[name] = dets.get(symbol, "N/A")
        except Exception as e:
            all_signals[name] = 0
            all_details[name] = f"Loi engine: {e}"

    bull = sum(1 for v in all_signals.values() if v > 0)
    bear = sum(1 for v in all_signals.values() if v < 0)
    n    = len(all_signals)
    verdict = (1 if bull > bear and bull > n * 0.4
               else -1 if bear > bull and bear > n * 0.4
               else 0)

    bull_names = [k for k,v in all_signals.items() if v > 0]
    bear_names = [k for k,v in all_signals.items() if v < 0]
    summary = (
        f"Vibe-Trading 16 engines: {bull}/{n} bullish, {bear}/{n} bearish. "
        f"Bull: {', '.join(bull_names) or 'none'}. "
        f"Bear: {', '.join(bear_names) or 'none'}."
    )

    return {
        "signals": all_signals,
        "details": all_details,
        "verdict": verdict,
        "bull":    bull,
        "bear":    bear,
        "n":       n,
        "summary": summary,
    }
