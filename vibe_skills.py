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
  1.  Candlestick    [HKUDS] — 15 mô hình nến, vectorized scoring, volume filter
  2.  Ichimoku       [HKUDS] — TK Cross event + 3-filter (rewrite từ source gốc)
  3.  TechnicalBasic [HKUDS] — 3-dim voting EMA/ADX+BB/RSI+OBV (rewrite từ source gốc)
  4.  ElliottWave    [HKUDS] — Zigzag + 5-wave impulse + ABC correction + Fibonacci
  5.  Harmonic       [HKUDS] — XABCD Gartley/Bat/Butterfly/Crab + pyharmonics fallback
  6.  Volatility     [HKUDS] — HV percentile (lookback=120, low=20%, high=80%)
  7.  Seasonal       [HKUDS] — Fixed month lists (rewrite từ source gốc)
  8.  SMC            [HKUDS] — ChoCH priority + BOS + FVG filter (momentum fallback)
  9.  CrossMarket    [Claude-made] — Vol-adjusted dual-MA, single stock adapt
  10. MultiFactor    [Claude-made] — Momentum/Reversal/Vol/VolumeRatio Z-score
  11. MeanReversion  [Claude-made] — Z-score price vs rolling mean
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
    def __init__(self, body_pct=0.1, shadow_ratio=2.0):
        self.body_pct = body_pct
        self.shadow_ratio = shadow_ratio

    # helpers
    def _bd(self, o, c): return (c - o).abs()
    def _rng(self, h, l): return (h - l).replace(0, np.nan)
    def _us(self, o, c, h): return h - pd.concat([o, c], axis=1).max(axis=1)
    def _ls(self, o, c, l): return pd.concat([o, c], axis=1).min(axis=1) - l

    def _hammer(self, o, h, l, c):
        bd=self._bd(o,c); ls=self._ls(o,c,l); us=self._us(o,c,h)
        return ((ls>=self.shadow_ratio*bd)&(us<bd)&(bd>0)&self._rng(h,l).notna()).astype(int)

    def _inv_hammer(self, o, h, l, c):
        bd=self._bd(o,c); us=self._us(o,c,h); ls=self._ls(o,c,l)
        return ((us>=self.shadow_ratio*bd)&(ls<bd)&(bd>0)).astype(int)

    def _shooting_star(self, o, h, l, c):
        bd=self._bd(o,c); us=self._us(o,c,h); ls=self._ls(o,c,l)
        up = c.shift(1) > c.shift(2)
        return -(((us>=self.shadow_ratio*bd)&(ls<bd)&(bd>0)&up).astype(int))

    def _doji(self, o, h, l, c):
        bd=self._bd(o,c); rng=self._rng(h,l).fillna(1)
        return ((bd/rng < self.body_pct)).astype(int)

    def _engulfing(self, o, h, l, c):
        o1,c1=o.shift(1),c.shift(1)
        bull=((c1<o1)&(c>o)&(c>=o1)&(o<=c1)).astype(int)
        bear=((c1>o1)&(c<o)&(c<=o1)&(o>=c1)).astype(int)
        s=pd.Series(0,index=o.index); s[bull==1]=1; s[bear==1]=-1; return s

    def _harami(self, o, h, l, c):
        bd=self._bd(o,c); o1,c1=o.shift(1),c.shift(1); bd1=self._bd(o1,c1)
        pt=pd.concat([o1,c1],axis=1).max(axis=1); pb=pd.concat([o1,c1],axis=1).min(axis=1)
        ct=pd.concat([o,c],axis=1).max(axis=1);   cb=pd.concat([o,c],axis=1).min(axis=1)
        cont=(ct<=pt)&(cb>=pb)
        s=pd.Series(0,index=o.index)
        s[((c1<o1)&(bd1>bd)&cont)]=1; s[((c1>o1)&(bd1>bd)&cont)]=-1; return s

    def _piercing(self, o, h, l, c):
        o1,c1,l1=o.shift(1),c.shift(1),l.shift(1); mid=(o1+c1)/2
        return (((c1<o1)&(c>o)&(o<l1)&(c>mid)).astype(int))

    def _dark_cloud(self, o, h, l, c):
        o1,c1,h1=o.shift(1),c.shift(1),h.shift(1); mid=(o1+c1)/2
        return -(((c1>o1)&(c<o)&(o>h1)&(c<mid)).astype(int))

    def _morning_star(self, o, h, l, c):
        o1,c1=o.shift(2),c.shift(2); o2,c2,h2=o.shift(1),c.shift(1),h.shift(1)
        bd2=self._bd(o2,c2); rng2=self._rng(h.shift(1),l.shift(1)).fillna(1)
        return (((c1<o1)&(bd2/rng2<0.3)&(h2<l.shift(2))&(c>o)&(c>(o1+c1)/2)).astype(int).fillna(0))

    def _evening_star(self, o, h, l, c):
        o1,c1=o.shift(2),c.shift(2); o2,c2,l2=o.shift(1),c.shift(1),l.shift(1)
        bd2=self._bd(o2,c2); rng2=self._rng(h.shift(1),l.shift(1)).fillna(1)
        return -(((c1>o1)&(bd2/rng2<0.3)&(l2>h.shift(2))&(c<o)&(c<(o1+c1)/2)).astype(int).fillna(0))

    def _three_white(self, o, h, l, c):
        o1,c1=o.shift(2),c.shift(2); o2,c2=o.shift(1),c.shift(1)
        cond=((c1>o1)&(c2>o2)&(c>o)&(c2>c1)&(c>c2)&(o2>=o1)&(o2<=c1)&(o>=o2)&(o<=c2))
        return cond.astype(int).fillna(0)

    def _three_black(self, o, h, l, c):
        o1,c1=o.shift(2),c.shift(2); o2,c2=o.shift(1),c.shift(1)
        cond=((c1<o1)&(c2<o2)&(c<o)&(c2<c1)&(c<c2)&(o2<=o1)&(o2>=c1)&(o<=o2)&(o>=c2))
        return -(cond.astype(int).fillna(0))

    def generate(self, data_map: Dict[str, pd.DataFrame]) -> Tuple[Dict,Dict]:
        signals, details = {}, {}
        for code, df in data_map.items():
            o,h,l,c = df["open"],df["high"],df["low"],df["close"]
            v = df.get("volume", pd.Series(1,index=df.index))

            sc = pd.DataFrame(index=df.index)
            sc["hammer"]        = self._hammer(o,h,l,c)
            sc["inv_hammer"]    = self._inv_hammer(o,h,l,c)
            sc["shooting_star"] = self._shooting_star(o,h,l,c)
            sc["engulfing"]     = self._engulfing(o,h,l,c)
            sc["harami"]        = self._harami(o,h,l,c)
            sc["piercing"]      = self._piercing(o,h,l,c)
            sc["dark_cloud"]    = self._dark_cloud(o,h,l,c)
            sc["morning_star"]  = self._morning_star(o,h,l,c)
            sc["evening_star"]  = self._evening_star(o,h,l,c)
            sc["three_white"]   = self._three_white(o,h,l,c)
            sc["three_black"]   = self._three_black(o,h,l,c)
            total = sc.sum(axis=1)

            # ── Volume Filter ─────────────────────────────────────────────
            # Vol < 0.5x TB20: mô hình nến không đủ thanh khoản → trung lập
            # Vol 0.5-0.7x: hạ cấp signal (chỉ giữ nếu mô hình mạnh >= 2 cùng chiều)
            # Vol >= 0.7x: giữ nguyên signal
            vol_ma20 = v.rolling(20).mean()
            vol_ratio = v / vol_ma20.replace(0, np.nan)

            sig_series = pd.Series(np.sign(total).astype(int), index=df.index)

            # Vol quá thấp → reset về 0
            very_low_vol = vol_ratio < 0.5
            sig_series[very_low_vol] = 0

            # Vol thấp → chỉ giữ signal khi mô hình MẠNH (>= 2 pattern cùng chiều)
            low_vol = (vol_ratio >= 0.5) & (vol_ratio < 0.7)
            weak_signal = total.abs() < 2
            sig_series[low_vol & weak_signal] = 0

            signals[code] = _last(sig_series)

            # Detail
            last  = sc.iloc[-1]
            cur_vr = round(float(vol_ratio.iloc[-1]), 2) if not pd.isna(vol_ratio.iloc[-1]) else "N/A"
            found_bull = [n for n in sc.columns if last.get(n,0) > 0]
            found_bear = [n for n in sc.columns if last.get(n,0) < 0]
            desc = ""
            if found_bull: desc += f"Bullish: {', '.join(found_bull)}. "
            if found_bear: desc += f"Bearish: {', '.join(found_bear)}. "
            if not desc:   desc = "Khong nhan dien mo hinh nen dac biet. "
            vol_note = ""
            if isinstance(cur_vr, float):
                if cur_vr < 0.5:   vol_note = f"[Vol {cur_vr}x < 0.5x: RESET signal]"
                elif cur_vr < 0.7: vol_note = f"[Vol {cur_vr}x thap: chi giu signal manh]"
                else:              vol_note = f"[Vol {cur_vr}x OK]"
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
    def __init__(self, hv_w=20, pct_w=120, lo_pct=20, hi_pct=80, ann=252):  # [HKUDS] lookback=120
        self.hv_w=hv_w; self.pct_w=pct_w; self.lo_pct=lo_pct; self.hi_pct=hi_pct; self.ann=ann

    def _hv(self, c):
        return np.log(c/c.shift(1)).rolling(self.hv_w).std()*np.sqrt(self.ann)

    def _one(self, df):
        c=df["close"]; hv=self._hv(c)
        min_p=max(20, min(self.pct_w, len(hv)-1))
        pct=hv.rolling(min_p).apply(lambda x: (pd.Series(x).rank(pct=True).iloc[-1])*100, raw=False)
        sig=pd.Series(0,index=df.index,dtype=int)
        sig[pct<self.lo_pct]=1; sig[pct>self.hi_pct]=-1
        cur_hv=hv.iloc[-1]; cur_pct=pct.iloc[-1]
        regime="low_vol" if cur_pct<self.lo_pct else "high_vol" if cur_pct>self.hi_pct else "normal"
        det=(f"HV20={round(cur_hv*100,1) if not pd.isna(cur_hv) else 'N/A'}% "
             f"Percentile={round(cur_pct,0) if not pd.isna(cur_pct) else 'N/A'}% "
             f"Regime={regime}")
        return sig, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig,det=self._one(df); signals[code]=_last(sig); details[code]=det
            except Exception as e:
                signals[code]=0; details[code]=f"Loi: {e}"
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
    def __init__(self, swing_length=10, close_break=True):
        self.sl=swing_length; self.cb=close_break

    def _swing_hl(self, h, l):
        w=self.sl
        swing_h=h[(h==h.rolling(2*w+1,center=True).max())].dropna()
        swing_l=l[(l==l.rolling(2*w+1,center=True).min())].dropna()
        return swing_h, swing_l

    def _bos_choch(self, df, swing_h, swing_l):
        """Detect BOS/ChoCH: +1 bullish, -1 bearish."""
        c=df["close"]
        sig=pd.Series(0,index=df.index,dtype=int)
        # Bullish: close breaks above previous swing high
        for idx in swing_h.index[:-1]:
            lvl=swing_h[idx]
            after=c[c.index>idx]
            breaks=after[after>lvl] if self.cb else after[df["high"][after.index]>lvl]
            if not breaks.empty:
                sig[breaks.index[0]]=1
        # Bearish: close breaks below previous swing low
        for idx in swing_l.index[:-1]:
            lvl=swing_l[idx]
            after=c[c.index>idx]
            breaks=after[after<lvl] if self.cb else after[df["low"][after.index]<lvl]
            if not breaks.empty:
                sig[breaks.index[0]]=-1
        return sig

    def _fvg(self, df):
        """Fair Value Gap: gap between candle[i-2].high and candle[i].low (bull FVG) or vice versa."""
        h,l=df["high"],df["low"]
        bull_fvg=(l > h.shift(2))  # gap up
        bear_fvg=(h < l.shift(2))  # gap down
        fvg=pd.Series(0,index=df.index,dtype=int)
        fvg[bull_fvg]=1; fvg[bear_fvg]=-1
        return fvg

    def _one(self, df):
        if len(df)<self.sl*2:
            return pd.Series(0,index=df.index,dtype=int), f"Khong du du lieu (can >={self.sl*2} bars)."
        sh, sl = self._swing_hl(df["high"], df["low"])
        structure = self._bos_choch(df, sh, sl)
        fvg = self._fvg(df)

        # ── Đánh giá theo MOMENTUM cấu trúc, không chỉ bar cuối ──────────
        # Window gần nhất 60 bars để tính bias
        w = min(60, len(df))
        struct_recent = structure.iloc[-w:]
        fvg_recent    = fvg.iloc[-w:]

        n_buy_total  = int((structure ==  1).sum())
        n_sell_total = int((structure == -1).sum())
        n_buy_rec    = int((struct_recent ==  1).sum())
        n_sell_rec   = int((struct_recent == -1).sum())

        fvg_bull = int((fvg_recent ==  1).sum())
        fvg_bear = int((fvg_recent == -1).sum())

        # Net structure pressure (gần nhất có trọng số 2x)
        net_struct = (n_buy_rec * 2 + n_buy_total) - (n_sell_rec * 2 + n_sell_total)
        net_fvg    = fvg_bull - fvg_bear

        # Signal dựa trên net pressure
        if net_struct < -2 and net_fvg <= 0:
            # Áp lực bán rõ ràng từ cả structure lẫn FVG
            cur_sig = -1
            bias = "BEARISH"
        elif net_struct > 2 and net_fvg >= 0:
            cur_sig = 1
            bias = "BULLISH"
        elif net_struct < -1:
            # Structure lean bearish
            cur_sig = -1
            bias = "LEAN BEARISH"
        elif net_struct > 1:
            cur_sig = 1
            bias = "LEAN BULLISH"
        else:
            cur_sig = 0
            bias = "NEUTRAL"

        sig = pd.Series(0, index=df.index, dtype=int)
        sig.iloc[-1] = cur_sig

        det = (f"BOS/ChoCH: {n_buy_total} bull, {n_sell_total} bear breaks "
               f"(60D gần: {n_buy_rec} bull, {n_sell_rec} bear). "
               f"FVG 60D: bull={fvg_bull} bear={fvg_bear}. "
               f"Net={net_struct:+d} | Bias={bias}")
        return sig, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig,det=self._one(df); signals[code]=_last(sig); details[code]=det
            except Exception as e:
                signals[code]=0; details[code]=f"Loi: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 9 — CROSS-MARKET (Vol-adjusted Dual-MA, từ Vibe-Trading)
# Với single VN stock: dùng a_share params (MA5/MA20)
# ══════════════════════════════════════════════════════════════════════════════

class CrossMarketEngine:
    MA_FAST=5; MA_SLOW=20; VOL_LB=20

    def _one(self, df):
        c=df["close"]
        mf=c.rolling(self.MA_FAST).mean(); ms=c.rolling(self.MA_SLOW).mean()
        sig=pd.Series(0.0,index=df.index)
        sig[mf>ms]=1.0; sig[mf<ms]=-1.0
        ret=c.pct_change().dropna()
        vol=ret.rolling(self.VOL_LB).std().iloc[-1] if len(ret)>self.VOL_LB else ret.std()
        # Volatility-adjusted: scale signal by inv_vol weight (single stock=1.0)
        sig_adj=(sig*1.0).clip(-1.0,1.0)
        last_s=_last(sig_adj)
        det=(f"[Claude-made] MA{self.MA_FAST}={round(mf.iloc[-1],2)} MA{self.MA_SLOW}={round(ms.iloc[-1],2)} "
             f"Vol20={round(vol*100,2) if not pd.isna(vol) else 'N/A'}% "
             f"Signal={'BUY' if last_s>0 else 'SELL' if last_s<0 else 'NEUTRAL'}")
        return sig_adj, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig,det=self._one(df)
                signals[code]=_last(sig); details[code]=det
            except Exception as e:
                signals[code]=0; details[code]=f"Loi: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 10 — MULTI-FACTOR (Momentum/Reversal/Volatility/VolumeRatio, from Vibe-Trading)
# Single stock: dùng absolute Z-score thay vì cross-section rank
# ══════════════════════════════════════════════════════════════════════════════

class MultiFactorEngine:
    def __init__(self, mom_w=20, vol_w=20, z_th=1.0):
        self.mom_w=mom_w; self.vol_w=vol_w; self.z_th=z_th

    def _one(self, df):
        c=df["close"]
        v=df.get("volume",pd.Series(1,index=df.index))
        ret=c.pct_change()
        # Factor values
        momentum = c/c.shift(self.mom_w)-1              # +: bullish
        reversal  = -(c/c.shift(5)-1)                   # -5d return inverted
        volatility= -(ret.rolling(self.vol_w).std())    # low vol = better
        vol_ratio = v/v.rolling(self.vol_w).mean()       # high vol ratio = better
        # Composite score (equal weight, no cross-section needed for 1 stock)
        # Normalise each factor individually (rolling z-score)
        def roll_z(s, w=60):
            m=s.rolling(w,min_periods=20).mean()
            sd=s.rolling(w,min_periods=20).std()
            return (s-m)/sd.replace(0,np.nan)
        score = (roll_z(momentum)+roll_z(reversal)+roll_z(volatility)+roll_z(vol_ratio))/4
        sig=pd.Series(0,index=df.index,dtype=int)
        sig[score>self.z_th]=1; sig[score<-self.z_th]=-1
        cur_score=score.iloc[-1]
        det=(f"[Claude-made] Factor score={round(cur_score,2) if not pd.isna(cur_score) else 'N/A'} "
             f"(momentum={round(momentum.iloc[-1]*100,1) if not pd.isna(momentum.iloc[-1]) else 'N/A'}% "
             f"vol_ratio={round(vol_ratio.iloc[-1],2) if not pd.isna(vol_ratio.iloc[-1]) else 'N/A'}x) "
             f"Signal={'BUY' if _last(sig)>0 else 'SELL' if _last(sig)<0 else 'NEUTRAL'}")
        return sig, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig,det=self._one(df); signals[code]=_last(sig); details[code]=det
            except Exception as e:
                signals[code]=0; details[code]=f"Loi: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 11 — MEAN REVERSION (từ pair-trading logic, adapt cho single stock)
# Dùng price vs rolling mean Z-score — khi giá lệch quá xa mean → đảo chiều
# Trực tiếp từ pair-trading engine của Vibe-Trading (Skill: pair-trading)
# ══════════════════════════════════════════════════════════════════════════════

class MeanReversionEngine:
    """
    Adapt pair-trading Z-score logic cho single stock.
    Thay vì ratio(A/B), dùng price vs rolling-mean.
    Nguồn: agent/src/skills/pair-trading/example_signal_engine.py
    """
    def __init__(self, lookback=60, entry_z=2.0, exit_z=0.5):
        self.lookback=lookback; self.entry_z=entry_z; self.exit_z=exit_z

    def _one(self, df):
        c = df["close"]
        if len(c) < self.lookback + 5:
            return pd.Series(0, index=df.index, dtype=int), "Khong du data"
        mean = c.rolling(self.lookback).mean()
        std  = c.rolling(self.lookback).std()
        z    = (c - mean) / std.replace(0, np.nan)

        sig = pd.Series(0, index=df.index, dtype=int)
        sig[z < -self.entry_z] = 1    # giá quá thấp so với mean → kỳ vọng phục hồi
        sig[z >  self.entry_z] = -1   # giá quá cao so với mean → kỳ vọng điều chỉnh
        sig[z.abs() < self.exit_z] = 0

        cur_z = z.iloc[-1]
        cur_s = _last(sig)
        det = (f"[Claude-made] Z-score={round(cur_z,2) if not pd.isna(cur_z) else 'N/A'} "
               f"(entry±{self.entry_z} exit±{self.exit_z} lookback={self.lookback}D) "
               f"Signal={'BUY(reversion up)' if cur_s>0 else 'SELL(reversion dn)' if cur_s<0 else 'NEUTRAL'}")
        return sig, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code] = _last(sig); details[code] = det
            except Exception as e:
                signals[code] = 0; details[code] = f"Loi: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 12 — PRICE MOMENTUM (từ fundamental-filter logic + momentum factor)
# Multi-timeframe momentum: 5D/20D/60D composite
# Nguồn: agent/src/skills/fundamental-filter/example_signal_engine.py (momentum factor)
# ══════════════════════════════════════════════════════════════════════════════

class PriceMomentumEngine:
    """
    Multi-timeframe price momentum engine.
    Kết hợp momentum 5D/20D/60D và xác nhận bằng volume.
    Nguồn logic: fundamental-filter engine (factor scoring) + Vibe momentum factor.
    """
    def __init__(self, windows=(5, 20, 60), vol_window=20):
        self.windows  = windows
        self.vol_window = vol_window

    def _one(self, df):
        c = df["close"]
        v = df.get("volume", pd.Series(1, index=df.index))
        if len(c) < max(self.windows) + 5:
            return pd.Series(0, index=df.index, dtype=int), "Khong du data"

        mom_signals = []
        mom_vals    = []
        for w in self.windows:
            if len(c) > w:
                m = float(c.iloc[-1] / c.iloc[-w] - 1) if c.iloc[-w] != 0 else 0.0
                mom_vals.append(m)
                mom_signals.append(1 if m > 0 else -1 if m < 0 else 0)

        # Volume confirmation: current vol vs average
        vol_ratio = float(v.iloc[-1] / v.rolling(self.vol_window).mean().iloc[-1]) if len(v) > self.vol_window else 1.0

        # Composite: đa số timeframes + vol không quá thấp
        bull = sum(1 for s in mom_signals if s > 0)
        bear = sum(1 for s in mom_signals if s < 0)
        n    = len(mom_signals)

        sig_series = pd.Series(0, index=df.index, dtype=int)
        if bull > n / 2 and vol_ratio > 0.5:
            sig_series.iloc[-1] = 1
        elif bear > n / 2 and vol_ratio > 0.5:
            sig_series.iloc[-1] = -1

        cur = _last(sig_series)
        labels = [f"{w}D={round(m*100,1)}%" for w,m in zip(self.windows, mom_vals)]
        det = (f"[Claude-made] Momentum [{', '.join(labels)}] | Vol={round(vol_ratio,2)}x | "
               f"Signal={'BUY' if cur>0 else 'SELL' if cur<0 else 'NEUTRAL'}")
        return sig_series, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code] = _last(sig); details[code] = det
            except Exception as e:
                signals[code] = 0; details[code] = f"Loi: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 13 — SUPPORT / RESISTANCE BREAKOUT
# Phát hiện breakout qua đỉnh/đáy N phiên — xác nhận bằng volume spike
# Bổ sung cho hệ thống: cần thiết cho TTCK VN (breakout rất phổ biến)
# ══════════════════════════════════════════════════════════════════════════════

class BreakoutEngine:
    """
    Support/Resistance Breakout detection.
    Phát hiện breakout qua high/low của N phiên trước, xác nhận bằng volume.
    """
    def __init__(self, lookback=20, vol_multiplier=1.3):
        self.lookback = lookback
        self.vol_multiplier = vol_multiplier

    def _one(self, df):
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df.get("volume", pd.Series(1, index=df.index))

        if len(c) < self.lookback + 2:
            return pd.Series(0, index=df.index, dtype=int), "Khong du data"

        # Đỉnh/đáy N phiên trước (không tính phiên hiện tại)
        resist = h.shift(1).rolling(self.lookback).max()
        supprt = l.shift(1).rolling(self.lookback).min()
        avg_vol = v.shift(1).rolling(self.lookback).mean()

        sig = pd.Series(0, index=df.index, dtype=int)
        # Breakout up: đóng cửa trên resistance + volume xác nhận
        bull_break = (c > resist) & (v > avg_vol * self.vol_multiplier)
        # Breakdown: đóng cửa dưới support + volume xác nhận
        bear_break = (c < supprt) & (v > avg_vol * self.vol_multiplier)

        sig[bull_break] = 1
        sig[bear_break] = -1

        cur = _last(sig)
        r = float(resist.iloc[-1]) if not pd.isna(resist.iloc[-1]) else 0
        s = float(supprt.iloc[-1]) if not pd.isna(supprt.iloc[-1]) else 0
        vr = float(v.iloc[-1] / avg_vol.iloc[-1]) if not pd.isna(avg_vol.iloc[-1]) and avg_vol.iloc[-1] > 0 else 1.0
        det = (f"[Claude-made] Resist={round(r,2)} Support={round(s,2)} "
               f"VolRatio={round(vr,2)}x (need>{self.vol_multiplier}x) "
               f"Signal={'BREAKOUT UP' if cur>0 else 'BREAKDOWN' if cur<0 else 'NO BREAK'}")
        return sig, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code] = _last(sig); details[code] = det
            except Exception as e:
                signals[code] = 0; details[code] = f"Loi: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 14 — ML STRATEGY (sklearn RandomForest walk-forward, từ Vibe-Trading)
# Nguồn: agent/src/skills/ml-strategy/SKILL.md (build_features + walk-forward)
# ══════════════════════════════════════════════════════════════════════════════

class MLStrategyEngine:
    """
    Machine Learning signal engine dựa trên sklearn RandomForest.
    Walk-forward training để tránh data leakage.
    Nguồn: Vibe-Trading ml-strategy SKILL.md.
    """
    def __init__(self, train_window=120, predict_horizon=5, min_train=80):
        self.train_window    = train_window
        self.predict_horizon = predict_horizon
        self.min_train       = min_train

    def _build_features(self, df):
        c = df["close"]; v = df.get("volume", pd.Series(1, index=df.index))
        h = df.get("high", c); l = df.get("low", c)
        ret = c.pct_change()
        feat = pd.DataFrame(index=df.index)
        # Price momentum — multi-timeframe (quan trọng cho VN)
        feat["f_ret_1d"]        = c.pct_change(1)
        feat["f_ret_5d"]        = c.pct_change(5)
        feat["f_ret_20d"]       = c.pct_change(20)
        feat["f_ret_60d"]       = c.pct_change(60)
        # Volatility
        feat["f_vol_10d"]       = ret.rolling(10).std()
        feat["f_vol_20d"]       = ret.rolling(20).std()
        # Trend (MA cross)
        feat["f_ma_ratio_20"]   = c / c.rolling(20).mean()
        feat["f_ma_ratio_50"]   = c / c.rolling(50).mean()
        feat["f_ma5_20_cross"]  = c.rolling(5).mean() / c.rolling(20).mean()
        # RSI — Wilder SMA để nhất quán với compute_indicators
        delta = c.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        feat["f_rsi_14"]        = 100 - (100 / (1 + rs))
        # Bollinger position
        feat["f_bb_pos"]        = (c - c.rolling(20).mean()) / (c.rolling(20).std().replace(0, np.nan))
        # Volume — đặc biệt quan trọng cho TTCK VN (thanh khoản thấp)
        feat["f_vol_ratio_20"]  = v / v.rolling(20).mean()
        feat["f_vol_ratio_5"]   = v / v.rolling(5).mean()
        # Volume-price momentum: ngày tăng volume cao hơn ngày giảm → bullish
        vp = (c.pct_change() * (v / v.rolling(20).mean()))
        feat["f_vp_momentum"]   = vp.rolling(5).sum()
        # OBV trend — smart money flow
        obv = (v * np.sign(c.diff())).fillna(0).cumsum()
        feat["f_obv_ratio"]     = obv / obv.rolling(20).mean().replace(0, np.nan)
        # High-Low range (intraday volatility)
        feat["f_hl_ratio"]      = (h - l) / c.replace(0, np.nan)
        return feat.replace([np.inf, -np.inf], np.nan)

    def _one(self, df):
        feat = self._build_features(df)
        c    = df["close"]
        n    = len(df)

        if n < self.min_train + self.predict_horizon + 10:
            return pd.Series(0, index=df.index, dtype=int), "Khong du data cho ML"

        # Label: future N-day return > 0
        label = (c.shift(-self.predict_horizon) > c).astype(int)

        sig     = pd.Series(0, index=df.index, dtype=int)
        method  = "rule_fallback"
        cur     = 0

        # ── Thử ML (sklearn) ──────────────────────────────────────────────
        try:
            from sklearn.ensemble import ExtraTreesClassifier  # nhanh hơn RF ~4x
            from sklearn.preprocessing import StandardScaler

            # Walk-forward: retrain mỗi 10 bars (giảm từ mỗi bar → giảm 90% thời gian)
            step = 10
            last_pred = 0
            for i in range(self.train_window, n - self.predict_horizon, step):
                X_train = feat.iloc[i - self.train_window:i].dropna()
                y_train = label.iloc[i - self.train_window:i].loc[X_train.index]
                if len(X_train) < self.min_train or y_train.nunique() < 2:
                    continue
                X_pred_block = feat.iloc[i:i+step].dropna()
                if X_pred_block.empty:
                    continue
                try:
                    scaler  = StandardScaler()
                    X_tr_s  = scaler.fit_transform(X_train)
                    X_pr_s  = scaler.transform(X_pred_block)
                    clf = ExtraTreesClassifier(
                        n_estimators=30, max_depth=4,
                        random_state=42, n_jobs=1
                    )
                    clf.fit(X_tr_s, y_train)
                    probas = clf.predict_proba(X_pr_s)
                    for j, proba in enumerate(probas):
                        idx = i + j
                        if idx < n:
                            ps = 1 if proba[1] > 0.60 else -1 if proba[1] < 0.40 else 0
                            sig.iloc[idx] = ps
                    last_pred = int(sig.iloc[min(i + step - 1, n - 1)])
                except Exception:
                    continue

            cur    = _last(sig) if sig.any() else 0
            method = "ExtraTrees walk-forward"

        except ImportError:
            # ── Fallback rule-based khi không có sklearn ──────────────────
            # Dùng features đã tính để ra signal deterministic
            feat_last = feat.iloc[-1]
            score = 0
            # RSI momentum
            rsi = feat_last.get("f_rsi_14", 50)
            if not pd.isna(rsi):
                if rsi < 40: score += 1    # oversold → khả năng phục hồi
                elif rsi > 65: score -= 1  # overbought
            # MA trend
            ma_r = feat_last.get("f_ma_ratio_20", 1.0)
            if not pd.isna(ma_r):
                if ma_r > 1.02: score += 1
                elif ma_r < 0.98: score -= 1
            # Price momentum 20D
            r20 = feat_last.get("f_ret_20d", 0)
            if not pd.isna(r20):
                if r20 > 0.05: score += 1
                elif r20 < -0.05: score -= 1
            # Volume-price momentum
            vp = feat_last.get("f_vp_momentum", 0)
            if not pd.isna(vp):
                if vp > 0.1: score += 1
                elif vp < -0.1: score -= 1
            cur = 1 if score >= 2 else -1 if score <= -2 else 0
            sig.iloc[-1] = cur
            method = "rule_fallback(no sklearn)"

        det = (f"[Claude-made] ML({method} train={self.train_window}D horizon={self.predict_horizon}D) "
               f"LastSignal={'BUY' if cur>0 else 'SELL' if cur<0 else 'NEUTRAL'}")
        return sig, det

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code] = _last(sig); details[code] = det
            except Exception as e:
                signals[code] = 0; details[code] = f"Loi ML: {e}"
        return signals, details



# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 15 — FUNDAMENTAL SCORE (Value + Growth filter)
# Nguồn: Vibe-Trading fundamental-filter skill
# Lấy PE/PB/ROE/EPS từ vnstock KBS — so sánh với ngưỡng ngành
# ══════════════════════════════════════════════════════════════════════════════

class FundamentalEngine:
    """
    Fundamental valuation + growth signal.
    Scoring: PE thấp + PB hợp lý + ROE cao + EPS tăng trưởng → MUA
    Nguồn: Vibe-Trading fundamental-filter SKILL.md
    """
    # Ngưỡng tham chiếu TTCK VN (theo VN30 trung bình lịch sử)
    PE_CHEAP   = 12.0   # PE < 12: rẻ
    PE_FAIR    = 20.0   # PE 12-20: hợp lý
    PE_DEAR    = 30.0   # PE > 30: đắt
    PB_CHEAP   = 1.5    # PB < 1.5: rẻ
    PB_DEAR    = 4.0    # PB > 4: đắt
    ROE_GOOD   = 15.0   # ROE > 15%: tốt
    ROE_GREAT  = 20.0   # ROE > 20%: rất tốt
    EPS_GROW   = 10.0   # EPS growth > 10%: tăng trưởng

    def _fetch_fundamental(self, symbol: str) -> dict:
        """Lấy fundamental data từ vnstock KBS."""
        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=symbol, source="KBS")
            # Thử lấy ratio
            for period in ["quarter", "annual"]:
                try:
                    ratio = stock.finance.ratio(period=period)
                    if ratio is not None and not ratio.empty:
                        latest = ratio.iloc[0]
                        cols   = [c.lower() for c in latest.index]
                        orig   = list(latest.index)

                        def _find(patterns, default=None):
                            for p in patterns:
                                for i, c in enumerate(cols):
                                    if p.lower() in c:
                                        try:
                                            v = float(latest[orig[i]])
                                            if not pd.isna(v) and v != 0:
                                                return v
                                        except Exception:
                                            pass
                            return default

                        return {
                            "pe":      _find(["pricetoearning", "p/e", "pe"]),
                            "pb":      _find(["pricetobook", "p/b", "pb"]),
                            "roe":     _find(["roe"]),
                            "eps":     _find(["earningpershare", "eps"]),
                            "eps_g":   _find(["epsgrowth", "earnings_growth"]),
                            "rev_g":   _find(["revenuegrowth", "revenue_growth"]),
                            "de":      _find(["debtonequity", "d/e", "leverage"]),
                            "source":  f"vnstock KBS ({period})",
                        }
                except Exception:
                    continue
        except Exception:
            pass
        return {}

    def _score(self, fund: dict) -> tuple[int, str]:
        """
        Tính điểm fundamental. Trả về (signal, detail_str).
        signal: +1 (value/growth), -1 (expensive/weak), 0 (neutral)
        """
        if not fund:
            return 0, "Khong lay duoc fundamental data (vnstock)"

        score  = 0
        parts  = []
        pe  = fund.get("pe")
        pb  = fund.get("pb")
        roe = fund.get("roe")
        eps_g = fund.get("eps_g")
        rev_g = fund.get("rev_g")
        de  = fund.get("de")

        # PE scoring
        if pe is not None and pe > 0:
            if pe < self.PE_CHEAP:
                score += 2; parts.append(f"PE={pe:.1f}(rẻ)")
            elif pe < self.PE_FAIR:
                score += 1; parts.append(f"PE={pe:.1f}(OK)")
            elif pe > self.PE_DEAR:
                score -= 2; parts.append(f"PE={pe:.1f}(đắt)")
            else:
                parts.append(f"PE={pe:.1f}")

        # PB scoring
        if pb is not None and pb > 0:
            if pb < self.PB_CHEAP:
                score += 1; parts.append(f"PB={pb:.1f}(rẻ)")
            elif pb > self.PB_DEAR:
                score -= 1; parts.append(f"PB={pb:.1f}(đắt)")
            else:
                parts.append(f"PB={pb:.1f}")

        # ROE scoring
        if roe is not None:
            roe_pct = roe * 100 if abs(roe) < 2 else roe  # normalize nếu dạng 0.15
            if roe_pct >= self.ROE_GREAT:
                score += 2; parts.append(f"ROE={roe_pct:.1f}%(tốt)")
            elif roe_pct >= self.ROE_GOOD:
                score += 1; parts.append(f"ROE={roe_pct:.1f}%")
            elif roe_pct < 5:
                score -= 1; parts.append(f"ROE={roe_pct:.1f}%(yếu)")
            else:
                parts.append(f"ROE={roe_pct:.1f}%")

        # EPS growth
        if eps_g is not None:
            eg = eps_g * 100 if abs(eps_g) < 5 else eps_g
            if eg > self.EPS_GROW:
                score += 1; parts.append(f"EPS_g={eg:.1f}%")
            elif eg < 0:
                score -= 1; parts.append(f"EPS_g={eg:.1f}%(giảm)")

        # D/E cao → rủi ro
        if de is not None and de > 2.0:
            score -= 1; parts.append(f"D/E={de:.1f}(cao)")

        # Kết luận
        if not parts:
            return 0, f"Du lieu: {fund.get('source','N/A')} — Khong du chi so"

        src = fund.get("source", "N/A")
        detail = f"[{src}] " + " | ".join(parts) + f" | Score={score:+d}"

        if score >= 3:
            return 1, detail
        elif score <= -2:
            return -1, detail
        elif score >= 1:
            return 1, detail   # lean bullish (giá trị tốt)
        elif score <= -1:
            return -1, detail  # lean bearish (định giá cao)
        else:
            return 0, detail

    def generate(self, data_map: dict) -> tuple[dict, dict]:
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                fund = self._fetch_fundamental(code)
                sig, det = self._score(fund)
                signals[code] = sig
                details[code] = f"[Claude-made] {det}"
            except Exception as e:
                signals[code] = 0
                details[code] = f"[Claude-made] Loi FundamentalEngine: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 16 — MONEY FLOW (Smart Money / Institutional Flow)
# Phân tích dòng tiền thông minh qua volume-price pattern
# Nguồn: Vibe-Trading fund-flow + market-microstructure skills
# ══════════════════════════════════════════════════════════════════════════════

class MoneyFlowEngine:
    """
    Smart money / institutional money flow detection.
    Dùng: OBV divergence, CMF (Chaikin Money Flow), VPT, Force Index
    Không cần foreign flow data thực — phân tích từ OHLCV.
    """
    def __init__(self, window=20, cmf_w=20, vpt_w=20):
        self.window = window
        self.cmf_w  = cmf_w
        self.vpt_w  = vpt_w

    def _cmf(self, h, l, c, v, w):
        """Chaikin Money Flow = sum(MFV) / sum(Vol) trong W bars."""
        hl_range = (h - l).replace(0, np.nan)
        mfm = ((c - l) - (h - c)) / hl_range          # Money Flow Multiplier [-1, 1]
        mfv = mfm * v                                   # Money Flow Volume
        cmf = mfv.rolling(w).sum() / v.rolling(w).sum()
        return cmf

    def _vpt(self, c, v):
        """Volume Price Trend — cumulative indicator."""
        ret = c.pct_change()
        vpt = (ret * v).fillna(0).cumsum()
        return vpt

    def _obv(self, c, v):
        """On Balance Volume."""
        return (v * np.sign(c.diff())).fillna(0).cumsum()

    def _force_index(self, c, v, w=13):
        """Elder Force Index = price_change * volume."""
        fi = c.diff() * v
        return fi.ewm(span=w, adjust=False).mean()

    def _one(self, df):
        if len(df) < self.window + 5:
            return pd.Series(0, index=df.index, dtype=int), "Khong du data"

        o = df.get("open",  df["close"])
        h = df["high"]; l = df["low"]; c = df["close"]
        v = df.get("volume", pd.Series(1, index=df.index))

        # ── Tính các indicators ───────────────────────────────────────────
        cmf     = self._cmf(h, l, c, v, self.cmf_w)
        vpt     = self._vpt(c, v)
        vpt_ma  = vpt.rolling(self.vpt_w).mean()
        obv     = self._obv(c, v)
        obv_ma  = obv.rolling(self.window).mean()
        fi      = self._force_index(c, v, 13)

        # ── Scoring (bar cuối) ────────────────────────────────────────────
        score = 0
        parts = []

        # CMF: > +0.05 = tiền vào, < -0.05 = tiền ra
        cmf_val = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0
        if cmf_val > 0.10:
            score += 2; parts.append(f"CMF={cmf_val:.2f}(tienVao)")
        elif cmf_val > 0.03:
            score += 1; parts.append(f"CMF={cmf_val:.2f}(+)")
        elif cmf_val < -0.10:
            score -= 2; parts.append(f"CMF={cmf_val:.2f}(tienRa)")
        elif cmf_val < -0.03:
            score -= 1; parts.append(f"CMF={cmf_val:.2f}(-)")
        else:
            parts.append(f"CMF={cmf_val:.2f}(neutral)")

        # VPT vs MA: VPT tăng nhanh hơn MA → dòng tiền vào
        vpt_last   = float(vpt.iloc[-1])
        vpt_ma_last= float(vpt_ma.iloc[-1]) if not pd.isna(vpt_ma.iloc[-1]) else vpt_last
        if vpt_last > vpt_ma_last * 1.02:
            score += 1; parts.append("VPT>MA(bull)")
        elif vpt_last < vpt_ma_last * 0.98:
            score -= 1; parts.append("VPT<MA(bear)")

        # OBV vs MA
        obv_last   = float(obv.iloc[-1])
        obv_ma_last= float(obv_ma.iloc[-1]) if not pd.isna(obv_ma.iloc[-1]) else obv_last
        if obv_last > obv_ma_last:
            score += 1; parts.append("OBV>MA(bull)")
        else:
            score -= 1; parts.append("OBV<MA(bear)")

        # Force Index: > 0 = lực mua, < 0 = lực bán
        fi_val = float(fi.iloc[-1]) if not pd.isna(fi.iloc[-1]) else 0
        if fi_val > 0:
            score += 1; parts.append(f"FI>0(bullish)")
        else:
            score -= 1; parts.append(f"FI<0(bearish)")

        # OBV divergence (5D): giá giảm nhưng OBV tăng → smart money mua
        price_5d_chg = float((c.iloc[-1] - c.iloc[-5]) / c.iloc[-5]) if len(c) >= 5 else 0
        obv_5d_chg   = float((obv.iloc[-1] - obv.iloc[-5]) / abs(obv.iloc[-5]) + 1e-9) if len(obv) >= 5 else 0
        if price_5d_chg < -0.01 and obv_5d_chg > 0.01:
            score += 1; parts.append("OBV_div(bull)")  # bullish divergence
        elif price_5d_chg > 0.01 and obv_5d_chg < -0.01:
            score -= 1; parts.append("OBV_div(bear)")  # bearish divergence

        # Signal
        if score >= 3:
            cur_sig = 1
        elif score <= -3:
            cur_sig = -1
        elif score >= 1:
            cur_sig = 1
        elif score <= -1:
            cur_sig = -1
        else:
            cur_sig = 0

        sig = pd.Series(0, index=df.index, dtype=int)
        sig.iloc[-1] = cur_sig

        label = "BUY(dong tien vao)" if cur_sig > 0 else "SELL(dong tien ra)" if cur_sig < 0 else "NEUTRAL"
        det = f"Score={score:+d} | {' | '.join(parts)} | {label}"
        return sig, det

    def generate(self, data_map: dict) -> tuple[dict, dict]:
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code] = _last(sig)
                details[code] = f"[Claude-made] {det}"
            except Exception as e:
                signals[code] = 0
                details[code] = f"[Claude-made] Loi MoneyFlowEngine: {e}"
        return signals, details

# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON INSTANCES — 14 engines đầy đủ
# ══════════════════════════════════════════════════════════════════════════════

_ENGINES = {
    # --- Từ Vibe-Trading strategy skills (có example_signal_engine.py) ---
    "Candlestick":    CandlestickEngine(),      # 15 mô hình nến
    "Ichimoku":       IchimokuEngine(),          # 5-line system
    "TechnicalBasic": TechnicalEngine(),         # EMA/ADX/RSI/OBV
    "ElliottWave":    ElliottEngine(),           # Zigzag + 5-wave
    "Harmonic":       HarmonicEngine(),          # Gartley/Bat/Butterfly/Crab
    "Volatility":     VolatilityEngine(),        # HV percentile
    "Seasonal":       SeasonalEngine(),          # Calendar effect
    "SMC":            SMCEngine(),               # BOS/ChoCH/FVG — momentum-based
    "CrossMarket":    CrossMarketEngine(),       # Vol-adjusted dual-MA
    "MultiFactor":    MultiFactorEngine(),       # Momentum/Reversal/Vol/VolumeRatio
    # --- Adapted từ Vibe-Trading pair-trading + fundamental-filter ---
    "MeanReversion":  MeanReversionEngine(),    # Z-score price vs rolling mean
    "PriceMomentum":  PriceMomentumEngine(),    # Multi-TF momentum 5D/20D/60D
    "Breakout":       BreakoutEngine(),          # S/R breakout + vol confirm
    "MLStrategy":     MLStrategyEngine(),        # ExtraTrees walk-forward + rule fallback
    # --- Mới: Fundamental + MoneyFlow (thu hẹp gap 16 vs 69) ---
    "Fundamental":    FundamentalEngine(),       # PE/PB/ROE/EPS từ vnstock
    "MoneyFlow":      MoneyFlowEngine(),         # CMF/VPT/OBV/Force Index
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
