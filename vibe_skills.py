"""
vibe_skills.py  — 10 Signal Engines từ Vibe-Trading (HKUDS/Vibe-Trading)
Tích hợp vào VN Signal Bot, chạy trực tiếp trên OHLCV Entrade.

Engines (pure pandas/numpy, không cần API server):
  1.  Candlestick       — 15 mô hình nến kinh điển (Hammer→3 White Soldiers)
  2.  Ichimoku          — 5-line system: TK cross + cloud position + Chikou
  3.  TechnicalBasic    — EMA/ADX/BB/RSI/OBV 3-dim voting
  4.  ElliottWave       — Zigzag swing → 5-wave impulse + Fibonacci validation
  5.  Harmonic          — Gartley/Bat/Butterfly/Crab XABCD PRZ signals
  6.  Volatility        — HV percentile mean-reversion
  7.  Seasonal          — Month-of-year calendar effect
  8.  SMC               — BOS/ChoCH/FVG Smart Money Concepts (pure pandas fallback)
  9.  CrossMarket       — Vol-adjusted dual-MA (dùng cho single stock = a_share params)
  10. MultiFactor       — Momentum/Reversal/Volatility/VolumeRatio cross-section rank

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
            sig_series = pd.Series(np.sign(total).astype(int), index=df.index)
            signals[code] = _last(sig_series)
            # Tên pattern xuất hiện ở cây nến cuối
            last = sc.iloc[-1]
            found_bull = [n for n in sc.columns if last.get(n,0)>0]
            found_bear = [n for n in sc.columns if last.get(n,0)<0]
            desc = ""
            if found_bull: desc += f"Bullish: {', '.join(found_bull)}. "
            if found_bear: desc += f"Bearish: {', '.join(found_bear)}. "
            details[code] = desc.strip() or "Khong nhan dien mo hinh nen dac biet."
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 2 — ICHIMOKU (from Vibe-Trading)
# ══════════════════════════════════════════════════════════════════════════════

class IchimokuEngine:
    def __init__(self, tenkan=9, kijun=26, senkou_b=52, displacement=26):
        self.tenkan=tenkan; self.kijun=kijun
        self.senkou_b=senkou_b; self.displacement=displacement

    def _mid(self, s, p): return (s.rolling(p).max()+s.rolling(p).min())/2

    def generate(self, data_map):
        signals, details = {}, {}
        warmup = self.senkou_b + self.displacement
        for sym, df in data_map.items():
            h,l,c = df["high"],df["low"],df["close"]
            if len(df) < warmup:
                signals[sym]=0; details[sym]="Khong du du lieu (can >= 78 bars)."; continue
            tenkan = self._mid(h,self.tenkan)
            kijun  = self._mid(h,self.kijun)
            span_a = ((tenkan+kijun)/2).shift(self.displacement)
            span_b = self._mid(h,self.senkou_b).shift(self.displacement)
            cloud_top = pd.concat([span_a,span_b],axis=1).max(axis=1)
            cloud_bot = pd.concat([span_a,span_b],axis=1).min(axis=1)
            tk_bull = (tenkan>kijun)&(tenkan.shift(1)<=kijun.shift(1))
            tk_bear = (tenkan<kijun)&(tenkan.shift(1)>=kijun.shift(1))
            above   = c > cloud_top
            below   = c < cloud_bot
            bull_cloud = span_a > span_b
            bear_cloud = span_a < span_b
            buy  = tk_bull & above & bull_cloud
            sell = tk_bear & below & bear_cloud
            sig  = pd.Series(0,index=df.index,dtype=int)
            sig[buy]=1; sig[sell]=-1
            signals[sym] = _last(sig)
            # Detail
            t_v = round(tenkan.iloc[-1],2) if not pd.isna(tenkan.iloc[-1]) else "N/A"
            k_v = round(kijun.iloc[-1],2)  if not pd.isna(kijun.iloc[-1]) else "N/A"
            ct  = round(cloud_top.iloc[-1],2) if not pd.isna(cloud_top.iloc[-1]) else "N/A"
            cb  = round(cloud_bot.iloc[-1],2) if not pd.isna(cloud_bot.iloc[-1]) else "N/A"
            tk_str = "bull" if tenkan.iloc[-1]>kijun.iloc[-1] else "bear"
            pos    = "TREN may" if c.iloc[-1]>cloud_top.iloc[-1] else "DUOI may" if c.iloc[-1]<cloud_bot.iloc[-1] else "TRONG may"
            details[sym] = f"TK={tk_str} | Tenkan={t_v} Kijun={k_v} | {pos} (top={ct} bot={cb})"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 3 — TECHNICAL BASIC (EMA/ADX/BB/RSI/OBV, from Vibe-Trading)
# ══════════════════════════════════════════════════════════════════════════════

class TechnicalEngine:
    def __init__(self, ef=12, es=26, adx_p=14, adx_th=25.0,
                 bb_w=20, bb_s=2.0, rsi_p=14, rsi_os=30, rsi_ob=70, vma=20):
        self.ef=ef; self.es=es; self.adx_p=adx_p; self.adx_th=adx_th
        self.bb_w=bb_w; self.bb_s=bb_s; self.rsi_p=rsi_p
        self.rsi_os=rsi_os; self.rsi_ob=rsi_ob; self.vma=vma

    def _rsi(self, c, p):
        d=c.diff(); g=d.where(d>0,0).ewm(alpha=1/p,adjust=False).mean()
        ls=(-d.where(d<0,0)).ewm(alpha=1/p,adjust=False).mean()
        return 100-100/(1+g/ls.replace(0,np.nan))

    def _adx(self, h, l, c, p):
        tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
        pdm=(h-h.shift()).where((h-h.shift())>(l.shift()-l),0).clip(lower=0)
        ndm=(l.shift()-l).where((l.shift()-l)>(h-h.shift()),0).clip(lower=0)
        atr=tr.ewm(alpha=1/p,adjust=False).mean()
        pdi=100*pdm.ewm(alpha=1/p,adjust=False).mean()/atr.replace(0,np.nan)
        ndi=100*ndm.ewm(alpha=1/p,adjust=False).mean()/atr.replace(0,np.nan)
        dx=100*(pdi-ndi).abs()/(pdi+ndi).replace(0,np.nan)
        return dx.ewm(alpha=1/p,adjust=False).mean(), pdi, ndi

    def _one(self, df):
        o,h,l,c = df["open"],df["high"],df["low"],df["close"]
        v = df.get("volume", pd.Series(1,index=df.index))
        ef=c.ewm(span=self.ef,adjust=False).mean()
        es=c.ewm(span=self.es,adjust=False).mean()
        adx,pdi,ndi = self._adx(h,l,c,self.adx_p)
        tbull=(ef>es)&(adx>self.adx_th)&(pdi>ndi)
        tbear=(ef<es)&(adx>self.adx_th)&(ndi>pdi)
        bm=c.rolling(self.bb_w).mean(); bs=c.rolling(self.bb_w).std()
        bu=bm+self.bb_s*bs; bl=bm-self.bb_s*bs
        rsi=self._rsi(c,self.rsi_p)
        mr_bull=(c<bl)&(rsi<self.rsi_os); mr_bear=(c>bu)&(rsi>self.rsi_ob)
        obv=(v*np.sign(c.diff())).fillna(0).cumsum()
        obv_ma=obv.rolling(self.vma).mean(); vm=v.rolling(self.vma).mean()
        vp_bull=(obv>obv_ma)&(v>vm); vp_bear=(obv<obv_ma)&(v>vm)
        sig=pd.Series(0,index=df.index,dtype=int)
        sig[tbull&~mr_bear&vp_bull]=1; sig[tbear&~mr_bull&vp_bear]=-1
        # Detail
        adx_v=round(adx.iloc[-1],1); rsi_v=round(rsi.iloc[-1],1)
        ef_v=round(ef.iloc[-1],2);   es_v=round(es.iloc[-1],2)
        trend="bull" if ef.iloc[-1]>es.iloc[-1] else "bear"
        detail=(f"EMA{self.ef}={ef_v} EMA{self.es}={es_v} ADX={adx_v} RSI={rsi_v} trend={trend}")
        return sig, detail

    def generate(self, data_map):
        signals, details = {}, {}
        for code, df in data_map.items():
            try:
                sig, det = self._one(df)
                signals[code]=_last(sig); details[code]=det
            except Exception as e:
                signals[code]=0; details[code]=f"Loi: {e}"
        return signals, details


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 4 — ELLIOTT WAVE (Zigzag + 5-wave + Fibonacci, from Vibe-Trading)
# ══════════════════════════════════════════════════════════════════════════════

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
                    det=(f"5-song {'tang' if d>0 else 'giam'} phat hien. "
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
    def __init__(self, hv_w=20, pct_w=252, lo_pct=20, hi_pct=80, ann=252):
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
    def __init__(self, lookback_years=5, min_count=3, threshold=0.60):
        self.lb=lookback_years; self.mc=min_count; self.th=threshold

    def _one(self, df):
        c=df["close"]
        mret=c.resample("ME").last().pct_change().dropna()
        sig=pd.Series(0,index=df.index,dtype=int)
        if len(mret)<self.mc*6: return sig, "Khong du du lieu lich su seasonal."
        cutoff=mret.index[-1]-pd.DateOffset(years=self.lb)
        hist=mret[mret.index>=cutoff]
        stats=hist.groupby(hist.index.month).apply(
            lambda x: (x>0).sum()/len(x) if len(x)>=self.mc else 0.5)
        for dt in df.index:
            m=dt.month
            if m in stats.index:
                pw=stats[m]
                if pw>=self.th: sig[dt]=1
                elif pw<=(1-self.th): sig[dt]=-1
        import datetime; cm=datetime.datetime.now().month
        mn=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        bias=int(round(stats.get(cm,0.5)*100))
        cur_sig=_last(sig)
        det=(f"Thang {mn[cm-1]}: lich su tang {bias}% lan. "
             +("Thang nay thuong TANG." if cur_sig>0 else
               "Thang nay thuong GIAM." if cur_sig<0 else
               "Khong co xu huong ro rang."))
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
# ENGINE 8 — SMC (Smart Money Concepts — pure pandas fallback, from Vibe-Trading)
# BOS/ChoCH/FVG không dùng external lib
# ══════════════════════════════════════════════════════════════════════════════

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
        sh,sl=self._swing_hl(df["high"],df["low"])
        structure=self._bos_choch(df,sh,sl)
        fvg=self._fvg(df)
        # Signal: structure xác nhận + FVG cùng chiều hoặc neutral
        buy =(structure==1)&(fvg>=0)
        sell=(structure==-1)&(fvg<=0)
        sig=pd.Series(0,index=df.index,dtype=int)
        sig[buy]=1; sig[sell]=-1
        last_s=_last(sig)
        n_buy=(structure==1).sum(); n_sell=(structure==-1).sum()
        det=(f"BOS/ChoCH: {n_buy} bullish, {n_sell} bearish breaks. "
             f"FVG: bull={((fvg==1)).sum()} bear={((fvg==-1)).sum()}. "
             f"Signal={'BUY' if last_s>0 else 'SELL' if last_s<0 else 'NEUTRAL'}")
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
        det=(f"MA{self.MA_FAST}={round(mf.iloc[-1],2)} MA{self.MA_SLOW}={round(ms.iloc[-1],2)} "
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
        det=(f"Factor score={round(cur_score,2) if not pd.isna(cur_score) else 'N/A'} "
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
# SINGLETON INSTANCES
# ══════════════════════════════════════════════════════════════════════════════

_ENGINES = {
    "Candlestick":    CandlestickEngine(),
    "Ichimoku":       IchimokuEngine(),
    "TechnicalBasic": TechnicalEngine(),
    "ElliottWave":    ElliottEngine(),
    "Harmonic":       HarmonicEngine(),
    "Volatility":     VolatilityEngine(),
    "Seasonal":       SeasonalEngine(),
    "SMC":            SMCEngine(),
    "CrossMarket":    CrossMarketEngine(),
    "MultiFactor":    MultiFactorEngine(),
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_vibe_agents(symbol: str, df: pd.DataFrame) -> dict:
    """
    Chạy tất cả 10 Vibe-Trading engines trên OHLCV DataFrame.

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
        f"Vibe-Trading 10 engines: {bull}/{n} bullish, {bear}/{n} bearish. "
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
