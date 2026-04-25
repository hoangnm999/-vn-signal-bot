"""
vn_loader.py — DataLoader cho thị trường chứng khoán Việt Nam.

BUGS ĐÃ FIX (từ log 2026-04-25):
  BUG 1: analyzer df có cột 'time' → rename sang 'date'
  BUG 2: Entrade data['t']=[] → loop qua nextTime pagination
  BUG 3: vnstock TCBS/MSN không support → dùng KBS + FMP
  BUG 4: vnstock VCI KeyError 'data' → xử lý đúng format response
"""

from __future__ import annotations
import os, logging, requests, pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DAYS    = 365
REQUEST_TIMEOUT = 20

_H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Referer": "https://banggia.vn/",
    "Origin":  "https://banggia.vn",
}

_DATE_ALIASES = ("time", "tradingdate", "trading_date", "datetime", "timestamp", "date_", "index")

def _ds(dt): return dt.strftime("%Y-%m-%d")


# ── SOURCE 0: analyzer.get_price_data() ────────────────────────────────────────
def _fetch_from_analyzer(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """BUG 1 FIX: map cot time → date truoc khi xuly."""
    try:
        from analyzer import get_price_data
        result = get_price_data(symbol, days + 60)
        if not result or not result.get("success"):
            logger.warning(f"analyzer fail {symbol}: {result.get('error') if result else 'no result'}")
            return None
        df = result.get("df")
        if df is None or df.empty:
            return None
        df = df.copy()
        df.columns = [str(c).lower().strip() for c in df.columns]
        # BUG 1 FIX: rename 'time' hoac cac alias khac → 'date'
        if "date" not in df.columns:
            for alias in _DATE_ALIASES:
                if alias in df.columns:
                    df = df.rename(columns={alias: "date"})
                    logger.info(f"analyzer: renamed '{alias}' -> 'date'")
                    break
            else:
                if isinstance(df.index, pd.DatetimeIndex):
                    df.index.name = "date"
                    df = df.reset_index()
                elif df.index.name and df.index.name.lower() in _DATE_ALIASES:
                    df.index.name = "date"
                    df = df.reset_index()
                else:
                    logger.warning(f"analyzer df ko co col date: {list(df.columns)}")
                    return None
        # map ten cot gia neu viet tat
        rn = {}
        for c in df.columns:
            if c == "o" and "open" not in df.columns:    rn[c] = "open"
            elif c == "h" and "high" not in df.columns:  rn[c] = "high"
            elif c == "l" and "low" not in df.columns:   rn[c] = "low"
            elif c in ("c","adjclose") and "close" not in df.columns: rn[c] = "close"
            elif c in ("vol","v","volume_match") and "volume" not in df.columns: rn[c] = "volume"
        if rn: df = df.rename(columns=rn)
        needed = ["date","open","high","low","close","volume"]
        miss = [c for c in needed if c not in df.columns]
        if miss:
            logger.warning(f"analyzer df thieu {miss}, co: {list(df.columns)[:8]}")
            return None
        df = df[needed].copy()
        df["date"]   = pd.to_datetime(df["date"], errors="coerce")
        df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df = df.dropna(subset=["date","close"])
        df = df[df["close"] > 0].sort_values("date").tail(days).reset_index(drop=True)
        logger.info(f"analyzer OK: {symbol} {len(df)} bars")
        return df
    except ImportError:
        logger.debug("analyzer.py ko import duoc")
        return None
    except Exception as e:
        logger.warning(f"analyzer exception {symbol}: {e}")
        return None


# ── SOURCE 1: Entrade v2 voi pagination ────────────────────────────────────────
def _fetch_entrade(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """BUG 2 FIX: loop qua nextTime khi data[t]=[]."""
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=days + 60)
    all_t=[];all_o=[];all_h=[];all_l=[];all_c=[];all_v=[]
    current_to = int(end_dt.timestamp())
    from_ts    = int(start_dt.timestamp())
    for _ in range(20):
        try:
            r = requests.get(
                "https://services.entrade.com.vn/chart-api/v2/ohlcs/stock",
                params={"resolution":"D","symbol":symbol,"from":from_ts,"to":current_to},
                headers=_H, timeout=REQUEST_TIMEOUT,
            )
        except Exception as e:
            logger.warning(f"Entrade request error {symbol}: {e}"); break
        if r.status_code == 403:
            logger.warning(f"Entrade 403 {symbol}: blocked"); break
        if not r.ok:
            logger.warning(f"Entrade HTTP {r.status_code} {symbol}"); break
        try: data = r.json()
        except: logger.warning(f"Entrade JSON error {symbol}"); break
        t_list    = data.get("t", [])
        next_time = data.get("nextTime")
        if t_list:
            all_t.extend(t_list);all_o.extend(data.get("o",[]));all_h.extend(data.get("h",[]))
            all_l.extend(data.get("l",[]));all_c.extend(data.get("c",[]));all_v.extend(data.get("v",[]))
        if next_time and next_time > from_ts:
            current_to = next_time - 1
        else:
            break
        if len(all_t) >= days + 30: break
    if not all_t:
        logger.warning(f"Entrade: no data for {symbol}")
        return None
    try:
        df = pd.DataFrame({"date":pd.to_datetime(all_t,unit="s"),"open":all_o,"high":all_h,
                           "low":all_l,"close":all_c,"volume":all_v})
        df = df.sort_values("date").drop_duplicates("date").tail(days).reset_index(drop=True)
        logger.info(f"Entrade OK: {symbol} {len(df)} bars")
        return df
    except Exception as e:
        logger.warning(f"Entrade parse error {symbol}: {e}"); return None


# ── SOURCE 2: DNSE v3 / SSI ────────────────────────────────────────────────────
def _fetch_dnse_v3(symbol: str, days: int) -> Optional[pd.DataFrame]:
    end_dt = datetime.now(); start_dt = end_dt - timedelta(days=days+60)
    endpoints = [
        ("https://services.entrade.com.vn/chart-api/v3/ohlcs/stock",
         {"resolution":"D","symbol":symbol,"from":int(start_dt.timestamp()),"to":int(end_dt.timestamp())}),
        ("https://fc-data.ssi.com.vn/api/v2/GetOhlcFromHistory",
         {"Symbol":symbol,"StartDate":_ds(start_dt),"EndDate":_ds(end_dt),
          "Market":"HOSE","Frequency":"D","PageIndex":1,"PageSize":days+60}),
    ]
    for url, params in endpoints:
        try:
            r = requests.get(url, params=params, headers=_H, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200: continue
            data = r.json()
            if data.get("t"):
                df = pd.DataFrame({"date":pd.to_datetime(data["t"],unit="s"),
                    "open":data["o"],"high":data["h"],"low":data["l"],
                    "close":data["c"],"volume":data["v"]})
                df = df.sort_values("date").tail(days).reset_index(drop=True)
                logger.info(f"DNSE v3 OK: {symbol} {len(df)} bars"); return df
            for key in ("data","items","Data","result"):
                records = data.get(key)
                if not records: continue
                first = records[0] if isinstance(records,list) else {}
                dk = next((k for k in ("TradingDate","tradingDate","Date","date","t") if k in first),None)
                if not dk: continue
                rows = []
                for b in records:
                    ts = b.get(dk)
                    try: dt = pd.to_datetime(ts,unit="s") if isinstance(ts,(int,float)) else pd.to_datetime(ts)
                    except: continue
                    rows.append({"date":dt,
                        "open":float(b.get("Open",b.get("open",b.get("o",0))) or 0),
                        "high":float(b.get("High",b.get("high",b.get("h",0))) or 0),
                        "low":float(b.get("Low",b.get("low",b.get("l",0))) or 0),
                        "close":float(b.get("Close",b.get("close",b.get("c",0))) or 0),
                        "volume":float(b.get("Volume",b.get("volume",b.get("v",0))) or 0)})
                if len(rows) >= 60:
                    df = pd.DataFrame(rows).sort_values("date").tail(days).reset_index(drop=True)
                    logger.info(f"DNSE v3 records OK: {symbol} {len(df)} bars"); return df
        except Exception as e:
            logger.debug(f"DNSE {url[-30:]} fail: {e}"); continue
    return None


# ── SOURCE 3: Fireant ──────────────────────────────────────────────────────────
def _fetch_fireant(symbol: str, days: int) -> Optional[pd.DataFrame]:
    token = os.environ.get("FIREANT_TOKEN","").strip()
    if not token: logger.debug("No FIREANT_TOKEN"); return None
    end_dt=datetime.now(); start_dt=end_dt-timedelta(days=days+60)
    h = {**_H, "Authorization": f"Bearer {token}"}
    try:
        r = requests.get(
            f"https://restv2.fireant.vn/symbols/{symbol}/historical-quotes",
            headers=h,
            params={"startDate":_ds(start_dt),"endDate":_ds(end_dt),"offset":0,"limit":days+80},
            timeout=REQUEST_TIMEOUT)
        if r.status_code == 403: logger.warning(f"Fireant 403 {symbol}: token expired"); return None
        if r.status_code == 401: logger.warning(f"Fireant 401 {symbol}: invalid token"); return None
        r.raise_for_status()
        raw = r.json()
        if not raw: return None
        records = []
        for item in raw:
            cr = float(item.get("priceClose",item.get("close",0)) or 0)
            if cr <= 0: continue
            m = 1000 if cr < 1000 else 1
            records.append({"date":pd.to_datetime(item.get("date",item.get("tradingDate",""))),
                "open":float(item.get("priceOpen",item.get("open",0)) or 0)*m,
                "high":float(item.get("priceHigh",item.get("high",0)) or 0)*m,
                "low":float(item.get("priceLow",item.get("low",0)) or 0)*m,
                "close":cr*m,
                "volume":float(item.get("totalVolume",item.get("volume",0)) or 0)})
        if not records: return None
        df = pd.DataFrame(records).sort_values("date").tail(days).reset_index(drop=True)
        logger.info(f"Fireant OK: {symbol} {len(df)} bars"); return df
    except Exception as e:
        logger.warning(f"Fireant fail {symbol}: {e}"); return None


# ── SOURCE 4: TCBS API ─────────────────────────────────────────────────────────
def _fetch_tcbs(symbol: str, days: int) -> Optional[pd.DataFrame]:
    end_dt=datetime.now(); start_dt=end_dt-timedelta(days=days+60)
    for url in [
        f"https://apipubaws.tcbs.com.vn/stock-insight/v2/stock/{symbol}/bars-long-term",
        f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/{symbol}/bars-long-term",
    ]:
        try:
            r = requests.get(url,
                params={"resolution":"D","from":int(start_dt.timestamp()),"to":int(end_dt.timestamp()),"type":"stock"},
                headers=_H, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200: continue
            data = r.json()
            bars = data.get("data", data.get("bars", []))
            if not bars: continue
            records = []
            for b in bars:
                ts = b.get("TradingDate",b.get("tradingDate",b.get("t")))
                try: dt = pd.to_datetime(ts,unit="s") if isinstance(ts,(int,float)) else pd.to_datetime(ts)
                except: continue
                records.append({"date":dt,
                    "open":float(b.get("Open",b.get("open",b.get("o",0))) or 0),
                    "high":float(b.get("High",b.get("high",b.get("h",0))) or 0),
                    "low":float(b.get("Low",b.get("low",b.get("l",0))) or 0),
                    "close":float(b.get("Close",b.get("close",b.get("c",0))) or 0),
                    "volume":float(b.get("Volume",b.get("volume",b.get("v",0))) or 0)})
            if len(records) >= 60:
                df = pd.DataFrame(records).sort_values("date").tail(days).reset_index(drop=True)
                logger.info(f"TCBS OK: {symbol} {len(df)} bars"); return df
        except Exception as e:
            logger.debug(f"TCBS {url[-30:]} error: {e}"); continue
    logger.warning(f"TCBS: all endpoints fail for {symbol}"); return None


# ── SOURCE 5: vnstock KBS/VCI/FMP ─────────────────────────────────────────────
def _fetch_vnstock(symbol: str, days: int) -> Optional[pd.DataFrame]:
    """BUG 3+4 FIX: chi dung KBS/VCI/FMP, xu ly dung column format."""
    try:
        from vnstock import Vnstock
    except ImportError:
        logger.debug("vnstock chua cai"); return None
    end_dt=datetime.now(); start_dt=end_dt-timedelta(days=days+60)
    start_s=_ds(start_dt); end_s=_ds(end_dt)
    # BUG 3 FIX: bo TCBS/MSN, chi dung KBS/VCI/FMP
    for src in ("KBS","VCI","FMP"):
        try:
            stock  = Vnstock().stock(symbol=symbol, source=src)
            df_raw = stock.quote.history(start=start_s, end=end_s, interval="1D")
            if df_raw is None or (hasattr(df_raw,"empty") and df_raw.empty): continue
            df = df_raw.copy()
            # Flatten MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(str(x) for x in c).lower().strip("_") for c in df.columns]
            else:
                df.columns = [str(c).lower().strip() for c in df.columns]
            # Reset index neu date o trong index
            if not isinstance(df.index, pd.RangeIndex):
                if isinstance(df.index, pd.DatetimeIndex) or (
                    df.index.name and df.index.name.lower() in _DATE_ALIASES + ("date",)):
                    df.index.name = "date"
                df = df.reset_index()
            df.columns = [str(c).lower().strip() for c in df.columns]
            # BUG 4 FIX: rename tat ca alias ngay
            if "date" not in df.columns:
                for alias in _DATE_ALIASES:
                    if alias in df.columns:
                        df = df.rename(columns={alias:"date"}); break
            # Map ten cot gia
            rn = {}
            cs = set(df.columns)
            for tgt, als in [("open",["o","open_price"]),("high",["h","high_price"]),
                              ("low",["l","low_price"]),("close",["c","close_price","adjclose"]),
                              ("volume",["vol","v","volume_match","klgd"])]:
                if tgt not in cs:
                    for a in als:
                        if a in cs: rn[a]=tgt; break
            if rn: df = df.rename(columns=rn)
            needed=["date","open","high","low","close","volume"]
            miss=[c for c in needed if c not in df.columns]
            if miss:
                logger.warning(f"vnstock {src}: missing {miss}, have={list(df.columns)[:8]}"); continue
            df = df[needed].copy()
            for col in needed[1:]: df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"]   = pd.to_datetime(df["date"], errors="coerce")
            df["volume"] = df["volume"].fillna(0)
            df = df.dropna(subset=["date","close"])
            df = df[df["close"] > 0]
            med = df["close"].median()
            if 0 < med < 1000:
                for col in ("open","high","low","close"): df[col] *= 1000
            df = df.sort_values("date").tail(days).reset_index(drop=True)
            if len(df) >= 60:
                logger.info(f"vnstock {src} OK: {symbol} {len(df)} bars"); return df
        except Exception as e:
            logger.warning(f"vnstock {src} fail {symbol}: {type(e).__name__}: {e}"); continue
    return None


# ── PUBLIC API ─────────────────────────────────────────────────────────────────
def load_vn_ohlcv(symbol: str, days: int = DEFAULT_DAYS, min_bars: int = 60) -> pd.DataFrame:
    symbol = symbol.upper().strip()
    failures = []
    for fn, name in [
        (_fetch_from_analyzer, "analyzer.get_price_data"),
        (_fetch_entrade,       "Entrade v2"),
        (_fetch_dnse_v3,       "DNSE v3/SSI"),
        (_fetch_fireant,       "Fireant"),
        (_fetch_tcbs,          "TCBS"),
        (_fetch_vnstock,       "vnstock KBS/VCI/FMP"),
    ]:
        try:
            df = fn(symbol, days)
            if df is None: failures.append(f"{name}: None"); continue
            if len(df) < min_bars: failures.append(f"{name}: {len(df)} bars < {min_bars}"); continue
            df = _clean_ohlcv(df)
            if len(df) < min_bars: failures.append(f"{name}: after_clean {len(df)} < {min_bars}"); continue
            logger.info(f"load_vn_ohlcv({symbol}): OK via {name} | {len(df)} bars")
            return df
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {str(e)[:60]}")
            logger.warning(f"load_vn_ohlcv [{name}]: {e}")
    raise ValueError(
        f"Khong the lay du lieu cho {symbol} ({days} ngay). "
        f"Da thu 6 sources. Chi tiet: {' | '.join(failures)}"
    )


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for col in ("open","high","low","close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df = df.dropna(subset=["open","high","low","close"])
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    return df.sort_values("date").drop_duplicates("date",keep="last").reset_index(drop=True)


def debug_sources(symbol: str, days: int = 100) -> str:
    import time
    symbol = symbol.upper()
    lines = [f"=== VN Loader Debug: {symbol} ({days}D) ==="]
    for fn, name in [
        (_fetch_from_analyzer, "analyzer.get_price_data"),
        (_fetch_entrade,       "Entrade v2"),
        (_fetch_dnse_v3,       "DNSE v3/SSI"),
        (_fetch_fireant,       "Fireant"),
        (_fetch_tcbs,          "TCBS"),
        (_fetch_vnstock,       "vnstock KBS/VCI/FMP"),
    ]:
        t0 = time.time()
        try:
            df = fn(symbol, days)
            el = round(time.time()-t0, 1)
            if df is not None and len(df) > 0:
                lc = df["close"].iloc[-1]; ld = str(df["date"].iloc[-1])[:10]
                lines.append(f"  ✅ {name}: {len(df)} bars | last={lc:,.0f} @ {ld} ({el}s)")
            else:
                lines.append(f"  ❌ {name}: None/empty ({el}s)")
        except Exception as e:
            el = round(time.time()-t0, 1)
            lines.append(f"  ❌ {name}: {type(e).__name__}: {str(e)[:70]} ({el}s)")
    return "\n".join(lines)


def get_vn_info(symbol: str) -> dict:
    token = os.environ.get("FIREANT_TOKEN","").strip()
    base  = {"symbol":symbol,"name":symbol,"exchange":"HSX/HNX","industry":"N/A"}
    if not token: return base
    try:
        r = requests.get(f"https://restv2.fireant.vn/symbols/{symbol}",
            headers={**_H,"Authorization":f"Bearer {token}"}, timeout=10)
        r.raise_for_status(); d = r.json()
        return {"symbol":symbol,"name":d.get("companyName",symbol),
                "exchange":d.get("exchange","N/A"),"industry":d.get("industryName","N/A")}
    except: return base
