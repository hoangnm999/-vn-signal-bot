import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ─── DATA LAYER ───────────────────────────────────────────────────────────────

def get_stock_data(symbol: str, days: int = 90) -> dict:
    """Lấy data từ vnstock"""
    try:
        from vnstock import stock_historical_data, financial_ratio
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        df = stock_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            resolution="1D",
            type="stock"
        )

        if df is None or df.empty:
            raise ValueError(f"Không có dữ liệu cho {symbol}")

        df = df.sort_values("time").reset_index(drop=True)
        return {"success": True, "df": df, "symbol": symbol}

    except Exception as e:
        return {"success": False, "error": str(e), "symbol": symbol}


def compute_indicators(df: pd.DataFrame) -> dict:
    """Tính toán các chỉ báo kỹ thuật"""
    close = df["close"]
    volume = df["volume"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - 100 / (1 + rs)).iloc[-1]

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9).mean()
    macd_val = macd.iloc[-1]
    signal_val = signal_line.iloc[-1]
    macd_hist = macd_val - signal_val

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = (sma20 + 2 * std20).iloc[-1]
    bb_lower = (sma20 - 2 * std20).iloc[-1]
    bb_mid = sma20.iloc[-1]

    # MA
    ma20 = sma20.iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None

    # Volume
    avg_vol_20 = volume.rolling(20).mean().iloc[-1]
    curr_vol = volume.iloc[-1]
    vol_ratio = curr_vol / avg_vol_20 if avg_vol_20 > 0 else 1

    # Price action
    current_price = close.iloc[-1]
    price_1w = close.iloc[-5] if len(close) >= 5 else close.iloc[0]
    price_1m = close.iloc[-20] if len(close) >= 20 else close.iloc[0]
    change_1w = (current_price - price_1w) / price_1w * 100
    change_1m = (current_price - price_1m) / price_1m * 100

    # Support / Resistance (20-day)
    high_20 = df["high"].tail(20).max()
    low_20 = df["low"].tail(20).min()

    return {
        "current_price": round(current_price, 2),
        "change_1w_pct": round(change_1w, 2),
        "change_1m_pct": round(change_1m, 2),
        "rsi": round(rsi, 1),
        "macd": round(macd_val, 4),
        "macd_signal": round(signal_val, 4),
        "macd_hist": round(macd_hist, 4),
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
        "bb_mid": round(bb_mid, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2) if ma50 else None,
        "volume_ratio": round(vol_ratio, 2),
        "resistance_20d": round(high_20, 2),
        "support_20d": round(low_20, 2),
    }


# ─── ANALYSIS LAYER ───────────────────────────────────────────────────────────

def call_deepseek(system_prompt: str, user_prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 800
    }
    resp = requests.post(DEEPSEEK_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def run_trend_agent(symbol: str, indicators: dict) -> str:
    system = (
        "Bạn là chuyên gia phân tích xu hướng giá cổ phiếu Việt Nam. "
        "Chỉ phân tích ngắn gọn, tập trung vào trend, MA, momentum. "
        "Kết luận bằng: TĂNG / GIẢM / SIDEWAY"
    )
    user = f"""
Cổ phiếu: {symbol}
Giá hiện tại: {indicators['current_price']:,}
Thay đổi 1 tuần: {indicators['change_1w_pct']}%
Thay đổi 1 tháng: {indicators['change_1m_pct']}%
MA20: {indicators['ma20']:,}
MA50: {indicators.get('ma50', 'N/A')}
RSI(14): {indicators['rsi']}
MACD histogram: {indicators['macd_hist']}

Phân tích xu hướng trong 2-3 câu. Kết luận: TĂNG/GIẢM/SIDEWAY
"""
    return call_deepseek(system, user)


def run_volume_agent(symbol: str, indicators: dict) -> str:
    system = (
        "Bạn là chuyên gia phân tích khối lượng giao dịch. "
        "Phân tích volume để xác nhận hay bác bỏ xu hướng giá. "
        "Kết luận: XÁC NHẬN / NGHI NGỜ / PHÂN KỲ"
    )
    user = f"""
Cổ phiếu: {symbol}
Tỷ lệ volume hôm nay / TB20 ngày: {indicators['volume_ratio']}x
RSI: {indicators['rsi']}
Giá vs BB_mid: {'+' if indicators['current_price'] > indicators['bb_mid'] else '-'}{abs(round((indicators['current_price']/indicators['bb_mid']-1)*100, 1))}%

Phân tích volume trong 2-3 câu. Kết luận: XÁC NHẬN/NGHI NGỜ/PHÂN KỲ
"""
    return call_deepseek(system, user)


def run_risk_agent(symbol: str, indicators: dict) -> str:
    system = (
        "Bạn là chuyên gia quản lý rủi ro cổ phiếu. "
        "Đánh giá mức rủi ro dựa trên vị trí giá, Bollinger Bands, RSI. "
        "Kết luận: RỦI RO THẤP / TRUNG BÌNH / CAO"
    )
    user = f"""
Cổ phiếu: {symbol}
Giá: {indicators['current_price']:,}
BB Upper: {indicators['bb_upper']:,} | BB Lower: {indicators['bb_lower']:,}
Kháng cự 20 ngày: {indicators['resistance_20d']:,}
Hỗ trợ 20 ngày: {indicators['support_20d']:,}
RSI: {indicators['rsi']}

Đánh giá rủi ro trong 2-3 câu. Kết luận: RỦI RO THẤP/TRUNG BÌNH/CAO
"""
    return call_deepseek(system, user)


def run_verdict_agent(symbol: str, trend: str, volume: str, risk: str) -> str:
    system = (
        "Bạn là trưởng nhóm phân tích, tổng hợp ý kiến từ các chuyên gia. "
        "Đưa ra kết luận cuối cùng: ĐỒNG THUẬN MUA / ĐỒNG THUẬN BÁN / TRUNG LẬP / PHẢN BÁC. "
        "Ngắn gọn, quyết đoán."
    )
    user = f"""
Cổ phiếu: {symbol}

Ý kiến chuyên gia Xu hướng:
{trend}

Ý kiến chuyên gia Volume:
{volume}

Ý kiến chuyên gia Rủi ro:
{risk}

Tổng hợp và đưa ra KẾT LUẬN CUỐI CÙNG trong 3-4 câu.
Kết luận: ĐỒNG THUẬN MUA / ĐỒNG THUẬN BÁN / TRUNG LẬP / PHẢN BÁC
"""
    return call_deepseek(system, user)


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def analyze_stock(symbol: str) -> str:
    """Full analysis pipeline cho 1 mã"""
    # 1. Lấy data
    data = get_stock_data(symbol)
    if not data["success"]:
        return f"❌ Không lấy được dữ liệu {symbol}: {data['error']}"

    # 2. Tính indicators
    try:
        ind = compute_indicators(data["df"])
    except Exception as e:
        return f"❌ Lỗi tính indicators {symbol}: {str(e)}"

    # 3. Chạy 3 agent song song (sequential cho đơn giản)
    try:
        trend_result = run_trend_agent(symbol, ind)
        volume_result = run_volume_agent(symbol, ind)
        risk_result = run_risk_agent(symbol, ind)
        verdict = run_verdict_agent(symbol, trend_result, volume_result, risk_result)
    except Exception as e:
        return f"❌ Lỗi phân tích DeepSeek {symbol}: {str(e)}"

    # 4. Xác định emoji kết luận
    verdict_upper = verdict.upper()
    if "ĐỒNG THUẬN MUA" in verdict_upper:
        verdict_emoji = "🟢"
    elif "ĐỒNG THUẬN BÁN" in verdict_upper:
        verdict_emoji = "🔴"
    elif "PHẢN BÁC" in verdict_upper:
        verdict_emoji = "🔴"
    else:
        verdict_emoji = "🟡"

    # 5. Format message
    now = datetime.now().strftime("%d/%m %H:%M")
    msg = f"""
{verdict_emoji} *Phân tích {symbol}* — {now}

📊 *Data kỹ thuật:*
• Giá: `{ind['current_price']:,.0f}` | 1W: `{ind['change_1w_pct']:+.1f}%` | 1M: `{ind['change_1m_pct']:+.1f}%`
• RSI: `{ind['rsi']}` | MACD Hist: `{ind['macd_hist']:+.4f}`
• Volume vs TB20: `{ind['volume_ratio']}x`
• Hỗ trợ: `{ind['support_20d']:,.0f}` | Kháng cự: `{ind['resistance_20d']:,.0f}`

📈 *Agent Xu hướng:*
{trend_result}

💧 *Agent Volume:*
{volume_result}

⚠️ *Agent Rủi ro:*
{risk_result}

{verdict_emoji} *KẾT LUẬN:*
{verdict}
""".strip()

    return msg


def scan_watchlist(watchlist: list) -> str:
    """Quét nhanh toàn bộ watchlist"""
    results = []
    for symbol in watchlist:
        try:
            data = get_stock_data(symbol, days=30)
            if not data["success"]:
                results.append(f"❌ {symbol}: Không lấy được data")
                continue

            ind = compute_indicators(data["df"])
            rsi = ind["rsi"]
            change_1w = ind["change_1w_pct"]
            vol_ratio = ind["volume_ratio"]

            # Quick signal dựa trên indicators
            signals = []
            if rsi < 35:
                signals.append("RSI quá bán")
            elif rsi > 70:
                signals.append("RSI quá mua")
            if vol_ratio > 1.5:
                signals.append(f"Volume cao {vol_ratio}x")
            if abs(change_1w) > 5:
                signals.append(f"Biến động mạnh {change_1w:+.1f}%")

            if not signals:
                signals.append("Bình thường")

            emoji = "🟢" if change_1w > 0 else "🔴" if change_1w < -2 else "🟡"
            signal_str = " | ".join(signals)
            results.append(
                f"{emoji} *{symbol}* — `{ind['current_price']:,.0f}` ({change_1w:+.1f}%)\n"
                f"   RSI: {rsi} | Vol: {vol_ratio}x | {signal_str}"
            )
        except Exception as e:
            results.append(f"❌ {symbol}: {str(e)[:50]}")

    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    header = f"📋 *Scan Watchlist* — {now}\n{'─'*30}\n"
    footer = "\n\n_Dùng /check <MÃ> để phân tích sâu_"
    return header + "\n".join(results) + footer
