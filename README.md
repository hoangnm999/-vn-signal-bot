# VN Signal Bot 🤖

Bot Telegram phân tích cổ phiếu Việt Nam dùng multi-agent AI (DeepSeek) + vnstock data.

## Lệnh bot

| Lệnh | Mô tả |
|------|-------|
| `/check VCB` | Phân tích sâu 1 mã (3 agent + verdict) |
| `/scan` | Quét nhanh toàn bộ watchlist |
| `/watchlist` | Xem danh sách theo dõi |
| `/help` | Hướng dẫn |

## Deploy lên Railway

### 1. Fork/Clone repo này lên GitHub

### 2. Tạo project trên Railway
- Vào [railway.app](https://railway.app)
- New Project → Deploy from GitHub repo
- Chọn repo này

### 3. Set Environment Variables trên Railway
Vào tab **Variables**, thêm:

```
TELEGRAM_TOKEN=your_bot_token
CHAT_ID=your_chat_id
DEEPSEEK_API_KEY=your_deepseek_key
WATCHLIST=VCB,HPG,FPT,VNM,MWG,TCB
```

### 4. Deploy
Railway tự động build và chạy.

## Kiến trúc

```
Telegram Command (/check VCB)
        ↓
Railway (bot.py)
        ↓
vnstock → OHLCV data 90 ngày
        ↓
compute_indicators() → RSI, MACD, BB, Volume...
        ↓
DeepSeek Agent 1: Xu hướng
DeepSeek Agent 2: Volume
DeepSeek Agent 3: Rủi ro
        ↓
DeepSeek Verdict Agent: Tổng hợp
        ↓
Telegram Response
```

## Lưu ý
- Mỗi lệnh `/check` tốn ~4 API calls DeepSeek (~1-2 phút)
- `/scan` chỉ dùng indicators, không gọi AI (nhanh hơn)
- Bot chỉ phản hồi đúng CHAT_ID đã cấu hình (bảo mật)
