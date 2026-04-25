#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════
#  migrate_to_render.sh
#  Script migrate VN Signal Bot từ Railway sang Render
#  Chạy từ máy local: bash migrate_to_render.sh
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

info()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()    { echo -e "${YELLOW}[!]${NC} $*"; }
error()   { echo -e "${RED}[✗]${NC} $*"; }
header()  { echo -e "\n${BOLD}━━━ $* ━━━${NC}"; }

# ════════════════════════════════════════════════════════════════════
header "BƯỚC 1: Kiểm tra công cụ cần thiết"
# ════════════════════════════════════════════════════════════════════

for tool in git psql pg_dump python3; do
    if command -v "$tool" &>/dev/null; then
        info "$tool OK ($(command -v $tool))"
    else
        warn "$tool chưa cài. Cài theo hướng dẫn trong README."
    fi
done

# ════════════════════════════════════════════════════════════════════
header "BƯỚC 2: Export database từ Railway"
# ════════════════════════════════════════════════════════════════════

echo ""
warn "Bạn cần RAILWAY_DATABASE_URL từ Railway dashboard:"
warn "  Railway → Project → PostgreSQL → Connect → Connection String"
echo ""
read -rp "Dán RAILWAY_DATABASE_URL vào đây: " RAILWAY_DB_URL

if [[ -z "$RAILWAY_DB_URL" ]]; then
    error "DATABASE_URL trống, bỏ qua bước export. Sẽ tạo DB mới trên Render."
    SKIP_EXPORT=true
else
    SKIP_EXPORT=false
    BACKUP_FILE="railway_backup_$(date +%Y%m%d_%H%M%S).sql"
    info "Đang export DB vào $BACKUP_FILE ..."
    pg_dump "$RAILWAY_DB_URL" \
        --no-owner \
        --no-acl \
        --if-exists \
        --clean \
        -f "$BACKUP_FILE" \
        && info "Export thành công: $BACKUP_FILE" \
        || { error "Export thất bại. Kiểm tra kết nối và thử lại."; SKIP_EXPORT=true; }
fi

# ════════════════════════════════════════════════════════════════════
header "BƯỚC 3: Chuẩn bị repo (render.yaml + requirements.txt)"
# ════════════════════════════════════════════════════════════════════

if [[ ! -f "render.yaml" ]]; then
    warn "render.yaml chưa có trong thư mục hiện tại."
    warn "Copy file deploy/render.yaml vào root project rồi commit."
else
    info "render.yaml đã tồn tại"
fi

if [[ ! -f "requirements.txt" ]]; then
    warn "requirements.txt chưa có — Render cần file này để build."
else
    info "requirements.txt OK"
fi

# ════════════════════════════════════════════════════════════════════
header "BƯỚC 4: Push code lên GitHub"
# ════════════════════════════════════════════════════════════════════

echo ""
info "Thêm các file mới vào git:"

FILES_TO_ADD=(
    "render.yaml"
    "requirements.txt"
    "scrapers/"
    "news_sentiment_patch.py"
)

for f in "${FILES_TO_ADD[@]}"; do
    if [[ -e "$f" ]]; then
        git add "$f" 2>/dev/null && info "  git add $f" || warn "  $f không tồn tại, bỏ qua"
    fi
done

# Hỏi có commit không
read -rp "Commit và push lên GitHub? (y/N): " DO_PUSH
if [[ "$DO_PUSH" =~ ^[Yy]$ ]]; then
    git commit -m "chore: add render.yaml + scrapers (F319/Fireant) for Render deploy"
    git push
    info "Push thành công!"
else
    warn "Bỏ qua push. Nhớ push trước khi deploy trên Render."
fi

# ════════════════════════════════════════════════════════════════════
header "BƯỚC 5: Tạo service trên Render"
# ════════════════════════════════════════════════════════════════════

cat <<'EOF'

Làm theo các bước sau trên https://dashboard.render.com:

  1. New → PostgreSQL
     - Name: vn-signal-db
     - Region: Singapore
     - Plan: Free (thử nghiệm) hoặc Starter ($7)
     - → Create Database
     - Copy "Internal Database URL" để dùng ở bước 6

  2. New → Background Worker  (QUAN TRỌNG: chọn Worker, không phải Web Service)
     - Connect GitHub repo của bạn
     - Name: vn-signal-bot
     - Region: Singapore
     - Branch: main
     - Build Command: pip install -r requirements.txt
     - Start Command: python bot.py
     - Plan: Starter ($7) → BẬT "Always On"

  3. Persistent Disk (trong trang service sau khi tạo):
     - Disks → Add Disk
     - Name: vn-signal-data
     - Mount Path: /data
     - Size: 1 GB

EOF

info "Sau khi tạo service, tiếp tục bước 6 để điền env vars."

# ════════════════════════════════════════════════════════════════════
header "BƯỚC 6: Import database vào Render"
# ════════════════════════════════════════════════════════════════════

if [[ "$SKIP_EXPORT" == "false" && -f "${BACKUP_FILE:-}" ]]; then
    echo ""
    read -rp "Dán RENDER_DATABASE_URL (External URL từ Render Postgres): " RENDER_DB_URL

    if [[ -n "$RENDER_DB_URL" ]]; then
        info "Import $BACKUP_FILE vào Render DB..."
        psql "$RENDER_DB_URL" < "$BACKUP_FILE" \
            && info "Import DB thành công!" \
            || warn "Import có warning (có thể do duplicate — thường không sao)"
    else
        warn "Bỏ qua import. Nếu cần, chạy tay: psql <RENDER_URL> < $BACKUP_FILE"
    fi
else
    info "Bỏ qua import DB (không có backup hoặc đã bỏ qua export)."
    info "Bot sẽ tự chạy db_migration.py khi khởi động để tạo schema."
fi

# ════════════════════════════════════════════════════════════════════
header "BƯỚC 7: Env Vars cần điền trên Render"
# ════════════════════════════════════════════════════════════════════

cat <<'EOF'

Vào Render dashboard → service vn-signal-bot → Environment → Add Env Var:

  ┌─────────────────────────┬──────────────────────────────────────────────────┐
  │ Key                     │ Value                                            │
  ├─────────────────────────┼──────────────────────────────────────────────────┤
  │ TELEGRAM_TOKEN          │ <từ BotFather>                                   │
  │ CHAT_ID                 │ <Telegram chat_id của bạn>                       │
  │ ALLOWED_CHAT_IDS        │ <chat_id1,chat_id2,...>                          │
  │ DEEPSEEK_API_KEY        │ <DeepSeek key>                                   │
  │ GROQ_API_KEY            │ <Groq key — free, nhanh>                        │
  │ GEMINI_API_KEY          │ <Gemini key>                                     │
  │ OPENROUTER_API_KEY      │ <OpenRouter key (optional)>                     │
  │ FIREANT_TOKEN           │ <Bearer token từ fireant.vn>                    │
  │ DATABASE_URL            │ <Internal Database URL từ Render Postgres>       │
  │ VIBE_API_URL            │ https://hkuds-vibe-trading-production.up.railway.app │
  │ VIBE_API_KEY            │ <password khớp API_AUTH_KEY bên Vibe>           │
  │ WATCHLIST               │ VCB,HPG,FPT,VNM,MWG,TCB,VHM,SSI,HDB,BID       │
  │ DATA_DIR                │ /data                                            │
  │ REPORT_DIR              │ /data/reports                                    │
  │ TZ                      │ Asia/Ho_Chi_Minh                                 │
  │ PYTHONUNBUFFERED        │ 1                                                │
  └─────────────────────────┴──────────────────────────────────────────────────┘

  ⚠️  DATABASE_URL: dùng "Internal Database URL" (không phải External)
      để tận dụng mạng nội bộ Render, nhanh hơn và miễn phí.

EOF

# ════════════════════════════════════════════════════════════════════
header "BƯỚC 8: Kiểm tra sau deploy"
# ════════════════════════════════════════════════════════════════════

cat <<'EOF'

Sau khi Render deploy xong (3-5 phút), kiểm tra:

  1. Render Logs → xem có lỗi không
     Dòng mong đợi: "Bot started. Polling..."

  2. Telegram → gửi /status
     Bot trả: API keys OK, DB connected, Vibe status

  3. Telegram → gửi /debug_loader VCB
     Kiểm tra vn_loader waterfall: Render không bị block như Railway

  4. Telegram → gửi /check VCB
     Full analysis 16 engines

  5. Telegram → gửi /vibe VCB
     Vibe-Trading swarm (vẫn trên Railway — giữ nguyên)

EOF

# ════════════════════════════════════════════════════════════════════
header "HOÀN TẤT"
# ════════════════════════════════════════════════════════════════════

info "Script hoàn thành. Theo dõi Render Logs để xác nhận deploy thành công."
echo ""
warn "TIPS:"
warn "  - Nếu bot bị sleep (Starter plan) → vào Settings → Always On: ON"
warn "  - Persistent Disk cần mount /data TRƯỚC lần deploy đầu tiên"
warn "  - Fireant token lấy từ: https://fireant.vn → Tài khoản → API Token"
echo ""
