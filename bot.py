import os
import logging
import asyncio
import re
import time
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.error import Conflict, NetworkError, TimedOut
from analyzer import (analyze_stock_full, scan_watchlist,
                      get_price_data, get_market_data, get_news_data)
from db import init_db, save_signal, run_evaluation_cron, get_report, get_history

# ── Logging setup TRƯỚC tất cả optional imports để tránh NameError ──────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Auto Context modules ────────────────────────────────────────────────────
try:
    from auto_context import (
        save_state_vector_to_signal, save_vibe_signal,
        parse_trade_plan_from_text, save_trade_plan_to_signal,
    )
    from state_vector import compute_state_vector_from_df
    _AUTO_CONTEXT = True
except ImportError:
    _AUTO_CONTEXT = False

try:
    from db_migration import run_migration as _run_db_migration
    _run_db_migration()
except Exception:
    pass

try:
    from alert_system import alert_cmd, alerts_cmd, _start_alert_cron
    _ALERT_AVAILABLE = True
except ImportError:
    _ALERT_AVAILABLE = False
    logger.warning("alert_system.py chua co — /alert va /alerts bi tat")
try:
    from local_swarm_cmd import (
        local_swarm_cmd, install_vibe_failover,
        cancel_swarm_cmd, lswarm_fast_cmd,
    )
    _LOCAL_SWARM = True
except ImportError:
    _LOCAL_SWARM = False
    logger.warning("local_swarm_cmd.py chua co — /local_swarm bi tat")

try:
    from backtest_rule_cmd import (
        backtest_rule_cmd,
        backtest_analog_cmd,
        backtest_analog_batch_cmd,
        backtest_analog_detail_cmd,
        walkforward_analog_cmd,
        analog_pipeline_cmd,
        analog_approve_cmd,
        analog_approve_callback,
        analog_configs_cmd,
        analog_remove_cmd,
        analog_regime_analysis_cmd,
        analog_sim_dist_cmd,
        _load_wf_config_from_db,
    )
    _BACKTEST_RULE = True
except ImportError:
    _BACKTEST_RULE = False
    backtest_analog_cmd        = None
    backtest_analog_batch_cmd  = None
    backtest_analog_detail_cmd = None
    walkforward_analog_cmd     = None
    analog_pipeline_cmd        = None
    analog_approve_cmd         = None
    analog_configs_cmd         = None
    analog_remove_cmd          = None
    analog_regime_analysis_cmd = None
    analog_sim_dist_cmd        = None
    analog_approve_callback    = None
    _load_wf_config_from_db    = None
    logger.warning("backtest_rule_cmd.py chua co — /backtest_rule bi tat")

try:
    from analog_signal import (
        signal_status_cmd, signal_history_cmd,
        _start_signal_cron,
    )
    _ANALOG_SIGNAL = True
    logger.info("analog_signal.py loaded OK")
except ImportError:
    _ANALOG_SIGNAL = False
    signal_status_cmd  = None
    signal_history_cmd = None
    logger.warning("analog_signal.py chua co — /signal_status bi tat")

try:
    from analog_cmd import analog_cmd
    _ANALOG = True
    logger.info("analog_cmd.py loaded OK")
except ImportError:
    _ANALOG = False
    logger.warning("analog_cmd.py chua co — /analog bi tat")

try:
    from cluster_scanner import cluster_scan_cmd, _start_cluster_scan_cron
    _CLUSTER_SCANNER = True
    logger.info("cluster_scanner.py loaded OK")
except ImportError:
    _CLUSTER_SCANNER = False
    logger.warning("cluster_scanner.py chua co — /cluster_scan bi tat")

try:
    from wave_pattern import wave_cmd
    _WAVE_PATTERN = True
except ImportError:
    _WAVE_PATTERN = False
    logger.warning("wave_pattern.py chua co — /wave bi tat")

try:
    from market_regime import regime_cmd, get_market_regime, format_regime_inline
    _MARKET_REGIME = True
    logger.info("market_regime.py loaded OK")
except ImportError:
    _MARKET_REGIME = False
    logger.warning("market_regime.py chua co — /regime bi tat")

try:
    from portfolio import buy_cmd, sell_cmd, portfolio_cmd, check_portfolio_alerts, format_morning_digest
    _PORTFOLIO = True
    logger.info("portfolio.py loaded OK")
except ImportError:
    _PORTFOLIO = False
    logger.warning("portfolio.py chua co — /buy /sell /portfolio bi tat")

# morning_briefing disabled S31 — logic Analog Engine, thay bằng cluster_scan cron 08:30+12:30
_MORNING = False

try:
    from vibe_client import (
        is_available as vibe_available,
        start_swarm, poll_swarm, format_swarm_result,
        SWARM_ALIASES, SWARM_LABELS, SWARM_GROUPS, QUICK_ALIASES, resolve_alias,
        build_local_context,
    )
    _VIBE_CLIENT = True
except ImportError:
    _VIBE_CLIENT = False

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

# ── Message split helper ──────────────────────────────────────────────────────
TELEGRAM_MAX = 4096

def _smart_split(text: str) -> list:
    """Tách text tại điểm tự nhiên, mỗi part <= TELEGRAM_MAX."""
    if len(text) <= TELEGRAM_MAX:
        return [text]
    # Tách tại markers tự nhiên theo thứ tự ưu tiên
    for marker in ["\nCHI TIET:", "\nLUU Y:", "\nRisk (", "\n================================"]:
        idx = text.find(marker)
        if 0 < idx < TELEGRAM_MAX:
            part1 = text[:idx].rstrip()
            part2 = text[idx:].lstrip('\n')
            if len(part1) <= TELEGRAM_MAX:
                return [part1] + _smart_split(part2)
    # Tách tại dòng trống gần nhất
    nl_idx = text.rfind('\n\n', 0, TELEGRAM_MAX - 100)
    if nl_idx > 0:
        return [text[:nl_idx].rstrip()] + _smart_split(text[nl_idx:].lstrip())
    # Hard cut
    return [text[:4000]] + _smart_split(text[4000:])


async def _split_and_send(message, text: str):
    """
    Gửi text dài qua Telegram, tự động tách nếu > 4096 ký tự.
    Phần đầu dùng reply_text, các phần sau reply tiếp để tạo thread.
    """
    parts = _smart_split(plain(text))
    sent = None
    for i, part in enumerate(parts):
        if i == 0:
            sent = await message.reply_text(part)
        else:
            # Reply vào message gốc (không phải part trước) để giữ context
            sent = await message.reply_text(part)
    return sent


async def _edit_or_split(msg, message, text: str):
    """
    Với message đã gửi trước (loading indicator):
      - Nếu chỉ 1 part: edit message cũ (trải nghiệm tốt hơn)
      - Nếu nhiều parts: delete/edit part1 + send phần còn lại
    """
    parts = _smart_split(plain(text))
    if len(parts) == 1:
        try:
            await msg.edit_text(parts[0])
        except Exception:
            await message.reply_text(parts[0])
    else:
        # Edit message loading thành part 1
        try:
            await msg.edit_text(parts[0])
        except Exception:
            await message.reply_text(parts[0])
        # Gửi các phần còn lại
        for part in parts[1:]:
            await message.reply_text(part)

# ── Authorization ─────────────────────────────────────────────────────────────
def _allowed_ids() -> set:
    """
    Tập hợp chat_id/user_id được phép dùng bot.
    Set env var:
      CHAT_ID = "123456789"           (user chính)
      ALLOWED_CHAT_IDS = "111,222"    (thêm user/group khác, tùy chọn)
    """
    ids = set()
    main_id = os.environ.get("CHAT_ID", "").strip()
    if main_id:
        ids.add(main_id)
    for part in os.environ.get("ALLOWED_CHAT_IDS", "").split(","):
        part = part.strip()
        if part:
            ids.add(part)
    return ids


def is_allowed(update: Update) -> bool:
    """Chỉ cho phép chat_id trong whitelist. Từ chối tất cả nếu chưa cấu hình."""
    allowed = _allowed_ids()
    if not allowed:
        return False
    user_id = str(update.effective_user.id) if update.effective_user else ""
    chat_id = str(update.effective_chat.id) if update.effective_chat else ""
    return user_id in allowed or chat_id in allowed


async def _deny(update: Update):
    await update.message.reply_text("Khong co quyen su dung bot nay.")


# ── Rate limiting ─────────────────────────────────────────────────────────────
_last_heavy_cmd: dict[str, float] = {}
HEAVY_COOLDOWN_SECS = 30  # /check và /scan tốn API, giới hạn 1 lệnh/30s per user


def _check_rate_limit(user_id: str) -> float:
    """Trả về giây còn lại nếu đang cooldown, 0.0 nếu OK."""
    elapsed = time.time() - _last_heavy_cmd.get(user_id, 0)
    return max(0.0, HEAVY_COOLDOWN_SECS - elapsed)


def _record_cmd(user_id: str):
    _last_heavy_cmd[user_id] = time.time()


# ── Input validation ──────────────────────────────────────────────────────────
def _validate_symbol(raw: str) -> tuple[bool, str]:
    """
    Validate mã cổ phiếu: chỉ chấp nhận 2-10 ký tự chữ/số.
    Trả về (is_valid, cleaned_symbol).
    """
    s = raw.upper().strip()[:10]
    if not re.match(r'^[A-Z0-9]{2,10}$', s):
        return False, s
    return True, s


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_watchlist(context) -> list:
    default = [s.strip() for s in
               os.environ.get("WATCHLIST", "VCB,HPG,FPT,VNM,MWG,TCB ,STB, VCI, VIX, HHS, QCG, CTS, HAH, HTG, LPB, ORS, GIL, VDS, PC1, HPG, FRT, MCH, DGC, VND, BSR, BSI, PDR, CNG, GAS, DPM, DVP, FPT, HBC, VIC, AGG, VCB, CSV, CTG, VTP, DCM, PHP, DXS, HCM, NKG, SSI, MWG, HDB, POW, OCB, MBS, SHS, SZC").split(",")]
    return context.bot_data.get("watchlist", default)


def plain(text: str) -> str:
    """Strip markdown để tránh Telegram parse error."""
    text = re.sub(r"[*`]", "", text)
    text = text.replace("\\_", "_")
    return text


# ── /start ────────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        await _deny(update); return
    msg = (
        "🤖 VN Signal Bot — San sang!\n\n"
        "━━━ PHAN TICH ━━━\n"
        "/check <MA>          — Phan tich 16 engines + Action Plan\n"
        "/vibe <MA>           — Vibe-Trading swarm (35 agents, AI)\n"
        "/local_swarm <MA>    — Hoi dong 5 chuyen gia AI noi bo\n"
        "/lswarm_fast <MA>    — Fast mode: 2 chuyen gia, ~60s\n"
        "/cancel              — Huy tac vu local_swarm dang chay\n"
        "/scan                — Quet nhanh RSI+Vol toan watchlist\n"
        "/deepscan            — Phan tich sau 16 engines toan watchlist\n\n"
        "━━━ BACKTEST & ANALOG ━━━\n"
        "/analog <MA>         — Phan tich tuong dong lich su (regime filter ON)\n"
        "/analog <MA> --raw   — Tat regime filter\n"
        "/backtest_rule <MA>  — Backtest tu dong (Auto Context)\n"
        "/backtest_rule <MA> \"entry_rule\" \"exit_rule\"  — Manual\n"
        "/wave <MA>           — Phan tich song lich su (Wave Pattern)\n\n"
        "━━━ THI TRUONG ━━━\n"
        "/regime              — Market Regime VNINDEX (Bull/Bear x Quiet/Volatile)\n"
        "/regime --history    — Lich su 90 ngay + chart regime\n\n"
        "━━━ CANH BAO GIA ━━━\n"
        "/alert <MA> <gia> [above|below] — Dat canh bao\n"
        "/alerts              — Xem canh bao dang hoat dong\n"
        "/alert cancel <id>   — Huy canh bao\n\n"
        "━━━ PORTFOLIO ━━━\n"
        "/buy <MA> <gia> <SL> <KL> [TP] — Ghi nhan vi the mua\n"
        "/sell <MA> [gia]     — Dong vi the\n"
        "/portfolio           — Tong quan P&L + goi y hanh dong\n"
        "/portfolio <MA>      — Chi tiet 1 ma\n"
        "/portfolio --history — Lich su lenh da dong\n\n"
        "━━━ LICH SU & BAO CAO ━━━\n"
        "/history <MA>        — Lich su signal cua ma\n"
        "/report [ngay]       — Accuracy tung agent (mac dinh 30 ngay)\n\n"
        "━━━ WATCHLIST ━━━\n"
        "/watchlist           — Xem danh sach ma theo doi\n"
        "/add <MA>            — Them ma vao watchlist\n"
        "/remove <MA>         — Xoa ma khoi watchlist\n\n"
        "━━━ HE THONG ━━━\n"
        "/status              — Trang thai API + DB\n"
        "/debug <MA>          — Debug data sources (price/market/news)\n"
        "/debug_loader <MA>   — Debug vn_loader 6 sources waterfall\n"
        "/vibestatus          — Trang thai Vibe-Trading service\n"
        "/vibetest            — Test Vibe-Trading API\n"
        "/help                — Huong dan chi tiet\n\n"
        "Vi du: /check VCB | /vibe HPG | /local_swarm FPT"
    )
    await update.message.reply_text(msg)


# ── /help ─────────────────────────────────────────────────────────────────────
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return

    # Part 1: Phan tich + Song gia + Analog System
    part1 = (
        "📖 VN TRADER BOT — HUONG DAN\n"
        "══════════════════════════════\n\n"

        "🔍 PHAN TICH CO PHIEU\n"
        "──────────────────────\n"
        "/check <MA>\n"
        "  Phan tich 16 engines: Candlestick, Ichimoku, Elliott,\n"
        "  Harmonic, GARCH, SMC, MLStrategy, Fundamental,...\n"
        "  Ket luan: DONG THUAN/NGHIENG/TRUNG LAP + Entry/SL/TP\n"
        "  Cooldown: 30 giay\n\n"
        "/vibe <MA> [swarm]\n"
        "  AI swarm 35 swarms / 71 skills, tranh luan da vong\n"
        "  Vi du: /vibe VCB | /vibe HPG swing\n\n"
        "/local_swarm <MA>  — 5 chuyen gia AI noi bo (2 vong)\n"
        "/lswarm_fast <MA>  — Phien ban nhanh (2 experts, ~60s)\n"
        "/scan              — Quet nhanh RSI + Volume watchlist\n"
        "/deepscan          — 16 engines toan watchlist (5-15 phut)\n"
        "/check_all         — Check nhanh tat ca ma trong watchlist\n"
        "/cluster_scan      — Scan cluster signals (MR + Momentum)\n\n"

        "🌊 SONG GIA & THI TRUONG\n"
        "──────────────────────\n"
        "/wave <MA> [--rebuild]\n"
        "  Song lich su: tim dau hieu truoc song tang/giam\n\n"
        "/analog <MA>\n"
        "  Tim ngay lich su tuong dong (cosine sim 15 chieu)\n"
        "  Du bao forward return 30 ngay\n\n"
        "/regime [--history] [--rebuild]\n"
        "  VNINDEX: Bull/Bear x Quiet/Volatile\n"
        "  --history: 90 ngay | --rebuild: tinh lai cache\n"
    )

    # Part 2: Analog System 3 Tang
    part2 = (
        "📊 ANALOG SYSTEM — 3 TANG\n"
        "══════════════════════════\n\n"

        "TANG 1 — BACKTEST (105 experiments)\n"
        "──────────────────────\n"
        "/backtest_analog <MA>\n"
        "  15 combo x 7 threshold, xep hang Exp → PF → Sharpe\n"
        "  Bo loc: Exp > 0 VA PF >= 1.5\n\n"
        "/backtest_analog_batch <MA1> <MA2>...\n"
        "  Nhieu ma, tim combo tot nhat moi ma\n\n"
        "/backtest_analog_detail <MA> <combo>\n"
        "  Bang 7 nguong cho 1 combo, phan tich rui ro\n\n"

        "TANG 2 — WALK-FORWARD OOS (2025 → nay)\n"
        "──────────────────────\n"
        "/walkforward_analog [MA1 MA2...]\n"
        "  Ma co config: chay nhanh (~2 phut/ma)\n"
        "  Ma CHUA co config: tu dong Pipeline T1+T2 (~10 phut)\n"
        "  Verdict: PASS / PARTIAL / FAIL\n\n"
        "/analog_pipeline <MA1> <MA2>...\n"
        "  T1 + T2 tu dong, toi da 5 ma, ~10 phut/ma\n\n"

        "TANG 3 — LIVE SIGNAL CRON\n"
        "──────────────────────\n"
        "/signal_status   — Signal dang active (chua du 30 ngay)\n"
        "/signal_history  — Lich su + ket qua thuc te\n"
        "  Cron: 15:30 VN | Cooldown: 7 ngay/ma\n"
        "  Ma hien tai: MWG STB HPG GAS DPM DCM\n\n"

        "QUAN LY CONFIG\n"
        "──────────────────────\n"
        "/analog_configs\n"
        "  Xem tat ca config (DB va hardcode)\n\n"
        "/analog_approve <MA> [combo] [threshold] [mae30] [sizing]\n"
        "  Luu config moi vao DB, persist qua restart\n"
        "  Sau khi WF tim duoc PASS: /analog_approve LPB la xong\n\n"
        "/analog_remove <MA>\n"
        "  Xoa config khoi DB, quay ve hardcode neu co\n"
    )

    # Part 3: Backtest Rule + Alert + Portfolio + He thong
    part3 = (
        "🖊 BACKTEST RULE (DSL)\n"
        "══════════════════════════\n"
        "/backtest_rule <MA>\n"
        "  Auto Context: lay trade_plan tu /vibe hoac analog tu /check\n\n"
        "/backtest_rule <MA> \"entry\" \"exit\"\n"
        "  INDICATORS: close, rsi, macd, bb, sma(N), ema(N), atr...\n"
        "  FUNCTIONS: crossover() crossunder() breakout() prev()\n"
        "  EXITS: trailing_stop(5%) take_profit(10%) stop_loss(5%)\n"
        "  Vi du: /backtest_rule VCB \"rsi < 30\" \"rsi > 70\"\n\n"

        "🔔 CANH BAO GIA\n"
        "──────────────────────\n"
        "/alert <MA> <gia> [above|below]  — Dat canh bao\n"
        "/alerts                          — Xem dang hoat dong\n"
        "/alert cancel <id>               — Huy canh bao\n"
        "  Cron: moi 15 phut\n\n"

        "💼 PORTFOLIO\n"
        "──────────────────────\n"
        "/buy <MA> <sl> [gia]   — Ghi nhan mua\n"
        "/sell <MA> <sl> [gia]  — Ghi nhan ban\n"
        "/portfolio             — Danh muc + PnL hien tai\n"
        "  Alert SL/TP: moi 30 phut | Digest: 8:15 sang\n\n"

        "📋 LICH SU & BAO CAO\n"
        "──────────────────────\n"
        "/history <MA>   — Signal cu + tag het han / dao chieu\n"
        "/report [ngay]  — Accuracy tung agent (mac dinh 30 ngay)\n\n"

        "⚙️ HE THONG\n"
        "──────────────────────\n"
        "/status          — API keys + DB + Watchlist\n"
        "/watchlist       — Xem danh sach theo doi\n"
        "/add <MA>        — Them ma vao watchlist\n"
        "/remove <MA>     — Xoa ma khoi watchlist\n"
        "/debug <MA>      — Kiem tra nguon du lieu\n"
        "/vibestatus      — Trang thai Vibe-Trading Railway\n"
        "/vibetest        — Test Vibe API\n\n"

        "══════════════════════════════\n"
        "⚠️  Cong cu phan tich ho tro.\n"
        "KHONG phai khuyen nghi dau tu."
    )

    for part in [part1, part2, part3]:
        await update.message.reply_text(part)



# ── /watchlist, /add, /remove ─────────────────────────────────────────────────
async def watchlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    wl = _get_watchlist(context)
    await update.message.reply_text(f"Watchlist ({len(wl)} ma): {', '.join(wl)}")


async def add_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    if not context.args:
        await update.message.reply_text("Dung: /add VCB"); return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le (2-10 chu cai/so, VD: VCB)."); return

    wl = _get_watchlist(context)
    if len(wl) >= 100:
        await update.message.reply_text("Watchlist da day (toi da 100 ma). Xoa bot roi them."); return
    if symbol not in wl:
        wl.append(symbol)
        context.bot_data["watchlist"] = wl
        await update.message.reply_text(f"Da them {symbol} vao watchlist")
    else:
        await update.message.reply_text(f"{symbol} da co trong watchlist roi")


async def remove_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    if not context.args:
        await update.message.reply_text("Dung: /remove VCB"); return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le."); return

    wl = _get_watchlist(context)
    if symbol in wl:
        wl.remove(symbol)
        context.bot_data["watchlist"] = wl
        await update.message.reply_text(f"Da xoa {symbol} khoi watchlist")
    else:
        await update.message.reply_text(f"{symbol} khong co trong watchlist")


# ── /status ───────────────────────────────────────────────────────────────────
async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    lines = ["Trang thai bot:\n"]
    lines.append(f"  DeepSeek API : {'Co' if os.environ.get('DEEPSEEK_API_KEY') else 'CHUA SET'}")
    lines.append(f"  Gemini API   : {'Co' if os.environ.get('GEMINI_API_KEY') else 'Chua set'}")
    lines.append(f"  VNAI API     : {'Co' if os.environ.get('VNAI_API_KEY') else 'Chua set'}")
    lines.append(f"  Fireant Token: {'Co' if os.environ.get('FIREANT_TOKEN') else 'Chua set'}")
    lines.append(f"  Watchlist    : {len(_get_watchlist(context))} ma")

    # Kiểm tra DB — KHÔNG expose DATABASE_URL
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        lines.append("  PostgreSQL   : CHUA SET DATABASE_URL")
    else:
        try:
            from db import get_conn
            conn = get_conn()
            conn.close()
            lines.append("  PostgreSQL   : Ket noi OK")
        except Exception as e:
            lines.append(f"  PostgreSQL   : LOI — {str(e)[:60]}")

    await update.message.reply_text("\n".join(lines))


# ── /healthcheck ──────────────────────────────────────────────────────────────
async def healthcheck_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Kiểm tra toàn bộ modules và dependencies, trả về trạng thái từng thành phần.
    """
    if not is_allowed(update): await _deny(update); return

    await update.message.reply_text("🔍 Đang kiểm tra hệ thống...")

    ok  = "✅"
    err = "❌"
    warn = "⚠️"
    lines = ["🩺 *HEALTHCHECK*\n"]

    # ── 1. ENV VARS ──────────────────────────────────────────────────────────
    lines.append("*ENV VARS*")
    env_checks = [
        ("TELEGRAM_TOKEN",   "Telegram"),
        ("DATABASE_URL",     "PostgreSQL"),
        ("DEEPSEEK_API_KEY", "DeepSeek"),
        ("GEMINI_API_KEY",   "Gemini"),
        ("CHAT_ID",          "Chat ID"),
    ]
    for key, label in env_checks:
        val = os.environ.get(key, "")
        lines.append(f"  {ok if val else err} {label}")
    lines.append("")

    # ── 2. DATABASE ──────────────────────────────────────────────────────────
    lines.append("*DATABASE*")
    try:
        from db import get_conn
        conn = get_conn(); conn.close()
        lines.append(f"  {ok} Kết nối PostgreSQL")
    except Exception as e:
        lines.append(f"  {err} PostgreSQL: {str(e)[:60]}")

    try:
        from db import load_scan_result
        r = load_scan_result(scan_type="watchlist")
        if r:
            ts = r.get("scanned_at") or r.get("saved_at") or r.get("ts") or "không rõ"
            ranked = len(r.get("ranked", r.get("signals", [])))
            lines.append(f"  {ok} Scan cache (batch cũ): {ranked} mã, lúc {ts}")
        else:
            lines.append(f"  {warn} Scan cache: chưa có")
    except Exception as e:
        lines.append(f"  {err} Scan cache: {str(e)[:60]}")
    lines.append("")

    # ── 3. CORE MODULES ──────────────────────────────────────────────────────
    lines.append("*CORE MODULES*")
    module_checks = [
        ("analyzer",         "analyzer"),
        ("vn_loader",        "vn_loader"),
        ("db",               "db"),
        ("cluster_scanner",  "cluster_scanner"),
    ]
    for mod, label in module_checks:
        try:
            __import__(mod)
            lines.append(f"  {ok} {label}")
        except Exception as e:
            lines.append(f"  {err} {label}: {str(e)[:60]}")
    lines.append("")

    # ── 4. OPTIONAL MODULES ──────────────────────────────────────────────────
    lines.append("*OPTIONAL MODULES*")
    optional_checks = [
        ("backtest_rule_cmd", "backtest_rule_cmd"),
        ("analog_signal",     "analog_signal"),
        ("analog_cmd",        "analog_cmd"),
        ("wave_pattern",      "wave_pattern"),
        ("local_swarm_cmd",   "local_swarm_cmd"),
        ("vibe_client",       "vibe_client"),
        ("portfolio",         "portfolio"),
        ("alert_system",      "alert_system"),
    ]
    for mod, label in optional_checks:
        try:
            __import__(mod)
            lines.append(f"  {ok} {label}")
        except ImportError:
            lines.append(f"  {warn} {label}: không có (optional)")
        except Exception as e:
            lines.append(f"  {err} {label}: {str(e)[:60]}")
    lines.append("")

    # ── 5. DATA SOURCE ───────────────────────────────────────────────────────
    lines.append("*DATA SOURCE*")
    try:
        from vn_loader import load_vn_ohlcv
        df = load_vn_ohlcv("VCB", n_bars=5)
        if df is not None and len(df) > 0:
            last_date = str(df.index[-1])[:10]
            lines.append(f"  {ok} vn_loader: VCB OK (last={last_date})")
        else:
            lines.append(f"  {warn} vn_loader: VCB trả về rỗng")
    except Exception as e:
        lines.append(f"  {err} vn_loader: {str(e)[:60]}")
    lines.append("")

    # ── 6. COMMANDS REGISTERED ───────────────────────────────────────────────
    lines.append("*COMMANDS ACTIVE*")
    handlers = context.application.handlers.get(0, [])
    cmds = sorted([
        h.commands for h in handlers
        if hasattr(h, "commands")
    ])
    cmd_list = ", ".join(f"/{list(c)[0]}" for c in cmds if c)
    lines.append(f"  {ok} {len(cmds)} lệnh: {cmd_list}")

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown"
    )


# ── /debug <MA> ───────────────────────────────────────────────────────────────
async def debug_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    if not context.args:
        await update.message.reply_text("Dung: /debug VCB"); return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le."); return

    msg = await update.message.reply_text(f"Debug {symbol}... (15-30 giay)")
    lines = [f"=== DEBUG {symbol} ===\n"]

    # 1. Price data (Entrade)
    try:
        t0 = time.time()
        price = get_price_data(symbol, 30)
        elapsed = round(time.time() - t0, 1)
        if price["success"]:
            df = price["df"]
            lines.append(f"PRICE (Entrade): OK ({elapsed}s) | {len(df)} bars")
            lines.append(f"  Last close: {df['close'].iloc[-1]:.2f}")
        else:
            lines.append(f"PRICE: FAIL | {price['error'][:80]}")
    except Exception as e:
        lines.append(f"PRICE: ERROR | {str(e)[:80]}")

    # 2. Market data
    try:
        t0 = time.time()
        market = get_market_data()
        elapsed = round(time.time() - t0, 1)
        if market.get("success"):
            lines.append(f"MARKET: OK ({elapsed}s) proxy={market.get('proxy','?')}")
            lines.append(f"  VNINDEX={market['vnindex']} | 5D={market['change_5d']}%")
        else:
            lines.append(f"MARKET: FAIL | {market.get('error','?')[:80]}")
    except Exception as e:
        lines.append(f"MARKET: ERROR | {str(e)[:80]}")

    # 3. News (10 sources)
    try:
        t0 = time.time()
        news = get_news_data(symbol)
        elapsed = round(time.time() - t0, 1)
        if news.get("success"):
            src_summary = news.get("source_summary", {})
            working = [k for k, v in src_summary.items() if v > 0]
            lines.append(f"NEWS: OK ({elapsed}s) | total={news['total']}")
            lines.append(f"  Sources OK: {working}")
            if news['headlines']:
                lines.append(f"  Sample: {news['headlines'][0][:80]}")
        else:
            lines.append(f"NEWS: FAIL | {news.get('error','?')[:80]}")
    except Exception as e:
        lines.append(f"NEWS: ERROR | {str(e)[:80]}")

    result = "\n".join(lines)
    await _edit_or_split(msg, update.message, result)


# ── /check ────────────────────────────────────────────────────────────────────
async def check_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    if not context.args:
        await update.message.reply_text("Vi du: /check VCB"); return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le (VD: VCB, HPG, FPT)."); return

    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    wait = _check_rate_limit(user_id)
    if wait > 0:
        await update.message.reply_text(f"Vui long cho {wait:.0f}s truoc khi /check tiep."); return

    _record_cmd(user_id)
    msg = await update.message.reply_text(f"Dang phan tich {symbol}... (~30-60 giay)")
    try:
        result, meta = await asyncio.to_thread(analyze_stock_full, symbol)
        await _edit_or_split(msg, update.message, result)

        # Gửi regime context ngay sau kết quả /check
        if _MARKET_REGIME:
            try:
                _regime_data = await asyncio.to_thread(get_market_regime)
                _regime_line = format_regime_inline(_regime_data)
                if _regime_line:
                    weights = _regime_data.get("weights", {})
                    _regime_msg = (
                        f"{_regime_line}\n"
                        f"   Tin hieu TANG x{weights.get('bull',1):.2f} | "
                        f"Tin hieu GIAM x{weights.get('bear',1):.2f}\n"
                        f"   /regime --history de xem chi tiet"
                    )
                    await update.message.reply_text(_regime_msg)
            except Exception as _re:
                logger.debug(f"regime context skip: {_re}")

        if meta:
            try:
                sid = save_signal(
                    symbol,
                    meta["verdict"],
                    meta["ind"],
                    meta["agent_verdicts"],
                    meta["macro_v"]["label"] if isinstance(meta["macro_v"], dict) else meta["macro_v"],
                )
                if sid > 0:
                    logger.info(f"Signal saved: {symbol} id={sid}")
                    if _AUTO_CONTEXT:
                        try:
                            from vn_loader import load_vn_ohlcv
                            _sv_df = await asyncio.to_thread(load_vn_ohlcv, symbol, 100)
                            _sv = compute_state_vector_from_df(_sv_df)
                            if _sv:
                                sv_json = {k: v for k, v in _sv.items() if k != "_array"}
                                await asyncio.to_thread(save_state_vector_to_signal, sid, sv_json)
                        except Exception as _sve:
                            logger.debug(f"state_vector save skip: {_sve}")
            except Exception as db_err:
                logger.warning(f"Khong luu duoc signal: {db_err}")

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        await msg.edit_text(f"Loi khi phan tich {symbol}: {str(e)[:200]}")



# ── /check_all ────────────────────────────────────────────────────────────────
CHECK_ALL_COOLDOWN = 600
_last_check_all: dict[str, float] = {}


async def check_all_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /check_all           — skip mã đã check trong 24h
    /check_all --force   — check lại tất cả
    Pipeline: analyze_stock_full → save_signal → save_state_vector → append_today_vector
    """
    if not is_allowed(update): await _deny(update); return
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    since   = time.time() - _last_check_all.get(user_id, 0)
    if since < CHECK_ALL_COOLDOWN:
        await update.message.reply_text(
            f"⏳ Chờ {int(CHECK_ALL_COOLDOWN - since)}s trước khi /check_all lại."
        ); return

    force_flag = "--force" in (context.args or [])
    wl = [s.strip().upper() for s in _get_watchlist(context) if s.strip()]
    if not wl:
        await update.message.reply_text("Watchlist trống."); return

    _last_check_all[user_id] = time.time()
    chat_id = update.effective_chat.id
    msg = await update.message.reply_text(
        f"🔍 /check_all: {len(wl)} mã\n"
        f"Mode: {'🔄 Force' if force_flag else '⏭️ Skip nếu đã check <24h'}\n"
        f"Ước tính {len(wl)*30//60}–{len(wl)*60//60} phút."
    )

    async def _run():
        ok_list, skip_list, fail_list = [], [], []
        progress_lines = []
        for i, sym in enumerate(wl, 1):
            # Skip nếu đã check gần đây
            if not force_flag:
                try:
                    from db import get_conn
                    with get_conn() as conn:
                        row = conn.execute(
                            "SELECT created_at FROM signals "
                            "WHERE symbol=? ORDER BY created_at DESC LIMIT 1", (sym,)
                        ).fetchone()
                        if row and row[0]:
                            from datetime import timezone as _tz
                            created = row[0]
                            if isinstance(created, str):
                                created = datetime.fromisoformat(created)
                            if created.tzinfo is None:
                                created = created.replace(tzinfo=_tz.utc)
                            age_h = (datetime.now(_tz.utc) - created).total_seconds() / 3600
                            if age_h < 24:
                                skip_list.append(sym)
                                progress_lines.append(f"  ⏭️ {sym} (skip {age_h:.0f}h)")
                                continue
                except Exception:
                    pass

            try:
                progress_lines.append(f"  🔄 [{i}/{len(wl)}] {sym}...")
                try:
                    preview = (f"🔍 /check_all {i}/{len(wl)} | "
                               f"OK:{len(ok_list)} Skip:{len(skip_list)} Fail:{len(fail_list)}\n"
                               + "\n".join(progress_lines[-6:]))
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg.message_id,
                        text=preview[:4000])
                except Exception:
                    pass

                result, meta = await asyncio.to_thread(analyze_stock_full, sym)
                if meta is None:
                    fail_list.append(sym)
                    progress_lines[-1] = f"  ❌ {sym}: analyze fail"
                    continue

                # Lưu signal
                sid = 0
                try:
                    sid = save_signal(sym, meta["verdict"], meta["ind"],
                                      meta["agent_verdicts"],
                                      meta["macro_v"]["label"] if isinstance(meta["macro_v"], dict)
                                      else meta["macro_v"])
                except Exception as _dbe:
                    logger.warning(f"check_all save_signal {sym}: {_dbe}")

                # Lưu state_vector (quan trọng cho scan_watchlist)
                sv_ok = False
                if sid > 0 and _AUTO_CONTEXT:
                    try:
                        from vn_loader import load_vn_ohlcv
                        _sv_df = await asyncio.to_thread(load_vn_ohlcv, sym, 100)
                        _sv    = compute_state_vector_from_df(_sv_df)
                        if _sv:
                            sv_json = {k: v for k, v in _sv.items() if k != "_array"}
                            await asyncio.to_thread(save_state_vector_to_signal, sid, sv_json)
                            sv_ok = True
                    except Exception as _sve:
                        logger.debug(f"check_all sv {sym}: {_sve}")

                # Cập nhật vector cache CSV cho historical_analog
                try:
                    from historical_analog import append_today_vector
                    await asyncio.to_thread(append_today_vector, sym)
                except Exception as _ve:
                    logger.debug(f"check_all vec_cache {sym}: {_ve}")

                ok_list.append(sym)
                tag = "✅+vec" if sv_ok else "✅"
                progress_lines[-1] = f"  {tag} {sym}"
                await asyncio.sleep(2)

            except Exception as e:
                fail_list.append(sym)
                progress_lines[-1] = f"  ❌ {sym}: {str(e)[:50]}"
                logger.error(f"check_all {sym}: {e}")

        summary = [
            f"✅ /check_all xong ({len(wl)} mã):",
            f"  OK  : {len(ok_list)} mã",
            f"  Skip: {len(skip_list)} mã (đã check <24h)",
            f"  Fail: {len(fail_list)}" + (f" — {', '.join(fail_list)}" if fail_list else ""),
            "",
            "→ Chạy /scan_watchlist để xem cơ hội tiềm năng.",
        ]
        if fail_list:
            summary.append(f"⚠️ Thử lại bằng /check <MA> cho mã lỗi.")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg.message_id,
                text="\n".join(summary)[:4000])
        except Exception:
            await context.bot.send_message(chat_id=chat_id,
                                           text="\n".join(summary)[:4000])

    asyncio.create_task(_run())

# ── /scan ─────────────────────────────────────────────────────────────────────
async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return

    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    wait = _check_rate_limit(user_id)
    if wait > 0:
        await update.message.reply_text(f"Vui long cho {wait:.0f}s."); return

    _record_cmd(user_id)
    wl = _get_watchlist(context)
    msg = await update.message.reply_text(f"Dang quet {len(wl)} ma... (~30 giay)")
    try:
        result = await asyncio.to_thread(scan_watchlist, wl)
        await _edit_or_split(msg, update.message, result)
    except Exception as e:
        logger.error(f"Scan error: {e}")
        await msg.edit_text(f"Loi khi quet watchlist: {str(e)[:200]}")




# ── /deepscan ─────────────────────────────────────────────────────────────────
# Phân tích toàn diện 13 engines cho từng mã trong watchlist.
# Chạy background để không block bot. Max 5 workers song song.
#
# TODO: migrate watchlist sang DB table (chat_id, symbol) thay bot_data.
# ─────────────────────────────────────────────────────────────────────────────

MAX_DEEPSCAN_WORKERS = 5        # số mã chạy song song tối đa
DEEPSCAN_COOLDOWN    = 300      # 5 phút cooldown giữa 2 lần /deepscan cùng user
_last_deepscan: dict[str, float] = {}


def _deepscan_report_path(suffix: str = "") -> str:
    """Đường dẫn file báo cáo: reports/YYYY-MM-DD[_suffix].txt"""
    import pathlib
    pathlib.Path("reports").mkdir(exist_ok=True)
    date_s = datetime.now().strftime("%Y-%m-%d")
    name   = f"deepscan_{date_s}{('_' + suffix) if suffix else ''}.txt"
    return str(pathlib.Path("reports") / name)


async def _analyze_one(symbol: str) -> dict:
    """
    Chạy analyze_stock_full() trong thread pool.
    Trả về dict chuẩn để deepscan tổng hợp.
    """
    try:
        result, meta = await asyncio.to_thread(analyze_stock_full, symbol)
        if meta is None:
            return {"symbol": symbol, "ok": False, "error": str(result)[:80]}
        v   = meta["verdict"]
        ind = meta["ind"]
        return {
            "symbol":       symbol,
            "ok":           True,
            "verdict":      v["verdict_label"],
            "confidence":   v["confidence_pct"],
            "bull":         v["bull_count"],
            "bear":         v["bear_count"],
            "n":            v["active_agents"],
            "price":        ind["current_price"],
            "rsi":          ind["rsi"],
            "vol_ratio":    ind["volume_ratio"],
            "change_1w":    ind["change_1w_pct"],
            "summary":      v.get("summary", "")[:120],
            "contradictions": v.get("contradictions", []),
            "meta":         meta,
        }
    except Exception as e:
        return {"symbol": symbol, "ok": False, "error": str(e)[:80]}


def _verdict_priority(verdict: str) -> int:
    """Sắp xếp: MUA(0) > TRUNG LAP(1) > BAN(2) > lỗi(3)"""
    if "MUA" in verdict:    return 0
    if "TRUNG" in verdict:  return 1
    if "BAN" in verdict:    return 2
    return 3


def _deepscan_summary_line(r: dict, idx: int, total: int) -> str:
    """Một dòng tóm tắt cho từng mã trong progress update."""
    if not r["ok"]:
        return f"❌ [{idx}/{total}] {r['symbol']}: LOI — {r.get('error','?')}"
    v    = r["verdict"]
    conf = r["confidence"]
    em   = "🟢" if "MUA" in v else "🔴" if "BAN" in v else "🟡"
    return f"{em} [{idx}/{total}] {r['symbol']}: {v} ({conf}%)"


def _build_deepscan_report(results: list, wl: list, elapsed: float) -> str:
    """
    Báo cáo tổng hợp cuối cùng sau khi scan xong.
    Sắp xếp: MUA đầu tiên, rồi TRUNG LẬP, rồi BÁN.
    """
    now_s    = datetime.now().strftime("%d/%m/%Y %H:%M")
    ok_res   = [r for r in results if r["ok"]]
    err_res  = [r for r in results if not r["ok"]]
    sorted_r = sorted(ok_res, key=lambda r: (
        _verdict_priority(r["verdict"]),
        -r["confidence"],       # cùng chiều thì confidence cao hơn lên trước
    ))

    # Stats
    n_buy  = sum(1 for r in ok_res if "MUA"   in r["verdict"])
    n_sell = sum(1 for r in ok_res if "BAN"   in r["verdict"])
    n_neut = sum(1 for r in ok_res if "TRUNG" in r["verdict"])

    lines = [
        f"DEEP SCAN — {now_s}",
        f"Scan {len(wl)} ma | {len(ok_res)} thanh cong | "
        f"{len(err_res)} loi | {elapsed:.0f}s",
        f"MUA:{n_buy}  TRUNG LAP:{n_neut}  BAN:{n_sell}",
        "═" * 38,
    ]

    # Chi tiết từng mã
    prev_priority = -1
    for r in sorted_r:
        pri = _verdict_priority(r["verdict"])
        # Separator giữa nhóm MUA / TRUNG LẬP / BÁN
        if pri != prev_priority and prev_priority != -1:
            lines.append("─" * 38)
        prev_priority = pri

        v    = r["verdict"]
        conf = r["confidence"]
        em   = "🟢" if "MUA" in v else "🔴" if "BAN" in v else "🟡"
        p    = r["price"]
        rsi  = r["rsi"]
        vol  = r["vol_ratio"]
        w1   = r["change_1w"]

        lines.append(
            f"{em} {r['symbol']:<6} {v:<16} {conf:>3}% | "
            f"Gia:{p:>8,.0f}  RSI:{rsi:>4.0f}  Vol:{vol:.1f}x  1W:{w1:>+5.1f}%"
        )
        if r["summary"]:
            lines.append(f"   └ {r['summary'][:90]}")
        for c in r["contradictions"][:1]:   # chỉ hiện 1 mâu thuẫn quan trọng nhất
            lines.append(f"   ⚡ {c[:80]}")

    if err_res:
        lines.append("─" * 38)
        lines.append("LOI:")
        for r in err_res:
            lines.append(f"  ❌ {r['symbol']}: {r.get('error','?')[:60]}")

    lines.append("═" * 38)
    lines.append(f"/check <MA> de phan tich sau tung ma")
    return "\n".join(lines)


async def deepscan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /deepscan — Phân tích toàn diện 13 engines cho toàn bộ watchlist.

    Flow:
      1. Kiểm tra auth + cooldown
      2. Gửi cảnh báo thời gian
      3. Chạy background task (không block)
      4. Background: phân tích max 5 mã song song, update progress
      5. Gửi báo cáo tổng hợp + lưu file
    """
    if not is_allowed(update):
        await _deny(update); return

    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    chat_id = update.effective_chat.id

    # Cooldown riêng cho deepscan (nặng hơn /check)
    elapsed_since = time.time() - _last_deepscan.get(user_id, 0)
    if elapsed_since < DEEPSCAN_COOLDOWN:
        wait = int(DEEPSCAN_COOLDOWN - elapsed_since)
        await update.message.reply_text(
            f"⏳ /deepscan can {wait}s nua moi co the chay lai.\n"
            f"(Cooldown {DEEPSCAN_COOLDOWN//60} phut)"
        )
        return

    wl = [s.strip().upper() for s in _get_watchlist(context) if s.strip()]
    # Dedup
    seen = set()
    wl   = [s for s in wl if s not in seen and not seen.add(s)]

    if not wl:
        await update.message.reply_text("Watchlist trong. Dung /add <MA> de them ma.")
        return

    # Cảnh báo thời gian
    est_mins = max(1, len(wl) * 45 // 60)  # ~45s/mã
    await update.message.reply_text(
        f"⏳ Bat dau Deep Scan {len(wl)} ma...\nUoc tinh ~{est_mins}-{est_mins+5} phut. Ban se nhan bao cao khi hoan tat.\nBot van nhan lenh khac trong khi cho."
    )

    _last_deepscan[user_id] = time.time()

    # Background task
    async def _run_deepscan():
        start_t  = time.time()
        results  = []
        progress_lines = []

        # Progress message (sẽ edit dần)
        prog_msg = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🔄 Dang phan tich 0/{len(wl)}..."
        )

        # Chạy theo batch MAX_DEEPSCAN_WORKERS
        for batch_start in range(0, len(wl), MAX_DEEPSCAN_WORKERS):
            batch   = wl[batch_start:batch_start + MAX_DEEPSCAN_WORKERS]
            tasks   = [_analyze_one(sym) for sym in batch]
            batch_r = await asyncio.gather(*tasks, return_exceptions=True)

            for i, r in enumerate(batch_r):
                global_idx = batch_start + i + 1
                if isinstance(r, Exception):
                    r = {"symbol": batch[i], "ok": False, "error": str(r)[:80]}
                results.append(r)
                line = _deepscan_summary_line(r, global_idx, len(wl))
                progress_lines.append(line)

                # Lưu signal vào DB nếu có meta (best-effort)
                if r.get("ok") and r.get("meta"):
                    try:
                        _mv = r["meta"]["macro_v"]
                        save_signal(
                            r["symbol"],
                            r["meta"]["verdict"],
                            r["meta"]["ind"],
                            r["meta"]["agent_verdicts"],
                            _mv["label"] if isinstance(_mv, dict) else _mv,
                        )
                    except Exception:
                        pass

            # Update progress message sau mỗi batch
            done_count = min(batch_start + MAX_DEEPSCAN_WORKERS, len(wl))
            try:
                prog_text = (
                    f"🔄 Dang phan tich {done_count}/{len(wl)}...\n"
                    + "\n".join(progress_lines[-10:])  # 10 dòng gần nhất
                )
                await prog_msg.edit_text(plain(prog_text)[:4000])
            except Exception:
                pass

        elapsed = time.time() - start_t

        # Build báo cáo tổng hợp
        report = _build_deepscan_report(results, wl, elapsed)

        # Lưu file
        try:
            fpath = _deepscan_report_path()
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(report)
            file_note = f"\n📁 Luu tai: {fpath}"
        except Exception as e:
            file_note = f"\n⚠️ Khong luu duoc file: {e}"

        # Gửi báo cáo cuối (split nếu dài)
        final_header = (
            f"✅ Deep Scan hoan tat! ({elapsed:.0f}s){file_note}\n"
            f"{'═'*38}\n"
        )
        for part in _smart_split(plain(final_header + report)):
            await context.bot.send_message(chat_id=chat_id, text=part)

        # Edit progress message thành "done"
        try:
            await prog_msg.edit_text(
                f"✅ Xong! Da phan tich {len(results)}/{len(wl)} ma trong {elapsed:.0f}s."
            )
        except Exception:
            pass

    asyncio.create_task(_run_deepscan())


# ── /report ───────────────────────────────────────────────────────────────────
async def report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    msg = await update.message.reply_text("Dang tinh toan report...")
    try:
        days = 30
        if context.args:
            try:
                days = max(1, min(int(context.args[0]), 365))
            except ValueError:
                pass
        result = await asyncio.to_thread(get_report, days)
        await _edit_or_split(msg, update.message, result)
    except Exception as e:
        await msg.edit_text(f"Loi khi lay report: {str(e)[:200]}")


# ── /history ──────────────────────────────────────────────────────────────────
async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update): await _deny(update); return
    if not context.args:
        await update.message.reply_text("Vi du: /history VCB"); return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le."); return

    msg = await update.message.reply_text(f"Dang lay lich su {symbol}...")
    try:
        result = await asyncio.to_thread(get_history, symbol)
        await _edit_or_split(msg, update.message, result)
    except Exception as e:
        await msg.edit_text(f"Loi khi lay history {symbol}: {str(e)[:200]}")


# ── Cron job 18:00 mỗi ngày ──────────────────────────────────────────────────
async def _start_cron():
    while True:
        import datetime as _dt
        now_dt = _dt.datetime.now()
        target = now_dt.replace(hour=18, minute=0, second=0, microsecond=0)
        if now_dt >= target:
            target += _dt.timedelta(days=1)
        wait_secs = (target - now_dt).total_seconds()
        logger.info(f"Cron: next run in {wait_secs/3600:.1f}h ({target.strftime('%d/%m %H:%M')})")
        await asyncio.sleep(wait_secs)
        try:
            updated = await asyncio.to_thread(run_evaluation_cron)
            logger.info(f"Cron xong: {updated} predictions da cham diem")
        except Exception as e:
            logger.error(f"Cron error: {e}")
        try:
            from historical_analog import update_all_caches
            _wl = [s.strip() for s in os.environ.get("WATCHLIST","VCB,HPG,FPT").split(",") if s.strip()]
            _cmsg = await asyncio.to_thread(update_all_caches, _wl)
            logger.info(f"Vector cache: {_cmsg}")
        except Exception as _ce:
            logger.debug(f"Cache update skip: {_ce}")


# ── Main ──────────────────────────────────────────────────────────────────────
# ── /vibe — Gọi Vibe-Trading swarm agents thật sự ────────────────────────────
# Lệnh: /vibe <MA> [preset]
# Preset: technical (mặc định), investment, risk, macro, fundamental, sector
# Ví dụ: /vibe VCB
#         /vibe HPG investment
#         /vibe FPT risk

async def vibe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Chạy Vibe-Trading swarm agents (29 swarms / 113 agents)."""
    if not is_allowed(update): await _deny(update); return

    if not _VIBE_CLIENT:
        await update.message.reply_text(
            "vibe_client.py chua duoc deploy.\n"
            "Them file vibe_client.py vao repo va set VIBE_API_URL."
        ); return

    # Nếu không có args → hiện menu
    if not context.args:
        lines = ["VIBE-TRADING — 35 Swarms / 71 Skills / 69 Agents", ""]
        for group, aliases in SWARM_GROUPS.items():
            lines.append(f"{group}:")
            for a in aliases:
                label = SWARM_LABELS.get(a, a)
                lines.append(f"  /vibe <MA> {a:<18} {label}")
            lines.append("")
        lines.append("Vi du: /vibe VCB technical")
        lines.append("       /vibe HPG investment")
        await update.message.reply_text("\n".join(lines))
        return

    valid, symbol = _validate_symbol(context.args[0])
    if not valid:
        await update.message.reply_text("Ma khong hop le (VD: VCB, HPG, FPT)."); return

    raw_alias = context.args[1].lower() if len(context.args) > 1 else "technical"
    alias = resolve_alias(raw_alias)
    if not alias:
        close = [a for a in SWARM_ALIASES if raw_alias in a]
        tip = f"Y ban noi: {', '.join(close[:3])}?" if close else ""
        await update.message.reply_text(
            f"Swarm '{raw_alias}' khong ton tai. {tip}\n"
            f"Goi /vibe de xem danh sach 35 swarms.\n"
            f"Quick alias: ta, wave, vol, cs, vi_mo, tamly, quant, rui_ro, danh_muc..."
        ); return

    if not vibe_available():
        await update.message.reply_text(
            "Vibe-Trading service OFFLINE.\n"
            "Kiem tra /vibestatus"
        ); return

    # Rate limit
    user_id = str(update.effective_user.id) if update.effective_user else "unknown"
    wait = _check_rate_limit(user_id)
    if wait > 0:
        await update.message.reply_text(f"Vui long cho {wait:.0f}s."); return
    _record_cmd(user_id)

    label   = SWARM_LABELS.get(alias, alias)
    chat_id = update.effective_chat.id
    msg = await update.message.reply_text(
        f"Vibe-Trading: {label}\n"
        f"Ma: {symbol} | Dang khoi dong...\n"
        f"(Co the mat 5-15 phut, bot van nhan lenh khac trong khi cho)"
    )
    msg_id = msg.message_id

    # Chạy swarm trong background task — KHÔNG block event loop
    async def _run_swarm_bg():
        try:
            # ── Thu thập local context từ /check để inject vào agents ────
            # Nếu có data OHLCV, chạy quick analysis để lấy signals thực
            extra_vars: dict = {}
            try:
                from analyzer import get_price_data, run_vibe_agents, get_indicators
                df = await asyncio.to_thread(get_price_data, symbol, 300)
                if df is not None and len(df) >= 60:
                    check_data  = await asyncio.to_thread(run_vibe_agents, symbol, df)
                    ind         = await asyncio.to_thread(get_indicators, symbol, df)
                    extra_vars  = build_local_context(symbol, {**check_data, "indicators": ind})
                    logger.info(f"vibe context injected for {symbol}: "
                                f"bull={check_data.get('bull',0)} bear={check_data.get('bear',0)}")
            except Exception as e:
                logger.warning(f"Could not build local context for {symbol}: {e}")
                # Không block — tiếp tục chạy vibe mà không có context

            # Start swarm (với hoặc không có local context)
            run_id = await asyncio.to_thread(start_swarm, alias, symbol,
                                             extra_vars=extra_vars or None)
            if not run_id:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg_id,
                    text=f"Loi: Khong khoi dong duoc swarm {alias}.\n"
                         f"Kiem tra VIBE_API_KEY."
                )
                return

            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=msg_id,
                text=f"Vibe-Trading: {label}\n"
                     f"Ma: {symbol} | Run ID: {run_id[:12]}...\n"
                     f"Agents dang khoi dong..."
            )

            # Poll với progress update định kỳ
            start_t = time.time()
            last_update = 0

            while True:
                elapsed = time.time() - start_t
                if elapsed > 900:
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg_id,
                        text=f"Timeout: Swarm chay qua 15 phut.\nThu lai sau."
                    )
                    return

                # Poll 1 lần
                try:
                    r = await asyncio.to_thread(
                        lambda: __import__('requests').get(
                            f"{__import__('vibe_client').VIBE_API_URL}/swarm/runs/{run_id}",
                            headers={"Authorization": f"Bearer {__import__('os').environ.get('VIBE_API_KEY','')}",
                                     "Content-Type": "application/json"},
                            timeout=15
                        )
                    )
                    data = r.json()
                except Exception as e:
                    logger.warning(f"poll error: {e}")
                    await asyncio.sleep(8)
                    continue

                status  = data.get("status", "")
                tasks   = data.get("tasks", [])
                n       = len(tasks)
                done    = sum(1 for t in tasks if t.get("status") in
                              ("completed","failed","cancelled"))
                in_prog = [t.get("agent_id", t.get("id","?")) for t in tasks
                           if t.get("status") == "in_progress"]

                # Update message mỗi 15s
                if time.time() - last_update > 15:
                    last_update = time.time()
                    prog = f"Agents: {done}/{n} xong ({int(elapsed)}s)"
                    if in_prog: prog += f"\nDang chay: {', '.join(in_prog[:3])}"
                    try:
                        await context.bot.edit_message_text(
                            chat_id=chat_id, message_id=msg_id,
                            text=f"Vibe-Trading: {label}\nMa: {symbol}\n\n{prog}"
                        )
                    except Exception:
                        pass

                if status == "completed":
                    final_report = data.get("final_report", "")
                    if not final_report and tasks:
                        done_tasks = [t for t in tasks if t.get("status")=="completed"]
                        if done_tasks:
                            last_t = done_tasks[-1]
                            final_report = f"[{last_t.get('agent_id','?')}]\n{last_t.get('summary','')}"

                    header = (f"VIBE-TRADING — {label}\n"
                              f"Ma: {symbol} | {done}/{n} agents | "
                              f"{int(elapsed)//60}p{int(elapsed)%60}s\n"
                              f"{'='*32}\n\n")
                    full_output = header + (final_report or "Khong co bao cao.")
                    parts = _smart_split(plain(full_output))

                    # Edit message loading thành part 1
                    try:
                        await context.bot.edit_message_text(
                            chat_id=chat_id, message_id=msg_id,
                            text=parts[0]
                        )
                    except Exception:
                        await context.bot.send_message(chat_id=chat_id, text=parts[0])

                    # Gửi phần còn lại (nếu có)
                    for part in parts[1:]:
                        await context.bot.send_message(chat_id=chat_id, text=part)
                    return

                if status in ("failed", "cancelled"):
                    failed = [t for t in tasks if t.get("status")=="failed"]
                    err = failed[0].get("error","")[:100] if failed else ""
                    await context.bot.edit_message_text(
                        chat_id=chat_id, message_id=msg_id,
                        text=f"Swarm {status}: {label}\nMa: {symbol}\n"
                             f"Loi: {err or 'Kiem tra Vibe-Trading logs'}"
                    )
                    return

                # Nếu tasks rỗng và < 60s → đang init
                if n == 0 and elapsed < 60:
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(8)

        except Exception as e:
            logger.error(f"_run_swarm_bg error: {e}")
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id, message_id=msg_id,
                    text=f"Loi khi chay Vibe-Trading {alias}: {str(e)[:200]}"
                )
            except Exception:
                pass

    # Tạo background task — không await, event loop tiếp tục nhận lệnh khác
    asyncio.create_task(_run_swarm_bg())


async def vibe_status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Kiểm tra Vibe-Trading service + liệt kê swarms."""
    if not is_allowed(update): await _deny(update); return

    if not _VIBE_CLIENT:
        await update.message.reply_text("vibe_client.py chua duoc cai dat."); return

    # Lấy URL đã normalize từ vibe_client
    try:
        from vibe_client import VIBE_API_URL as _vibe_url
    except Exception:
        _vibe_url = os.environ.get("VIBE_API_URL", "")

    msg = await update.message.reply_text("Dang kiem tra Vibe-Trading service...")

    online  = vibe_available()
    lines = [
        "Vibe-Trading Service:",
        f"  URL    : {_vibe_url or 'CHUA SET VIBE_API_URL'}",
        f"  Status : {'Online ✓' if online else 'OFFLINE ✗'}",
        f"  API Key: {'Co' if os.environ.get('VIBE_API_KEY') else 'Chua set (dev mode)'}",
        f"  Swarms : {len(SWARM_ALIASES)} presets | 71 skills | 69 agents",
    ]
    if not online:
        lines += [
            "",
            "Nguyen nhan co the:",
            "  1. VIBE_API_URL sai (phai co https://)",
            "  2. Vibe Railway service chua chay",
            "  3. Network/firewall block",
            "",
            "Dung /vibetest de debug chi tiet",
        ]
    else:
        lines += ["", "Dung: /vibe <MA> <swarm>", "Xem list: /vibe"]

    await msg.edit_text("\n".join(lines))


async def vibetest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Debug chi tiết kết nối Vibe-Trading — gọi /health và in raw response."""
    if not is_allowed(update): await _deny(update); return

    if not _VIBE_CLIENT:
        await update.message.reply_text("vibe_client.py chua duoc cai dat."); return

    try:
        from vibe_client import VIBE_API_URL as _vibe_url
    except Exception:
        _vibe_url = os.environ.get("VIBE_API_URL", "")

    msg = await update.message.reply_text(f"Testing: {_vibe_url}/health ...")

    lines = [f"Vibe-Trading Debug", f"URL: {_vibe_url}"]

    if not _vibe_url:
        await msg.edit_text("VIBE_API_URL chua duoc set tren bot service!"); return

    # Test 1: /health
    import requests as _req
    try:
        r = _req.get(f"{_vibe_url}/health", timeout=15)
        lines.append(f"GET /health → HTTP {r.status_code}")
        if r.status_code == 200:
            lines.append(f"Response: {str(r.json())[:100]}")
        else:
            lines.append(f"Body: {r.text[:100]}")
    except _req.exceptions.ConnectionError as e:
        lines.append(f"GET /health → ConnectionError")
        lines.append(f"  {str(e)[:120]}")
    except _req.exceptions.Timeout:
        lines.append(f"GET /health → Timeout (15s)")
    except Exception as e:
        lines.append(f"GET /health → {type(e).__name__}: {str(e)[:100]}")

    # Test 2: /swarm/presets với auth header
    api_key = os.environ.get("VIBE_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r2 = _req.get(f"{_vibe_url}/swarm/presets",
                      headers=headers, timeout=15)
        lines.append(f"GET /swarm/presets → HTTP {r2.status_code}")
        if r2.status_code == 200:
            data = r2.json()
            n = len(data) if isinstance(data, list) else "?"
            lines.append(f"  {n} presets available")
        elif r2.status_code == 401:
            lines.append("  → 401 Unauthorized: VIBE_API_KEY sai hoac thieu")
        else:
            lines.append(f"  Body: {r2.text[:80]}")
    except Exception as e:
        lines.append(f"GET /swarm/presets → {type(e).__name__}: {str(e)[:80]}")

    await msg.edit_text("\n".join(lines))


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler — log lỗi nhưng KHÔNG crash bot."""
    err = context.error
    if isinstance(err, Conflict):
        # 2 instances đang chạy — chờ instance cũ tự dừng, không crash
        logger.warning(
            "Telegram 409 Conflict — co 2 bot instances dang chay. "
            "Instance nay se tiep tuc sau khi instance cu dung."
        )
        return
    if isinstance(err, (NetworkError, TimedOut)):
        logger.warning(f"Network error (se tu retry): {err}")
        return
    # Lỗi khác → log đầy đủ
    logger.error(f"Unhandled exception: {err}", exc_info=err)


# ── Portfolio cron ─────────────────────────────────────────────────────────────

async def _start_portfolio_cron(bot, chat_ids: list):
    """
    Portfolio alert cron:
      - Tầng 1 (CRITICAL): kiểm tra mỗi 30 phút → push ngay nếu có SL/TP/vol breach
      - Tầng 2 (DIGEST): tổng hợp 1 lần/ngày vào 8:15 AM, gộp với morning report

    chat_ids: danh sách user_id được push alert (đọc từ ALLOWED_CHAT_IDS).
    """
    if not _PORTFOLIO:
        return

    import asyncio as _asyncio

    logger.info(f"Portfolio cron started: {len(chat_ids)} users, check every 30min")

    # Track để tránh gửi digest 2 lần trong cùng 1 ngày
    _digest_sent_date = {}

    while True:
        try:
            now = datetime.now()
            hour, minute = now.hour, now.minute

            for user_id in chat_ids:
                try:
                    # ── Tầng 1: CRITICAL — kiểm tra mỗi 30 phút ─────────────
                    critical, digest_items = await _asyncio.to_thread(
                        check_portfolio_alerts, user_id
                    )

                    for alert_msg in critical:
                        try:
                            await bot.send_message(chat_id=user_id, text=alert_msg)
                            logger.info(f"Portfolio CRITICAL alert → user {user_id}")
                        except Exception as e:
                            logger.warning(f"Portfolio alert send error {user_id}: {e}")

                    # ── Tầng 2: DIGEST — chỉ gửi 1 lần lúc 8:15 AM ──────────
                    today_str = now.strftime("%Y-%m-%d")
                    already_sent = _digest_sent_date.get(user_id) == today_str

                    if (hour == 8 and 15 <= minute <= 25
                            and digest_items and not already_sent):
                        # Lấy regime để gắn vào digest
                        regime_label = ""
                        try:
                            from market_regime import get_market_regime
                            mr = await _asyncio.to_thread(get_market_regime)
                            if mr:
                                regime_label = mr.get("label", "")
                        except Exception:
                            pass

                        digest_text = format_morning_digest(digest_items, regime_label)
                        if digest_text:
                            try:
                                await bot.send_message(chat_id=user_id,
                                                       text=digest_text)
                                _digest_sent_date[user_id] = today_str
                                logger.info(f"Portfolio digest sent → user {user_id}")
                            except Exception as e:
                                logger.warning(f"Portfolio digest send error: {e}")

                except Exception as e:
                    logger.error(f"Portfolio cron user {user_id}: {e}")

        except Exception as e:
            logger.error(f"_start_portfolio_cron outer error: {e}")

        # Ngủ 30 phút rồi check lại
        await _asyncio.sleep(30 * 60)


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN chua duoc set")

    allowed = _allowed_ids()
    if not allowed:
        logger.warning("CANH BAO: CHAT_ID/ALLOWED_CHAT_IDS chua set — bot se tu choi tat ca!")
    else:
        logger.info(f"Authorization OK: {len(allowed)} id duoc phep")

    try:
        init_db()
        logger.info("DB initialized OK")
    except Exception as e:
        logger.warning(f"DB init failed: {e}")

    # Load analog config từ DB — merge với hardcode defaults
    if _BACKTEST_RULE and _load_wf_config_from_db:
        try:
            _load_wf_config_from_db()
        except Exception as e:
            logger.warning(f"load_wf_config_from_db failed (using defaults): {e}")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",        start))
    app.add_handler(CommandHandler("help",         help_cmd))
    app.add_handler(CommandHandler("watchlist",    watchlist_cmd))
    app.add_handler(CommandHandler("add",          add_cmd))
    app.add_handler(CommandHandler("remove",       remove_cmd))
    app.add_handler(CommandHandler("status",       status_cmd))
    app.add_handler(CommandHandler("healthcheck",  healthcheck_cmd))
    app.add_handler(CommandHandler("debug",        debug_cmd))
    app.add_handler(CommandHandler("check",        check_cmd))
    app.add_handler(CommandHandler("scan",         scan_cmd))
    app.add_handler(CommandHandler("report",       report_cmd))
    app.add_handler(CommandHandler("history",      history_cmd))
    app.add_handler(CommandHandler("vibe",         vibe_cmd))
    app.add_handler(CommandHandler("vibestatus",   vibe_status_cmd))
    app.add_handler(CommandHandler("vibetest",     vibetest_cmd))
    app.add_handler(CommandHandler("deepscan",    deepscan_cmd))
    app.add_handler(CommandHandler("check_all",   check_all_cmd))
    if _BACKTEST_RULE:
        app.add_handler(CommandHandler("backtest_rule",          backtest_rule_cmd))
        app.add_handler(CommandHandler("backtest_analog",        backtest_analog_cmd))
        app.add_handler(CommandHandler("backtest_analog_batch",  backtest_analog_batch_cmd))
        app.add_handler(CommandHandler("backtest_analog_detail", backtest_analog_detail_cmd))
        app.add_handler(CommandHandler("walkforward_analog",     walkforward_analog_cmd))
        app.add_handler(CommandHandler("analog_pipeline",        analog_pipeline_cmd))
        app.add_handler(CommandHandler("analog_approve",         analog_approve_cmd))
        app.add_handler(CommandHandler("analog_configs",         analog_configs_cmd))
        app.add_handler(CommandHandler("analog_remove",          analog_remove_cmd))
        from telegram.ext import CallbackQueryHandler
        app.add_handler(CallbackQueryHandler(
            analog_approve_callback, pattern=r"^approve_(yes|no)_"
        ))
        app.add_handler(CommandHandler("analog_regime_analysis", analog_regime_analysis_cmd))
        app.add_handler(CommandHandler("analog_sim_dist",        analog_sim_dist_cmd))
        if _ANALOG:
            app.add_handler(CommandHandler("analog", analog_cmd))
    if _ANALOG_SIGNAL:
        app.add_handler(CommandHandler("signal_status",  signal_status_cmd))
        app.add_handler(CommandHandler("signal_history", signal_history_cmd))
    if _CLUSTER_SCANNER:
        app.add_handler(CommandHandler("cluster_scan",   cluster_scan_cmd))
    if _WAVE_PATTERN:
        app.add_handler(CommandHandler("wave", wave_cmd))
    if _MARKET_REGIME:
        app.add_handler(CommandHandler("regime", regime_cmd))
    if _LOCAL_SWARM:
        app.add_handler(CommandHandler("local_swarm",  local_swarm_cmd))
        app.add_handler(CommandHandler("cancel",       cancel_swarm_cmd))
        app.add_handler(CommandHandler("lswarm_fast",  lswarm_fast_cmd))   # fast mode: 2 experts, ~60s
        install_vibe_failover(app)   # failover khi Vibe offline
    if _ALERT_AVAILABLE:
        app.add_handler(CommandHandler("alert",    alert_cmd))
        app.add_handler(CommandHandler("alerts",   alerts_cmd))
    if _PORTFOLIO:
        app.add_handler(CommandHandler("buy",       buy_cmd))
        app.add_handler(CommandHandler("sell",      sell_cmd))
        app.add_handler(CommandHandler("portfolio", portfolio_cmd))
    # /morning disabled S31
    app.add_error_handler(_error_handler)

    async def post_init(application):
        asyncio.create_task(_start_cron())
        # Cluster scanner — đọc chat_ids từ env vars
        _raw_ids = (
            os.environ.get("ALLOWED_CHAT_IDS", "") or
            os.environ.get("ALLOWED_IDS", "") or
            os.environ.get("CHAT_ID", "")
        )
        _scan_ids = [
            int(x.strip()) for x in _raw_ids.split(",")
            if x.strip().lstrip("-").isdigit()
        ]
        if _CLUSTER_SCANNER and _scan_ids:
            asyncio.create_task(_start_cluster_scan_cron(application.bot, _scan_ids))
            logger.info(f"ClusterScanCron: {len(_scan_ids)} chat_ids, 08:30+12:30 VN daily")
        elif not _scan_ids:
            logger.warning("ClusterScanCron: khong co chat_id — set ALLOWED_CHAT_IDS")
        if _ALERT_AVAILABLE:
            asyncio.create_task(_start_alert_cron(application))
        if _PORTFOLIO:
            asyncio.create_task(_start_portfolio_cron(application.bot, _scan_ids))
        # morning cron disabled S31
        if _ANALOG_SIGNAL:
            _signal_ids = _scan_ids
            if _signal_ids:
                asyncio.create_task(_start_signal_cron(application.bot, _signal_ids))
                logger.info(f"AnalogSignalCron: {len(_signal_ids)} chat_ids, 15:30 VN daily")

    app.post_init = post_init

    logger.info("Bot dang chay...")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
