"""
db.py — PostgreSQL layer cho VN Signal Bot
Chức năng:
  1. Khởi tạo schema (signals + agent_predictions)
  2. Lưu signal sau mỗi /check
  3. Cron job tự chấm điểm agent sau N phiên giao dịch
  4. /report — tổng kết accuracy từng agent
  5. /history <MÃ> — lịch sử signal của mã
"""

import os
import json
import logging
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)

# DATABASE_URL được đọc lazy trong get_conn() để chắc chắn env đã inject

def get_conn():
    """
    Lấy connection PostgreSQL.
    - Thử psycopg2 trước
    - Fallback pg8000.dbapi (tương thích DB-API 2.0, dùng %s như psycopg2)
    """
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise RuntimeError("DATABASE_URL chua duoc set trong environment variables")

    # Thử psycopg2 trước
    try:
        import psycopg2
        return psycopg2.connect(db_url)
    except ImportError:
        pass
    except Exception as e:
        raise RuntimeError(f"psycopg2 connect failed: {e}")

    # Fallback: pg8000.dbapi — tương thích DB-API 2.0, dùng %s như psycopg2
    try:
        import pg8000.dbapi
        import urllib.parse
        p = urllib.parse.urlparse(db_url)
        conn = pg8000.dbapi.connect(
            host=p.hostname,
            port=p.port or 5432,
            database=p.path.lstrip("/"),
            user=p.username,
            password=p.password,
            ssl_context=True,
        )
        return conn
    except ImportError:
        raise RuntimeError("Chua cai pg8000. Kiem tra requirements.txt co 'pg8000'.")
    except Exception as e:
        raise RuntimeError(f"pg8000 connect failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA INIT
# ══════════════════════════════════════════════════════════════════════════════

INIT_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(10)  NOT NULL,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMP WITH TIME ZONE NOT NULL
                        DEFAULT (NOW() + INTERVAL '5 days'),
    verdict_label   VARCHAR(30)  NOT NULL,
    confidence_pct  FLOAT        NOT NULL,
    entry_price     FLOAT        NOT NULL,
    tp              FLOAT,
    sl              FLOAT,
    rr              FLOAT,
    bull_count      INT,
    bear_count      INT,
    agent_verdicts  TEXT,
    market_regime   VARCHAR(20),
    macro_label     VARCHAR(20),
    summary         TEXT,
    negative        TEXT
);

CREATE TABLE IF NOT EXISTS agent_predictions (
    id                  SERIAL PRIMARY KEY,
    signal_id           INT          NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    agent_name          VARCHAR(30)  NOT NULL,
    prediction_label    VARCHAR(30)  NOT NULL,
    metric_type         VARCHAR(30)  NOT NULL,
    target_value        FLOAT        NOT NULL,
    entry_price         FLOAT        NOT NULL,
    entry_date          DATE         NOT NULL,
    check_after_sessions INT         NOT NULL,
    actual_value        FLOAT,
    result              VARCHAR(10)  DEFAULT 'PENDING',
    checked_at          TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol     ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created    ON signals(created_at);
CREATE INDEX IF NOT EXISTS idx_pred_result        ON agent_predictions(result);
CREATE INDEX IF NOT EXISTS idx_pred_entry_date    ON agent_predictions(entry_date);
"""

# Migration an toàn cho DB cũ (dùng tên cột cũ: verdict, confidence, agent_scores)
# Chạy sau INIT_SQL trong init_db() — IF NOT EXISTS / IF EXISTS đảm bảo idempotent
MIGRATION_SQL = """
-- Đổi tên cột cũ sang tên mới nếu còn tên cũ
DO $$
BEGIN
    -- verdict -> verdict_label
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='signals' AND column_name='verdict'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='signals' AND column_name='verdict_label'
    ) THEN
        ALTER TABLE signals RENAME COLUMN verdict TO verdict_label;
    END IF;

    -- confidence -> confidence_pct
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='signals' AND column_name='confidence'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='signals' AND column_name='confidence_pct'
    ) THEN
        ALTER TABLE signals RENAME COLUMN confidence TO confidence_pct;
    END IF;

    -- agent_scores (JSONB) -> agent_verdicts (TEXT)
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='signals' AND column_name='agent_scores'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='signals' AND column_name='agent_verdicts'
    ) THEN
        ALTER TABLE signals RENAME COLUMN agent_scores TO agent_verdicts;
        -- Cast JSONB -> TEXT nếu cần
        ALTER TABLE signals ALTER COLUMN agent_verdicts TYPE TEXT
            USING agent_verdicts::text;
    END IF;

    -- Thêm cột created_at timezone nếu thiếu timezone info
    -- (idempotent — sẽ không làm gì nếu đã đúng type)
END $$;

-- Đảm bảo INDEX vẫn tồn tại sau rename
CREATE INDEX IF NOT EXISTS idx_signals_symbol   ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_created  ON signals(created_at);
CREATE INDEX IF NOT EXISTS idx_signals_expires  ON signals(expires_at);

-- Migration: thêm cột expires_at nếu chưa có (cho DB cũ không có cột này)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='signals' AND column_name='expires_at'
    ) THEN
        ALTER TABLE signals
            ADD COLUMN expires_at TIMESTAMP WITH TIME ZONE
                NOT NULL DEFAULT (NOW() + INTERVAL '5 days');
        -- Backfill: các row cũ chưa có expires_at → set = created_at + 5 days
        UPDATE signals SET expires_at = created_at + INTERVAL '5 days'
        WHERE expires_at IS NULL;
        RAISE NOTICE 'Migration: expires_at column added and backfilled.';
    END IF;
END $$;
"""

def init_db():
    """Khởi tạo schema + migration + alerts table. Gọi 1 lần khi bot start."""
    try:
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute(INIT_SQL)
        cur.execute(MIGRATION_SQL)   # rename cột cũ → tên mới, idempotent
        conn.commit()
        cur.close()
        conn.close()
        logger.info("DB schema initialized OK")
    except Exception as e:
        logger.error(f"init_db error: {e}")
        return False

    # Khởi tạo bảng alerts (tách riêng để lỗi alerts không block bot chính)
    try:
        from alert_system import init_alerts_schema
        init_alerts_schema()
    except Exception as e:
        logger.warning(f"init_alerts_schema failed (non-critical): {e}")

    return True


# ══════════════════════════════════════════════════════════════════════════════
# LƯU SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

# Time horizon (số phiên giao dịch) cho từng agent
AGENT_HORIZONS = {
    "trend":       10,
    "volume":       3,
    "risk":        10,
    "fundamental": 10,
    "smart_money":  5,
    "news":         3,
    "market":      10,
    "macro":        5,
}

# Metric type mỗi agent tự chấm
# PRICE_UP   : giá phiên N > entry_price
# PRICE_DOWN : giá phiên N < entry_price
# NO_SL_HIT  : giá không chạm SL trong N phiên
# VOL_UP     : volume TB N phiên > volume TB 20 phiên tại entry
# NO_DROP3   : giá không giảm quá 3% trong N phiên
AGENT_METRICS = {
    "trend":       ("PRICE_UP",   None),   # target_value = None → dùng entry_price
    "volume":      ("VOL_UP",     None),
    "risk":        ("NO_SL_HIT",  None),   # target_value = sl
    "fundamental": ("PRICE_UP",   None),
    "smart_money": ("PRICE_UP",   0.02),   # cần tăng > 2%
    "news":        ("NO_DROP3",   0.03),   # không giảm > 3%
    "market":      ("PRICE_UP",   None),
    "macro":       ("NO_DROP3",   0.02),
}

def save_signal(symbol: str, verdict: dict, ind: dict,
                agent_verdicts: dict, macro_v: str) -> int:
    """
    Lưu 1 signal vào DB sau khi /check hoàn thành.
    Trả về signal_id hoặc -1 nếu lỗi.
    """
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()

        ap    = verdict["action_plan"]
        price = ind["current_price"]
        today = date.today()

        confidence_val = verdict.get("confidence_pct", verdict.get("confidence", 0))

        # Shelf life theo độ mạnh của tín hiệu:
        # DONG THUAN (đồng thuận cao) → 7 ngày | NGHIENG (đa số) → 5 ngày | TRUNG LAP → 3 ngày
        _verdict_lbl = verdict.get("verdict_label", "")
        if   "DONG THUAN" in _verdict_lbl: _shelf_days = 7
        elif "NGHIENG"    in _verdict_lbl: _shelf_days = 5
        else:                               _shelf_days = 3

        cur.execute("""
            INSERT INTO signals
              (symbol, verdict_label, confidence_pct, entry_price, tp, sl, rr,
               bull_count, bear_count, agent_verdicts, market_regime,
               macro_label, summary, negative,
               expires_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    NOW() + INTERVAL '1 day' * %s)
            RETURNING id
        """, (
            symbol.upper()[:10],
            verdict["verdict_label"][:30],
            float(confidence_val),
            float(price),
            float(ap["tp"]),
            float(ap["sl"]),
            float(ap["rr"]),
            int(verdict["bull_count"]),
            int(verdict["bear_count"]),
            json.dumps(agent_verdicts),
            str(agent_verdicts.get("market", ""))[:20],
            str(macro_v)[:20],
            str(verdict.get("summary", ""))[:500],
            str(verdict.get("negative", ""))[:500],
            _shelf_days,
        ))
        row_id = cur.fetchone()
        signal_id = int(row_id[0]) if row_id else -1
        if signal_id < 0:
            raise ValueError("RETURNING id tra ve None")

        # Insert agent_predictions cho 6 agents hiện tại
        for agent, label in agent_verdicts.items():
            if agent not in AGENT_HORIZONS:
                continue
            horizon    = AGENT_HORIZONS[agent]
            metric, extra = AGENT_METRICS.get(agent, ("PRICE_UP", None))

            if metric == "NO_SL_HIT":
                target = float(ap["sl"])
            elif metric == "PRICE_UP" and extra:
                target = float(price) * (1 + float(extra))
            elif metric == "NO_DROP3":
                target = float(extra or 0.03)
            else:
                target = float(price)

            cur.execute("""
                INSERT INTO agent_predictions
                  (signal_id, agent_name, prediction_label, metric_type,
                   target_value, entry_price, entry_date, check_after_sessions)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (signal_id, str(agent)[:30], str(label)[:30], str(metric)[:30],
                  target, float(price), today, int(horizon)))

        conn.commit()
        cur.close()
        logger.info(f"Signal saved: {symbol} id={signal_id}")
        return signal_id

    except Exception as e:
        logger.error(f"save_signal error: {e}")
        if conn:
            try: conn.rollback()
            except Exception: pass
        return -1
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# CRON JOB — Tự chấm điểm agent
# ══════════════════════════════════════════════════════════════════════════════

def _count_trading_sessions(from_date: date, to_date: date) -> int:
    """Đếm số phiên giao dịch thực (bỏ T7, CN) giữa 2 ngày"""
    count = 0
    d = from_date + timedelta(days=1)
    while d <= to_date:
        if d.weekday() < 5:   # 0=Mon .. 4=Fri
            count += 1
        d += timedelta(days=1)
    return count


def _get_ohlcv_since(symbol: str, from_date: date, sessions_needed: int):
    """
    Lấy OHLCV từ Entrade từ from_date, đủ sessions_needed phiên.
    Trả về DataFrame hoặc None.
    """
    try:
        from analyzer import get_price_data
        # Lấy dư hơn để chắc chắn đủ phiên
        days_buffer = int(sessions_needed * 1.6) + 10
        result = get_price_data(symbol, days_buffer + 10)
        if not result.get("success"):
            return None
        df = result["df"]
        # Lọc các phiên sau from_date
        df = df[df["time"] > from_date.strftime("%Y-%m-%d")].reset_index(drop=True)
        return df if len(df) >= sessions_needed else None
    except Exception as e:
        logger.warning(f"_get_ohlcv_since {symbol}: {e}")
        return None


def _evaluate_prediction(pred: dict, df) -> tuple:
    """
    Chấm điểm 1 prediction dựa trên OHLCV thực.
    Trả về (result, actual_value) — result = WIN / LOSS
    """
    metric   = pred["metric_type"]
    target   = float(pred["target_value"])
    entry    = float(pred["entry_price"])
    n        = pred["check_after_sessions"]

    close  = df["close"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    if metric == "PRICE_UP":
        # Giá phiên N > entry
        actual = float(close.iloc[n - 1]) if len(close) >= n else float(close.iloc[-1])
        result = "WIN" if actual > entry else "LOSS"
        return result, actual

    elif metric == "PRICE_DOWN":
        actual = float(close.iloc[n - 1]) if len(close) >= n else float(close.iloc[-1])
        result = "WIN" if actual < entry else "LOSS"
        return result, actual

    elif metric == "NO_SL_HIT":
        # Giá không chạm SL (target = sl) trong N phiên
        sl          = target
        window_low  = low.iloc[:n]
        min_low     = float(window_low.min())
        result      = "WIN" if min_low > sl else "LOSS"
        return result, min_low

    elif metric == "VOL_UP":
        # Volume TB N phiên tới > volume TB 20 phiên trước (dùng target = avg_vol_entry)
        avg_vol_new = float(volume.iloc[:n].mean()) if len(volume) >= n else float(volume.mean())
        result      = "WIN" if avg_vol_new > target else "LOSS"
        return result, avg_vol_new

    elif metric == "NO_DROP3":
        # Giá không giảm quá target% trong N phiên
        threshold   = entry * (1 - target)
        window_low  = low.iloc[:n]
        min_low     = float(window_low.min())
        result      = "WIN" if min_low > threshold else "LOSS"
        return result, min_low

    return "LOSS", 0.0


def run_evaluation_cron():
    """
    Cron job chạy mỗi ngày 18:00 sau đóng cửa.
    Query tất cả predictions PENDING đã đủ N phiên → chấm điểm → cập nhật DB.
    """
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()

        today = date.today()

        # Lấy tất cả predictions chưa chấm
        cur.execute("""
            SELECT id, signal_id, agent_name, prediction_label,
                   metric_type, target_value, entry_price, entry_date,
                   check_after_sessions
            FROM agent_predictions
            WHERE result = 'PENDING'
            ORDER BY entry_date ASC
        """)
        rows = cur.fetchall()

        # Group theo symbol để giảm số lần gọi API
        from collections import defaultdict
        symbol_map = {}  # signal_id -> symbol
        cur.execute("SELECT id, symbol FROM signals")
        for sid, sym in cur.fetchall():
            symbol_map[sid] = sym

        updated = 0
        for row in rows:
            (pred_id, signal_id, agent_name, pred_label,
             metric_type, target_value, entry_price,
             entry_date, check_after_sessions) = row

            sessions_passed = _count_trading_sessions(entry_date, today)
            if sessions_passed < check_after_sessions:
                continue   # chưa đủ phiên, bỏ qua

            symbol = symbol_map.get(signal_id)
            if not symbol:
                continue

            df = _get_ohlcv_since(symbol, entry_date, check_after_sessions)
            if df is None or df.empty:
                continue

            pred_dict = {
                "metric_type":          metric_type,
                "target_value":         float(target_value) if target_value else 0,
                "entry_price":          float(entry_price) if entry_price else 0,
                "check_after_sessions": int(check_after_sessions),
            }
            result, actual = _evaluate_prediction(pred_dict, df)

            cur.execute("""
                UPDATE agent_predictions
                SET result = %s, actual_value = %s, checked_at = NOW()
                WHERE id = %s
            """, (result, float(actual), int(pred_id)))
            updated += 1

        conn.commit()
        cur.close()
        logger.info(f"Evaluation cron: {updated}/{len(rows)} predictions checked")
        return updated

    except Exception as e:
        logger.error(f"run_evaluation_cron error: {e}")
        if conn:
            try: conn.rollback()
            except Exception: pass
        return 0
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# /report — Tổng kết accuracy
# ══════════════════════════════════════════════════════════════════════════════

def get_report(days: int = 30) -> str:
    """Tổng kết accuracy từng agent trong N ngày gần nhất"""
    # Validate input
    days = max(1, min(int(days), 365))
    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()

        since = date.today() - timedelta(days=days)
        cur.execute("""
            SELECT ap.agent_name,
                   COUNT(*)                                          AS total,
                   SUM(CASE WHEN ap.result = 'WIN' THEN 1 ELSE 0 END) AS wins,
                   AVG(CASE WHEN ap.result = 'WIN' THEN 1.0 ELSE 0.0 END) * 100 AS acc
            FROM agent_predictions ap
            JOIN signals s ON s.id = ap.signal_id
            WHERE ap.result != 'PENDING'
              AND s.created_at >= %s
            GROUP BY ap.agent_name
            ORDER BY acc DESC
        """, (since,))
        rows = cur.fetchall()

        cur.execute("SELECT COUNT(*) FROM signals WHERE created_at >= %s", (since,))
        row_ts = cur.fetchone()
        total_signals = int(row_ts[0]) if row_ts else 0

        cur.execute("""
            SELECT COUNT(*) FROM agent_predictions ap
            JOIN signals s ON s.id = ap.signal_id
            WHERE ap.result = 'PENDING' AND s.created_at >= %s
        """, (since,))
        row_p = cur.fetchone()
        pending = int(row_p[0]) if row_p else 0
        cur.close()

        if not rows:
            return (f"Chua co du lieu du danh gia trong {days} ngay qua.\n"
                    f"Tong signal: {total_signals} | Pending: {pending}")

        lines = [
            f"AGENT ACCURACY ({days} ngay | {total_signals} signals)",
            "─" * 38,
        ]
        for row in rows:
            agent, total, wins, acc = row[0], row[1], row[2], row[3]
            acc_f   = float(acc) if acc is not None else 0.0
            total_i = int(total) if total else 0
            wins_i  = int(wins) if wins else 0
            horizon = AGENT_HORIZONS.get(str(agent), "?")
            bar     = "█" * int(acc_f / 10) + "░" * (10 - int(acc_f / 10))
            lines.append(
                f"  {str(agent):<13} {acc_f:5.1f}%  [{bar}]  "
                f"{wins_i}/{total_i}  ({horizon}p)"
            )

        lines.append("─" * 38)
        lines.append(f"Pending chua chot: {pending} predictions")

        if rows:
            best_acc  = float(rows[0][3]) if rows[0][3] else 0.0
            worst_acc = float(rows[-1][3]) if rows[-1][3] else 0.0
            lines.append("\nINSIGHT:")
            if best_acc >= 70:
                lines.append(f"  Manh nhat: {rows[0][0]} ({best_acc:.0f}%) — nen tang weight")
            if worst_acc < 55:
                lines.append(f"  Yeu nhat:  {rows[-1][0]} ({worst_acc:.0f}%) — nen giam weight")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"get_report error: {e}")
        return f"Loi khi lay report: {e}"
    finally:
        if conn:
            try: conn.close()
            except Exception: pass


def get_history(symbol: str, limit: int = 10) -> str:
    """Lấy lịch sử signal của 1 mã"""
    # Validate + sanitize input
    symbol = symbol.upper().strip()[:10]
    if not symbol.isalpha():
        return "Ma co phieu khong hop le."
    limit = max(1, min(int(limit), 50))

    conn = None
    try:
        conn = get_conn()
        cur  = conn.cursor()

        cur.execute("""
            SELECT created_at, verdict_label, confidence_pct,
                   entry_price, tp, sl, summary,
                   expires_at
            FROM signals
            WHERE symbol = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (symbol, limit))
        rows = cur.fetchall()
        cur.close()

        if not rows:
            return f"Chua co signal nao duoc luu cho {symbol}."

        from datetime import timezone as _tz
        now_utc = datetime.utcnow().replace(tzinfo=_tz.utc)

        lines = [f"LICH SU SIGNAL — {symbol} ({len(rows)} gan nhat)", "─" * 32]
        for i, row in enumerate(rows):
            created_at = row[0]
            verdict    = str(row[1])
            conf       = float(row[2]) if row[2] else 0
            entry      = float(row[3]) if row[3] else 0
            tp         = float(row[4]) if row[4] else 0
            sl         = float(row[5]) if row[5] else 0
            summary    = str(row[6]) if row[6] else ""
            expires_at = row[7]

            if hasattr(created_at, 'strftime'):
                dt = created_at.strftime("%d/%m %H:%M")
            else:
                dt = str(created_at)[:16]

            emoji = "🟢" if "MUA" in verdict else "🔴" if "BAN" in verdict else "🟡"

            # Tag HẾT HẠN nếu expires_at đã qua
            expired_tag = ""
            if expires_at is not None:
                exp = expires_at
                if hasattr(exp, "tzinfo") and exp.tzinfo is None:
                    exp = exp.replace(tzinfo=_tz.utc)
                if now_utc > exp:
                    expired_tag = " [HET HAN]"

            # Tag ĐẢO CHIỀU so với lần trước
            change_tag = ""
            if i < len(rows) - 1:
                prev_v = str(rows[i + 1][1])
                _BULL = {"DONG THUAN MUA", "NGHIENG MUA"}
                _BEAR = {"DONG THUAN BAN", "NGHIENG BAN"}
                if verdict in _BULL and prev_v in _BEAR:
                    change_tag = " ↑ DAO CHIEU"
                elif verdict in _BEAR and prev_v in _BULL:
                    change_tag = " ↓ DAO CHIEU"

            lines.append(
                f"{emoji} {dt}  {verdict}  {conf:.0f}%{expired_tag}{change_tag}\n"
                f"   Entry:{entry:,.1f}  TP:{tp:,.1f}  SL:{sl:,.1f}\n"
                f"   {summary[:80]}"
            )

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"get_history error: {e}")
        return f"Loi khi lay history {symbol}: {e}"
    finally:
        if conn:
            try: conn.close()
            except Exception: pass
