"""
db_migration.py — Migration thêm cột state_vector và trade_plan vào bảng signals.

Chạy 1 lần khi deploy, idempotent (an toàn nếu chạy lại).

Usage:
    python db_migration.py
    hoặc từ bot.py: from db_migration import run_migration; run_migration()
"""
from __future__ import annotations
import logging, os
logger = logging.getLogger(__name__)

MIGRATION_SQL = """
-- Thêm cột state_vector (JSONB) nếu chưa có
ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS state_vector JSONB DEFAULT NULL;

-- Thêm cột trade_plan (JSONB) nếu chưa có
ALTER TABLE signals
    ADD COLUMN IF NOT EXISTS trade_plan JSONB DEFAULT NULL;

-- Index để tìm context nhanh
CREATE INDEX IF NOT EXISTS idx_signals_symbol_created
    ON signals(symbol, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_signals_trade_plan
    ON signals(symbol, created_at DESC)
    WHERE trade_plan IS NOT NULL;
"""


def run_migration() -> tuple[bool, str]:
    """Chạy migration. Trả về (success, message)."""
    try:
        from db import get_conn  # type: ignore
        conn = get_conn()
        cur  = conn.cursor()
        for stmt in MIGRATION_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
        conn.commit()
        cur.close()
        conn.close()
        msg = "Migration OK: state_vector + trade_plan columns added."
        logger.info(msg)
        return True, msg
    except Exception as e:
        msg = f"Migration failed: {e}"
        logger.warning(msg)
        return False, msg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok, msg = run_migration()
    print(f"{'OK' if ok else 'FAIL'}: {msg}")
