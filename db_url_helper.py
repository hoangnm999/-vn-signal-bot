# db_url_helper.py
# Parse DATABASE_URL tương thích Railway + Render + local
# Render inject: DATABASE_URL = postgresql://user:pass@host:port/dbname
# Railway inject: DATABASE_URL = postgresql://user:pass@host:port/dbname  (giống)
# Cả 2 đều dùng postgres:// hoặc postgresql:// — cần normalize

import os
import re

def get_db_params() -> dict:
    """
    Parse DATABASE_URL thành dict params cho pg8000.
    Hỗ trợ cả postgres:// và postgresql:// scheme.
    """
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise ValueError("DATABASE_URL chưa được set trong environment variables")

    # Normalize scheme
    url = url.replace("postgresql://", "postgres://", 1)

    # Parse: postgres://user:pass@host:port/dbname
    pattern = r"postgres://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)"
    m = re.match(pattern, url)
    if not m:
        raise ValueError(f"DATABASE_URL format không hợp lệ: {url[:40]}...")

    return {
        "user":     m.group(1),
        "password": m.group(2),
        "host":     m.group(3),
        "port":     int(m.group(4)),
        "database": m.group(5).split("?")[0],  # bỏ query params nếu có
    }


def get_connection():
    """Tạo pg8000 connection từ DATABASE_URL"""
    import pg8000.dbapi
    params = get_db_params()
    return pg8000.dbapi.connect(**params)
