"""
batch_scanner.py — STUB (S31 cleanup)

File gốc (~1900 dòng) đã được archive.
Các hàm cốt lõi mà module khác cần đã được chuyển vào db.py:
  - get_last_scan_result  → db.get_last_scan_result
  - load_watchlist        → db.load_watchlist
  - save_scan_result (viết) → db.save_scan_result  (đã có từ trước)

Stub này chỉ để backward-compat nếu còn module nào chưa update import.
"""

from db import get_last_scan_result, load_watchlist, save_scan_result  # noqa: F401

__all__ = ["get_last_scan_result", "load_watchlist", "save_scan_result"]
