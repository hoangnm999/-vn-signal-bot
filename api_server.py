"""
api_server.py — FastAPI REST server cho VN Signal Bot.

Endpoints:
  POST /api/backtest   — Chạy backtest, trả về metrics + chart path
  GET  /health         — Health check
  GET  /api/backtest/status/{job_id} — Poll trạng thái job async

Deploy cùng process với bot.py (uvicorn thread) hoặc chạy riêng port 8080.

Usage:
    # Standalone
    uvicorn api_server:app --host 0.0.0.0 --port 8080

    # Từ bot.py (async startup)
    import asyncio, uvicorn
    asyncio.create_task(uvicorn.Server(uvicorn.Config(app, port=8080)).serve())
"""

from __future__ import annotations

import os
import json
import uuid
import time
import logging
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)

# ── FastAPI ───────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse, FileResponse
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    logger.warning("fastapi/uvicorn chưa cài — api_server sẽ không hoạt động")
    # Tạo stub để bot.py không crash khi import
    class FastAPI:  # type: ignore
        def post(self, *a, **kw): return lambda f: f
        def get(self,  *a, **kw): return lambda f: f
    class BaseModel: pass  # type: ignore
    def Field(*a, **kw): return None  # type: ignore
    app = FastAPI()


# ── Job store (in-memory, đủ cho single-instance Railway) ─────────────────────
_jobs: dict[str, dict] = {}
_JOBS_TTL = 3600  # giữ job tối đa 1 giờ


def _cleanup_old_jobs():
    now = time.time()
    dead = [k for k, v in _jobs.items() if now - v.get("created_at", now) > _JOBS_TTL]
    for k in dead:
        _jobs.pop(k, None)


# ── Request / Response models ─────────────────────────────────────────────────
if _HAS_FASTAPI:
    class BacktestRequest(BaseModel):
        symbol:           str          = Field(..., description="Mã CK VN, VD: VCB")
        days:             int          = Field(365, ge=60, le=1500)
        initial_capital:  float        = Field(100_000_000, ge=1_000_000)
        commission_pct:   float        = Field(0.0015, ge=0, le=0.01)
        slippage_pct:     float        = Field(0.001,  ge=0, le=0.05)
        position_size:    float        = Field(1.0,    ge=0.1, le=1.0)
        allow_short:      bool         = Field(False)
        signal_code:      str          = Field(
            ...,
            description=(
                "Nội dung file signal_engine.py — phải chứa hàm "
                "generate_signals(df) -> pd.Series với giá trị +1/0/-1"
            )
        )

    class BacktestResponse(BaseModel):
        job_id:   str
        status:   str
        message:  str

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VN Signal Bot API",
    description="Backtest API cho thị trường chứng khoán Việt Nam",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "vn-signal-bot-api", "timestamp": time.time()}


# ── POST /api/backtest ─────────────────────────────────────────────────────────
if _HAS_FASTAPI:
    @app.post("/api/backtest", response_model=BacktestResponse)
    async def create_backtest(req: BacktestRequest, background_tasks: BackgroundTasks):
        """
        Tạo job backtest mới (async).

        Body JSON:
        {
          "symbol": "VCB",
          "days": 365,
          "initial_capital": 100000000,
          "signal_code": "import pandas as pd\ndef generate_signals(df):\n    ..."
        }

        Returns: {"job_id": "...", "status": "queued"}
        Poll: GET /api/backtest/status/{job_id}
        """
        _cleanup_old_jobs()

        # Validate symbol
        sym = req.symbol.upper().strip()
        if not sym or not sym.isalnum() or len(sym) > 10:
            raise HTTPException(400, "symbol không hợp lệ (2-10 ký tự chữ/số)")

        # Validate signal_code có generate_signals
        if "generate_signals" not in req.signal_code:
            raise HTTPException(
                400,
                "signal_code phải chứa hàm generate_signals(df: pd.DataFrame) -> pd.Series"
            )

        job_id = str(uuid.uuid4())[:12]
        _jobs[job_id] = {
            "status":     "queued",
            "created_at": time.time(),
            "symbol":     sym,
            "result":     None,
            "error":      None,
        }

        # Chạy backtest trong background
        background_tasks.add_task(_run_backtest_job, job_id, req)

        return BacktestResponse(
            job_id=job_id,
            status="queued",
            message=f"Job {job_id} đã được tạo. Poll tại /api/backtest/status/{job_id}",
        )


    @app.get("/api/backtest/status/{job_id}")
    async def get_backtest_status(job_id: str):
        """
        Kiểm tra trạng thái job backtest.

        Trả về:
        - Khi đang chạy: {"status": "running"}
        - Khi xong:      {"status": "ok", "metrics": {...}, "chart_url": "..."}
        - Khi lỗi:       {"status": "error", "error": "..."}
        """
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} không tồn tại hoặc đã hết hạn")

        if job["status"] in ("queued", "running"):
            elapsed = round(time.time() - job["created_at"], 0)
            return {"status": job["status"], "elapsed_s": elapsed, "job_id": job_id}

        if job["status"] == "error":
            return {"status": "error", "error": job["error"], "job_id": job_id}

        # Done — trả về full result
        result = job["result"]
        # Thêm URL chart nếu có
        chart_url = None
        if result.get("chart_path") and Path(result["chart_path"]).exists():
            chart_url = f"/api/backtest/chart/{job_id}"

        return {
            "status":    "ok",
            "job_id":    job_id,
            "symbol":    result.get("symbol"),
            "metrics":   result.get("metrics"),
            "n_trades":  result.get("n_trades"),
            "trades":    result.get("trades", []),
            "chart_url": chart_url,
            "stdout":    result.get("stdout", "")[-500:],
            "elapsed_s": result.get("elapsed_s"),
        }


    @app.get("/api/backtest/chart/{job_id}")
    async def get_backtest_chart(job_id: str):
        """Download equity curve chart PNG."""
        job = _jobs.get(job_id)
        if not job or job["status"] != "done":
            raise HTTPException(404, "Chart chưa có hoặc job không tồn tại")
        chart_path = job.get("result", {}).get("chart_path", "")
        if not chart_path or not Path(chart_path).exists():
            raise HTTPException(404, "File chart không tồn tại")
        return FileResponse(chart_path, media_type="image/png",
                            filename=f"backtest_{job_id}.png")


# ── Background task ────────────────────────────────────────────────────────────
async def _run_backtest_job(job_id: str, req: "BacktestRequest"):
    """Chạy backtest trong thread pool (không block event loop)."""
    _jobs[job_id]["status"] = "running"
    try:
        result = await asyncio.to_thread(_execute_backtest_sync, req)
        _jobs[job_id]["result"] = result
        _jobs[job_id]["status"] = "done" if result["status"] == "ok" else "error"
        if result["status"] == "error":
            _jobs[job_id]["error"] = result.get("error", "Unknown error")
    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"]  = str(e)
        logger.error(f"Job {job_id} failed: {e}")


def _execute_backtest_sync(req: "BacktestRequest") -> dict:
    """
    Tạo run_dir tạm, viết config.json + signal_engine.py, gọi backtest_engine.
    Chạy trong thread (blocking OK).
    """
    with tempfile.TemporaryDirectory(prefix="vn_backtest_") as tmpdir:
        run_path = Path(tmpdir)
        code_dir = run_path / "code"
        code_dir.mkdir()

        # Ghi config.json
        config = {
            "source":           "entrade",
            "symbol":           req.symbol.upper(),
            "days":             req.days,
            "initial_capital":  req.initial_capital,
            "commission_pct":   req.commission_pct,
            "slippage_pct":     req.slippage_pct,
            "position_size":    req.position_size,
            "allow_short":      req.allow_short,
        }
        (run_path / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Ghi signal_engine.py
        (code_dir / "signal_engine.py").write_text(req.signal_code, encoding="utf-8")

        # Chạy engine
        from backtest_engine import run_vn_backtest
        result = run_vn_backtest(str(run_path))

        # Copy chart ra ngoài tmpdir trước khi tmpdir bị xoá
        if result.get("chart_path") and Path(result["chart_path"]).exists():
            import shutil
            persistent_dir = Path("backtest_charts")
            persistent_dir.mkdir(exist_ok=True)
            dest = persistent_dir / f"{req.symbol}_{int(time.time())}.png"
            shutil.copy(result["chart_path"], dest)
            result["chart_path"] = str(dest)
            result["artifacts"] = result.get("artifacts", {})
            result["artifacts"]["chart"] = str(dest)

        return result


# ── Standalone launch helper ───────────────────────────────────────────────────
def start_api_server(port: int = 8080):
    """
    Khởi động uvicorn server trong thread riêng.
    Gọi từ bot.py post_init nếu muốn chạy cùng process.
    """
    try:
        import uvicorn
        import threading

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        t = threading.Thread(target=server.run, daemon=True)
        t.start()
        logger.info(f"API server started on port {port}")
        return server
    except ImportError:
        logger.warning("uvicorn chưa cài — API server không khởi động")
        return None


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8080)))
