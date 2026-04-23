# ============================================================================
# Vibe-Trading Railway Deploy — API only (no frontend build)
# Đặt file này vào root của repo hoangnm999/HKUDS-Vibe-Trading
# ============================================================================
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để cache layer
COPY agent/requirements.txt agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

# Copy toàn bộ source
COPY pyproject.toml LICENSE README.md ./
COPY agent/ agent/

# Install CLI
RUN pip install --no-cache-dir -e .

# Tạo thư mục runtime
RUN mkdir -p agent/runs agent/sessions agent/uploads

# Script tạo agent/.env từ Railway env vars lúc runtime
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8899/health')" || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["vibe-trading", "serve", "--host", "0.0.0.0", "--port", "8899"]
