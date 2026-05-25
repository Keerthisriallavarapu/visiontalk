FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

COPY visiontalk/ ./visiontalk/

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD curl -fsS http://localhost:8090/health || exit 1

CMD ["python", "-m", "visiontalk.cli", "serve", "--host", "0.0.0.0", "--port", "8090"]
