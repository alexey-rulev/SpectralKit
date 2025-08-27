# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY . .
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
