FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data ./data
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
# Secrets are injected at runtime (ECS task secrets / Secrets Manager), never baked into the image:
#   WFR_API_KEY, and for real LLM use WFR_LLM=bedrock + an IAM task role (no static AWS keys).
ENV WFR_LLM=mock WFR_TRACE_DIR=/data/traces WFR_CHECKPOINT_DB=/data/checkpoints.sqlite WFR_RELEASE_DIR=/data/released
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "wfr.api:app", "--host", "0.0.0.0", "--port", "8000"]
