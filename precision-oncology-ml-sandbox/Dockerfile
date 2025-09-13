FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libjpeg62-turbo-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[dev]

COPY src ./src
RUN mkdir -p artifacts/mlruns

CMD ["python", "-m", "pon_ml.genomics.train"]

