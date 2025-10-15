# syntax=docker/dockerfile:1.7

############################
# Base builder
############################
FROM python:3.11-slim AS builder

# 1) System sanity
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

# 2) OS deps for wheels you actually use (pandas, lxml/bs4 sometimes need these; psycopg needs libpq)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      libpq-dev \
      curl \
      ca-certificates \
      tzdata \
      libxml2-dev \
      libxslt1-dev \
      && rm -rf /var/lib/apt/lists/*

# 3) Create user so you’re not root inside the container like a gremlin
RUN useradd -m -u 1000 app
WORKDIR /app

# 4) Install Python deps first to leverage Docker layer cache
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install -r requirements.txt

# 5) Copy the code last (so you don’t bust cache every time you sneeze)
COPY . /app

# Keep permissions friendly
RUN chown -R app:app /app
USER app

############################
# Runtime (single-stage is fine; slimming optional)
############################
FROM builder AS runtime
WORKDIR /app

# Expose API port; ETL ignores this, relax
EXPOSE 8000

# Default command is a no-op so compose can override with uvicorn or the ETL
# You can set a helpful default to run the API locally without compose:
CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
