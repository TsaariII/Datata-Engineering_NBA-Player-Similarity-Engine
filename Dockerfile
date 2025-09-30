# Dockerfile
FROM python:3.11-slim

# Set these at build time if you want, or just run as a user at runtime
ARG UID=1000
ARG GID=1000

# Create app dir and a user/group that match host
RUN groupadd -g ${GID} appgroup \
    && useradd -m -u ${UID} -g appgroup appuser

WORKDIR /app

# Install deps first (better cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only code, not data
COPY data_scraper.py .

# Create the data folder in the image so paths exist,
# but we'll bind-mount a host dir over it at run time
RUN mkdir -p /app/data \
    && chown -R appuser:appgroup /app

USER appuser

# Default: run the scraper (you can override with docker run)
CMD ["python", "data_scraper.py"]