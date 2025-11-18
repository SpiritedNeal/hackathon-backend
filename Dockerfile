# Dockerfile - reproducible environment for Railway
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# install minimal system deps (kept small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install python deps
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port and run uvicorn using the correct module name
ENV PORT 8080
CMD ["python", "-m", "uvicorn", "Quadcore:app", "--host", "0.0.0.0", "--port", "8080"]
