# === STAGE 1: Build stage ===
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools for compiling tricky packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        git \
        curl \
        cmake \
        rustc \
        cargo \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install into a temporary directory
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# === STAGE 2: Runtime stage ===
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source code
COPY . .

# Default command
CMD ["python", "main.py"]
