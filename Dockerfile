FROM python:3.12

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \ build-essential \ && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Expose API port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Default command: start API
CMD ["python", "main.py", "api", "--host", "0.0.0.0", "--port", "9000"]
