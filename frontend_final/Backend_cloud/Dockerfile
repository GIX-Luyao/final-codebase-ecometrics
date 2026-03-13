FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer for Docker cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for logs (all data storage is in S3)
RUN mkdir -p logs

# Environment defaults (overridden by ECS task definition)
ENV FLASK_DEBUG=false
ENV FLASK_PORT=8000
ENV STORAGE_BACKEND=s3

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "app:app"]
