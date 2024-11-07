FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app \
    FLASK_ENV=development \
    FLASK_DEBUG=1

# Debug: List directories
RUN echo "Contents of /app:" && ls -la /app && \
    echo "\nContents of /app/app:" && ls -la /app/app && \
    echo "\nContents of /app/app/templates:" && ls -la /app/app/templates

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--log-level=debug"]