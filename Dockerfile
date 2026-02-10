FROM python:3.10-slim

# 1. Install System Dependencies (Crucial for OCR & Audio)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup App
WORKDIR /app

# Upgrade pip to ensure latest functionality
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Code
COPY . .

# 4. Expose Port & Run
EXPOSE 8000

# Use gunicorn with uvicorn workers and 600s timeout
CMD ["gunicorn", "src.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "600"]
