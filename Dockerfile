FROM python:3.10-slim

# 1. Install System Dependencies (Crucial for OCR & Audio)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup App
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Code
COPY . .

# 4. Expose Port & Run
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
