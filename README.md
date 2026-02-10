# AI OCR Service

This is a FastAPI-based AI OCR project (`aitigravity`) designed for deployment on DigitalOcean App Platform.

## Features

- Optical Character Recognition (OCR) using Typhoon OCR
- PDF processing
- AI integration via OpenAI compatible API

## Setup

### Local Development

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Application:**
    ```bash
    uvicorn src.main:app --reload
    ```

### Docker

Build and run the container:

```bash
docker build -t ai-ocr-service .
docker run -p 8000:8000 ai-ocr-service
```
