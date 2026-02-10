FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY docker/init-mock-db.sql ./docker/init-mock-db.sql

CMD ["python", "-m", "src.main"]
