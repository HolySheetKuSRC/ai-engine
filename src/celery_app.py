import os
from celery import Celery

# Use Redis as broker and backend
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")

celery_app = Celery(
    "ai_ocr_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True
)
