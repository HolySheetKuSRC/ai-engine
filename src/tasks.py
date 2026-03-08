import asyncio
import uuid
import logging
from src.celery_app import celery_app
from src.services.sheet_processor import process_analysis_task
from src.services.audio_processor import process_audio_job

logger = logging.getLogger(__name__)

@celery_app.task(name="src.tasks.process_sheet_task")
def process_sheet_task(job_id_str: str, file_path: str, webhook_url: str = None):
    job_id = uuid.UUID(job_id_str)
    logger.info(f"Starting Celery task process_sheet_task for job {job_id}")
    asyncio.run(process_analysis_task(job_id, file_path, webhook_url))
    return f"Completed sheet task for {job_id}"

@celery_app.task(name="src.tasks.process_audio_task")
def process_audio_task(job_id_str: str, file_path: str):
    job_id = uuid.UUID(job_id_str)
    logger.info(f"Starting Celery task process_audio_task for job {job_id}")
    asyncio.run(process_audio_job(job_id, file_path))
    return f"Completed audio task for {job_id}"
