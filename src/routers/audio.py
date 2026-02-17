import os
import uuid
import shutil
import asyncio
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_async_session, AnalyzeJob
from src.services.audio_processor import process_audio_job

router = APIRouter(
    prefix="/api/audio",
    tags=["Audio"]
)

TEMP_DIR = "./temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def save_file_sync(file_obj, dest_path):
    """Sync function to save file, to be run in executor."""
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)

@router.post("/transcribe")
async def transcribe_audio_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Upload an audio file (mp3/wav) for background processing.
    Returns a job_id immediately.
    """
    # 1. Validate file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".wav", ".mp3"]:
         raise HTTPException(status_code=400, detail="Unsupported file format. Allowed: .wav, .mp3")

    # 2. Create Job
    job_id = uuid.uuid4()
    # User requested status="processing" immediately
    job = AnalyzeJob(id=job_id, status="processing", result={})
    session.add(job)
    await session.commit()
    
    # 3. Save file temporarily
    unique_filename = f"{job_id}{file_extension}"
    file_path = os.path.join(TEMP_DIR, unique_filename)
    
    try:
        # Use run_in_executor to avoid blocking the event loop during file I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_file_sync, file.file, file_path)
            
        # 4. Dispatch Background Task
        # Critical: process_audio_job must match the signature expected by background_tasks
        background_tasks.add_task(process_audio_job, job_id, file_path)
        
        # 5. Return Immediately
        return {
            "job_id": str(job_id),
            "status": "processing"
        }
        
    except Exception as e:
        # Cleanup if initial save fails
        await session.delete(job)
        await session.commit()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
