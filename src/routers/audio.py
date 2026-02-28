import os
import uuid
import shutil
import asyncio
import subprocess
from typing import List

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import CurrentUser, get_current_user
from src.database import AnalyzeJob, get_async_session
from src.schemas.audio import AudioHistoryItem
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

async def convert_to_optimized_mp3(input_path: str) -> str:
    """
    Convert audio to optimized mono 64k mp3 using ffmpeg via subprocess.
    """
    output_path = f"/tmp/{uuid.uuid4().hex}.mp3"
    
    # Run ffmpeg asynchronously
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", input_path, 
        "-ac", "1", "-b:a", "64k", output_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode()
        print(f"FFmpeg Error: {error_msg}")
        raise HTTPException(status_code=500, detail="Audio conversion failed.")
        
    return output_path


@router.post("/transcribe")
async def transcribe_audio_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Upload an audio file (mp3/wav) for background processing.
    Returns a job_id immediately.
    """
    # 1. Validate file extension (now accepting more formats since ffmpeg handles it)
    file_extension = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]
    if file_extension not in allowed_extensions:
         raise HTTPException(status_code=400, detail=f"Unsupported file format. Allowed: {allowed_extensions}")

    # 2. Create Job
    job_id = uuid.uuid4()
    # User requested status="processing" immediately
    job = AnalyzeJob(
        id=job_id,
        status="processing",
        result={},
        user_id=current_user.id,
        filename=file.filename,
        source_type="audio",
    )
    session.add(job)
    await session.commit()
    
    # 3. Save file temporarily
    unique_filename = f"{job_id}{file_extension}"
    file_path = os.path.join(TEMP_DIR, unique_filename)
    
    try:
        # Use run_in_executor to avoid blocking the event loop during file I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, save_file_sync, file.file, file_path)
            
        # Convert to optimized MP3 (asynchronous subprocess)
        optimized_path = await convert_to_optimized_mp3(file_path)
        
        # 4. Dispatch Background Task with the OPTIMIZED path
        # Note: audio_processor needs to clean up the optimized_path
        # but we also must ensure we clean up the original file_path.
        # It's safer to let the router clean up the original and pass the optimized one.
        # However, to be completely safe with background tasks, we should let the
        # background task clean up both, or clean up the original here if we don't need it.
        # Actually, since audio_processor receives exactly one path, we'll let it process and clean 
        # the optimized path. We should clean up the RAW file here right after conversion.
        
        background_tasks.add_task(process_audio_job, job_id, optimized_path)
        
        # Fast cleanup of the raw uploaded file, since we have the optimized one
        if os.path.exists(file_path):
             os.remove(file_path)
        
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


@router.get("/history", response_model=List[AudioHistoryItem])
async def get_audio_history(
    session: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user),
) -> List[AudioHistoryItem]:
    """
    Return all audio transcription jobs belonging to the authenticated user,
    ordered by creation time (newest first).
    """
    stmt = (
        select(AnalyzeJob)
        .where(
            AnalyzeJob.user_id == current_user.id,
            AnalyzeJob.source_type == "audio",
        )
        .order_by(AnalyzeJob.created_at.desc())
    )
    result = await session.execute(stmt)
    jobs = result.scalars().all()

    return [
        AudioHistoryItem(
            job_id=job.id,
            filename=job.filename,
            status=job.status,
            created_at=job.created_at,
        )
        for job in jobs
    ]
