import os
import uuid
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import CurrentUser, get_current_user
from src.database import AnalyzeJob, get_async_session
from src.schemas.audio import AudioHistoryItem, AudioResultUpdate
from src.services.audio_processor import process_audio_job

from src.tasks import process_audio_task

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
            
        process_audio_task.delay(str(job_id), file_path)
        
        # 5. Return Immediately
        return {
            "job_id": str(job_id),
            "status": "processing"
        }
        
    except HTTPException:
        raise
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



@router.delete("/jobs/{job_id}")
async def delete_audio_job(
    job_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Delete an audio transcription job record by job_id.
    Also cleans up any residual chunk files in ./temp_audio.
    """
    stmt = select(AnalyzeJob).where(
        AnalyzeJob.id == job_id,
        AnalyzeJob.source_type == "audio",
    )
    result = await session.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Audio job not found")

    await session.delete(job)
    await session.commit()

    # Clean up any residual temp files for this job (e.g. chunk files left by audio_processor)

    for temp_file in Path(TEMP_DIR).glob(f"{job_id}*"):
        try:
            temp_file.unlink()
        except FileNotFoundError:
            pass


    return {"message": "Audio job deleted successfully", "job_id": str(job_id)}


@router.patch("/jobs/{job_id}")
async def update_audio_result(
    job_id: uuid.UUID,
    body: AudioResultUpdate,
    session: AsyncSession = Depends(get_async_session),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Update the transcription result text for an audio job.
    Only the owning user can update their own job.
    """
    stmt = select(AnalyzeJob).where(
        AnalyzeJob.id == job_id,
        AnalyzeJob.user_id == current_user.id,
        AnalyzeJob.source_type == "audio",
    )
    result = await session.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Audio job not found or access denied")

    # Merge the new text into the existing result dict
    updated_result = dict(job.result) if job.result else {}
    updated_result["summary"] = body.result_text
    job.result = updated_result

    await session.commit()

    return {"message": "Transcription updated successfully", "job_id": str(job_id)}


