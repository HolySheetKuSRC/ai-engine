import os
import uuid
import aiofiles
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
    if file_extension not in [".wav", ".mp3", ".m4a"]: # Allow m4a as we handle it in processor too (or should we?)
        # User requirement said "only .mp3/.wav" for this specific request, but previously we supported m4a. 
        # Let's keep supporting m4a if pydub can handle it, but for strict compliance with prompt:
        # "only .mp3/.wav" -> actually, let's allow m4a as the user previously asked for it and pydub handles it.
        # But if the user strictly wants mp3/wav now, I should probably stick to that?
        # "Change POST ... to accept an UploadFile (only .mp3/.wav)"
        # Okay, I will restrict to mp3/wav to follow the prompt exactly, but I'll add m4a comment.
        if file_extension not in [".wav", ".mp3"]:
             raise HTTPException(status_code=400, detail="Unsupported file format. Allowed: .wav, .mp3")

    # 2. Create Job
    job_id = uuid.uuid4()
    job = AnalyzeJob(id=job_id, status="pending", result={})
    session.add(job)
    await session.commit()
    
    # 3. Save file temporarily
    unique_filename = f"{job_id}{file_extension}"
    file_path = os.path.join(TEMP_DIR, unique_filename)
    
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):
                await out_file.write(content)
                
        # 4. Dispatch Background Task
        background_tasks.add_task(process_audio_job, job_id, file_path)
        
        return {
            "job_id": str(job_id),
            "status": "processing"
        }
        
    except Exception as e:
        # If saving fails, we should probably fail the job or just error out
        # Since we haven't dispatched the task yet, we can just error out
        # But we should probably cleanup the DB record or mark it failed?
        await session.delete(job)
        await session.commit()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))
