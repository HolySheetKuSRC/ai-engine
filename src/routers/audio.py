import os
import shutil
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.database import get_async_session
from src.models.ai_dataset import AiDatasetRecord
from src.services.asr_service import transcribe_audio
from src.services.summary_service import summarize_lecture

router = APIRouter(
    prefix="/api/audio",
    tags=["Audio"]
)

TEMP_DIR = "./temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/transcribe")
async def transcribe_and_summarize(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Upload an audio file, transcribe it, summarize it, and store the result.
    """
    # Validate file extension
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in [".wav", ".mp3", ".m4a"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Allowed: .wav, .mp3, .m4a")

    # Generate a unique filename to prevent collisions
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(TEMP_DIR, unique_filename)

    try:
        # Save the uploaded file temporarily
        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):  # Read in chunks
                await out_file.write(content)

        # Transcribe audio
        raw_text = await transcribe_audio(file_path)

        # Summarize transcript
        summary_text = await summarize_lecture(raw_text)

        # Store in database
        record = AiDatasetRecord(
            filename=file.filename,
            source_type='audio',
            raw_text=raw_text,
            summary_text=summary_text
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)

        return {
            "id": record.id,
            "filename": record.filename,
            "raw_text": record.raw_text,
            "summary_text": record.summary_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
