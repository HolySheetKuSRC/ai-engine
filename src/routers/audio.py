import os
import shutil
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydub import AudioSegment
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
    Supports various audio formats by converting to mp3 if necessary.
    """
    # 1. Save the uploaded file temporarily
    file_extension = os.path.splitext(file.filename)[1].lower()
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    original_file_path = os.path.join(TEMP_DIR, unique_filename)
    converted_file_path = None

    try:
        async with aiofiles.open(original_file_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):  # Read in chunks
                await out_file.write(content)

        # 2. Check if conversion is needed
        # Typhoon ASR likely supports wav, mp3, mpeg, mpga, m4a, ogg, wav, webm. 
        # But if the user specifically asked to convert if not in specific list, we follow that.
        # User said: "if it's not wav, mp3, flac, ogg, opus" -> convert.
        # However, pydub depends on ffmpeg.
        target_file_path = original_file_path
        supported_formats = [".wav", ".mp3", ".flac", ".ogg", ".opus"]
        
        if file_extension not in supported_formats:
            # Convert to mp3
            converted_filename = f"{uuid.uuid4()}_converted.mp3"
            converted_file_path = os.path.join(TEMP_DIR, converted_filename)
            
            # Load and export using pydub
            # Note: This operation blocks the event loop. For production with high load, 
            # consider running in a thread pool executor.
            audio = AudioSegment.from_file(original_file_path)
            audio.export(converted_file_path, format="mp3")
            
            target_file_path = converted_file_path

        # 3. Transcribe audio
        raw_text = await transcribe_audio(target_file_path)

        # 4. Summarize transcript
        summary_text = await summarize_lecture(raw_text)

        # 5. Store in database
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
        # Cleanup temporary files
        if os.path.exists(original_file_path):
            os.remove(original_file_path)
        if converted_file_path and os.path.exists(converted_file_path):
            os.remove(converted_file_path)
