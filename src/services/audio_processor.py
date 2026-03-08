
import os
import asyncio
import uuid
import math
from pydub import AudioSegment
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update
from src.database import async_session_maker, AnalyzeJob
from src.models.ai_dataset import AiDatasetRecord
from src.services.asr_service import transcribe_audio
from src.services.summary_service import summarize_lecture

# Constants
CHUNK_LENGTH_MS = 10 * 60 * 1000  # 10 minutes in milliseconds
TEMP_DIR = "./temp_audio"

async def process_audio_job(job_id: uuid.UUID, file_path: str):
    """
    Background task to process audio:
    1. Update job status to processing.
    2. Load audio (pydub) in a separate thread.
    3. Chunk into 10-min segments.
    4. Transcribe each chunk.
    5. Combine text.
    6. Summarize.
    7. Save result to DB (AiDatasetRecord + Job Result).
    8. Cleanup files.
    """
    optimized_path = None
    async with async_session_maker() as session:
        try:
            # 1. Update Job Status to Processing
            await _update_job_status(session, job_id, "processing")
            
            # 1.5. Convert to optimized MP3
            optimized_path = f"/tmp/{job_id.hex}_optimized.mp3"
            process = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-i", file_path, 
                "-ac", "1", "-b:a", "64k", optimized_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise Exception(f"FFmpeg Error: {stderr.decode()}")
            
            # 2. Load Audio
            # Use asyncio.to_thread to run blocking I/O in a separate thread
            # preventing the event loop from freezing.
            audio = await asyncio.to_thread(AudioSegment.from_file, optimized_path)
            
            # 3. Chunking
            duration_ms = len(audio)
            chunks_count = math.ceil(duration_ms / CHUNK_LENGTH_MS)
            
            full_transcript = []
            
            for i in range(chunks_count):
                start_ms = i * CHUNK_LENGTH_MS
                end_ms = min((i + 1) * CHUNK_LENGTH_MS, duration_ms)
                chunk = audio[start_ms:end_ms]
                
                # Export chunk to temp file
                chunk_filename = f"{job_id}_chunk_{i}.mp3"
                chunk_path = os.path.join(TEMP_DIR, chunk_filename)
                
                # Exporting is blocking IO, offload to thread
                await asyncio.to_thread(chunk.export, chunk_path, format="mp3")
                
                try:
                    # 4. Transcribe Chunk
                    chunk_text = await transcribe_audio(chunk_path)
                    full_transcript.append(chunk_text)
                finally:
                    # Cleanup chunk immediately using os.remove (blocking but fast enough, 
                    # or could use to_thread but typically fine for unlink)
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)

            raw_text = " ".join(full_transcript)
            
            # 6. Summarize
            summary_text = await summarize_lecture(raw_text)
            
            # 7. Save Result
            # Save permanently to AiDatasetRecord
            dataset_record = AiDatasetRecord(
                filename=os.path.basename(file_path),
                source_type='audio',
                raw_text=raw_text,
                summary_text=summary_text
            )
            session.add(dataset_record)
            
            # Ensure ID is generated
            await session.flush()
            
            # Update Job with processing results
            stmt = (
                update(AnalyzeJob)
                .where(AnalyzeJob.id == job_id)
                .values(
                    status="completed",
                    result={
                      "dataset_id": dataset_record.id,
                      "summary": summary_text,
                      "raw_text_snippet": raw_text[:200] + "..." if len(raw_text) > 200 else raw_text,
                      "full_text_saved_in_dataset": True
                    } 
                )
            )
            await session.execute(stmt)
            await session.commit()
            
        except Exception as e:
            await session.rollback()
            await _update_job_status(session, job_id, "failed", error_message=str(e))
            print(f"Job {job_id} failed: {e}")
            
        finally:
            # 8. Cleanup Original File and Optimized File
            if os.path.exists(file_path):
                os.remove(file_path)
            if optimized_path and os.path.exists(optimized_path):
                os.remove(optimized_path)

async def _update_job_status(session: AsyncSession, job_id: uuid.UUID, status: str, error_message: str = None):
    stmt = (
        update(AnalyzeJob)
        .where(AnalyzeJob.id == job_id)
        .values(status=status, error_message=error_message)
    )
    await session.execute(stmt)
    await session.commit()
