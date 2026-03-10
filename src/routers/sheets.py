import os
from pathlib import Path
from src.services.ocr_service import extract_text_from_pdf, calculate_file_hash, is_junk_content
from src.services.analysis_service import analyze_sheet_content
from src.services.webhook_service import send_webhook
from src.services.download_service import download_pdf_from_url
from src.database import get_async_session, AnalyzeJob, async_session_maker
from src.models.ai_dataset import AiDatasetRecord
from src.schemas.ai_response import AIAnalysisResult
import logging
import os
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form, status
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from src.tasks import process_sheet_task

router = APIRouter(prefix="/sheets", tags=["Sheets"])

PROCESSED_CACHE = {}
TEMP_DIR = "./temp_sheets"
os.makedirs(TEMP_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@router.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
async def analyze_sheet(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    webhook_url: str | None = Form(None),
    sheet_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Async AI Analysis Pipeline:
    1. Upload File (PDF) OR Provide URL
    2. Create Job (Pending)
    3. Return Job ID (202 Accepted)
    4. Background: OCR -> AI -> DB Update -> Webhook
    """
    # Normalize inputs
    normalized_url = (file_url or "").strip()
    is_url_provided = normalized_url.startswith(("http://", "https://"))
    
    # Sanitize Swagger default "string" value
    if webhook_url == "string":
        webhook_url = None

    is_file_provided = False
    file_content = b""
    if file and file.filename:
        file_content = await file.read()
        if len(file_content) > 0:
            is_file_provided = True

    try:
        content = b""
        if is_url_provided:
            # Download from URL
            content = await download_pdf_from_url(normalized_url)
        elif is_file_provided:
            # Use uploaded file
            if file.content_type != "application/pdf":
                # Basic check, background task does deeper OCR/PDF check
                raise HTTPException(status_code=400, detail="File must be a PDF")
            content = file_content
        else:
            raise HTTPException(
                status_code=400, 
                detail="Please provide either a PDF file or a valid file_url."
            )
            
        if len(content) == 0:
             raise HTTPException(status_code=400, detail="Empty content or download failed")

        # Create Job
        new_job = AnalyzeJob(
            webhook_url=webhook_url, 
            sheet_id=sheet_id,
            status="pending"
        )
        db.add(new_job)
        await db.commit()
        await db.refresh(new_job)

        # Save to temp file
        temp_path = os.path.join(TEMP_DIR, f"{new_job.id}.pdf")
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Start Celery Task
        process_sheet_task.delay(str(new_job.id), temp_path, webhook_url)
        
        return {
            "job_id": new_job.id,
            "sheet_id": new_job.sheet_id,
            "status": "pending",
            "message": "Job accepted. Processing in background."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: UUID, db: AsyncSession = Depends(get_async_session)):
    stmt = select(AnalyzeJob).where(AnalyzeJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # Backward Compatibility: Flatten ocr_content if it's a list
    formatted_result = job.result
    if formatted_result and isinstance(formatted_result.get("ocr_content"), list):
        formatted_result = dict(formatted_result)  # Copy
        formatted_result["ocr_content"] = "\n\n".join(
            [block.get("text", "") for block in formatted_result["ocr_content"] if isinstance(block, dict)]
        )

    return {
        "job_id": job.id,
        "status": job.status,
        "result": formatted_result,
        "error_message": job.error_message,
        "created_at": job.created_at
    }

@router.get("/jobs/by-sheet/{sheet_id}")
async def get_job_by_sheet(sheet_id: str, db: AsyncSession = Depends(get_async_session)):
    """
    Poll job status by sheet_id.
    Returns the latest job associated with this sheet_id.
    """
    stmt = (
        select(AnalyzeJob)
        .where(AnalyzeJob.sheet_id == sheet_id)
        .order_by(AnalyzeJob.created_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail=f"No job found for sheet_id: {sheet_id}")
        
    # Backward Compatibility: Flatten ocr_content if it's a list
    formatted_result = job.result
    if formatted_result and isinstance(formatted_result.get("ocr_content"), list):
        formatted_result = dict(formatted_result)  # Copy
        formatted_result["ocr_content"] = "\n\n".join(
            [block.get("text", "") for block in formatted_result["ocr_content"] if isinstance(block, dict)]
        )
        
    return {
        "job_id": job.id,
        "sheet_id": job.sheet_id,
        "status": job.status,
        "result": formatted_result,
        "error_message": job.error_message,
        "created_at": job.created_at
    }


@router.delete("/jobs/{job_id}", status_code=status.HTTP_200_OK)
async def delete_sheet_job(job_id: UUID, db: AsyncSession = Depends(get_async_session)):
    """
    Delete a sheet OCR job record by job_id.
    Also cleans up any residual files in ./temp_sheets if present.
    """
    stmt = select(AnalyzeJob).where(AnalyzeJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    await db.delete(job)
    await db.commit()

    # Clean up any residual temp files (gracefully ignore if already removed)
    temp_dir = Path("./temp_sheets")
    if temp_dir.is_dir():
        for temp_file in temp_dir.glob(f"{job_id}*"):
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass

    return {"message": "Sheet job deleted successfully", "job_id": job_id}
