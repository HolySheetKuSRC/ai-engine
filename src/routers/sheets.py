from src.services.ocr_service import extract_text_from_pdf, calculate_file_hash, is_junk_content
from src.services.analysis_service import analyze_sheet_content
from src.services.webhook_service import send_webhook
from src.services.download_service import download_pdf_from_url
from src.database import get_async_session, AnalyzeJob, async_session_maker
from src.schemas.ai_response import AIAnalysisResult
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Depends, Form, status
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

router = APIRouter(prefix="/sheets", tags=["Sheets"])

PROCESSED_CACHE = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Background Task Logic
async def process_analysis_task(job_id: UUID, file_bytes: bytes, webhook_url: str | None):
    async with async_session_maker() as session:
        try:
            # Update status to processing
            stmt = select(AnalyzeJob).where(AnalyzeJob.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                job.status = "processing"
                await session.commit()
            
            # 1. OCR
            # Security: Check file size/bytes/hash checks could be done here or in endpoint.
            # (Doing it here to keep endpoint fast, but repeated hash calc is okay)
            
            ocr_text, page_count = await extract_text_from_pdf(file_bytes)
            
            # Security: Junk Filter
            if is_junk_content(ocr_text):
                raise ValueError("Invalid or unreadable content (Junk Filter)")

            # 2. AI Analysis (Chunking handled in service)
            ai_data = await analyze_sheet_content(ocr_text)
            
            summary = ai_data.get("summary", "No summary available")
            file_hash = calculate_file_hash(file_bytes)
            
            # Security: Watermarking
            summary += f"\n\n(Verified by AI - Ref: {file_hash[:8]})"
            
            # Construct Result
            final_result = {
                "filename": "async_job", # We might want to pass filename too, but for now simple
                "ocr_content": ocr_text,
                "summary": summary,
                "assessment": ai_data.get("assessment", []),
                "tags": ai_data.get("tags", []),
                "page_count": page_count
            }
            
            # 3. Update DB (Success)
            # Re-fetch job to avoid detached instance issues if session closed/re-opened? 
            # We are in same session.
            job.status = "completed"
            job.result = final_result
            await session.commit()
            
            # 4. Webhook
            if webhook_url:
                await send_webhook(webhook_url, final_result)
                
        except Exception as e:
            logger.error(f"Background task failed for job {job_id}: {e}")
            # Update DB (Failure)
            # Need to re-fetch if transaction rolled back?
            # We'll try to update in a new transaction block if needed, but here simple:
            try:
                stmt = select(AnalyzeJob).where(AnalyzeJob.id == job_id)
                result = await session.execute(stmt)
                job = result.scalar_one_or_none()
                if job:
                    job.status = "failed"
                    job.error_message = str(e)
                    await session.commit()
            except Exception as db_e:
                logger.error(f"Failed to update job status to failed: {db_e}")

@router.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
async def analyze_sheet(
    background_tasks: BackgroundTasks,
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
        
        # Start Background Task
        background_tasks.add_task(process_analysis_task, new_job.id, content, webhook_url)
        
        return {
            "job_id": new_job.id,
            "sheet_id": new_job.sheet_id,
            "status": "pending",
            "message": "Job accepted. Processing in background."
        }

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
        
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result,
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
        
    return {
        "job_id": job.id,
        "sheet_id": job.sheet_id,
        "status": job.status,
        "result": job.result,
        "error_message": job.error_message,
        "created_at": job.created_at
    }
