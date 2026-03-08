import os
import logging
from uuid import UUID
from sqlalchemy import select
from src.services.ocr_service import extract_text_from_pdf, calculate_file_hash, is_junk_content
from src.services.analysis_service import analyze_sheet_content
from src.services.webhook_service import send_webhook
from src.database import AnalyzeJob, async_session_maker
from src.models.ai_dataset import AiDatasetRecord

logger = logging.getLogger(__name__)

async def process_analysis_task(job_id: UUID, file_path: str, webhook_url: str | None):
    async with async_session_maker() as session:
        try:
            # Update status to processing
            stmt = select(AnalyzeJob).where(AnalyzeJob.id == job_id)
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job:
                job.status = "processing"
                await session.commit()
            
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            
            # 1. OCR
            # Security: Check file size/bytes/hash checks could be done here or in endpoint.
            # (Doing it here to keep endpoint fast, but repeated hash calc is okay)
            
            ocr_blocks, page_count = await extract_text_from_pdf(file_bytes)
            
            # Concatenate all text for analysis and filtering
            full_text = "\n\n".join([block["text"] for block in ocr_blocks])
            
            # Security: Junk Filter
            if is_junk_content(full_text):
                raise ValueError("Invalid or unreadable content (Junk Filter)")

            # 2. AI Analysis (Chunking handled in service)
            ai_data = await analyze_sheet_content(full_text)
            
            summary = ai_data.get("summary", "No summary available")
            file_hash = calculate_file_hash(file_bytes)
            
            # Security: Watermarking
            summary += f"\n\n(Verified by AI - Ref: {file_hash[:8]})"

            # 3a. Canonical identifier — must be computed BEFORE final_result so it's
            #     used consistently in both the job result JSON and ai_dataset_records.
            record_filename: str = job.sheet_id if job.sheet_id else str(job_id)
            tags_str: str = ", ".join(ai_data.get("tags", []))

            # Construct Result
            final_result = {
                "filename": record_filename, 
                "ocr_content": ocr_blocks,
                "summary": summary,
                "assessment": ai_data.get("assessment", []),
                "tags": ai_data.get("tags", []),
                "page_count": page_count
            }
            
            # 3. Update DB (Success)
            job.status = "completed"
            job.result = final_result
            await session.commit()

            existing_stmt = select(AiDatasetRecord).where(
                AiDatasetRecord.filename == record_filename
            )
            existing_result = await session.execute(existing_stmt)
            existing_record = existing_result.scalar_one_or_none()

            if existing_record:
                # Update in place so chat tutor mode also gets fresh content
                existing_record.raw_text = full_text
                existing_record.summary_text = summary
                existing_record.tags = tags_str
                existing_record.source_type = "sheet"
            else:
                dataset_record = AiDatasetRecord(
                    filename=record_filename,
                    source_type="sheet",
                    raw_text=full_text,
                    summary_text=summary,
                    tags=tags_str,
                )
                session.add(dataset_record)

            await session.commit()
            logger.info(f"Synced '{record_filename}' to ai_dataset_records (tags: {tags_str[:80]})")

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
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
