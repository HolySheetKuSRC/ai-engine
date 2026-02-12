from src.services.ocr_service import extract_text_from_pdf, calculate_file_hash, is_junk_content
from src.services.analysis_service import analyze_sheet_content
from src.schemas.ai_response import AIAnalysisResult
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from uuid import UUID

router = APIRouter(prefix="/sheets", tags=["Sheets"])

PROCESSED_CACHE = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=AIAnalysisResult)
async def analyze_sheet(file: UploadFile = File(...)):
    """
    Stateless AI Analysis Pipeline:
    1. Upload File (PDF)
    2. Extract Text (OCR)
    3. Analyze Content (AI)
    4. Return Analysis Result (No DB Save)
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # 1. Read File Logic
        content = await file.read()
        logger.info(f"Processing file: {file.filename} ({len(content)} bytes)")

        # Security: Check file size/bytes before hashing (implicit in read())
        if len(content) == 0:
             raise HTTPException(status_code=400, detail="Empty file")

        # Security: Processing Cache (DoS Protection)
        file_hash = calculate_file_hash(content)
        
        if file_hash in PROCESSED_CACHE:
            logger.info(f"Cache hit for hash: {file_hash}")
            return PROCESSED_CACHE[file_hash]

        # 2. OCR Extraction
        ocr_text, page_count = await extract_text_from_pdf(content)
        
        if not ocr_text:
             raise HTTPException(status_code=400, detail="Could not extract text from PDF")
             
        # Security: Junk Filter
        if is_junk_content(ocr_text):
            logger.warning(f"Junk content detected for file: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid or unreadable content")

        logger.info("OCR Extraction completed.")
        
        # 3. AI Analysis
        ai_data = await analyze_sheet_content(ocr_text)
        logger.info("AI Analysis completed.")
        
        summary = ai_data.get("summary", "No summary available")
        
        # Security: Watermarking
        summary += f"\n\n(Verified by AI - Ref: {file_hash[:8]})"

        # 4. Return Result (Stateless)
        result = AIAnalysisResult(
            filename=file.filename,
            ocr_content=ocr_text,
            summary=summary,
            assessment=ai_data.get("assessment", []),
            tags=ai_data.get("tags", []),
            page_count=page_count
        )
        
        # Update Cache
        if len(PROCESSED_CACHE) >= 100:
            PROCESSED_CACHE.clear()
            logger.info("Cache cleared (limit reached)")
            
        PROCESSED_CACHE[file_hash] = result
        
        return result

    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))
