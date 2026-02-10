from src.services.ocr_service import extract_text_from_pdf
from src.services.analysis_service import analyze_sheet_content
from src.schemas.ai_response import AIAnalysisResult
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException
from uuid import UUID

router = APIRouter(prefix="/sheets", tags=["Sheets"])

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

        # 2. OCR Extraction
        ocr_text = await extract_text_from_pdf(content)
        if not ocr_text:
             raise HTTPException(status_code=400, detail="Could not extract text from PDF")
             
        logger.info("OCR Extraction completed.")
        
        # 3. AI Analysis
        ai_data = await analyze_sheet_content(ocr_text)
        logger.info("AI Analysis completed.")
        
        # 4. Return Result (Stateless)
        return AIAnalysisResult(
            filename=file.filename,
            ocr_content=ocr_text,
            summary=ai_data.get("summary", "No summary available"),
            assessment=ai_data.get("assessment", []),
            tags=ai_data.get("tags", []),
            suggested_price=0.0, # Placeholder logic
            page_count=None # OCR service currently returns string, page count logic can be added later
        )

    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))
