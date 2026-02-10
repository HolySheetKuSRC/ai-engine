
from fastapi import APIRouter, UploadFile, HTTPException, File
from src.services.ocr_service import extract_text_from_pdf

router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extracts text from a PDF file using Typhoon OCR.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        file_bytes = await file.read()
        text = await extract_text_from_pdf(file_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
