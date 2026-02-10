
import os
import asyncio
import aiofiles
from pypdf import PdfReader
from typhoon_ocr import ocr_document

async def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extracts text from a PDF file (bytes) using Typhoon OCR.
    Handles multi-page PDFs by processing pages concurrently.
    """
    temp_filename = f"/tmp/temp_ocr_{os.urandom(8).hex()}.pdf"
    
    try:
        # 1. Save bytes to a temporary file
        async with aiofiles.open(temp_filename, "wb") as f:
            await f.write(file_bytes)
            
        # 2. Count pages
        reader = PdfReader(temp_filename)
        num_pages = len(reader.pages)
        
        # Limit to 50 pages as per requirement
        max_pages = min(num_pages, 50)
        
        # 3. Process pages concurrently with a Semaphore
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        results = [None] * max_pages

        async def process_page(page_idx):
            async with semaphore:
                # page_num in typhoon_ocr is 1-indexed (based on docs example page_num=2)
                # But let's verify if default is 1. Docs say "default is 1".
                # So loop from 1 to max_pages.
                page_number = page_idx + 1
                return await asyncio.to_thread(
                    ocr_document,
                    pdf_or_image_path=temp_filename,
                    page_num=page_number
                )

        tasks = [process_page(i) for i in range(max_pages)]
        pages_text = await asyncio.gather(*tasks)
        
        # 4. Concatenate results
        full_text = "\n\n".join(pages_text)
        return full_text

    except Exception as e:
        print(f"Error during OCR processing: {e}")
        raise e
        
    finally:
        # 5. Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
