
import os
import asyncio
import base64
import hashlib
import re
from typing import Tuple, List, Dict, Any
from openai import AsyncOpenAI
import numpy as np
import cv2
from pdf2image import convert_from_bytes
from src.services.image_processor import preprocess_image, detect_text_blocks

def calculate_file_hash(file_bytes: bytes) -> str:
    """Calculates MD5 hash of file bytes."""
    return hashlib.md5(file_bytes).hexdigest()

def is_junk_content(text: str) -> bool:
    """
    Returns True if content is considered junk.
    Logic: Length < 10 chars. 
    (Trusting LLM to handle technical/symbol-heavy documents)
    """
    return len(text) < 10

async def _ocr_base64_image(client: AsyncOpenAI, base64_image: str) -> str:
    """Calls Typhoon OCR using the OpenAI-compatible endpoint with a base64 payload."""
    prompt = (
        "Extract all text from the image.\n\n"
        "Instructions:\n"
        "- Only return the clean Markdown.\n"
        "- Do not include any explanation or extra text.\n"
        "- You must include all information on the page.\n\n"
        "Formatting Rules:\n"
        "- Tables: Render tables using <table>...</table> in clean HTML format.\n"
        "- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$)."
    )
    
    response = await client.chat.completions.create(
        model="typhoon-ocr",
        messages=[
           {
             "role": "user",
             "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
             ],
           }
        ],
        max_tokens=16384,
        extra_body={
            "repetition_penalty": 1.1,
            "temperature": 0.1,
            "top_p": 0.6,
        }
    )
    return response.choices[0].message.content

async def extract_text_from_pdf(file_bytes: bytes) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extracts structured layout coordinates and text from a PDF file using a Two-Pass Hybrid Pipeline.
    Returns:
       Tuple containing:
         - List of text blocks: [{"text": "...", "bbox": {"x":..., "y":..., "w":..., "h":...}}]
         - Total number of pages processed
    """
    # Initialize client for async Typhoon API calls
    api_key = os.getenv("TYPHOON_API_KEY") or os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(
        base_url=os.getenv("TYPHOON_BASE_URL", 'https://api.opentyphoon.ai/v1'),
        api_key=api_key,
        timeout=45.0,
        max_retries=0  # avoid silent double-waits; failures surface immediately
    )
    
    # 1. Convert PDF bytes to PIL Images
    # Convert up to max 50 pages.
    try:
        pages_pil = convert_from_bytes(file_bytes, fmt="jpeg", last_page=50)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        raise ValueError(f"Failed to read PDF: {e}")
        
    num_pages = len(pages_pil)
    semaphore = asyncio.Semaphore(1)  # Strictly 1 concurrent Typhoon API call to prevent 429 Errors
    all_blocks = []
    
    async def process_crop(client, base64_img, bbox):
        async with semaphore:
            try:
                # Add slight delay to respect rate limits
                await asyncio.sleep(1)
                text = await _ocr_base64_image(client, base64_img)
                return {"text": text, "bbox": bbox}
            except Exception as e:
                print(f"Error during OCR of crop: {e}")
                return {"text": "", "bbox": bbox}

    for page_num, img in enumerate(pages_pil):
        # Convert PIL to cv2 format (BGR)
        open_cv_image = np.array(img)
        
        # Convert RGB to BGR for cv2
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        # 2. Add Preprocessing 
        color_deskewed, binary_mask = preprocess_image(open_cv_image)
        
        # 3. Pass 1: Layout Analysis (Paragraph Bounding Boxes)
        bboxes = detect_text_blocks(binary_mask)
        
        # 4. Pass 2: Base64 Encoding and Text Extraction
        crop_tasks = []
        for bbox in bboxes:
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # Crop using numpy slicing
            cropped_img = color_deskewed[max(0, y-5):min(color_deskewed.shape[0], y+h+5), 
                                         max(0, x-5):min(color_deskewed.shape[1], x+w+5)]
                                         
            # Encode frame directly from memory to Avoid Disk I/O
            success, encoded_image = cv2.imencode('.jpg', cropped_img)
            if not success:
                continue
            
            base64_crop = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
            
            # Schedule the task
            crop_tasks.append(process_crop(client, base64_crop, bbox))
            
        # Run OCR calls for this page concurrently
        page_results = await asyncio.gather(*crop_tasks)
        
        # Filter out empty text blocks
        page_results = [block for block in page_results if block["text"].strip()]
        
        all_blocks.extend(page_results)

    return all_blocks, num_pages
