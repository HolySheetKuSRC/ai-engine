import os
import asyncio
import hashlib
import json
import base64
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
from pypdf import PdfReader
import aiofiles
from pdf2image import convert_from_bytes
from openai import AsyncOpenAI

from src.services.image_processor import deskew_image, preprocess_for_layout_analysis, detect_text_blocks
from typhoon_ocr.ocr_utils import get_prompt

# Async OpenAI client for Typhoon OCR
typhoon_client = AsyncOpenAI(
    base_url=os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1"),
    api_key=os.getenv("TYPHOON_OCR_API_KEY") or os.getenv("TYPHOON_API_KEY") or os.getenv("OPENAI_API_KEY")
)

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

async def call_typhoon_vision_base64(base64_img: str, prompt: str, model: str = "typhoon-ocr") -> str:
    """Sends a single base64 image block directly to Typhoon Vision API."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}},
            ],
        }
    ]
    
    for attempt in range(3):
        try:
            response = await typhoon_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=16384,
                extra_body={
                    "repetition_penalty": 1.1, # v1.5 standard
                    "temperature": 0.1,
                    "top_p": 0.6,
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                await asyncio.sleep(2 ** attempt) # Exponential backoff: 1s, 2s
            else:
                raise e

async def extract_text_from_pdf(file_bytes: bytes) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extracts text from a PDF file (bytes) using a Two-Pass Pipeline:
    1. OpenCV Layout Analysis (Bounding Boxes).
    2. Typhoon OCR on cropped image blocks.
    
    Returns:
        A tuple of (List of layout blocks, total page count)
        Layout blocks format: [{"text": str, "bbox": {"x": int, "y": int, "w": int, "h": int}, "page_number": int}, ...]
    """
    # 1. Convert PDF to images
    images = await asyncio.to_thread(convert_from_bytes, file_bytes)
    
    num_pages = len(images)
    max_pages = min(num_pages, 50)
    
    ocr_results = []
    
    prompt_fn = get_prompt("v1.5")
    prompt_text = prompt_fn(figure_language="Thai")
    
    semaphore = asyncio.Semaphore(5)  # Global limit of 5 concurrent requests
    
    for page_idx in range(max_pages):
        pil_image = images[page_idx]
        
        # Convert PIL to cv2 BGR format
        open_cv_image = np.array(pil_image)
        if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
            open_cv_image = open_cv_image[:, :, ::-1].copy() # RGB to BGR
            
        # 1. Image Preprocessing (Deskewing)
        aligned_img = deskew_image(open_cv_image)
        
        # 2. Pass 1 - Layout Analysis
        binary_mask = preprocess_for_layout_analysis(aligned_img)
        blocks = detect_text_blocks(binary_mask)
        
        # If no blocks detected, fallback to full page OCR
        if not blocks:
            height, width = aligned_img.shape[:2]
            blocks = [{"x": 0, "y": 0, "w": width, "h": height}]
        
        # 3. Pass 2 - Text Recognition (Async)
        async def process_block(block: Dict[str, int]) -> Dict[str, Any]:
            x, y, w, h = block["x"], block["y"], block["w"], block["h"]
            
            cropped_img = aligned_img[y:y+h, x:x+w]
            
            success, encoded_image = cv2.imencode('.png', cropped_img)
            if not success:
                return {"text": "", "bbox": block, "page_number": page_idx + 1}
                
            base64_str = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
            
            async with semaphore:
                try:
                    text = await call_typhoon_vision_base64(base64_str, prompt_text)
                    return {"text": text.strip(), "bbox": block, "page_number": page_idx + 1}
                except Exception as e:
                    print(f"Error OCR'ing block on page {page_idx + 1}: {e}")
                    return {"text": "", "bbox": block, "page_number": page_idx + 1}

        # Run OCR concurrently for all blocks on this page
        # Note: semaphore intrinsically limits to 5 at a time
        tasks = [process_block(block) for block in blocks]
        page_results = await asyncio.gather(*tasks)
        
        valid_results = [res for res in page_results if res["text"]]
        ocr_results.extend(valid_results)

    return ocr_results, num_pages
