
import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.ocr_service import extract_text_from_pdf

async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_ocr_manual.py <path_to_pdf>")
        # Default to sample if exists
        default_path = "sample_data/test.pdf"
        if os.path.exists(default_path):
            print(f"No file specified, using default: {default_path}")
            file_path = default_path
        else:
            return
    else:
        file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Processing {file_path}...")
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        import json
        ocr_blocks, page_count = await extract_text_from_pdf(file_bytes)
        print("\n--- Extracted Layout Blocks ---\n")
        print(json.dumps(ocr_blocks, indent=2, ensure_ascii=False))
        print(f"\nTotal Pages: {page_count}\n----------------------\n")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
