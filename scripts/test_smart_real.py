
import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.ocr_service import extract_text_from_pdf
from src.services.analysis_service import analyze_sheet_content

async def main():
    file_path = "sample_data/test.pdf"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please place a real PDF there.")
        return

    print(f"🚀 Starting Real Smart Pipeline Test (Stateless - No DB Write)...")
    print(f"📄 File: {file_path}")
    
    try:
        # 1. OCR Step
        print("\n⏳ Step 1: Extracting text via Typhoon OCR (Multi-page)...")
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        
        ocr_text = await extract_text_from_pdf(file_bytes)
        print(f"✅ OCR Done! Extracted {len(ocr_text)} characters.")
        print("-" * 30)
        print(f"Sample Text: {ocr_text[:300]}...")
        print("-" * 30)

        # 2. AI Analysis Step
        print("\n⏳ Step 2: Analyzing content via Typhoon AI (v2.5)...")
        ai_data = await analyze_sheet_content(ocr_text)
        
        print("\n✨ --- REAL AI ANALYSIS RESULTS --- ✨")
        print(f"\n📝 [SUMMARY]\n{ai_data.get('summary')}")
        
        print("\n📊 [ASSESSMENT]")
        for point in ai_data.get('assessment', []):
            print(f"  • {point}")
            
        print("\n🏷️  [TAGS]")
        print(f"  {', '.join(ai_data.get('tags', []))}")
        print("\n" + "✨" * 20)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
