
import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4
import sys
import os

# Set dummy API key for testing purposes (before importing service)
os.environ["TYPHOON_API_KEY"] = "sk-dummy-key"

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.routers.sheets import analyze_sheet
from src.services.analysis_service import analyze_sheet_content

class TestSmartUpload(unittest.TestCase):
    def test_analysis_service_mock(self):
        """Test the AI analysis logic with a mock response"""
        asyncio.run(self._test_analysis())

    async def _test_analysis(self):
        with patch("src.services.analysis_service.client") as mock_client:
            # Mock OpenAI response
            mock_response = MagicMock()
            mock_response.choices[0].message.content = '{"summary": "Test Summary", "assessment": ["Point 1", "Point 2"], "tags": ["#Tag1", "#Tag2"]}'
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await analyze_sheet_content("Sample OCR Text")
            
            self.assertEqual(result["summary"], "Test Summary")
            self.assertEqual(len(result["tags"]), 2)
            print("AI Analysis Service Test Passed!")

    def test_router_logic(self):
        """Test the analysis router flow (Stateless)"""
        asyncio.run(self._test_router())

    async def _test_router(self):
        with patch("src.routers.sheets.extract_text_from_pdf", new_callable=AsyncMock) as mock_ocr, \
             patch("src.routers.sheets.analyze_sheet_content", new_callable=AsyncMock) as mock_ai:
            
            # Mocks
            mock_ocr.return_value = "Mocked PDF Content"
            mock_ai.return_value = {
                "summary": "Mock Summary", 
                "assessment": ["Good"], 
                "tags": ["#Mock"]
            }
            
            mock_file = MagicMock()
            mock_file.filename = "test.pdf"
            mock_file.content_type = "application/pdf"
            mock_file.read = AsyncMock(return_value=b"PDF_BYTES")
            
            # Execute
            response = await analyze_sheet(file=mock_file)
            
            # Verify
            self.assertEqual(response.filename, "test.pdf")
            self.assertEqual(response.summary, "Mock Summary")
            self.assertEqual(response.ocr_content, "Mocked PDF Content")
            print("Router Flow Test Passed! (Stateless)")

if __name__ == "__main__":
    unittest.main()
