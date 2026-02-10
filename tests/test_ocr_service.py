
import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.services.ocr_service import extract_text_from_pdf

class TestOCRService(unittest.TestCase):
    def test_extract_text_logic(self):
        asyncio.run(self._test_async_logic())

    async def _test_async_logic(self):
        # Mock dependencies
        with patch("src.services.ocr_service.aiofiles.open") as mock_aio_open, \
             patch("src.services.ocr_service.PdfReader") as mock_pdf_reader, \
             patch("src.services.ocr_service.ocr_document") as mock_ocr_document, \
             patch("os.path.exists") as mock_exists, \
             patch("os.remove") as mock_remove:
            
            # Setup mocks
            mock_file_handle = AsyncMock()
            mock_aio_open.return_value.__aenter__.return_value = mock_file_handle
            
            # Mock PDF with 3 pages
            mock_reader_instance = MagicMock()
            mock_reader_instance.pages = [1, 2, 3] # length 3
            mock_pdf_reader.return_value = mock_reader_instance
            
            # Mock OCR results
            mock_ocr_document.side_effect = lambda pdf_or_image_path, page_num: f"Page {page_num} Content"
            
            mock_exists.return_value = True

            # Execute
            file_bytes = b"fake_pdf_content"
            result = await extract_text_from_pdf(file_bytes)
            
            # Verify
            print(f"Result Preview: {result[:50]}...")
            
            # Check temp file write
            mock_aio_open.assert_called()
            mock_file_handle.write.assert_called_with(file_bytes)
            
            # Check page count
            self.assertEqual(len(mock_reader_instance.pages), 3)
            
            # Check OCR calls (should be 3 times)
            self.assertEqual(mock_ocr_document.call_count, 3)
            
            # Check correct page numbers (1-based)
            calls = mock_ocr_document.call_args_list
            page_nums = sorted([call.kwargs['page_num'] for call in calls])
            self.assertEqual(page_nums, [1, 2, 3])
            
            # Check cleanup
            mock_remove.assert_called()
            
            # Check result content
            expected_text = "Page 1 Content\n\nPage 2 Content\n\nPage 3 Content"
            self.assertEqual(result, expected_text)
            print("Verification Successful: logical flow covers all requirements.")

if __name__ == "__main__":
    unittest.main()
