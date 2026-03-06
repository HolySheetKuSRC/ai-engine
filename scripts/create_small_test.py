from PIL import Image, ImageDraw, ImageFont
import os

img = Image.new('RGB', (800, 600), color = (255, 255, 255))
d = ImageDraw.Draw(img)

# Add some text
d.text((50, 50), "Hello World", fill=(0, 0, 0))
d.text((50, 100), "This is a synthetic test for OCR.", fill=(0, 0, 0))
d.text((50, 150), "Testing bounding box accuracy.", fill=(0, 0, 0))

# Save as PDF
os.makedirs("sample_data", exist_ok=True)
img.save('sample_data/small_test.pdf', "PDF" ,resolution=100.0)
print("Created small_test.pdf")
