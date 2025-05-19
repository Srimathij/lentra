import google.generativeai as genai
from dotenv import load_dotenv
import os
import PIL.Image

load_dotenv()

GOOGLE_API = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API)
 
# Load the image file
image = PIL.Image.open("voter.jpga")

prompt="""
Extract the text content and You are a document classification assistant. You will receive OCR-extracted text from an image of a government-issued document.
Your task is to classify the type of card shown in the image by analyzing the text content.

The possible classifications are:
- Aadhaar Card
- PAN Card
- Udyam Certificate
- Unknown (if it doesn't match any of the above)

🔍 Classification Rules:
- Aadhaar: Look for patterns like 12-digit numbers, mentions of "Unique Identification Authority of India", "Aadhaar", "VID", or QR codes.
- PAN Card: Look for "Income Tax Department", "Permanent Account Number", 10-character alphanumeric PAN format (ABCDE1234F).
- Udyam Certificate: Look for “Udyam Registration”, “Ministry of MSME”, or registration numbers starting with "UDYAM-".
- Unknown: Use this if the text doesn't match any known patterns.

Return only the classification as one of the four options.
[Aadhaar Card / PAN Card / Udyam Certificate / Unknown]
"""
# Generate content using the image
model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content(
    contents=[image, prompt],
    generation_config=genai.types.GenerationConfig(
        temperature=0
    )
)

print(response.text)