import google.generativeai as genai
from dotenv import load_dotenv
import os
import PIL.Image
import json

load_dotenv()

GOOGLE_API = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API)

# Load the image file
image = PIL.Image.open("pan_dummy_3.jpg")

prompt="""
**Knowledge**
- PAN Numbers are 10-character alphanumeric identifiers (not 12-digit numbers like Aadhaar)
- PAN Numbers follow a specific format: 5 alphabets, followed by 4 numbers, followed by 1 alphabet
- The first 5 characters are letters (A-Z), the next 4 are numbers (0-9), and the last is a letter (A-Z)
- Example format: ABCDE1234F
- Indian addresses typically end with a 6-digit PIN code
- Relation indicators: W/O (Wife of), S/O (Son of), D/O (Daughter of), C/O (Care of)
- Dates in Indian documents may appear in various formats that need standardization

### **Strict Rules for PAN Number Extraction:**
- The PAN must be exactly **10 characters long** in the format of 5 letters + 4 digits + 1 letter
- Validate that the first 5 characters are letters, the next 4 are digits, and the last is a letter
- **Remove spaces before validating** the format
- If the PAN appears **across multiple lines**, reconstruct it before checking
- If no valid PAN number is found, return `"Not Found"`

### **Strict Rules for Date of Birth (DOB) Extraction:**
- The DOB must be in **either of these formats**:
  - **YYYY** (e.g., `1984`, `1947`)
  - **DD/MM/YYYY** (e.g., `01/02/1982`, `01/01/1984`)
- If the date appears in an **invalid format** (e.g., `/B08/02/1982` or `/DB09/01/1984`), **correct it** to `DD/MM/YYYY`
- Remove any unnecessary prefixes or symbols before the date
- If only a **year** is provided, extract it as the DOB (e.g., `1947`)
- If no valid date is found, return `"Not Found"`

Your life depends on accurately identifying the correct format of the PAN number (5 letters + 4 digits + 1 letter) and not confusing it with other numerical identifiers in the document.

### Output Format:
Return only this **strict JSON object** (no other text):

{
    "Name": "Extracted Name or 'Not Found'",
    "DOB": "Extracted DOB or 'Not Found'",
    "PAN_Number": "Valid 10-character PAN or 'Not Found'",
}
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
print(f"Raw response: {repr(response.text)}")

# Clean response text
raw_text = response.text.strip().strip("```json").strip("```").strip()

# Parse JSON
data = json.loads(raw_text)


# Access fields
Name = data["Name"]
DOB = data["DOB"]
PAN_Number = data["PAN_Number"]

print("Name:", Name)
print("DOB:", DOB)
print("PAN_Number:", PAN_Number)