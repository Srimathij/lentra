import google.generativeai as genai
from dotenv import load_dotenv
import os
import PIL.Image

load_dotenv()

GOOGLE_API = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API)
 
# Load the image file
image = PIL.Image.open("udayam_2.webp")

prompt="""
Analyze the following text and extract the following details relevant to a Udyam Registration Certificate:

- Enterprise Name (typically the registered business name).
- Udyam Registration Number (exactly 16 characters, alphanumeric, usually starts with 'UDYAM-' and includes hyphens).
- Type of Enterprise (e.g., Micro, Small, Medium).
- Owner Name (extract the individual’s name associated with the enterprise).
- Official Address block including State, District, and a 6-digit PIN code.

### **Strict Rules for Udyam Registration Number:**
- Must be exactly **16 characters long** (including hyphens).
- Usually in the format: `UDYAM-XX-00-0000000` (alphanumeric).
- Ignore any sequences that are **not** exactly 16 characters or that **don’t** start with `UDYAM-`.
- If the number is split across multiple lines, reconstruct it.
- If no valid number is found, return `"Not Found"`.

### **Rules for Type of Enterprise Extraction:**
- Extract terms such as **Micro**, **Small**, or **Medium** based on their mention in the document.
- Ignore any unrelated business terms or categories.

### **Rules for Address Extraction:**
- Extract the **complete address block** containing street/locality, district, state, and 6-digit PIN code.
- If parts of the address appear across multiple lines, reconstruct into a **single paragraph**.
- Always include a **6-digit PIN code**.
- Ignore extra unrelated info like email, phone number, or industry classification.

### **Rules for Owner Name Extraction:**
- Find and extract the name associated with labels like **"Name of Entrepreneur"**, **"Proprietor"**, or similar.
- Only extract the actual name — remove any prefixes like “Mr.”, “Ms.”, or label words.

### **Expected Output Format:**
Enterprise Name: [Extracted Enterprise Name]  
Udyam Registration Number: [Extracted Number] (or "Not Found")  
Type of Enterprise: [Micro/Small/Medium] (or "Not Found")  
Owner Name: [Extracted Name] (or "Not Found")  
Address: [Complete Address in a single paragraph.]

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