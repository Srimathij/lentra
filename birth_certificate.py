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
Analyze the following text and extract the following details relevant to an Indian Birth Certificate:
- Full Name of the Child  
- Date of Birth (DOB) in a valid format  
- Gender  
- Father's Name  
- Mother's Name  
- Place of Birth (typically includes hospital/house name, locality, district, and state)  
- Registration Number (alphanumeric code associated with the certificate)  

### **Rules for Date of Birth (DOB) Extraction:**  
- Acceptable formats:  
  - **DD/MM/YYYY** (e.g., 01/01/2001)  
  - **DD-MM-YYYY**  
  - **DD Month YYYY** (e.g., 01 January 2001)  
- If the date appears with extra symbols or typos (like “/D0B: 12/03/1994”), correct and extract it.  
- If no valid date is found, return `"DOB: Not Found"`.

### **Rules for Gender Extraction:**  
- Extract gender as **Male**, **Female**, or **Other**.  
- Ignore variants or extra words like “Sex:”, “Gender:”, etc.  
- If not found, return `"Gender: Not Found"`.

### **Rules for Parent Names Extraction:**  
- Look for **Father's Name** and **Mother's Name** using common labels:  
  - Father’s Name: “Father”, “Father's Name”, “S/O”, etc.  
  - Mother’s Name: “Mother”, “Mother's Name”, “M/O”, etc.  
- Only extract the full names, without labels or prefixes like “Mr.” or “Mrs.”  
- If a name is missing, return `"Not Found"` for that field.

### **Rules for Place of Birth Extraction:**  
- Extract a complete location including **hospital/home name**, **area/locality**, **district**, and **state**.  
- If the place is split across lines, reconstruct it into a **single paragraph**.  
- Must not include unrelated data like phone numbers or certificate authority info.

### **Rules for Registration Number Extraction:**  
- Look for any **alphanumeric code labeled as Registration Number**, “Reg. No.”, etc.  
- It is typically a mix of numbers and/or letters and uniquely identifies the birth record.  
- If not found, return `"Registration Number: Not Found"`.

### **Expected Output Format:**
Child Name: [Extracted Name]  
DOB: [Extracted DOB]  
Gender: [Male/Female/Other]  
Father’s Name: [Extracted Name]  
Mother’s Name: [Extracted Name]  
Place of Birth: [Extracted Address or Location]  
Registration Number: [Extracted Reg. No.] (or "Not Found")
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