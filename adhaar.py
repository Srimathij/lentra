import google.generativeai as genai
from dotenv import load_dotenv
import os
import PIL.Image
import json

load_dotenv()
GOOGLE_API = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API)

# Define the function
def extract_adhaar_details_from_image(image_path: str) -> dict:
    prompt = """
        Analyze the following text and extract the following details:  
    - Name (typically the card holder's name).
    - A date formatted as a date of birth (DOB).  
    - A valid 12-digit number (may contain spaces or appear on multiple lines).  
    - Relation Name (extract the correct name mentioned with W/O, S/O, D/O, or C/O).
    - A text block that resembles an address, including the 6-digit PIN code.
 
    ### **Strict Rules for the 12-digit Aadhaar Number:**  
    - The number must be exactly **12 digits long** (e.g., "1234 5678 9012" or "123456789012").  
    - If the number appears **across multiple lines**, reconstruct it before checking the length.  
    - **Ignore numbers shorter than 12 digits** (like pincodes).  
    - **Ignore numbers longer than 12 digits** (like 16-digit VID numbers).  
    - **Remove spaces before validating the 12-digit length**.  
    - If no valid 12-digit number is found, return `"Not Found"`.
   
    ### **Strict Rules for Date of Birth (DOB) Extraction:**  
    - The DOB must be in **either of these formats**:  
      - **YYYY** (e.g., `1984`, `1947`)  
      - **DD/MM/YYYY** (e.g., `01/02/1982`, `01/01/1984`)  
    - If the date appears in an **invalid format** (e.g., `/B08/02/1982` or `/DB09/01/1984`), **correct it** to `DD/MM/YYYY`.  
    - Remove any unnecessary prefixes or symbols before the date.  
    - If only a **year** is provided, extract it as the DOB (e.g., `1947`).  
    - If no valid date is found, return `"Not Found"`.  
 
    ### Output Format:
    Return only this **strict JSON object** (no other text):

    {
      "Name": "Extracted Name or 'Not Found'",
      "DOB": "Extracted DOB or 'Not Found'",
      "Number": "Valid 12-digit number or 'Not Found'"
    }

    """
    
    try:
        # Load the image
        image = PIL.Image.open(image_path)

        # Generate content
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            contents=[image, prompt],
            generation_config=genai.types.GenerationConfig(temperature=0)
        )

        # Extract text and clean it
        raw_text = response.text.strip().strip("```json").strip("```").strip()
        
        print("collectd_daya==============>",json.loads(raw_text))
        # Parse and return JSON
        return json.loads(raw_text)
    
    except Exception as e:
        return {
            "Name": "Error",
            "DOB": "Error",
            "Number": f"Error: {str(e)}"
        }
