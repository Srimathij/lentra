import google.generativeai as genai
from dotenv import load_dotenv
import os
import PIL.Image
import json

load_dotenv()
GOOGLE_API = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API)

# Define the function
def extract_udayam_details_from_image(image_path: str) -> dict:
    prompt = """
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

    ### Output Format:
    Return only this **strict JSON object** (no other text):

    {
        "Name": "Extracted Enterprise Name or 'Not Found'",
        "DOB": "Extracted Udyam Registration Number or 'Not Found'",
        "Number": "Extracted Number or 'Not Found'",
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
        
        # Parse and return JSON
        return json.loads(raw_text)
    
    except Exception as e:
        return {
            "Enterprise_Name": "Error",
            "Udyam_Registration_Number": "Error",
            "Type_of_Enterprise": "Error",
            "Owner_Name": "Error",
            "Address": f"Error: {str(e)}"
        }
