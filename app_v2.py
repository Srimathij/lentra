
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import base64
import io
import requests
import os
import cv2
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import json
from paddleocr import PaddleOCR  # ✅ PaddleOCR

# 🌍 App setup
app = Flask(__name__)
CORS(app)

# 🔐 Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLAMA_MODEL = "llama3-70b-8192"

# 🖼️ Save directory
IMAGE_SAVE_DIR = "saved_images"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# ✅ Initialize PaddleOCR model (English)
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.5)

# 🧪 Image preprocessing
def preprocess_image(image):
    try:
        opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

        (h, w) = gray.shape
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(opencv_img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        threshold = cv2.adaptiveThreshold(gray_rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 3)
        denoised = cv2.GaussianBlur(threshold, (5, 5), 0)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        enhanced_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
        return enhanced_image
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return image

# 🧠 Call Groq LLaMA
def call_groq_llama(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": LLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an AI that extracts structured personal details from OCR text extracted from Indian ID cards."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=body)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# 🧹 Clean OCR text
def clean_ocr_text(raw_text):
    # Remove unwanted extra spaces and newlines
    clean_text = " ".join(raw_text.split())  # This will remove leading/trailing spaces and replace multiple spaces with one.
    clean_text = clean_text.replace("\n", " ").replace("\r", " ")  # Handle newlines and carriage returns.
    return clean_text.strip()  # Ensure no leading/trailing spaces remain.

# 🚀 Main extraction route
@app.route("/extract", methods=["POST"])
def extract_card_info():
    try:
        data = request.get_json()
        image_base64 = data.get("image")
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # Save input image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_image_path = os.path.join(IMAGE_SAVE_DIR, f"input_image_{timestamp}.png")
        image.save(saved_image_path)
        print(f"✅ Image saved at {saved_image_path}")

        # Preprocess image
        enhanced_image = preprocess_image(image)
        enhanced_path = os.path.join(IMAGE_SAVE_DIR, f"enhanced_image_{timestamp}.png")
        enhanced_image.save(enhanced_path)

        # 🧾 Run PaddleOCR
        opencv_img = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
        ocr_results = ocr_model.ocr(opencv_img, cls=True)
        raw_text = " ".join([line[1][0] for line in ocr_results[0]])

        print("🧾 OCR Text:\n", raw_text)

        if not raw_text.strip():
            return jsonify({"error": "OCR returned no readable text. Please check image quality."}), 400

        prompt = f"""
    OCR Text:
    {clean_ocr_text(raw_text)}

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
   
    ### **Rules for Address Extraction:**  
    - Extract only the **address portion** without including gender information.  
    - Ignore words like "Male," "Female," "M/F," or similar gender-related terms.  
    - The address typically contains house numbers, streets, cities, states, and **PIN codes** (which are 6-digit numbers).  
    - **Always include any 6-digit numbers (PIN codes)** that appear near the address block.  
    - Even if the PIN code appears on a new line or at the end, **attach it to the address**.  
    - Ensure the extracted address does not contain **extra demographic details** like gender.    
   
    ### Rules for Relation Name Extraction (W/O, S/O, D/O, C/O):
    - Identify and extract the name associated with **W/O** (Wife of), **S/O** (Son of), **D/O** (Daughter of), or **C/O** (Child of).
    - Extract only the **person's name following the relation tag**, not the tag itself.
    - Example: If the text contains "S/O: Murat Singh", extract only `"Murat Singh"`.
    - **Only one relation** will be mentioned in the text.
    - Do not mention other relations or return "Not Found" for them.
    - If no W/O, S/O, D/O, or C/O is found at all, return `"Relation Name: Not Found"`.
 
    ### **Expected Output Format:**  
    Name: [Extracted Name]  
    DOB: [Extracted DOB]  
    12-digit Identifier: [Valid 12-digit Number] (or "Not Found")  
    Relation Name: [Extracted Relation Name] (or "Not Found")  
    Address: [Extracted Address in a single paragraph.]
    
    """
 

        structured_response = call_groq_llama(prompt)
        print("✅ Groq Response:\n", structured_response)

        try:
            extracted_data = json.loads(structured_response)
        except Exception as e:
            extracted_data = {"error": "Failed to parse Groq response", "details": str(e)}

        return jsonify({"result": extracted_data})

    except Exception as e:
        print(f"❌ Error during OCR extraction: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# 🎨 Frontend
@app.route("/")
def home():
    return render_template("index.html")

# 🏁 Run server
if __name__ == "__main__":
    app.run(debug=True)
