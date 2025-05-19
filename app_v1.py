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
from paddleocr import PaddleOCR

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

# 🧪 Enhanced Image preprocessing with CLAHE + Bilateral + Better Skew Correction
def preprocess_image(image):
    try:
        # Convert PIL to OpenCV BGR format
        opencv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Apply adaptive thresholding for better skew detection
        thresh_for_skew = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV, 15, 10)

        # Find coordinates of non-zero pixels
        coords = np.column_stack(np.where(thresh_for_skew > 0))
        if coords.shape[0] < 10:
            print("⚠️ Not enough points for skew detection")
            return image

        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

        (h, w) = gray.shape
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(opencv_img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Apply bilateral filtering (denoise while preserving edges)
        filtered = cv2.bilateralFilter(rotated, d=9, sigmaColor=75, sigmaSpace=75)

        # Convert to grayscale again after filtering
        gray_filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold for binarization
        threshold = cv2.adaptiveThreshold(gray_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 3)

        # Sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(threshold, -1, kernel)

        # Convert back to PIL Image
        final_image = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
        return final_image

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
    clean_text = " ".join(raw_text.split())
    clean_text = clean_text.replace("\n", " ").replace("\r", " ")
    return clean_text.strip()

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
            You are a strict JSON generator AI. From the following OCR text extracted from an Indian ID card, extract ONLY clearly visible fields and return the output as a **valid, parsable JSON object**.

            ❗ Absolutely no explanation, commentary, markdown formatting, or code block syntax — just the JSON.

            Extract only the following fields if found in the text:
            - Full Name
            - Date of Birth
            - Address
            - ID Number
            - Contact Number
            - Father's Name
            - Gender
            - Organization / Issuing Authority

            OCR Text:
            {clean_ocr_text(raw_text)}

            Output:
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