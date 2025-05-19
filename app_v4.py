
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
from pan import extract_pan_details_from_image
from udayam import extract_udayam_details_from_image
from adhaar import extract_adhaar_details_from_image
from card_classifier import classify_document_type_from_image

import google.generativeai as genai
import PIL.Image
from main import classify


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

# 🧹 Clean OCR text
def clean_ocr_text(raw_text):
    # Remove unwanted extra spaces and newlines
    clean_text = " ".join(raw_text.split())  # This will remove leading/trailing spaces and replace multiple spaces with one.
    clean_text = clean_text.replace("\n", " ").replace("\r", " ")  # Handle newlines and carriage returns.
    return clean_text.strip()  # Ensure no leading/trailing spaces remain.



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

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_image_path = os.path.join(IMAGE_SAVE_DIR, f"input_image_{timestamp}.png")
        image.save(saved_image_path)

        # Preprocess
        enhanced_image = preprocess_image(image)
        enhanced_path = os.path.join(IMAGE_SAVE_DIR, f"enhanced_image_{timestamp}.png")
        enhanced_image.save(enhanced_path)

        # 🔥 Get classification + result
        collected_data = classify(enhanced_path)

        return jsonify(collected_data)

    except Exception as e:
        print(f"❌ Error during OCR extraction: {e}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500




# # 🚀 Main extraction route
# @app.route("/extract", methods=["POST"])
# def extract_card_info():
#     try:
#         data = request.get_json()
#         image_base64 = data.get("image")
#         if not image_base64:
#             return jsonify({"error": "No image provided"}), 400
#         if "," in image_base64:
#             image_base64 = image_base64.split(",")[1]

#         image_bytes = base64.b64decode(image_base64)
#         image = Image.open(io.BytesIO(image_bytes))

#         # Save input image
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         saved_image_path = os.path.join(IMAGE_SAVE_DIR, f"input_image_{timestamp}.png")
#         image.save(saved_image_path)
#         print(f"✅ Image saved at {saved_image_path}")

#         # Preprocess image
#         enhanced_image = preprocess_image(image)
#         enhanced_path = os.path.join(IMAGE_SAVE_DIR, f"enhanced_image_{timestamp}.png")
#         print("image path========>, ", enhanced_path)
#         enhanced_image.save(enhanced_path)

#         collected_data = classify(enhanced_path)
#         return jsonify(collected_data)


#     except Exception as e:
#         print(f"❌ Error during OCR extraction: {e}")
#         return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# 🎨 Frontend
@app.route("/")
def home():
    return render_template("index.html")

# 🏁 Run server
if __name__ == "__main__":
    app.run(debug=True)
