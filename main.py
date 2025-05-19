from flask import Flask, request, jsonify
from pan import extract_pan_details_from_image
from udayam import extract_udayam_details_from_image
from adhaar import extract_adhaar_details_from_image
from card_classifier import classify_document_type_from_image
import os

app = Flask(__name__)  # ✅ This is what Gunicorn needs

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = f"temp_{file.filename}"
    file.save(file_path)

    try:
        card_type = classify_document_type_from_image(file_path)
        print("card type========>", card_type)

        if card_type == "Aadhaar Card":
            result = extract_adhaar_details_from_image(file_path)
        elif card_type == "PAN Card":
            result = extract_pan_details_from_image(file_path)
        elif card_type == "Udyam Certificate":
            result = extract_udayam_details_from_image(file_path)
        else:
            return jsonify({"error": "Invalid Card Type"}), 400

        return jsonify({
            "card_type": card_type,
            "data": result
        })
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
