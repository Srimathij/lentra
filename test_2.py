import cv2
from pyzbar.pyzbar import decode
from PIL import Image

# Load the Aadhaar card image (make sure the QR code is visible and clear)
image_path = 'add.jpg'  # Path to your Aadhaar card image
img = cv2.imread(image_path)

print("code running...")
# Decode the QR code
qr_codes = decode(img)

# Extract the QR code data
for qr in qr_codes:
    qr_code_data = qr.data
    print("QR Code Data (bytes):", qr_code_data)
