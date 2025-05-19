from aadhaar.secure_qr import extract_data

# Replace 'received_qr_code_data' with the actual QR code data extracted from the Aadhaar card
received_qr_code_data = 12345678
extracted_data = extract_data(received_qr_code_data)

# Accessing the extracted information
print(extracted_data.text_data)
print(extracted_data.image)
print(extracted_data.contact_info)

# Converting to dictionary format
data_dict = extracted_data.to_dict()
