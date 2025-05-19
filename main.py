import json
from pan import extract_pan_details_from_image
from udayam import extract_udayam_details_from_image
from adhaar import extract_adhaar_details_from_image
from card_classifier import classify_document_type_from_image

def classify(file_path):
    card_type = classify_document_type_from_image(file_path)
    print("card type========>", card_type)

    if card_type == "Aadhaar Card":
        result = extract_adhaar_details_from_image(file_path)

    elif card_type == "PAN Card":
        result = extract_pan_details_from_image(file_path)

    elif card_type == "Udyam Certificate":
        result = extract_udayam_details_from_image(file_path)

    else:
        return {"error": "Invalid Card Type"}

    # ✅ Return structured JSON-like dict (not string)
    return {
        "card_type": card_type,
        "data": result
    }



# def classify(file_path):
#     card_type=classify_document_type_from_image(file_path)
#     print("card type========>",card_type)

#     if card_type=="Aadhaar Card":
#         result = extract_adhaar_details_from_image(file_path)
#         print(json.dumps(result, indent=2))
#         return json.dumps(result, indent=2)

#     elif card_type=="PAN Card":
#         result = extract_pan_details_from_image(file_path)
#         print(json.dumps(result, indent=2))
#         return json.dumps(result, indent=2)

#     elif card_type=="Udyam Certificate":
#         result = extract_udayam_details_from_image(file_path)
#         print(json.dumps(result, indent=2))
#         return json.dumps(result, indent=2)

#     else:
#         return {"error": "Invalid Card Type"}



# # file_path="pan_dummy_2.jpg"
# file_path="udayam_2.webp"


# card_type=classify_document_type_from_image(file_path)
# print("card type===>",card_type)


# if card_type=="Aadhaar Card":
#     result = extract_adhaar_details_from_image(file_path)
#     print(json.dumps(result, indent=2))
#     # return json.dumps(result, indent=2)

# elif card_type=="PAN Card":
#     result = extract_pan_details_from_image(file_path)
#     print(json.dumps(result, indent=2))
#     # return json.dumps(result, indent=2)

# elif card_type=="Udyam Certificate":
#     result = extract_udayam_details_from_image(file_path)
#     print(json.dumps(result, indent=2))
#     # return json.dumps(result, indent=2)

# else:
#     print("Invalid...")