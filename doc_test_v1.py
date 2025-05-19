from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import pathlib

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Generative AI client
genai.configure(api_key=api_key)
client = genai.Client()

# Load local PDF file
filepath = pathlib.Path('test_doc.pdf')  # Change to your actual file name

# Ensure the file exists
if not filepath.exists():
    raise FileNotFoundError(f"{filepath} not found.")

# Generate content from PDF and prompt
prompt = "Summarize this document"
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        types.Part.from_bytes(
            data=filepath.read_bytes(),
            mime_type='application/pdf',
        ),
        prompt
    ]
)

# Print the result
print(response.text)
