import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
import os
import pathlib
import httpx

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the genai client with the API key
genai.configure(api_key=api_key)
# client = genai.Client()


# doc_url = "https://discovery.ucl.ac.uk/id/eprint/10089234/1/343019_3_art_0_py4t4l_convrt.pdf"
doc_url = "https://www.paradisegp.com/wp-content/uploads/2025/03/Viewing_Seafood-Menu-Vivo-Mar-2.pdf"


# Download and save the PDF
filepath = pathlib.Path('file.pdf')
filepath.write_bytes(httpx.get(doc_url).content)

prompt = "Summarize this document"
response = models.generate_content(
    model="gemini-2.0-flash",
    contents=[
        types.Part.from_bytes(
            data=filepath.read_bytes(),
            mime_type='application/pdf',
        ),
        prompt
    ]
)

print(response.text)
