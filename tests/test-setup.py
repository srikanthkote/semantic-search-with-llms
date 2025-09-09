import os
import getpass
from dotenv import load_dotenv
import google.generativeai as genai
import certifi

# Load environment variables
load_dotenv()

print(certifi.where())

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key: ")

for model in genai.list_models():
    print(model.name)
