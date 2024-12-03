# Constants.
CHUNK_SIZE = 5 # Chunk size in terms of pages.
K = 5 # How many documents to retrieve?
MCL = 5 # Maximum context length.
API_KEY = "your-api-key-here"
model_list = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
MODEL_NAME = model_list[0]
import openai
openai.api_key = API_KEY
PAGE_RANGE = False
PDF_LOCATION = "pdf/can_parse/project_manual.pdf"
CHAT = False
