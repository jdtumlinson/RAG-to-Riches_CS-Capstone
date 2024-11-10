# Constants.

CHUNK_SIZE = 5 # Chunk size in terms of pages.
K = 5 # How many documents to retrieve?
MCL = 5 # Maximum context length.
API_KEY = "your-api-key-here"

import openai
openai.api_key = API_KEY

PAGE_RANGE = False
PDF_LOCATION = "project_manual.pdf"