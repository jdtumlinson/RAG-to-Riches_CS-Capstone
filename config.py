# Constants.

CHUNK_SIZE = 5 # Chunk size in terms of pages.
K = 5 # How many documents to retrieve?
MCL = 5 # Maximum context length.
API_KEY = "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"

import openai
openai.api_key = API_KEY

PAGE_RANGE = True
PDF_LOCATION = "project_manual.pdf"