# M1 v0.1

## Overview
This program extracts text from a PDF, splits the text into chunks, and allows you to query the content using natural language. It uses OpenAI's embeddings to convert the text into vectors for efficient retrieval based on your questions.

## Key Tools

- **LangChain**: A framework for handling document processing, embedding, and querying.
- **Chroma**: A vector store used to store and retrieve embedded text.
- **OpenAI Embeddings**: The program uses OpenAI embeddings to convert the document text into vectors.

## How to Run

### 1. Run the Setup Script
1. Make sure the `test.sh` script is executable. If not, run the following command:
   ```bash
   chmod +x test.sh
   ```

2. Run the `test.sh` script to create and set up the virtual environment:
   ```bash
   ./test.sh
   ```

### 2. Activate the Virtual Environment
After running the setup script, activate the virtual environment:
```bash
source test_env/bin/activate
```

### 3. Run the Program
Once the virtual environment is active, run the program:
```bash
python main.py
```

### 4. Provide PDF Path and Ask Questions
- Enter the path to your PDF when prompted.
- Ask questions in natural language, and the program will return relevant sections from the document.
