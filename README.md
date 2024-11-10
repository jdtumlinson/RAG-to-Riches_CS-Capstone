# RAG to Riches

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

## Extra

### Chunk Generation

#### What are chunks?
A chunk is a list of LangChain documents.

#### What is a LangChain document?
A LangChain document is a way to represent data. It might be overcomplicating things, but it was used in the template notebook, and it could change in the future.

#### How to generate chunks from a page range?
To generate chunks from a page range, you can use the following approach:
```python
Document(
    page_content=chunk[idx]["string"],
    metadata={
        "start_index": chunk[idx]["start_index"],
        "end_index": chunk[idx]["end_index"],
    }
)
```
#### How to generate chunks from sections?
To generate chunks from sections, you can use the following approach:
```python
Document(
    page_content=chunk[idx]["string"],
    metadata={
        "section_id": section_id
    }
)