import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
import numpy as np
#from util import string_length_statistics
import PyPDF2

def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text_by_page = []
        # Extract text from each page and store it in the list
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            text_by_page.append(text)
        return text_by_page
    
def get_pdf_location():
    """Function to get input from the user"""
    file_path = input("Please enter the relative pdf location:")
    return file_path

def get_question():
    """Function to get input from the user"""
    question = input("Please enter a question in natural language:")
    return question

def greet():
    print("WELCOME")

def string_length_statistics(strings):
    # Calculate the length of each string
    lengths = [len(s) for s in strings]
    
    # Calculate statistics
    mean_length = np.mean(lengths)
    variance_length = np.var(lengths)
    std_dev_length = np.std(lengths)
    min_length = np.min(lengths)
    max_length = np.max(lengths)

    # Print the results
    print(f"Average length: {mean_length}")
    print(f"Variance: {variance_length}")
    print(f"Standard deviation: {std_dev_length}")
    print(f"Minimum length: {min_length}")
    print(f"Maximum length: {max_length}")

def print_doc_stats(pages):
    print("Some information about the length of each page. (length of strings after parsing, not the word count)")
    string_length_statistics(pages)

def chunk_generator(chunk_size, string_list):
    # Calculate the number of strings each element in the new list should have
    count = len(string_list) // chunk_size
    
    # Create the new list of combined strings
    combined_strings = []
    for i in range(count):
        # Calculate start and end indices for slicing
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        
        # Ensure the slice does not exceed the list boundaries
        if start_index < len(string_list):
            # Extracting string from each dictionary assuming the key for string is 'string'
            combined_string = ' '.join(item['string'] if isinstance(item, dict) else item 
                                       for item in string_list[start_index:end_index])
            combined_strings.append({"start_index": start_index, "end_index": end_index, "string": combined_string})
    
    return combined_strings

# Post-processing
def format_docs(docs):
    docs_update = []
    for doc in docs:
        start_index = doc.metadata.get('start_index', 'no idea')
        end_index = doc.metadata.get('end_index', 'no idea')-1
        docs_update.append(f"start_index: {start_index}, end_index: {end_index}")# - Content: {doc.page_content}
    return "\n".join(docs_update)

# Retrieve-sections
def retrieve_sections(question, count, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": count})
    rag_chain = (
        retriever | format_docs
    )
    # Run the chain to retrieve and format the documents
    retrieved_docs = rag_chain.invoke(question)
    return retrieved_docs

def output(question, vector_store):
    result = retrieve_sections(question, 5, vector_store)
    print(result)

def main():
    greet()
    got_pdf_location=False
    while got_pdf_location==False:
        try:
            pdf_location = get_pdf_location()
            pages = extract_text_from_pdf(pdf_location)
            print_doc_stats(pages)
            got_pdf_location=True
        except:
            print("TRY AGAIN")
    chunk_sizes = [5]
    chunks = []
    # Generate chunks and create Document objects
    for c_size in chunk_sizes:
        chunk = chunk_generator(c_size, pages)
        for idx in range(len(chunk)):
            chunks.append(
                Document(
                    page_content=chunk[idx]["string"],
                    metadata={
                        "start_index": chunk[idx]["start_index"],
                        "end_index": chunk[idx]["end_index"],
                    },
                )
            )
    # Initialize the embedding model with the API key
    api_key = "your-api-key-here"
    embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Initialize an empty Chroma vector store with the embedding function
    vector_store = Chroma(embedding_function=embedding_model)
    
    # Why? TPM limit.
    # Batch processing with batch size of 10
    batch_size = 10
    num_chunks = len(chunks)
    for i in range(0, num_chunks, batch_size):
        # Get the batch of documents
        batch_chunks = chunks[i:i + batch_size]
        texts = [doc.page_content for doc in batch_chunks]
        metadatas = [doc.metadata for doc in batch_chunks]
        # Embed the batch of documents
        embeddings = embedding_model.embed_documents(texts)
        # Add texts, embeddings, and metadata to the vector store
        vector_store.add_texts(texts=texts, embeddings=embeddings, metadatas=metadatas)

    # Loop
    while True:
        question = get_question()
        output(question, vector_store)

if __name__ == "__main__":
    main()
