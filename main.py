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

# CONSTANTS

CHUNK_SIZE = 5 # Chunk size in terms of pages.
K = 5 # How many documents to retrieve?
MCL = 5 # Maximum context length.
API_KEY = "your-api-key-here"

########################################################
# The functions below are utility functions.
########################################################

# Greet
def greet():
    print("WELCOME")

# Get question
def get_question():
    """Function to get input from the user"""
    question = input("Please enter a question in natural language:")
    return question

# Output
def output(question, vector_store):
    result = retrieve_sections(question, K, vector_store)
    print(result)

# Extract text from pdf
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
    
# Get pdf location
def get_pdf_location():
    """Function to get input from the user"""
    file_path = input("Please enter the relative pdf location:")
    return file_path

# String length statistics
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

# Print document statistics
def print_doc_stats(pages):
    print("Some information about the length of each page. (length of strings after parsing, not the word count)")
    string_length_statistics(pages)

########################################################
# The functions above are utility functions.
########################################################

# Generate chunks (a list of strings) from the whole text.
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

# Format the retrieved documents.
def format_docs(docs):
    docs_update = []
    for doc in docs:
        start_index = doc.metadata.get('start_index', 'no idea')
        end_index = doc.metadata.get('end_index', 'no idea')-1
        docs_update.append(f"start_index: {start_index}, end_index: {end_index}")# - Content: {doc.page_content}
    return "\n".join(docs_update)

# Retrieve sections from the vector store.
def retrieve_sections(question, count, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": count})
    rag_chain = (
        retriever | format_docs
    )
    # Run the chain to retrieve and format the documents
    retrieved_docs = rag_chain.invoke(question)
    return retrieved_docs

# Initialize the vector store and store the documents.
def store_chunks(chunks):
    # Initialize the embedding model and vector store.
    # Store the chunks in the vector store.
    # Batch processing with batch size of 10.
    # Token per minute limit.
    api_key = API_KEY
    embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = Chroma(embedding_function=embedding_model)
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

    return vector_store
    
# Divide the pages into chunks of different sizes.
def pages_to_chunks(pages):
    chunk_sizes = [CHUNK_SIZE]
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

    return chunks

# Prompt for pdf location until a valid one is provided.
def generate_pages():

    got_pdf_location=False
    while got_pdf_location==False:
        try:
            pdf_location = get_pdf_location()
            pages = extract_text_from_pdf(pdf_location)
            print_doc_stats(pages)
            got_pdf_location=True
        except:
            print("TRY AGAIN")

    return pages

# Question loop.
def question_loop(vector_store):
    # Question loop.
    while True:
        question = get_question()
        output(question, vector_store)

# M2.
def m2():

    def format_docs(docs):
        docs_update = []
        for doc in docs:
            docs_update.append(f"Content: {doc.page_content}")
        return "\n".join(docs_update)

    # Function to retrieve context (stubbed for now)
    def retrieve_context(question):
        retriever = vector_store.as_retriever(search_kwargs={"k": K})
        rag_chain = (
            retriever | format_docs
        )
        # Run the chain to retrieve and format the documents
        retrieved_docs = rag_chain.invoke(question)
        return retrieved_docs
    
    import openai
    openai.api_key = API_KEY
    
    greet()
    pages = generate_pages()
    chunks = pages_to_chunks(pages)
    vector_store = store_chunks(chunks)

    """def get_response_from_openai(chat_history):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_history
        )
        return response.choices[0].message.content"""
    
    def get_response_from_openai(chat_history):
        # Set max_context_length to either MCL or current chat history length, whichever is smaller
        max_context_length = min(MCL, len(chat_history))
        min_context_length = 1   # Minimum of 1 message

        while max_context_length >= min_context_length:
            try:
                # Limit the context window to the last N messages
                truncated_chat_history = chat_history[-max_context_length:]
                
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=truncated_chat_history
                )
                return response.choices[0].message.content
            
            except Exception as e:
                print(f"Error with context length {max_context_length}: {e}")
                max_context_length -= 1  # Reduce the context length and try again

        # If all attempts fail, raise an exception
        raise RuntimeError("Failed to get a response from OpenAI with any context length.")
    
    def chat():

        # Initialize the chat history with the system message
        chat_history = [{"role": "system", "content": "You are a helpful assistant."}]
        # Main chat function
        while True:
            # Step 1: Get the user's question
            user_question = input("Enter your question (or type 'exit' to quit): ")
            if user_question.lower() == 'exit':
                break
            # Step 2: Retrieve context using the retrieve_context function
            context = retrieve_context(user_question)
            # Step 3: Combine the user's question with the context
            combined_input = f"Context:You are an AI LLM that is used in a RAG system (AEC industry, document reviews). Your goal is \
                to use the context below: (that is retrieved with semantic search from \
                    a vector database to answer the user's question.){context}\n\nUser's Question: {user_question}"
            # Add the user's question to the chat history
            chat_history.append({"role": "user", "content": combined_input})
            # Step 4: Get the response from OpenAI API
            assistant_response = get_response_from_openai(chat_history)
            # Add the assistant's response to the chat history
            chat_history.append({"role": "assistant", "content": assistant_response})
            # Step 5: Display the response to the user
            print("Assistant:", assistant_response)

    # Start the chat application
    chat()   

# M1.
def m1():
    greet()
    pages = generate_pages()
    chunks = pages_to_chunks(pages)
    vector_store = store_chunks(chunks)
    question_loop(vector_store)

# Main
def main():
    m2()
    
if __name__ == "__main__":
    main()
