#
#   File:           main.py
#   Description:    Main loop of program.
#

#   Includes
#----------------------------------------------------------------------#
from config import *
from pdfExtract import *
from chat import *
from langchain_chroma.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
import PyPDF2
from langchain.schema import HumanMessage
#**********************************************************************#



#----------------------------------------------------------------------#
def generate_pages(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text_by_page = []
        # Extract text from each page and store it in the list
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            text_by_page.append(text)
        return text_by_page
#**********************************************************************#



#----------------------------------------------------------------------#
def generate_chunks(pages):

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
#**********************************************************************#



#----------------------------------------------------------------------#
def pdf_location_to_chunks(pdf_location):
    # if the system is using a page range.
    if PAGE_RANGE == True:
        # parse the pdf, make pages
        pages = generate_pages(pdf_location)
        # generate chunks by grouping the pages
        chunks = generate_chunks(pages)
    # if the system is using sections.
    else:
        # get the sections
        doc = pdf.open(pdf_location)
        # sections[i][0] = title, sections[i][1] = text.
        sections = extract_sections(doc)
        # create the chunks
        chunks = []
        for section in sections:
            chunks.append(
                Document(
                    page_content=section.text,
                    metadata={
                        "title": section.title,
                    },
                )
            )
    return chunks
#**********************************************************************#



#----------------------------------------------------------------------#
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
#**********************************************************************#



#----------------------------------------------------------------------#
def get_user_question():
    user_question = input("Enter your question (or type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        return 0
    return user_question
#**********************************************************************#



#----------------------------------------------------------------------#
def main():
    # 1. Parse the PDF, seperate it into chunks.
    chunks = pdf_location_to_chunks(PDF_LOCATION)
    # 2. Store the chunks in a vector database.
    db = store_chunks(chunks)
    if CHAT == True:
        # 3. Chat.
        chat(db)
    # Assuming 'db' is your vector database and 'K' is the number of documents to retrieve
    else:
        # Get the question.
        user_question = input("Enter your question (or type 'exit' to quit'): ")
        if user_question.lower() != 'exit':
            # Create a retriever
            retriever = db.as_retriever(search_kwargs={"k": K})
            # Retrieve relevant documents
            docs = retriever.invoke(user_question)
            # Display the titles of the retrieved documents
            for i, doc in enumerate(docs, 1):
                print(f"[{i}] {doc.metadata['title']}")
            print("Example response: 1, 2")
            indices_string = input("Enter response: ")
            indices = [int(i) for i in indices_string.replace(" ", "").split(",")]
            print(indices)
            # Collect the selected documents
            selected_docs = [docs[index - 1] for index in indices]
            print(len(selected_docs))
            context = "\n".join([doc.page_content for doc in selected_docs])
            # Create the combined input
            combined_input = f"Context: You are an AI LLM used in a RAG system (AEC industry, document reviews). Your goal is to use the context below (retrieved with semantic search from a vector database) to answer the user's question.\n\n{context}\n\nUser's Question: {user_question}"
            try:
                # Initialize the ChatOpenAI model
                llm = ChatOpenAI(model_name=MODEL_NAME, openai_api_key=API_KEY)  # or pass openai_api_key='your-api-key-here' if not set as env variable
                # Create a HumanMessage with the input
                human_message = HumanMessage(content=combined_input)
                # Generate the response using invoke
                response = llm.invoke([human_message])
                # Access the content of the response
                print("Assistant:", response.content)
            except Exception as e:
                print(f"Error generating response: {e}")
#**********************************************************************#



if __name__ == "__main__":
    main()
    