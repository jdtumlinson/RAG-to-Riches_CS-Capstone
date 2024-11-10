from config import * 

def chat(db):

    def format_docs(docs):
        docs_update = []
        for doc in docs:
            docs_update.append(f"Content: {doc.page_content}")
        return "\n".join(docs_update)

    # Function to retrieve context (stubbed for now)
    def retrieve_context(question):
        retriever = db.as_retriever(search_kwargs={"k": K})
        rag_chain = (
            retriever | format_docs
        )
        # Run the chain to retrieve and format the documents
        retrieved_docs = rag_chain.invoke(question)
        return retrieved_docs
    
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

        