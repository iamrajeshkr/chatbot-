#!/usr/bin/env python

import asyncio
import os
from langchain_ollama import OllamaLLM
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.embeddings import OllamaEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.memory import ConversationSummaryBufferMemory

# Import websockets correctly
try:
    import websockets
except ImportError:
    print("Websockets module not found. Please install with 'pip install websockets'")
    exit(1)

# Connect to the persisted Chroma database.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma(
    embedding_function=embeddings, 
    persist_directory=r"C:\Users\20368750\Documents\vercel_chatbot\chatbot-ui\testbackend\chroma_db3"
)
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# Custom callback handler to stream tokens via websocket and accumulate tokens.
class AccumulatingCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket, loop, accumulator):
        self.websocket = websocket
        self.loop = loop
        self.accumulator = accumulator

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Append the token to the accumulator.
        self.accumulator.append(token)
        # Send the token to the client.
        asyncio.run_coroutine_threadsafe(self.websocket.send(token), self.loop)

async def handle_message(websocket):
    try:
        # Capture the current event loop.
        loop = asyncio.get_running_loop()
        
        # Create a separate summarization LLM (without streaming) for conversation memory.
        # You can adjust the model and temperature as needed.
        summarization_llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0.2)
        
        # Initialize ConversationSummaryBufferMemory for this connection.
        memory = ConversationSummaryBufferMemory(
            llm=summarization_llm,
            max_token_limit=1000,  # adjust as needed
            memory_key="chat_history",
            return_messages=True
        )
        
        async for message in websocket:
            print("Received message:", message, flush=True)
            
            # Check for flags in the message
            use_knowledge_base = True
            is_guest_chat = False
            
            # Process the |noKB flag (for knowledge base toggle)
            if "|noKB" in message:
                use_knowledge_base = False
                message = message.replace("|noKB", "")
                print("Knowledge base disabled for this message", flush=True)
            
            # Process the |guest flag (for guest chat)
            if "|guest" in message:
                is_guest_chat = True
                message = message.replace("|guest", "")
                print("Guest chat message received", flush=True)
            
            # Retrieve context from the vector database only if using knowledge base
            context = ""
            if use_knowledge_base:
                docs = retriever.get_relevant_documents(message)
                context = "\n".join([doc.page_content for doc in docs])
                print(f"Retrieved {len(docs)} documents from knowledge base", flush=True)
            else:
                print("Skipping knowledge base retrieval", flush=True)
            
            # Define the system prompt.
            system_prompt = (
                "You are an AI chatbot that leverages a knowledge base built from uploaded documents to answer specific queries. "
                "When provided with document context, carefully extract and use only the information that directly answers the user's question—avoid including extraneous details. "
            )
            
            # Extend system prompt based on whether knowledge base is used
            if use_knowledge_base:
                system_prompt += (
                    "If the user's query clearly refers to the documents, answer strictly based on the provided context. "
                    "However, if the available context does not fully address the question, or if the user's message is casual (like a greeting), "
                    "then either indicate that the documents do not contain a complete answer or write the best close answer you can make from the the available data. "
                )
            else:
                system_prompt += (
                    "For this message, you're not using the knowledge base, so you should respond based on your general knowledge. "
                    "Be helpful, creative, and answer to the best of your ability without referencing specific documents. "
                )
            
            # Add guest chat information if applicable
            if is_guest_chat:
                system_prompt += "This is a guest chat without user authentication. Keep responses professional but engaging. "
            
            system_prompt += "Respond in a friendly, human-like manner."
            
            # Skip memory for guest chat or handle differently if needed
            memory_summary = ""
            if not is_guest_chat:
                # Retrieve the current conversation summary from memory.
                memory_summary = memory.load_memory_variables({}).get("chat_history", "")
            
            # Construct the final prompt including the system prompt, memory summary, retrieved document context, and user message.
            modified_message = (
                f"System: {system_prompt}\n\n"
            )
            
            if memory_summary:
                modified_message += f"Conversation History:\n{memory_summary}\n\n"
            
            if context:
                modified_message += f"Context:\n{context}\n\n"
            
            modified_message += f"User: {message}"
            
            # Prepare an accumulator for the output tokens.
            output_tokens = []
            
            # Instantiate the accumulating callback handler.
            callback_handler = AccumulatingCallbackHandler(websocket, loop, output_tokens)
            
            # Instantiate the LLM with the streaming callback.
            llm = OllamaLLM(
                model="qwen2.5:3b",
                temperature=0.2,
                callbacks=[callback_handler]
            )
            
            # Call the LLM in a thread-safe way with the modified message.
            await asyncio.to_thread(llm, modified_message)
            
            # Combine the tokens to form the full response.
            full_response = "".join(output_tokens)
            
            # Save the current conversation turn to memory, but only if not a guest chat
            if not is_guest_chat:
                memory.save_context({"input": message}, {"output": full_response})
            
            # Send a final marker to indicate the end of the streamed response.
            await websocket.send("[END]")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

async def main():
    print("WebSocket server starting", flush=True)
    async with websockets.serve(
        handle_message,
        "0.0.0.0",
        int(os.environ.get('PORT', 8090))
    ) as server:
        print("WebSocket server running on port 8090", flush=True)
        await asyncio.Future()  # Run forever.

if __name__ == "__main__":
    asyncio.run(main())
