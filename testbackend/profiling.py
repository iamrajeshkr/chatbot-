#!/usr/bin/env python

import asyncio
import websockets
import os
import time
from langchain_ollama import OllamaLLM  # Updated import from langchain-ollama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory

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
        summarization_llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0.2)
        
        # Initialize ConversationSummaryBufferMemory for this connection.
        memory = ConversationSummaryBufferMemory(
            llm=summarization_llm,
            max_token_limit=1000,  # adjust as needed
            memory_key="chat_history",
            return_messages=True
        )
        
        async for message in websocket:
            overall_start = time.perf_counter()
            print("Received message:", message, flush=True)
            
            # Measure document retrieval time.
            retrieval_start = time.perf_counter()
            docs = await asyncio.to_thread(retriever.get_relevant_documents, message)
            retrieval_time = time.perf_counter() - retrieval_start
            print(f"Document retrieval took {retrieval_time:.4f} seconds", flush=True)
            
            context = "\n".join([doc.page_content for doc in docs])
            
            # Define the system prompt.
            system_prompt = (
                "You are an AI chatbot that leverages a knowledge base built from uploaded documents to answer specific queries. "
                "When provided with document context, carefully extract and use only the information that directly answers the user's question—avoid including extraneous details. "
                "If the user's query clearly refers to the documents, answer strictly based on the provided context. "
                "However, if the available context does not fully address the question, or if the user's message is casual (like a greeting), "
                "then either indicate that the documents do not contain a complete answer or write the best close answer you can make from the available data, "
                "respond in a friendly, human-like manner."
            )
            
            # Measure conversation memory load time.
            memory_load_start = time.perf_counter()
            memory_summary = memory.load_memory_variables({}).get("chat_history", "")
            memory_load_time = time.perf_counter() - memory_load_start
            print(f"Conversation memory load took {memory_load_time:.4f} seconds", flush=True)
            
            # Construct the final prompt.
            modified_message = (
                f"System: {system_prompt}\n\n"
                f"Conversation History:\n{memory_summary}\n\n"
                f"Context:\n{context}\n\n"
                f"User: {message}"
            )
            
            # Prepare an accumulator for the output tokens.
            output_tokens = []
            
            # Instantiate the accumulating callback handler.
            callback_handler = AccumulatingCallbackHandler(websocket, loop, output_tokens)
            
            # Instantiate the LLM with the streaming callback.
            llm = OllamaLLM(
                model="qwen2.5:7b",
                temperature=0.2,
                callbacks=[callback_handler]
            )
            
            # Measure LLM call time.
            llm_start = time.perf_counter()
            await asyncio.to_thread(llm, modified_message)
            llm_time = time.perf_counter() - llm_start
            print(f"LLM call took {llm_time:.4f} seconds", flush=True)
            
            # Combine the tokens to form the full response.
            full_response = "".join(output_tokens)
            
            # Measure conversation memory save time.
            memory_save_start = time.perf_counter()
            memory.save_context({"input": message}, {"output": full_response})
            memory_save_time = time.perf_counter() - memory_save_start
            print(f"Memory save took {memory_save_time:.4f} seconds", flush=True)
            
            # Send a final marker to indicate the end of the streamed response.
            await websocket.send("[END]")
            
            overall_time = time.perf_counter() - overall_start
            print(f"Total processing time for the message: {overall_time:.4f} seconds", flush=True)
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
