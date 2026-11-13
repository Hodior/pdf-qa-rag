import gradio as gr
from markitdown import MarkItDown
import numpy as np
from numpy.linalg import norm
import google.generativeai as genai
import os

# Globals to act as our simple in-memory vector store
vector_store = None
chunks = []

def process_pdf(pdf_file):
    global vector_store, chunks
    if pdf_file is None:
        return "Please upload a PDF file."
    
    # 1. Convert PDF to text using MarkItDown
    md = MarkItDown()
    try:
        result = md.convert(pdf_file.name)
        text = result.text_content
    except Exception as e:
        return f"Error converting PDF: {e}"
    
    # 2. Chunk the text
    # Simple chunking by splitting words into fixed-size chunks with some overlap
    chunk_size = 400
    overlap = 50
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
            
    if not chunks:
        return "No text found or extracted from the PDF."
        
    # 3. Create Vector Store using NumPy (with Google Gemini)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Please set GEMINI_API_KEY environment variable. It is required for embeddings."
    
    genai.configure(api_key=api_key)
    try:
        response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=chunks,
            task_type="retrieval_document"
        )
        embeddings = response['embedding']
        vector_store = np.array(embeddings)
    except Exception as e:
        return f"Error creating embeddings: {e}"
    
    return f"PDF processed successfully! Created {len(chunks)} chunks. You can now ask questions."

def chat(message, history):
    global vector_store, chunks
    if vector_store is None or not chunks:
        return "Please upload and process a PDF first."
        
    # 4. Encode the user's query
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Please set the GEMINI_API_KEY environment variable."
    genai.configure(api_key=api_key)
    try:
        query_response = genai.embed_content(
            model="models/gemini-embedding-001",
            content=message,
            task_type="retrieval_query"
        )
        query_embedding = query_response['embedding']
    except Exception as e:
        return f"Error creating query embedding: {e}"
    
    # 5. Cosine similarity search (using raw numpy)
    # Cosine similarity = dot(A, B) / (norm(A) * norm(B))
    similarities = np.dot(vector_store, query_embedding) / (norm(vector_store, axis=1) * norm(query_embedding))
    
    # Retrieve top 5 chunks
    top_k = min(5, len(chunks))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    retrieved_chunks = [chunks[i] for i in top_indices]
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    # 6. Send to LLM
    prompt = (
        f"Context information is below.\n"
        f"---------------------\n"
        f"{context}\n"
        f"---------------------\n"
        f"Given the context information, answer the user's question: {message}"
    )
    
    
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Fast Gemini model for answering
            model = genai.GenerativeModel(
                "gemini-3-flash-preview",
                system_instruction="You are a helpful assistant that answers questions accurately based on the provided PDF context."
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"LLM Error: {e}"
    else:
        # Fallback if no API key is set
        return (
            f"**LLM Mock Response (No GEMINI_API_KEY found):**\n\n"
            f"I retrieved the top {top_k} chunks from the PDF based on your question.\n\n"
            f"Here is the context I would have sent to the LLM:\n"
            f"> {retrieved_chunks[0][:200]}...\n\n"
            f"*(Set your GEMINI_API_KEY environment variable to use the actual LLM)*"
        )

# 7. Wrap this in gradio.ChatInterface
with gr.Blocks() as demo:
    gr.Markdown("# PDF RAG Chatbot")
    gr.Markdown("Upload a PDF document. The app uses `markitdown` for parsing, Google Gemini for encoding and answering, and `numpy` for search.")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        process_button = gr.Button("Process PDF")
    
    status_output = gr.Textbox(label="Status", interactive=False)
    
    process_button.click(fn=process_pdf, inputs=pdf_input, outputs=status_output)
    
    # Setup the Chat Interface
    gr.ChatInterface(fn=chat)

if __name__ == "__main__":
    demo.launch()
