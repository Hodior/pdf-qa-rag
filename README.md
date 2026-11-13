# PDF Q&A chatbot (RAG)

Upload a PDF and ask questions about it. The app finds the most relevant passages and has Google Gemini
answer from them only.

## How it works

1. **Parse:** [MarkItDown](https://github.com/microsoft/markitdown) turns the PDF into text.
2. **Chunk:** text is split into 400-word chunks with a 50-word overlap.
3. **Embed:** each chunk is embedded with the Gemini embedding model (`gemini-embedding-001`).
4. **Search:** the question is embedded too, and cosine similarity in NumPy picks the top 5 chunks.
5. **Answer:** the chunks go to a Gemini model with an instruction to answer from that context.
6. **Interface:** Gradio, with a PDF upload and a chat window.

The vector store is a plain NumPy array in memory, which keeps the retrieval step easy to read.

## Run it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your-key          # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
python app.py
```

Then open the local URL Gradio prints, upload a PDF, click **Process PDF** and ask a question.

## Stack

Python, Gradio, MarkItDown, Google Gemini API (embeddings + generation), NumPy.
