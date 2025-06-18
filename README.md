# Medical Multimodal Assistant

This project provides a simple web interface built with [Gradio](https://gradio.app) for interacting with the **Lingshu-7B** model (GGUF format) via `llama-cpp`. The assistant is designed to help clinicians with general questions and supports reasoning over multiple modalities.

## Features

- **Long chain-of-thought reasoning** guided by a custom system prompt.
- **Retrieval augmented generation (RAG)** using DuckDuckGo web search.
- **Episodic memory** stores past conversations in a FAISS index for context.
- **Multimodal input**: optional image uploads are captioned automatically.
- **Document ingestion**: text, PDF and DOCX files can be uploaded and will be
  included in the context for the language model.
- **User accounts**: visitors must register and log in. Only subscribed users
  can access the assistant.
- **Admin panel** for managing subscriptions.
- **Russian and English locales** with a simple language switch.

## Usage

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Building `llama-cpp-python` requires a C++ compiler and other development
   libraries. Installation may fail if those dependencies are missing.
2. Place the Lingshu-7B GGUF model file on disk and set the `MODEL_PATH`
   environment variable to its location (default:
   `models/Lingshu-7B-Q4_0.gguf`).
3. Set a random `SECRET_KEY` environment variable for session cookies.
4. Run the app:
   ```bash
   python app.py
   ```
5. Open the URL printed by Uvicorn in your browser (default: `http://localhost:8000`).

Open `/register` to create your first account, then log in at `/login`.
Users marked as `is_subscriber` can access the assistant. Admin users can
toggle subscription status for any account at `/admin`.

After logging in with a subscribed account, open `/chat/en` or `/chat/ru` to
access the assistant. It will reason step by step about the question, using
uploaded images and documents, as well as recent web search results. The final
answer is generated after the reasoning steps and includes a short disclaimer
that the response should not replace professional medical advice.

Each dialogue is stored in a semantic memory using sentence-transformer
embeddings and a FAISS index. When you ask a new question, the assistant
retrieves the most relevant past exchanges to provide additional context.

## Limitations

- The web search uses a public DuckDuckGo API and may return limited results.
- Uploaded files are read synchronously and should be reasonably small.
- Model inference can be slow on machines without GPU acceleration.

## License

This project is provided as-is for educational purposes.

## Deployment

### Docker

1. Build the image:
   ```bash
   docker build -t medical-assistant .
   ```
2. Run the container, mounting the model directory and setting environment variables:
   ```bash
   docker run -p 8000:8000 \
     -e MODEL_PATH=/models/Lingshu-7B-Q4_0.gguf \
     -e SECRET_KEY=$(openssl rand -hex 16) \
     -v /path/to/models:/models \
     medical-assistant
   ```

Persist `users.db` and the FAISS index by mounting a directory at `/app` if you
want to keep data between restarts.

