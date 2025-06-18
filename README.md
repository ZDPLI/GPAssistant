# Medical Multimodal Assistant

This project demonstrates the **Lingshu‑7B** multimodal model running locally with [llama.cpp](https://github.com/ggerganov/llama.cpp).  The web UI follows a ChatGPT-style layout and accepts medical images or a short video.  The model runs with GPU acceleration when available.

Uploaded documents (PDF, DOCX or text) are indexed for retrieval‑augmented generation (RAG). The assistant can also perform web searches to enrich its answers.

## Requirements

- Python 3.10+
- A GPU supported by `llama-cpp-python`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Download the model

Download the GGUF weights and multimodal projection from the [Lingshu‑7B‑GGUF repository](https://huggingface.co/mradermacher/Lingshu-7B-GGUF):

```bash
wget -O Lingshu-7B.Q8_0.gguf \
  https://huggingface.co/mradermacher/Lingshu-7B-GGUF/resolve/main/Lingshu-7B.Q8_0.gguf
wget -O Lingshu-7B.mmproj-Q8_0.gguf \
  https://huggingface.co/mradermacher/Lingshu-7B-GGUF/resolve/main/Lingshu-7B.mmproj-Q8_0.gguf
```

Set the paths before starting the app:

```bash
export MODEL_PATH="$(pwd)/Lingshu-7B.Q8_0.gguf"
export MM_PROJ_PATH="$(pwd)/Lingshu-7B.mmproj-Q8_0.gguf"
# Optional: number of layers to offload to GPU
export N_GPU_LAYERS=100
```

## Running

Start the Gradio interface:

```bash
python app.py
```

Open the printed URL in your browser to chat with the assistant.  You can change the maximum number of images processed from a video with the `MAX_NUM_IMAGES` environment variable.

### Episodic memory and documents

Uploaded PDF, DOCX or text files are parsed and stored in a local FAISS index (`memory.index`).  The assistant retrieves the most relevant snippets for each question and combines them with optional web search results from DuckDuckGo.
