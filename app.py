import os
import re
import tempfile
from collections.abc import Iterator
from threading import Thread

import cv2
import gradio as gr
from loguru import logger
from PIL import Image
from llama_cpp import Llama, llama_chat_format
import torch
from sentence_transformers import SentenceTransformer
import faiss
from duckduckgo_search import DDGS
import pdfplumber
import docx
import pickle

# Paths to model files
MODEL_PATH = os.getenv("MODEL_PATH", "Lingshu-7B.Q8_0.gguf")
MM_PROJ_PATH = os.getenv("MM_PROJ_PATH", "Lingshu-7B.mmproj-Q8_0.gguf")
GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "100"))
if not torch.cuda.is_available():
    GPU_LAYERS = 0
logger.info(f"GPU layers: {GPU_LAYERS}")

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not os.path.isfile(MM_PROJ_PATH):
    raise FileNotFoundError(f"Projection file not found: {MM_PROJ_PATH}")

chat_handler = llama_chat_format.Llava15ChatHandler(clip_model_path=MM_PROJ_PATH)

llama = Llama(
    model_path=MODEL_PATH,
    chat_handler=chat_handler,
    chat_format="qwen2",
    n_gpu_layers=GPU_LAYERS,
    n_ctx=4096,
    n_batch=512,
    verbose=True,
)

# Embedding setup for episodic memory
embed_device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device=embed_device,
)
FAISS_INDEX = "memory.index"
FAISS_TEXTS = "memory.pkl"
if os.path.exists(FAISS_INDEX) and os.path.exists(FAISS_TEXTS):
    index = faiss.read_index(FAISS_INDEX)
    with open(FAISS_TEXTS, "rb") as f:
        memory_texts = pickle.load(f)
else:
    index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
    memory_texts = []

MAX_NUM_IMAGES = int(os.getenv("MAX_NUM_IMAGES", "5"))


def add_to_memory(text: str) -> None:
    embedding = embedder.encode([text])
    index.add(embedding)
    memory_texts.append(text)
    faiss.write_index(index, FAISS_INDEX)
    with open(FAISS_TEXTS, "wb") as f:
        pickle.dump(memory_texts, f)


def search_memory(query: str, top_k: int = 3) -> list[str]:
    if index.ntotal == 0:
        return []
    embedding = embedder.encode([query])
    distances, ids = index.search(embedding, top_k)
    return [memory_texts[i] for i in ids[0] if i != -1]


def web_search(query: str, num_results: int = 3) -> list[str]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            results.append(f"{r['title']}: {r['href']}")
    return results


def extract_text_from_file(path: str) -> str:
    try:
        if path.lower().endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        if path.lower().endswith(".docx"):
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to parse {path}: {e}")
        return ""


def handle_docs(paths: list[str]) -> None:
    for path in paths:
        if path.lower().endswith((".pdf", ".docx", ".txt")):
            text = extract_text_from_file(path)
            if text:
                add_to_memory(text)


def _url_from_path(path: str) -> str:
    return f"file://{os.path.abspath(path)}"


def _is_image(path: str) -> bool:
    return path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))


def count_files_in_new_message(paths: list[str]) -> tuple[int, int]:
    image_count = 0
    video_count = 0
    for path in paths:
        if path.lower().endswith(".mp4"):
            video_count += 1
        elif _is_image(path):
            image_count += 1
    return image_count, video_count


def count_files_in_history(history: list[dict]) -> tuple[int, int]:
    image_count = 0
    video_count = 0
    for item in history:
        if item["role"] != "user" or isinstance(item["content"], str):
            continue
        path = item["content"][0]
        if path.lower().endswith(".mp4"):
            video_count += 1
        elif _is_image(path):
            image_count += 1
    return image_count, video_count


def validate_media_constraints(message: dict, history: list[dict]) -> bool:
    new_image_count, new_video_count = count_files_in_new_message(message["files"])
    history_image_count, history_video_count = count_files_in_history(history)
    image_count = history_image_count + new_image_count
    video_count = history_video_count + new_video_count
    if video_count > 1:
        gr.Warning("Only one video is supported.")
        return False
    if video_count == 1:
        if image_count > 0:
            gr.Warning("Mixing images and videos is not allowed.")
            return False
        if "<image>" in message["text"]:
            gr.Warning("Using <image> tags with video files is not supported.")
            return False
    if video_count == 0 and image_count > MAX_NUM_IMAGES:
        gr.Warning(f"You can upload up to {MAX_NUM_IMAGES} images.")
        return False
    if "<image>" in message["text"] and message["text"].count("<image>") != new_image_count:
        gr.Warning("The number of <image> tags in the text does not match the number of images.")
        return False
    return True


def downsample_video(video_path: str) -> list[tuple[Image.Image, float]]:
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(total_frames // MAX_NUM_IMAGES, 1)
    frames: list[tuple[Image.Image, float]] = []

    for i in range(0, min(total_frames, MAX_NUM_IMAGES * frame_interval), frame_interval):
        if len(frames) >= MAX_NUM_IMAGES:
            break

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))

    vidcap.release()
    return frames


def process_video(video_path: str) -> list[dict]:
    content = []
    frames = downsample_video(video_path)
    for frame in frames:
        pil_image, timestamp = frame
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            pil_image.save(temp_file.name)
            content.append({"type": "text", "text": f"Frame {timestamp}:"})
            content.append({"type": "image_url", "image_url": _url_from_path(temp_file.name)})
    logger.debug(f"{content=}")
    return content


def process_interleaved_images(message: dict) -> list[dict]:
    logger.debug(f"{message['files']=}")
    parts = re.split(r"(<image>)", message["text"])
    logger.debug(f"{parts=}")

    content = []
    image_index = 0
    for part in parts:
        if part == "<image>":
            path = message["files"][image_index]
            content.append({"type": "image_url", "image_url": _url_from_path(path)})
            image_index += 1
        elif part.strip():
            content.append({"type": "text", "text": part.strip()})
    logger.debug(f"{content=}")
    return content


def process_new_user_message(message: dict) -> list[dict]:
    text = message["text"]
    files = [p for p in message["files"] if _is_image(p) or p.lower().endswith(".mp4")]
    if not files:
        return [{"type": "text", "text": text}]
    if files[0].lower().endswith(".mp4"):
        return [{"type": "text", "text": text}, *process_video(files[0])]
    if "<image>" in message["text"]:
        message_local = {"text": text, "files": [p for p in files if _is_image(p)]}
        return process_interleaved_images(message_local)
    return [
        {"type": "text", "text": text},
        *[{"type": "image_url", "image_url": _url_from_path(path)} for path in files if _is_image(path)],
    ]


def process_history(history: list[dict]) -> list[dict]:
    messages = []
    current_user_content: list[dict] = []
    for item in history:
        if item["role"] == "assistant":
            if current_user_content:
                messages.append({"role": "user", "content": current_user_content})
                current_user_content = []
            if isinstance(item["content"], str):
                messages.append({"role": "assistant", "content": item["content"]})
            else:
                messages.append({"role": "assistant", "content": item["content"]})
        else:
            content = item["content"]
            if isinstance(content, str):
                current_user_content.append({"type": "text", "text": content})
            else:
                current_user_content.append({"type": "image_url", "image_url": _url_from_path(content[0])})
    if current_user_content:
        messages.append({"role": "user", "content": current_user_content})
    return messages


def generate_stream(messages: list[dict], max_new_tokens: int) -> Iterator[str]:
    stream = llama.create_chat_completion(
        messages=messages,
        temperature=0.2,
        top_p=0.95,
        repeat_penalty=1.1,
        stop=["USER:"],
        max_tokens=max_new_tokens,
        stream=True,
    )
    output = ""
    for chunk in stream:
        delta = chunk["choices"][0]["delta"].get("content", "")
        output += delta
        yield output


@gr.experimental.Function
def run(message: dict, history: list[dict], system_prompt: str = "", max_new_tokens: int = 2048) -> Iterator[str]:
    if not validate_media_constraints(message, history):
        yield ""
        return

    handle_docs(message["files"])

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(process_history(history))

    context_parts = []
    retrieved = search_memory(message["text"])
    if retrieved:
        context_parts.append("Memory:\n" + "\n".join(retrieved))
    search_results = web_search(message["text"])
    if search_results:
        context_parts.append("Web search:\n" + "\n".join(search_results))
    if context_parts:
        messages.append({"role": "system", "content": "\n\n".join(context_parts)})

    messages.append({"role": "user", "content": process_new_user_message(message)})

    output_text = ""
    for delta in generate_stream(messages, max_new_tokens):
        output_text = delta
        yield delta

    add_to_memory(f"Q: {message['text']}\nA: {output_text}")


DESCRIPTION = """\
This is a demo of Lingshu 7B running with llama-cpp and GPU acceleration.\
Upload images or a short video and ask questions in text.\
"""

demo = gr.ChatInterface(
    fn=run,
    type="messages",
    chatbot=gr.Chatbot(type="messages", scale=1, allow_tags=["image"]),
    textbox=gr.MultimodalTextbox(file_types=["image", ".mp4"], file_count="multiple", autofocus=True),
    multimodal=True,
    additional_inputs=[
        gr.Textbox(label="System Prompt", value="You are a helpful medical expert."),
        gr.Slider(label="Max New Tokens", minimum=100, maximum=4096, step=10, value=2048),
    ],
    stop_btn=False,
    title="Lingshu 7B",
    description=DESCRIPTION,
    run_examples_on_click=False,
    cache_examples=False,
    css_paths="style.css",
    delete_cache=(1800, 1800),
)

if __name__ == "__main__":
    demo.launch()
