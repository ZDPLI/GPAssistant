"""Medical assistant web application powered by the Lingshu-7B model.

The app exposes a Gradio based chat UI themed to resemble LM Studio. Users
can tweak decoding parameters such as temperature and top-p during
inference.  A small FastAPI layer provides authentication with optional
subscription gating.  Conversations are stored in a FAISS index so relevant
exchanges can be retrieved for context.

Both English and Russian locales are supported via a simple language switch.
"""

import os
import pickle
from functools import partial
from typing import Iterable, List

import numpy as np
import faiss

import gradio as gr
from duckduckgo_search import DDGS
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    pipeline,
)
from sentence_transformers import SentenceTransformer
import torch
from pypdf import PdfReader
from docx import Document

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
)
from sqlalchemy.orm import sessionmaker, declarative_base
from werkzeug.security import generate_password_hash, check_password_hash


# System prompts for both locales. They instruct the model to provide
# step-by-step medical reasoning and end with a short disclaimer.
SYSTEM_PROMPT_EN = (
    "You are DoctorGPT, a medical assistant helping clinicians.\n"
    "1. Carefully analyse the question and available context from web search,\n"
    "documents and images.\n"
    "2. Think through the problem in a detailed chain-of-thought.\n"
    "3. State your final answer clearly. If uncertain, say so.\n"
    "Always finish with a reminder that you do not replace a professional doctor."
)

SYSTEM_PROMPT_RU = (
    "Вы DoctorGPT, медицинский помощник для врачей.\n"
    "1. Внимательно анализируйте вопрос и контекст из поиска, документов и изображений.\n"
    "2. Размышляйте последовательно, описывая цепочку рассуждений.\n"
    "3. Чётко формулируйте итоговый ответ и сообщайте, если не уверены.\n"
    "В конце напомните, что ответ не заменяет консультацию специалиста."
)

SYSTEM_PROMPT = {"en": SYSTEM_PROMPT_EN, "ru": SYSTEM_PROMPT_RU}

# Simple localisation dictionary for UI texts.
LOCALES = {
    "en": {
        "title": "Medical Multimodal Assistant",
        "ask": "Ask a question",
        "image": "Upload an image (optional)",
        "docs": "Upload documents",
        "send": "Send",
        "subscription_required": "An active subscription is required to access the assistant.",
        "login": "Login",
        "register": "Register",
        "username": "Username",
        "password": "Password",
        "submit": "Submit",
        "logout": "Logout",
        "admin_panel": "Admin panel",
        "not_admin": "Admin access required.",
        "users": "Users",
        "subscriber": "Subscriber",
        "start_chat": "Start Chat",
        "landing_info": "AI medical assistant with multimodal capabilities.",
    },
    "ru": {
        "title": "Медицинский мультимодальный ассистент",
        "ask": "Задайте вопрос",
        "image": "Загрузите изображение (необязательно)",
        "docs": "Загрузите документы",
        "send": "Отправить",
        "subscription_required": "Для доступа к ассистенту требуется активная подписка.",
        "login": "Вход",
        "register": "Регистрация",
        "username": "Имя пользователя",
        "password": "Пароль",
        "submit": "Отправить",
        "logout": "Выйти",
        "admin_panel": "Панель администратора",
        "not_admin": "Требуются права администратора.",
        "users": "Пользователи",
        "subscriber": "Подписка",
        "start_chat": "Начать чат",
        "landing_info": "Медицинский ИИ ассистент с поддержкой нескольких модальностей.",
    },
}

# --- Episodic memory setup using FAISS ---
MEMORY_INDEX_PATH = "memory.index"
MEMORY_STORE_PATH = "memory.pkl"
# Use GPU for embedding model if available
EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

if os.path.exists(MEMORY_INDEX_PATH) and os.path.exists(MEMORY_STORE_PATH):
    memory_index = faiss.read_index(MEMORY_INDEX_PATH)
    with open(MEMORY_STORE_PATH, "rb") as f:
        memory_texts = pickle.load(f)
else:
    memory_index = faiss.IndexFlatL2(EMBED_MODEL.get_sentence_embedding_dimension())
    memory_texts: List[str] = []

def _save_memory() -> None:
    faiss.write_index(memory_index, MEMORY_INDEX_PATH)
    with open(MEMORY_STORE_PATH, "wb") as f:
        pickle.dump(memory_texts, f)

def add_to_memory(question: str, answer: str) -> None:
    """Store a Q&A pair in the semantic memory."""
    text = f"Q: {question}\nA: {answer}"
    vector = EMBED_MODEL.encode([text]).astype("float32")
    memory_index.add(vector)
    memory_texts.append(text)
    _save_memory()

def retrieve_memory(query: str, k: int = 3) -> List[str]:
    """Return similar stored dialogues to the query."""
    if not memory_texts:
        return []
    vector = EMBED_MODEL.encode([query]).astype("float32")
    _, idx = memory_index.search(vector, k)
    return [memory_texts[i] for i in idx[0] if i < len(memory_texts)]

# --- Database setup for user accounts ---

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password_hash = Column(String)
    is_admin = Column(Boolean, default=False)
    is_subscriber = Column(Boolean, default=False)


engine = create_engine("sqlite:///users.db")
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_lang(request: Request) -> str:
    return request.session.get("lang", "en")


def set_lang(request: Request, lang: str) -> None:
    if lang in LOCALES:
        request.session["lang"] = lang


def get_current_user(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    with SessionLocal() as db:
        return db.query(User).filter(User.id == user_id).first()

# Paths to model and multimodal projection
MODEL_PATH = os.getenv("MODEL_PATH", "lingshu-medical-mllm/Lingshu-7B")

# Load the model with GPU acceleration when available
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# Translation pipelines for input and output
device = 0 if torch.cuda.is_available() else -1
translator_ru_en = pipeline(
    "translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en", device=device
)
translator_en_ru = pipeline(
    "translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru", device=device
)

def search_web(query, k=3):
    """Return web search summaries using DuckDuckGo."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=k):
            results.append(f"{r['title']}: {r['body']} ({r['href']})")
    return "\n".join(results)


def _extract_text_from_pdf(path: str) -> str:
    """Return extracted text from a PDF file."""
    reader = PdfReader(path)
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )


def _extract_text_from_docx(path: str) -> str:
    """Return extracted text from a docx file."""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_documents(files: Iterable[gr.File]) -> str:
    """Concatenate text extracted from uploaded files."""
    texts: List[str] = []
    for f in files or []:
        if not f:
            continue
        path = f.name
        if path.endswith(".pdf"):
            texts.append(_extract_text_from_pdf(path))
        elif path.endswith(".docx"):
            texts.append(_extract_text_from_docx(path))
        else:  # assume plain text
            with open(path, "r", errors="ignore") as fh:
                texts.append(fh.read())
    return "\n".join(texts)

def generate_answer(
    question,
    image=None,
    documents=None,
    chat_history=None,
    lang="en",
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_tokens=768,
    repeat_penalty=1.1,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    system_prompt=None,
):
    """Generate a response given user question, image and documents."""

    chat_history = chat_history or []
    context_parts: List[str] = []

    # Translate question to English for processing
    question_en = translator_ru_en(question)[0]["translation_text"]

    memory_snippets = retrieve_memory(question_en)
    if memory_snippets:
        context_parts.append("Previous dialogues:\n" + "\n".join(memory_snippets))


    if documents:
        docs_text = load_documents(documents)
        if docs_text:
            context_parts.append(f"Document excerpts:\n{docs_text}")

    search_context = search_web(question_en)
    context_parts.append(f"Search results:\n{search_context}")
    context = "\n".join(context_parts)

    user_prompt = (
        f"Question: {question_en}\n\nContext:\n{context}\n\n"
        "Please reason step by step, then give your final answer prefixed "
        "with 'Final Answer:'."
    )

    system_prompt = system_prompt or SYSTEM_PROMPT.get(lang, SYSTEM_PROMPT_EN)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image is not None:
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repeat_penalty,
        )
        answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repeat_penalty,
        )
        answer = tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()

    # Translate model answer to Russian
    answer_ru = translator_en_ru(answer)[0]["translation_text"]

    chat_history.append((question, answer_ru))
    add_to_memory(question_en, answer)
    return "", chat_history, chat_history


def create_demo(lang: str) -> gr.Blocks:
    texts = LOCALES[lang]
    theme = gr.themes.Default(primary_hue="blue").set(
        body_background_fill="#fafafa",
        body_text_color="#111",
        input_background_fill="white",
        input_text_color="#111",
        block_background_fill="white",
    )
    css = ".input-text input{color:#111}!important;"
    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(f"# {texts['title']}")
        chatbot = gr.Chatbot(height=500)
        state = gr.State([])

        with gr.Row():
            txt = gr.Textbox(label=texts['ask'], scale=6, elem_classes="input-text")
            send = gr.Button(texts['send'], scale=1)
        img = gr.Image(type="pil", label=texts['image'])
        docs = gr.File(label=texts['docs'], file_count="multiple")

        with gr.Accordion("Advanced", open=False):
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top-p")
            top_k = gr.Slider(1, 100, value=40, step=1, label="Top-k")
            max_tokens = gr.Slider(64, 2048, value=768, step=1, label="Max tokens")
            repeat_penalty = gr.Slider(0.5, 2.0, value=1.1, step=0.05, label="Repeat penalty")
            presence_penalty = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Presence penalty")
            frequency_penalty = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Frequency penalty")
            system_prompt = gr.Textbox(label="System prompt", value=SYSTEM_PROMPT[lang], lines=3)

        send.click(
            partial(generate_answer, lang=lang),
            inputs=[
                txt,
                img,
                docs,
                state,
                temperature,
                top_p,
                top_k,
                max_tokens,
                repeat_penalty,
                presence_penalty,
                frequency_penalty,
                system_prompt,
            ],
            outputs=[txt, chatbot, state],
        )
    return demo


app = FastAPI()
SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


demo_en = create_demo("en")
demo_ru = create_demo("ru")

app = gr.mount_gradio_app(app, demo_en, path="/chat/en")
app = gr.mount_gradio_app(app, demo_ru, path="/chat/ru")


def render_template(title: str, body: str) -> HTMLResponse:
    head = (
        "<head>"
        "<meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
        "<link href='https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css' rel='stylesheet'>"
        "<script src='https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js'></script>"
        "</head>"
    )
    return HTMLResponse(f"<html>{head}<body class='container'>{body}</body></html>")


@app.get("/set_lang/{lang}")
async def set_language(request: Request, lang: str):
    set_lang(request, lang)
    return RedirectResponse("/", status_code=302)


@app.get("/register", response_class=HTMLResponse)
async def register_form(request: Request):
    lang = get_lang(request)
    t = LOCALES[lang]
    body = (
        f"<h3 class='center-align'>{t['register']}</h3>"
        f"<form method='post' class='row'>"
        f"<div class='input-field col s12'><input id='username' name='username' type='text'/><label for='username'>{t['username']}</label></div>"
        f"<div class='input-field col s12'><input id='password' name='password' type='password'/><label for='password'>{t['password']}</label></div>"
        f"<button class='btn waves-effect waves-light' type='submit'>{t['submit']}</button>"
        f"</form>"
        f"<p class='center-align'><a href='/login'>{t['login']}</a></p>"
    )
    return render_template(t['register'], body)


@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, username: str = Form(...), password: str = Form(...)):
    lang = get_lang(request)
    with SessionLocal() as db:
        if db.query(User).filter(User.username == username).first():
            return render_template("error", "User exists")
        user = User(
            username=username,
            password_hash=generate_password_hash(password),
            is_admin=False,
            is_subscriber=False,
        )
        db.add(user)
        db.commit()
    return RedirectResponse("/login", status_code=302)


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    lang = get_lang(request)
    t = LOCALES[lang]
    body = (
        f"<h3 class='center-align'>{t['login']}</h3>"
        f"<form method='post' class='row'>"
        f"<div class='input-field col s12'><input id='username' name='username' type='text'/><label for='username'>{t['username']}</label></div>"
        f"<div class='input-field col s12'><input id='password' name='password' type='password'/><label for='password'>{t['password']}</label></div>"
        f"<button class='btn waves-effect waves-light' type='submit'>{t['submit']}</button>"
        f"</form>"
        f"<p class='center-align'><a href='/register'>{t['register']}</a></p>"
    )
    return render_template(t['login'], body)


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return render_template("error", "Invalid credentials")
        request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    lang = get_lang(request)
    t = LOCALES[lang]
    user = get_current_user(request)
    if not user or not user.is_admin:
        return render_template(t['admin_panel'], t['not_admin'])
    rows = []
    with SessionLocal() as db:
        for u in db.query(User).all():
            toggle_link = f"/toggle_sub/{u.id}"
            rows.append(
                f"<tr><td>{u.username}</td>"
                f"<td>{'✔' if u.is_subscriber else ''}</td>"
                f"<td><a class='btn-small' href='{toggle_link}'>{t['subscriber']}</a></td></tr>"
            )
    table = "<table class='striped'><thead><tr><th>" + t['users'] + "</th><th>" + t['subscriber'] + "</th><th></th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    body = f"<h3 class='center-align'>{t['admin_panel']}</h3>" + table + f"<p class='center-align'><a href='/'>{t['logout']}</a></p>"
    return render_template(t['admin_panel'], body)


@app.get("/toggle_sub/{user_id}")
async def toggle_subscription(request: Request, user_id: int):
    user = get_current_user(request)
    if not user or not user.is_admin:
        return RedirectResponse("/", status_code=302)
    with SessionLocal() as db:
        target = db.query(User).get(user_id)
        if target:
            target.is_subscriber = not target.is_subscriber
            db.add(target)
            db.commit()
    return RedirectResponse("/admin", status_code=302)


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    lang = get_lang(request)
    t = LOCALES[lang]
    user = get_current_user(request)
    body = (
        f"<div class='section center-align'>"
        f"<h3>{t['title']}</h3>"
        f"<p>{t['landing_info']}</p>"
        f"</div>"
    )
    if user and user.is_subscriber:
        body += (
            f"<div class='section center-align'>"
            f"<a class='btn' href='/chat/en'>English</a> "
            f"<a class='btn' href='/chat/ru'>Русский</a>"
            f"</div>"
            f"<p class='center-align'><a href='/logout'>{t['logout']}</a></p>"
        )
    elif user:
        body += (
            f"<p class='red-text center-align'>{t['subscription_required']}</p>"
            f"<p class='center-align'><a href='/logout'>{t['logout']}</a></p>"
        )
    else:
        body += (
            f"<div class='section center-align'>"
            f"<a class='btn' href='/login'>{t['login']}</a> "
            f"<a class='btn' href='/register'>{t['register']}</a>"
            f"</div>"
        )
    if user and user.is_admin:
        body += f"<p class='center-align'><a href='/admin'>{t['admin_panel']}</a></p>"
    return render_template(t['title'], body)


if __name__ == "__main__":
    import uvicorn
    import secrets
    from gradio.networking import setup_tunnel

    host = "0.0.0.0"
    port = 8000

    if os.getenv("SHARE", "false").lower() == "true":
        try:
            url = setup_tunnel(
                local_host=host,
                local_port=port,
                share_token=secrets.token_urlsafe(32),
                share_server_address=None,
                share_server_tls_certificate=None,
            )
            print(f"* Running on public URL: {url}")
        except Exception as exc:
            print("Could not create share link:", exc)

    uvicorn.run(app, host=host, port=port)
