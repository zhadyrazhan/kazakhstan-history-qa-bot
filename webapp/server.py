import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent  # kazakhstan-history-qa-bot/
HISTORY_TEXT_PATH = ROOT_DIR / "history_text.txt"
STATIC_DIR = Path(__file__).parent / "static"
CHAT_MODEL = "gpt-5-mini"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in webapp/.env (copy .env.example)")

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_INSTRUCTION = """Ты — дружелюбный помощник по истории Казахстана для учеников 11 класса.
Отвечай ТОЛЬКО на основе приведённого ниже фрагмента учебника. Если в тексте
нет ответа на вопрос, честно скажи, что в этом фрагменте учебника такой
информации нет — не придумывай факты. Отвечай кратко и по существу
(2-4 предложения), на том языке, на котором задан вопрос.

Фрагмент учебника:
{context}
"""

app = FastAPI()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


def load_context() -> str:
    if not HISTORY_TEXT_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "history_text.txt ещё не сгенерирован. Запусти секции 1-6 "
                "history_finetuning.ipynb (OCR + датасет) и положи файл в корень проекта."
            ),
        )
    text = HISTORY_TEXT_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise HTTPException(status_code=503, detail="history_text.txt пустой.")
    return text


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Пустой вопрос.")

    context = load_context()
    try:
        response = client.responses.create(
            model=CHAT_MODEL,
            instructions=SYSTEM_INSTRUCTION.format(context=context),
            input=question,
            reasoning={"effort": "minimal"},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}")

    return AskResponse(answer=response.output_text)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
