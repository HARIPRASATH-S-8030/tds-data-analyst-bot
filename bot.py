import json
import time
import os
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from openai import OpenAI
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    ContextTypes,
    filters,
)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
LOG_URL = os.getenv("LOG_URL", "")

client = OpenAI(
    base_url="https://aipipe.org/openai/v1",
    api_key=AIPIPE_TOKEN,
)

LOG_FILE = "run.jsonl"

conversation_history = {}


def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    log_event({
        "type": "incoming",
        "chat_id": chat_id,
        "text": user_text,
    })

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    system_prompt = (
        "You are a careful data analyst. "
        "The user's LAST message asks a data-analysis question and tells you exactly "
        "what JSON shape to reply with. "
        "Work out the real answer. "
        "Reply with ONLY that exact JSON object and absolutely nothing else."
    )

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": system_prompt}] + history[-6:],
    )

    reply_text = response.choices[0].message.content.strip()

    history.append({
        "role": "assistant",
        "content": reply_text,
    })

    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        start = reply_text.find("{")
        end = reply_text.rfind("}")
        parsed = json.loads(reply_text[start:end + 1])

    parsed["log_url"] = LOG_URL

    final_reply = json.dumps(parsed)

    log_event({
        "type": "outgoing",
        "chat_id": chat_id,
        "text": final_reply,
    })

    await update.message.reply_text(final_reply)


telegram_app = (
    ApplicationBuilder()
    .token(TELEGRAM_BOT_TOKEN)
    .build()
)

telegram_app.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_message,
    )
)


async def run_bot():
    await telegram_app.initialize()
    await telegram_app.start()
    await telegram_app.updater.start_polling()


@asynccontextmanager
async def lifespan(app: FastAPI):
    bot_task = asyncio.create_task(run_bot())
    print("Telegram bot started.")
    yield
    bot_task.cancel()
    await telegram_app.stop()
    await telegram_app.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def health():
    return {"status": "ok"}


@app.get("/run.jsonl")
async def get_log():
    if not os.path.exists(LOG_FILE):
        open(LOG_FILE, "a").close()

    return FileResponse(
        LOG_FILE,
        media_type="application/json",
        filename="run.jsonl",
    )