import os
import re
import threading
import time
import requests
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from milvus_client import connect_to_milvus, get_or_create_collection, search
from modules.market_cron import (
    market_analysis_job,
    start_scheduler,
    SYMBOLS,
    fetch_binance_data,
    shared_embedder,
)
import re

load_dotenv()

app = Flask(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = "trading-model"
TOP_K = 3

# Milvus init
connect_to_milvus()
embedder = shared_embedder  # Use shared instance from market_cron
collection = get_or_create_collection(dim=384)


# ---- Warm up Ollama ----
def warm_up_model():
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "ping",
                "stream": False,
                "raw": True,
                "options": {"num_predict": 2},
                "keep_alive": "10m",
            },
            timeout=20,
        )
        print(f"Ollama warm-up status: {r.status_code}")
    except Exception as e:
        print(f"Ollama warm-up failed: {e}")


threading.Thread(target=warm_up_model, daemon=True).start()


# ---- Ollama Call Helper ----
def call_ollama(prompt: str, timeout: int = 180):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {
            "num_predict": 256,
            "temperature": 0.2,
        },
        "stop": ["</Answer>"],
        "keep_alive": "5m",
    }
    return requests.post(OLLAMA_URL, json=payload, timeout=timeout)


def strip_html_tags(text):
    # Remove anything that looks like an HTML tag
    return re.sub(r"<[^>]*>", "", text).strip()


# ---- API Routes ----
@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question")
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400

        ql = question.lower()
        is_trading_query = any(
            t in ql
            for t in [
                "bitcoin",
                "crypto",
                "stock",
                "trade",
                "market",
                "price",
                "btc",
                "eth",
            ]
        )

        # RAG retrieve
        query_embedding = embedder.encode([question])[0]
        retrieved_docs = search(collection, query_embedding, TOP_K)
        context = "\n".join(retrieved_docs)

        # Optional: fetch live market data
        market_data = None
        if is_trading_query:
            market_data = {}
            for symbol in SYMBOLS:
                market_data[symbol] = fetch_binance_data(symbol)

        prompt = f"""Use the following context and market data to answer the question. 
Stick strictly to the provided data—do not invent facts. Keep reasoning brief.

Context:
{context or "No retrieved context."}

Market Data:
{str(market_data) if market_data else "No live market data available—use context only."}

Question: {question}

Response format:
<Answer>
[Concise final answer]
</Answer>"""

        try:
            response = call_ollama(prompt, timeout=180)
        except requests.exceptions.ReadTimeout:
            return jsonify({"error": "LLM timed out"}), 504

        if response.status_code != 200:
            return (
                jsonify(
                    {
                        "error": "LLM call failed",
                        "status": response.status_code,
                        "body": response.text,
                    }
                ),
                502,
            )

        try:
            result = response.json()
        except ValueError:
            return (
                jsonify(
                    {"error": "LLM returned non-JSON", "body": response.text[:5000]}
                ),
                502,
            )

        raw_response = (
            (result.get("response") or "").replace("<|endoftext|>", "").strip()
        )

        # extract <Answer>...</Answer>
        answer_start = raw_response.find("<Answer>")
        answer_end = raw_response.find("</Answer>")
        if answer_start != -1 and answer_end != -1:
            answer = raw_response[answer_start + len("<Answer>") : answer_end].strip()
        else:
            answer = raw_response

        # 🔹 SHORT-TERM FIX: strip HTML tags from whatever is left
        answer = strip_html_tags(answer)

        payload = {"answer": answer, "context": retrieved_docs}
        if market_data:
            payload["market_data"] = market_data
        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": "Unhandled server error", "details": str(e)}), 500

# @app.route('/query', methods=['POST'])
# def query():
#     data = request.json
#     question = data['question']
#     response = call_ollama(question)  # returns a Response object
    
#     # If it's a `requests.Response`
#     model_response = response.json()  # if API returns JSON
#     # or: model_response = response.text  # if it's plain text

#     return jsonify({
#         "answer": model_response
#     })



@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Prevent duplicate jobs in debug mode
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        start_scheduler()
        market_analysis_job()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
