from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import requests
# from modules.market_data import fetch_market_data_and_log
from milvus_client import connect_to_milvus, get_or_create_collection, search
from dotenv import load_dotenv
import os
from modules.market_cron import market_analysis_job, start_scheduler

app = Flask(__name__)
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
OLLAMA_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "trading-model"
TOP_K = 3

# Initialize Milvus and SentenceTransformer
connect_to_milvus()
print("Loading SentenceTransformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

start_scheduler()
market_analysis_job()
# Initialize gold trading documents
collection = get_or_create_collection(
    dim=384
)  # all-MiniLM-L6-v2 produces 384-dim embeddings


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # Determine query type
    is_trading_query = any(
        term in question.lower()
        for term in [
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

    # Always fetch embeddings and retrieved docs
    query_embedding = embedder.encode([question])[0]
    retrieved_docs = search(collection, query_embedding, TOP_K)
    context = "\n".join(retrieved_docs)

    # Trading-specific logic
    market_data = None
    # if is_trading_query:
    #     market_data = fetch_market_data_and_log(collection)

    # Always create a prompt
    prompt = f"""Use the following context{" and market data" if is_trading_query else ""} 
to answer the question. Provide a step-by-step reasoning process labeled 'Thinking,' 
followed by the final answer labeled 'Answer'.

Context:
{context}

{"Market Data:\n" + str(market_data) if market_data else ""}

Question: {question}

Response format:
<Thinking>
[Your step-by-step reasoning here]
</Thinking>
<Answer>
[Your final answer here]
</Answer>"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        result = response.json()
        raw_response = result.get("response", "")

        # thinking_start = raw_response.find("<Thinking>")
        # thinking_end = raw_response.find("</Thinking>")
        answer_start = raw_response.find("<Answer>")
        answer_end = raw_response.find("</Answer>")

        # thinking_start != -1 and thinking_end != -1 and
        if answer_start != -1 and answer_end != -1:
            # thinking = raw_response[thinking_start + len("<Thinking>"):thinking_end].strip()
            answer = raw_response[answer_start + len("<Answer>") : answer_end].strip()
        else:
            thinking = "The model did not provide a clear thinking process."
            answer = raw_response

        response_data = {
            "answer": answer,
            # "thinking": thinking,
            "context": retrieved_docs,
        }

        if market_data:
            response_data["market_data"] = market_data

        return jsonify(response_data)
    except Exception as e:
        return (
            jsonify({"error": "Failed to get valid response", "details": str(e)}),
            500,
        )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
