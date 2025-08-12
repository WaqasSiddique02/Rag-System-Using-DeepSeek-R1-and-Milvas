from flask import Flask, request, jsonify
import requests
import re
from sentence_transformers import SentenceTransformer
from milvus_client import connect_to_milvus, get_or_create_collection, search
from dotenv import load_dotenv
import os
from modules.market_cron import market_analysis_job, start_scheduler, fetch_binance_data, SYMBOLS

app = Flask(__name__)
load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = "trading-model"
TOP_K = 3
# Initialize Milvus and SentenceTransformer
connect_to_milvus()
print("Loading SentenceTransformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

start_scheduler()
market_analysis_job()
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
    if is_trading_query:
        market_data = {}
        for symbol in SYMBOLS:
            market_data[symbol] = fetch_binance_data(symbol)

    # Always create a prompt
    prompt = f"""Use the following context and market data to answer the question. Stick strictly to the provided data—do not invent facts, speculate, or add unrelated information. For comparisons or calculations, extract numbers step-by-step, perform arithmetic accurately (e.g., subtract precisely), and explain each step. Do not include any special tokens like <|endoftext|> or repeats.

Context:
{context}

Market Data:
{str(market_data) if market_data else "No live market data available—use context only."}

Question: {question}

Response format:
<Thinking>
[Step-by-step reasoning: Extract data, calculate if needed, compare explicitly.]
</Thinking>
<Answer>
[Concise final answer based on reasoning.]
</Answer>"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        result = response.json()
        raw_response = result.get("response", "")

        # Clean up common Llama artifacts
        raw_response = raw_response.replace("<|endoftext|>", "").strip()
        # Remove duplicates (simple heuristic)
        if raw_response.count("<Answer>") > 1:
            raw_response = raw_response[:raw_response.find("<Answer>", raw_response.find("<Answer>") + 1)] + raw_response[raw_response.rfind("<Answer>"):]

        thinking_start = raw_response.find("<Thinking>")
        thinking_end = raw_response.find("</Thinking>")
        answer_start = raw_response.find("<Answer>")
        answer_end = raw_response.find("</Answer>")

        if thinking_start != -1 and thinking_end != -1 and answer_start != -1 and answer_end != -1:
            thinking = raw_response[thinking_start + len("<Thinking>"):thinking_end].strip()
            answer = raw_response[answer_start + len("<Answer>"):answer_end].strip()
        else:
            thinking = "The model did not provide a clear thinking process."
            answer = raw_response

        # Optional: Basic validation (e.g., re-compute if difference mentioned)
        if "difference" in answer.lower() or "lower" in answer.lower() or "higher" in answer.lower():
            # Extract numbers heuristically and validate
            prices = re.findall(r'\d+\.\d+', answer)
            if len(prices) >= 2:
                btc_price = float(prices[0])
                eth_price = float(prices[1])
                diff = abs(btc_price - eth_price)
                if f"${diff:.2f}" not in answer:
                    answer += f" (Note: Verified difference is ${diff:.2f}; BTC is {'higher' if btc_price > eth_price else 'lower'} than ETH.)"

        response_data = {
            "answer": answer,
            "thinking": thinking,
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