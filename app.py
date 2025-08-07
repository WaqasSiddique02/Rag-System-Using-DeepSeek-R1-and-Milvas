from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import requests
import numpy as np
from milvus_client import connect_to_milvus, get_or_create_collection, insert_documents, search, check_existing_documents
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

OLLAMA_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"
TOP_K = 3

connect_to_milvus()

documents = [
    "Gold trading involves speculating on the price of gold, often through spot markets, futures, or CFDs. The spot price (XAU) is the current price for immediate delivery, quoted per troy ounce (31.103 grams).",
    "The London Fix is a twice-daily meeting of five LBMA firms to set the global benchmark gold price, influencing worldwide trading.",
    "Bullion refers to gold bars or ingots at least 99.5% pure, traded as good delivery bars (400 oz) in the LBMA system with a traceable chain of integrity.",
    "Bullion coins (e.g., Canadian Maple Leaf, American Buffalo) are valued by their gold content, with a premium of 5-15% over the spot price.",
    "Numismatic coins are valued for rarity or historical significance, not just gold content, unlike bullion coins.",
    "Arbitrage is buying and selling gold in different markets to profit from price differences (e.g., USD vs. GBP).",
    "An assay tests golds purity and weight, using methods like Fire Assay or X-Ray Fluorescence (XRF).",
    "Gold ETFs (e.g., SPDR Gold Shares, GLD) track gold prices without physical ownership, but carry risks like management fees.",
    "XAU/USD is the forex pair for one troy ounce of gold in US dollars, used in spot trading.",
    "Gold prices rise during inflation or geopolitical uncertainty (safe haven) and fall with high interest rates or a strong USD.",
    "COMEX is a major New York-based exchange for gold futures trading, alongside the London OTC market and Shanghai Gold Exchange.",
    "A troy ounce (31.103 grams) is used for gold, distinct from an avoirdupois ounce (28.35 grams).",
    "The chain of integrity ensures LBMA bullion bars are trusted without re-assaying, maintaining a traceable custody chain.",
    "Futures are contracts to buy/sell gold at a future date, while CFDs allow speculation without physical delivery.",
    "The bid price is what a dealer pays for gold, and the ask price is what they sell it for; the spread is the difference."
]

print("Loading SentenceTransformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents).tolist()

dim = len(doc_embeddings[0])
collection = get_or_create_collection(dim)

# Check and insert only new documents
new_documents = check_existing_documents(collection, documents)
if new_documents:
    new_embeddings = embedder.encode(new_documents).tolist()
    insert_documents(collection, new_documents, new_embeddings)
else:
    print("No new documents to insert; all documents already exist in collection")

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # Fetch gold price (XAU/USD)
    try:
        api_url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={ALPHA_VANTAGE_API_KEY}"
        api_response = requests.get(api_url).json()
        quote = api_response.get("Realtime Currency Exchange Rate", {})
        gold_data = (f"Latest gold price (XAU/USD): ${quote.get('5. Exchange Rate', 'N/A')} "
                     f"as of {quote.get('6. Last Refreshed', 'N/A')} "
                     f"(Bid: {quote.get('8. Bid Price', 'N/A')}, Ask: {quote.get('9. Ask Price', 'N/A')})")
    except Exception as e:
        gold_data = f"Failed to fetch gold price: {str(e)}"

    # Fetch gold-related news
    try:
        news_url = f"https://newsapi.org/v2/everything?q=gold+market&sortBy=publishedAt&apiKey={NEWSAPI_API_KEY}"
        news_response = requests.get(news_url).json()
        articles = news_response.get("articles", [])[:3]
        news_data = "\n".join([f"News: {article['title']} ({article['publishedAt']})" for article in articles])
    except Exception as e:
        news_data = f"Failed to fetch gold news: {str(e)}"

    query_embedding = embedder.encode([question])[0]
    retrieved_docs = search(collection, query_embedding, TOP_K)

    context = "\n".join(retrieved_docs + [gold_data, news_data])
    prompt = f"""You are a gold trading assistant. Use the following context to answer the question, focusing on gold trading concepts, price data, and news. Provide a step-by-step reasoning process labeled 'Thinking,' analyzing the context (price, news, indicators). If the question is broad (e.g., 'which gold to buy'), suggest up to three options (e.g., bullion, ETFs, futures). Clearly label the final recommendation as 'Answer'.

Context:
{context}

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

        return jsonify({
            "answer": answer,
            "thinking": thinking,
            "context": retrieved_docs + [gold_data, news_data]
        })
    except Exception as e:
        return jsonify({"error": "Failed to get valid response", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)