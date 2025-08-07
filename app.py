from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import requests
import numpy as np
from milvus_client import connect_to_milvus, get_or_create_collection, insert_documents, search

# Flask setup
app = Flask(__name__)

# Config
OLLAMA_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"
TOP_K = 3

# Connect to Milvus
connect_to_milvus()

# Load docs
documents = [
    "DeepSeek is a powerful code and text language model.",
    "Retrieval-Augmented Generation (RAG) combines document retrieval with language generation.",
    "Milvus is a scalable vector database designed for similarity search.",
    "Ollama allows running large language models locally via an API.",
    "Sentence Transformers convert text into embeddings for semantic search."
]

# Embedding model
print("Loading SentenceTransformer...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents).tolist()

# Milvus collection
dim = len(doc_embeddings[0])
collection = get_or_create_collection(dim)

# Insert documents into Milvus
insert_documents(collection, documents, doc_embeddings)

# Query route
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    query_embedding = embedder.encode([question])[0]
    retrieved_docs = search(collection, query_embedding, TOP_K)
    context = "\n".join(retrieved_docs)
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        result = response.json()
        return jsonify({
            "answer": result.get("response", ""),
            "context": retrieved_docs
        })
    except Exception as e:
        return jsonify({"error": "Failed to get valid response", "details": str(e)}), 500

# Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
