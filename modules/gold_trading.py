# File: gold_trading_chatbot/modules/gold_trading.py
from sentence_transformers import SentenceTransformer
import requests
import numpy as np
from milvus_client import check_existing_documents, insert_documents


def initialize_gold_documents(collection, embedder):
    """
    Initialize gold trading documents and insert only if not already in Milvus.
    Returns the collection for further use.
    """
    documents = [
        "Gold trading involves speculating on the price of gold, often through spot markets, futures, or CFDs. The spot price (XAU) is the current price for immediate delivery, quoted per troy ounce (31.103 grams).",
        "The London Fix is a twice-daily meeting of five LBMA firms to set the global benchmark gold price, influencing worldwide trading.",
        "Bullion refers to gold bars or ingots at least 99.5% pure, traded as good delivery bars (400 oz) in the LBMA system with a traceable chain of integrity.",
        "Bullion coins (e.g., Canadian Maple Leaf, American Buffalo) are valued by their gold content, with a premium of 5-15% over the spot price.",
        "Numismatic coins are valued for rarity or historical significance, not just gold content, unlike bullion coins.",
        "Arbitrage is buying and selling gold in different markets to profit from price differences (e.g., USD vs. GBP).",
        "An assay tests gold’s purity and weight, using methods like Fire Assay or X-Ray Fluorescence (XRF).",
        "Gold ETFs (e.g., SPDR Gold Shares, GLD) track gold prices without physical ownership, but carry risks like management fees.",
        "XAU/USD is the forex pair for one troy ounce of gold in US dollars, used in spot trading.",
        "Gold prices rise during inflation or geopolitical uncertainty (safe haven) and fall with high interest rates or a strong USD.",
        "COMEX is a major New York-based exchange for gold futures trading, alongside the London OTC market and Shanghai Gold Exchange.",
        "A troy ounce (31.103 grams) is used for gold, distinct from an avoirdupois ounce (28.35 grams).",
        "The chain of integrity ensures LBMA bullion bars are trusted without re-assaying, maintaining a traceable custody chain.",
        "Futures are contracts to buy/sell gold at a future date, while CFDs allow speculation without physical delivery.",
        "The bid price is what a dealer pays for gold, and the ask price is what they sell it for; the spread is the difference.",
    ]

    # Check for existing documents
    new_documents = check_existing_documents(collection, documents)
    if new_documents:
        new_embeddings = embedder.encode(new_documents).tolist()
        insert_documents(collection, new_documents, new_embeddings)
    else:
        print("No new gold trading documents to insert; all documents already exist")

    return documents


def fetch_gold_data(alpha_vantage_key, newsapi_key):
    """
    Fetch real-time gold price and news.
    Returns gold_data and news_data as strings for context.
    """
    # Fetch gold price (XAU/USD)
    try:
        api_url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=XAU&to_currency=USD&apikey={alpha_vantage_key}"
        api_response = requests.get(api_url).json()
        quote = api_response.get("Realtime Currency Exchange Rate", {})
        gold_data = (
            f"Latest gold price (XAU/USD): ${quote.get('5. Exchange Rate', 'N/A')} "
            f"as of {quote.get('6. Last Refreshed', 'N/A')} "
            f"(Bid: {quote.get('8. Bid Price', 'N/A')}, Ask: {quote.get('9. Ask Price', 'N/A')})"
        )
    except Exception as e:
        gold_data = f"Failed to fetch gold price: {str(e)}"

    # Fetch gold-related news
    try:
        news_url = f"https://newsapi.org/v2/everything?q=gold+market&sortBy=publishedAt&apiKey={newsapi_key}"
        news_response = requests.get(news_url).json()
        articles = news_response.get("articles", [])[:3]
        news_data = "\n".join(
            [
                f"News: {article['title']} ({article['publishedAt']})"
                for article in articles
            ]
        )
    except Exception as e:
        news_data = f"Failed to fetch gold news: {str(e)}"

    return gold_data, news_data


def get_gold_prompt(question, retrieved_docs, gold_data, news_data):
    """
    Generate the prompt for gold trading queries.
    """
    context = "\n".join(retrieved_docs + [gold_data, news_data])
    return f"""You are a gold trading assistant. Use the following context to answer the question, focusing on gold trading concepts, price data, and news. Provide a step-by-step reasoning process labeled 'Thinking,' analyzing the context (price, news, indicators). If the question is broad (e.g., 'which gold to buy'), suggest up to three options (e.g., bullion, ETFs, futures). Clearly label the final recommendation as 'Answer'.

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

def get_trading_prompt(question, retrieved_docs, market_data):
    """
    Generate prompt for crypto/stock trading queries
    """
    context = "\n".join(retrieved_docs + [market_data])
    
    return f"""You are a professional trading assistant specializing in cryptocurrencies and stocks. 
Analyze the following information and provide a detailed response with step-by-step reasoning.
Consider technical indicators, market trends, and risk factors.

Context:
{context}

Question: {question}

Required Response Format:
<Thinking>
[Your detailed analysis process]
[Consideration of relevant indicators]
[Evaluation of potential risks]
</Thinking>
<Answer>
[Clear, actionable conclusion]
[Optional: Suggested next steps or additional analysis needed]
</Answer>"""