# import requests
# import os
# from dotenv import load_dotenv
# from datetime import datetime
# from milvus_client import check_existing_documents, insert_documents
# from sentence_transformers import SentenceTransformer

# load_dotenv()
# embedder = SentenceTransformer("all-MiniLM-L6-v2")  # same as app.py


# def fetch_market_data_and_log(collection):
#     """
#     Fetch crypto and stock market data, save them into Milvus with timestamps,
#     and return the entries for immediate use.
#     """
#     data_entries = []
#     timestamp = datetime.utcnow().isoformat()

#     # Crypto data from CoinGecko
#     try:
#         response = requests.get(
#             "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc"
#         )
#         response.raise_for_status()
#         cryptos = response.json()[:3]  # Top 3 cryptos
#         for crypto in cryptos:
#             data_entries.append(
#                 f"[{timestamp}] {crypto['symbol'].upper()}: ${crypto['current_price']} "
#                 f"(24h: {crypto['price_change_percentage_24h']}%)"
#             )
#     except Exception as e:
#         data_entries.append(f"[{timestamp}] Crypto data unavailable: {str(e)}")

#     # Stock data from Alpha Vantage
#     try:
#         api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
#         stocks = ["MSFT", "AAPL", "TSLA"]  # Example stocks
#         for stock in stocks:
#             response = requests.get(
#                 f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock}&apikey={api_key}"
#             )
#             response.raise_for_status()
#             quote = response.json().get("Global Quote", {})
#             data_entries.append(
#                 f"[{timestamp}] {stock}: ${quote.get('05. price', 'N/A')} "
#                 f"(Change: {quote.get('10. change percent', 'N/A')})"
#             )
#     except Exception as e:
#         data_entries.append(f"[{timestamp}] Stock data unavailable: {str(e)}")

#     # Save into Milvus
#     try:
#         new_entries = check_existing_documents(collection, data_entries)
#         if new_entries:
#             embeddings = embedder.encode(new_entries)
#             insert_documents(collection, new_entries, embeddings)
#             print(f"Logged {len(new_entries)} new market data entries to Milvus.")
#         else:
#             print("No new market data entries to log.")
#     except Exception as e:
#         print(f"Failed to log market data to Milvus: {str(e)}")

#     return "\n".join(data_entries)
