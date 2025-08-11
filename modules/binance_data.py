# modules/binance_data.py
import os
from datetime import datetime
from dotenv import load_dotenv
from binance.client import Client
from sentence_transformers import SentenceTransformer
from milvus_client import check_existing_documents, insert_documents

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

client = Client(api_key, api_secret)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def fetch_binance_data_and_log(collection, symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"]):
    """
    Fetch market prices from Binance for given symbols and store in Milvus.
    """
    timestamp = datetime.utcnow().isoformat()
    data_entries = []

    try:
        for symbol in symbols:
            ticker = client.get_ticker(symbol=symbol)
            price = ticker["lastPrice"]
            change = ticker["priceChangePercent"]
            data_entries.append(f"[{timestamp}] {symbol}: ${price} (24h: {change}%)")
    except Exception as e:
        data_entries.append(f"[{timestamp}] Binance data unavailable: {str(e)}")

    # Save to Milvus
    try:
        new_entries = check_existing_documents(collection, data_entries)
        if new_entries:
            embeddings = embedder.encode(new_entries)
            insert_documents(collection, new_entries, embeddings)
            print(f"Logged {len(new_entries)} new Binance market entries to Milvus.")
        else:
            print("No new Binance data entries to log.")
    except Exception as e:
        print(f"Failed to log Binance data to Milvus: {str(e)}")

    return "\n".join(data_entries)
