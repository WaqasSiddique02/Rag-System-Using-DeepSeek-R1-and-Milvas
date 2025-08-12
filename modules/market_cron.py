import os
from datetime import datetime, timedelta
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from milvus_client import connect_to_milvus, get_or_create_collection, insert_documents, check_existing_documents
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

SPOT_BASE = "https://api.binance.com"
FUTURES_BASE = "https://fapi.binance.com"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]

def get_json(url, params=None):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def fetch_binance_data(symbol):
    """
    Fetch Spot + Futures data for a given symbol.
    Returns a list of text entries describing the state of the market.
    """
    entries = []
    timestamp = datetime.utcnow().isoformat()

    # Spot latest price & 24h stats
    stats = get_json(f"{SPOT_BASE}/api/v3/ticker/24hr", {"symbol": symbol})
    if "error" not in stats:
        entries.append(f"[{timestamp}] {symbol} Spot Price: ${stats['lastPrice']} "
                       f"(24h Change: {stats['priceChangePercent']}%, High: {stats['highPrice']}, Low: {stats['lowPrice']}, Volume: {stats['volume']})")

    # Order book depth
    depth = get_json(f"{SPOT_BASE}/api/v3/depth", {"symbol": symbol, "limit": 10})
    if "error" not in depth:
        best_bid = depth['bids'][0][0] if depth['bids'] else 'N/A'
        best_ask = depth['asks'][0][0] if depth['asks'] else 'N/A'
        entries.append(f"[{timestamp}] {symbol} Order Book: Best Bid ${best_bid}, Best Ask ${best_ask}")

    # Recent trades
    trades = get_json(f"{SPOT_BASE}/api/v3/trades", {"symbol": symbol, "limit": 5})
    if "error" not in trades:
        total_qty = sum(float(t["qty"]) for t in trades)
        entries.append(f"[{timestamp}] {symbol} Recent Trades: {len(trades)} trades, total qty {total_qty}")

    # Historical candlesticks (10 min window to detect moves)
    now = datetime.utcnow()
    klines = get_json(f"{SPOT_BASE}/api/v3/klines", {"symbol": symbol, "interval": "1m", "limit": 10})
    if "error" not in klines:
        start_price = float(klines[0][1])
        end_price = float(klines[-1][4])
        change_pct = ((end_price - start_price) / start_price) * 100
        entries.append(f"[{timestamp}] {symbol} Last 10 min: Start ${start_price}, End ${end_price}, Change {change_pct:.2f}%")

    # Futures open interest
    oi = get_json(f"{FUTURES_BASE}/fapi/v1/openInterest", {"symbol": symbol})
    if "error" not in oi:
        entries.append(f"[{timestamp}] {symbol} Futures Open Interest: {oi['openInterest']} contracts")

    # Futures liquidation orders (last few mins)
    liquidations = get_json(f"{FUTURES_BASE}/fapi/v1/allForceOrders", {"symbol": symbol, "limit": 5})
    if "error" not in liquidations:
        entries.append(f"[{timestamp}] {symbol} Recent Liquidations: {len(liquidations)} orders")

    # Futures long/short ratio (top traders)
    long_short = get_json(f"{FUTURES_BASE}/futures/data/topLongShortAccountRatio", {"symbol": symbol, "period": "5m", "limit": 1})
    if isinstance(long_short, list) and long_short:
        ratio = long_short[0]["longAccount"] + ":" + long_short[0]["shortAccount"]
        entries.append(f"[{timestamp}] {symbol} Long/Short Account Ratio (5m): {ratio}")

    return entries

def market_analysis_job():
    try:
        print(f"[{datetime.utcnow().isoformat()}] Starting market data job...")
        connect_to_milvus()

        dim = embedder.get_sentence_embedding_dimension()
        collection = get_or_create_collection(dim)

        all_entries = []
        for symbol in SYMBOLS:
            all_entries.extend(fetch_binance_data(symbol))

        # Remove already stored entries
        new_entries = check_existing_documents(collection, all_entries)

        if new_entries:
            embeddings = embedder.encode(new_entries)
            insert_documents(collection, new_entries, embeddings)
            print(f"Inserted {len(new_entries)} new entries.")
        else:
            print("No new entries to insert.")

        print(f"[{datetime.utcnow().isoformat()}] Market data job complete.")

    except Exception as e:
        print(f"Job failed: {str(e)}")


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(market_analysis_job, 'interval', minutes=15, id='market_analysis')
    scheduler.start()
    print("Market analysis scheduler started (runs every 15 minutes).")

if __name__ == "__main__":
    start_scheduler()
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
