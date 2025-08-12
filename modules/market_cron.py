import os
from datetime import datetime, timedelta
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from milvus_client import (
    connect_to_milvus,
    get_or_create_collection,
    insert_documents,
    check_existing_documents,
)
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """Fetch Spot + Futures data for a given symbol in parallel."""
    timestamp = datetime.utcnow().isoformat()
    urls = {
        "stats": (f"{SPOT_BASE}/api/v3/ticker/24hr", {"symbol": symbol}),
        "depth": (f"{SPOT_BASE}/api/v3/depth", {"symbol": symbol, "limit": 10}),
        "trades": (f"{SPOT_BASE}/api/v3/trades", {"symbol": symbol, "limit": 5}),
        "klines": (
            f"{SPOT_BASE}/api/v3/klines",
            {"symbol": symbol, "interval": "1m", "limit": 10},
        ),
        "oi": (f"{FUTURES_BASE}/fapi/v1/openInterest", {"symbol": symbol}),
        "liquidations": (
            f"{FUTURES_BASE}/fapi/v1/allForceOrders",
            {"symbol": symbol, "limit": 5},
        ),
        "long_short": (
            f"{FUTURES_BASE}/futures/data/topLongShortAccountRatio",
            {"symbol": symbol, "period": "5m", "limit": 1},
        ),
    }

    entries = []

    # Run all Binance requests for this symbol in parallel
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        futures = {
            executor.submit(get_json, url, params): name
            for name, (url, params) in urls.items()
        }
        results = {}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()

    # Process responses
    stats = results.get("stats", {})
    if "error" not in stats:
        entries.append(
            f"[{timestamp}] {symbol} Spot Price: ${stats['lastPrice']} "
            f"(24h Change: {stats['priceChangePercent']}%, High: {stats['highPrice']}, Low: {stats['lowPrice']}, Volume: {stats['volume']})"
        )

    depth = results.get("depth", {})
    if "error" not in depth:
        best_bid = depth["bids"][0][0] if depth["bids"] else "N/A"
        best_ask = depth["asks"][0][0] if depth["asks"] else "N/A"
        entries.append(
            f"[{timestamp}] {symbol} Order Book: Best Bid ${best_bid}, Best Ask ${best_ask}"
        )

    trades = results.get("trades", {})
    if "error" not in trades:
        total_qty = sum(float(t["qty"]) for t in trades)
        entries.append(
            f"[{timestamp}] {symbol} Recent Trades: {len(trades)} trades, total qty {total_qty}"
        )

    klines = results.get("klines", {})
    if "error" not in klines and len(klines) >= 2:
        start_price = float(klines[0][1])
        end_price = float(klines[-1][4])
        change_pct = ((end_price - start_price) / start_price) * 100
        entries.append(
            f"[{timestamp}] {symbol} Last 10 min: Start ${start_price}, End ${end_price}, Change {change_pct:.2f}%"
        )

    oi = results.get("oi", {})
    if "error" not in oi:
        entries.append(
            f"[{timestamp}] {symbol} Futures Open Interest: {oi['openInterest']} contracts"
        )

    liquidations = results.get("liquidations", {})
    if "error" not in liquidations:
        entries.append(
            f"[{timestamp}] {symbol} Recent Liquidations: {len(liquidations)} orders"
        )

    long_short = results.get("long_short", [])
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

        # Check for duplicates
        new_entries_with_hashes = check_existing_documents(collection, all_entries)
        if new_entries_with_hashes:
            new_entries, _ = zip(*new_entries_with_hashes) if new_entries_with_hashes else ([], [])
            embeddings = embedder.encode(new_entries)
            insert_documents(collection, new_entries_with_hashes, embeddings)
            print(f"Inserted {len(new_entries)} new entries.")
        else:
            print("No new entries to insert.")

        print(f"[{datetime.utcnow().isoformat()}] Market data job complete.")
    except Exception as e:
        print(f"Job failed: {str(e)}")
        


def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(market_analysis_job, "interval", minutes=15, id="market_analysis")
    scheduler.start()
    print("Market analysis scheduler started (runs every 15 minutes).")


if __name__ == "__main__":
    start_scheduler()
    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")
