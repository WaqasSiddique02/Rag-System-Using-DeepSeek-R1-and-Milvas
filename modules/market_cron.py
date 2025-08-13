import os
from datetime import datetime
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

# Shared embedder instance
print("[market_cron] Loading SentenceTransformer model...")
shared_embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        futures = {
            executor.submit(get_json, url, params): name
            for name, (url, params) in urls.items()
        }
        results = {
            name: future.result()
            for future, name in ((f, futures[f]) for f in as_completed(futures))
        }

    # Spot stats
    stats = results.get("stats", {})
    if "error" not in stats:
        entries.append(
            f"[{timestamp}] {symbol} Spot Price: ${stats['lastPrice']} "
            f"(24h Change: {stats['priceChangePercent']}%, High: {stats['highPrice']}, Low: {stats['lowPrice']}, Volume: {stats['volume']})"
        )

    # Depth
    depth = results.get("depth", {})
    if "error" not in depth:
        best_bid = depth["bids"][0][0] if depth.get("bids") else "N/A"
        best_ask = depth["asks"][0][0] if depth.get("asks") else "N/A"
        entries.append(
            f"[{timestamp}] {symbol} Order Book: Best Bid ${best_bid}, Best Ask ${best_ask}"
        )

    # Trades
    trades = results.get("trades", {})
    if "error" not in trades:
        total_qty = sum(float(t["qty"]) for t in trades)
        entries.append(
            f"[{timestamp}] {symbol} Recent Trades: {len(trades)} trades, total qty {total_qty}"
        )

    # Klines
    klines = results.get("klines", {})
    if "error" not in klines and len(klines) >= 2:
        start_price = float(klines[0][1])
        end_price = float(klines[-1][4])
        change_pct = ((end_price - start_price) / start_price) * 100
        entries.append(
            f"[{timestamp}] {symbol} Last 10 min: Start ${start_price}, End ${end_price}, Change {change_pct:.2f}%"
        )

    # Open Interest
    oi = results.get("oi", {})
    if "error" not in oi:
        entries.append(
            f"[{timestamp}] {symbol} Futures Open Interest: {oi['openInterest']} contracts"
        )

    # Liquidations
    liquidations = results.get("liquidations", {})
    if "error" not in liquidations:
        entries.append(
            f"[{timestamp}] {symbol} Recent Liquidations: {len(liquidations)} orders"
        )

    # Long/Short Ratio
    long_short = results.get("long_short", [])
    if isinstance(long_short, list) and long_short:
        ratio = f"{long_short[0]['longAccount']}:{long_short[0]['shortAccount']}"
        entries.append(f"[{timestamp}] {symbol} Long/Short Account Ratio (5m): {ratio}")

    return entries


def market_analysis_job():
    try:
        print(f"[{datetime.utcnow().isoformat()}] Starting market data job...")
        connect_to_milvus()

        dim = shared_embedder.get_sentence_embedding_dimension()
        collection = get_or_create_collection(dim)

        all_entries = []
        for symbol in SYMBOLS:
            all_entries.extend(fetch_binance_data(symbol))

        # Check for duplicates in Milvus
        new_entries = check_existing_documents(collection, all_entries)
        if new_entries:
            embeddings = shared_embedder.encode(new_entries)
            insert_documents(collection, new_entries, embeddings)
            print(f"[market_cron] Inserted {len(new_entries)} new entries.")
        else:
            print("[market_cron] No new entries to insert.")

        print(f"[{datetime.utcnow().isoformat()}] Market data job complete.")
    except Exception as e:
        print(f"[market_cron] Job failed: {str(e)}")


# Prevent multiple scheduler starts in debug mode
_scheduler = BackgroundScheduler()


def start_scheduler():
    if not _scheduler.get_jobs():
        _scheduler.add_job(
            market_analysis_job,
            "interval",
            minutes=15,
            id="market_analysis",
            replace_existing=True,
        )
        _scheduler.start()
        print("[market_cron] Scheduler started (runs every 15 minutes).")
