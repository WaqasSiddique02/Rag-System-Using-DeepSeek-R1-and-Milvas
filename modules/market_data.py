import requests
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_market_data():
    """Fetch crypto and stock market data from various APIs"""
    data = []
    
    # Crypto data from CoinGecko
    try:
        response = requests.get("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc")
        cryptos = response.json()[:3]  # Top 3 cryptos
        for crypto in cryptos:
            data.append(
                f"{crypto['symbol'].upper()}: ${crypto['current_price']} "
                f"(24h: {crypto['price_change_percentage_24h']}%)"
            )
    except Exception as e:
        data.append(f"Crypto data unavailable: {str(e)}")
    
    # Stock data from Alpha Vantage
    try:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        stocks = ["MSFT", "AAPL", "TSLA"]  # Example stocks
        for stock in stocks:
            response = requests.get(
                f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock}&apikey={api_key}"
            )
            quote = response.json().get("Global Quote", {})
            data.append(
                f"{stock}: ${quote.get('05. price', 'N/A')} "
                f"(Change: {quote.get('10. change percent', 'N/A')})"
            )
    except Exception as e:
        data.append(f"Stock data unavailable: {str(e)}")
    
    return "\n".join(data)