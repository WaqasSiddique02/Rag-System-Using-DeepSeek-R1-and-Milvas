import requests

def test_trading_queries():
    queries = [
        "Should I buy Bitcoin now?",
        "Explain MACD indicator",
        "What's the best strategy for day trading stocks?",
        "How do I read order book data?"
    ]
    
    for query in queries:
        response = requests.post(
            "http://localhost:5000/query",
            json={"question": query}
        )
        print(f"Question: {query}")
        print("Response:", response.json())
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    test_trading_queries()