import requests
import streamlit as st

class WhaleClient:
    def __init__(self):
        self.api_key = st.secrets.get("WHALE_ALERT_API_KEY")
        self.base_url = "https://api.whale-alert.io/v1"

    def get_recent_whales(self, min_value=500000):
        if not self.api_key:
            return []
        
        url = f"{self.base_url}/transactions?api_key={self.api_key}&min_value={min_value}"
        try:
            response = requests.get(url, timeout=10).json()
            transactions = response.get('transactions', [])
            return self.process_transactions(transactions)
        except Exception:
            return []

    def process_transactions(self, txs):
        processed = []
        for tx in txs:
            # Logic: Transfers TO exchanges are often bearish (selling)
            # Transfers FROM exchanges are often bullish (accumulation)
            impact = "Neutral"
            if tx['to']['owner_type'] == 'exchange': impact = "Bearish (Inflow)"
            elif tx['from']['owner_type'] == 'exchange': impact = "Bullish (Outflow)"
            
            processed.append({
                "time": tx['timestamp'],
                "amount": f"{tx['amount']:,.0f} {tx['symbol'].upper()}",
                "value_usd": tx['amount_usd'],
                "impact": impact,
                "blockchain": tx['blockchain']
            })
        return processed
