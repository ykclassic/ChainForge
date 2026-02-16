import requests
import streamlit as st

class WhaleClient:
    def __init__(self):
        # Using .get to avoid KeyErrors if secret is missing
        self.api_key = st.secrets.get("WHALE_ALERT_API_KEY")
        self.base_url = "https://api.whale-alert.io/v1"

    def get_recent_whales(self, min_value=500000):
        if not self.api_key:
            return []
        
        url = f"{self.base_url}/transactions?api_key={self.api_key}&min_value={min_value}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                transactions = response.json().get('transactions', [])
                return self.process_transactions(transactions)
            return []
        except Exception:
            return []

    def process_transactions(self, txs):
        processed = []
        for tx in txs:
            # Logic: Transfers TO exchanges = Selling pressure; FROM = Accumulation
            to_owner = tx.get('to', {}).get('owner_type', 'unknown')
            from_owner = tx.get('from', {}).get('owner_type', 'unknown')
            
            impact = "Neutral"
            if to_owner == 'exchange': 
                impact = "Bearish (Inflow)"
            elif from_owner == 'exchange': 
                impact = "Bullish (Outflow)"
            
            processed.append({
                "time": tx.get('timestamp'),
                "amount": f"{tx.get('amount', 0):,.0f} {tx.get('symbol', '').upper()}",
                "value_usd": tx.get('amount_usd', 0),
                "impact": impact,
                "blockchain": tx.get('blockchain', 'N/A')
            })
        return processed
