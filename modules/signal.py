import numpy as np

def generate_institutional_signal(current_price, predicted_price, sentiment_score, imbalance, volatility):
    """
    Fuses all engine data to produce a precise Entry, Stop Loss, and Take Profit.
    
    Args:
        current_price (float): The actual market price right now.
        predicted_price (float): The target from the LSTM Fusion engine.
        sentiment_score (float): VADER sentiment (-1 to 1).
        imbalance (float): Order book pressure (-1 to 1).
        volatility (float): Standard deviation of recent returns (from Risk Lab).
    """
    
    # 1. Calculate the 'Delta' (Expected Move)
    expected_change = ((predicted_price / current_price) - 1) * 100
    
    # 2. Define Institutional Signal Thresholds
    # Signal requires agreement between Prediction, Sentiment, and Order Book.
    verdict = "NEUTRAL"
    if expected_change > 0.5 and sentiment_score > 0.1 and imbalance > 0.15:
        verdict = "BUY"
    elif expected_change < -0.5 and sentiment_score < -0.1 and imbalance < -0.15:
        verdict = "SELL"
    
    if verdict == "NEUTRAL":
        return {
            "verdict": "NEUTRAL",
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "confidence": 0
        }

    # 3. Dynamic Stop Loss (Based on Volatility)
    # Institutional standard: SL is set at 1.5x the current market volatility (ATR equivalent)
    sl_pct = max(volatility * 1.5, 0.01) # Minimum 1% stop to avoid noise
    
    # 4. Dynamic Take Profit (Based on Risk/Reward Ratio)
    # We target a 1:2.5 Risk-to-Reward ratio as a baseline.
    tp_pct = sl_pct * 2.5 

    if verdict == "BUY":
        entry = current_price
        stop_loss = entry * (1 - sl_pct)
        take_profit = entry * (1 + tp_pct)
    else: # SELL
        entry = current_price
        stop_loss = entry * (1 + sl_pct)
        take_profit = entry * (1 - tp_pct)

    # 5. Confidence Score (0-100%)
    # Based on the strength of the three supporting factors
    confidence = (abs(expected_change) * 10 + abs(sentiment_score) * 20 + abs(imbalance) * 20)
    confidence = min(round(confidence, 1), 100.0)

    return {
        "verdict": verdict,
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "confidence": confidence,
        "rr_ratio": "1:2.5"
    }
