import numpy as np

def generate_institutional_signal(current_price, predicted_price, sentiment_score, imbalance, volatility):
    """
    Fuses all engine data to produce a precise Entry, Stop Loss, and Take Profit.
    """
    # 1. Calculate the 'Delta' (Expected Move)
    expected_change = ((predicted_price / current_price) - 1) * 100
    
    # 2. Define Institutional Signal Thresholds
    verdict = "NEUTRAL"
    if expected_change > 0.5 and sentiment_score > 0.1 and imbalance > 0.15:
        verdict = "BUY"
    elif expected_change < -0.5 and sentiment_score < -0.1 and imbalance < -0.15:
        verdict = "SELL"
    
    # Return 0/None safely if Neutral
    if verdict == "NEUTRAL":
        return {
            "verdict": "NEUTRAL",
            "entry": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "confidence": 0.0
        }

    # 3. Dynamic Stop Loss (Based on Volatility)
    sl_pct = max(volatility * 1.5, 0.01) 
    
    # 4. Dynamic Take Profit (1:2.5 Risk/Reward)
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
