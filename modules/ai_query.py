import streamlit as st
import google.generativeai as genai
from openai import OpenAI

def heuristic_fallback(context):
    """
    Uses hard-coded quant rules to provide a bias when APIs fail.
    """
    rsi = context.get('rsi', 50)
    # Default to neutral if bb_status is missing
    bb_status = context.get('bb_status', "Neutral")
    pair = context.get('pair', "Asset")

    if rsi < 35 and "Lower" in str(bb_status):
        bias = "STRONGLY BULLISH"
        reason = "Oversold conditions met with volatility band contact."
    elif rsi > 65 and "Upper" in str(bb_status):
        bias = "STRONGLY BEARISH"
        reason = "Overbought conditions met with upper band contact."
    else:
        bias = "NEUTRAL"
        reason = f"RSI at {rsi:.1f} is within balanced territory."

    return f"### 🛡️ Heuristic Report\n**Bias:** {bias}\n**Analysis:** {reason}"

def process_query(query: str, context: dict):
    # 1. Try OpenAI
    try:
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"Context: {context}"}, {"role": "user", "content": query}],
            timeout=10
        )
        return res.choices[0].message.content
    except Exception:
        pass

    # 2. Try Gemini
    try:
        genai.configure(api_key=st.secrets.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(f"Context: {context}\nQuery: {query}")
        return res.text
    except Exception:
        pass

    # 3. Final Fallback
    return heuristic_fallback(context)
