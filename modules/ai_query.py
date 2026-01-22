import streamlit as st
import google.generativeai as genai
from openai import OpenAI

def heuristic_fallback(context):
    """
    The 'Emergency Brain'. No API needed. 
    Uses hard-coded quant rules to provide a bias.
    """
    rsi = context.get('rsi', 50)
    bb_status = context.get('bb_status', "Neutral")
    sentiment = context.get('sentiment', 0)
    pair = context.get('pair', "Asset")

    # Logic Engine
    if rsi < 35 and bb_status == "Touching Lower":
        bias = "STRONGLY BULLISH"
        reason = "Asset is deeply oversold and hugging the lower volatility band. High probability of mean reversion."
    elif rsi > 65 and bb_status == "Touching Upper":
        bias = "STRONGLY BEARISH"
        reason = "Asset is overextended and touching the upper volatility limit. High risk of a pullback."
    else:
        bias = "NEUTRAL / CAUTIOUS"
        reason = f"RSI is at {rsi:.1f} (Balanced). No extreme volatility squeeze detected."

    return f"### 🛡️ Heuristic Report (Non-AI Fallback)\n**Bias:** {bias}\n**Analysis:** {reason}\n**Note:** This analysis was generated locally because AI services were unavailable."

def process_query(query: str, context: dict):
    # 1. Try OpenAI Primary
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"Context: {context}"}, {"role": "user", "content": query}],
            timeout=10
        )
        return res.choices[0].message.content
    except Exception as e:
        st.warning(f"OpenAI Failed. Attempting Gemini...")

    # 2. Try Gemini Secondary
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(f"Context: {context}\nQuery: {query}")
        return res.text
    except Exception as e:
        st.error(f"Gemini Failed. Engaging Heuristic Safety Engine...")

    # 3. Final Hard Fallback
    return heuristic_fallback(context)
