import streamlit as st
import google.generativeai as genai
from openai import OpenAI

# Load keys securely
try:
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except KeyError:
    openai_client = None

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # Update if needed; 2.5-flash may exist in 2026
except KeyError:
    gemini_model = None

def process_query(query: str, context: dict):
    if not openai_client and not gemini_model:
        return "Error: No API keys configured for OpenAI or Gemini."

    # Format context for prompt
    vol_data = context.get('volatility_data', [])
    pairs = context.get('pairs', [])
    vol_str = "\n".join([f"- {item['Pair']}: {item['Volatility %']}%" for item in vol_data if item['Volatility %'] != 'N/A'])

    system_prompt = f"""
You are a crypto analyst for ChainForge Analytics.
Use this data:
Available pairs: {', '.join(pairs)}
30d Annualized Volatility:
{vol_str}

Answer concisely and insightfully. Use markdown/tables if helpful.
"""

    full_prompt = f"{system_prompt}\nUser query: {query}"

    # Try OpenAI first
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Or gpt-4o for better reasoning
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": query}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"OpenAI failed ({str(e)}), falling back to Gemini...")

    # Fallback to Gemini
    if gemini_model:
        try:
            response = gemini_model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            return f"Gemini failed: {str(e)}"

    return "Both AI providers unavailable."
