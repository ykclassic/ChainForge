# modules/ai_query.py

import os
import openai
import google.generativeai as genai

# === API KEYS ===
# Use Streamlit secrets in production (st.secrets["OPENAI_API_KEY"])
# For now, hardcode dummy OpenAI key from user
OPENAI_API_KEY = "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"

# Gemini key — get a free one from https://aistudio.google.com/app/apikey
GEMINI_API_KEY = "AIzaSyDkpALD1EIsWi7aGSk1X8aQDvfBWbjddMY"  # Replace with valid key

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

def process_query(query: str, data_context: dict = None):
    """
    Process natural language query with OpenAI primary + Gemini fallback.
    Updated for OpenAI v1.0+ and Gemini correct model.
    """
    context_str = str(data_context) if data_context else "General crypto market data."
    prompt = f"You are a professional crypto analyst. Use the provided data to answer concisely.\nData: {context_str}\nUser query: {query}"

    # === PRIMARY: OpenAI (v1.0+ syntax) ===
    try:
        client = openai.OpenAI()  # New client syntax
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful crypto analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e_openai:
        print(f"OpenAI failed: {e_openai}")

        # === FALLBACK: Google Gemini (correct model name) ===
        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')  # Valid model
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e_gemini:
            return f"Both AI providers failed.\nOpenAI: {str(e_openai)}\nGemini: {str(e_gemini)}\nCheck API keys and models."

    return "AI query completed."
