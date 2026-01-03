import openai
import google.generativeai as genai
import os

# API Keys (replace with real ones in production; use st.secrets or env vars)
OPENAI_API_KEY = "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"  # Dummy from you
GEMINI_API_KEY = "AIzaSyDkpALD1EIsWi7aGSk1X8aQDvfBWbjddMY"  # Get free from https://ai.google.dev/gemini-api

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

def process_query(query: str, data_context: dict = None):
    """
    Process a natural language query.
    - Primary: OpenAI (gpt-3.5-turbo)
    - Fallback: Google Gemini (gemini-1.5-flash)
    """
    context_str = str(data_context) if data_context else "General crypto market data available."
    prompt = f"You are a crypto analyst. Use the provided data to answer.\nData: {context_str}\nQuery: {query}"

    # Primary: OpenAI
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful crypto analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except Exception as e_openai:
        print(f"OpenAI failed: {e_openai}")

        # Fallback: Gemini
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e_gemini:
            return f"Both AI providers failed.\nOpenAI error: {str(e_openai)}\nGemini error: {str(e_gemini)}"
