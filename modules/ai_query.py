import openai
import google.generativeai as genai
import streamlit as st

def process_query(query: str, data_context: dict = None):
    """
    Secure AI query with OpenAI primary + Gemini fallback.
    Keys from st.secrets (no hardcoding).
    """
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        gemini_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        return "API keys missing. Add OPENAI_API_KEY and GEMINI_API_KEY to Streamlit secrets."

    context_str = str(data_context) if data_context else "General crypto market data."
    prompt = f"You are a professional crypto analyst. Answer concisely using data.\nData: {context_str}\nQuery: {query}"

    # Primary: OpenAI
    try:
        client = openai.OpenAI(api_key=openai_key)
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
        st.warning("OpenAI failed — trying Gemini fallback...")

        # Fallback: Gemini (updated model)
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.5-flash')  # Valid Jan 2026 model
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e_gemini:
            return f"Both providers failed.\nOpenAI: {str(e_openai)}\nGemini: {str(e_gemini)}\nAdd valid keys to secrets."
