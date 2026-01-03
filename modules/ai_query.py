import openai

openai.api_key = "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"  # From config.py or environment

def process_query(query, data_context):
    """
    Process a natural language query using AI and app data.
    - query: User input string.
    - data_context: Dict of app data (e.g., {'volatility': df_vol, 'on_chain': metrics}).
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a crypto analyst. Use the provided data to answer."},
                {"role": "user", "content": f"Data: {str(data_context)}\nQuery: {query}"}
            ]
        )
        return response.choices[0].message['content']
    except:
        return "AI query unavailable."
