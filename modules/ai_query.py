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
