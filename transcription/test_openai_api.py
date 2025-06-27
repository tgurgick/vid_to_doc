import openai
import os
from dotenv import load_dotenv

def test_openai_api_connection():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("OPENAI_API_KEY not set. Please set it in your .env file.")
        return
    openai.api_key = api_key
    try:
        # Lightweight model list check
        response = openai.models.list()
        if hasattr(response, 'data') or isinstance(response, list):
            print("OpenAI API connection successful (model list).")
        else:
            print("OpenAI API connection failed: Unexpected response.")
        # Robust: send a simple chat completion prompt
        chat_response = openai.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ],
            max_tokens=10
        )
        print("\nFull chat completion response:")
        print(chat_response)
        reply = chat_response.choices[0].message.content.strip()
        print(f"\nChat completion test successful. Model replied: {reply}")
    except Exception as e:
        print(f"OpenAI API connection failed: {e}")

if __name__ == "__main__":
    test_openai_api_connection()
