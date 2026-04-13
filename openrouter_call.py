import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Simple test call with qwen3.6-plus:free
response = client.chat.completions.create(
    model="qwen/qwen3.6-plus:free",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one sentence."
        }
    ],
)

print("Model:", response.model)
print("Response:", response.choices[0].message.content)
