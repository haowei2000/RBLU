import os
from openai import OpenAI


client = OpenAI(
    base_url="https://www.gptapi.us/v1/chat/completions", api_key=os.getenv("GPTAPI_KEY")
)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "developer", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
)

print(completion.choices[0].message.content)
