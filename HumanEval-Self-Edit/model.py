import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

from openai import OpenAI

OPENROUTER_API_KEY = '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'

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"  # this overrides the default openai endpoint
)

model_name = "deepseek/deepseek-r1:free"

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "user", "content": "Write a hello world program in python."}
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)