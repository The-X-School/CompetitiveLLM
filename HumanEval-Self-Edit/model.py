import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from tqdm import tqdm
from openai import OpenAI

# Load dataset
dataset = load_dataset("openai_humaneval")["test"]

OPENROUTER_API_KEY = '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'
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
model_name = "deepseek/deepseek-r1:free"

outputs = {}

for i, sample in enumerate(tqdm(dataset)):
    prompt = sample["prompt"]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=256,
        )

        generated_code = response.choices[0].message.content
        outputs[str(i)] = [generated_code]

    except Exception as e:
        print(f"Error at index {i}: {e}")
        outputs[str(i)] = ["<ERROR>"]

# Save to output.json
with open("output.json", "w") as f:
    json.dump(outputs, f, indent=4)
