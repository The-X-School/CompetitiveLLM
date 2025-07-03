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
            # max_tokens=256,
        )

        msg = response.choices[0].message
        generated_code = msg.content.strip() or msg.reasoning.strip() or "<EMPTY>"
        #generated_code = response.choices[0].message.content
        print(f"[{i}] Raw response:\n{response}\n")

        print(f"[{i}] Prompt:\n{prompt}\n---\nOutput:\n{generated_code}\n")

        outputs[str(i)] = [generated_code]

    except Exception as e:
        print(f"Error at index {i}: {e}")
        outputs[str(i)] = ["<ERROR>"]

# Save to output.json
with open("output.json", "w") as f:
    json.dump(outputs, f, indent=4)
