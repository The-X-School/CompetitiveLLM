from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model

bnb = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype="bfloat16"
)

model_id = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

peft_config = get_peft_config(
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, peft_config)

from datasets import load_dataset

ds = load_dataset("BAAI/TACO")
# Inspect its structure to map inputs/outputs, e.g.:
# ds["train"][0]


