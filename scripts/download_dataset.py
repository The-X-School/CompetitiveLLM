from datasets import load_dataset
import json

ds = load_dataset("BAAI/TACO", split="train")
sample = next(iter(ds))
sample["solutions"] = json.loads(sample["solutions"])
sample["input_output"] = json.loads(sample["input_output"])
sample["raw_tags"] = eval(sample["raw_tags"])
sample["tags"] = eval(sample["tags"])
sample["skill_types"] = eval(sample["skill_types"])
print(sample["question"])
