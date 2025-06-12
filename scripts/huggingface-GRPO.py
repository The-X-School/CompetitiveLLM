# train_grpo.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, get_peft_model

train = load_dataset("trl-lib/tldr", split="train")
valid = load_dataset("trl-lib/tldr", split="validation")

# TACO = load_dataset("BAAI/TACO", split="train")
# TLDR = load_dataset("trl-lib/tldr", split="train")
#
# TACO = TACO.rename_column('question', 'prompt')
# TACO = TACO.rename_column('solutions', 'completion')

def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10, per_device_train_batch_size=24)

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = get_peft_model(model, peft_config)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=train,
    eval_dataset=valid
)
trainer.train()

