from trl import RewardTrainer, GRPOTrainer, GRPOConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType

TACO = load_dataset("BAAI/TACO", split="train")
TLDR = load_dataset("trl-lib/tldr", split="train")

TACO = TACO.rename_column('question', 'prompt')
TACO = TACO.rename_column('solutions', 'completion')
print(TACO)
print(TLDR)

def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]

config = GRPOConfig(
    per_device_train_batch_size=24,  # Set your batch size here
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_num_unique_chars,
    args=config,
    train_dataset=TACO,
)

trainer.train()
