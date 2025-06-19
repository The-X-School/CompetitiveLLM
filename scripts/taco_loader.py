# train_grpo.py
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, get_peft_model
import multiprocessing
import psutil
import time
import builtins
import io
import sys
import re
import os
import json

def get_taco_data():
    if (os.path.exists('./taco_train')):
        TACO_train = load_from_disk('./taco_train')
    else:
        TACO_train = load_dataset("BAAI/TACO", split="train")
        TACO_train.save_to_disk('./taco_train')

    if (os.path.exists('./taco_valid')):
        TACO_valid = load_from_disk('./taco_valid')
    else:
        TACO_valid = load_dataset("BAAI/TACO", split="test")
        TACO_valid.save_to_disk('./taco_valid')

    TACO_train = TACO_train.rename_column('question', 'prompt')
    TACO_valid = TACO_valid.rename_column('question', 'prompt')

    TACO_train = TACO_train.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <think> and </think>. Then, provide your solution between <solution> and </solution>"},
            {"role": "user",   "content": x["prompt"]},
        ]
    })

    TACO_valid = TACO_valid.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <think> and </think>. Then, provide your solution between <solution> and </solution>"},
            {"role": "user",   "content": x["prompt"]},
        ]
    })

    return TACO_train, TACO_valid