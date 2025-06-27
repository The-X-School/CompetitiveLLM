from data_structures import *
from datasets import load_dataset
from gv_system import GVSystem, GVRunner
from llm_client import LLMClient
from utils import extract_code, extract_configuration, test_code
from typing import Optional
import logging
import json
import asyncio
from tqdm import trange
import sys

sys.set_int_max_str_digits(0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def map_taco(problem: dict, idx: int) -> Problem:
    in_out = json.loads(problem["input_output"])
    return Problem(
        id = idx + 1,
        name = problem["name"],
        statement = problem["question"],
        sample_inputs = in_out["inputs"],
        sample_outputs = in_out["outputs"]
    )

if __name__ == '__main__':
    config = Config(
        generator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        validator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        good_cases_path = "data/good_cases.json",
        bad_cases_path = "data/bad_cases.json"
    )
    
    dataset_mapped = []
    dataset = load_dataset("BAAI/TACO", split="train")
    logging.info("Finished logging TACO dataset")
    
    for i in trange(len(dataset), desc="Mapping TACO dataset"):
        dataset_mapped.append(map_taco(dataset[i], i))
    
    system = GVRunner(config)
    system.run_multi(dataset_mapped[:100])