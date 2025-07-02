from data_structures import *
from datasets import load_dataset
from gv_system import GVSystem, GVRunner
from llm_client import LLMClient
from utils import extract_code, extract_configuration, test_code_multi_cases
from typing import Optional, List
import logging
import json
import pickle
import os
from tqdm import trange
import sys

sys.set_int_max_str_digits(0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def map_taco_problem(problem: dict, idx: int) -> Problem:
    in_out = json.loads(problem["input_output"])
    return Problem(
        id = str(idx + 1),
        name = problem["name"],
        statement = problem["question"],
        sample_inputs = in_out["inputs"],
        sample_outputs = in_out["outputs"],
        solutions = eval(problem["solutions"]),
        time_limit = problem["time_limit"],
        memory_limit = problem["memory_limit"]
    )

def get_mapped_taco(config: Config, split="train") -> List[Problem]:
    def map_full():
        dataset_mapped = []
        dataset = load_dataset("BAAI/TACO", split=split)
        for i in trange(len(dataset), desc="Mapping TACO dataset"):
            dataset_mapped.append(map_taco_problem(dataset[i], i))
        with open(config.mapped_taco_path, "wb") as f:
            pickle.dump(dataset_mapped, f)
        return dataset_mapped
    
    if os.path.exists(config.mapped_taco_path):
        with open(config.mapped_taco_path, "rb") as f:
            try: return pickle.load(f)
            except: return map_full()
    else: return map_full()

if __name__ == '__main__':
    config = Config(
        generator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        validator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        postive_cases_path = "data/postive_cases.json",
        negative_cases_path = "data/negative_cases.json",
        processes = 32
    )
    
    dataset = get_mapped_taco(config)
    logging.info("Finished loading TACO dataset")
    
    GVRunner.run_multi(dataset[:100], config)