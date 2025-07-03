from typing import List, Dict, Tuple, Optional
from utils import test_multi_code, load_json, save_json, queue_result
from taco_generator import get_mapped_taco
from data_structures import Config, ClientConfig, Problem
from generator_agent import GeneratorAgent
from llm_client import LLMClient
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import multiprocessing as mp
import asyncio
import sys
import logging

sys.set_int_max_str_digits(0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# returns two lists: true cases and false cases
def evaluate_truths(inputs: List[str], solutions: List[str], solution_limit: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Evaluates if certain inputs for a problem are valid or not based on solution verdict"""
    if solution_limit: sliced_sols = solutions[:solution_limit]
    else: sliced_sols = solutions

    outs = test_multi_code(sliced_sols, inputs, 2)
    # for out in outs: print(out)
    true_cases = []
    false_cases = []
    for i in range(len(inputs)):
        passed = True
        for j in range(len(sliced_sols)):            # if outs[j][i].verdict != 'OK' or (j > 0 and outs[j][i].output != outs[j-1][i].output):
            if outs[j][i].verdict != 'OK':
                false_cases.append(inputs[i])
                passed = False
                break
            
            if j > 0 and outs[j][i].output != outs[j-1][i].output:
                false_cases.append(inputs[i])
                passed = False
                break
        
        if passed:
            true_cases.append(inputs[i])
    
    return true_cases, false_cases

def evaluate_all_inputs(config: Config, limit_problems: Optional[int] = None) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """Evaluate truths for all inputs generated"""
    all_inputs = load_json(config.all_cases_path)
    
    dataset = get_mapped_taco(config)
    if limit_problems:
        dataset = dataset[:limit_problems]
        
    true_cases = {}
    false_cases = {}
    for idx, inputs in all_inputs.items(): 
        true_cases[idx], false_cases[idx] = evaluate_truths(inputs, dataset[int(idx) - 1].solutions)

    # NOTE: yutang's great code
    # NOTE: very cool one-liner by yutang
    # true_cases, false_cases = ({id: t for id, (t, f) in (evaluate_truths(inputs, dataset[int(id) - 1]['solutions'], config) for id, inputs in all_inputs.items())}, {id: f for id, (t, f) in (evaluate_truths(inputs, dataset[int(id) - 1]['solutions'], config) for id, inputs in all_inputs.items())})

    # save true and false cases

    save_json(config.true_cases_path, true_cases)
    save_json(config.false_cases_path, false_cases)
    return true_cases, false_cases

def generate_all_inputs(config: Config, use_async = True, limit_problems: Optional[int] = None) -> Dict[str, List[str]]:
    generator = GeneratorAgent(LLMClient(config.generator), config)
    
    dataset = get_mapped_taco(config)
    if limit_problems:
        dataset = dataset[:limit_problems]

    all_inputs = load_json(config.all_cases_path, {})
    if use_async:
        sem = asyncio.Semaphore(config.processes)
        async def process_problem(problem: Problem) -> Tuple[str, List[Optional[str]]]:
            async with sem:
                result = await generator.generate_generator(problem)
                return problem.id, result.inputs
        
        async def async_generate():
            return await asyncio.gather(*(process_problem(problem) for problem in dataset))
        
        results = asyncio.run(async_generate())
        for result in results:
            value = list(filter(lambda x : x is not None, result[1]))
            if result[0] in all_inputs: all_inputs[result[0]].extend(value)
            else: all_inputs[result[0]] = value
        all_inputs = asyncio.run(async_generate())
            
        # with ProcessPoolExecutor(max_workers=config.processes) as executor:
        #     ids, results = list(executor.map(process_problem, dataset))
        #     all_inputs = dict(zip(ids, results))
    else:
        for problem in dataset:
            print("Problem id", problem.id)
            all_inputs[problem.id] = list(filter(
                lambda x : x is not None,
                asyncio.run(generator.generate_generator(problem)).inputs
            ))
            print(f"inputs for problem {problem.id}: {all_inputs[problem.id]}")

        # all_inputs = list(executor.map(generator.generate_generator, dataset))
        # all_inputs = dict(executor.map(lambda pair : (pair[0], pair[1]), zip(generator.generate_generator, dataset)))
    
    save_json(config.all_cases_path, all_inputs) # type: ignore
    return all_inputs # type: ignore

if __name__ == '__main__':
    problem_limit = 4
    thread_limit = 4
    config = Config(
        generator = ClientConfig("async_openrouter", "deepseek/deepseek-chat-v3-0324"),
        validator = ClientConfig("async_openrouter", "deepseek/deepseek-chat-v3-0324"),
        processes = thread_limit
    )
    # evaluate truths
    generate_all_inputs(config, limit_problems=problem_limit)
    evaluate_all_inputs(config, limit_problems=problem_limit)
