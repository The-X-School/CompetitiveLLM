from typing import List, Dict, Tuple
from utils import test_multi_code, load_json, save_json, queue_result
from taco_generator import get_mapped_taco
from data_structures import Config, ClientConfig, Problem
from generator_agent import GeneratorAgent
from llm_client import LLMClient
from concurrent.futures import ProcessPoolExecutor
from multiprocess import Pool
import multiprocess as mp
import sys
import logging

sys.set_int_max_str_digits(0)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# returns two lists: true cases and false cases
def evaluate_truths(idx, inputs: List[str], solutions: List[str]) -> Tuple[List[str], List[str]]:
    """Evaluates if certain inputs for a problem are valid or not based on problem constraints"""
    outs = test_multi_code(solutions, inputs, 2)
    true_cases = []
    false_cases = []
    for i in range(len(inputs)):
        passed = True
        for j in range(len(solutions)):
            if idx == 2: print(outs[j][i].output)
            # if outs[j][i].verdict != 'OK' or (j > 0 and outs[j][i].output != outs[j-1][i].output):
            #     false_cases.append(inputs[i])
            #     passed = False
            #     break 
            
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

def evaluate_all_inputs(config: Config, limit_problems: int = None) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """Evaluate truths for all inputs generated"""
    all_inputs = load_json(config.all_cases_path)
    
    dataset = get_mapped_taco(config)
    if limit_problems:
        dataset = dataset[:limit_problems]
        
    true_cases = {}
    false_cases = {}
    for idx, inputs in all_inputs.items(): 
        true_cases[idx], false_cases[idx] = evaluate_truths(idx, inputs, dataset[int(idx) - 1].solutions)
        
    # yutang's great code
    # very cool one-liner by yutang
    # true_cases, false_cases = ({id: t for id, (t, f) in (evaluate_truths(inputs, dataset[int(id) - 1]['solutions'], config) for id, inputs in all_inputs.items())}, {id: f for id, (t, f) in (evaluate_truths(inputs, dataset[int(id) - 1]['solutions'], config) for id, inputs in all_inputs.items())})

    # save true and false cases

    save_json(config.true_cases_path, true_cases)
    save_json(config.false_cases_path, false_cases)
    return true_cases, false_cases

def generate_all_inputs(config: Config, use_multiprocess: bool = True, limit_problems: int = None) -> Dict[str, List[str]]:
    generator_client = LLMClient(config.generator)
    
    dataset = get_mapped_taco(config)
    if limit_problems:
        dataset = dataset[:limit_problems]

    all_inputs = {}
    if use_multiprocess:
        def process_problem(problem: Problem) -> Tuple[str, List[str]]:
            generator = GeneratorAgent(generator_client, config)
            manager = mp.Manager()
            queue = manager.Queue()
            p = mp.Process(
                target=queue_result(generator.generate_generator),
                args=(problem,),
                kwargs={"queue": queue}
            )
            p.start()
            p.join()
            
            return problem.id, generator.generate_generator(problem).inputs
        
        with Pool(processes=config.processes) as pool:
            ids, results = list(pool.map(process_problem, dataset))
            all_inputs = dict(zip(ids, results))
            
        # with ProcessPoolExecutor(max_workers=config.processes) as executor:
        #     ids, results = list(executor.map(process_problem, dataset))
        #     all_inputs = dict(zip(ids, results))
    else:
        for problem in dataset:
            generator = GeneratorAgent(generator_client, config)
            print("Problem id", problem.id)
            all_inputs[problem.id] = list(filter(
                lambda x : x is not None,
                generator.generate_generator(problem).inputs
            ))
            print(f"inputs for problem {problem.id}: {all_inputs[problem.id]}")

        # all_inputs = list(executor.map(generator.generate_generator, dataset))
        # all_inputs = dict(executor.map(lambda pair : (pair[0], pair[1]), zip(generator.generate_generator, dataset)))
    
    save_json(config.all_cases_path, all_inputs)
    return all_inputs

if __name__ == '__main__':
    config = Config(
        generator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        validator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        processes = 8
    )
    # evaluate truths
    #generate_all_inputs(config, use_multiprocess=False, limit_problems=8)
    evaluate_all_inputs(config, limit_problems=8)
