from typing import List, Dict, Tuple
from utils import test_multi_code, load_json, save_json
from taco_generator import get_mapped_taco
from data_structures import Config, ClientConfig, Problem
from generator_agent import GeneratorAgent
from llm_client import LLMClient
from concurrent.futures import ProcessPoolExecutor

# returns two lists: true cases and false cases
def evaluate_truths(inputs: List[str], solutions: List[str]) -> Tuple[List[str], List[str]]:
    """Evaluates if certain inputs for a problem are valid or not based on problem constraints"""
    outs = test_multi_code(solutions, inputs, 2)
    true_cases = []
    false_cases = []
    for i in range(len(inputs)):
        passed = True
        for j in range(len(solutions)):
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
        true_cases[idx], false_cases[idx] = evaluate_truths(inputs, dataset[int(idx) - 1].solutions, config)
        
    # yutang's great code
    # very cool one-liner by yutang
    # true_cases, false_cases = ({id: t for id, (t, f) in (evaluate_truths(inputs, dataset[int(id) - 1]['solutions'], config) for id, inputs in all_inputs.items())}, {id: f for id, (t, f) in (evaluate_truths(inputs, dataset[int(id) - 1]['solutions'], config) for id, inputs in all_inputs.items())})

    # save true and false cases

    save_json(config.true_cases_path, true_cases)
    save_json(config.false_cases_path, false_cases)
    return true_cases, false_cases

def generate_all_inputs(config: Config, use_multiprocessing: bool = True, limit_problems: int = None) -> Dict[str, List[str]]:
    generator = GeneratorAgent(LLMClient(config.generator), config)
    
    dataset = get_mapped_taco(config)
    if limit_problems:
        dataset = dataset[:limit_problems]

    all_inputs = {}
    if use_multiprocessing:
        def process_problem(problem: Problem) -> Tuple[str, List[str]]:
            return problem.id, generator.generate_generator(problem).inputs
        
        with ProcessPoolExecutor(max_workers=config.processes) as executor:
            ids, results = list(executor.map(process_problem, dataset))
            all_inputs = dict(zip(ids, results))
    else:
        for problem in dataset:
            all_inputs[problem.id] = generator.generate_generator(problem).inputs

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
    generate_all_inputs(config, False, 8)
    evaluate_all_inputs(config, 8)
