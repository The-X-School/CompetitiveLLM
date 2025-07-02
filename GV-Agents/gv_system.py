import logging
import json
import tqdm
import dataclasses
import multiprocess
import asyncio
from concurrent.futures import ProcessPoolExecutor
from utils import load_json, save_json, queue_result
from data_structures import *
from llm_client import LLMClient
from generator_agent import GeneratorAgent
from validator_agent import ValidatorAgent

multiprocess.set_start_method('fork')
logger = logging.getLogger(__name__)

class GVSystem:
    """Generator-Validator System for the two agents to communicate"""
    def __init__(self, generator: LLMClient, validator: LLMClient, config: Config):
        self.generator_agent = GeneratorAgent(generator, config)
        self.validator_agent = ValidatorAgent(validator)
        self.max_retries = config.max_retries
        self.postive_cases_path = config.postive_cases_path
        self.negative_cases_path = config.negative_cases_path
    
    async def generate_test_cases(self, problem: Problem, retries: int = 0) -> List[str]:
        """Parallelized of generation  test cases for a given problem"""
        test_cases = []
    
        # q1 = multiprocess.Queue()
        # q2 = multiprocess.Queue()
        # p1 = multiprocess.Process(
        #     target=self.validator_agent.generate_validator,
        #     args=(problem,),
        #     kwargs={"queue": q1}
        # )
        # p2 = multiprocess.Process(
        #     target=self.generator_agent.generate_generator,
        #     args=(problem,),
        #     kwargs={"queue": q2}
        # )
        # p1.start()
        # p2.start()
        # p1.join()
        # p2.join()
        # validator_result = q1.get()
        # generator_result = q2.get()
        await asyncio.gather()
        validator_result = await self.validator_agent.generate_validator(problem)
        generator_result = await self.generator_agent.generate_generator(problem)
        
        if not validator_result or not generator_result:
            if retries < self.max_retries:
                return self.generate_test_cases(problem, retries + 1)
            else:
                print(f"Skipping problem with id {problem.id} after {self.max_retries} retries.")
                return []

        #print("Validator:", validator_result.code)
        #print("\nGenerator:", generator_result.inputs)
        #print("\nGenerator:", generator_result.response)
        
        for i in range(self.max_retries):
            test_cases = self.validator_agent.test_inputs(
                validator_result.code,
                generator_result.inputs
            )
            
            if (i < self.max_retries - 1):
                feedback = self.validator_agent.give_feedback(
                    generator_result.commands, test_cases
                )
                
            #print("\nValidator:", feedback)
            if feedback == "All test cases passed!": break
            self.generator_agent.messages.append({"role": "user", "content": feedback})
            
            generator_result = self.generator_agent.generate_generator(problem)
            #print("\nGenerator:", generator_result.inputs)
            #print("\nGenerator:", generator_result.response)
        
        inputs = []
        postive_cases = {}
        negative_cases = {}
        postive_cases = load_json(self.postive_cases_path, {})
        negative_cases = load_json(self.negative_cases_path, {})
        
        if problem.id not in postive_cases:
            postive_cases[problem.id] = []
        
        if problem.id not in negative_cases:
            negative_cases[problem.id] = []
        
        print("Good cases keys:", postive_cases.keys())
        print("Bad cases keys:", negative_cases.keys())
        for i in range(len(test_cases)):
            if test_cases[i].verdict == "OK":
                postive_cases[problem.id].append(generator_result.inputs[i])
                inputs.append(generator_result.inputs[i])
            else:
                negative_cases[problem.id].append(generator_result.inputs[i])
        
        save_json(self.postive_cases_path, postive_cases)
        save_json(self.negative_cases_path, negative_cases)
        
        #logging.info(f"Finished generating test cases for problem {problem.name} with ID {problem.id}")
        return inputs

# class GVRunner:
#     """Runner class for the GV system"""
#     def __init__(self, config: Config):
#         self.config = config
#         self.generator = LLMClient(config.generator)
#         self.validator = LLMClient(config.validator)
    
#     def run_single(self, problem: Problem):
#         logging.info(f"Generating test cases for problem {problem.name} with ID {problem.id}")
#         self.system = GVSystem(self.generator, self.validator, self.config)
#         return self.system.generate_test_cases(problem)
        
#     def run_multi(self, problems: List[Problem]):
#         with multiprocess.Pool(self.config.processes) as pool:
#             return pool.map(self.run_single, problems)

class GVRunner:
    @staticmethod
    def _run_single(problem: Problem, config: Config):
        """Standalone function that can be pickled for multiprocess"""
        logging.info(f"Generating test cases for problem {problem.name} with ID {problem.id}")

        # Create fresh instances for each process
        generator = LLMClient(config.generator)
        validator = LLMClient(config.validator)
        system = GVSystem(generator, validator, config)

        return system.generate_test_cases(problem)

    @staticmethod
    def run_multi(problems: List[Problem], config: Config):
        # Create tuples of (config, problem) for the standalone function
        # config_problem_pairs = [(self.config, problem) for problem in problems]
        # with multiprocess.Pool(self.config.processes) as pool:
        #     return pool.map(_run_single_problem, config_problem_pairs)
        with ProcessPoolExecutor(max_workers=config.processes) as executor:
            return list(executor.map(GVRunner._run_single, problems, [config] * len(problems)))

if __name__ == '__main__':
    config = Config(
        generator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        validator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        processes = 32
    )
    generator = LLMClient(config.generator)
    validator = LLMClient(config.validator)
    system = GVSystem(generator, validator, config)
    
    # statement pulled from codeforces "Tanya and Colored Candies" (https://codeforces.com/problemset/problem/1057/C)
    statement = \
"""
There are $n$ candy boxes in front of Tanya. The boxes are arranged in a row from left to right, numbered from $1$ to $n$. The $i$-th box contains $r_i$ candies, candies have the color $c_i$ (the color can take one of three values - red, green, or blue). All candies inside a single box have the same color (and it is equal to $c_i$).
Initially, Tanya is next to the box number $s$. Tanya can move to the neighbor box (that is, with a number that differs by one) or eat candies in the current box. Tanya eats candies instantly, but the movement takes one second.
If Tanya eats candies from the box, then the box itself remains in place, but there is no more candies in it. In other words, Tanya always eats all the candies from the box and candies in the boxes are not refilled.
It is known that Tanya cannot eat candies of the same color one after another (that is, the colors of candies in two consecutive boxes from which she eats candies are always different). In addition, Tanya's appetite is constantly growing, so in each next box from which she eats candies, there should be strictly more candies than in the previous one.
Note that for the first box from which Tanya will eat candies, there are no restrictions on the color and number of candies.
Tanya wants to eat at least $k$ candies. What is the minimum number of seconds she will need? Remember that she eats candies instantly, and time is spent only on movements.

-----Input-----
The first line contains three integers $n$, $s$ and $k$ ($1 \\le n \\le 50$, $1 \\le s \\le n$, $1 \\le k \\le 2000$) - number of the boxes, initial position of Tanya and lower bound on number of candies to eat. The following line contains $n$ integers $r_i$ ($1 \\le r_i \\le 50$) - numbers of candies in the boxes. The third line contains sequence of $n$ letters 'R', 'G' and 'B', meaning the colors of candies in the correspondent boxes ('R' for red, 'G' for green, 'B' for blue). Recall that each box contains candies of only one color. The third line contains no spaces.

-----Output-----
Print minimal number of seconds to eat at least $k$ candies. If solution doesn't exist, print "-1".

-----Examples-----
Input
5 3 10
1 2 3 4 5
RGBRR

Output
4

Input
2 1 15
5 6
RG

Output
-1

-----Note-----
The sequence of actions of Tanya for the first example:
  move from the box $3$ to the box $2$;  eat candies from the box $2$;  move from the box $2$ to the box $3$;  eat candy from the box $3$;  move from the box $3$ to the box $4$;  move from the box $4$ to the box $5$;  eat candies from the box $5$. 
Since Tanya eats candy instantly, the required time is four seconds.
"""
    
    # example problem pulled from codeforces, we are actually using BAAI/TACO
    problem = Problem(
        id="1",
        name="Tanya and Colored Candies",
        statement=statement,
        sample_inputs=["5 3 10\n1 2 3 4 5\nRGBRR", "2 1 15\n5 6\nRG"],
        sample_outputs=["4", "-1"]
    )
    
    system.generate_test_cases(problem)
