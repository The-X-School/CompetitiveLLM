import logging
import json
import tqdm
import dataclasses
import asyncio
from utils import load_json, save_json
from data_structures import *
from llm_client import LLMClient
from generator_agent import GeneratorAgent
from validator_agent import ValidatorAgent

logger = logging.getLogger(__name__)

class GVSystem:
    """Generator-Validator System for the two agents to communicate"""
    def __init__(self, generator: LLMClient, validator: LLMClient, config: Config):
        self.generator_agent = GeneratorAgent(generator, config)
        self.validator_agent = ValidatorAgent(validator)
        self.max_retries = config.max_retries
        self.good_cases_path = config.good_cases_path
        self.bad_cases_path = config.bad_cases_path
    
    async def generate_test_cases(self, problem: Problem) -> List[str]:
        """Generate test cases for a given problem"""
        test_cases = []
        generator_result = []
        if (
            self.generator_agent.client.backend == "openrouter" and
            self.validator_agent.client.backend == "openrouter"
        ):
            validator_result, generator_result = await asyncio.gather(
                self.validator_agent.generate_validator(problem),
                self.generator_agent.generate_generator(problem)
            )
        else:
            validator_result = await self.validator_agent.generate_validator(problem)
            generator_result = await self.generator_agent.generate_generator(problem)
            
        print("Validator:", validator_result.code)
        print("\nGenerator:", generator_result.inputs)
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
                
            print("\nValidator:", feedback)
            if feedback == "All test cases passed!": break
            self.generator_agent.messages.append({"role": "user", "content": feedback})
            
            generator_result = await self.generator_agent.generate_generator(problem)
            print("\nGenerator:", generator_result.inputs)
            #print("\nGenerator:", generator_result.response)
        
        inputs = []
        good_cases = load_json(self.good_cases_path, {})
        bad_cases = load_json(self.bad_cases_path, {})
        
        if problem.id not in good_cases:
            good_cases[problem.id] = []
        
        if problem.id not in bad_cases:
            bad_cases[problem.id] = []
        
        for i in range(len(test_cases)):
            if test_cases[i].verdict == "OK":
                good_cases[problem.id].append(generator_result.inputs[i])
                inputs.append(generator_result.inputs[i])
            else:
                bad_cases[problem.id].append("inputs": generator_result.inputs[i])
        
        save_json(self.good_cases_path, good_cases)
        save_json(self.bad_cases_path, bad_cases)
            
        return inputs

def run_multi_gv(problems: List[Problem], config: Config):
    """Run the GV system on multiple problems"""
    generator = LLMClient(config.generator)
    validator = LLMClient(config.validator)
    
    for problem in tqdm.tqdm(problems):
        print(problem.statement)
        logging.info(f"Generating test cases for problem {problem.name} with ID {problem.id}")
        system = GVSystem(generator, validator, config)
        asyncio.run(system.generate_test_cases(problem))

if __name__ == '__main__':
    config = Config(
        generator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        validator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        good_cases_path = "data/good_cases.json",
        bad_cases_path = "data/bad_cases.json"
    )
    generator = LLMClient(config.generator)
    validator = LLMClient(config.validator)
    system = GVSystem(generator, validator, config)
    
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
    
    problem = Problem(
        id="1",
        name="Tanya and Colored Candies",
        statement=statement,
        sample_inputs=["5 3 10\n1 2 3 4 5\nRGBRR", "2 1 15\n5 6\nRG"],
        sample_outputs=["4", "-1"]
    )
    
    asyncio.run(system.generate_test_cases(problem))