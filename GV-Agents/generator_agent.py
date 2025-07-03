from data_structures import *
from llm_client import LLMClient
from utils import extract_code, extract_configuration, test_code_multi_cases, queue_result
from typing import Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

class GeneratorAgent:
    """Agent responsible for generating test cases for a given problem."""
    def __init__(self, client: LLMClient, config: Config):
        """Initializes the agent."""
        self.client = client
        if client.backend != "async_openrouter":
            raise ValueError("Validator agent requires async_openrouter backend")
        self.messages = {}
        self.num_inputs = config.num_inputs_per_problem
        self.system_prompt = \
f"""
You are an expert Python coder and an expert in writing test cases for competitive programming problems. Your task is to write high-quality test-case generators in Python for competitive programming problems.

**Your Responsibilities:**
- Understand the problem and its input constraints.
- Identify edge and corner cases for inputs.
    - Design varied and diverse test cases.
    - Write a test-case generator in Python for the given problem. The generator must have the following rules:
        - Reads input from stdin (e.g., int, str)
        - Outputs the test case to stdout
        - Uses randomness (no hardcoding), but is deterministic based on the input
- GENERATED TEST CASES MUST MAINTAIN STRICT INPUT CONSISTENCY (CONSTANT NUMBER OF INPUTS EVERY TIME)
    - You MUST FOLLOW strict input consistency, or else the input will be invalid
    - Make sure you ONLY take in the constraints as the input requirements, NOT anything else
    - For example, if the input asks for three (3) integer variables, IT MUST GENERATE THREE EVERY TIME (e.g. "1 2 3", "4 2 3", but not "1 4 2 1" or "1 4 RGB")

**Guidelines:**
- Always respect problem constraints, don't generate anything extra or not enough input.
- Cover a range of test case sizes: small, medium, large
- Include edge cases (e.g., min/max values, all 0s/1s, special graph structures)
- Format generator input examples exactly like this:
    - **Configuration:** `<inputs>`
    - **Description:** `<description>`
- The generator must read from stdin and print to stdout

Output format:
1. Problem constraint analysis
2. Complete generator code (in Python)
3. Exactly {self.num_inputs} configuration examples with descriptions

Be precise, deterministic, and thorough.
"""
    
    async def generate_generator(
        self,
        problem: Problem
    ) -> GeneratorResult:
        logger.info(f"Generating test generator for problem: {problem.name}")
        
        if problem.id not in self.messages:
            self.messages[problem.id] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": problem.get_description()}
            ]
        
        output = await self.client.async_chat(self.messages[problem.id], temperature=0.0)
        self.messages[problem.id].append({"role": "assistant", "content": output})
        code = extract_code(output)
        commands = extract_configuration(output)
        logger.info("Running generator with commands")
        if code:
            tests = test_code_multi_cases(code, commands)
            logger.info("Finished running generator with commands")
            return GeneratorResult(
                response = output,
                code = code,
                commands = commands,
                inputs = [test.output for test in tests]
            )
        else:
            logger.info("Code could not be extracted")
            return GeneratorResult(
                response = output,
                code = code,
                commands = commands,
                inputs = []
            )
        
if __name__ == '__main__':
    config = Config(
        generator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        validator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324")
    )
    client = LLMClient(config.generator)
    agent = GeneratorAgent(client, config)
    
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
        solutions = [],
        sample_inputs=["5 3 10\n1 2 3 4 5\nRGBRR", "2 1 15\n5 6\nRG"],
        sample_outputs=["4", "-1"]
    )
    
    response = asyncio.run(agent.generate_generator(problem))
    print(response.response)
    print(response.code)
    print(response.commands)
    print(response.inputs)
    