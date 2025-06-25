from data_structures import *
from llm_client import LLMClient
from utils import extract_code, test_code

class ValidatorAgent:
    def __init__(self, client: LLMClient):
        self.client = client
        self.system_prompt = \
f"""You are an expert competitive programming judge and test case validator. Your task is to create validators that check if a test case input satisfy all problem constraints.

Your responsibilities:
1. Carefully analyze the problem statement to identify ALL constraints
2. Write a Python validator program that checks every constraint
3. Throw an exception with a descriptive error message when a constraint is violated
4. Handle edge cases and boundary conditions properly
5. Ensure the validator is robust and catches all invalid inputs

Guidelines:
- Check data ranges, format requirements, and structural constraints
- Provide specific error messages with line numbers when possible
- Handle all edge cases mentioned in the problem
- Be thorough - missing a constraint can lead to invalid test cases

Output format:
1. First, provide analysis of all constraints that need to be checked
2. Then provide the complete Python validator code
3. Explain the validation logic briefly

Be extremely thorough in identifying and checking constraints."""
    
    def generate_validator(self, problem: Problem) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem.get_description()}
        ]
        output = self.client.chat(messages, temperature=0.0)
        self.validator_code = extract_code(output)
        return output
    
    def test_inputs(self, test_cases: List[TestCase]) -> List[CodeResult]:
        return test_code(self.validator_code, test_cases)

if __name__ == "__main__":
    client = LLMClient("openrouter", "google/gemma-3-27b-it")
    agent = ValidatorAgent(client)
    
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
    
    agent.generate_validator(problem)
    print("Validator Code:", agent.validator_code)
    print(agent.test_inputs(problem.sample_inputs))
