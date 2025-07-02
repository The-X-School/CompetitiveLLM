from data_structures import *
from llm_client import LLMClient
from utils import extract_code, test_code_multi_cases, queue_result
import logging

logger = logging.getLogger(__name__)
class ValidatorAgent:
    def __init__(self, client: LLMClient):
        self.client = client
        if client.backend != "async_openrouter":
            raise ValueError("Validator agent requires async_openrouter backend")
        self.messages = {}
        self.system_prompt = \
f"""
You are an expert competitive programming test case validator, which means your job is to validate test cases, given both a problem and a test case. Your task is to write validators for test cases in Python that ensure that a test case inputs fully comply with the problem constraints.

**Your Responsibilities:**
- Carefully read the problem to clearly identify all constraints of the problem's input requirements.
- Write a test case validator in Python that:
    - Reads input from stdin
    - Checks every input constraint
    - Raises exceptions with clear error messages (include line numbers when possible) if there is an error detected with the given input for a problem
- Properly handle edge cases and boundary conditions.
- Make the validator strict and robust - it must catch ANY invalid input.
    - Example: if the problem's input constraints say that the first integer *n* ranges from 1 to 10 (1 <= *n* <= 10), and if the given input has the first integer (*n*) as 13, then invalidate the input.

**Guidelines:**
- Validate data ranges, input format, and structural rules
- Be explicit and informative in error messages
- Do not skip rare or tricky edge cases
- Validators must read from stdin and perform real checks (not placeholders)

**Output format:**
1. A clear summary of all constraints to be validated
2. Full Python validator code
3. A brief explanation of the validation logic

Be precise and exhaustive - missing a single constraint could break the problem."""
    
    async def generate_validator(self, problem: Problem) -> ValidatorResult:
        logger.info(f"Generating validator...")
        if problem.id not in self.messages:
            self.messages[problem.id] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": problem.get_description()}
            ]
        
        output = await self.client.async_chat(self.messages[problem.id], temperature=0.0)
        #print(output)
        self.messages[problem.id].append({"role": "assistant", "content": output})
        self.code = extract_code(output)
        return ValidatorResult(output, self.code)
    
    def test_inputs(self, code: str, test_cases: List[str]) -> List[CodeResult]:
        logger.info("Validating inputs generator by the generator agent")
        return test_code_multi_cases(code, test_cases)
    
    def give_feedback(self, commands: List[str], test_results: List[CodeResult]) -> str:
        feedback = ""
        for i in range(len(commands)):
            if test_results[i].verdict == "OK": continue
            feedback += f"- Test case {i+1} that was generated with command `{commands[i]}` failed with verdict {test_results[i].verdict}. Error: {test_results[i].error}\n"
        
        if feedback == "": feedback = "All test cases passed!"
        else:
            feedback = \
f"""
Some test cases did not meet the problem constraints:
{feedback}

Please recode the generator given these errors to meet the problem constraints, and provide the generator commands to fix these cases.
"""
        return feedback

if __name__ == "__main__":
    config = Config(
        generator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324"),
        validator = ClientConfig("openrouter", "deepseek/deepseek-chat-v3-0324")
    )
    
    client = LLMClient(config.validator)
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
    
    result = agent.generate_validator(problem)
    print("Validator Code:", result.code)
    print(agent.test_inputs(result.code, problem.sample_inputs))
