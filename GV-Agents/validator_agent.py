from data_structures import *
from llm_client import LLMClient
from utils import extract_code

class ValidatorAgent:
    def __init__(self, client: LLMClient):
        self.client = client
        self.system_prompt = \
f"""You are an expert competitive programming judge and test case validator. Your task is to create validators that check if test cases satisfy all problem constraints.

Your responsibilities:
1. Carefully analyze the problem statement to identify ALL constraints
2. Write a C++ validator program that checks every constraint
3. Provide detailed error messages when constraints are violated
4. Handle edge cases and boundary conditions properly
5. Ensure the validator is robust and catches all invalid inputs

Guidelines:
- Use testlib library for input reading and validation
- Check data ranges, format requirements, and structural constraints
- Provide specific error messages with line numbers when possible
- Handle all edge cases mentioned in the problem
- Be thorough - missing a constraint can lead to invalid test cases
- Use appropriate testlib functions like readInt(), readString(), etc.
- Call quitf() with appropriate verdict when validation fails

Validation verdicts:
- Use _ok for valid input
- Use _wa for constraint violations
- Use _pe for format errors

Output format:
1. First, provide analysis of all constraints that need to be checked
2. Then provide the complete C++ validator code
3. Explain the validation logic briefly

Be extremely thorough in identifying and checking constraints."""
    
    def generate_validator(self, problem: Problem, test_case: TestCase) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem.get_full_description()}
        ]
        output = self.client.chat(messages, temperature=0.0)
        self.validator_code = extract_code(output)
        return output
    
    def test_input(self, test_case: TestCase) -> ValidationResult:
        self.validator_code
        pass

if __name__ == "__main__":
    client = LLMClient("")