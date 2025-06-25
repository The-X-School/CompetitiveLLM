from data_structures import *
from llm_client import LLMClient
from typing import List
import logging

logger = logging.getLogger(__name__)

class GeneratorAgent:
    """Agent responsible for generating test cases for a given problem."""
    def __init__(self, client: LLMClient, num_cases: int = 50):
        """Initializes the agent."""
        self.client = client
        self.num_cases = num_cases
        self.system_prompt = \
f"""
You are an expert competitive programming problem setter and test case generator. Your task is to create high-quality test case generators for competitive programming problems.

Your responsibilities:
1. Analyze the problem statement carefully to understand all constraints
2. Identify potential edge cases and corner cases
3. Design adversarial test cases that can catch common mistakes
4. Write the generator program in Python that takes command line arguments
5. Provide generator commands that cover various data sizes and special cases

Guidelines:
- Ensure the generator respects all problem constraints (very important)
- Generate diverse test cases including small, medium, and large inputs
- Some ideas for covering edge cases:
    - Minimum/maximum values
    - Extreme inputs (with all 0s, all 1s, etc.)
    - For graphs, try trees, chain, and dense graphs
- Make the generator deterministic based on command-line arguments
- Provide clear, descriptive commands with expected properties

Output format:
1. First, provide analysis of the problem constraints
2. Then provide the complete Python generator code
3. Finally, provide exactly {num_cases} generator commands with descriptions

Be thorough and precise in your implementation.
"""

    def generate_generator(self, problem: Problem) -> str:
        logger.info(f"Generating test generator for problem: {problem.name}")
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem.get_full_description()}
        ]
        output = self.client.chat(messages, temperature=0.0)
        return output
        
if __name__ == '__main__':
    client = LLMClient("openrouter", "google/gemma-3-27b-it")
    agent = GeneratorAgent(client)
    problem = Problem(
        id="1",
        name="Test Problem",
        statement="This is a test problem.",
        sample_inputs=["1", "2", "3"],
        sample_outputs=["1", "2", "3"]
    )
    print(agent.generate_generator(problem))
    