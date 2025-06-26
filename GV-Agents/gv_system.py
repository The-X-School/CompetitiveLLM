from data_structures import *
from generator_agent import GeneratorAgent
from validator_agent import ValidatorAgent

class GVSystem:
    """Generator-Validator System for the two agents to communicate"""
    def __init__(self, config: Config):
        self.generator_agent = GeneratorAgent(config)
        self.validator_agent = ValidatorAgent(config)
        self.max_retries = config.max_retries
    
    # TODO: finish ts
    def generate_test_cases(self, problem: Problem) -> GeneratorResult:
        """Generate test cases for a given problem"""
        generator_result = self.generator_agent.generate_generator(problem)
        validator_code = self.validator_agent.generate_validator(problem)
        test_cases = self.validator_agent.test_inputs(generator_result.test_cases)
        return GeneratorResult(
            generator_code=generator_result.generator_code,
            commands=generator_result.commands,
            compilation_successful=generator_result.compilation_successful,
            compilation_error=generator_result.compilation_error,
            test_cases=test_cases
        )
    
        
        
