from data_structures import *
from generator_agent import GeneratorAgent
from validator_agent import ValidatorAgent

class GVSystem:
    """Generator-Validator System"""
    def __init__(self, config: Config):
        self.generator_agent = GeneratorAgent(config)
        self.validator_agent = ValidatorAgent(config)
        
        
