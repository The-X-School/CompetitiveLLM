from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

@dataclass(frozen=True)
class ClientConfig:
    backend: str
    model: str

@dataclass
class Config:
    generator: ClientConfig = ClientConfig("async_openrouter", "deepseek/deepseek-chat-v3-0324")
    validator: ClientConfig = ClientConfig("async_openrouter", "deepseek/deepseek-chat-v3-0324")
    
    postive_cases_path: str = 'data/postive_cases.json'
    negative_cases_path: str = 'data/negative_cases.json'
    true_cases_path: str = 'data/true_cases.json'
    false_cases_path: str = 'data/false_cases.json'
    all_cases_path: str = 'data/all_cases.json'
    mapped_taco_path: str = 'data/mapped_taco.pkl'
    
    no_response_retires: int = 3
    max_retries: int = 3
    num_inputs_per_problem: int = 15
    
    processes: int = 8

@dataclass
class Problem:
    """Represents a competitive programming problem."""
    id: str
    name: str
    statement: str
    sample_inputs: List[str]
    sample_outputs: List[str]
    solutions: List[str]
    time_limit: float = 2.0  # seconds
    memory_limit: int = 256  # MB

    def get_description(self) -> str:
        """Get the complete problem description."""
        description = f"Problem: {self.name}\n\n"
        description += f"Statement:\n{self.statement}\n\n"
        description += f"Time Limit: {self.time_limit} seconds\n"
        description += f"Memory Limit: {self.memory_limit} MB\n"
        
        return description

@dataclass
class TestCase:
    """Represents a test case."""
    input_data: str
    expected_output: str
    description: Optional[str] = None

@dataclass
class CodeResult:
    """Result from executing a piece of code."""
    time: float  # seconds
    memory: float  # MB
    verdict: str  # "OK", "TLE", "MLE", "RTE", "WA"
    output: Optional[str] = None
    error: Optional[str] = None

@dataclass
class Command:
    """Represents a terminal command with parameters."""
    command: str
    description: str
    expected_properties: List[str]

@dataclass
class GeneratorResult:
    """Result from the Generator Agent."""
    response: str
    code: Optional[str]
    commands: List[str]
    inputs: List[Optional[str]]

@dataclass
class ValidatorResult:
    """Result from the Validator Agent."""
    response: str
    code: Optional[str]