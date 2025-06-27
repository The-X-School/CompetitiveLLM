from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

@dataclass
class ClientConfig:
    backend: str
    model: str

@dataclass
class Config:
    generator: ClientConfig
    validator: ClientConfig
    
    good_cases_path: str = 'data/good_cases.json'
    bad_cases_path: str = 'data/bad_cases.json'
    
    max_retries: int = 3
    num_inputs_per_problem: int = 15

@dataclass
class Problem:
    """Represents a competitive programming problem."""
    id: str
    name: str
    statement: str
    sample_inputs: List[str]
    sample_outputs: List[str]
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
    code: str
    commands: List[Command]
    inputs: Optional[List[str]] = None

@dataclass
class ValidatorResult:
    """Result from the Validator Agent."""
    response: str
    code: str