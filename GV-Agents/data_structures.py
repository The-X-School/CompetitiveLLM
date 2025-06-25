from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class ValidationResult(Enum):
    """Validation result status."""
    VALID = "valid"
    INVALID = "invalid"

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
    
    def get_full_description(self) -> str:
        """Get the complete problem description."""
        description = f"Problem: {self.name}\n\n"
        description += f"Statement:\n{self.statement}\n\n"
        
        if self.sample_inputs and self.sample_outputs:
            description += "Sample Test Cases:\n"
            for i, (inp, out) in enumerate(zip(self.sample_inputs, self.sample_outputs)):
                description += f"Input {i+1}:\n{inp}\n"
                description += f"Output {i+1}:\n{out}\n\n"
        
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
class GeneratorCommand:
    """Represents a generator command with parameters."""
    command: str
    description: str
    expected_properties: List[str]

@dataclass
class GeneratorResult:
    """Result from the Generator Agent."""
    generator_code: str
    commands: List[GeneratorCommand]
    compilation_successful: bool = False
    compilation_error: Optional[str] = None
    test_cases: List[TestCase] = None

@dataclass
class ValidationError:
    """Represents a validation error."""
    error_type: str
    message: str
    line_number: Optional[int] = None
    constraint_violated: Optional[str] = None

@dataclass
class ValidatorResult:
    """Result from the Validator Agent."""
    validator_code: str
    compilation_successful: bool = False
    compilation_error: Optional[str] = None
    validation_results: List[ValidationResult] = None
    validation_errors: List[ValidationError] = None

@dataclass
class AgentFeedback:
    """Feedback provided to agents for improvement."""
    success: bool
    message: str
    errors: List[str] = None
    suggestions: List[str] = None
