import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../LiveCodeBench')
    )
)

from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.benchmarks import load_code_generation_dataset
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

check_correctness(
    load_code_generation_dataset()[0].get_evaluation_sample(),
    extract_code("```python\ndef mostFrequentIDs(nums):\n    return [1, 2, 3]\n```", "default"),
    timeout=6,
    debug=True,
)
