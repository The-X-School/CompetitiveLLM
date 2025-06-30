import re
import time
import builtins
import json
import io
import sys
import multiprocessing
import resource
import ast
from functools import wraps
from pathlib import Path
from typing import List
from data_structures import CodeResult

# Import libraries to make available for the test code that is generated
import math
import random
import numpy as np

default_globals = {
    "__builtins__": __builtins__,
    "math": math,
    "random": random,
    "np": np,
    "sys": sys
}

# Copied from LCB (livecodebench)
def clean_if_main(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except:
        pass

    return code

def extract_code(text: str, language: str = "python") -> str:
    """Extracts markdown code from a text given a language"""
    compiled = re.findall(f"```{language}(.*?)```", text, re.DOTALL)
    if len(compiled) > 0:
        return clean_if_main(compiled[-1].strip())
    else:
        return None

def extract_configuration(text: str) -> List[str]:
    return re.findall(r"\*\*Configuration:\*\* `(.*?)`", text, re.DOTALL)

# TODO: Fix ts
def reliability_guard(memory_limit: float = 256):
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    memory_limit_bytes = memory_limit * 1024 * 1024  # in bytes

    if memory_limit_bytes > hard:
        memory_limit_bytes = hard  # Cap to avoid ValueError
        
    print(memory_limit_bytes, hard)
        
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    resource.setrlimit(resource.RLIMIT_DATA, (memory_limit_bytes, memory_limit_bytes))
    if not sys.platform == "darwin":
       resource.setrlimit(resource.RLIMIT_STACK, (memory_limit_bytes, memory_limit_bytes))

def run_code(
    code: str,
    case_input: str,
    output_queue: multiprocessing.Queue,
    memory_limit: float = 256
) -> CodeResult:
    """Runs a piece of code with given inputs and captures the outputs"""
    
    #reliability_guard(memory_limit)
    
    def get_peak_memory_mb():
        """Get peak memory usage"""
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin': # darwin is Mac OS X
            return usage / (1024 * 1024)  # Convert bytes -> MB
        else:
            return usage / 1024 # Convert KB -> MB  
            
    start = time.time()

    # Mock input
    input_stream = io.StringIO(case_input)
    sys.stdin = input_stream
    builtins.input = lambda: input_stream.readline().rstrip('\n')  # Optional override

    # Capture output
    buf = io.StringIO()
    sys.stdout = buf
    try:
        exec(code, default_globals)
    except Exception as e:
        if (get_peak_memory_mb() >= memory_limit):
            output_queue.put(CodeResult(
                time=time.time() - start,
                memory=get_peak_memory_mb(),
                verdict="Memory Limit Error",
                error=str(e)
            ), 0)
            return
        
        output_queue.put(CodeResult(
            time=time.time() - start,
            memory=get_peak_memory_mb(),
            verdict="Runtime Error",
            error=str(e)
        ), 0)
        return
    finally:
        sys.stdout = sys.__stdout__

    result = CodeResult(
        output=buf.getvalue(),
        time=time.time() - start,
        memory=get_peak_memory_mb(),
        verdict="OK"
    )
    output_queue.put(result, 0)
    return

def test_code(code: str, cases: List[str], time_limit: float = 2) -> List[CodeResult]:
    outs = []
    manager = multiprocessing.Manager()

    for i in range(len(cases)):
        output_queue = manager.Queue()
        p = multiprocessing.Process(target=run_code, args=(code, cases[i], output_queue))
        p.start()
        p.join(timeout=time_limit)
        if p.is_alive():
            p.terminate()
            out = CodeResult(
                time=time_limit,
                memory=0,
                verdict="Time Limit Error"
            )
        else:
            try:
                out = output_queue.get(timeout=1)
            except Exception as e:
                out = CodeResult(
                    time=0,
                    memory=0,
                    verdict="Runtime Error",
                    error=str(e)
                )

        outs.append(out)

    return outs

if __name__ == '__main__':
    code = \
"""
outs = []
for i in range(1000000):
    outs.append(i)
print(outs[:100])
"""
    cases = ["1 2 3", "4 5 6"]
    outs = test_code(code, cases)
    for out in outs:
        print(out)
        
def load_json(file_path: str, default: dict = {}):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except:
        return default
    
def save_json(file_path: str, data: dict):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(file_path, "w"), indent=2)
    
def queue_result(func):
    @wraps(func)
    def wrapper(*args, queue=None, **kwargs):
        result = func(*args, **kwargs)
        if queue is not None:
            queue.put(result)
    return wrapper