import re
import os
import time
import builtins
import io
import sys
import multiprocessing
import resource
import asyncio
from typing import List
from data_structures import CodeResult

def extract_code(text: str, language: str):
    """Extracts markdown code from a text given a language"""
    compiled = re.findall(f"```{language}(.*?)```", text)
    if len(compiled) > 0:
        return compiled[-1].rsplit()
    else:
        return None

def run_code(
    code: str,
    case_input: List[str],
    output_queue: multiprocessing.Queue,
    memory_limit: float = 256
) -> CodeResult:
    """Runs a piece of code with given inputs and captures the outputs"""
    
    hard = resource.getrlimit(resource.RLIMIT_AS)[1]
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit * 1024 * 1024, hard)) # converted to megabytes
    
    def get_peak_memory_mb():
        """Get peak memory usage"""
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin': # darwin is Mac OS X
            return usage / (1024 * 1024)  # Convert bytes -> MB
        else:
            return usage / 1024 # Convert KB -> MB  
            
    start = time.time()

    # Mock input
    input_lines = iter(case_input.splitlines())
    builtins.input = lambda: next(input_lines)

    # Capture output
    buf = io.StringIO()
    sys.stdout = buf
    try:
        exec(code)
    except Exception as e:
        if (get_peak_memory_mb() > memory_limit):
            return CodeResult(
                output="",
                time=time.time() - start,
                memory=get_peak_memory_mb(),
                verdict="MLE",
                error=str(e)
            )
        
        return CodeResult(
            output="",
            time=time.time() - start,
            memory=get_peak_memory_mb(),
            verdict="RTE",
            error=str(e)
        )
    finally:
        sys.stdout = sys.__stdout__

    result = CodeResult(
        output=buf.getvalue(),
        time=time.time() - start,
        memory=get_peak_memory_mb(),
        verdict="OK"
    )
    
    output_queue.put(result, 0)

def test_code(code, cases, time_limit):
    outs = []

    for i in range(len(cases)):
        output_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_code, args=(code, cases[i], output_queue))
        p.start()
        p.join(timeout=time_limit)
        if p.is_alive():
            p.terminate()
            out = "TIME LIMIT EXCEEDED\n"
        else:
            try:
                out, _ = output_queue.get(timeout=1)
            except:
                out = "ERROR\n"

        outs.append(out)

    return outs