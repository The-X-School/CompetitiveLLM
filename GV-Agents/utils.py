import re
import time
import builtins
import io
import sys
import multiprocessing
import resource
from typing import List
from data_structures import CodeResult

def extract_code(text: str, language: str = "python"):
    """Extracts markdown code from a text given a language"""
    compiled = re.findall(f"```{language}(.*?)```", text, re.DOTALL)
    if len(compiled) > 0:
        return compiled[-1].strip()
    else:
        return None

def run_code(
    code: str,
    case_input: str,
    output_queue: multiprocessing.Queue,
    memory_limit: float = 256
) -> CodeResult:
    """Runs a piece of code with given inputs and captures the outputs"""

    # TODO: error on setting limit, always reports memory limit error
    # example is when i gave it 32gb of memory but it was reporting out of memory even after using only 16mb (1/2048)
    # Set memory limit
    # memory_limit_bytes = memory_limit * 1024 * 1024 # convert to bytes
    # resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
    # resource.setrlimit(resource.RLIMIT_DATA, (memory_limit_bytes, memory_limit_bytes))
    # if not sys.platform == "darwin":
    #     resource.setrlimit(resource.RLIMIT_STACK, (memory_limit_bytes, memory_limit_bytes))
    
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
        if (get_peak_memory_mb() >= memory_limit):
            return CodeResult(
                time=time.time() - start,
                memory=get_peak_memory_mb(),
                verdict="Memory Limit Error",
                error=str(e)
            )
        
        return CodeResult(
            time=time.time() - start,
            memory=get_peak_memory_mb(),
            verdict="Runtime Error",
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

def test_code(code: str, cases: List[str], time_limit: float = 256) -> List[CodeResult]:
    outs = []

    for i in range(len(cases)):
        output_queue = multiprocessing.Queue()
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