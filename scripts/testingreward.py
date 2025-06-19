from unittest.mock import patch
import io
from contextlib import redirect_stdout
import multiprocessing
import time
import builtins
import io
import sys
import re
import os
def test_code(code, cases, ex_out, time_limit):
    score = 0
    correct_cases = 0

    def run_code(code, case_input, output_queue):
        # Mock input
        builtins.input = lambda: case_input

        # Capture output
        buf = io.StringIO()
        sys.stdout = buf
        try:
            exec(code)
        except Exception as e:
            output_queue.put(("ERROR\n", 0))
            return
        finally:
            sys.stdout = sys.__stdout__

        output_queue.put((buf.getvalue(), 0))  # memory will be added later

    for i in range(len(cases)):
        output_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=run_code, args=(code, cases[i], output_queue))
        p.start()
        start_time = time.time()
        while p.is_alive() and time.time() - start_time < time_limit:
            time.sleep(0.05)

        if p.is_alive():
            p.terminate()
            out = "TIME LIMIT EXCEEDED\n"
            score -= 1 / len(cases) * 10
        else:
            try:
                out, _ = output_queue.get(timeout=1)
            except:
                out = "ERROR\n"
                score -= 1 / len(cases) * 50

        #print(f"Case {i+1}: Memory used ≈ {int(max_mem)} KB")

        if out == ex_out[i]:
            correct_cases += 1
    score += correct_cases / len(cases) * 100
    if correct_cases == len(cases):
        score += 25
    return score

code = """a,b = map(int, input().split())
print(a+b)"""

cases = [["1 2"], ["3 4"]]
ex_out = ["3", "7"]
time_limit = 1

print(test_code(code, cases, ex_out, time_limit))

