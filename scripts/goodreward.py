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
        input_lines = iter(case_input.splitlines())
        builtins.input = lambda: next(input_lines)

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
        p.join(timeout=2)
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

        if out == ex_out[i]:
            correct_cases += 1
        else:
            print(out)
    score += correct_cases / len(cases) * 100
    if correct_cases == len(cases):
        score += 25
    return score