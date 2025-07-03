import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from taco_generator import get_mapped_taco
from data_structures import Config
from untrained_generator import evaluate_truths

if __name__ == '__main__':
    ds = get_mapped_taco(Config())

    problem_2 = ds[1]
    print(problem_2.statement)

    true_cases, false_cases = evaluate_truths(["1 1 1\n1\nR\n", "2 1 15\n5 6\nRG"], problem_2.solutions, pass_threshold=0.8)

    # print(f"\n{problem_2.solutions[-2]}\n")

    print(f"In true: {len(true_cases)}")
    print(f"True cases:")
    for i in range(len(true_cases)):
        print(f"{true_cases[i]}")

    print(f"In false: {len(false_cases)}")
    print(f"False cases:")
    for i in range(len(false_cases)):
        print(f"{false_cases[i]}")

    # NOTE: input format:
    # the first line contains three integers n, s, and k
    # 1 <= n <= 50
    # 1 <= s <= n
    # 1 <= k <= 2000
    # the second line contains n integers
    # the third line contains n characters

    # TODO: validator marks all test cases generated about problem 2 as incorrect, when in reality they are valid
    # Found the problem: one of the solutions just gives a runtime error due to it being incorrect
