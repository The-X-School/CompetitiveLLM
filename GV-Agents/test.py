code = \
"""
import sys
import random

def generate_test_case(config):
    random.seed(config)
    n_digits = random.randint(2, 100000)
    digits = [str(random.randint(1, 9)) for _ in range(n_digits)]
    print(''.join(digits))

config = int(sys.stdin.readline())
generate_test_case(config)
"""

from utils import test_code

if __name__ == "__main__":
    print(test_code(code, ["2", "27"]))