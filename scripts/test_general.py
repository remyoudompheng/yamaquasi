"""
A test script for general-purpose factoring.
"""

import argparse
import random
import time
from multiprocessing.dummy import Pool
import os

import gmpy2
import pymqs

p = argparse.ArgumentParser()
p.add_argument(
    "-j", dest="threads", type=int, default=os.cpu_count(), help="number of threads"
)
p.add_argument("--algo", default="auto", help="auto or ecm")
p.add_argument("minsize", metavar="SIZE1", type=int, help="lower range of number size")
p.add_argument("maxsize", metavar="SIZE2", type=int, help="upper range of number size")
args = p.parse_args()

print(f"Test using {args.threads} threads")
for sz in range(args.minsize, args.maxsize + 1):
    numbers = []
    if sz <= 11:
        numbers = list(range(2 ** (sz - 1), 2**sz))
    else:
        for i in range(1000):
            # Try both squarefree and non-squarefree integers
            if i % 10 == 0:
                exp = random.randint(2, 10)
                x = random.getrandbits(sz // 2 // exp)
                y = random.getrandbits(sz // 2)
                n = y * x**exp
                adj = "non squarefree "
            else:
                n = random.getrandbits(sz)
                adj = ""
            numbers.append(n)

    def test_number(n):
        try:
            factors = pymqs.factor(n, algo=args.algo)
        except Exception:
            print(f"Exception during factor({n})")
            raise
        if n == 0:
            assert factors == [0], n
            return
        prod = 1
        for f in factors:
            prod *= f
            assert gmpy2.is_prime(f), (n, factors)
        assert n == prod, (n, factors)

    t = time.time()
    with Pool(args.threads) as pool:
        pool.map(test_number, numbers)
    dt = time.time() - t
    print(f"Bit length {sz}: tested {len(numbers)} numbers in {dt:.3}s")
