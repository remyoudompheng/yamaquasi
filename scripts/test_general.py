"""
A test script for general-purpose factoring.

It checks factoring results and print indicative timings
(that may be affected by FFI boilerplate and Python performance).
"""

import argparse
import random
import time
from multiprocessing.dummy import Pool
import os

import gmpy2
import pymqs

try:
    import sage.all

    sage.all.proof.arithmetic(False)
except ImportError:
    pass

p = argparse.ArgumentParser()
p.add_argument("-j", dest="threads", type=int, default=None, help="number of threads")
p.add_argument("--algo", default="auto", help="auto, ecm, siqs.. or 'sage' or sage-ecm")
p.add_argument("--semiprimes", action="store_true")
p.add_argument("minsize", metavar="SIZE1", type=int, help="lower range of number size")
p.add_argument("maxsize", metavar="SIZE2", type=int, help="upper range of number size")
args = p.parse_args()

if args.threads is None:
    args.threads = 1 if args.algo.startswith("sage") else os.cpu_count()

print(f"Test using {args.threads} threads")
for sz in range(args.minsize, args.maxsize + 1):
    numbers = []
    if args.semiprimes:
        numbers = [
            int(
                gmpy2.next_prime(random.getrandbits(sz // 2))
                * gmpy2.next_prime(random.getrandbits(sz - sz // 2))
            )
            for i in range(1000)
        ]
    elif sz <= 11:
        numbers = list(range(2 ** (sz - 1), 2**sz))
    else:
        for i in range(1000):
            # Try (nonzero) both squarefree and non-squarefree integers
            if i % 10 == 0:
                exp = random.randint(2, 10)
                x = 1 + random.getrandbits(sz // 2 // exp)
                y = 1 + random.getrandbits(sz - (x**exp).bit_length())
                n = y * x**exp
                adj = "non squarefree "
            else:
                n = 1 + random.getrandbits(sz)
                adj = ""
            numbers.append(n)

    def test_number(n):
        try:
            if args.algo == "sage":
                factors = []
                for p, k in sage.all.factor(n):
                    factors += k * [p]
                return factors
            elif args.algo == "sage-ecm":
                # For Sage ECM, cheat by hinting at factor size.
                if args.semiprimes:
                    return sage.all.ecm.find_factor(n, factor_digits=sz // 7)
                else:
                    return sage.all.ecm.factor(n)
            else:
                return pymqs.factor(n, algo=args.algo)
        except Exception:
            print(f"Exception during factor({n})")
            raise

    t = time.time()
    if args.threads == 1:
        results = [test_number(_n) for _n in numbers]
    else:
        with Pool(args.threads) as pool:
            results = pool.map(test_number, numbers)
    dt = time.time() - t
    print(f"Bit length {sz}: tested {len(numbers)} numbers in {dt:.3}s")
    # Check results
    for n, factors in zip(numbers, results):
        if n == 0:
            assert factors == [0], n
            continue
        prod = 1
        for f in factors:
            prod *= f
            assert gmpy2.is_prime(f), (n, factors)
        assert n == prod, (n, factors)
