"""
A test script for general-purpose factoring.
"""

import argparse
import random
import subprocess
import time

import gmpy2

p = argparse.ArgumentParser()
p.add_argument("bitsize", metavar="BITSIZE", type=int, help="target size of input integers")
args = p.parse_args()

sz = args.bitsize
for i in range(1000):
    # Try both squarefree and non-squarefree integers
    if i % 10 == 0:
        exp = random.randint(2, 10)
        x = random.getrandbits(sz // 2 // exp)
        y = random.getrandbits(sz // 2)
        n = y*x**exp
        adj = "non squarefree "
    else:
        n = random.getrandbits(sz)
        adj = ""
    if n < 2:
        continue
    print(f"Testing {adj}number {n}")
    t0 = time.time()
    out = subprocess.check_output(["bin/ymqs", str(n)],
        stderr=subprocess.DEVNULL)
    dt = time.time() - t0
    factors = [int(f) for f in out.split()]
    prod = 1
    for f in factors:
        prod *= f
        if not gmpy2.is_prime(f):
            print(f"WARNING: factor {f} is composite")
        assert gmpy2.is_prime(f)
    assert n == prod
    print(f"OK in {dt:.3}s {factors}")


