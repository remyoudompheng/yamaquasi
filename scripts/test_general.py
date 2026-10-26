"""
A test script for general-purpose factoring.

It checks factoring results and print indicative timings
(that may be affected by FFI boilerplate and Python performance).
"""

import argparse
import math
import os
import random
import sys
import time
from multiprocessing.dummy import Pool

import gmpy2

import pymqs

try:
    import cypari2

    pari = cypari2.Pari()
    pari.allocatemem(32_000_000)
except ImportError:
    pass

try:
    import sage.all

    sage.all.proof.arithmetic(False)
except ImportError:
    pass

p = argparse.ArgumentParser()
p.add_argument("-j", dest="threads", type=int, default=None, help="number of threads")
p.add_argument(
    "--algo", default="auto", help="auto, ecm, siqs.. or 'pari', 'sage' or sage-ecm"
)
p.add_argument(
    "--sampling",
    default="auto",
    choices=("auto", "uniform", "smooth", "semiprimes", "semiprimes_unbalanced"),
)
p.add_argument("minsize", metavar="SIZE1", type=int, help="lower range of number size")
p.add_argument("maxsize", metavar="SIZE2", type=int, help="upper range of number size")
p.add_argument("step", nargs="?", type=int, default=1)
args = p.parse_args()

if args.threads is None:
    args.threads = 1 if args.algo.startswith("sage") else os.cpu_count()


def randbits(bitsize):
    if bitsize <= 0:
        # Python <= 3.8 rejects random.getrandbits(0)
        return 0
    return random.getrandbits(bitsize)


BATCHSIZE = 1000


def factor_pari(n):
    factors = []
    ps, es = pari.factor(n)
    for p, k in zip(ps, es):
        factors += int(k) * [int(p)]
    return factors


def factor_sage(n):
    factors = []
    for p, k in sage.all.factor(n):
        factors += k * [p]
    return factors


def factor_sage_ecm(n):
    # For Sage ECM, cheat by hinting at factor size.
    if args.semiprimes:
        sz = n.bit_length()
        return sage.all.ecm.find_factor(n, factor_digits=sz // 7)
    else:
        return sage.all.ecm.factor(n)


def factor_smooth(n):
    return pymqs.factor_smooth(n, 20 + n.bit_length() // 30)


print(f"Test using {args.threads} threads")
for sz in range(args.minsize, args.maxsize + 1, args.step):
    numbers = []
    if sz <= 11 and ("semiprime" not in args.sampling):
        numbers = list(range(2 ** (sz - 1), 2**sz))
    elif args.sampling == "semiprimes":
        for i in range(BATCHSIZE):
            p = gmpy2.next_prime(random.getrandbits(sz // 2))
            l = sz - p.bit_length()
            q = gmpy2.next_prime((1 << (l - 1)) | random.getrandbits(l - 1))
            numbers.append(int(p * q))
    elif args.sampling == "semiprimes_unbalanced":
        for i in range(BATCHSIZE):
            p = gmpy2.next_prime(random.getrandbits(sz // 3))
            l = sz - p.bit_length()
            q = gmpy2.next_prime((1 << (l - 1)) | random.getrandbits(l - 1))
            numbers.append(int(p * q))
    elif args.sampling == "smooth":
        for _ in range(BATCHSIZE):
            n = 1
            # Generate a LCM of odd integers
            if sz < 64:
                n = random.getrandbits(sz // 2) | 1
            for b in range(20, 48):
                n = math.lcm(n, random.getrandbits(min(sz - n.bit_length(), b)) | 1)
                if n.bit_length() >= sz - 2:
                    break
            # pad with 0s
            n <<= sz - n.bit_length()
            assert n.bit_length() == sz
            numbers.append(n)
    elif args.sampling == "uniform":
        for i in range(BATCHSIZE):
            n = 2 ** (sz - 1) + random.getrandbits(sz - 1)
            numbers.append(n)
    elif args.sampling == "auto":
        for i in range(BATCHSIZE):
            # Try (nonzero) both squarefree and non-squarefree integers
            if i % 10 == 0:
                exp = random.randint(2, 10)
                x = 1 + randbits(sz // 2 // exp)
                y = 1 + randbits(sz - (x**exp).bit_length())
                n = y * x**exp
                # adj = "non squarefree "
            else:
                n = 1 + random.getrandbits(sz)
                # adj = ""
            numbers.append(n)

    match args.algo:
        case "pari":
            factor_func = factor_pari
        case "sage":
            factor_func = factor_sage
        case "sage-ecm":
            factor_func = factor_sage_ecm
        case "smooth":
            factor_func = factor_smooth
        case _:
            factor_func = lambda _n: pymqs.factor(_n, algo=args.algo)

    def test_number(n):
        try:
            t0 = time.monotonic()
            facs = factor_func(n)
            return facs, time.monotonic() - t0
        except Exception:
            print(f"Exception during factor({n})", file=sys.stderr)
            return None, time.monotonic() - t0

    # Warmup
    [test_number(_n) for _n in numbers[:10]]

    if args.threads == 1:
        results = [test_number(_n) for _n in numbers]
    else:
        with Pool(args.threads) as pool:
            results = pool.map(test_number, numbers)

    t1 = sum(t for (_, t) in results)
    t2 = sum(t * t for (_, t) in results)
    mean = t1 / len(results)
    # Bootstrap sampling
    subsample = (len(results) + 1) // 2
    submeans = [
        sum(t for (_, t) in random.sample(results, subsample)) / subsample
        for _ in range(10)
    ]
    mean1, mean2 = min(submeans), max(submeans)
    std = math.sqrt(t2 / len(results) - (t1 / len(results)) ** 2)
    if mean2 < 1e-3:
        print(
            f"Bit length {sz}: tested {len(numbers)} numbers in {t1:.3f}s (mean {mean1 * 1e6:.3f}-{mean2 * 1e6:.3f}us, std {std * 1e6:.3f}us)"
        )
    elif mean2 < 1.0:
        print(
            f"Bit length {sz}: tested {len(numbers)} numbers in {t1:.3f}s (mean {mean1 * 1000:.3f}-{mean2 * 1000:.3f}ms, std {std * 1000:.3f}ms)"
        )
    else:
        print(
            f"Bit length {sz}: tested {len(numbers)} numbers in {t1:.3f}s (mean {mean1:.3f}-{mean2:.3f}s, std {std:.3f}s)"
        )
    # Check results
    for n, (factors, _) in zip(numbers, results):
        if n == 0:
            assert factors == [0], n
            continue
        if factors is None:
            continue
        prod = 1
        for f in factors:
            prod *= f
            assert gmpy2.is_prime(f), (n, factors)
        assert n == prod, (n, factors)
