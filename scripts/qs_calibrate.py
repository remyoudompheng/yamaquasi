"""
Explore factor base sizes for optimal performance of the classical
quadratic sieve.

This is similar to script mpqs_calibrate_fbase.py
"""

from math import log, sqrt
from random import randint
import time

from gmpy2 import next_prime

import pymqs

SIZES = [
    # Bit length, Factor base size, FB size for 2x large prime, Samples
    (70, 120, 60, 400),
    (80, 120, 60, 300),
    (90, 200, 90, 200),
    (100, 300, 120, 100),
    (110, 500, 200, 40),
    (120, 800, 300, 20),
    (130, 1200, 450, 20),
    (140, 2000, 650, 10),
    (150, 3200, 1000, 8),
    (160, 5500, 1500, 6),
    (170, 8000, 2500, 5),
    (180, 13000, 4000, 4),
    (190, 20000, 7000, 3),
    (200, 26000, 10000, 2),
    # Double large primes activated
    (210, 30000, 14000, 2),
    (220, 35000, 20000, 2),
    (230, 50000, 35000, 2),
    (240, 70000, 50000, 2),
    (250, 90000, 65000, 2),
    (260, 120000, 80000, 2),
]


def get_prime(bits):
    return int(next_prime(randint(2 ** (bits - 1), 2**bits)))


for sz, fb1, fb2, count in SIZES:
    use_double = sz >= 205
    fb0 = fb2 if use_double else fb1
    print(
        f"Bit length {sz} default factor base size {fb0} (double large prime = {use_double})"
    )
    step = int(fb0 * sqrt(log(fb0)) / 25)
    # Generate fixed numbers for the various factor base size
    # (otherwise the comparison does not make sense)
    ns = [get_prime(sz // 2) * get_prime(sz // 2) for _ in range(count)]
    for fb in (fb0 - 2 * step, fb0 - step, fb0, fb0 + step, fb0 + 2 * step):
        # Add threads to speedup large numbers.
        if sz < 225:
            threads = 1
        else:
            threads = 2
        try:
            t = time.time()
            for n in ns:
                pymqs.factor(
                    n,
                    qs_fb_size=fb,
                    qs_use_double=use_double,
                    algo="qs",
                    threads=threads,
                )
            elapsed = time.time() - t
            avg = elapsed / float(len(ns))
            print(
                f"{sz} bit FB={fb} inputs={count} threads={threads} avgtime={avg:.3}s"
            )
        except Exception as e:
            print(e)
