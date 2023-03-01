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
    # The parameters follow roughly a Fibonacci relation
    (70, 100, 80, 400),
    (80, 150, 100, 300),
    (90, 200, 200, 200),
    (100, 280, 250, 100),
    (110, 500, 350, 40),
    (120, 800, 600, 20),
    (130, 1200, 800, 20),
    (140, 2000, 1300, 10),
    (150, 3200, 2000, 8),
    (160, 5500, 3000, 6),
    (170, 8000, 5000, 5),
    (180, 13000, 8000, 4),
    (190, 18000, 13000, 3),
    (200, 30000, 20000, 2),
    # Double large primes activated
    (210, 50000, 25000, 2),
    (220, 90000, 45000, 2),
    # Extrapolated
    (230, 180000, 80000, 2),
    (240, 240000, 120000, 2),
    (250, 400000, 200000, 2),
    (260, 600000, 300000, 3),
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
        elif sz < 255:
            threads = 2
        elif sz < 285:
            threads = 3
        else:
            threads = 4
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
