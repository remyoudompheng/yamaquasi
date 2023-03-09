"""
Explore factor base sizes around a central value for optimal performance
according to input size.

When factor base size is optimal, the cost function is usually "flat"
enough so that a relative difference of 10-20% still has a similar
performance, giving a large interval of suitable sizes.

If the factor base is not optimal, the performance difference
between probed parameters can reach 10%-20%.

The script generates a set of semiprime numbers for each size and factors
that fixed set of numbers with several parameters. Although the performance
may vary depending on arithmetical properties and CPU architecture,
the ideal factor base size range should not vary too much (at most 20-30%),
even if the underlying times have large variance.
"""

from math import log, sqrt
from random import randint
import time

from gmpy2 import next_prime

import pymqs

SIZES = [
    # Bit length, Factor base size, FB size for 2x large prime, Samples
    (70, 100, 60, 500),
    (80, 130, 70, 500),
    (90, 160, 90, 500),
    (100, 200, 120, 500),
    (110, 300, 150, 500),
    (120, 400, 200, 500),
    (130, 550, 250, 400),
    (140, 750, 300, 300),
    (150, 1000, 380, 200),
    (160, 1400, 500, 150),
    (170, 2000, 800, 100),
    (180, 3000, 1300, 60),
    (190, 5000, 2000, 40),
    (200, 8000, 3000, 25),
    (210, 11000, 4500, 15),
    (220, 15000, 7000, 10),
    (230, 20000, 10000, 8),
    # Double large primes activated by default
    (240, 26000, 14000, 6),
    (250, 35000, 18000, 4),
    (260, 40000, 23000, 4),
    (270, 48000, 30000, 4),
    (280, 60000, 40000, 3),
    (290, 75000, 55000, 3),
    (300, 90000, 75000, 3),
    (310, 110000, 90000, 3),
    (320, 140000, 110000, 3),
    (330, 200000, 130000, 3),
    (340, 260000, 170000, 3),
    (350, 330000, 220000, 3),
    (360, 400000, 280000, 3),
]


def get_prime(bits):
    return int(next_prime(randint(2 ** (bits - 1), 2**bits)))


for sz, fb1, fb2, count in SIZES:
    # Generate fixed numbers for the various factor base size
    # (otherwise the comparison does not make sense)
    ns = [get_prime(sz // 2) * get_prime(sz // 2) for _ in range(count)]
    use_double = sz > 224
    fb0 = fb2 if use_double else fb1
    print(
        f"Bit length {sz} default factor base size {fb0} (double large prime = {use_double})"
    )
    step = int(fb0 * sqrt(log(fb0)) / 25)
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
                    algo="mpqs",
                    threads=threads,
                )
            elapsed = time.time() - t
            avg = elapsed / float(len(ns))
            print(
                f"{sz} bit FB={fb} inputs={count} threads={threads} avgtime={avg:.3}s"
            )
        except Exception as e:
            print(e)
