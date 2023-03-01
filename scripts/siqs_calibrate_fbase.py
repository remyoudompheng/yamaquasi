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

# The following sizes are selected so that running the script
# shows that it is nearly optimal.
SIZES = [
    # Bit length, Factor base size, FB size for 2x large prime, Samples
    (70, 100, 80, 600),
    (80, 150, 100, 400),
    (90, 200, 120, 400),
    (100, 230, 150, 100),
    (110, 280, 200, 80),
    (120, 400, 250, 40),
    (130, 450, 300, 40),
    (140, 600, 360, 20),
    (150, 800, 480, 10),
    (160, 1000, 500, 10),
    (170, 1500, 800, 10),
    (180, 3000, 1800, 8),
    (190, 5000, 2000, 7),
    (200, 6000, 3000, 6),
    (210, 10000, 4500, 5),
    (220, 15000, 7000, 4),
    (230, 17000, 10000, 4),
    (240, 22000, 15000, 4),
    (250, 28000, 18000, 4),
    # Double large primes activated by default
    (260, 37000, 23000, 4),
    (270, 40000, 30000, 4),
    (280, 50000, 40000, 3),
    (290, 60000, 50000, 3),
    (300, 70000, 60000, 3),
    (310, 90000, 80000, 3),
    # Extrapolated
    (330, 300000, 130000, 3),
    (360, 500000, 280000, 3),
]


def get_prime(bits):
    return int(next_prime(randint(2 ** (bits - 1), 2**bits)))


for sz, fb1, fb2, count in SIZES:
    use_double = sz >= 256
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
                    algo="siqs",
                    threads=threads,
                )
            elapsed = time.time() - t
            avg = elapsed / float(len(ns))
            print(
                f"{sz} bit FB={fb} inputs={count} threads={threads} avgtime={avg:.3}s"
            )
        except Exception as e:
            print(e)
