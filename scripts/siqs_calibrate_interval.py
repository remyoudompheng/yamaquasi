"""
Run SIQS with a range of interval sizes to evaluate preferred parameters.

Interval size can affect performance wildly depending on the size of CPU caches
(for each level of the cache hierarchy) and the number of threads (that may
or may not share cache levels), and depending on mathematical properties
(smaller intervals have a higher density of smooth numbers).

Results may vary considerably depending on CPU architecture and
selected parallelism.

However, if the optimal interval size is very small, the result
is probably valid on CPUs with a small L3 cache (less than 2MB-4MB).

The exact running times can vary a lot depending on the input integer,
but comparing several interval times for a fixed set of integers is expected
to produce consistent results about the relative performance differences.
"""

from random import randint
import time
from gmpy2 import next_prime

import pymqs

SIZES = [
    # Bit length, Interval size / 1024, Samples
    (70, 32, 600),
    (80, 32, 400),
    (90, 32, 400),
    (100, 32, 100),
    (110, 32, 80),
    (120, 32, 40),
    (130, 32, 40),
    (140, 32, 20),
    (150, 32, 10),
    (160, 32, 10),
    (170, 32, 10),
    (180, 32, 8),
    (190, 64, 6),
    (200, 96, 4),
    (210, 128, 4),
    (220, 160, 3),
    (230, 192, 3),
    (240, 256, 3),
    (250, 288, 2),
    # Double large prime starts here
    (260, 256, 2),
    (270, 288, 2),
    (280, 320, 2),
    (290, 352, 2),
    (300, 384, 2),
    (310, 416, 2),
    (320, 480, 2),
    (330, 512, 2),
]


def get_prime(bits):
    return int(next_prime(randint(2 ** (bits - 1), 2**bits)))


for sz, isize, count in SIZES:
    print(f"Bit length {sz} default interval size {isize}k")
    # Generate fixed numbers (otherwise the comparison does not make sense)
    ns = [get_prime(sz // 2) * get_prime(sz // 2) for _ in range(count)]
    isizes = sorted(
        set(
            [
                isize + 32 * i
                for i in (-4, -1, -isize // 128, 0, isize // 128, isize // 64, 1, 4)
                if isize + 32 * i > 0
            ]
        )
    )
    for isz in isizes:
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
                    n, qs_interval_size=isz * 1024, algo="siqs", threads=threads
                )
            elapsed = time.time() - t
            avg = elapsed / float(len(ns))
            print(
                f"{sz} bit interval={isz}k inputs={count} threads={threads} avgtime={avg:.3}s"
            )
        except Exception as e:
            print(e)
