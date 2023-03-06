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
    (110, 64, 80),
    (120, 128, 40),
    (130, 192, 40),
    (140, 256, 20),
    (150, 320, 10),
    (160, 384, 10),
    (170, 448, 10),
    (180, 512, 8),
    (190, 768, 6),
    (200, 1024, 4),
    (210, 1280, 4),
    (220, 1536, 3),
    (230, 1792, 3),
    # Double large prime starts here
    (240, 1536, 3),
    (250, 1664, 2),
    (260, 1792, 2),
    (270, 1920, 2),
    (280, 2048, 2),
    (290, 3072, 2),
    (300, 4096, 2),
    (310, 5120, 2),
    (320, 6144, 2),
    (330, 7168, 2),
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
                for i in (-isize // 128, 0, isize // 128, isize // 64, 1, 2, 4, 8)
                if isize + 32 * i > 0
            ]
        )
    )
    for isz in isizes:
        # Add threads to speedup large numbers.
        if sz < 205:
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
                    n, qs_interval_size=isz * 1024, algo="mpqs", threads=threads
                )
            elapsed = time.time() - t
            avg = elapsed / float(len(ns))
            print(
                f"{sz} bit interval={isz}k inputs={count} threads={threads} avgtime={avg:.3}s"
            )
        except Exception as e:
            print(e)
