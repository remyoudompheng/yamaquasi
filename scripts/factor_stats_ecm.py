"""
Distribution of factors of Suyama-11 curve orders
"""

import math
import random
import time

import cypari2
import gmpy2
import pymqs

pari = cypari2.Pari()
pari.allocatemem(32_000_000)

CUMUL = True

for size in (64, 80, 96, 112, 128):
    SAMPLES = 100000
    stats = [[0 for _ in range(16)] for _ in range(8)]
    t0 = time.monotonic()
    for _ in range(SAMPLES):
        l = gmpy2.next_prime(random.getrandbits(size - 1) | (1 << (size - 1)))
        # Rescaled Suyama σ=11 curve
        E = pari.ellinit([0, -679442, 0, 71973621841, 0], l)
        order = int(pari.ellcard(E))
        assert order % 12 == 0

        facs = pymqs.factor(order)
        p1 = facs[-1]
        p2 = facs[-2]

        r1 = min(8, math.floor(math.log(l) / math.log(p1))) - 1
        r2 = min(16, math.floor(math.log(l) / math.log(p2))) - 1
        stats[r1][r2] += 1

    print(size, SAMPLES, time.monotonic() - t0)
    if CUMUL:
        cumul = [[0 for _ in range(16)] for _ in range(8)]
        for i in range(8):
            for j in range(i, 16):
                cumul[i][j] = sum(sum(row[j:]) for row in stats[i:])

        for i, row in enumerate(cumul):
            print(f"{i + 1:>2} {row}")
    else:
        for i, row in enumerate(stats):
            print(f"{i + 1:>2} {sum(row):>6} {row}")
