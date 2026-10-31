"""
Distribution of factors of uniform numbers
"""

import math
import random

import pymqs


for size in (64, 80, 96, 112, 128, 160, 192):
    if size < 100:
        SAMPLES = 100000
    else:
        SAMPLES = 10000
    stats2 = 16 * [0]
    stats3 = 16 * [0]
    # Deciles of the semiprime part
    statsS = 10 * [0]
    for _ in range(SAMPLES):
        n = random.getrandbits(size - 1) | (1 << (size - 1))
        facs = pymqs.factor(n)
        p1 = facs[-1]
        p2 = facs[-2] if len(facs) >= 2 else 2
        p3 = facs[-3] if len(facs) >= 3 else 2
        ratio = math.log(n) / math.log(p2)
        stats2[min(15, math.floor(ratio))] += 1
        sratio = math.log(p1 * p2) / math.log(n)
        statsS[min(9, math.floor(10 * sratio))] += 1
        ratio = math.log(n) / math.log(p3)
        stats3[min(15, math.floor(ratio))] += 1
    print(size)
    print(stats2)
    print([sum(stats2[i:]) for i in range(1, 11)])
    print(stats3)
    print([sum(stats3[i:]) for i in range(1, 11)])
    print(statsS)
    print([sum(statsS[i:]) for i in range(10)])
