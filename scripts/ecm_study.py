import random
from random import getrandbits
from sage.all import factor, primes, proof, GF, EllipticCurve, QQ
from math import gcd

proof.arithmetic(False)

def analyze(primes, x, y, expect=1):
    stats1, stats2 = [], []
    exp2, exp3 = 0, 0
    t1, t2 = 0, 0
    for p in primes:
        K = GF(p)
        d = K(x * x + y * y - 1) / K(x * x * y * y)
        c = (1 - d) / 4
        E = EllipticCurve(GF(p), [0, 1 - 2 * c, 0, c * c, 0])
        fs = factor(E.order())
        assert E.order() % expect == 0
        d1, d2, *_ = E.abelian_group().invariants() + (0,)
        t1 = gcd(t1, d1)
        t2 = gcd(t2, d2)
        if len(fs) == 1:
            max1, max2 = fs[0][0], fs[0][0]
        else:
            max1 = sorted(p**k for p, k in fs)[-2]
            max2 = max(p for p, _ in fs)
        stats1.append(max1)
        stats2.append(max2)
        exp2 += E.order().valuation(2)
        exp3 += E.order().valuation(3)

    stats1.sort()
    stats2.sort()
    print(f"Generic torsion: ({t1}, {t2})")
    l = len(primes)
    exp2, exp3 = float(exp2), float(exp3)
    print(f"avg exponent of 2: {exp2/l:.3f}")
    print(f"avg exponent of 3: {exp3/l:.3f}")
    p25, p50 = stats1[l // 4], stats1[-l // 2]
    p66, p75 = stats1[-l // 3], stats1[-l // 4]
    print(f"needs B1 {p25=} {p50=} {p66=} {p75=}")
    p25, p50 = stats2[l // 4], stats2[-l // 2]
    p66, p75 = stats2[-l // 3], stats2[-l // 4]
    print(f"needs B2 {p25=} {p50=} {p66=} {p75=}")

for size in (32, 48, 64):
    print(f"=== ===")
    print(f"=== Working with primes of size {size} ===")
    print(f"=== ===")
    p0 = getrandbits(size)
    ps = random.sample(list(primes(p0, p0 + 1e6)), 2000)

    # Basic Edwards curves
    # Average exp2 = 3 + 2/3
    # Average exp3 = 2/3
    for y in (3, 4, 5, 6):
        print(f"=== Trying Edwards curve through (2, {y})")
        analyze(ps, 2, y, expect=4)

    # Extra 2-torsion (Z/2 x Z/4) when d is a square
    # (3a+5)² + (4a+5)² = 1 + (5a+7)²
    # Average exp2 = 4 + 1/3
    # Average exp3 = 2/3
    for a in (1, 2, 3):
        x, y = 3*a+5, 4*a+5
        print(f"=== Trying Edwards curve Z/2 x Z/4 ({x}, {y})")
        analyze(ps, x, y, expect=8)

    # Good curve Z/12
    # Average exp2 = 3 + 2/3
    # Average exp3 = 1 + 2/3
    x, y = QQ("5/23"), QQ("-1/7")
    print(f"=== Trying Edwards curve Z/12 ({x}, {y})")
    analyze(ps, x, y, expect=12)
    x, y = QQ("8/17"), QQ("20/19")
    print(f"=== Trying Edwards curve Z/12 ({x}, {y})")
    analyze(ps, x, y, expect=12)

    # Good curve 2x8
    # Average exp2 = 5 + 1/3
    # Average exp3 = 2/3
    x, y = QQ("17/19"), QQ("17/33")
    print(f"=== Trying Edwards curve Z/2 x Z/8 ({x}, {y})")
    analyze(ps, x, y, expect=16)
    x, y = QQ("125/91"), QQ("841/791")
    print(f"=== Trying Edwards curve Z/2 x Z/8 ({x}, {y})")
    analyze(ps, x, y, expect=16)

