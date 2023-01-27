"""
Study prime factors of Di(y)-Di(x) where Di are polynomials.
"""

from math import gcd
from sage.all import chebyshev_T, ZZ, QQ, primes, euler_phi, prime_pi
from pymqs import factor
import random
import time

D = []
Zt = ZZ["t"]
for i in range(0, 256):
    x = QQ["x"].gen()
    pi = 2 * chebyshev_T(i, x / 2)
    D.append(Zt(pi))

Zta = Zt["a"]


def delta(p):
    t = Zt.gen()
    a = Zta.gen()
    return p(t=t + a) - p


def delta_k(p, k):
    for _ in range(k):
        p = delta(p)
    return p


for i in (2, 3, 4, 6, 12):
    print(f"Dickson degree {i}: {D[i]}")
    print("Start index 0, step a")
    for j in range(1, i + 1):
        print(f"diff^{j}", delta_k(D[i], j)(t=0))
    print("Start index 1, step 6")
    for j in range(1, i + 1):
        print(f"diff^{j}", delta_k(D[i], j)(t=1, a=6))

# Validate delta.
def test_d4():
    # D4 = t^4 - 4*t^2 + 2
    a = 71
    b = 12 * a**4
    x = 2
    dx = a**4 - 4 * a**2
    d2x = b + 2 * dx  # 14*a^4 - 8*a^2
    d3x = 3 * b
    d4x = 2 * b
    for j in range(1000):
        x += dx
        dx += d2x
        d2x += d3x
        d3x += d4x
        assert x == D[4](t=71 * (j + 1))


def test_d6():
    # Finite differences at t=0 for step=a
    # D6 = t^6 - 6*t^4 + 9*t^2 - 2
    # diff^1 a^6 - 6*a^4 + 9*a^2
    # diff^2 62*a^6 - 84*a^4 + 18*a^2
    # diff^3 540*a^6 - 216*a^4
    # diff^4 1560*a^6 - 144*a^4
    # diff^5 1800*a^6
    # diff^6 720*a^6
    a = 71
    b1 = a**6 - 6 * a**4 + 9 * a**2
    b2 = 360 * a**6
    b3 = 60 * a**6 - 72 * a**4
    x = -2
    dx = b1
    d2x = b3 + 2 * b1
    d3x = b2 + 3 * b3
    d4x = 4 * b2 + 2 * b3
    d5x = 5 * b2
    d6x = 2 * b2
    for j in range(1000):
        x += dx
        dx += d2x
        d2x += d3x
        d3x += d4x
        d4x += d5x
        d5x += d6x
        assert x == D[6](t=71 * (j + 1))


test_d4()
test_d6()

# Study the P-1 case: polynomials (even degree) are evaluated on giant steps
# and baby steps until d/2. Then G - B can find a hidden prime divisor.

for d, k in [(2310, 3 * 256), (9240, 3 * 1024), (39270, 3 * 4096)]:
    primecount = prime_pi(d * k)
    allpairs = k * int(euler_phi(d))
    print(f"==> P-1 stage 2 with B2={float(d*k):.2e} ({d}*{k}, {primecount} primes)")

    for i in (4, 6, 8, 12):
        if i > 10 and d > 4000:
            continue
        print("")
        for poly in (D[i], Zt.gen() ** i):
            print(f"Degree {i}: {poly}")
            gs = [int(poly(x)) for x in range(d, d * k + 1, d)]
            bs = [int(poly(x)) for x in range(d // 2) if gcd(x, d) == 1]
            extra = set()
            # Don't test all giant steps
            pairs = 0
            t0 = time.time()
            for idx in range(10000):
                g, b = random.choice(gs), random.choice(bs)
                for f in factor(g - b, timeout=1):
                    if f > d * k:
                        extra.add(f)
                pairs += 2
                if idx % 1000 == 100:
                    pass  # print("typical", g - b, g + b)
                if time.time() > t0 + 10:
                    pass  # print("typical", g - b, g + b)
                    break
            extra = sorted(extra)
            lex = len(extra)
            pairs = pairs / (allpairs / primecount)
            print(
                f"{lex}/{pairs:.1f} extra primes {extra[0]}..{extra[lex//3]}..{extra[2*lex//3]}..{extra[-1]}"
            )
            extra10 = len([e for e in extra if e < 10 * d * k])
            print(f"{100 * extra10 / pairs:.1f}% gain in [B2, 10 B2] range")
            extra100 = len([e for e in extra if e < 100 * d * k])
            print(f"{100 * extra100 / pairs:.1f}% gain in [B2, 100 B2] range")
            extra1000 = len([e for e in extra if e < 1000 * d * k])
            print(f"{100 * extra1000 / pairs:.1f}% gain in [B2, 1000 B2] range")

# Study the ECM case: polynomials are evaluated on giant steps
# and baby steps until d/2. Then both g-b and g+b can find the hidden prime.

for d, k in [(2310, 3 * 256), (9240, 3 * 1024), (39270, 3 * 4096)]:
    primecount = prime_pi(d * k)
    allpairs = k * int(euler_phi(d))
    print(f"==> ECM stage 2 with B2={float(d*k):.2e} ({d}*{k}, {primecount} primes)")

    for i in (2, 3, 6, 8, 12):
        if i > 10 and d > 4000:
            continue
        print("")
        for poly in (D[i], Zt.gen() ** i):
            print(f"Degree {i}: {poly}")
            gs = [int(poly(x)) for x in range(d, d * k + 1, d)]
            bs = [int(poly(x)) for x in range(d // 2) if gcd(x, d) == 1]
            extra = set()
            # Don't test all giant steps
            pairs = 0
            t0 = time.time()
            for idx in range(10000):
                g, b = random.choice(gs), random.choice(bs)
                for f in factor(g - b, timeout=1):
                    if f > d * k:
                        extra.add(f)
                for f in factor(g + b, timeout=1):
                    if f > d * k:
                        extra.add(f)
                pairs += 2
                if idx % 1000 == 100:
                    pass  # print("typical", g - b, g + b)
                if time.time() > t0 + 10:
                    pass  # print("typical", g - b, g + b)
                    break
            extra = sorted(extra)
            lex = len(extra)
            pairs = pairs / (allpairs / primecount)
            print(
                f"{lex}/{pairs:.1f} extra primes {extra[0]}..{extra[lex//3]}..{extra[2*lex//3]}..{extra[-1]}"
            )
            extra10 = len([e for e in extra if e < 10 * d * k])
            print(f"{100 * extra10 / pairs:.1f}% gain in [B2, 10 B2] range")
            extra100 = len([e for e in extra if e < 100 * d * k])
            print(f"{100 * extra100 / pairs:.1f}% gain in [B2, 100 B2] range")
            extra1000 = len([e for e in extra if e < 1000 * d * k])
            print(f"{100 * extra1000 / pairs:.1f}% gain in [B2, 1000 B2] range")
