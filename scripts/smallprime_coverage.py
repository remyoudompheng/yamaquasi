"""
Study coverage of P-1 and ECM over small primes

We only consider twisted Edwards, using equivalent Montgomery form.
"""

import random
from math import gcd, isqrt

import cypari2
import gmpy2
import numpy

# SAMPLES = 10000
SAMPLES = 1000
MOD4 = None


def primerange(B1, B2):
    B1 = max(B1, 2)
    s = numpy.ones(B2, dtype=numpy.uint8)
    for l in range(2, B2):
        if s[l] == 1:
            s[2 * l :: l] = 0
        if l * l > B2:
            break
    return [int(l) for l in s.nonzero()[0] if l >= B1]


def primesamples(pbits):
    if pbits < 18 and SAMPLES > 1000:
        ps = list(primerange(2 ** (pbits - 1), 2**pbits))
    else:
        ps = [
            int(gmpy2.next_prime(random.randrange(2 ** (pbits - 1), 2**pbits)))
            for _ in range(3 * SAMPLES)
        ]

    if MOD4:
        ps = [l for l in ps if l & 3 == MOD4]
    return ps[:SAMPLES]


def make_params(B1, B2):
    ps = primerange(0, B1)
    prodB1 = 1
    for p in ps:
        pk = p
        while pk * p < B1:
            pk *= p
        prodB1 *= pk
    return prodB1, list(primerange(B1, B2))


pari = cypari2.Pari()


def rho_coverage(primes, iters):
    c = 2
    good = 0
    for l in primes:
        ok = False
        u = pari.Mod(2, l)
        for _ in range(iters):
            u = u * u + c
        v = u
        for _ in range(iters):
            v = v * v + c
            if v == u:
                ok = True
                break
        good += int(ok)

    return good / len(primes)


primes = primerange(0, 500_000)


def pm1_params(budget, verbose=False):
    # Cost is 2 B1 + 1 (π(B2)-π(B1))
    # Stage 1 is 1.44 B1 bits and 1.5 MUL/bit
    # Allocate half to stage1 and half to stage2
    params = {
        100: (40, 200),
        200: (40, 700),
        500: (130, 1600),
        1000: (250, 4000),
        2000: (320, 12000),
        4000: (500, 28_000),
        8000: (1500, 50_000),
        16000: (2000, 130_000),
        32000: (3000, 300_000),
        50000: (9000, 400_000),
    }
    B1, B2 = params[budget]
    if verbose:
        print(f"PM1 budget={budget} B1={B1} B2={B2}")
    return make_params(B1, B2)


def pm1_coverage(primes, prodB1, p2s):
    good = 0
    for l in primes:
        lz = gmpy2.mpz(l)
        g = pow(gmpy2.mpz(2), prodB1, lz)
        ok = any(pow(g, e, lz) == 1 for e in p2s)
        good += int(ok)

    return good / len(primes)


def ecm_params(budget, verbose=False):
    # Cost = 13.7 B1 + stage2 cost
    # B2=660 costs ~30 point adds + 100 muls = 340 muls
    # B2=1920 costs ~40 point adds + 144 muls = 450 muls
    # B2=5040 costs about 1000 MUL
    # B2=7700 costs about 1500 MUL
    # B2=13e3 costs about 2400 MUL
    # B2=20e3 costs about 3200 MUL
    # B2=53e3 costs about 7800 MUL
    # B2=126e3 costs about 16000 MUL
    # B2=181e3 costs about 23000 MUL
    params = {
        100: (10, 10),
        200: (15, 15),
        500: (10, 660),
        1000: (50, 1920),
        2000: (75, 5040),
        4000: (120, 13_200),
        8000: (280, 33_000),
        16000: (360, 126_000),
        32000: (660, 181_000),
        50000: (2000, 181_000),
    }
    if budget in params:
        B1, B2 = params[budget]
    if verbose:
        print(f"ECM budget={budget} B1={B1} B2={B2}")
    return make_params(B1, B2)


def ecm_coverage(primes, snum, sden, prodB1, p2s):
    good = 0
    for l in primes:
        # Suyama curve
        sigma = pari.Mod(snum, l) / pari.Mod(sden, l)
        u = pari.Mod(sigma**2 - 5, l)
        v = pari.Mod(4 * sigma, l)
        A = (v - u) ** 3 * (3 * u + v) / (4 * u**3 * v) - 2
        x0 = u**3 / v**3
        # x0^2 + A x0 + 1 = ((s²-1)(s²-25)(s²+5)/v^3)**2
        B = x0
        y0 = (u + 4) * (u - 20) * (u + 10) / v**3
        assert B * y0**2 == x0**3 + A * x0**2 + x0
        # B y^2 = x^3 + A x^2 + x
        # <=> (y/B)^2 = (x/B)^3 + (A/B) (x/B)^2 + (1/B²) x/B
        E = pari.ellinit([0, A / B, 0, 1 / B**2, 0], l)
        if not E:
            # ill defined curve
            continue
        G = [x0 / B, y0 / B]
        assert pari.ellisoncurve(E, G)
        G1 = pari.ellmul(E, G, prodB1)
        if len(p2s) < 100:
            ok = any(pari.ellmul(E, G1, e) == 0 for e in p2s)
        else:
            r = 6 * isqrt(p2s[-1])
            gs = set(pari.ellmul(E, G1, e)[0] for e in range(0, p2s[-1], r))
            ok = any(pari.ellmul(E, G1, e)[0] in gs for e in range(1, r // 2) if gcd(e, r) == 1)

        good += int(ok)

    return good / len(primes)


for budget in (100, 200, 500, 1000, 2000, 4000, 8000, 16000):
    pm1_params(budget, verbose=True)
for budget in (100, 200, 500, 1000, 2000, 4000, 8000, 16000):
    ecm_params(budget, verbose=True)

# P-1 parameter sweep
if False:
    print("PM1 200")
    for pbits in (14, 16, 18, 20):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            pm1_coverage(ps, *make_params(25, 900)),
            pm1_coverage(ps, *make_params(40, 700)), # Best
            pm1_coverage(ps, *make_params(60, 500)),
        )
    # Budget 500
    for pbits in (14, 16, 18, 20, 22):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            pm1_coverage(ps, *make_params(70, 2400)),
            pm1_coverage(ps, *make_params(100, 2000)),
            pm1_coverage(ps, *make_params(130, 1600)),  # Best
            pm1_coverage(ps, *make_params(170, 1000)),
        )
    # Budget 1000
    for pbits in (16, 18, 20, 22, 24):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            pm1_coverage(ps, *make_params(200, 5000)),
            pm1_coverage(ps, *make_params(250, 4000)),  # Best
            pm1_coverage(ps, *make_params(300, 3000)),
            pm1_coverage(ps, *make_params(350, 2000)),
        )
    # Budget 2000
    for pbits in (18, 20, 22, 24, 26):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            pm1_coverage(ps, *make_params(120, 15000)),
            pm1_coverage(ps, *make_params(180, 14000)),
            pm1_coverage(ps, *make_params(240, 13000)),
            pm1_coverage(ps, *make_params(320, 12000)),  # Best
            pm1_coverage(ps, *make_params(500, 8000)),
        )
    # Budget 4000
    for pbits in (20, 22, 24, 26):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            pm1_coverage(ps, *make_params(300, 32000)),
            pm1_coverage(ps, *make_params(500, 28000)),  # Best
            pm1_coverage(ps, *make_params(1000, 18000)),
        )

# ECM parameter sweep
if False:
    print("ECM 500")
    for pbits in (16, 18, 20):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            ecm_coverage(ps, 11, 1, *make_params(50, 50)),
            ecm_coverage(ps, 11, 1, *make_params(10, 660)),  # Best
        )
    print("ECM 1000")
    for pbits in (16, 18, 20, 22, 24):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            ecm_coverage(ps, 11, 1, *make_params(50, 660)),
            ecm_coverage(ps, 11, 1, *make_params(40, 1920)),  # Best
            ecm_coverage(ps, 11, 1, *make_params(7, 5040)),
        )
    print("ECM 2000")
    for pbits in (18, 20, 22, 24, 26):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            ecm_coverage(ps, 11, 1, *make_params(120, 660)),
            ecm_coverage(ps, 11, 1, *make_params(110, 1920)),
            ecm_coverage(ps, 11, 1, *make_params(75, 5040)),  # Best
            ecm_coverage(ps, 11, 1, *make_params(35, 7700)),
        )
    print("ECM 4000")
    for pbits in (20, 22, 24, 26):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            ecm_coverage(ps, 11, 1, *make_params(220, 5040)),
            ecm_coverage(ps, 11, 1, *make_params(180, 7700)),  # Best
            ecm_coverage(ps, 11, 1, *make_params(120, 13200)),
            ecm_coverage(ps, 11, 1, *make_params(60, 20000)),
        )


print(
    "pbits\tbudget\trho\tpm1\tecm\tcost(rho)\tcost(ecm)"
)  # + "\trgain\tpgain\tegain1")
for pbits in (10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36):
    ps = sorted(primesamples(pbits))
    assert ps[0] > 3

    best = None
    pm1_scores = {}

    for budget in (100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 50000):
        if pbits <= 16 and budget > 2000:
            continue
        if pbits <= 20 and budget > 8000:
            continue
        if pbits <= 26 and budget > 16000:
            continue
        if pbits <= 32 and budget > 32000:
            continue
        if pbits > 20 and budget < 500:
            continue
        if pbits > 22 and budget < 1000:
            continue
        if pbits > 28 and budget < 2000:
            continue
        rscore = rho_coverage(ps, budget // 3)
        prodB1, p2s = pm1_params(budget)
        pscore = pm1_coverage(ps, prodB1, p2s)
        prodB1, p2s = ecm_params(budget)
        escore = ecm_coverage(ps, 11, 1, prodB1, p2s)
        # escore = ecm_coverage(ps, 9, 4, prodB1, p2s)
        line = f"{pbits}\t{budget}\t{rscore:.3f}\t{pscore:.3f}\t{escore:.3f}"

        # Effectiveness in bits
        cost_rho = int(budget / rscore) if rscore > 0.01 else float("inf")
        cost_ecm = int(budget / escore) if escore > 0.01 else float("inf")
        line += f"\t{cost_rho}\t{cost_ecm}"

        pm1_scores[budget] = pscore
        if best is None or best[0] > cost_rho:
            best = (cost_rho, f"rho({budget})")
        if best is None or best[0] > cost_ecm:
            best = (cost_ecm, f"ecm({budget})")

        print(line)

    # Compute best strategy
    strategy = min(
        (pcost + (1 - p) * best[0], f"PM1({pcost})+{best[1]}")
        for pcost, p in pm1_scores.items()
    )
    print(f"{pbits} bits => {strategy[1]} cost={strategy[0]:.0f}")
