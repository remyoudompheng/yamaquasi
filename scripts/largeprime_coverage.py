"""
Study coverage of P-1 and ECM over small primes

We only consider twisted Edwards, using equivalent Montgomery form.
"""

import random
from math import gcd, isqrt

import cypari2
import gmpy2
import numpy

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
    return prodB1, B2


pari = cypari2.Pari()
pari.allocatemem(32_000_000)

primes = primerange(0, 500_000)


def pm1_params(budget, verbose=False):
    # Cost is 2 B1 + 1 (π(B2)-π(B1))
    # Polyeval is useful for B2 > 500k
    params = {
        4000: (500, 28_000),
        8000: (2000, 40_000),
        16000: (3000, 100_000),
        32000: (8000, 200_000),
        50000: (12000, 350_000),
        100000: (15000, 4_000_000),
        200000: (30000, 15_000_000),
        500000: (50000, 120_000_000),
    }
    B1, B2 = params[budget]
    if verbose:
        print(f"PM1 budget={budget} B1={B1} B2={B2}")
    return make_params(B1, B2)


def pm1_coverage(primes, prodB1, B2):
    good = 0
    r = 6 * (isqrt(B2) // 6)
    for l in primes:
        lz = gmpy2.mpz(l)
        g = pow(gmpy2.mpz(2), prodB1, lz)

        gbig = set(int(pow(g, e, lz)) for e in range(0, B2, r))
        g1, g5, g6 = g, pow(g, 5, lz), pow(g, 6, lz)
        ok = False
        for _ in range(0, r, 6):
            g1 = (g1 * g6) % lz
            g5 = (g5 * g6) % lz
            if g1 in gbig or g5 in gbig:
                ok = True
                break
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
        500: (40, 40),
        1000: (50, 1920),
        2000: (75, 5040),
        4000: (120, 13_200),
        8000: (280, 33_000),
        16000: (360, 126_000),
        32000: (660, 181_000),
        50000: (2000, 181_000),
        100000: (4000, 450_000),
        200000: (7000, 1_200_000),
        500000: (12000, 6_000_000),
    }
    if budget in params:
        B1, B2 = params[budget]
    if verbose:
        print(f"ECM budget={budget} B1={B1} B2={B2}")
    return make_params(B1, B2)


def ecm_coverage(primes, snum, sden, prodB1, B2):
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
        r = 6 * isqrt(B2)
        gs = set(pari.ellmul(E, G1, e)[0] for e in range(r, B2, r))
        ok = any(
            pari.ellmul(E, G1, e)[0] in gs for e in range(1, r // 2) if gcd(e, r) == 1
        )
        good += int(ok)

    return good / len(primes)


for budget in (4000, 8000, 16000):
    pm1_params(budget, verbose=True)
for budget in (4000, 8000, 16000):
    ecm_params(budget, verbose=True)

# P-1 parameter sweep
if False:
    print("PM1 8000")
    for pbits in (32, 36, 40, 48):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            pm1_coverage(ps, *make_params(1000, 50000)),
            pm1_coverage(ps, *make_params(2000, 40000)),  # Best
            pm1_coverage(ps, *make_params(3000, 30000)),
        )
    print("PM1 16000")
    for pbits in (32, 36, 40, 48, 56):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            pm1_coverage(ps, *make_params(2000, 110000)),
            pm1_coverage(ps, *make_params(3000, 100000)),  # Best
            pm1_coverage(ps, *make_params(4000, 90000)),
        )
    print("PM1 32000")
    for pbits in (32, 40, 48, 56):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            pm1_coverage(ps, *make_params(4000, 250_000)),
            pm1_coverage(ps, *make_params(6000, 220_000)),
            pm1_coverage(ps, *make_params(8000, 200_000)),  # Best
            pm1_coverage(ps, *make_params(10000, 180_000)),
        )
    print("PM1 50000")
    for pbits in (32, 40, 48, 56, 64):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            pm1_coverage(ps, *make_params(8000, 400_000)),
            pm1_coverage(ps, *make_params(10000, 380_000)),
            pm1_coverage(ps, *make_params(12000, 350_000)),  # Best
            pm1_coverage(ps, *make_params(15000, 300_000)),
        )
    print("PM1 100000")
    for pbits in (32, 40, 48, 56, 64):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            pm1_coverage(ps, *make_params(10000, 5_000_000)),
            pm1_coverage(ps, *make_params(15000, 4_000_000)),  # Best
            pm1_coverage(ps, *make_params(20000, 3_000_000)),
        )
# ECM parameter sweep
if False:
    print("ECM 4000")
    for pbits in (28, 32, 36, 40):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            ecm_coverage(ps, 11, 1, *make_params(220, 5040)),
            ecm_coverage(ps, 11, 1, *make_params(180, 7700)),  # Best
            ecm_coverage(ps, 11, 1, *make_params(120, 13200)),
            ecm_coverage(ps, 11, 1, *make_params(60, 20000)),
        )
    print("ECM 16000")
    for pbits in (32, 36, 40, 48):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            ecm_coverage(ps, 11, 1, *make_params(1000, 20000)),
            ecm_coverage(ps, 11, 1, *make_params(800, 33000)),
            ecm_coverage(ps, 11, 1, *make_params(600, 53000)),  # Best
            ecm_coverage(ps, 11, 1, *make_params(400, 81000)),
        )
    print("ECM 100k")
    for pbits in (32, 40, 48):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            ecm_coverage(ps, 11, 1, *make_params(3000, 600000)),
            ecm_coverage(ps, 11, 1, *make_params(4000, 450000)),  # Best
            ecm_coverage(ps, 11, 1, *make_params(5000, 350000)),
            ecm_coverage(ps, 11, 1, *make_params(6000, 200000)),
        )
    print("ECM 200k")
    for pbits in (40, 48, 56, 64):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            ecm_coverage(ps, 11, 1, *make_params(5000, 1500_000)),
            ecm_coverage(ps, 11, 1, *make_params(7000, 1200_000)),  # Best
            ecm_coverage(ps, 11, 1, *make_params(10000, 700_000)),
            ecm_coverage(ps, 11, 1, *make_params(12000, 400_000)),
        )
    # Crossover for polyeval is around 1M
    print("ECM 500k")
    for pbits in (48, 56, 64, 72):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            ecm_coverage(ps, 11, 1, *make_params(9000, 7000_000)),
            ecm_coverage(ps, 11, 1, *make_params(12000, 6000_000)),  # Best
            ecm_coverage(ps, 11, 1, *make_params(15000, 4500_000)),
            ecm_coverage(ps, 11, 1, *make_params(20000, 3000_000)),
        )
    print("ECM 1M")
    for pbits in (56, 64, 72, 80):
        ps = sorted(primesamples(pbits))
        print(
            pbits,
            ecm_coverage(ps, 11, 1, *make_params(15000, 30_000_000)),
            ecm_coverage(ps, 11, 1, *make_params(20000, 25_000_000)),  # Best
            ecm_coverage(ps, 11, 1, *make_params(25000, 20_000_000)),
            ecm_coverage(ps, 11, 1, *make_params(30000, 16_000_000)),
        )


print("pbits\tbudget\tpm1\tecm\tcost(ecm)")  # + "\trgain\tpgain\tegain1")
for pbits in (32, 36, 40, 44, 48, 56, 64, 72, 80):
    ps = sorted(primesamples(pbits))
    assert ps[0] > 3

    best = None
    pm1_scores = {}

    for budget in (4000, 8000, 16000, 32000, 50000, 100000, 200000, 500000):
        if pbits <= 44 and budget > 100000:
            continue
        if pbits >= 48 and budget < 16000:
            continue
        if pbits >= 64 and budget < 50000:
            continue
        prodB1, B2 = pm1_params(budget)
        pscore = pm1_coverage(ps, prodB1, B2)
        prodB1, B2 = ecm_params(budget)
        escore = ecm_coverage(ps, 11, 1, prodB1, B2)
        # escore = ecm_coverage(ps, 9, 4, prodB1, p2s)
        line = f"{pbits}\t{budget}\t{pscore:.3f}\t{escore:.3f}"

        # Effectiveness in bits
        cost_ecm = int(budget / escore) if escore > 0.01 else float("inf")
        line += f"\t{cost_ecm}"

        pm1_scores[budget] = pscore
        if best is None or best[0] > cost_ecm:
            best = (cost_ecm, f"ecm({budget})")

        print(line)

    # Compute best strategy
    strategy = min(
        (pcost + (1 - p) * best[0], f"PM1({pcost})+{best[1]}")
        for pcost, p in pm1_scores.items()
    )
    print(f"{pbits} bits => {strategy[1]} cost={strategy[0]:.0f}")
