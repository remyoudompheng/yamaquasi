"""
Theoretical study of optimal B1/B2 for ECM

The probabilities of smoothness are estimated following
Bach, Eric; Peralta, René (1996). "Asymptotic Semismoothness Probabilities"
Mathematics of Computation. 65 (216): 1701–1715

The optimal choice of B1/B2 depends on the relative cost
of stage 1 and stage 2 and can vary after algorithm changes.

Usually doubling B1 doubles the cost of a curve but also
the probability of success. Theoretical computation shows
that best values of (B1,B2) can have a 5%-10% difference
in efficiency.

Sample results:

32 bit (10 digits): B1=500 B2=65000 or B1=200 B2=15000 (< 10 curves)
48 bit (15 digits): B1=2000 B2=130k-268k (20-25 curves)
64 bit (20 digits): B1=10k-20k B2=1.2-1.8M (70-50 curves)
80 bit (25 digits): B1=100k B2=19-38M (100 curves)
96 bit (30 digits): B1=500k B2=150-650M (150-200 curves)
112 bit (35 digits): B1=1M-2M B2=1.3e9-2.6e9 (400-600 curves)
128 bit (40 digits): B1=5M B2=21e9 (~900 curves)
"""

from math import log2, sqrt
from sage.all import dickman_rho

# fmt:off
B1 = [
    100, 200, 500, 1000, 2000,
    5000, 10000, 20000, 50_000,
    100_000, 200_000, 500_000, 1e6,
    2e6, 5e6, 10e6, 20e6, 50e6,
    100e6, 200e6, 300e6,
]
B2 = [
    10e3, 15e3, 30e3, 65e3,
    98e3, 130e3, 268e3, 400e3,
    540e3, 1.2e6, 1.8e6, 2.4e6, 4.7e6, 7.1e6,
    9.5e6, 19e6, 28e6, 38e6, 78e6, 117e6, 156e6,
    322e6, 643e6, 1.3e9, 2.6e9, 5.2e9, 10.5e9,
    21e9, 32e9, 43e9, 136e9, 362e9, 543e9,
]
# fmt:on


# Ordinary integers have average factors 2^1 3^0.5
# Edwards curves with (2,4)-torsion have average 2^4.33 3^0.68 (alpha=-2.75)
# => equivalent to 4 fewer bits
# Suyama-11 curves have average 2^3.66 3^1.69 (alpha=-3.38)
# => equivalent to 4.8 fewer bits
EXTRA_SMOOTHNESS = 4.8


def semismooth(u, v):
    # Probability of semismoothness (B1=x^1/u B2=x^1/v for u >= v)
    # G(1/u, 1/v) = ρ(u) + int(t=v..u, ρ(u - u/t)/t dt)
    # where ρ is the Dickman function.
    # The integral is computed using naïve Riemann summation.
    g = dickman_rho(u)
    for i in range(100):
        t = v + (u - v) * (i + 0.5) / 100
        g += (u - v) * dickman_rho(u - u / t) / t / 100.0
    return g


def pm1_cost(b1, b2):
    # Constants are calibrated to follow trends of the P-1 benchmark
    # Stage 1 costs 1.2 MULMOD per exponent bit (1.44 B1 bits)
    # Stage 2 constant is 3 for smaller inputs, 2 for larger inputs
    # On a Cortex-A76 constant was 2 for smaller inputs, 1.5 for larger inputs
    stage1 = 1.44 * 1.2 * b1
    if b2 < 80e3:
        stage2 = b2 / log2(b2)
    elif b2 < 200e9:
        stage2 = 3 * sqrt(b2) * log2(b2)
    else:
        stage2 = 2 * sqrt(b2) * log2(b2)
    return stage1, stage2


def ecm_cost(b1, b2):
    # Run the ECM benchmark to calibrate constants.
    # Stage 1 costs 8.92 MULMOD per exponent bit (1.44 B1 bits)
    # Stage 2 is O(sqrt(B2) log2(B2))
    # Stage 2 constant is 6-7 for large inputs, 9-11 for small inputs.
    # On a Cortex-A76 constant was 5.5 for large inputs, 8-9 for small inputs
    stage1 = 1.44 * 8.92 * b1
    if b2 < 3e6:
        stage2 = 2.2 * b2 / log2(b2)
    else:
        stage2 = 9 * sqrt(b2) * log2(b2)
    return stage1, stage2


print("Semisoothness probabilities (compare with Bach-Peralta)")
for u in range(2, 6):
    for v in range(2, u + 1):
        g = semismooth(u, v)
        print(f"G({u=:.2f}, {v=:.2f}) = {g:.3e}")

print(f"=== P-1 EFFICIENCY ===")
PM1_PARAMS = [
    (400, 15e3),
    (600, 40e3),
    (10_000, 270e3),
    (50_000, 8e6),
    (500_000, 300e6),
    (1_000_000, 1.2e9),
    (2_000_000, 5e9),
    (7_000_000, 18e9),
    (16 << 20, 150e9),
    (45e6, 2.5e12),
    (160e6, 20e12),
]
for b1ref, b2ref in PM1_PARAMS:
    print(f"Compare B1={b1ref:.3e} B2={b2ref:.3e}")
    for b1, b2 in [
        (b1ref, b2ref),
        (0.4 * b1ref, 2 * b2ref),
        (1.4 * b1ref, 0.5 * b2ref),
    ]:
        lb1, lb2 = log2(b1), log2(b2)
        p = [
            100 * semismooth((bits - 1.5) / lb1, (bits - 1.5) / lb2)
            for bits in (24, 32, 48, 64, 80, 96)
        ]
        c1, c2 = pm1_cost(b1, b2)
        cost = c1 + c2
        print(
            f"B1={b1:.2e} B2={b2:.2e} cost={cost:.2e} success ratio 24b={p[0]:.1f}% 32b={p[1]:.1f}% 48b={p[2]:.1f}% 64b={p[3]:.1f}% 80b={p[4]:.1f}% 96b={p[5]:.1f}%"
        )

for bits in (24, 32, 36, 40, 44, 48, 52, 56, 64, 72, 80, 96, 112, 128, 144, 160):
    print(f"=== ECM for {bits}-bit factor ===")
    extra_bits = EXTRA_SMOOTHNESS
    # Find best cost
    best = 1e100
    for b1 in B1:
        for b2 in B2:
            if b2 < b1:
                continue
            u = (bits - extra_bits) / log2(b1)
            v = (bits - extra_bits) / log2(b2)
            g = semismooth(u, v)
            c1, c2 = ecm_cost(b1, b2)
            best = min(best, (c1 + c2) / g)
    # Find neighbourhood
    for b1 in B1:
        for b2 in B2:
            if b2 < b1:
                continue
            u = (bits - extra_bits) / log2(b1)
            v = (bits - extra_bits) / log2(b2)
            g = semismooth(u, v)
            c1, c2 = ecm_cost(b1, b2)
            cost = (c1 + c2) / g
            if cost > 1.5 * best:
                continue
            pc1, pc2 = 100 * c1 / (c1 + c2), 100 * c2 / (c1 + c2)
            print(
                f"B1={b1} B2={b2:.2e} G({u=:.2f}, {v=:.2f}) = 1/{1/g:.2f} success, cost {cost:.3e} (stage1 {pc1:.0f}% stage2 {pc2:.0f}%)"
            )
