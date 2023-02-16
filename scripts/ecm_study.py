"""
Study efficiency of ECM parameters

The study of curve orders uses the
equivalence between twisted Edwards and Weierstrass forms:
    a x^2 + y^2 = 1 + d x^2 y^2
is equivalent to:
    B y^2 = x^3 + A x^2 + x
    where A = 2(a+d)/(a-d), B=4/(a-d)
and equivalent to
    y^2 = x^3 + (A/B) x + (1/B^2)

If c=(a-d)/4
1/B^2 = c^2
A/B = (a+d)/2 = a-2c

References:

Daniel J. Bernstein, Peter Birkner, Tanja Lange, Christiane Peters. ECM using Edwards curves
https://eecm.cr.yp.to/eecm-20111008.pdf

Razvan Barbulescu, Joppe W. Bos, Cyril Bouvier, Thorsten Kleinjung, Peter L.
Montgomery. Finding ECM-friendly curves through a study of Galois properties.
ANTS-X 10th Algorithmic Number Theory Symposium - 2012
https://hal.inria.fr/hal-00671948v2
"""

from sage.all import (
    euler_phi,
    factor,
    product,
    random_prime,
    proof,
    GF,
    EllipticCurve,
    QQ,
)
from math import gcd
import itertools

proof.arithmetic(False)

# Ideal values of D/B2
# We cannot reach D > 6 φ(D) (it requires D >= 223092870)
# fmt:off
smooths = sorted([
    product(_pks) for _pks in itertools.product(
        (1, 2, 4, 8, 16, 32), (1, 3, 9, 27), (1, 5, 25), (1, 7, 49), (1, 11),
        (1, 13), (1, 17), (1, 19), (1, 23), (1, 29), (1, 31), (1, 37)
    )
])
# fmt:on
# For small values, we want "square" patterns (1 block)
for t in [32, 40, 48, 64, 80, 100, 120, 150, 180, 210, 240, 320, 360, 420]:
    nbest = 0
    for n in smooths:
        if 0.80 <= euler_phi(n) / 2 / t <= 1.01:
            nbest = n
    phi = euler_phi(nbest) // 2
    print(
        f"D1={nbest} D2={phi} B2={float(phi*nbest):.3e} φ(D)/2={phi} (cost {phi**2} products)"
    )
for k in range(9, 25):
    t = 1 << k
    nbest = 0
    for n in smooths:
        if 0.80 <= euler_phi(n) / 2 / t <= 1.01:
            nbest = n
    phi = euler_phi(nbest) // 2
    for d2 in [2 * t, 3 * t, 4 * t]:
        print(
            f"D1={nbest} D2={d2} B2={float(d2*nbest):.3e} φ(D)/2={phi} ({d2//t} blocks size 2^{k-1})"
        )

# Moduli space of Suyama-11 curves.
X13f = EllipticCurve(QQ, [0, -1, 0, -120, 432])


def x13f_params(P):
    PX, PY = P.xy()
    sigma = 24 / PX - 1
    alpha = sigma**2 - 5
    beta = 4 * sigma
    r = 48 * PY / PX**2
    if sigma not in (0, 1, 5, -5):
        x = (2 * r * sigma) / (sigma - 1) / (sigma + 5) / (sigma**2 + 5)
        y = (alpha**3 - beta**3) / (alpha**3 + beta**3)
    else:
        x, y = "invalid", "invalid"
    print(f"P={P} σ={sigma} r={r} G=({x}, {y})")
    return sigma, r, x, y


print("Suyama-11 family details:")
x13f_t = X13f.torsion_points()
print("Torsion points (invalid)")
for t in x13f_t:
    if t[0] == 0:
        continue
    x13f_params(t)

pg = X13f.gens()[0]
for k in (1, 2, 3):
    print(f"{k}G + torsion")
    for t in x13f_t:
        x13f_params(k * pg + t)


def analyze(primes, b1, b2, x, y, a=1, expect=1):
    stats1, stats2 = [], []
    exp2, exp3 = {}, {}
    exp2s, exp3s = 0, 0
    t1, t2 = 0, 0
    success = 0
    for p in primes:
        K = GF(p)
        d = K(a * x * x + y * y - 1) / K(x * x * y * y)
        c = (K(a) - d) / 4
        E = EllipticCurve(GF(p), [0, K(a) - 2 * c, 0, c * c, 0])
        Eord = E.order()
        fs = factor(Eord)
        assert Eord % expect == 0
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
        if max1 <= b1 and max2 <= b2:
            success += 1
        # Examine valuation of 2, 3 depending on p%12
        val2, val3 = int(Eord.valuation(2)), int(Eord.valuation(3))
        exp2.setdefault(p % 12, []).append(val2)
        exp3.setdefault(p % 12, []).append(val3)
        exp2s += val2
        exp3s += val3

    stats1.sort()
    stats2.sort()
    print(f"Generic torsion: ({t1}, {t2})")
    print(f"Success rate {success/len(primes)*100:.2f}% (B1={b1} B2={b2})")
    l = len(primes)
    print(f"avg exponent of 2,3: {exp2s/l:.3f}, {exp3s/l:.3f}")
    for r in (1, 5, 7, 11):
        v2, v3 = sum(exp2[r]) / len(exp2[r]), sum(exp3[r]) / len(exp3[r])
        print(f"avg exponent of 2,3 for primes 12k+{r}: ({v2:.3f}, {v3:.3f})")
    p25, p50 = stats1[l // 4], stats1[-l // 2]
    p66, p75 = stats1[-l // 3], stats1[-l // 4]
    print(f"needs B1 {p25=} {p50=} {p66=} {p75=:.3g}")
    p25, p50 = stats2[l // 4], stats2[-l // 2]
    p66, p75 = stats2[-l // 3], stats2[-l // 4]
    print(f"needs B2 {p25=:.3g} {p50=:.3g} {p66=:.3g} {p75=:.3g}")


# For each curve family, we break after the first curve,
# but test many primes to improve statistics.
EXTRA_CURVES = False

for size in (24, 32, 48, 64, 80, 96, 112, 128):
    samples = 8000 if size <= 64 else 2000
    print(f"################################################")
    print(f"# Working with {samples} primes of size {size} bits")
    print(f"################################################")
    b1 = 1 << (size // 4)
    b2 = size * b1
    ps = [random_prime(2 ** (size - 1), 2**size) for _ in range(samples)]

    # Basic Edwards curves
    # Average exp2 = 3.66
    # Average exp3 = 0.68
    # Family X13h, (alpha=-2.28?)
    for y in (3, 4, 5, 6):
        print(f"\n=== Trying generic Edwards curve through (2, {y})")
        analyze(ps, b1, b2, 2, y, expect=4)
        if not EXTRA_CURVES:
            break

    # Extra 2-torsion (Z/2 x Z/4) when d is a square
    # (3a+5)² + (4a+5)² = 1 + (5a+7)²
    # Average exp2 = 4.33
    # Average exp3 = 0.68
    # Family X25n (alpha=-2.75?)
    for a in (1, 2, 3):
        x, y = 3 * a + 5, 4 * a + 5
        print(f"\n=== Trying Edwards curve Z/2 x Z/4 ({x}, {y})")
        analyze(ps, b1, b2, x, y, expect=8)
        if not EXTRA_CURVES:
            break

    # Generic Suyama curve
    # Average exp2 = 3.33
    # Average exp3 = 1.69
    # Family X13 (alpha=-3.15)
    for sigma in (6, 7, 8):
        alpha = sigma**2 - 5
        beta = 4 * sigma
        a = (beta - alpha) ** 3 * (3 * alpha + beta)
        v1 = (sigma**2 - 1) * (sigma**2 - 25) * (sigma**4 - 25)
        x = QQ(alpha * beta) / QQ(2 * v1)
        y = QQ(alpha**3 - beta**3) / QQ(alpha**3 + beta**3)
        print(f"\n=== Trying Generic Suyama (σ={sigma})")
        analyze(ps, b1, b2, x, y, a=a, expect=1)
        if not EXTRA_CURVES:
            break

    # Suyama-11 Twisted Edwards (X13f)
    # https://eecm.cr.yp.to/a1ecm-20100614.pdf (section 5.3)
    # Average exp2 = 3.66
    # Average exp3 = 1.69
    # Family X13f-3B0-3aT2 in Barbulescu et al. (alpha=-3.38)
    C = EllipticCurve(QQ, [0, -1, 0, -120, 432])
    PC = C(12, 24)
    for t in (2, 3, 4, 5):
        S = t * PC
        sigma = 24 / S[0] - 1
        alpha = sigma**2 - 5
        beta = 4 * sigma
        r = 48 * S[1] / S[0] ** 2
        x = (2 * r * sigma) / (sigma - 1) / (sigma + 5) / (sigma**2 + 5)
        y = (alpha**3 - beta**3) / (alpha**3 + beta**3)
        print(f"\n=== Trying Suyama-11 Twisted Edwards Z/6 (σ={sigma})")
        analyze(ps, b1, b2, x, y, a=-1, expect=12)
        if not EXTRA_CURVES:
            break

    # Good curve Z/12
    # Average exp2 = 3.66
    # Average exp3 = 1.69
    # Family X13h-3B0-3aT2 (alpha=-3.38)
    x, y = QQ("5/23"), QQ("-1/7")
    print(f"=== Trying Edwards curve Z/12 ({x}, {y})")
    analyze(ps, b1, b2, x, y, expect=12)

    if EXTRA_CURVES:
        x, y = QQ("8/17"), QQ("20/19")
        print(f"=== Trying Edwards curve Z/12 ({x}, {y})")
        analyze(ps, b1, b2, x, y, expect=12)

    # Good curve 2x8
    # Average exp2 = 5.33
    # Average exp3 = 0.68
    # Family X193n (alpha=-3.43)
    x, y = QQ("17/19"), QQ("17/33")
    print(f"=== Trying Edwards curve Z/2 x Z/8 ({x}, {y})")
    analyze(ps, b1, b2, x, y, expect=16)

    if EXTRA_CURVES:
        x, y = QQ("125/91"), QQ("841/791")
        print(f"=== Trying Edwards curve Z/2 x Z/8 ({x}, {y})")
        analyze(ps, b1, b2, x, y, expect=16)
