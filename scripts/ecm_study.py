from random import getrandbits
from sage.all import factor, primes, proof

proof.arithmetic(False)

SIZES = [
    # size, B1, B2
    (32, 200, 5_000),
    (40, 800, 25_000),
    (48, 1_000, 200_000),
    (56, 3_000, 500_000),
    (64, 5_000, 1_000_000),
    (72, 15_000, 3_000_000),
    (80, 25_000, 7_500_000),
]

# Study the distribution of prime factors of random integers.
for sz, B1, B2 in SIZES:
    print(f"Factor size {sz} bits")
    stats1 = []
    stats2 = []
    powers = 0
    ecmok = 0
    for _ in range(5000):
        n = getrandbits(sz)
        # Edwards curves can have torsion 12 or 16
        n -= n % 4
        for f, k in factor(n):
            if k > 1 and f**k > B1:
                # print(f"large factor {f}^{k} in prime {p}")
                powers += 1
        ps = [p for p, _ in factor(n)]
        q1, q2 = sorted(ps)[-2:] if len(ps) > 1 else (ps[0], ps[0])
        stats1.append(q1)
        stats2.append(q2)
        if q1 < B1 and q2 < B2:
            ecmok += 1
    l = len(stats1)
    stats1.sort()
    stats2.sort()
    p50, p66, p75 = stats1[-l // 2], stats1[-l // 3], stats1[-l // 4]
    p90, p99 = stats1[-l // 10], stats1[-l // 100]
    print(f"2nd largest {p50=} {p66=} {p75=} {p90=} {p99=}")
    p10, p25, p50, p66 = (
        stats2[l // 10],
        stats2[l // 4],
        stats2[-l // 2],
        stats2[-l // 3],
    )
    print(f"largest {p10=} {p25=} {p50=} {p66=}")
    print(f"{powers/50:.2}% misses due to small prime powers")
    print(f"{ecmok/50:.1f}% success with B1={B1} and B2={B2}")

# Study the gaps between primes
ps = list(primes(1_000_000))
gaps = set(q - p for p, q in zip(ps, ps[1:]))
print(sorted(gaps))

# Find a nice ECM curve for a given semiprime
from sage.all import GF, Zmod, EllipticCurve

# 50-bit factors
TESTS = [
    # (p, q, B1, B2)
    # 50-bit
    (602768606663711, 957629686686973, 2_000, 400_000),
    # 64-bit factors
    (12811221319803074089, 12815927653690616723, 5_000, 1_000_000),
    # 80-bit factors
    (1174273970803390465747303, 607700066377545220515437, 25_000, 5_000_000),
]

for p, q, B1, B2 in TESTS:
    print(f"n={p*q} (p has {p.bit_length()} bits) B1={B1} B2={B2}")
    Zn = Zmod(p * q)
    x = 2
    done = False
    for y in range(3, 500):
        d = Zn(x * x + y * y - 1) / Zn(x * x * y * y)
        c = (1 - d) / 4
        for prime in [p, q]:
            E = EllipticCurve(GF(prime), [0, 1 - 2 * c, 0, c * c, 0])
            fs = [f for f, _ in factor(E.order())]
            q1, q2 = sorted(fs)[-2:] if len(fs) > 1 else (fs[0], fs[0])
            if q1 < B1 and q2 < B2:
                print(x, y, factor(E.order()), "at prime", prime)
                done = True
        if done:
            break
