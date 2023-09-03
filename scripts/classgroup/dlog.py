"""
Given a class group structure, compute the coordinates of a prime ideal

This is done by sieving a few polynomials corresponding to known
multiples of this ideal.

This is extremely slow, a faster way would be to run yamaquasi sieve.
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time

from sage.all import (
    legendre_symbol,
    BinaryQF,
    cached_function,
    ZZ,
    mod,
    polygen,
    GF,
    factor,
)


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--all", action="store_true")
    argp.add_argument("DIRECTORY")
    argp.add_argument("PRIME", type=int)
    args = argp.parse_args()

    dir = Path(args.DIRECTORY)
    with open(dir / "args.json") as f:
        meta = json.load(f)
    D = int(meta["d"])
    print("Discriminant", D)

    mods = []
    fbase = {}
    gen = None
    with open(dir / "group.structure") as f:
        header = next(f).split()
        assert header[0] == "G"
        mods = [int(z) for z in header[1:]]
        for line in f:
            v = [int(z) for z in line.strip().split()]
            fbase[v[0]] = v[1:]
            if v[1:] == [1]:
                gen = v[0]
    if (dir / "group.structure.extra").is_file():
        with open(dir / "group.structure.extra") as f:
            for line in f:
                v = [int(z) for z in line.strip().split()]
                fbase[v[0]] = v[1:]

    print(f"Loaded group structure with {len(fbase)} factor base elements")
    print(f"Group generator is [{gen}]")
    if len(mods) > 1:
        raise NotImplementedError("non-cyclic groups are not supported yet")

    if D.bit_length() < 220:
        M = 1_000_000
    else:
        M = 10_000_000

    prime = args.PRIME
    assert legendre_symbol(D, prime) == 1
    done = False
    while not done:
        A, facs = choose_a(prime, D, fbase, M)
        q = qf(prime, D)
        for l in facs:
            q *= qf(l, D)
        assert q[0] == A
        print(f"Sampled ideal {prime} *", "*".join(str(q) for q in facs))
        print("Sieving", q)
        for x, val in sieve(q, M, list(fbase)):
            print(f"q({x}) =", factor(val))
            bx = -2 * q[0] * x - q[1]
            done = True
            break

    rel = []
    for f in facs:
        rel.append((f, -1))
    for l, e in factor(val):
        b = qf(l, D)[1]
        assert bx % l in (b, l - b)
        if bx % l == b:
            rel.append((l, e))
        else:
            rel.append((l, -e))
    print(prime, "=", "*".join(f"{l}^{e}" for l, e in rel))

    dlog = [0 for _ in mods]
    for l, e in rel:
        v = fbase[l]
        for i in range(len(dlog)):
            dlog[i] += e * v[i]
    for i in range(len(dlog)):
        dlog[i] %= mods[i]

    gl = qpow(qf(gen, D), dlog[0], D)
    assert gl.reduced_form() == qf(prime, D).reduced_form()
    print(f"Discrete log [{prime}] = [{gen}]^{dlog}")


def choose_a(prime, D, fbase, M):
    # Choose some multiples of the factor base to produce
    # A ~= sqrt(D)/M
    A = prime
    facs = []
    target = ZZ(abs(D)).isqrt() // M
    fb = sorted(fbase)
    while A < target // 100:
        q = random.choice(fb)
        if A * q < target:
            A *= q
            facs.append(q)

    if A * min(fb) < target:
        q = max(_q for _q in fb if A * _q < target)
        A *= q
        facs.append(q)
    return A, facs


def sieve(q, M, fbase: list):
    # Start sieving
    I = [1 for _ in range(-M, M)]
    x = polygen(ZZ)
    a, b, c = list(q)
    vmin = c
    vmax = c + a * M**2
    print(f"Value range {vmin.bit_length()}-{vmax.bit_length()} bits")
    pol = a * x * x + b * x + c
    for p in fbase:
        rs = pol.change_ring(GF(p)).roots(multiplicities=False)
        rs = [ZZ(r) for r in rs]
        start = -p * (M // p + 1)
        for r in rs:
            for i in range(r + start, M, p):
                if abs(i) < M:
                    I[i + M] *= p
    maxl = 0
    for i, x in enumerate(I):
        xl = x.bit_length()
        maxl = max(maxl, xl)
        if x.bit_length() >= c.bit_length() - 10:
            facs = factor(pol(i - M))
            # print(pol(i - M), "smooth value", facs)
            yield i - M, pol(i - M)
    print("Largest smooth part", maxl, "bits")


@cached_function
def qf(p, D):
    # Convert to binary quadratic form
    assert p == 2 or legendre_symbol(D, p) >= 0
    if D & 1:
        b = ZZ(min(mod(D, 2 * p).sqrt(all=True)))
        c = (ZZ(b) ** 2 - D) // (4 * p)
        assert b * b - 4 * p * c == D
        assert (q := BinaryQF([p, b, c])) == q.reduced_form()
        return BinaryQF(p, b, c)
    else:
        if p == 2:
            if D % 8 == 0:
                return (2, 0, -D // 8)
            else:
                return (2, 2, (4 - D) // 8)
        b = ZZ(min(mod(D, 2 * p).sqrt(all=True)))
        c = (ZZ(b) ** 2 - D) // (4 * p)
        assert b * b - 4 * p * c == D
        assert (q := BinaryQF([p, b, c])) == q.reduced_form()
        return BinaryQF(p, b, c)


def qpow(qf, e, D):
    if D % 4 == 1:
        res = BinaryQF([1, 1, (1 - D) // 4])
    else:
        raise NotImplementedError
    sq = qf
    while e:
        if e % 2 == 1:
            res = (res * sq).reduced_form()
        sq = (sq * sq).reduced_form()
        e //= 2
    return res


if __name__ == "__main__":
    main()
