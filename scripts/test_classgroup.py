"""
Random tests for class group computations
"""

import argparse
import sys
from random import randint
import subprocess
import tempfile
import time

from gmpy2 import mpz, next_prime, is_prime, legendre, invert


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--check", action="store_true")
    argp.add_argument("--exec", action="store_true")
    argp.add_argument("-v", action="store_true")
    argp.add_argument("MODE", choices=("worst", "average", "best"))
    argp.add_argument("N_BITS", type=int)
    argp.add_argument("ITERS", type=int)
    args = argp.parse_args()

    t0 = time.time()
    fails = 0
    if args.exec:
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(args.ITERS):
                D = gendisc(args.MODE, args.N_BITS)
                proc = subprocess.Popen(
                    ["bin/ymcls", str(D), tmpdir],
                    encoding="utf-8",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                out, err = proc.communicate()
                if proc.returncode == 0:
                    print("OK", D)
                else:
                    print(err)
                    print(out)
                    fails += 1
    else:
        import pymqs

        for _ in range(args.ITERS):
            D = gendisc(args.MODE, args.N_BITS)
            try:
                start = time.time()
                h, invs, gens = pymqs.quadratic_classgroup(int(D))
                dt = time.time() - start
                if args.check:
                    check_group(D, h, invs, gens)
                if args.v:
                    print("OK", D, h, f"{dt:.5f}s")
            except BaseException as err:
                print("KO", int(D), err)
                fails += 1

    print(args.ITERS, f"tested in {time.time()-t0:.3}s")
    print(fails, "failures")


def gendisc(mode, size):
    assert size < 400

    if mode == "average":
        factors = randint(1, 4)
        prod = 1
        facs = []
        for _ in range(factors - 1):
            psize = size // factors
            p = next_prime(randint(2 ** (psize - 1), 2**psize))
            prod *= p
            facs.append(int(p))
        bound = 2**size // prod
        p = next_prime(randint(bound // 2, bound))
        prod *= p
        facs.append(int(p))
        assert size - 4 < prod.bit_length() < size + 2
        if prod % 4 == 1:
            prod *= 4
        return -prod

    # Random selection with prescribed Legendre symbols
    target_legendre = 1 if mode == "best" else -1
    ps = primes(2 * size)[1:]
    js = []
    for p in ps:
        while True:
            j = randint(1, p)
            if legendre(j, p) == target_legendre:
                break
        js.append(j)

    prod = 1
    for i, p in enumerate(ps):
        prod_ = prod * p
        if prod_.bit_length() < size:
            prod = prod_
        else:
            ps = ps[:i]
            break

    b = crt_basis(ps)
    n = sum(j * bi for j, bi in zip(js, b))
    assert all(legendre(n, p) == target_legendre for p in ps)
    n0 = n % prod
    if mode == "best":
        for k in range(2 ** (size - 1) // prod, 2 ** (size + 2) // prod + 8):
            n = n0 - k * prod
            if n < 0 and n % 8 == 1:
                break
        assert all(legendre(n, p) == 1 for p in ps)

        return n

    else:
        # Worst case example
        for k in range(2 ** (size - 1) // prod, 2 ** (size + 2) // prod + 8):
            n = n0 - k * prod
            if n < 0 and n % 4 == 3:
                break
        assert all(legendre(4 * n, p) == -1 for p in ps)

        return 4 * n


def crt_basis(primes):
    prod = mpz(1)
    for p in primes:
        prod *= p
    basis = []
    for p in primes:
        q = prod // p
        inv = invert(prod // p, p)
        basis.append(inv * q)
    return basis


def primes(bound):
    ps = []
    p = 2
    while p < bound:
        ps.append(p)
        p = next_prime(p)
    return ps


pari = None


def check_group(D, h, invs, gens):
    global pari
    from cypari2 import Pari
    from sage.all import ZZ, span

    if pari is None:
        pari = Pari()

    prod = 1
    for d in invs:
        prod *= d
    assert h == prod
    # Prepare generators as binary forms
    qgens = [pari.qfbprimeform(D, p) for p, _ in gens]
    # Convert coordinates to relations
    n, r = len(invs), len(gens)
    G = ZZ**n / span([_e * gi for _e, gi in zip(invs, (ZZ**n).gens())])
    R = ZZ**r / (ZZ**r).zero_submodule()
    f = R.hom([G(v) for _, v in gens])
    assert (f.domain() / f.kernel()).invariants() == G.invariants()
    rels = f.kernel().V().basis()
    for r in rels:
        res = pari.qfbpow(qgens[0], r[0])
        for q, ri in zip(qgens[1:], r[1:]):
            res = pari.qfbcomp(res, pari.qfbpow(q, ri))
        if res[0] != 1:
            rel = " ".join(f"{p}^{e}" for (p, _), e in zip(gens, r) if e)
            raise ValueError(f"failed relation {rel}")


if __name__ == "__main__":
    main()
