"""
Generates random inputs for ymcls.
"""

import argparse
import sys
from random import randint
from gmpy2 import mpz, next_prime, is_prime, legendre, invert


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("MODE", choices=("worst", "average", "best"))
    argp.add_argument("N_BITS", type=int)
    args = argp.parse_args()

    size = args.N_BITS
    assert size < 400

    if args.MODE == "average":
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
        print(f"primes = {facs}", file=sys.stderr)
        print(f"D = -{prod} {prod.bit_length()} bits", file=sys.stderr)
        print(-prod)
        return

    # Random selection with prescribed Legendre symbols
    target_legendre = 1 if args.MODE == "best" else -1
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
    if args.MODE == "best":
        for k in range(2 ** (size - 1) // prod, 2 ** (size + 2) // prod + 8):
            n = n0 - k * prod
            if n < 0 and n % 8 == 1:
                break
        assert all(legendre(n, p) == 1 for p in ps)

        print(n, "bits", n.bit_length(), file=sys.stderr)
        print(n)

    else:
        # Worst case example
        for k in range(2 ** (size - 1) // prod, 2 ** (size + 2) // prod + 8):
            n = n0 - k * prod
            if n < 0 and n % 4 == 3:
                break
        assert all(legendre(4 * n, p) == -1 for p in ps)

        print(4 * n, "bits", (4 * n).bit_length(), file=sys.stderr)
        print(4 * n)


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


if __name__ == "__main__":
    main()
