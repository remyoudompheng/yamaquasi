"""
Verify the contents of ymcls output directory
"""

import argparse
import json
from pathlib import Path
import sys
import time

from sage.all import *


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--all", action="store_true")
    argp.add_argument("DIRECTORY")
    args = argp.parse_args()

    dir = Path(args.DIRECTORY)
    with open(dir / "args.json") as f:
        meta = json.load(f)
    D = int(meta["d"])
    print("Discriminant", D)

    if args.all:
        if (rels := dir / "relations.sieve").is_file():
            verify_rels(rels, D)

    if (rels := dir / "relations.pruned").is_file():
        verify_rels(rels, D)

    if not (gextra := dir / "group.structure.extra").is_file():
        gextra = None

    if (g := dir / "group.structure").is_file():
        verify_group(g, D, gextra=gextra)


def verify_rels(rels, D):
    match D % 4:
        case 0:
            ONE = BinaryQF([1, 0, -D // 4])
        case 1:
            ONE = BinaryQF([1, 1, (1 - D) // 4])
        case _:
            raise ValueError(f"D must be 0 or 1 modulo 4")

    t0 = time.time()
    with open(rels) as f:
        count = 0
        for line in f:
            line = line.strip()
            facs = [int(p) for p in line.split()]
            pos, neg = [], []
            for f in facs:
                if f > 0:
                    pos.append(f)
                else:
                    neg.append(-f)
            # Check
            qpos = ONE
            for l in pos:
                qpos *= qf(l, D)
            qneg = ONE
            for l in neg:
                qneg *= qf(l, D)
            assert qpos.reduced_form() == qneg.reduced_form()
            count += 1
        print(f"{rels}: {count} relations verified in {time.time()-t0:.3f}s")


def verify_group(g, D, gextra=None):
    with open(g) as f:
        structure = next(f)
        assert structure.startswith("G ")
        invariants = [int(z) for z in structure[2:].strip().split()]
        if len(invariants) > 1:
            raise NotImplementedError

        ls = []
        dlogs = []
        gen = None
        for line in f:
            l, dlog = line.strip().split()
            l, dlog = int(l), int(dlog)
            ls.append(l)
            dlogs.append(dlog)
            if dlog == 1:
                gen = l

        for l, dlog in zip(ls, dlogs):
            q = qpow(qf(gen, D), dlog, D).reduced_form()
            assert q == qf(l, D)
            print(f"OK [{l}] == [{gen}]^{dlog}")

    if gextra:
        with open(gextra) as f:
            for line in f:
                l, dlog = line.strip().split()
                l, dlog = int(l), int(dlog)
                q = qpow(qf(gen, D), dlog, D).reduced_form()
                assert q == qf(l, D)
                print(f"OK [{l}] == [{gen}]^{dlog}")


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
