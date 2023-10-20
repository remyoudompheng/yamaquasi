"""
This step computes the class number by computing
the GCD of minors of the relation matrix.

It requires SageMath/FLINT for the Berlekamp-Massey algorithm,
and uses scipy sparse linear algebra for the Wiedemann algorithm.

Libraries numpy and scipy are bundled with SageMath.
"""

from pathlib import Path
import random
import time

import numpy as np
from scipy.sparse import csr_matrix, vstack

from sage.all import (
    ZZ,
    GF,
    CRT,
    random_prime,
    parallel,
    cputime,
    walltime,
    proof,
    prod,
    gcd,
)

proof.arithmetic(False)


def classno(datadir: Path, meta, nthreads):
    assert isinstance(nthreads, int) and nthreads > 0
    print("==> DETERMINANT STEP")

    # Crude estimate
    happ = (meta["h_estimate_min"] + meta["h_estimate_max"]) / 2.0
    print(f"Approximate class number is {happ:.6e}")

    M, primes, maxw = read_matrix(datadir / "relations.filtered")
    avgw = M.count_nonzero() / M.shape[0]
    print(repr(M))
    print(f"average row weight {avgw:.1f} max sum(row) {maxw}")
    N = len(primes)
    nrows = M.shape[0]

    # We need to know the class number prime factors
    dets = []
    subrows = []
    # FIXME: 2 determinants should be enough?
    while True:
        t0 = walltime()
        select = random.sample(list(range(nrows)), N)
        Msub = vstack([M.getrow(i) for i in select])
        det = sparsedet(Msub, maxw, nthreads)
        if det == 0:
            continue
        print(f"Computed determinant ({det.bit_length()} bits) in {walltime(t0):.3f}s")
        dets.append(det)
        subrows.append(select)
        gd = gcd(dets)
        ratio = float(gd) / float(happ)
        if ratio < 1000:
            print("GCD is", gd, f"~{round(ratio, 6)} h_approx")
        if ratio < 50 and abs(ratio - round(ratio)) < 0.1:
            ratio = int(round(ratio))
            if ratio and gd % ratio == 0:
                gd //= ratio
                print("Found exact class number", gd)
                with open(datadir / "classnumber", "w") as w:
                    w.write(str(gd) + "\n")
                break


def read_matrix(filename: Path):
    # Determine factor base
    primes = set()
    with open(filename) as f:
        for l in f:
            for w in l.split():
                p, _, _ = w.partition("^")
                primes.add(int(p))
    # Build directly the CSR representation.
    primes = sorted(primes)
    pidx = {p: i for i, p in enumerate(primes)}
    values = []
    indices = []
    offsets = [0]
    maxeplus = 0
    maxeminus = 0
    with open(filename) as f:
        for l in f:
            eplus = 0
            eminus = 0
            for w in l.split():
                p, _, e = w.partition("^")
                p, e = int(p), int(e)
                if e > 0:
                    eplus += e
                else:
                    eminus += -e
                values.append(e)
                indices.append(pidx[p])
            offsets.append(len(values))
            maxeplus = max(maxeplus, eplus)
            maxeminus = max(maxeminus, eminus)

    M = csr_matrix((values, indices, offsets), dtype=np.int64)
    return M, primes, max(maxeplus, maxeminus)


def sparsedet(M, norm, nthreads):
    N = M.shape[0]
    detps = []
    ps = []
    candidate = -1

    def wiedemann(p):
        assert p * norm < 2**63
        t = cputime()
        V = np.array([random.randint(0, 2) for _ in range(N)], dtype=np.int64)
        krylov = [np.array(V[:64], dtype=np.int64)]
        for _ in range(2 * N + 64 - 1):
            V = (M @ V) % p
            krylov.append(np.array(V[:64], dtype=np.int64))
        detp = []
        Fp, (FpX, x) = GF(p), GF(p)["x"].objgen()
        for i in range(10):
            v = [Fp(V[i]) for V in krylov]
            # FIXME: SageMath (as of 10.1) berlekamp_massey function
            # has quadratic space complexity
            vp = FpX(list(reversed(v)))
            _, pol = vp.rational_reconstruction(x ** (2 * N + 64), N, N)
            if pol.degree() == N:
                detp.append(pol[0] if N % 2 == 0 else -pol[0])
            if len(detp) >= 3:
                break
        if not detp:
            print("degree", pol.degree())
            return 0, 0, 0
        assert all(_d == detp[0] for _d in detp[1:])
        return int(p), int(detp[0]), cputime(t)

    pmin = (2**63) // (3 * norm + 1)
    pmax = (2**63) // (2 * norm)
    pargs = [random_prime(pmax, lbound=pmin) for _ in range(N)]
    for _, (p, detp, cput) in parallel(nthreads)(wiedemann)(pargs):
        if detp == 0:
            print("Matrix is probably singular")
            return 0
        prodsize = sum(_p.bit_length() for _p in ps)
        print(
            f"wiedemann {N}x{N} mod {p} det={detp} cputime={cput:.3f} total",
            prodsize,
        )
        detps.append(ZZ(detp))
        ps.append(ZZ(p))
        assert len(detps) == len(ps)
        if sum(p.bit_length() for p in ps) > N:
            # Beware we may be looking for a negative integer.
            c = CRT(detps, ps)
            pps = prod(ps)
            c -= ZZ((c / pps).round()) * pps
            if c == candidate:
                # twice the same
                return c
            candidate = c
