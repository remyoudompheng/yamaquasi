"""
Determination of group structure

This step uses the (filtered) relation matrix
and the knowledge of the class number factorization
to determine the full group structure.

This is done by computing the right kernel of the relation matrix.

Supported methods:
* numpy/scipy for moduli below 55 bits using Wiedemann algorithm
* SageMath (Wiedemann) using generic sparse matrices (slow)
* Block Wiedemann using Cado-NFS binaries

Currently, the case where the class group has mixed torsion
(Z/p^a x Z/p^b for a != b) is unsupported.
"""

import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack

from sage.all import (
    Matrix,
    MatrixSpace,
    GF,
    Zmod,
    factor,
    cputime,
    lcm,
    vector,
    prod,
    CRT_basis,
)
from step2_classno import read_matrix


def structure(datadir: Path, meta, sage=False):
    print("==> GROUP STRUCTURE STEP")

    M, primes, maxw = read_matrix(datadir / "relations.filtered")
    print(repr(M))
    print("maxw", maxw)

    # Read class number
    with open(datadir / "classnumber") as f:
        h = int(f.read().strip())

    hfacs = factor(h)
    print("Class number factors", hfacs)

    # Once the class number is known, compute generators for the group
    # invariant.
    # Given a presentation of the group Z^m --M-> Z^n -->> Gp -> 0
    # a coordinate on the p-torsion subgroup is a GF(p)-linear map L
    # such that L vanishes on the image of M, meaning that L
    # belongs to the right kernel of M.

    moduli = []
    vlogs = []

    for l, e in hfacs:
        print(f"Invariant factor {l}")
        if l == 2:
            M2 = Matrix(GF(2), M.shape[0], M.shape[1])
            for (i, j), v in M.todok().items():
                M2[i, j] = v
            ker = M2.right_kernel().basis()
            if len(ker) < e:
                # Lift them
                if len(ker) == 1:
                    sol = ker[0]
                    for exponent in range(2, e + 1):
                        print(f"Lifting modulo {l}^{exponent}")
                        sol = sparsekernel_pk(M, maxw, l, exponent, sol0=sol)
                    moduli.append(l**e)
                    vlogs.append(vector(Zmod(l**e), sol))
                    # FIXME FIXME
                else:
                    raise NotImplementedError(
                        f"Failed to determine 2-torsion: kernel has rank {len(ker)}"
                    )
            else:
                for k in ker:
                    moduli.append(2)
                    vlogs.append(ker[0])
            continue

        ker = []
        while True:
            sol = sparsekernel(M, maxw, l, sage=sage)
            if sol == 0:
                print("solution is a null vector, retry")
            else:
                ker.append(sol)
                if e == 1 or Matrix(GF(l), ker).rank() == e or len(ker) > e + 8:
                    break
                else:
                    rk = Matrix(GF(l), ker).rank()
                    print(f"Found rank {rk} subgroup of {l}-torsion")

        if (rk := Matrix(GF(l), ker).rank()) < e:
            if rk == 1:
                sol = ker[0]
                for exponent in range(2, e + 1):
                    print(
                        f"Only {rk} solutions modulo {l}^{exponent-1}, trying modulo {l}^{exponent}"
                    )
                    sol = sparsekernel_pk(M, maxw, l, exponent, sol0=sol)
                moduli.append(l**e)
                vlogs.append(vector(Zmod(l**e), sol))
            else:
                raise NotImplementedError
        else:
            for b in Matrix(GF(l), ker).row_module().basis():
                moduli.append(l)
                vlogs.append(b)

    print("Coordinates found modulo", moduli)
    assert len(moduli) == len(vlogs)

    # Build a large CRT
    seen = set()
    used = []
    for i, l in enumerate(moduli):
        if l not in seen:
            used.append(i)
            seen.add(l)
    bigm = prod(moduli[i] for i in used)
    crtb = CRT_basis([moduli[i] for i in used])
    bigvlog = []
    Rbig = Zmod(bigm)
    for i in range(len(primes)):
        bigvlog.append(Rbig(sum(int(vlogs[j][i]) * _c for j, _c in zip(used, crtb))))

    invariants = [bigm] + [moduli[i] for i in range(len(moduli)) if i not in used]
    vlogs = [bigvlog] + [vlogs[i] for i in range(len(moduli)) if i not in used]
    assert len(invariants) == len(vlogs)

    # Normalize all kernels whenever possible
    # This is to (tentatively) express all primes as multiples of the first one.
    for idx, (m, v) in enumerate(zip(invariants, vlogs)):
        for i in range(10):
            if v[i].is_unit():
                vlogs[idx] = vector(Zmod(m), v) / v[i]
                break

    print("Group structure", "*".join(f"Z/{l}" for l in invariants))
    with open(datadir / "group.structure", "w") as w:
        print("G", " ".join(str(int(z)) for z in invariants), file=w)
        for idx, p in enumerate(primes):
            coords = [v[idx] for v in vlogs]
            print(p, " ".join(str(c) for c in coords), file=w)
        print("Discrete logs output to", w.name)


def sparsekernel(M, norm, p, dim=None, sage=False):
    if p * norm < 2**63:
        return _sparsekernel_numpy(M, norm, p, dim=dim)
    else:
        if os.getenv("CADONFS_BWCDIR"):
            return _sparsekernel_cado(M, p)
        else:
            if not sage:
                raise ValueError(
                    "Please define CADONFS_BWCDIR to use Cado-NFS or set --sage option"
                )
            return _sparsekernel_sage(M, p, dim=dim)


def sparsekernel_pk(M, norm, p, k, sol0):
    """
    Lift a solution mod p^(k-1) to a solution modulo p^k

    If M V0 = 0 mod p^(k-1)
    the equation M (V0 + V1 p^(k-1)) = 0 mod p^k
    is equivalent to M V1 = - (M V0 / p^(k-1)) mod p^k
    """
    assert (p**k).bit_length() < 30

    R, N = M.shape
    # print("shape", M.shape)
    # Use uint64 scipy arithmetic.
    # If Y = MX and P(M)X=0 where P=-1+XQ
    # then X = Q(M)MX = Q(M) * Y
    t = cputime()
    v0 = np.array([int(x) for x in sol0], dtype=np.int64)
    assert all(x == 0 for x in (M @ v0) % (p ** (k - 1)))
    mv1 = (M @ v0) // p ** (k - 1)
    # Note that mv1 coefficients have about the same size as M,
    # but mv1 is usually not sparse.
    norm1 = sum(abs(x) for x in mv1)
    # Build an augmented matrix
    mv1 = mv1.reshape((R, 1))
    M1 = csr_matrix(hstack([M, mv1]))
    M1.sort_indices()
    # FIXME: Why do we need dim=N ?
    while True:
        sol1 = sparsekernel(M1, max(norm, norm1), p, dim=N)
        if sol1[-1]:
            break
    # Normalize to get M V1 + (M v0 / p) == 0
    sol1 /= sol1[-1]
    sol = [int(sol0[i]) + p ** (k - 1) * int(sol1[i]) for i in range(N)]
    assert all(int(x) % (p**k) == 0 for x in (M @ np.array(sol, dtype=np.int64)))
    return sol


def _sparsekernel_numpy(M, norm, p, dim=None):
    """
    Assumes that the kernel exists

    We compute some random projection PM which is a square NxN matrix
    """
    R, N = M.shape
    dim = dim or N
    # Use uint64 scipy arithmetic.

    t = cputime()
    Fp, (FpX, x) = GF(p), GF(p)["x"].objgen()
    for iter in range(50):
        V0 = np.array([random.randint(0, p - 1) for _ in range(N)], dtype=np.int64)
        V = V0
        # Random coordinates for Berlekamp-Massey
        indices = [random.randint(0, N - 1) for _ in range(64)]
        proj = random.randint(1, norm - 1)
        krylov = [[V[i] for i in indices]]
        for _ in range(2 * N + 64 - 1):
            V = (M @ V) % p
            V = (V[:N] + proj * V[-N:]) % p
            krylov.append([V[i] for i in indices])
        # we need to take the lcm of a few polynomials
        idx = list(range(len(indices)))
        random.shuffle(idx)
        minpoly = FpX(1)
        for i in idx:
            v = [Fp(V[i]) for V in krylov]
            vp = FpX(list(reversed(v)))
            _, pol = vp.rational_reconstruction(x ** (2 * N + 64), N, N)
            minpoly = lcm(minpoly, pol)
            if minpoly.degree() == dim:
                break
        # The minimal polynomial may have smaller degree, it's fine
        assert minpoly.is_monic()
        if minpoly[0] != 0 or minpoly.degree() > N:
            print(
                f"Failed to find kernel, retry (deg={minpoly.degree()} val={minpoly.valuation()})"
            )
            continue
        # Assume that the minimal polynomial is correct
        # print(f"Minimal polynomial has degree {minpoly.degree()} and valuation {minpoly.valuation()}")
        polred = minpoly >> 1
        # Polynomial with zero coefficient is found
        if p.bit_length() < 30:
            # Small p, we can use numpy only to reduce mod p
            W = V0
            for ai in reversed(polred.list()[:-1]):
                MW = M @ W
                W = (MW[:N] + proj * MW[-N:] + int(ai) * V0) % p
        else:
            Vp = vector(GF(p), V0)
            W = V0
            for ai in reversed(polred.list()[:-1]):
                MW = vector(GF(p), M @ W)
                MWn = MW[:N] + proj * MW[-N:] + ai * Vp
                W = np.array([int(z) for z in MWn], dtype=np.int64)
            # apply the x^k factor
            for _ in range(dim - polred.degree()):
                if not ((M @ W[:N]) % p).any():
                    break
                W = (M @ W[:N]) % p

        if all(int(x) % p == 0 for x in M @ W):
            break
        # not in kernel, minpoly was probably wrong?
        print(
            f"Failed to find kernel, retry (deg={minpoly.degree()} val={minpoly.valuation()})"
        )
        assert minpoly.degree() < dim

    assert all(int(x) % p == 0 for x in M @ W[:N])
    sol = vector(GF(p), W[:N])
    print(f"Solved modulo {p} t={cputime(t):.3f}")
    return sol


def _sparsekernel_sage(M, p, dim=None):
    R, N = M.shape
    dim = dim or N
    Fp, (FpX, x) = GF(p), GF(p)["x"].objgen()
    Mp = MatrixSpace(Fp, R, N, sparse=True)()
    for (i, j), v in M.todok().items():
        Mp[i, j] = v
    print("WARNING: solving sparse system using Sage. This is VERY SLOW")
    t = cputime()
    V0 = vector(Fp, [random.randint(0, p - 1) for i in range(N)])
    V = V0
    # Random coordinates for Berlekamp-Massey
    indices = [random.randint(0, N - 1) for _ in range(64)]
    proj = random.randint(1, p - 1)
    krylov = [[V[i] for i in indices]]
    for i in range(2 * N + 64 - 1):
        if i and i % 100 == 0:
            print(i, "iterations done")
        V = Mp * V
        V = V[:N] + proj * V[-N:]
        krylov.append([V[i] for i in indices])

    idx = list(range(len(indices)))
    random.shuffle(idx)
    minpoly = FpX(1)
    for i in idx:
        v = [Fp(V[i]) for V in krylov]
        vp = FpX(list(reversed(v)))
        _, pol = vp.rational_reconstruction(x ** (2 * N + 64), N, N)
        minpoly = lcm(minpoly, pol)
        if minpoly.degree() == dim:
            break

    # The minimal polynomial may have smaller degree, it's fine
    assert minpoly[0] == 0
    assert minpoly.is_monic()
    minpoly //= x

    V = V0
    for ai in reversed(minpoly.list()[:-1]):
        V = Mp * V
        V = V[:N] + proj * V[-N:]
        V += ai * V0

    assert Mp * V == 0
    sol = V
    print(f"Solved modulo {p} t={cputime(t):.3f}")
    return sol


def _sparsekernel_cado(M, p):
    R, N = M.shape
    K = GF(p)
    Mp = MatrixSpace(GF(p), R, R, sparse=True)()
    for (i, j), v in M.todok().items():
        Mp[i, j] = v

    with tempfile.TemporaryDirectory() as tmpdir:
        bwc = CadoBWC(tmpdir=Path(tmpdir))
        bwc.export_matrix(M.todok(), R, N)
        sol = bwc.solve_modp(None, p, check=True, M=Mp)
    return sol


class CadoBWC:
    def __init__(self, tmpdir: Path):
        os.makedirs(tmpdir / "bwc", exist_ok=True)
        self.BWC_DIR = Path(os.getenv("CADONFS_BWCDIR"))
        assert (self.BWC_DIR / "bwc.pl").is_file()
        assert (self.BWC_DIR / "mf_scan2").is_file()
        self.LOGFILE = tmpdir / "ymqs.cado.log"
        self.TMPDIR = tmpdir
        self.BWC_TMPDIR = self.TMPDIR / "bwc"

        if os.path.isfile(self.LOGFILE):
            os.remove(self.LOGFILE)
            print("Cado-NFS log file is", self.LOGFILE)

        os.makedirs(self.BWC_TMPDIR, exist_ok=True)
        print("Cado-NFS work directory is", self.BWC_TMPDIR)

    def export_matrix(self, Mdict, nrows, ncols):
        """
        Export a matrix using Cado-NFS internal format.
        The format may be specific to a Cado-NFS version and is
        NOT guaranteed to be supported.

        The format uses signed 32-bit integers.
        The input must be a matrix of small integers encoded as a dictionary
        of scipy DOK matrix.
        """
        with open(self.TMPDIR / "matrix.bin", "wb") as w:
            rows = [set() for i in range(nrows)]
            for i, j in Mdict.keys():
                rows[i].add(j)
            for i in range(nrows):
                rowdata = []
                for j in sorted(rows[i]):
                    if k := Mdict[(i, j)]:
                        rowdata += [j, k]
                rowdata = [len(rowdata) // 2] + rowdata
                blob = b"".join(int(x % 2**32).to_bytes(4, "little") for x in rowdata)
                w.write(blob)
            print("Matrix written to", w.name)
            mfile = w.name

        with open(self.LOGFILE, "ab") as w:
            subprocess.check_call(
                f"{self.BWC_DIR}/mf_scan2 {mfile} --withcoeffs",
                shell=True,
                stdout=w,
                stderr=w,
            )

    def export_rhs(self, V, p):
        with open(self.TMPDIR + "/rhs.txt", "w") as w:
            w.write(f"{V.nrows()} {V.ncols()} {p}\n")
            for j in range(V.nrows()):
                w.write(str(V[j, 0]) + "\n")
            # print("RHS written to", w.name)

    def solve_modp(self, V, p, check=False, M=None):
        mfile = self.TMPDIR / "matrix.bin"
        vfile = self.TMPDIR / "rhs.txt"
        with open(self.LOGFILE, "ab") as w:
            shutil.rmtree(self.BWC_TMPDIR, ignore_errors=True)
            t0 = time.time()
            if V is None:
                # compute right kernel
                args = f"matrix={mfile} wdir={self.BWC_TMPDIR}"
            else:
                self.export_rhs(Matrix(V).T, p)
                args = f"matrix={mfile} rhs={vfile} wdir={self.BWC_TMPDIR}"
            blk = 64 if p == 2 else 1
            # We need to say RIGHT in uppercase to explain that this is not
            # a typo when p=2
            subprocess.check_call(
                f"{self.BWC_DIR}/bwc.pl :complete prime={p} nullspace=RIGHT mn={blk} "
                + args,
                shell=True,
                stdout=w,
                stderr=w,
            )
            print(f"solved at p={p} in {time.time()-t0:.3f}s")
            t0 = time.time()
            with open(self.BWC_TMPDIR / "K.sols0-1.0.txt") as f:
                if V is None:
                    # Kernel: keep one value per line
                    # FIXME
                    X = vector(GF(p), [int(l.split()[0]) for l in f])
                else:
                    # Solution of M X0 + v (X1=1) = 0
                    lines = f.read().split()
                    assert lines[-1] == "1"
                    X = -vector(GF(p), lines[:-1])
                    if check and M:
                        assert M * X == V.change_ring(GF(p))
                        print("check OK", time.time() - t0)
                return X
