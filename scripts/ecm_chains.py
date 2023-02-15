"""
Evaluation of sliding window addition chains as described in
D.J. Berstein, Analysis and optimization of elliptic-curve single-scalar multiplication
https://eprint.iacr.org/2007/455

This roughly amounts to processing blocks of bits during
fast exponentiation.

Optimal values are: 
m=3 for 32-bit blocks (0.31 ADD per bit)
m=7 for 64-bit blocks (0.26 ADD per bit)
m=11 for 128-bit blocks (0.22 ADD per bit)
m=19 for 256-bit blocks (0.20 ADD per bit)
m=31 for 512 bit blocks (0.17 ADD per bit)
m=63 for 1024 bit blocks (0.15 ADD per bit)
m=73 for 2048 bit blocks (0.14 ADD per bit)
"""

import sys
from sage.all import primes


def pblocks(sz):
    blocks = []
    buf = 1
    for p in primes(500_000):
        if (buf * p).bit_length() > sz:
            blocks.append(buf)
            buf = 1
        buf *= p
    blocks.append(buf)
    return blocks


def chain(n, m):
    if n == 2 or (n <= m and n % 2 == 1):
        c = [("double", 1)]
        for k in range(3, m + 2, 2):
            c.append(("xadd", k - 2, 2))
        return c
    if n > 16 * m and n & 15 == 0:
        return chain(n // 16, m) + [
            ("double", x) for x in (n // 16, n // 8, n // 4, n // 2)
        ]
    elif n & 1 == 0:
        return chain(n // 2, m) + [("double", n // 2)]
    else:
        a_s = [-(n & (2**k - 1)) for k in range(1, m.bit_length() + 1)] + [
            2**k - (n & (2**k - 1)) for k in range(1, m.bit_length() + 1)
        ]
        (k, a) = max(
            ((n + _a) ^ (n + _a - 1), _a)
            for _a in a_s
            if n + _a > 0 and abs(_a) <= min(m, n)
        )
        return chain(n + a, m) + [("add", n + a, -a)]
        # return chain(n-1) + [("add", n-1, 1)]


def show(c):
    words = [f"D({op[1]})" if op[0] == "double" else f"A({op[1]}, {op[2]})" for op in c]
    return " ".join(words)


def cost(c):
    ops = [x[0] for x in c]
    # print(ops.count("double"), "D", ops.count("add"), "A")
    return ops.count("double"), ops.count("add"), ops.count("xadd")


def find_m(blks, ms, verbose=False):
    for m in ms:
        print("M", m)
        ds = []
        a_s = []
        xa_s = []
        bits = []
        for n in blks:
            # print(chain(n))
            d, a, xa = cost(chain(n, m))
            ds.append(d)
            a_s.append(a)
            xa_s.append(xa)
            bits.append(n.bit_length() - 1)
        if verbose:
            print(f"Example {n=} {d}D {a}A")
            print(show(chain(n, m)))
        avd = sum(ds) / len(ds)
        ava = sum(a_s) / len(a_s)
        avx = sum(xa_s) / len(xa_s)
        avb = sum(bits) / len(bits)
        print(f"average D={avd:.2f} ADD={ava:.2f} XADD={avx:.2f} bits={avb:.2f}")


print("== 32 bit example ==")
for m in (1, 3, 5, 7):
    n = 1511 * 1523 * 1531
    d, a, x = cost(chain(n, m))
    print(f"{n=} {m=} {d}D {a}A, {x}XADD")
    print(show(chain(n, m)))

blocks32 = pblocks(32)
find_m(blocks32, [1, 3, 5, 7, 11, 13], verbose=True)

print("=== 64 bit blocks ===")
blocks64 = pblocks(64)
find_m(blocks64, [1, 3, 5, 7, 11, 15], verbose=True)

print(show(chain(1234567890, 7)))

print("=== 128 bit blocks ===")
blocks128 = pblocks(128)
find_m(blocks128, [1, 3, 7, 15, 23, 31])

print("=== 256 bit blocks ===")
blocks256 = pblocks(256)
find_m(blocks256, [1, 3, 7, 15, 23, 31])

print("=== 512 bit blocks ===")
blocks512 = pblocks(512)
find_m(blocks512, [1, 7, 15, 31, 47, 63])

sys.setrecursionlimit(2000)

print("=== 1024 bit blocks ===")
blocks1024 = pblocks(1024)
find_m(blocks1024, [1, 7, 15, 31, 47, 63, 95, 127])

sys.setrecursionlimit(4000)

print("=== 4096 bit blocks ===")
blocks4096 = pblocks(4096)
find_m(blocks4096, [15, 31, 63, 127, 255])
