"""
Evaluation of sliding window addition chains as described in
D.J. Berstein, Analysis and optimization of elliptic-curve single-scalar multiplication
https://eprint.iacr.org/2007/455
"""

from sage.all import primes

blocks = []
buf = 1
for p in primes(100000):
    if (buf * p).bit_length() > 64:
        blocks.append(buf)
        buf = 1
    buf *= p

blocks32 = []
buf = 1
for p in primes(100000):
    if (buf * p).bit_length() > 32:
        blocks32.append(buf)
        buf = 1
    buf *= p

blocks128 = []
buf = 1
for p in primes(100000):
    if (buf * p).bit_length() > 128:
        blocks128.append(buf)
        buf = 1
    buf *= p

def chain(n, m):

    if n == 2 or (n <= m and n % 2 == 1):
        c = [("double", 1)]
        for k in range(3, m + 2, 2):
            c.append(("add", k - 2, 2))
        return c
    if n % 2 == 0:
        return chain(n // 2, m) + [("double", n // 2)]
    elif m == 0:
        # Standard double-and-add
        return chain(n - 1, 0) + [("add", n - 1, 1)]
    elif n == m + 2:
        return chain(m, m) + [("add", m, 2)]
    elif m + 4 <= n <= 3 * m:
        k = n // 6
        return chain(2 * k + 1, m) + [
            ("double", 2 * k + 1),
            ("add", n - 4 * k - 2, 4 * k + 2),
        ]
    else:
        (k, a) = max(
            ((n + _a) ^ (n + _a - 1), _a) for _a in range(-m, m + 1, 2) if n + _a > 0
        )
        return chain(n + a, m) + [("add", n + a, -a)]
        # return chain(n-1) + [("add", n-1, 1)]


def show(c):
    words = [f"D({op[1]})" if op[0] == "double" else f"A({op[1]}, {op[2]})" for op in c]
    return " ".join(words)


def cost(c):
    ops = [x[0] for x in c]
    # print(ops.count("double"), "D", ops.count("add"), "A")
    return ops.count("double"), ops.count("add")


print("== 32 bit example ==")
for m in (1, 3, 5, 7):
    n = 1511 * 1523 * 1531
    d, a = cost(chain(n, m))
    print(f"{n=} {m=} {d}D {a}A")
    print(show(chain(n, m)))

for m in (0, 1, 3, 5, 7, 9, 11, 13):
    print("M", m)
    ds = []
    a_s = []
    for n in blocks32:
        # print(chain(n))
        d, a = cost(chain(n, m))
        ds.append(d)
        a_s.append(a)
    print(show(chain(n, m)))
    avd = sum(ds) / len(ds)
    ava = sum(a_s) / len(a_s)
    print(f"average D={avd:.2f} A={ava:.2f}")

print("=== 64 bit blocks ===")
for m in (0, 1, 3, 5, 7, 9, 11, 13):
    print("M", m)
    ds = []
    a_s = []
    bits = []
    for n in blocks:
        # print(chain(n))
        d, a = cost(chain(n, m))
        ds.append(d)
        a_s.append(a)
        bits.append(n.bit_length() - 1)
    print(f"Example {n=} {d}D {a}A")
    print(show(chain(n, m)))
    avd = sum(ds) / len(ds)
    ava = sum(a_s) / len(a_s)
    avb = sum(bits) / len(bits)
    print(f"average D={avd:.2f} A={ava:.2f} bits={avb:.2f}")

print(show(chain(1234567890, 7)))

print("=== 128 bit blocks ===")
for m in (0, 1, 3, 5, 7, 9, 11, 13):
    print("M", m)
    ds = []
    a_s = []
    bits = []
    for n in blocks128:
        # print(chain(n))
        d, a = cost(chain(n, m))
        ds.append(d)
        a_s.append(a)
        bits.append(n.bit_length() - 1)
    print(f"Example {n=} {d}D {a}A")
    print(show(chain(n, m)))
    avd = sum(ds) / len(ds)
    ava = sum(a_s) / len(a_s)
    avb = sum(bits) / len(bits)
    print(f"average D={avd:.2f} A={ava:.2f} bits={avb:.2f}")
