"""
Enumerate NTT friendly primes and primitive roots of unity with order 2^32.
"""

from gmpy2 import is_prime

for p in range(1 + 2**58, 2**59, 2**49):
    if not is_prime(p):
        continue
    # Find an order 2^k element
    g2k = 0
    for g in range(2, 10000):
        if pow(g, 2**64, p) != 1:
            continue
        for k in range(64, 0, -1):
            if pow(g, 2**k, p) == p - 1:
                g2k = pow(g, 2 ** (k - 31), p)
                assert pow(g2k, 2**32, p) == 1
                assert pow(g2k, 2**31, p) == p - 1
                break
        if g2k:
            break
    print(hex(p), hex(g2k), "has order 2^32")
    print(f"({hex(p)}, {hex(g2k)}),")
