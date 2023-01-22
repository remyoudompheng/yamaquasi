from random import getrandbits
from sage.all import next_prime, factor, primes

# Cost of exponentiation chains when taking w bits at a time.
def cost(n, w):
    if w == 1:
        sq, mul = 0, 0
    else:
        # compute 1, ... 2**w-1
        sq, mul = 1, 2**(w-1) - 1
    while n >= 0:
        if n % 2 == 0:
            sq += 1
            n = n // 2
        elif n < 2**w:
            break
        else:
            mul += 1
            n = n - (n % 2**w)
    return sq, mul

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

for blksize in (32, 64, 128, 256):
    blks = pblocks(blksize)
    print(f"exponent blocks {len(blks)}x {blksize} bits")
    for w in range(1, 10):
        sqs, muls = 0, 0
        for b in blks:
            sq, mul = cost(b, w)
            sqs += sq
            muls += mul
        print(f"{blksize=} {w=} cost={sqs+muls} ({sqs}S + {muls}M)")

# Study efficiency of Pollard P-1 for random large primes.
for sz in list(range(22, 28)) + [48, 60, 72, 84, 96]:
    print(f"Prime size {sz} bits")
    stats1 = []
    stats2 = []
    powers = 0
    for _ in range(5000):
        p0 = getrandbits(sz)
        p = next_prime(p0)
        ps = [f for f, _ in factor(p-1)]
        for f, k in factor(p-1):
            if k > 1 and f**k > 500:
                #print(f"large factor {f}^{k} in prime {p}")
                powers += 1
        q1, q2 = sorted(ps)[-2:] if len(ps) > 1 else (ps[0], ps[0])
        stats1.append(q1)
        stats2.append(q2)
    l = len(stats1)
    stats1.sort()
    stats2.sort()
    p50, p66, p75 = stats1[-l//2], stats1[-l//3], stats1[-l//4]
    p90, p99 = stats1[-l//10], stats1[-l//100]
    print(f"2nd largest {p50=} {p66=} {p75=} {p90=} {p99=}")
    p50, p66, p75 = stats2[-l//2], stats2[-l//3], stats2[-l//4]
    p90, p99 = stats2[-l//10], stats2[-l//100]
    print(f"largest {p50=} {p66=} {p75=} {p90=} {p99=}")
    print(f"{powers/50:.2}% misses due to small prime powers")

