"""
Generates random composite numbers with unbalanced primes.
This is useful as a ECM test script.
"""

import sys
from random import randint
from gmpy2 import mpz, next_prime, is_prime

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} N_BITS")
        exit(1)
    size = int(sys.argv[1])
    primes = []
    prod = 1
    while prod.bit_length() < size:
        rem = size - prod.bit_length()
        if rem < 64:
            psize = rem + 1
        else:
            psize = randint(rem // 3, 3 * rem // 4)
        seed = mpz(randint(2 ** (psize - 1), 2**psize))
        seed = next_prime(seed)
        prod *= seed
        primes.append(int(seed))
    print(f"{primes =}", file=sys.stderr)
    print(f"n = {prod} {prod.bit_length()} bits", file=sys.stderr)
    print(prod)


if __name__ == "__main__":
    main()
