import sys
from random import randint
from gmpy2 import mpz, next_prime, is_prime


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} N_BITS")
        exit(1)
    size = int(sys.argv[1])
    primes = []
    while len(primes) < 2:
        seed = mpz(randint(2 ** (size // 2 - 1), 2**(size // 2)))
        seed = next_prime(seed)
        primes.append(int(seed))
    p, q = primes
    print(f"{p =}", file=sys.stderr)
    print(f"{q =}", file=sys.stderr)
    print(f"n = {p*q} {(p*q).bit_length()} bits", file=sys.stderr)
    print(p * q)


if __name__ == "__main__":
    main()
