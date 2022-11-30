import sys
from random import randint
from gmpy2 import mpz, next_prime, is_prime


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} N_BITS")
        exit(1)
    size = int(sys.argv[1])
    seed = mpz(randint(2 ** (size // 2 - 1), 2**(size // 2)))
    primes = []
    while len(primes) < 2:
        seed = next_prime(seed)
        if is_prime(seed // 2):
            primes.append(int(seed))
            seed = mpz(randint(2 ** (size // 2 - 1), 2**(size // 2)))
    p, q = primes
    print(f"{p =}")
    print(f"{q =}")
    print(f"n = {p*q} {(p*q).bit_length()} bits")


if __name__ == "__main__":
    main()
