Yamaquasi is a Rust implementation of the Quadratic sieve and its multiple
polynomial variant (MPQS) aiming at a balance between efficiency and readability.

# Performance

The following benchmarks use balanced semiprimes as input (product of 2 primes
of similar size).

Benchmarks on Ryzen 5500U

| Input size | msieve | flintqs | ymqs | ymqs (6 cores) |
| --- | --- | --- | --- | --- |
|  40 bits |   2-3ms | — |  15-30ms | — |
|  60 bits |  5-10ms | — |   5-10ms | — |
|  80 bits |  5-10ms | — |   5-10ms | — |
| 100 bits | 30-35ms | — |  10-30ms | 10-150ms |
| 120 bits | 40-50ms | — | 30-130ms | 15-70ms |
| 140 bits | 75-90ms |  130-200ms| 100-400ms | 60-200ms |
| 160 bits | 170-190ms| 300-450ms| 0.5-2.5s | 0.2-1.2s |
| 180 bits | 400-600ms| 0.9-1.1s | 3-10s | 1-5s |
| 200 bits | 2.0-3.5s |  3-4s    | 10-40s | 5-20s |
| 220 bits | 8-12s    | 11-15s   | 60-100s | 15-60s |
| 240 bits | 25-40s   | 40-60s   | — | 100-160s |
| 260 bits | — | — | — | 1000s |

flintqs rejects inputs smaller than 40 decimal digits.

The linear algebra implementation of yamaquasi is single-threaded.

# Choice of thresholds and parameters

The chosen parameters received tweaks to accomodate very small
inputs for which the quadratic sieve is not efficient (under 100 bits).

# Implementation

The implementation is a "textbook" implementation following the papers:

[Carl Pomerance, A Tale of Two Sieves
](https://www.ams.org/notices/199612/pomerance.pdf)

[J. Gerver, Factoring Large Numbers with a Quadratic Sieve
](https://www.jstor.org/stable/2007781)

[Peter L. Montgomery, A Block Lanczos Algorithm for Finding Dependencies over GF(2)
](https://doi.org/10.1007/3-540-49264-X_9)

[Robert D. Silverman, The multiple polynomial quadratic sieve
,Math. Comp. 48, 1987](https://doi.org/10.1090/S0025-5718-1987-0866119-8)

[Wikipedia page](https://en.wikipedia.org/wiki/Quadratic_sieve)

## Integer arithmetic

Since it is unreasonable to use the quadratic sieve to factor numbers larger
than 512 bits, yamaquasi uses fixed width 1024-bit arithmetic for
modular arithmetic provided by the `bnum` crate, and 64-bit arithmetic
for computations involving the factor base.

## Polynomial selection

The polynomial selection looks for pseudoprimes D such that the input number
N is a quadratic residue modulo D. The search uses slight sieving to
find multiple appropriate values of D in a single pass.

## Linear algebra

Kernel computation for mod 2 matrices is done through a naïve Gauss reduction
using bit vectors from crate `bitvec_simd`. It will typically take less than
1 second for a size 5000 matrix, with a O(n³) complexity.

Above size 5000, the Block Lanczos algorithm is used. However, the matrices
D, E, F from Montgomery's article are not used: instead, the vectors are chosen
to optimistically achieve the correct condition for O(n²) complexity
using simple formulas.

