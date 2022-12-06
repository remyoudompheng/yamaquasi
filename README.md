Yamaquasi is a Rust implementation of the Quadratic sieve and its multiple
polynomial variant (MPQS) aiming at a balance between efficiency and readability.

# Performance

The following benchmarks use balanced semiprimes as input (product of 2 primes
of similar size).

Benchmarks on Ryzen 5500U

| Input size | msieve | flintqs | ymqs (QS) | ymqs | ymqs (6 cores) |
| -------- | ------- | ------- | ------- | ------- | -------------- |
|  40 bits |   2-3ms | —       |  3-4ms  | 25-60ms |  |
|  60 bits |  5-10ms | —       |  3-5ms  |  6-10ms | — |
|  80 bits |  5-10ms | —       |  5-7ms  |  5-10ms | — |
| 100 bits | 30-35ms | —       |10-25ms  | 10-25ms |  8-20ms  |
| 120 bits | 40-50ms | —       |50-150ms | 20-75ms | 15-30ms  |
| 140 bits | 75-90ms |  130-200ms| 0.5-1s|100-300ms| 50-100ms |
| 160 bits | 170-190ms| 300-450ms| 4-8s  |0.4-1.3s | 150-350ms|
| 180 bits | 400-600ms| 0.9-1.1s |  —    | 2.5-7s  | 0.7-2.0s |
| 200 bits | 2.0-3.5s |  3-4s    |  —    |   — |  3-5s |
| 220 bits | 8-12s    | 11-15s   |  —    |   — | 15-30s |
| 240 bits | 25-40s   | 40-60s   |  —    |   — | 100-160s |
| 260 bits | 120s     |   160s   |  —    |   — | — |
| 280 bits | 480s     | 500-600s |  —    |   — | — |
| 300 bits | 1400s    |  2300s   |  —    |   — | — |
| 320 bits | 5400s    | 20700s   |  —    |   — | — |
| RSA-100  | 11400s   |  — | — | — | —  |

flintqs rejects inputs smaller than 40 decimal digits.

The linear algebra implementation of yamaquasi is single-threaded.

# Choice of thresholds and parameters

The chosen parameters received tweaks to accomodate very small
inputs for which the quadratic sieve is not efficient (under 100 bits).

A maximum input size (500 bits) is enforced and data structures assume
that bound.

The sieve assumes that the factor base can be represented by 24-bit
integers (enough for the first million prime numbers).

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

