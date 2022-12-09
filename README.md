Yamaquasi is a Rust implementation of several variants of the Quadratic sieve
factorisation method. It attempts to balance efficiency and readability.

# Performance

The following benchmarks use balanced semiprimes as input (product of 2 primes
of similar size).

Benchmarks on Ryzen 5500U

|Input size| msieve  | flintqs |   QS    |  MPQS   |  SIQS   | MPQS (6 cores) |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- |
|  40 bits |   2-3ms | —       |   4-6ms | 40-80ms |  5-7ms  |    —    |
|  60 bits |   4-6ms | —       |   3-5ms |  6-10ms |  5-7ms  |    —    |
|  80 bits |   5-8ms | —       |   6-8ms |  5-10ms |  12-15ms|    —    |
| 100 bits | 30-35ms | —       | 15-30ms | 10-20ms |  15-35ms|   8-20ms|
| 120 bits | 40-50ms | —       |70-170ms | 30-60ms |  40-70ms|  15-30ms|
| 140 bits | 75-90ms |120-160ms| 0.5-1.3s|140-230ms|150-250ms| 50-100ms|
| 160 bits |150-190ms|300-340ms|    3-9s | 0.4-1.1s|500-900ms|200-300ms|
| 180 bits | 0.5-0.7s| 0.8-1.1s|  20-50s | 2.5-4.5s| 2.3-3.1s| 0.7-1.3s|
| 200 bits | 2.2-3.1s| 2.6-4.9s|140-220s | 10-20s  | 9.5-14s |  3-4.5s |
| 220 bits | 10-14s  |  13-19s | ~1000s  |  ~100s  |  45-55s |  15-25s |
| 240 bits | 25-40s  | 40-60s  |    —    |    —    |   ­     |100-160s |
| 260 bits | 120s    |   160s  |    —    |    —    |   ­     |    —    |
| 280 bits | 480s    |   ~500s |    —    |    —    |   ­     |    —    |
| 300 bits | 1400s   |  ~2000s |    —    |    —    |   ­     |    —    |
| 320 bits | 5400s   | ~20000s |    —    |    —    |   ­     |    —    |
| RSA-100  | 11400s  |  41400s |    —    |    —    |   ­     |    —    |

`msieve` uses SQUFOF and ordinary quadratic sieve below 85 bits.
`flintqs` rejects inputs smaller than 40 decimal digits.

The linear algebra implementation of yamaquasi is single-threaded.

# Choice of thresholds and parameters

The chosen parameters received tweaks to accomodate very small
inputs (under 100 bits) for which the quadratic sieve
is not the most effective method.

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

[W. R. Alford, Carl Pomerance, Implementing the self-initializing quadratic sieve
](https://math.dartmouth.edu/~carlp/implementing.pdf)

[Wikipedia page](https://en.wikipedia.org/wiki/Quadratic_sieve)

## Integer arithmetic

Since it is unreasonable to use the quadratic sieve to factor numbers larger
than 512 bits, yamaquasi uses fixed width 1024-bit arithmetic for
modular arithmetic provided by the `bnum` crate, and 64-bit arithmetic
for computations involving the factor base.

## Polynomial selection (MPQS)

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

