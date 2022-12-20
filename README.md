Yamaquasi is a Rust implementation of several variants of the Quadratic sieve
factorisation method. It attempts to balance efficiency and readability.

# Performance

The following benchmarks use balanced semiprimes as input (product of 2 primes
of similar size). The figures below are not guaranteed to follow a rigorous
measurement methodology. Timings are performed against a random sample
of numbers with the specified size.

Benchmarks on Ryzen 5500U

|Input size| msieve  | flintqs |   QS    |  MPQS   |  SIQS   | SIQS (6 cores) |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- |
|  40 bits |   2-3ms | —       |   3-5ms | 15-40ms |  5-10ms |   8-15ms|
|  60 bits |   4-6ms | —       |   3-5ms |  5-10ms |  5-10ms |   5-10ms|
|  80 bits |   5-8ms | —       |   6-8ms |  5-10ms |  5-10ms |  12-15ms|
| 100 bits | 30-35ms | —       | 10-25ms | 10-20ms |  8-20ms |  10-35ms|
| 120 bits | 40-50ms | —       | 30-80ms | 25-50ms |  20-40ms|  20-50ms|
| 140 bits | 75-90ms |120-160ms| 0.2-0.5s| 80-200ms| 70-130ms| 50-150ms|
| 160 bits |150-190ms|300-340ms|  1.5-3s |300-800ms|300-400ms|120-300ms|
| 180 bits | 0.5-0.7s| 0.8-1.1s|   7-15s | 1.5-4.0s| 1.0-1.8s| 0.4-0.9s|
| 200 bits | 2.2-3.1s| 2.6-4.9s|  30-70s |   7-15s |   4-7s  | 1.7-2.5s|
| 220 bits |  10-14s |  13-19s | 180-270s|  40-80s |  15-25s |    6-9s |
| 240 bits |  25-40s |  40-60s |750-1200s| 150-300s|  60-90s |  30-40s |
| 260 bits | 120-150s| 140-180s|3200-3500s|450-800s| 240-320s| 90-120s |
| 280 bits | 400-650s| 550-700s|    —    |1800-3000s| 900-1300s| 450-700s|
| 300 bits | 1400s   |  ~2000s |    —    | ~2 hours |2800-3800s|  ~1200s |
| 320 bits | 5400s   | ~20000s |    —    |    —     | ~15000s  |  ~5500s |
| RSA-100  | 11400s  |  41400s |    —    |    —     |  23000s  |   7450s |

The CPU clock rate is usually slower when multiple cores are active.

`msieve` uses SQUFOF and ordinary quadratic sieve below 85 bits.
`flintqs` rejects inputs smaller than 40 decimal digits.

The linear algebra implementation of yamaquasi is single-threaded.
There is no plan to use architecture-specific inline assembly,
but unsafe Rust or crates providing optimizations (such as
SIMD-enabled routines) are used.

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

[Brian Carrier, Samuel G. Wagstaff, Implementing the Hypercube Quadratic Sieve
with Two Large Primes](https://homes.cerias.purdue.edu/~ssw/qs4.pdf)

[Wikipedia page](https://en.wikipedia.org/wiki/Quadratic_sieve)

It features the 3 variants: classical quadratic sieve, multiple polynomial
quadratic sieve, self-initializing (hypercube) multiple polynomial
quadratic sieve.

## Integer arithmetic

Since it is unreasonable to use the quadratic sieve to factor numbers larger
than 512 bits, yamaquasi uses fixed width 1024-bit arithmetic for
modular arithmetic provided by the `bnum` crate, and 64-bit arithmetic
for computations involving the factor base. Polynomial coefficients are
guaranteed to be smaller than 256 bits.

## Polynomial selection (MPQS)

The polynomial selection looks for pseudoprimes D such that the input number
N is a quadratic residue modulo D. The search uses slight sieving to
find multiple appropriate values of D in a single pass.

## Parallel processing

Yamaquasi uses the `rayon` Rust crate to provide parallel computation capabilities.

In classical quadratic sieve, 2 threads can process the forward sieve
interval `[0,M]` and the backward sieve interval `[-M,0]` simultaneously.

It is used in the MPQS and the SIQS/HMPQS implementation to process a batch
of polynomials over a thread pool.

# Memory usage

Implementation choices do not attempt to aggressively reduce memory usage
and targets home devices with fewer than 16 cores and more than 4GB of memory.
In particular it does not write relations to disk, but keeps them entirely
in memory. Data structures are optimized only for cache-sensitive parts of the code.

## Linear algebra

Kernel computation for mod 2 matrices is done through a naïve Gauss reduction
using bit vectors from crate `bitvec_simd`. It will typically take less than
1 second for a size 5000 matrix, with a O(n³) complexity.

Above size 5000, the Block Lanczos algorithm is used. However, the matrices
D, E, F from Montgomery's article are not used: instead, the vectors are chosen
to optimistically achieve the correct condition for O(n²) complexity
using simple formulas. The implementation uses blocks of 256 vectors
using 256-bit wide variables provided by the `wide` crate.

# Bugs

The program is not guaranteed to provide a factorization for all inputs in
the supported range due to invariant failures or incorrect parameter choices.
