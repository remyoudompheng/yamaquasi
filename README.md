Yamaquasi is a Rust implementation of several variants of the Quadratic sieve
factorisation method. It attempts to balance efficiency and readability.

# Description

Yamaquasi can be used as a general-purpose factoring utility through command `ymqs`.

It implements several variants of the Quadratic sieve factoring method with light
boilerplate allowing it to completely factor most integers with fewer than
110 decimal digits.

There is no guarantee to obtain a successful factorization even when input size
is small enough due to edge cases (non squarefree integers, small factors,
many factors), but most integers will be processed successfully.

Utilities like YAFU (https://github.com/bbuhrow/yafu) can be used for more
reliable or efficient factoring.

# Choices

To keep a good balance with readability Yamaquasi does not include inline assembly
nor architecture-specific code. However optimization choices are guided by tests
on desktop x64-64 platforms (less than 10 cores with a few MB of L3 cache), and
unsafe Rust is used to skip bound checks or force using SIMD.

There is no plan to implement non-quadratic sieve methods (such as ECM or Pollard P-1).
A SQUFOF implementation is used internally for small integers.

Parameter choices were tested on integers from 40 to 330 bits. By default, the command
line utility will combine this with trial division to factor arbitrary inputs.

Input integers over 500 bit size cannot be factored in a reasonable time
using quadratic sieve methods and will be rejected. Internal data structures
assume that this upper bound is enforced.

# Benchmarks

The following benchmarks use balanced semiprimes as input (product of 2 primes
of similar size). The figures below are not guaranteed to follow a rigorous
measurement methodology. Timings are performed against a random sample
of numbers with the specified size.

Benchmarks on Ryzen 5500U

|Input size| msieve  | flintqs |   QS    |  MPQS   |  SIQS   | SIQS (6 cores) |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- |
|  40 bits |   2-3ms | —       |   3-5ms | 15-40ms |  5-20ms |   5-30ms|
|  60 bits |   4-6ms | —       |   3-5ms |  5-10ms |  5-10ms |   5-15ms|
|  80 bits |   5-8ms | —       |   6-8ms |  5-10ms |  5-10ms |   5-15ms|
| 100 bits | 30-35ms | —       | 10-25ms | 10-20ms |  8-20ms |   5-15ms|
| 120 bits | 40-50ms | —       | 30-80ms | 25-50ms |  20-30ms|  15-25ms|
| 140 bits | 75-90ms |120-160ms| 0.2-0.5s| 80-200ms|  60-90ms|  40-60ms|
| 160 bits |150-190ms|300-340ms|  1.5-3s |300-800ms|200-300ms| 80-150ms|
| 180 bits | 0.5-0.7s| 0.8-1.1s|   7-15s | 1.5-4.0s| 0.8-1.1s| 0.3-0.4s|
| 200 bits | 2.2-3.1s| 2.6-4.9s|  30-70s |   7-15s | 3.5-5.0s| 1.5-2.0s|
| 220 bits |  10-14s |  13-19s | 180-270s|  40-80s |  11-20s |    4-7s |
| 240 bits |  25-40s |  40-60s |750-1200s| 150-300s|  40-70s |  16-24s |
| 260 bits | 120-150s| 140-180s|3200-3500s|450-800s| 150-200s|  40-60s |
| 280 bits | 400-650s| 550-700s|    —    |1800-3000s| 550-700s | 160-180s |
| 300 bits |1300-1800s|  ~2000s |    —    | ~2 hours |1400-2000s| 450-550s |
| 320 bits | 5400s   | ~20000s |    —    |    —     |  ~7200s  |1900-2500s|
| RSA-100  | 11400s  |  41400s |    —    |    —     |  16400s  |  4650s |

RSA-100 is number 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139.

The CPU clock rate is usually slower when multiple cores are active.

`msieve` uses SQUFOF and ordinary quadratic sieve below 85 bits.
`flintqs` rejects inputs smaller than 40 decimal digits.

On platforms supporting SMT (Simultaneous Multi-Threading) a small benefit
can be observed when fully using all available logical cores.

On x86-64 architectures, several parts of the code can be accelerated
by enabling AVX2 using `RUSTFLAGS='-C target-cpu=native'`
or `RUSTFLAGS='-C target-feature=+avx2'`.

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

[A. K. Lenstra, M. S. Manasse, Factoring with two large primes
](https://doi.org/10.1090/S0025-5718-1994-1250773-9)

[Brian Carrier, Samuel G. Wagstaff, Implementing the Hypercube Quadratic Sieve
with Two Large Primes](https://homes.cerias.purdue.edu/~ssw/qs4.pdf)

[Wikipedia page](https://en.wikipedia.org/wiki/Quadratic_sieve)

It features the 3 variants: classical quadratic sieve, multiple polynomial
quadratic sieve, self-initializing (hypercube) multiple polynomial
quadratic sieve.

Several optimization techniques are not used in Yamaquasi: the Gray code enumeration
for SIQS polynomials is not used (instead a base 16 representation avoids excessive
initialization cost), and the formulas avoiding a few inner products in Block Lanczos
are not implemented.

## Integer arithmetic

Since it is unreasonable to use the quadratic sieve to factor numbers larger
than 512 bits, yamaquasi uses fixed width 1024-bit arithmetic for
modular arithmetic provided by the `bnum` crate, and 64-bit arithmetic
for computations involving the factor base. Polynomial coefficients are
guaranteed to be smaller than 256 bits.

## Multithread support

Yamaquasi uses the `rayon` Rust crate to provide parallel computation capabilities.

In classical quadratic sieve, 2 threads can process the forward sieve
interval `[0,M]` and the backward sieve interval `[-M,0]` simultaneously.

It is used in the MPQS and the SIQS/HMPQS implementation to process a batch
of polynomials over a thread pool.

## Memory usage

Yamaquasi does not write any relations to disk and keeps everything in memory
in a compact format. Memory usage will be less than 1GB even when factoring
100-digit inputs.

## Linear algebra

Kernel computation for mod 2 matrices is done through a naïve Gauss reduction
using bit vectors from crate `bitvec_simd`. It will typically take less than
a fraction of second for a size 5000 matrix, with a O(n³) complexity.

Above size 5000, the Block Lanczos algorithm is used (complexity O(n²), without
the optimized matrices `Di`, `Ei`, `Fi` from Montgomery's article that avoid
several inner products of blocks.
The implementation uses width 64 blocks with `u64` word type.

