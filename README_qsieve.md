# Quadratic sieve

## Sample benchmarks

The following benchmarks use balanced semiprimes as input (product of 2 primes
of similar size). The figures below are not guaranteed to follow a rigorous
measurement methodology. Timings are performed against a random sample
of numbers with the specified size. The performance can vary wildly depending
on how many roots are available on small primes.

Usually the performance is very similar (sometimes better) than msieve,
but much slower than YAFU.

THe numbers are relative to the entire program execution and do not reflect
the actual performance of the `factor` function.

Benchmarks on Ryzen 5500U

|Input size| msieve  | flintqs |   QS    |  MPQS   |  SIQS   | SIQS (6 cores) |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- |
|  40 bits |   2-3ms | —       |   1-3ms |   2-4ms |   2-4ms |   5-10ms|
|  60 bits |   4-6ms | —       |   2-4ms |   2-4ms |   2-4ms |   3-10ms|
|  80 bits |   5-8ms | —       |   3-6ms |   3-5ms |   2-5ms |   3-10ms|
| 100 bits | 30-35ms | —       |  7-15ms |  5-15ms |  5-10ms |   5-10ms|
| 120 bits | 40-50ms | —       | 25-50ms | 10-20ms |  10-20ms|   7-15ms|
| 140 bits | 75-90ms |120-160ms|120-200ms| 40-80ms |  40-80ms|  20-30ms|
| 160 bits |150-190ms|300-340ms| 0.6-1.0s|150-350ms|160-250ms|  50-80ms|
| 180 bits | 0.5-0.7s| 0.8-1.1s|    3-6s | 0.8-1.5s| 0.5-0.8s|150-250ms|
| 200 bits | 2.2-3.1s| 2.6-4.9s|  20-35s |   3-5s  | 2.0-3.0s| 0.5-1.0s|
| 220 bits |   9-13s |  13-19s | 90-110s |  15-25s |   8-15s |    2-4s |
| 240 bits |  25-40s |  40-60s |  ~500s  | 60-100s |  30-50s |   8-12s |
| 260 bits | 120-150s| 140-180s|3200-3500s|200-300s| 100-150s|  30-40s |
| 280 bits | 400-650s| 550-700s|    —    |1800-3000s| 350-550s | 100-120s |
| 300 bits |1300-1800s| ~2000s |    —    | ~2 hours |1200-1800s| 350-500s |
| 320 bits |  ~4300s  | ~20000s|    —    |    —     |4800-5400s|1200-2000s|
| RSA-100  | 10500s   | 41400s |    —    |    —     |   8500s  |  2400s   |

RSA-100 is number 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139.

The CPU clock rate is usually slower when multiple cores are active.

`msieve` uses SQUFOF and ordinary quadratic sieve below 85 bits.
`flintqs` rejects inputs smaller than 40 decimal digits.

On platforms supporting SMT (Simultaneous Multi-Threading) a small benefit
can be observed when fully using all available logical cores.

## Implementation choices

The implementation is a "textbook" implementation following the papers mentioned in bibliography.

It features the 3 variants: classical quadratic sieve, multiple polynomial
quadratic sieve, self-initializing (hypercube) multiple polynomial
quadratic sieve.

Several optimization techniques are not used in Yamaquasi: the Gray code enumeration
for SIQS polynomials is not used (instead a base 16 representation avoids excessive
initialization cost), and the formulas avoiding a few inner products in Block Lanczos
are not implemented.

## Case of numbers n=8k+1

The numbers such that `n % 8 == 1` are special because they have modular square
roots modulo every power of 2, providing additional factors, and because
`x^2 - n` is divisible by 8 for every odd `x`.

In MPQS and SIQS this is used by polynomials `Ax²+Bx+C = ((2Ax+B)²-n)/4`
whose values are both twice smaller and always even.

In classical QS this is used by sieving only odd values so that
`(2k+1)²-n` is always divisible by 8.

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

The implementation uses width 64 blocks with `u64` word type. It is considerably
slower than the version used by `msieve` or YAFU, but this is usually not a concern
because it is never a significant part of the algorithm cost.

## Bibliography

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

