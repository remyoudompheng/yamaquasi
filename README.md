Yamaquasi is a Rust implementation of several factoring algorithms with
a focus on the Quadratic sieve factorisation method. It is a hobby project
attempting to balance efficiency and readability, with references to research
articles when applicable.

# Description

Yamaquasi can be used as a general-purpose factoring utility through command `ymqs`.

It implements several variants of the Quadratic sieve factoring method with light
boilerplate allowing it to completely factor most integers with fewer than
110 decimal digits. In many cases larger numbers (up to 150 digits) can also
be factored thanks to the ECM method.

There is no guarantee to obtain a successful factorization even when input size
is small enough due to edge cases (non squarefree integers, small factors,
many factors), but most integers will be processed successfully.

Utilities like YAFU (https://github.com/bbuhrow/yafu) can be used for more
reliable or efficient factoring.

# Usage

To compile Yamaquasi, install Rust [https://rust-lang.org] and
build command `ymqs` using:

```
cargo install --root . --path . --bin ymqs
```

To factor an integer:

```
$ bin/ymqs 803469022129495137770981046170581301261101496891396417650687
[various logs to stderr]
164504919713
4884164093883941177660049098586324302977543600799
```

Factor an integer using only ECM:

```
$ bin/ymqs --mode ecm 803469022129495137770981046170581301261101496891396417650687
[various logs to stderr]
164504919713
4884164093883941177660049098586324302977543600799
```

# Python module

An experimental Python module is available. You can build it
with `maturin`:

```
maturin build -r -m pymqs/Cargo.toml -o OUTDIR
```

It has a single function `factor`:

```python
>>> import pymqs
>>> pymqs.factor(803469022129495137770981046170581301261101496891396417650687)
>>> pymqs.factor(803469022129495137770981046170581301261101496891396417650687, algo="ecm", verbose="info")
```

The `verbose` argument accepts values `silent` (default), `info`, `verbose`.

The `timeout` argument accepts an optional numerical value in seconds
and the library will tentatively stop computation after that duration
and possibly return a partial factorization.

The Python API returns factors as a list of Python integers. This is similar
to SageMath API `ecm.factor()` but not SageMath `factor()` which returns
a factorization object. When not using a timeout, the elements of the list
are expected to be pseudoprimes.

# Available factoring algorithms

The main algorithms used by the default strategy are:
- Pollard's Rho algorithm: it is efficient for numbers under 52 bits, with complexity O(sqrt(p))
  where p is the smallest prime factor
- ECM: the variant using 128-bit arithmetic is efficient for number under 80 bits
- Self-initializing quadratic sieve, suitable from 80 to 330 bits

Other algorithms are available but may have limited use for general integers:
- The P-1 algorithm, using the efficient chrip-z transform for stage 2, which
  cannot find general factors
- The P+1 algorithm, using the traditional (suboptimal) FFT continuation, which is actually
  less effective than ECM in most cases, in addition to being restricted to specific types of factors
  and requiring several retries
- Shanks' SQUFOF, previously used in the double large prime variation of the
  quadratic sieve, but Pollard's Rho is usually more efficient

Most algorithms have been tuned so that they also work outside their ideal
range (for example, they can factor numbers under 10 digits).

# Performance

## Orders of magnitude

When using direct library calls, the performance (measured on a Zen2 core)
is as follows:

- Pollard's rho can factor a 32-bit number in less than 30k CPU cycles
  and a 48-bit number in less than 150k CPU cycles
- ECM can factor a 64-bit number in 600k CPU cycles on average (a fraction of millisecond)
- SIQS can factor a 128-bit semiprime in less than 100M CPU cycles
  (25ms at 4GHz), a 192-bit semiprime in about a second,
  a 256-bit semiprime in about a minute, a 320-bit integer in a couple of hours.

For numbers under 128-bit, calling the Python API will usually be faster than SageMath
`factor()` or `ecm.factor()` with default options.

## Multiprecision arithmetic

Yamaquasi includes various homemade multiprecision routines which are used intensively
in the P-1 and ECM algorithms.

Multiplication can be sped up on `x86-64` using the `mulx` instruction. This is
done automatically by the Rust compiler when enabling `target-feature=+bmi2`
on supported CPUs (or microarchitecture level `target-cpu=x86-64-v3`). The improvement
may range from 0% to 10%.

## Vectorization

The polynomial roots step of SIQS is written to benefit from vectorization.
On architecture `x86-64` you can compile with `RUSTFLAGS='-C target-feature=+avx2'`
or `target-cpu=x86-64-v3` to use AVX2 which can often be faster.

The linear algebra step common to quadratic sieve variants can also benefit
from AVX2 through LLVM automatic vectorization (possibly 5%-10% improvement).
The NTT implementation with u64-sized primes also benefits from AVX2 auto-vectorization.

On `aarch64` architecture, the NEON feature is usually enabled by default.

# Choices

To keep a good balance with readability Yamaquasi does not include inline assembly
nor architecture-specific code. However optimization choices are guided by tests
on desktop x64-64 platforms (less than 10 cores with a few MB of L3 cache), and
unsafe Rust is used to skip bound checks or force using SIMD.

Parameter choices were tested on integers from 40 to 330 bits. By default, the command
line utility will combine this with trial division to factor arbitrary inputs.

Input integers over 500 bit size cannot be factored in a reasonable time
using quadratic sieve methods and will be rejected. Internal data structures
assume that this upper bound is enforced.

The purpose of Yamaquasi is to reinvent the wheel in various ways and thus
it often contains reimplementations instead of importing libraries.

# Notes on quadratic sieve implementation

See dedicated [README](README_qsieve.md)

# Notes on ECM implementation

See dedicated [README](README_ecm.md)
