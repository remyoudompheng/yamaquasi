# Class groups of imaginary quadratic fields

It is known that the quadratic sieve provides an efficient
algorithm to compute the structure of class groups of fields
Q(sqrt(-D)).

The utility `ymcls` implements the sieve part of this computation.
Linear algebra is handled by separate scripts using external algebra
libraries.

This program is a work-in-progress hobby and should not be trusted
for serious computations.

## Differences with factorization

Multipliers cannot be used: as a result some "unlucky" integers
will be processed much slower.

The polynomials are definite positive, in particular, their values
are never close to zero, so the smoothness probability is lower.

## Performance

Since the computation is identical to SIQS, the performance behaves
similarly. However, the parameters are chosen to avoid blowing up
the complexity of the linear algebra step, slowing down the sieve.

Expected execution times, depending on discriminant size
are approximately (for 1 CPU core):

* 200 bits: less than 1 minute sieve, 1-5mn linear algebra
* 220 bits: about 5 minutes for sieve, 5-10mn for linear algebra
* 250 bits: about 10 minutes for sieve, 20-30mn for linear algebra
* 280 bits: 30-40 minutes for sieve, more than 1 hour for linear algebra
* 300 bits: 2-3 hours for sieve, many hours for linear algebra

Parallel computing applies to the sieve in the same way
as the factoring implementation.

Some computer algebra systems like PARI implement the Buchmann-McCurley
algorithm which tries to find smooth ideals by random sampling.
This is slower for large numbers (above 128 bits).

## Linear algebra

Unlike integer factorization which only requires linear algebra
over GF(2), computing the class group requires computation on large
integer matrices, mainly a determinant computation to obtain the
class number.

For large sizes, the difference between dense linear algebra
libraries and sparse linear algebra routines may be smaller than
expected, and depend highly on the quality of matrix filtering.

To handle linear algebra, 2 methods are suggested:

* using SageMath with FLINT dense linear algebra routines
* calling into Cado-NFS `bwc` utility to apply Block Wiedemann
  algorithm to a sparse matrix, using Victor Pan's reduction of
  determinants to linear system solutions via Cramer's rule

Note that the interface of the `bwc` utility is an internal detail of
Cado-NFS implementation which is not meant to compute determinants,
and may change in incompatible ways in the future.

The other linear algebra steps are less computationally intensive.

## Examples

Small random numbers:
```
h(-672772578839) = 959482
h(-560979400532204839) = 594097986
h(-367908113612190744468907) = 73361502182
h(-835530720420926898479116136119) = 589709265061998
```

Example Δ5 from Jacobson's original article:
```
h(-4695204673552245130677413787157851216622805893433443043026971349460603)
= 18410786327386854870530065299887240
```

## Bibliography

Michael Jacobson, Applying sieving to the computation of class groups
Math. Comp. 68 (226), 1999, 859-867
<https://www.ams.org/journals/mcom/1999-68-226/S0025-5718-99-01003-0/S0025-5718-99-01003-0.pdf>

Jean-François Biasse, Improvements to the computation
of ideal class groups of imaginary quadratic number fields
<https://arxiv.org/pdf/1204.1300.pdf>

Thorsten Kleinjung, Quadratic Sieving
Math. Comp. 85 (300), 2016, 1861-1873
<https://www.ams.org/journals/mcom/2016-85-300/S0025-5718-2015-03058-0/S0025-5718-2015-03058-0.pdf>

Victor Pan, Computing the determinant and the characteristic polynomial of a matrix via solving linear systems of equations,
Information Processing Letters, Volume 28, Issue 2, 1988
<https://doi.org/10.1016/0020-0190(88)90166-4>

Erich Kaltofen, Gilles Villard. On the complexity of computing determinants
LIP RR-2003-36, Laboratoire de l’informatique du parallélisme. 2003, 2+35p. hal-02102099
<https://hal-lara.archives-ouvertes.fr/hal-02102099/file/RR2003-36.pdf>
