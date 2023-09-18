# Class groups of imaginary quadratic fields

It is known that the quadratic sieve provides an efficient
algorithm to compute the structure of class groups of fields
Q(sqrt(-D)).

The utility `ymcls` implements the sieve part of this computation.
Linear algebra is handled by separate scripts using external algebra
libraries.

This program is a work-in-progress hobby and should not be trusted
for serious computations.

Usage:
```
ymcls --threads N DISCRIMINANT OUTPUTDIR
sage -python scripts/classgroup/compute.py OUTPUTDIR
```

## Relation and differences with factorization

Multipliers cannot be used: as a result some "unlucky" integers
will be processed much slower.

The polynomials are definite positive, in particular, their values
are never close to zero, so the smoothness probability is lower.

The class group computation is closely related to factorization,
since 2-torsion elements of the class group are in 1-1 correspondance
with non trivial divisors of the discriminant.

Intermediate computation results contain interesting information
so they should be written to files.

The sieve effectiveness varies depending on how many small primes
are such that the Legendre symbol `(D|l)` is 1. The luckiest
discriminants are such that `h(D)/sqrt(D)` is largest.

The result is unspecified if the discriminant is not a "fundamental"
discriminant.

## Computation output

The output directory will contain the following files:

* `args.json`: a JSON file describing some input parameters

Example:
```json
{
   "d": "-75742003803548105271232793285527826539",
   "h_estimate_min": 1.22220e18,
   "h_estimate_max": 1.22979e18,
}
```

* `relations.sieve`: a large text file containing 1 relation per line.
  Each relation is a space-separated list of primes `p` or `-p` denoting
  a base prime ideal or its conjugate, such that their product is trivial
  in the class group. Primes can be repeated if they appear with some
  exponent in the relation.

Example:
```
-3 -3 13 19 73 -163 -2557 4229 6689 -12659
5 7 -11 17 -107 233 -4751 -9461 -12007 -145723
-11 -31 -59 -523 -1979 2447 4703 72019 -80489 -112207
...
```

* `relations.pruned`: same format as `relations.sieve`, a smaller set of
  relations resulting from pruning step.

* `relations.filtered`: a smaller-size text file containing the relations
  kept for linear algebra steps

Example:
```
3^5 5^-2 11^3 17^1
3^-1 17^3 23^2
```

* `relations.removed`: saved relations during elimination. Each relation
  depends on `relations.filtered` or on previous lines in the same file.

Example:
```
20707 = 3^2 5^-1 13^1 47^-2
20717 = 5^1 7^1 19^-1 97^2
```

* `classnumber`: a one-line file containing the value of the class number

* `group.structure`: a text file containing a decomposition into a
  product of cyclic groups and the coordinates of generators found in
  `relations.filter`.

Example:
```
G 826781660038665892291 2
3 1 0
5 141654769174164315103 1
7 484348438480025487173 0
...
```

* `group.structure.extra`: coordinates for elements of `relations.removed`

Example:
```
20011 639335354182027341157 1
20021 167479567451417040664 0
...
```

## Performance

Since the computation is identical to SIQS, the performance behaves
similarly. However, the parameters are chosen to avoid blowing up
the complexity of the linear algebra step, slowing down the sieve.

The best inputs (discriminant is a square modulo many small primes)
behave as if they were 15-20 bits smaller, and the worst inputs
behave as if they were 15-20 bits bigger.

Approximate execution times for average inputs, depending on discriminant size
are approximately (for 1 CPU core, Zen2 4 GHz):

| Input size | Sieve time | Linear algebra time |
| ---------- | ---------- | ------------------- |
|  100 bits  |    50 ms  |     5ms   |
|  120 bits  |   150 ms  |    20ms   |
|  140 bits  |   300 ms  |    80ms   |
|  160 bits  |   1 s     |   800ms   |
|  180 bits  |   3 s     |     2s    |
|  200 bits  |   15 s    |    15s    |
|  220 bits  |   1 min   |    30s    |
|  240 bits  |   5 mins  |   2 mins  |
|  260 bits  |  15 mins  |   5 mins  |
|  280 bits  |  60 mins  |  15 mins  |
|  300 bits  |   3 hours |  1 hour   |
|  320 bits  |10-20 hours|  2 hours  |

The sieving step and the computation of the class number can use multiple cores.
The output of the sieve is usually less than 250 MiB even for 100-digit inputs.

Some computer algebra systems like PARI implement the Buchmann-McCurley
algorithm which tries to find smooth ideals by random sampling.
This is slower for large discriminants (above 64-80 bits).

## Linear algebra

Unlike integer factorization which only requires linear algebra
over GF(2), to compute the 2-torsion subgroup of the class group,
computing the full class group structure requires computation on large
integer matrices, mainly a determinant computation to obtain the
class number.

The main computation is the class number, obtained as the GCD
of NxN minors of the relation matrix. Following [Kleinjung]
it is computed using 2 determinants, via CRT and the Wiedemann algorithm.

The Wiedemann algorithm is implemented in Python using
scipy uint64 matrices and uses 54-bit moduli.

For small discriminants, an implementation using dense linear algebra
is built in `ymcls` for faster computation.

## Group structure

The group structure (equivalent to Smith normal form of the relation matrix)
is determined by computing the kernel of the relation matrix modulo
prime power divisors of the class number.

Cado-NFS is the fastest option for primes over 55 bits that cannot use
the scipy-based Wiedemann algorithm implementation. A very slow implementation
using generic sparse matrices from SageMath is also available as a fallback.

The current implementation can fail if the group has unusually large p^k torsion.

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

Computations inspired by CSI-Fish:
```
D = 1 - 4*prod(primes(3,248))*661 (a 338-bit prime)
h(D) = 3^4 * 489089703283 * 1378350008771 * 11058447639442327727284973
```

## Bibliography

Henri Cohen, Calcul du nombre de classes d'un corps quadratique imaginaire ou
réel, d'après Shanks, Williams, McCurley, A. K. Lenstra et Schnorr
Séminaire de théorie des nombres de Bordeaux, Série 2, Tome 1 (1989) no. 1, pp. 117-135.
<http://www.numdam.org/item/JTNB_1989__1_1_117_0/>

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
