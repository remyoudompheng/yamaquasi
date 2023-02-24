# ECM implementation

To make it easier to factorize general numbers (not necessarily balanced
products of 2 primes), an implementation of ECM using Edwards curves
is provided. It partially follows the EECM paper (Bernstein-Birkner-Lange-Peters)
available at https://eecm.cr.yp.to/index.html

It uses twisted Edwards curves from the Suyama-11 family, which have
guaranteed order divisible by 12, and a fallback on a
family of curves with rational Z/2 x Z/4 torsion which is not optimal
but has a very simple definition.

The implementation uses several state-of-the-art techniques such as
Montgomery arithmetic, Schonhage-Strassen FFT multiplication, and
scaled remainder trees for multipoint polynomial evaluation.
It will usually be 2x-3x slower than GMP-ECM.

The ECM implementation is run when using the "automatic" factoring mode,
with very low parameters (that can easily catch factors with 1/5th
of input integer digits).

## Small modulus variant

A simplified version using 128-bit arithmetic, is also provided.

It also uses twisted Edwards curves from the Suyama-11 family containing
a guaranteed torsion subgroup of order 12, which makes ECM especially quick
for small prime factors.

Because it only handles small values, it doesn't use any FFT for stage 2.

It is used for general factoring (even for products of 2 primes with
similar size) and can factor a 64-bit integer in less than 600k CPU
cycles on a relatively modern x86-64 CPU.

## Limitations

The ECM implementation assumes the same bounds as the rest of the program.
In particular, only input integers below 512 bits are accepted.

## Bibliography

Peter. L. Montgomery. An FFT extension of the elliptic curve method of factorization.
PhD thesis, University of California, 1992.

[B. Dodson, P. Zimmermann, 20 Years of ECM](https://hal.archives-ouvertes.fr/inria-00070192)

[G. Hanrot, M. Quercia, P. Zimmermann, The Middle Product Algorithm I](https://hal.inria.fr/inria-00071921)

[D.J. Bernstein, P. Birkner, T. Lange, C. Peters, ECM Using Edwards curves](https://eecm.cr.yp.to/eecm-20111008.pdf)

[R. Barbulescu, J.W. Bos, C. Bouvier, T. Kleinjung, P.L. Montgomery,
Finding ECM-friendly curves through a study of Galois properties 
](https://hal.archives-ouvertes.fr/hal-00671948)

[Bos, Kleinjung, ECM at Work](https://eprint.iacr.org/2012/089)
