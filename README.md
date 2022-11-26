# Fixed-size arithmetic

Since it is unreasonable to use the quadratic sieve to factor numbers larger
than 512 bits, yamaquasi uses 512-bit arithmetic for exact computations
and 8-bit precision logarithms during the sieving process.

The block size is fixed to consume 512kB of memory per core.

