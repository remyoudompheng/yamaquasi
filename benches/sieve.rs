use brunch::Bench;
use std::str::FromStr;
use yamaquasi::poly::{primes, select_polys};
use yamaquasi::Uint;

const PQ256: &str =
    "104567211693678450173299212092863908236097914668062065364632502155864426186497";

brunch::benches! {
    // Eratosthenes sieve
    Bench::new("sieve 10000 primes")
    .run_seeded(10000, primes),
    Bench::new("sieve 50000 primes")
    .run_seeded(50000, primes),
    // Polynomial selection
    {
        let n = Uint::from_str(PQ256).unwrap();
        Bench::new("select_polys(256-bit n) = Some(...)")
        .run_seeded(n, |n| select_polys(25, 0, n))
    },
}
