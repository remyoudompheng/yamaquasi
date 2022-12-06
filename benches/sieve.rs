use brunch::Bench;
use std::str::FromStr;
use yamaquasi::arith::isqrt;
use yamaquasi::Uint;
use yamaquasi::{fbase, mpqs};

const PQ128: &str = "138775954839724585441297917764657773201";
const PQ256: &str =
    "104567211693678450173299212092863908236097914668062065364632502155864426186497";

brunch::benches! {
    // Eratosthenes sieve
    Bench::new("sieve 1000 primes")
    .run_seeded(1000, fbase::primes),
    Bench::new("sieve 10000 primes")
    .run_seeded(10000, fbase::primes),
    Bench::new("sieve 50000 primes")
    .run_seeded(50000, fbase::primes),
    // Polynomial selection
    {
        let n = Uint::from_str(PQ256).unwrap();
        let mut polybase: Uint = isqrt(n >> 1) >> 24;
        polybase = isqrt(polybase);
        Bench::new("select_polys(256-bit n) = Some(...)")
        .run_seeded(n, |n| mpqs::select_poly(polybase, 0, n))
    },
    // Mass polynomial selection
    // Generate 1000 polys, density is 1 / 2(log polybase)
    // = log 2 / 2 log2 n ~ 7/ 20 log2(n)
    {
        let n = Uint::from_str(PQ128).unwrap();
        let mut polybase: Uint = isqrt(n >> 1) >> 24;
        polybase = isqrt(polybase);
        let width = 100 * 20 / 7 * polybase.bits() as usize;
        Bench::new("select 100 polys 128-bit n")
        .run_seeded(n, |n| {
            let v = mpqs::select_polys(polybase, width, &n);
            assert!(90 < v.len() && v.len() < 110);
        })
    },
    {
        let n = Uint::from_str(PQ256).unwrap();
        let mut polybase: Uint = isqrt(n >> 1) >> 24;
        polybase = isqrt(polybase);
        let width = 100 * 20 / 7 * polybase.bits() as usize;
        Bench::new("select 100 polys 256-bit n")
        .run_seeded(n, |n| {
            let v = mpqs::select_polys(polybase, width, &n);
            assert!(90 < v.len() && v.len() < 110);
        })
    },
    // Prepare primes
    {
        let n = Uint::from_str(PQ256).unwrap();
        let primes = fbase::primes(10000);
        let fb = fbase::prepare_factor_base(&n, &primes[..]);
        let polybase: Uint = isqrt(isqrt(n));
        let pol = mpqs::select_poly(polybase, 0, n);
        Bench::new("prepare 5000 primes for poly (n: 256 bit)")
        .run_seeded((&pol, &fb), |(pol, fb)| fb.iter().map(|p| pol.prepare_prime(p)).collect::<Vec<_>>())
    }
}
