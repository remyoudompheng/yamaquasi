use brunch::Bench;
use std::str::FromStr;
use yamaquasi::arith::isqrt;
use yamaquasi::Uint;
use yamaquasi::{fbase, mpqs, qsieve, sieve, siqs};

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
        Bench::new("prepare 5000 primes for MPQS poly (n: 256 bit)")
        .run_seeded((&pol, &fb), |(pol, fb)| fb.iter().map(|p| pol.prepare_prime(p)).collect::<Vec<_>>())
    },
    // SIQS primitives
    {
        // Prepare 40 A (amortized cost for 10000 polynomials)
        let n = Uint::from_str(PQ256).unwrap();
        let primes = fbase::primes(10000);
        let fb = fbase::prepare_factor_base(&n, &primes[..]);
        Bench::new("prepare 50 A values for SIQS (n = 256 bits)")
        .run_seeded((&fb, &n), |(fb, n)| {
            let f = siqs::select_siqs_factors(fb, n, 9);
            siqs::prepare_as(&f, fb, 40);
        })
    },
    {
        // Fully prepare one polynomial for a given A.
        // It is 6 times faster than MPQS preparation.
        let n = Uint::from_str(PQ256).unwrap();
        let primes = fbase::primes(10000);
        let fb = fbase::prepare_factor_base(&n, &primes[..]);
        let f = siqs::select_siqs_factors(&fb[..], &n, 9);
        let a_s = siqs::prepare_as(&f, &fb, 40);
        Bench::new("prepare 1 SIQS polynomial (n = 256 bits)")
        .run_seeded((&n, &fb, a_s.first().unwrap()), |(n, fb, a)| {
            siqs::make_polynomial(n, fb, a, 123);
        })
    },
    // Block sieve
    {
        let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
        let primes = fbase::primes(5133);
        let fb = fbase::prepare_factor_base(&n, &primes[..]);
        let nsqrt = isqrt(n);
        let s = qsieve::init_sieves(&fb, nsqrt).0;
        Bench::new("clone sieve structure (no-op)")
        .run_seeded(s, |s| {
            let _ =  s.clone();
        })
    },
    {
        let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
        let primes = fbase::primes(5133);
        let fb = fbase::prepare_factor_base(&n, &primes[..]);
        let nsqrt = isqrt(n);
        let s = qsieve::init_sieves(&fb, nsqrt).0;
        Bench::new("sieve 32k block with ~2500 primes")
        .run_seeded(s, |s| {
            let mut s1 = s.clone();
            s1.sieve_block();
            s1.next_block();
        })
    },
    {
        let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
        let primes = fbase::primes(20000);
        let fb = fbase::prepare_factor_base(&n, &primes[..]);
        let nsqrt = isqrt(n);
        let s = qsieve::init_sieves(&fb, nsqrt).0;
        Bench::new("sieve 32k block with ~10000 primes")
        .run_seeded(s, |s| {
            let mut s1 = s.clone();
            s1.sieve_block();
            s1.next_block();
        })
    },
    {
        let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
        let primes = fbase::primes(100000);
        let fb = fbase::prepare_factor_base(&n, &primes[..]);
        let nsqrt = isqrt(n);
        let s = qsieve::init_sieves(&fb, nsqrt).0;
        Bench::new("sieve 32k block with ~50000 primes")
        .run_seeded(s, |s| {
            let mut s1 = s.clone();
            s1.sieve_block();
            s1.next_block();
        })
    },
    {
        let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
        let primes = fbase::primes(5133);
        let fb = fbase::prepare_factor_base(&n, &primes[3..]);
        let nsqrt = isqrt(n);
        let s = qsieve::init_sieves(&fb, nsqrt).0;
        Bench::new("sieve 32k block with [3..2500] primes")
        .run_seeded(s, |s| {
            let mut s1 = s.clone();
            s1.sieve_block();
            s1.next_block();
        })
    },
    {
        let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
        let primes = fbase::primes(20000);
        let fb = fbase::prepare_factor_base(&n, &primes[5..]);
        let nsqrt = isqrt(n);
        let s = qsieve::init_sieves(&fb, nsqrt).0;
        Bench::new("sieve 32k block with [5..10000] primes")
        .run_seeded(s, |s| {
            let mut s1 = s.clone();
            s1.sieve_block();
            s1.next_block();
        })
    },
    {
        let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
        let primes = fbase::primes(100000);
        let fb = fbase::prepare_factor_base(&n, &primes[10..]);
        let nsqrt = isqrt(n);
        let s = qsieve::init_sieves(&fb, nsqrt).0;
        Bench::new("sieve 32k block with [10..50000] primes")
        .run_seeded(s, |s| {
            let mut s1 = s.clone();
            s1.sieve_block();
            s1.next_block();
        })
    },
    {
        let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
        let primes = fbase::primes(5133);
        let fb = fbase::prepare_factor_base(&n, &primes[..]);
        let nsqrt = isqrt(n);
        let s = qsieve::init_sieves(&fb, nsqrt).0;
        Bench::new("sieve+factor 32k block with ~2500 primes")
        .run_seeded(s, |s| {
            let mut s1 = s.clone();
            s1.sieve_block();
            let idxs = s1.smooths(70).0;
            assert!(10 <= idxs.len() && idxs.len() <= 20);
            s1.next_block();
        })
    },
    {
        let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
        let primes = fbase::primes(20000);
        let fb = fbase::prepare_factor_base(&n, &primes[..]);
        let nsqrt = isqrt(n);
        let s = qsieve::init_sieves(&fb, nsqrt).0;
        Bench::new("sieve+factor 32k block with ~10000 primes")
        .run_seeded(s, |s| {
            let mut s1 = s.clone();
            s1.sieve_block();
            let idxs = s1.smooths(84).0;
            assert!(10 <= idxs.len() && idxs.len() <= 20);
            s1.next_block();
        })
    },
    {
        let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
        let primes = fbase::primes(100000);
        let fb = fbase::prepare_factor_base(&n, &primes[..]);
        let nsqrt = isqrt(n);
        let s = qsieve::init_sieves(&fb, nsqrt).0;
        Bench::new("sieve+factor 32k block with ~50000 primes")
        .run_seeded(s, |s| {
            let mut s1 = s.clone();
            s1.sieve_block();
            let idxs = s1.smooths(86).0;
            assert!(15 <= idxs.len() && idxs.len() <= 30);
            s1.next_block();
        })
    },
}
