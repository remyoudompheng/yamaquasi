use bnum::cast::CastFrom;
use brunch::Bench;
use std::str::FromStr;
use yamaquasi::arith::isqrt;
use yamaquasi::{fbase, mpqs, qsieve, siqs};
use yamaquasi::{Int, Uint, Verbosity};

const PQ128: &str = "138775954839724585441297917764657773201";
const PQ256: &str =
    "104567211693678450173299212092863908236097914668062065364632502155864426186497";

fn main() {
    brunch::benches! {
        inline:
        // Eratosthenes sieve
        Bench::new("sieve 1000 primes")
        .run_seeded(1000, fbase::primes),
        Bench::new("sieve 10000 primes (until ~100e3)")
        .run_seeded(10000, fbase::primes),
        Bench::new("sieve 50000 primes (until ~600e3)")
        .run_seeded(50000, fbase::primes),
        // Multiplier selection
        {
            let n = Uint::from_str(PQ128).unwrap();
            Bench::new("select_multiplier(128-bit n) = Some(...)")
            .run_seeded(n, |n| { fbase::select_multiplier(n) })
        },
        // Polynomial selection
        {
            let n = Uint::from_str(PQ256).unwrap();
            let polybase: Uint = isqrt(n >> 1) >> 24;
            let polybase = isqrt(polybase);
            let width = 20 / 7 * polybase.bits() as usize;
            let polybase = u128::cast_from(polybase);
            Bench::new("sieve_for_polys(256-bit n) = Some(...)")
            .run_seeded(n, |n| { _ = mpqs::sieve_for_polys(&n, polybase, width).first().unwrap() })
        },
        // Mass polynomial selection
        // Generate 1000 polys, density is 1 / 2(log polybase)
        // = log 2 / 2 log2 n ~ 7/ 20 log2(n)
        {
            let n = Uint::from_str(PQ128).unwrap();
            let polybase: Uint = isqrt(n >> 1) >> 24;
            let polybase = isqrt(polybase);
            let width = 100 * 20 / 7 * polybase.bits() as usize;
            let polybase = u128::cast_from(polybase);
            Bench::new("select 100 polys 128-bit n")
            .run_seeded(n, |n| {
                let v = mpqs::sieve_for_polys(&n, polybase, width);
                assert!(90 < v.len() && v.len() < 110);
            })
        },
        {
            let n = Uint::from_str(PQ256).unwrap();
            let polybase: Uint = isqrt(n >> 1) >> 24;
            let polybase = isqrt(polybase);
            let width = 100 * 20 / 7 * polybase.bits() as usize;
            let polybase = u128::cast_from(polybase);
            Bench::new("select 100 polys 256-bit n")
            .run_seeded(n, |n| {
                let v = mpqs::sieve_for_polys(&n, polybase, width);
                assert!(90 < v.len() && v.len() < 110);
            })
        },
        // SIQS primitives
        {
            // Prepare 40 A (amortized cost for 10000 polynomials)
            let n = Int::from_str(PQ256).unwrap();
            let fb = fbase::FBase::new(n, 5000);
            Bench::new("prepare 50 A values for SIQS (n = 256 bits)")
            .run_seeded((&fb, &n), |(fb, n)| {
                let f = siqs::select_siqs_factors(fb, n, 9, 1 << 20, Verbosity::Silent);
                let a_ints = siqs::select_a(&f, 40, Verbosity::Silent);
                for a_int in &a_ints {
                    siqs::prepare_a(&f, a_int, fb, 0);
                }
            })
        },
        {
            // Fully prepare one polynomial for a given A.
            // It is 6 times faster than MPQS preparation.
            let n = Int::from_str(PQ256).unwrap();
            let fb = fbase::FBase::new(Int::cast_from(n), 5000);
            let f = siqs::select_siqs_factors(&fb, &n, 9, 1 << 20, Verbosity::Silent);
            let a_ints = siqs::select_a(&f, 40, Verbosity::Silent);
            let a_s: Vec<_> = a_ints.iter().map(|a_int| siqs::prepare_a(&f, a_int, &fb, 0)).collect();
            let prefs = yamaquasi::Preferences::default();
            let s = siqs::SieveSIQS::new(n, &fb, fb.bound() as u64, 0, 1 << 20, &prefs);
            Bench::new("prepare 100 SIQS polynomial (n = 256 bits)")
            .with_samples(20_000)
            .run_seeded((&s, a_s.first().unwrap()), |(s, a)| {
                let mut pol = siqs::Poly::first(s, a);
                for _ in 0..100 {
                    pol.next(s, a);
                }
            })
        },
        // Block sieve
        {
            let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
            let fb = fbase::FBase::new(Int::cast_from(n), 2566);
            let qs = qsieve::SieveQS::new(n, &fb, 10_000_000, false);
            let (s, _) = qs.init_sieve_for_test();
            Bench::new("clone sieve structure (no-op)")
            .run_seeded(s, |s| {
                let _ = s.clone();
            })
        },
        {
            let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
            let fb = fbase::FBase::new(Int::cast_from(n), 2566);
            let qs = qsieve::SieveQS::new(n, &fb, 10_000_000, false);
            let s = qs.init_sieve_for_test().0;
            Bench::new("sieve 32k block with ~2500 primes")
            .run_seeded(s, |s| {
                let mut s1 = s.clone();
                s1.idxskip = 0;
                s1.sieve_block();
                s1.next_block();
            })
        },
        {
            let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
            let fb = fbase::FBase::new(Int::cast_from(n), 10000);
            let qs = qsieve::SieveQS::new(n, &fb, 10_000_000, false);
            let s = qs.init_sieve_for_test().0;
            Bench::new("sieve 32k block with ~10000 primes")
            .run_seeded(s, |s| {
                let mut s1 = s.clone();
                s1.idxskip = 0;
                s1.sieve_block();
                s1.next_block();
            })
        },
        {
            let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
            let fb = fbase::FBase::new(Int::cast_from(n), 50_000);
            let qs = qsieve::SieveQS::new(n, &fb, 10_000_000, false);
            let s = qs.init_sieve_for_test().0;
            Bench::new("sieve 32k block with ~50000 primes")
            .run_seeded(s, |s| {
                let mut s1 = s.clone();
                s1.idxskip = 0;
                s1.sieve_block();
                s1.next_block();
            })
        },
        {
            let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
            let fb = fbase::FBase::new(Int::cast_from(n), 2566);
            let qs = qsieve::SieveQS::new(n, &fb, 10_000_000, false);
            let (s, [r1, r2]) = qs.init_sieve_for_test();
            Bench::new("sieve+factor 32k block with ~2500 primes")
            .run_seeded(s, |s| {
                let mut s1 = s.clone();
                s1.sieve_block();
                let idxs = s1.smooths(70, None, [&r1, &r2]).0;
                assert!(10 <= idxs.len() && idxs.len() <= 20);
                s1.next_block();
            })
        },
        {
            let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
            let fb = fbase::FBase::new(Int::cast_from(n), 10000);
            let qs = qsieve::SieveQS::new(n, &fb, 10_000_000, false);
            let (s, [r1, r2]) = qs.init_sieve_for_test();
            Bench::new("sieve+factor 32k block with ~10000 primes")
            .run_seeded(s, |s| {
                let mut s1 = s.clone();
                s1.sieve_block();
                let idxs = s1.smooths(84, None, [&r1, &r2]).0;
                assert!(10 <= idxs.len() && idxs.len() <= 20);
                s1.next_block();
            })
        },
        {
            let n = Uint::from_str("176056248311966088405511077755578022771").unwrap();
            let fb = fbase::FBase::new(Int::cast_from(n), 50000);
            let qs = qsieve::SieveQS::new(n, &fb, 10_000_000, false);
            let s = qs.init_sieve_for_test().0;
            Bench::new("sieve+factor 32k block with ~50000 primes")
            .run_seeded(s, |s| {
                let mut s1 = s.clone();
                s1.sieve_block();
                // Inaccurate tests due to sieve optimisations.
                //let idxs = s1.smooths(86).0;
                //assert!(15 <= idxs.len() && idxs.len() <= 30);
                //eprintln!("{}", idxs.len());
                s1.next_block();
            })
        },
    }

    for b1 in [100_000, 1_000_000, 10_000_000, 100_000_000] {
        let start = std::time::Instant::now();
        let mut ps = fbase::PrimeSieve::new();
        let mut s = vec![];
        loop {
            let b = ps.next();
            s.extend_from_slice(b);
            if b[b.len() - 1] > b1 as u32 {
                break;
            }
        }
        eprintln!(
            "Sieved {} primes until {} in {:.3}s",
            s.len(),
            s.last().unwrap(),
            start.elapsed().as_secs_f64()
        );
    }
    // Full prime sieve.
    let start = std::time::Instant::now();
    let mut s = fbase::PrimeSieve::new();
    let mut last = 0_u32;
    let mut count = 0;
    while let b @ &[_, ..] = s.next() {
        last = b[b.len() - 1];
        count += b.len();
    }
    // primecount(2**32)
    assert_eq!(count, 203280221);
    assert_eq!(last, 4294967291);
    eprintln!(
        "Enumerated {count} 32-bit primes in {:.3}s",
        start.elapsed().as_secs_f64()
    );
}
