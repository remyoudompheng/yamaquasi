use std::str::FromStr;
use std::time::Duration;

use brunch::Bench;
use yamaquasi::arith_montgomery::ZmodN;
use yamaquasi::pollard_pm1::pm1_impl;
use yamaquasi::{ecm, Preferences, Uint, Verbosity};

fn main() {
    // A 256-bit prime.
    let p256 = Uint::from_str(
        "92786510271815932444618978328822237837414362351005653014234479629925371473357",
    )
    .unwrap();
    // A 480-bit prime.
    let p480 = Uint::from_str(
        "1814274712676087950344811991522598371991048724422784825007845656050800905627423692122807639509275259938192211611976651772022623688843091923010451",
    )
    .unwrap();
    let mut prefs = Preferences::default();
    prefs.verbosity = Verbosity::Silent;

    let b2_values = [
        15e3, 65e3, 268e3, 1.18e6, 7.1e6, 28e6, 117e6, 643e6, 2.6e9, 10.5e9, 43e9, 136e9, 543e9,
    ];
    for &b2 in &b2_values {
        let b1 = 200;
        let start = std::time::Instant::now();
        // Use P256 so what ECM cannot work.
        let res = pm1_impl(p256, b1, b2, Verbosity::Silent);
        assert!(res.is_none());
        eprintln!(
            "p256 PM1(B1={b1},B2={b2:.2e}) in {:.3}s",
            start.elapsed().as_secs_f64()
        );

        let start = std::time::Instant::now();
        let res = pm1_impl(p480, b1, b2, Verbosity::Silent);
        assert!(res.is_none());
        eprintln!(
            "p480 PM1(B1={b1},B2={b2:.2e}) in {:.3}s",
            start.elapsed().as_secs_f64()
        );
    }

    brunch::benches! {
        inline:
        {
            let zn = ZmodN::new(p256);
            let c = ecm::Curve::from_point(zn, 8, 9).unwrap();
            let g = c.gen();
            let n: u64 = 1511 * 1523 * 1531;
            Bench::new("scalar mul n32 x G (p256)")
                .with_timeout(Duration::from_secs(3))
                .run_seeded((), |_| c.scalar64_mul(n, &g))
        },
        {
            let zn = ZmodN::new(p256);
            let c = ecm::Curve::from_point(zn, 8, 9).unwrap();
            let g = c.gen();
            let n: u64 = 1511 * 1523 * 1531 * 1543 * 1549 * 1553;
            Bench::new("scalar mul n64 x G (p256)")
                .with_timeout(Duration::from_secs(3))
                .run_seeded((), |_| c.scalar64_mul(n, &g))
        },
        {
            let zn = ZmodN::new(p256);
            let c = ecm::Curve::from_point(zn, 8, 9).unwrap();
            let g = c.gen();
            let n: u64 = 1511 * 1523 * 1531;
            Bench::new("chain mul n32 x G (p256)")
                .with_timeout(Duration::from_secs(3))
                .run_seeded((), |_| c.scalar64_chainmul(n, &g))
        },
        {
            let zn = ZmodN::new(p256);
            let c = ecm::Curve::from_point(zn, 8, 9).unwrap();
            let g = c.gen();
            let n: u64 = 1511 * 1523 * 1531 * 1543 * 1549 * 1553;
            Bench::new("chain mul n64 x G (p256)")
                .with_timeout(Duration::from_secs(3))
                .run_seeded((), |_| c.scalar64_chainmul(n, &g))
        },
        {
            // 5 primes where the "good curves" often have non-smooth order.
            let primes24: &[u64] = &[
                13377491,
                13613153,
                13757371,
                14356327,
                14747881,
            ];
            Bench::new("5x ECM p24*p256")
                .with_timeout(Duration::from_secs(10))
                .run_seeded((), |_| for &p in primes24 {
                    let n = Uint::from(p) * p256;
                    ecm::ecm(n, 16, 120, 280.*280., &prefs, None).unwrap();
                })
        },
    }

    // ECM complexity (O(b1 log(b1)) + O(b2^(0.5+eps)))
    eprintln!("ECM timings");
    for &b2 in &b2_values {
        for b1 in [100, 150, 200] {
            if b1 != 200 && b2 > 1e6 {
                continue;
            }
            let start = std::time::Instant::now();
            // Use P256 so what ECM cannot work.
            let res = ecm::ecm(p256, 1, b1, b2, &prefs, None);
            assert!(res.is_none());
            eprintln!(
                "p256 ECM(B1={b1},B2={b2:.3e}) in {:.3}s",
                start.elapsed().as_secs_f64()
            );

            let start = std::time::Instant::now();
            let res = ecm::ecm(p480, 1, b1, b2, &prefs, None);
            assert!(res.is_none());
            eprintln!(
                "p480 ECM(B1={b1},B2={b2:.3e}) in {:.3}s",
                start.elapsed().as_secs_f64()
            );
        }
    }
    for b1 in [1000, 10_000, 100_000, 1_000_000] {
        let b2 = 38e6;
        let start = std::time::Instant::now();
        // Use P256 so what ECM cannot work.
        let res = ecm::ecm(p256, 1, b1, b2, &prefs, None);
        assert!(res.is_none());
        eprintln!(
            "p256 ECM(B1={b1},B2={b2:.3e}) in {:.3}s",
            start.elapsed().as_secs_f64()
        );

        let start = std::time::Instant::now();
        let res = ecm::ecm(p480, 1, b1, b2, &prefs, None);
        assert!(res.is_none());
        eprintln!(
            "p480 ECM(B1={b1},B2={b2:.3e}) in {:.3}s",
            start.elapsed().as_secs_f64()
        );
    }
}
