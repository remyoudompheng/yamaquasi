use std::str::FromStr;
use std::time::Duration;

use brunch::Bench;
use yamaquasi::arith_montgomery::ZmodN;
use yamaquasi::{ecm, Uint};

fn main() {
    // A 256-bit prime.
    let p256 = Uint::from_str(
        "92786510271815932444618978328822237837414362351005653014234479629925371473357",
    )
    .unwrap();

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
                    ecm::ecm(n, 16, 120, 280, 0, None).unwrap();
                })
        },
    }

    // ECM complexity (O(b1 log(b1)) + O(b2^2))
    eprintln!("ECM timings");
    // Ideal b2 are such that phi(b2)/2 is less than a power of two.
    // 120 => 32
    // 210 => 48
    // 462 => 120
    // 1050 => 240
    // 2310 => 480
    // 4410 => 1008
    // 8820 => 2016
    // 19110 => 4032
    // 38220 => 8064
    // 76440 => 16128
    for b2 in [120, 210, 462, 1050, 2310, 4410, 8820, 19110, 38220, 76440] {
        for b1 in [100, 150, 200] {
            if b1 != 200 && b2 > 1000 {
                continue;
            }
            let start = std::time::Instant::now();
            // Use P256 so what ECM cannot work.
            let res = ecm::ecm(p256, 1, b1, b2, 0, None);
            assert!(res.is_none());
            eprintln!(
                "ECM(B1={b1},B2={b2}) in {:.3}s",
                start.elapsed().as_secs_f64()
            );
        }
    }
    for b1 in [1000, 10_000, 100_000, 1_000_000] {
        let b2 = 8820;
        let start = std::time::Instant::now();
        // Use P256 so what ECM cannot work.
        let res = ecm::ecm(p256, 1, b1, b2, 0, None);
        assert!(res.is_none());
        eprintln!(
            "ECM(B1={b1},B2={b2}) in {:.3}s",
            start.elapsed().as_secs_f64()
        );
    }
}
