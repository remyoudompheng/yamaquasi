use std::str::FromStr;
use std::time::Duration;

use brunch::Bench;
use yamaquasi::arith_montgomery::ZmodN;
use yamaquasi::pollard_pm1::pm1_impl;
use yamaquasi::{ecm, Uint};

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

    let d_values = [
        120, 210, 462, 1050, 2310, 4620, 9240, 19110, 39270, 79170, 159390, 324870, 649740,
    ];
    for &d in &d_values {
        let b1 = 200;
        let start = std::time::Instant::now();
        // Use P256 so what ECM cannot work.
        let res = pm1_impl(p256, b1, d);
        assert!(res.is_none());
        eprintln!(
            "p256 PM1(B1={b1},D={d}) in {:.3}s",
            start.elapsed().as_secs_f64()
        );

        let start = std::time::Instant::now();
        let res = pm1_impl(p480, b1, d);
        assert!(res.is_none());
        eprintln!(
            "p480 PM1(B1={b1},D={d}) in {:.3}s",
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
                    ecm::ecm(n, 16, 120, 280, 0, None).unwrap();
                })
        },
    }

    // ECM complexity (O(b1 log(b1)) + O(d^2))
    eprintln!("ECM timings");
    // Ideal d are such that phi(d)/2 is less than a power of two.
    // 120 => 32
    // 210 => 48
    // 462 => 120
    // 1050 => 240
    // D=2310 B2=5.336e+06 φ(D)/2=240 (10 blocks size 2^8)
    // D=4620 B2=2.134e+07 φ(D)/2=480 (10 blocks size 2^9)
    // D=9240 B2=8.538e+07 φ(D)/2=960 (10 blocks size 2^10)
    // D=19110 B2=3.652e+08 φ(D)/2=2016 (10 blocks size 2^11)
    // D=39270 B2=1.542e+09 φ(D)/2=3840 (10 blocks size 2^12)
    // D=79170 B2=6.268e+09 φ(D)/2=8064 (10 blocks size 2^13)
    // D=159390 B2=2.541e+10 φ(D)/2=15840 (10 blocks size 2^14)
    // D=324870 B2=1.055e+11 φ(D)/2=32256 (10 blocks size 2^15)
    // D=649740 B2=4.222e+11 φ(D)/2=64512 (10 blocks size 2^16)
    // D=690690 B2=4.771e+11 φ(D)/2=63360 (11 blocks size 2^16)
    // D=1299480 B2=1.689e+12 φ(D)/2=129024 (10 blocks size 2^17)
    // D=1381380 B2=1.908e+12 φ(D)/2=126720 (11 blocks size 2^17)
    // D=2612610 B2=6.826e+12 φ(D)/2=241920 (10 blocks size 2^18)
    // D=2852850 B2=8.139e+12 φ(D)/2=259200 (11 blocks size 2^18)
    // D=5238870 B2=2.745e+13 φ(D)/2=518400 (10 blocks size 2^19)
    // D=5705700 B2=3.256e+13 φ(D)/2=518400 (11 blocks size 2^19)
    for d in [
        120, 210, 462, 1050, 2310, 4620, 9240, 19110, 39270, 79170, 159390, 324870, 649740,
    ] {
        for b1 in [100, 150, 200] {
            if b1 != 200 && d > 1000 {
                continue;
            }
            let start = std::time::Instant::now();
            // Use P256 so what ECM cannot work.
            let res = ecm::ecm(p256, 1, b1, d, 0, None);
            assert!(res.is_none());
            eprintln!(
                "p256 ECM(B1={b1},D={d}) in {:.3}s",
                start.elapsed().as_secs_f64()
            );

            let start = std::time::Instant::now();
            let res = ecm::ecm(p480, 1, b1, d, 0, None);
            assert!(res.is_none());
            eprintln!(
                "p480 ECM(B1={b1},D={d}) in {:.3}s",
                start.elapsed().as_secs_f64()
            );
        }
    }
    for b1 in [1000, 10_000, 100_000, 1_000_000] {
        let d = 8820;
        let start = std::time::Instant::now();
        // Use P256 so what ECM cannot work.
        let res = ecm::ecm(p256, 1, b1, d, 0, None);
        assert!(res.is_none());
        eprintln!(
            "p256 ECM(B1={b1},D={d}) in {:.3}s",
            start.elapsed().as_secs_f64()
        );

        let start = std::time::Instant::now();
        let res = ecm::ecm(p480, 1, b1, d, 0, None);
        assert!(res.is_none());
        eprintln!(
            "p480 ECM(B1={b1},D={d}) in {:.3}s",
            start.elapsed().as_secs_f64()
        );
    }
}
