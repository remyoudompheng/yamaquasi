use std::str::FromStr;
use std::time::Duration;

use brunch::Bench;
use yamaquasi::arith_montgomery::ZmodN;
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

    brunch::benches! {
        inline:
        {
            let zn = ZmodN::new(p256);
            let c = ecm::Curve::from_point(zn, 8, 9).unwrap();
            let g = c.gen();
            let n: u64 = 1511 * 1523 * 1531;
            Bench::new("scalar mul n32 x G (p256)")
                .with_timeout(Duration::from_secs(3))
                .run_seeded((), |_| c.scalar64_mul_dbladd(n, &g))
        },
        {
            let zn = ZmodN::new(p256);
            let c = ecm::Curve::from_point(zn, 8, 9).unwrap();
            let g = c.gen();
            let n: u64 = 1511 * 1523 * 1531 * 1543 * 1549 * 1553;
            Bench::new("scalar mul n64 x G (p256)")
                .with_timeout(Duration::from_secs(3))
                .run_seeded((), |_| c.scalar64_mul_dbladd(n, &g))
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
            let zn = ZmodN::new(p256);
            let c = ecm::Suyama11::new(&zn).unwrap();
            Bench::new("generate Suyama curve (p256)")
                .with_timeout(Duration::from_secs(3))
                .run_seeded((), |_| c.params_point(&c.element(123_456_789).unwrap()))
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
    eprintln!("ECM timings stage 2");
    let b2_values = [
        7.7e3, 20e3, 33e3, 81e3, 181e3, 554e3, 1.37e6, 2.3e6, 7.1e6, 28e6, 117e6, 643e6, 2.6e9,
        10.5e9, 43e9, 136e9, 543e9,
    ];
    let mut tt256 = vec![];
    let mut tt480 = vec![];
    for &b2 in &b2_values {
        let b1 = 200;
        let start = std::time::Instant::now();
        // Use P256 so what ECM cannot work.
        let iters = if b2 < 100e3 { 1000 } else { 1 };
        for _ in 0..iters {
            let res = ecm::ecm(p256, 1, b1, b2, &prefs, None);
            assert!(res.is_none());
        }
        let t = start.elapsed().as_secs_f64() / (iters as f64);
        eprintln!("p256 ECM(B1={b1},B2={b2:.3e}) in {t:.3}s");
        tt256.push(t);

        let start = std::time::Instant::now();
        let res = ecm::ecm(p480, 1, b1, b2, &prefs, None);
        assert!(res.is_none());
        let t = start.elapsed().as_secs_f64();
        eprintln!("p480 ECM(B1={b1},B2={b2:.3e}) in {t:.3}s");
        tt480.push(t);
    }

    eprintln!("ECM timings stage 1");
    let mut t256 = vec![];
    let mut t480 = vec![];
    let b1s = &[100, 1000, 10_000, 100_000, 1_000_000, 10_000_000];
    for &b1 in b1s {
        let b2 = 38e6;
        let start = std::time::Instant::now();
        // Use P256 so what ECM cannot work.
        let iters = if b2 < 100e3 { 1000 } else { 1 };
        for _ in 0..iters {
            let res = ecm::ecm(p256, 1, b1, b2, &prefs, None);
            assert!(res.is_none());
        }
        let t = start.elapsed().as_secs_f64() / (iters as f64);
        eprintln!("p256 ECM(B1={b1},B2={b2:.3e}) in {t:.3}s");
        t256.push(t);

        let start = std::time::Instant::now();
        let res = ecm::ecm(p480, 1, b1, b2, &prefs, None);
        assert!(res.is_none());
        let t = start.elapsed().as_secs_f64();
        eprintln!("p480 ECM(B1={b1},B2={b2:.3e}) in {t:.3}s");
        t480.push(t);
    }
    // Compare with theoretical complexity.
    // In stage 1, we need 8.9 modular multiplications per bit.
    let mut stage1_c256 = vec![];
    let mut stage1_c480 = vec![];
    for i in 2..b1s.len() {
        let b1 = b1s[i] as f64;
        let c = (t256[i] - t256[0]) / (1.44 * 8.92 * b1) * 1e9;
        stage1_c256.push(c.round() as u64);
        let c = (t480[i] - t480[0]) / (1.44 * 8.92 * b1) * 1e9;
        stage1_c480.push(c.round() as u64);
    }
    eprintln!("p256: stage 1 cost 8.92 * 1.44 B1 * {stage1_c256:?}ns");
    eprintln!("p480: stage 1 cost 8.92 * 1.44 B1 * {stage1_c480:?}ns");
    let mut small2_c256 = vec![];
    let mut small2_c480 = vec![];
    let mut large2_c256 = vec![];
    let mut large2_c480 = vec![];
    for i in 1..b2_values.len() {
        let b2 = b2_values[i] as f64;
        if b2 < 2e6 {
            let c = (tt256[i] - tt256[0]) / (b2 / b2.log2()) * 1e9;
            small2_c256.push(c.round() as u64);
            let c = (tt480[i] - tt480[0]) / (b2 / b2.log2()) * 1e9;
            small2_c480.push(c.round() as u64);
        } else {
            let c = (tt256[i] - tt256[0]) / (b2.sqrt() * b2.log2()) * 1e9;
            large2_c256.push(c.round() as u64);
            let c = (tt480[i] - tt480[0]) / (b2.sqrt() * b2.log2()) * 1e9;
            large2_c480.push(c.round() as u64);
        }
    }
    eprintln!("p256: small stage 2 cost B2/log2(B2) * {small2_c256:?}ns");
    eprintln!("p480: small stage 2 cost B2/log2(B2) * {small2_c480:?}ns");
    eprintln!("p256: large stage 2 cost sqrt(B2)*log2(B2) * {large2_c256:?}ns");
    eprintln!("p480: large stage 2 cost sqrt(B2)*log2(B2) * {large2_c480:?}ns");
}
