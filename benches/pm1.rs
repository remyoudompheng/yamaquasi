use std::str::FromStr;

use yamaquasi::pollard_pm1::pm1_impl;
use yamaquasi::{Preferences, Uint, Verbosity};

fn main() {
    // A 256-bit semiprime.
    // P-1 cannot possibly find the factors:
    // 283591646828730759669387134803840520563
    // 275050624619969308694260926840424191881
    let p256 = Uint::from_str(
        "78002059597248133810934168287429564748767875757275938068446100287361638149003",
    )
    .unwrap();
    // A 480-bit semiprime. Factors are not smooth:
    // 1556203143478091261684849091823555915389977651408594595152237580146768707
    // 1542388964472894082635452919698060312796671005618203059551010504722243229
    let p480 = Uint::from_str(
        "2400270554978635795737027370707206321618362754354145259391031820937407231529834691694780546243889645833224971047420763151985948675846852859834903"
    )
    .unwrap();
    let mut prefs = Preferences::default();
    prefs.verbosity = Verbosity::Silent;

    let b2_values = [
        30e3, 60e3, 100e3, 200e3, 450e3, 1.9e6, 8.3e6, 33e6, 133e6, 550e6, 2.3e9, 7.9e9, 37e9,
        150e9, 640e9, 2.5e12, 11e12,
    ];
    let mut tt256 = vec![];
    let mut tt480 = vec![];
    for &b2 in &b2_values {
        let b1 = 200;
        let start = std::time::Instant::now();
        let res = pm1_impl(&p256, b1, b2, Verbosity::Silent);
        assert!(res.is_none());
        let t = start.elapsed().as_secs_f64();
        eprintln!("p256 PM1(B1={b1},B2={b2:.2e}) in {t:.3}s");
        tt256.push(t);

        let start = std::time::Instant::now();
        let res = pm1_impl(&p480, b1, b2, Verbosity::Silent);
        assert!(res.is_none());
        let t = start.elapsed().as_secs_f64();
        eprintln!("p480 PM1(B1={b1},B2={b2:.2e}) in {t:.3}s");
        tt480.push(t);
    }

    let mut t256 = vec![];
    let mut t480 = vec![];
    let b1_values = &[1000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000];
    for &b1 in b1_values {
        let b2 = 8.3e6;

        let start = std::time::Instant::now();
        let res = pm1_impl(&p256, b1, b2, Verbosity::Silent);
        assert!(res.is_none());
        let t = start.elapsed().as_secs_f64();
        eprintln!("p256 PM1(B1={b1},B2={b2:.2e}) in {t:.3}s");
        t256.push(t);

        let start = std::time::Instant::now();
        let res = pm1_impl(&p480, b1, b2, Verbosity::Silent);
        assert!(res.is_none());
        let t = start.elapsed().as_secs_f64();
        eprintln!("p480 PM1(B1={b1},B2={b2:.2e}) in {t:.3}s");
        t480.push(t);
    }

    // Compare with theoretical complexity.
    // In stage 1, we need 1.2 modular multiplications per bit.
    let mut stage1_c256 = vec![];
    let mut stage1_c480 = vec![];
    for i in 2..b1_values.len() {
        let b1 = b1_values[i] as f64;
        let c = (t256[i] - t256[0]) / (1.44 * 1.2 * b1) * 1e9;
        stage1_c256.push(c.round() as u64);
        let c = (t480[i] - t480[0]) / (1.44 * 1.2 * b1) * 1e9;
        stage1_c480.push(c.round() as u64);
    }
    eprintln!("p256: stage 1 cost 1.2 * 1.44 B1 * {stage1_c256:?}ns");
    eprintln!("p480: stage 1 cost 1.2 * 1.44 B1 * {stage1_c480:?}ns");
    let mut small2_c256 = vec![];
    let mut small2_c480 = vec![];
    let mut large2_c256 = vec![];
    let mut large2_c480 = vec![];
    for i in 1..b2_values.len() {
        let b2 = b2_values[i] as f64;
        if b2 < 80e3 {
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
