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
    for &b2 in &b2_values {
        let b1 = 200;
        let start = std::time::Instant::now();
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

    for b1 in [1000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000] {
        let b2 = 8.3e6;

        let start = std::time::Instant::now();
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
}
