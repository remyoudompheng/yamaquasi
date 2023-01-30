use std::str::FromStr;

use yamaquasi::pollard_pm1::pm1_impl;
use yamaquasi::{Preferences, Uint, Verbosity};

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
        30e3, 60e3, 100e3, 200e3, 450e3, 1.9e6, 8.3e6, 33e6, 133e6, 550e6, 2.3e9, 7.9e9, 37e9,
        150e9, 640e9, 2.5e12, 11e12,
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
}
