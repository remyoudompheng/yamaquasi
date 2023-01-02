use std::str::FromStr;
use std::time::Duration;

use brunch::Bench;
use yamaquasi::{ecm, Uint};

brunch::benches! {
    {
        // 5 primes where the "good curves" often have non-smooth order.
        let primes24: &[u64] = &[
            13377491,
            13613153,
            13757371,
            14356327,
            14747881,
        ];
        // A 256-bit prime
        let p256 = Uint::from_str("92786510271815932444618978328822237837414362351005653014234479629925371473357").unwrap();
        Bench::new("5x ECM p24*p256")
            .with_timeout(Duration::from_secs(10))
            .run_seeded((), |_| for &p in primes24 {
                let n = Uint::from(p) * p256;
                ecm::ecm(n, 16, 120, 280, 0).unwrap();
            })
    },
}
