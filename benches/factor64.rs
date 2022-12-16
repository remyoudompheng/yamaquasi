use std::time::Duration;

use brunch::Bench;
use yamaquasi::{fbase, qsieve64, squfof};

brunch::benches! {
    {
        let n42: &[u64] = &[
            2965576997959,
            2631165445817,
            2794378024157,
            2822044701943,
            3052120253579,
        ];
        Bench::new("5x qsieve64 n=42 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n42, |ns| for &n in ns {
                qsieve64::qsieve(n).unwrap();
            })
    },
    {
        let n48: &[u64] = &[
            235075827453629,
            166130059616737,
            159247921097933,
            224077614412439,
            219669028971857,
        ];
        Bench::new("5x qsieve64 n=48 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n48, |ns| for &n in ns {
                qsieve64::qsieve(n).unwrap();
            })
    },
    {
        let n56: &[u64] = &[
            42795961034553971,
            39128513926139749,
            44643473083983271,
            40952332749496541,
            56396468816856241,
        ];
        Bench::new("5x qsieve64 n=56 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n56, |ns| for &n in ns {
                qsieve64::qsieve(n).unwrap();
            })
    },
    {
        let n42: &[u64] = &[
            2965576997959,
            2631165445817,
            2794378024157,
            2822044701943,
            3052120253579,
        ];
        Bench::new("5x SQUFOF n=42 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n42, |ns| for &n in ns {
                squfof::squfof(n).unwrap();
            })
    },
    {
        let n48: &[u64] = &[
            235075827453629,
            166130059616737,
            159247921097933,
            224077614412439,
            219669028971857,
        ];
        Bench::new("5x SQUFOF n=48 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n48, |ns| for &n in ns {
                squfof::squfof(n).unwrap();
            })
    },
    {
        let n56: &[u64] = &[
            42795961034553971,
            39128513926139749,
            44643473083983271,
            40952332749496541,
            56396468816856241,
        ];
        Bench::new("5x SQUFOF n=56 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n56, |ns| for &n in ns {
                squfof::squfof(n).unwrap();
            })
    },
    {
        let n48: &[u64] = &[
            235075827453629,
            166130059616737,
            159247921097933,
            224077614412439,
            219669028971857,
        ];
        Bench::new("5x pseudoprime n=48 bits")
            .run_seeded(n48, |ns| for &n in ns {
                assert!(fbase::certainly_composite(n));
            })
    },
}
