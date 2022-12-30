use std::time::Duration;

use brunch::Bench;
use yamaquasi::{fbase, pollard_pm1, qsieve64, squfof};

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
        let n59hard: &[u64] = &[
            28379446573532569,
            56181815869659373,
            87824287206965359,
            89058989051837809,
            89859608810317271,
        ];
        Bench::new("5x SQUFOF n=~56 bits with hard multiplier")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n59hard, |ns| for &n in ns {
                squfof::squfof(n).unwrap();
            })
    },
    {
        let n56: &[u64] = &[
            // from n56 above
            39128513926139749, // B=2693
            40952332749496541, // B=127
            56396468816856241, // B=1663
            // from n59hard
            28379446573532569, // B=11447
            56181815869659373, // B=17489
        ];
        let pb = pollard_pm1::PM1Base::new();
        Bench::new("5x P-1 n=56 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n56, |ns| for &n in ns {
                let Some(_) = pb.factor(n, 3000)
                    else { panic!("failure on {n}") };
            })
    },
    {
        let n52hard: &[u64] = &[
            1229054827205021, // B=55213
            1728782896618079, // B=52627
            2531488701194069, // B=53923
            1122065566442519, // B=57527
            5733442064937071, // B=55127

        ];
        let pb = pollard_pm1::PM1Base::new();
        Bench::new("5x P-1 n=52 bits with large bound")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n52hard, |ns| for &n in ns {
                let Some(_) = pb.factor(n, 8000)
                    else { panic!("failure on {n}") };
            })
    },
    {
        let n52hard: &[u64] = &[
            1834558324821379, // B=750223
            2804193742926553, // B=621133
            4216338048915137, // B=717683
            14385962168468899, // B=772847
            4263495252146089, // B=804511
        ];
        let pb = pollard_pm1::PM1Base::new();
        Bench::new("5x P-1 n=52 bits with very large bound")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n52hard, |ns| for &n in ns {
                let Some(_) = pb.factor(n, 70000)
                    else { panic!("failure on {n}") };
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
