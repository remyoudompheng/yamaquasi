use brunch::Bench;
use yamaquasi::{fbase, qsieve64};

brunch::benches! {
    {
        let n48: &[u64] = &[
            235075827453629,
            166130059616737,
            159247921097933,
            224077614412439,
            219669028971857,
        ];
        Bench::new("5x qsieve64 n=48 bits")
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
            .run_seeded(n56, |ns| for &n in ns {
                qsieve64::qsieve(n).unwrap();
            })
    },
    {
        let n60: &[u64] = &[
            581158550236064243,
            674589653504729947,
            585976505371483507,
            666172519548269753,
            909417471716788523,
        ];
        Bench::new("5x qsieve64 n=60 bits")
            .run_seeded(n60, |ns| for &n in ns {
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
        Bench::new("5x pseudoprime n=48 bits")
            .run_seeded(n48, |ns| for &n in ns {
                assert!(fbase::certainly_composite(n));
            })
    },
}
