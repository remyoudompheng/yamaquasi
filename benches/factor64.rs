use std::time::Duration;

use brunch::Bench;
use yamaquasi::{fbase, pollard_pm1, pollard_rho, qsieve64, squfof, Uint, Verbosity};

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
                qsieve64::qsieve(n, Verbosity::Silent).unwrap();
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
                qsieve64::qsieve(n, Verbosity::Silent).unwrap();
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
                qsieve64::qsieve(n, Verbosity::Silent).unwrap();
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
        let n42: &[u64] = &[
            2965576997959,
            2631165445817,
            2794378024157,
            2822044701943,
            3052120253579,
        ];
        Bench::new("5x Rho n=42 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n42, |ns| for &n in ns {
                pollard_rho::rho64(n, 1, 2000)
                    .or_else(||  pollard_rho::rho64(n, 2, 2000))
                    .or_else(||  pollard_rho::rho64(n, 3, 2000))
                    .unwrap();
            })
    },
    {
        let n48: &[u64] = &[
            235075827453629,
            166130059616737,
            159247921097933,
            224077614412439,
            153661921148731,
        ];
        Bench::new("5x Rho n=48 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n48, |ns| for &n in ns {
                pollard_rho::rho_semiprime(n).unwrap();
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
        Bench::new("5x Rho n=56 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n56, |ns| for &n in ns {
                let _ = pollard_rho::rho_semiprime(n);
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
        let pb = pollard_pm1::PM1Base::newsmall(512, 30000);
        Bench::new("5x P-1 n=56 bits")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n56, |ns| for &n in ns {
                pb.pm1(&Uint::from(n)).first().unwrap_or_else(|| panic!("failure on {n}"));
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
        let pb = pollard_pm1::PM1Base::newsmall(512, 80000);
        Bench::new("5x P-1 n=52 bits with large bound")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(n52hard, |ns| for &n in ns {
                pb.pm1(&Uint::from(n)).first().unwrap_or_else(|| panic!("failure on {n}"));
            })
    },
    /*
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
    */
    {
        let n16: &[u64] = &[
            47161, 60293, 48611, 33679, 38677
        ];
        Bench::new("5x pseudoprime n=16 bits prime")
            .with_samples(50_000)
            .run_seeded(n16, |ns| for &n in ns {
                assert!(yamaquasi::isprime64(n));
            })
    },
    {
        let ns: &[u64] = &[
            13383037, 15626777, 9736099, 9240683, 16070149
        ];
        Bench::new("5x pseudoprime n=24 bits prime")
            .with_samples(50_000)
            .run_seeded(ns, |ns| for &n in ns {
                assert!(yamaquasi::isprime64(n));
            })
    },
    {
        let ns: &[u64] = &[
            7977469, 12134833, 15380059, 10786771, 7881443
        ];
        Bench::new("5x pseudoprime n=24 bits composite")
            .with_samples(50_000)
            .run_seeded(ns, |ns| for &n in ns {
                assert!(!yamaquasi::isprime64(n));
            })
    },
    {
        let ns: &[u64] = &[
            219161743, 200625133, 231110093, 198868123, 252992219
        ];
        Bench::new("5x pseudoprime n=28 bits prime")
            .with_samples(50_000)
            .run_seeded(ns, |ns| for &n in ns {
                assert!(yamaquasi::isprime64(n));
            })
    },
    {
        let ns: &[u64] = &[
            127831157,88986389,89747717,72698897,181819793
        ];
        Bench::new("5x pseudoprime n=28 bits composite")
            .run_seeded(ns, |ns| for &n in ns {
                assert!(!yamaquasi::isprime64(n));
            })
    },
    {
        let n32: &[u64] = &[
            2656530319,
            3302512583,
            2849385853,
            2230843471,
            2564848859,
        ];
        Bench::new("5x pseudoprime n=32 bits prime")
            .with_samples(50_000)
            .run_seeded(n32, |ns| for &n in ns {
                assert!(yamaquasi::isprime64(n));
            })
    },
    {
        let n32: &[u64] = &[
            3175244279,
            2508597067,
            2975415077,
            2795977133,
            2249742151,
        ];
        Bench::new("5x pseudoprime n=32 bits composite")
            .with_samples(50_000)
            .run_seeded(n32, |ns| for &n in ns {
                assert!(fbase::certainly_composite(n));
            })
    },
    {
        let n48: &[u64] = &[
            248121875018861,
            264260672274707,
            196470019792451,
            227125290206527,
            221249422787797,
        ];
        Bench::new("5x pseudoprime n=48 bits prime")
            .with_samples(50_000)
            .run_seeded(n48, |ns| for &n in ns {
                assert!(yamaquasi::isprime64(n));
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
        Bench::new("5x pseudoprime n=48 bits composite")
            .with_samples(50_000)
            .run_seeded(n48, |ns| for &n in ns {
                assert!(fbase::certainly_composite(n));
            })
    },
    {
        let ns: &[u64] = &[
            707333682626518879,
            1058212004227469521,
            1037671100376454927,
            623197911999278353,
            1012592912978925827
        ];
        Bench::new("5x pseudoprime n=60 bits prime")
            .with_samples(50_000)
            .run_seeded(ns, |ns| for &n in ns {
                assert!(yamaquasi::isprime64(n));
            })
    },
    {
        let ns: &[u64] = &[
            590475982812721451,
            741485386388025119,
            483522723378941153,
            699484560272004929,
            634432471312362553
        ];
        Bench::new("5x pseudoprime n=60 bits composite")
            .with_samples(50_000)
            .run_seeded(ns, |ns| for &n in ns {
                assert!(fbase::certainly_composite(n));
            })
    },
    {
        let mut primes = vec![];
        let mut q = 0xabcdef;
        while primes.len() < 200 {
            q += 31;
            if !fbase::certainly_composite(q) {
                primes.push(q)
            }
        }
        Bench::new("100x try_factor composite n=48 bits")
            .with_samples(1000)
            .run_seeded((), |_| for i in 0..100 {
                let n = primes[2*i] * primes[2*i+1];
                fbase::try_factor64(n).unwrap();
            })
    },
}
