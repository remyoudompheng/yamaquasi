use std::str::FromStr;
use std::time::Duration;

use brunch::Bench;
use yamaquasi::{self, Uint, Verbosity};

const MAX_SAMPLES: u32 = 10000;

brunch::benches! {
    // 1619364369680082014535399761: 271 2131 5903 15263 199877 155709937
    // 1639767035017446322767934727: 193 797 14593 23857 199447 153524821
    // 1678020989439186534979875539: 461 977 22273 26953 40387 153664829
    // 1627017796754642154672796991: 257 3049 17011 145451 323149 2596883
    // 1706068244626777020404349883: 13 107 277 8117 209621 521471 4990327
    {
        let ns: &[u128] = &[
            1619364369680082014535399761,
            1639767035017446322767934727,
            1678020989439186534979875539,
            1627017796754642154672796991,
            1706068244626777020404349883,
        ];
        Bench::new("5x factor_smooth n=91b B=2^22")
            .with_samples(MAX_SAMPLES)
            .with_timeout(Duration::from_secs(3))
            .run_seeded(ns, |ns| for &n in ns {
                let facs = yamaquasi::factor_smooth(n.into(), 22);
                assert!(facs.len() > 5);
            })
    },
    {
        let ns: &[u128] = &[
            1619364369680082014535399761,
            1639767035017446322767934727,
            1678020989439186534979875539,
            1627017796754642154672796991,
            1706068244626777020404349883,
        ];
        Bench::new("5x factor_small n=91b B=2^22")
            .with_samples(MAX_SAMPLES)
            .with_timeout(Duration::from_secs(3))
            .run_seeded(ns, |ns| for &n in ns {
                let facs = yamaquasi::find_small_factors(n.into(), 22);
                assert!(facs.len() > 5);
            })
    },
    {
        let ns: &[u128] = &[
            26366582694208476520828554904669145, // 5 7 281 661 1069 1259 19037 48479 633253 5156383
            1552874776102907730900115628935835,  // 5 7 457 947 4567 10651 56377 75521 495006751
            6762236801171728559458267005774099,  // 3 7 17 967 1039 7523 38351 53819 559133 2171509
            17227965250634428706527504247997363, // 3 13 701 5399 10979 11423 47149 1005287 19635073
            7894246493576145500729101679488217,  // 7 43 47 379 3433 7247 183917 779927 412571501
        ];
        Bench::new("5x factor_smooth n=114b B=2^24")
            .with_timeout(Duration::from_secs(3))
            .run_seeded(ns, |ns| for &n in ns {
                let facs = yamaquasi::factor_smooth(n.into(), 24);
                assert!(facs.len() > 8);
            })
    },
    {
        let ns: &[u128] = &[
            26366582694208476520828554904669145, // 5 7 281 661 1069 1259 19037 48479 633253 5156383
            1552874776102907730900115628935835,  // 5 7 457 947 4567 10651 56377 75521 495006751
            6762236801171728559458267005774099,  // 3 7 17 967 1039 7523 38351 53819 559133 2171509
            17227965250634428706527504247997363, // 3 13 701 5399 10979 11423 47149 1005287 19635073
            7894246493576145500729101679488217,  // 7 43 47 379 3433 7247 183917 779927 412571501
        ];
        Bench::new("5x factor_small n=114b B=2^24")
            .with_samples(MAX_SAMPLES)
            .with_timeout(Duration::from_secs(3))
            .run_seeded(ns, |ns| for &n in ns {
                let facs = yamaquasi::find_small_factors(n.into(), 24);
                assert!(facs.len() > 8, "{}", facs.len());
            })
    },
    // 27046428237735521937698621600942432775345: 3 3 3 3 3 5 13 17 38611 143527 273901 345139 192268906869881
    // 17917681567176975564890649792772767247239: 3 3 13 43 173 11423 302593 1122841 6350219 835286330713
    // 30938317774575759057866971442921838963057: 3 7 17 31 727 38677 48073 2949931 4345343 161340147061
    // 16008651129507811679184207766759569066777: 3 3 3 487 1123 1549 13693 376039 855089 158960062195633
    // 48227137632137432994650208818675620122165: 3 5 23 8093 388541 2422351 4932139 3720948138410801
    {
        let ns: &[Uint] = &[
            Uint::from_str("27046428237735521937698621600942432775345").unwrap(),
            Uint::from_str("17917681567176975564890649792772767247239").unwrap(),
            Uint::from_str("30938317774575759057866971442921838963057").unwrap(),
            Uint::from_str("16008651129507811679184207766759569066777").unwrap(),
            Uint::from_str("48227137632137432994650208818675620122165").unwrap(),
        ];
        Bench::new("5x factor_smooth n=135b B=2^24")
            .with_samples(MAX_SAMPLES)
            .with_timeout(Duration::from_secs(3))
            .run_seeded(ns, |ns| for &n in ns {
                let facs = yamaquasi::factor_smooth(n.into(), 24);
                assert!(facs.len() >= 8);
            })
    },
    {
        let ns: &[Uint] = &[
            Uint::from_str("27046428237735521937698621600942432775345").unwrap(),
            Uint::from_str("17917681567176975564890649792772767247239").unwrap(),
            Uint::from_str("30938317774575759057866971442921838963057").unwrap(),
            Uint::from_str("16008651129507811679184207766759569066777").unwrap(),
            Uint::from_str("48227137632137432994650208818675620122165").unwrap(),
        ];
        Bench::new("5x factor_small n=135b B=2^24")
            .with_samples(MAX_SAMPLES)
            .with_timeout(Duration::from_secs(3))
            .run_seeded(ns, |ns| for &n in ns {
                let facs = yamaquasi::find_small_factors(n.into(), 24);
                assert!(facs.len() >= 8);
            })
    },
    // 44787627309529707312627764656534371155276280609020817807: 3 71 919 8311 53411 330313 464089 561943 1134673 1330309 3964025982923
    // 86710226916342934785784639648639193682768135431599457897: 797 13537 15823 110989 160409 318247 859787 1082093 96354564796325663
    // 87549538969133232697531126588719571179794594548406215861: 19 251 5437 6553 17657 27061 354169 885403 2787259 1233779133581183029
    // 28430495716025308846834128727575403606731564103738279227: 3 13 173 78643 514529 645907 2392147 2920523 2949049 23619173 331313117
    // 48966597617671202139464580151879657167535951923129890945: 5 13 599 7883 9743 16661 54851 71293 1015661 2176477 9902257 11481693889
    {
        let ns: &[Uint] = &[
            Uint::from_str("44787627309529707312627764656534371155276280609020817807").unwrap(),
            Uint::from_str("86710226916342934785784639648639193682768135431599457897").unwrap(),
            Uint::from_str("87549538969133232697531126588719571179794594548406215861").unwrap(),
            Uint::from_str("28430495716025308846834128727575403606731564103738279227").unwrap(),
            Uint::from_str("48966597617671202139464580151879657167535951923129890945").unwrap(),
        ];
        Bench::new("5x factor_smooth n=185b B=2^24")
            .with_samples(MAX_SAMPLES)
            .with_timeout(Duration::from_secs(3))
            .run_seeded(ns, |ns| for &n in ns {
                let facs = yamaquasi::factor_smooth(n.into(), 24);
                assert!(facs.len() >= 9);
            })
    },
    {
        let ns: &[Uint] = &[
            Uint::from_str("44787627309529707312627764656534371155276280609020817807").unwrap(),
            Uint::from_str("86710226916342934785784639648639193682768135431599457897").unwrap(),
            Uint::from_str("87549538969133232697531126588719571179794594548406215861").unwrap(),
            Uint::from_str("28430495716025308846834128727575403606731564103738279227").unwrap(),
            Uint::from_str("48966597617671202139464580151879657167535951923129890945").unwrap(),
        ];
        Bench::new("5x factor_small n=185b B=2^24")
            .with_samples(MAX_SAMPLES)
            .with_timeout(Duration::from_secs(3))
            .run_seeded(ns, |ns| for &n in ns {
                let facs = yamaquasi::find_small_factors(n.into(), 24);
                assert!(facs.len() >= 9, "{n} => {facs:?}");
            })
    },
    Bench::new("trialdiv 5x n=32b")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            [3992007061, 4217676929, 1063407835, 3769986693, 1264635665],
            |ns: [u32; 5]| {
                for n in ns {
                    let mut facs = Vec::with_capacity(8);
                    yamaquasi::trialdiv::factor_u32(n, &mut facs);
                    assert!(facs.len() >= 2);
                }
            }
        ),
    Bench::new("trialdiv 5x n=64b bound=4096")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            [9877823563940357795, 2248911720047503905, 3075973318107414345, 17849812464295589477, 9835686156758565361],
            |ns: [u64; 5]| {
                for n in ns {
                    let mut facs = Vec::with_capacity(8);
                    yamaquasi::trialdiv::factor4096(&mut Uint::from_digit(n), &mut facs);
                    assert!(facs.len() >= 4);
                }
            }
        ),
    Bench::new("trialdiv n=114b bound=small dividers")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from(7894246493576145500729101679488217_u128),
            |n| {
                use yamaquasi::fbase;

                let mut nred = n;
                let mut facs: Vec<Uint> = Vec::with_capacity(8);
                for (&p, div) in fbase::SMALL_PRIMES
                    .iter()
                    .zip(&fbase::SMALL_PRIMES_DIVIDERS)
                {
                    loop {
                        let (q, r) = div.divmod_uint(&nred);
                        if r != 0 {
                            break;
                        }
                        nred = q;
                        facs.push(p.into());
                    }
                }
            }),
    Bench::new("trialdiv n=114b bound=4096")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from(7894246493576145500729101679488217_u128),
            |n| {
                let mut n = n;
                let mut facs = Vec::with_capacity(8);
                yamaquasi::trialdiv::factor4096(&mut n, &mut facs);
                assert!(facs.len() == 5, "{facs:?}");
            },
        ),
    Bench::new("trialdiv n=135b bound=4096")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from_str("17917681567176975564890649792772767247239").unwrap(),
            |n| {
                let mut n = n;
                let mut facs = Vec::with_capacity(8);
                yamaquasi::trialdiv::factor4096(&mut n, &mut facs);
                assert!(facs.len() == 5, "{facs:?}");
            },
        ),
    Bench::new("trialdiv n=135b bound=16K")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from_str("17917681567176975564890649792772767247239").unwrap(),
            |n| {
                let mut n = n;
                let mut facs = Vec::with_capacity(8);
                yamaquasi::trialdiv::factor16384(&mut n, &mut facs);
                assert!(facs.len() == 6, "{facs:?}");
            },
        ),
    Bench::new("trialdiv n=135b bound=128K")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from_str("17917681567176975564890649792772767247239").unwrap(),
            |n| {
                let mut n = n;
                let mut facs = Vec::with_capacity(8);
                yamaquasi::trialdiv::factor128k(&mut n, &mut facs);
                assert!(facs.len() >= 5, "{facs:?}");
            },
        ),
    Bench::new("trialdiv n=185b bound=128K")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from_str("44787627309529707312627764656534371155276280609020817807").unwrap(),
            |n| {
                let mut n = n;
                let mut facs = Vec::with_capacity(8);
                yamaquasi::trialdiv::factor128k(&mut n, &mut facs);
                assert!(facs.len() >= 4, "{facs:?}");
            },
        ),
    {
        let ns: &[u128] = &[
            1619364369680082014535399761,
            1639767035017446322767934727,
            1678020989439186534979875539,
            1627017796754642154672796991,
            1706068244626777020404349883,
        ];
        Bench::new("trialdiv n=5x91b bound=128K")
            .with_samples(MAX_SAMPLES)
            .with_timeout(Duration::from_secs(3))
            .run_seeded(ns, |ns| {
                let mut x: Uint = ns.iter().copied().map(Uint::from).product();
                let mut facs = Vec::with_capacity(16);
                yamaquasi::trialdiv::factor128k(&mut x, &mut facs);
                assert!(facs.len() > 10, "{facs:?}");
            })
    },
    Bench::new("rho_smooth u64 iters=1024")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from(113831488164011_u64),
            |n| {
                let res = yamaquasi::pollard_rho::rho_impl(&n, 2, 1024, Verbosity::Silent);
                assert!(res.is_none());
            },
        ),
    Bench::new("rho_smooth n=114b iters=1024")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from(20586446766667442854177681947386053_u128),
            |n| {
                let (facs, _) = yamaquasi::pollard_rho::rho_impl(&n, 2, 1024, Verbosity::Silent).unwrap();
                assert!(facs.len() >= 1, "{facs:?}");
            },
        ),
    Bench::new("rho_smooth n=114b iters=2048")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from(20586446766667442854177681947386053_u128),
            |n| {
                let (facs, _ ) = yamaquasi::pollard_rho::rho_impl(&n, 2, 2048, Verbosity::Silent).unwrap();
                assert!(facs.len() >= 1, "{facs:?}");
            },
        ),
    Bench::new("rho_smooth n=135b iters=1024")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from_str("17917681567176975564890649792772767247239").unwrap(),
            |n| {
                let (facs, _) = yamaquasi::pollard_rho::rho_impl(&n, 2, 1024, Verbosity::Silent).unwrap();
                assert!(facs.len() >= 5, "{facs:?}");
            },
        ),
    Bench::new("rho_smooth n=185b iters=1024")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            Uint::from_str("44787627309529707312627764656534371155276280609020817807").unwrap(),
            |n| {
                let (facs, _) = yamaquasi::pollard_rho::rho_impl(&n, 2, 1024, Verbosity::Silent).unwrap();
                assert!(facs.len() >= 5, "{facs:?}");
            },
        ),
    Bench::new("pm1_smooth u64 budget=1000")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            (
            Uint::from(545498994516194557_u64),
            std::rc::Rc::new(yamaquasi::pollard_pm1::PM1Base::newsmall(250, 4000)),
            ),
            |(n, pbase)| {
                let facs = pbase.pm1(&n);
                assert!(facs.len() >= 1, "{facs:?}");
            },
        ),
    Bench::new("pm1_smooth n=114b budget=1000")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            (
                Uint::from(20586446766667442854177681947386053_u128),
                std::rc::Rc::new(yamaquasi::pollard_pm1::PM1Base::newsmall(250, 4000)),
            ),
            |(n, pbase)| {
                let facs = pbase.pm1(&n);
                assert!(facs.len() >= 2);
            },
        ),
    Bench::new("pm1_smooth n=114b budget=2000")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            (
                Uint::from(20586446766667442854177681947386053_u128),
                std::rc::Rc::new(yamaquasi::pollard_pm1::PM1Base::newsmall(320, 12000)),
            ),
            |(n, pbase)| {
                let facs = pbase.pm1(&n);
                assert!(facs.len() >= 2);
            },
        ),
    Bench::new("pm1_smooth n=135b budget=1000")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            (
                Uint::from_str("17917681567176975564890649792772767247239").unwrap(),
                std::rc::Rc::new(yamaquasi::pollard_pm1::PM1Base::newsmall(250, 4000)),
            ),
            |(n, pbase)| {
                let facs = pbase.pm1(&n);
                assert!(facs.len() >= 2, "{facs:?}");
            },
        ),
    Bench::new("pm1_smooth n=185b budget=1000")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            (
                Uint::from_str("44787627309529707312627764656534371155276280609020817807").unwrap(),
                std::rc::Rc::new(yamaquasi::pollard_pm1::PM1Base::newsmall(250, 4000)),
            ),
            |(n, pbase)| {
                let facs = pbase.pm1(&n);
                assert!(facs.len() >= 2);
            },
        ),
    Bench::new("pm1_smooth n=185b budget=20000")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            (
                Uint::from_str("113675669689084719521492435318032135717384168831434093549").unwrap(),
                std::rc::Rc::new(yamaquasi::pollard_pm1::PM1Base::newsmall(1000, 200_000)),
            ),
            |(n, pbase)| {
                let facs = pbase.pm1(&n);
                assert!(facs.len() >= 2);
            },
        ),
    Bench::new("pm1_smooth n=250b budget=20000")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            (
                Uint::from_str("107910248100432407082438802565921895527548119627537727229429245116458288637047").unwrap(),
                std::rc::Rc::new(yamaquasi::pollard_pm1::PM1Base::newsmall(1000, 200_000)),
            ),
            |(n, pbase)| {
                let facs = pbase.pm1(&n);
                assert!(facs.len() <= 1);
            },
        ),
    Bench::new("ecm_smooth n=60b budget=1000 (B1=50, B2=1920)")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            (
                692341156564642103_u128,
                std::rc::Rc::new(yamaquasi::ecm::SmoothBase::new(50, false)),
            ),
            |(n, pbase)| {
                let n = &Uint::from(n);
                let facs = yamaquasi::ecm::ecm_curve_smooth(n, 1, &pbase, 1920.0);
                assert!(facs.len() == 0, "{facs:?}");
            },
        ),
    Bench::new("ecm_smooth n=114b budget=1000 (B1=50, B2=1920)")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            (
                20586446766667442854177681947386053_u128,
                std::rc::Rc::new(yamaquasi::ecm::SmoothBase::new(50, false)),
            ),
            |(n, pbase)| {
                let n = &Uint::from(n);
                let facs = yamaquasi::ecm::ecm_curve_smooth(n, 1, &pbase, 1920.0);
                assert!(facs.len() >= 2, "{facs:?}");
            },
        ),
    Bench::new("ecm_smooth n=114b budget=2000 (B1=75, B2=5040)")
        .with_samples(MAX_SAMPLES)
        .with_timeout(Duration::from_secs(3))
        .run_seeded(
            (
                20586446766667442854177681947386053_u128,
                std::rc::Rc::new(yamaquasi::ecm::SmoothBase::new(75, false)),
            ),
            |(n, pbase)| {
                let n = &Uint::from(n);
                let facs = yamaquasi::ecm::ecm_curve_smooth(n, 1, &pbase, 5040.0);
                assert!(facs.len() >= 2, "{facs:?}");
            },
        ),
}
