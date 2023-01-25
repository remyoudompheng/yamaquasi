use std::str::FromStr;

use brunch::Bench;
use yamaquasi::arith_montgomery::{self, ZmodN};
use yamaquasi::Uint;

fn main() {
    // A 127-bit prime
    let p127 = Uint::from_str("147996737836217455754024183878179633223").unwrap();
    // A 256-bit prime
    let p256 = Uint::from_str(
        "92786510271815932444618978328822237837414362351005653014234479629925371473357",
    )
    .unwrap();
    let p480 = Uint::from_str(
        "1620223189206300768240993211435566664455395791250863601354022165952995299867069991801589111822422435474968009734022186278969436824263868898099719",
    ).unwrap();
    brunch::benches! {
        inline:
        // Small Montgomery arithmetic.
        {
            let p: u64 = 0x3460000000000001;
            let pinv = arith_montgomery::mg_2adic_inv(p);
            assert!(pinv == p-2);
            let x = 0x123abc123abcdef;
            let y = 0x456def456def123;
            let mut z = 0;
            Bench::new("1000x Z/pZ mul (p=p62)")
                .with_samples(1_000)
                .run_seeded((x, y), |(x, y)| {
                    for _ in 0..1_000 {
                        z = arith_montgomery::mg_mul(p, pinv, x, y);
                    }
                })
        },
        {
            use yamaquasi::arith_fft;
            let n = Uint::from_str("2953951639731214343967989360202131868064542471002037986749").unwrap();
            let zn = ZmodN::new(n);
            let x = zn.from_int(12345_u64.into());
            let mzn = arith_fft::MultiZmodP::new(&zn, 3);
            Bench::new("1000x roundtrip MultiZN")
                .run_seeded(x, |x| {
                    let mut mx = vec![0; 8];
                    for _ in 0..1000 {
                        mzn.from_mint(&mut mx, &x);
                        assert_eq!(x, mzn.to_mint(&mx))
                    }
                })
        },
        {
            let zn = ZmodN::new(p127);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("1000x Z/pZ add (p=p127)")
                .with_samples(10_000)
                .run_seeded((x, y), |(x, y)| {
                    for _ in 0..1000 {
                        zn.add(&x, &y);
                    }
                })
        },
        {
            let zn = ZmodN::new(p127);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("1000x Z/pZ sub (p=p127)")
                .with_samples(10_000)
                .run_seeded((x, y), |(x, y)| {
                    for _ in 0..1000 {
                        zn.sub(&x, &y);
                    }
                })
        },
        {
            let zn = ZmodN::new(p127);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("1000x Z/pZ mul (p=p127)")
                .with_samples(10_000)
                .run_seeded((x, y), |(x, y)| {
                    for _ in 0..1000 {
                        zn.mul(&x, &y);
                    }
                })
        },
        {
            let zn = ZmodN::new(p256);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("1000x Z/pZ add (p=p256)")
                .with_samples(10_000)
                .run_seeded((x, y), |(x, y)| {
                    for _ in 0..1000 {
                        zn.add(&x, &y);
                    }
                })
        },
        {
            let zn = ZmodN::new(p256);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("1000x Z/pZ sub (p=p256)")
                .with_samples(10_000)
                .run_seeded((x, y), |(x, y)| {
                    for _ in 0..1000 {
                        zn.sub(&x, &y);
                    }
                })
        },
        {
            let zn = ZmodN::new(p256);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("1000x Z/pZ mul (p=p256)")
                .with_samples(10_000)
                .run_seeded((x, y), |(x, y)| {
                    for _ in 0..1000 {
                        zn.mul(&x, &y);
                    }
                })
        },
        {
            let zn = ZmodN::new(p256);
            let mut x = [0_u64; 16];
            x[0..7].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7]);
            Bench::new("1000x Z/pZ REDC (p=p256)")
                .with_samples(10_000)
                .run_seeded(x, |x| {
                    for _ in 0..1000 {
                        zn.redc(&x);
                    }
                })
        },
        {
            let zn = ZmodN::new(p256);
            let x = zn.from_int(Uint::from(12345_u64));
            Bench::new("1000x Z/pZ inv (p=p256)").run_seeded(x, |x| {
                for _ in 0..1000 {
                    zn.inv(x);
                }
            })
        },
        {
            let zn = ZmodN::new(p480);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("1000x Z/pZ add (p=p480)")
                .with_samples(10_000)
                .run_seeded((x, y), |(x, y)| {
                    for _ in 0..1000 {
                        zn.add(&x, &y);
                    }
                })
        },
        {
            let zn = ZmodN::new(p480);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("1000x Z/pZ sub (p=p480)")
                .with_samples(10_000)
                .run_seeded((x, y), |(x, y)| {
                    for _ in 0..1000 {
                        zn.sub(&x, &y);
                    }
                })
        },
        {
            let zn = ZmodN::new(p480);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("1000x Z/pZ mul (p=p480)")
                .with_samples(10_000)
                .run_seeded((x, y), |(x, y)| {
                    for _ in 0..1000 {
                        zn.mul(&x, &y);
                    }
                })
        },
        {
            let zn = ZmodN::new(p480);
            let x = zn.from_int(Uint::from(12345_u64));
            Bench::new("1000x Z/pZ inv (p=p480)").run_seeded(x, |x| {
                for _ in 0..1000 {
                    zn.inv(x);
                }
            })
        },
    }
}
