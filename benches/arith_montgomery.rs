use std::str::FromStr;

use brunch::Bench;
use yamaquasi::arith_montgomery::ZmodN;
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
