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
    brunch::benches! {
        inline:
        {
            let zn = ZmodN::new(p127);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("Z/pZ add (p=p127)")
                .with_samples(5_000_000)
                .run_seeded((x, y), |(x, y)| zn.add(x, y))
        },
        {
            let zn = ZmodN::new(p127);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("Z/pZ sub (p=p127)")
                .with_samples(5_000_000)
                .run_seeded((x, y), |(x, y)| zn.sub(x, y))
        },
        {
            let zn = ZmodN::new(p127);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("Z/pZ mul (p=p127)")
                .with_samples(5_000_000)
                .run_seeded((x, y), |(x, y)| zn.mul(x, y))
        },
        {
            let zn = ZmodN::new(p256);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("Z/pZ add (p=p256)")
                .with_samples(5_000_000)
                .run_seeded((x, y), |(x, y)| zn.add(x, y))
        },
        {
            let zn = ZmodN::new(p256);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("Z/pZ sub (p=p256)")
                .with_samples(5_000_000)
                .run_seeded((x, y), |(x, y)| zn.sub(x, y))
        },
        {
            let zn = ZmodN::new(p256);
            let x = zn.from_int(Uint::from(12345_u64));
            let y = zn.from_int(Uint::from(1234567_u64));
            Bench::new("Z/pZ mul (p=p256)")
                .with_samples(5_000_000)
                .run_seeded((x, y), |(x, y)| zn.mul(x, y))
        },
    }
}
