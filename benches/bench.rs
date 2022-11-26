use brunch::Bench;
use std::time::Duration;
use yamaquasi::{isqrt, sqrt_mod, U1024, U256};

brunch::benches! {
    Bench::new("isqrt(u256)")
    .run_seeded(U256::from_dec_str("23374454829417248628572084580131596971714744792262629806178559231363799527559").unwrap(), isqrt),
    Bench::new("isqrt(u1024)")
    .run_seeded(U1024::from_dec_str("151952459753478002695019426760010155060843495222227274132379609296400121039669231304773230812180118038110720749720126892606028066428592635259881846540972318178085451540072789829262653604582400850027888747669577446006250152212830539247245081046528476394714357280530544575057923657219245807858740056085355550029").unwrap(),
    isqrt),

    Bench::new("sqrt_mod(6, 2500213) = None")
    .run_seeded(6, |k| sqrt_mod(U256::from(k), 2500213)),
    Bench::new("sqrt_mod(7, 2500213) = Some(...)")
    .run_seeded(7, |k| sqrt_mod(U256::from(k), 2500213)),
    Bench::new("sqrt_mod(7, 2500363) = Some(...)")
    .run_seeded(7, |k| sqrt_mod(U256::from(k), 2500363)),
    Bench::new("sqrt_mod(11, 300*1024+1) = Some(...)")
    .run_seeded(11, |k| sqrt_mod(U256::from(k), 300*1024 +1 )),
    Bench::new("sqrt_mod(13, 421*65536+1) = Some(...)")
    .with_timeout(Duration::from_secs(1))
    .run_seeded(13, |k| sqrt_mod(U256::from(k), 421*65536 +1 )),
}
