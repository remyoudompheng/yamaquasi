use brunch::Bench;
use std::str::FromStr;
use std::time::Duration;
use yamaquasi::arith::{inv_mod, isqrt, sqrt_mod, U1024, U256, U512};
use yamaquasi::Uint;

const N256: &str = "23374454829417248628572084580131596971714744792262629806178559231363799527559";
const N1024: &str = "151952459753478002695019426760010155060843495222227274132379609296400121039669231304773230812180118038110720749720126892606028066428592635259881846540972318178085451540072789829262653604582400850027888747669577446006250152212830539247245081046528476394714357280530544575057923657219245807858740056085355550029";
const P160: &str = "1267700734046967910160193878489299434564357851243";
const PQ256: &str =
    "104567211693678450173299212092863908236097914668062065364632502155864426186497";

brunch::benches! {
    Bench::new("isqrt(u256)").run_seeded(U256::from_str(N256).unwrap(), isqrt),
    Bench::new("isqrt(u1024)").run_seeded(U1024::from_str(N1024).unwrap(), isqrt),
    {
        let prime160 = U512::from_str(P160).unwrap();
        Bench::new("inv_mod(3, 160-bit prime) = Some(...)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded(3, |k: u64| inv_mod(U512::from(k), prime160))
    },

    Bench::new("sqrt_mod(6, 2500213) = None")
    .run_seeded(6, |k:u64| sqrt_mod(k, 2500213)),
    Bench::new("sqrt_mod(7, 2500213) = Some(...)")
    .run_seeded(7, |k:u64| sqrt_mod(k, 2500213)),
    Bench::new("sqrt_mod(7, 2500363) = Some(...)")
    .run_seeded(7, |k:u64| sqrt_mod(k, 2500363)),
    Bench::new("sqrt_mod(11, 300*1024+1) = Some(...)")
    .run_seeded(11, |k:u64| sqrt_mod(k, 300*1024 +1 )),
    Bench::new("sqrt_mod(13, 421*65536+1) = Some(...)")
    .with_timeout(Duration::from_secs(1))
    .run_seeded(13, |k:u64| sqrt_mod(k, 421*65536 +1 )),
    {
        let prime160 = U512::from_str(P160).unwrap();
        Bench::new("sqrt_mod(3, 160-bit prime) as U512 = Some(...)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded(3, |k: u64| sqrt_mod(U512::from(k), prime160))
    },
    {
        let prime160 = U1024::from_str(P160).unwrap();
        Bench::new("sqrt_mod(3, 160-bit prime) as U1024 = Some(...)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded(3, |k: u64| sqrt_mod(U1024::from(k), prime160))
    },
}
