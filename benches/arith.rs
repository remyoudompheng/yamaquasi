use brunch::Bench;
use std::str::FromStr;
use std::time::Duration;
use yamaquasi::arith::{self, inv_mod, isqrt, sqrt_mod, U1024, U256, U512};
use yamaquasi::Uint;

const N256: &str = "23374454829417248628572084580131596971714744792262629806178559231363799527559";
const N1024: &str = "151952459753478002695019426760010155060843495222227274132379609296400121039669231304773230812180118038110720749720126892606028066428592635259881846540972318178085451540072789829262653604582400850027888747669577446006250152212830539247245081046528476394714357280530544575057923657219245807858740056085355550029";
const P160: &str = "1267700734046967910160193878489299434564357851243";
const PQ256: &str =
    "104567211693678450173299212092863908236097914668062065364632502155864426186497";

brunch::benches! {
    Bench::new("isqrt(u256)").run_seeded(U256::from_str(N256).unwrap(), isqrt),
    Bench::new("isqrt(u1024)").run_seeded(U1024::from_str(N1024).unwrap(), isqrt),
    // Small modular inverses
    {
        let div = arith::Dividers::new(257);
        Bench::new("1000x inv_mod binary(17, 257)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded((17, &div), |(k, div)| for _ in 0..1000 { div.inv(k); })
    },
    {
        let div = arith::Dividers::new(65537);
        Bench::new("1000x inv_mod binary(17, 65537)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded((17, &div), |(k, div)| for _ in 0..1000 { div.inv(k); })
    },
    {
        let div = arith::Dividers::new(65537);
        Bench::new("1000x inv_mod binary(40507, 65537)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded((40507, &div), |(k, div)| for _ in 0..1000 { div.inv(k); })
    },
    {
        let div = arith::Dividers::new(1048583);
        Bench::new("1000x inv_mod binary(4057, 1048583)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded((4057, &div), |(k, div)| for _ in 0..1000 { div.inv(k); })
    },
    {
        let div = arith::Dividers::new(1048583);
        Bench::new("1000x inv_mod binary(1234567, 1048583)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded((1234567, &div), |(k, div)| for _ in 0..1000 { div.inv(k); })
    },
    {
        let inv = arith::Inverter::new(257);
        Bench::new("1000x inv_mod fast(17, 257)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded((17, &inv), |(k, inv)| for _ in 0..1000 { inv.invert(k); })
    },
    {
        let inv = arith::Inverter::new(65537);
        Bench::new("1000x inv_mod fast(17, 65537)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded((17, &inv), |(k, inv)| for _ in 0..1000 { inv.invert(k); })
    },
    {
        let inv = arith::Inverter::new(65537);
        Bench::new("1000x inv_mod fast(40507, 65537)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded((40507, &inv), |(k, inv)| for _ in 0..1000 { inv.invert(k); })
    },
    {
        let inv = arith::Inverter::new(1048583);
        Bench::new("1000x inv_mod fast(4057, 1048583)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded((4057, &inv), |(k, inv)| for _ in 0..1000 { inv.invert(k); })
    },
    {
        let inv = arith::Inverter::new(1048583);
        Bench::new("1000x inv_mod fast(1234567, 1048583)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded((1234567, &inv), |(k, inv)| for _ in 0..1000 { inv.invert(k); })
    },
    // Large modular inverses
    {
        let prime160 = U512::from_str(P160).unwrap();
        Bench::new("inv_mod(3, 160-bit prime) = Some(...)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded(3, |k: u64| inv_mod(U512::from(k), prime160).unwrap())
    },
    {
        let prime160 = U1024::from_str(P160).unwrap();
        let n = U1024::from_str(PQ256).unwrap();
        Bench::new("inv_mod(160-bit prime, 256-bit modulus) = Some(...)")
        .with_timeout(Duration::from_secs(1))
        .run_seeded(prime160, |k: Uint| inv_mod(k, n).unwrap())
    },
    // Modular square roots
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
    .run_seeded(13, |k:u64| sqrt_mod(k, 421*65536 + 1)),
    Bench::new("sqrt_mod(13, 13*2^20+1) = Some(...)")
    .with_timeout(Duration::from_secs(1))
    .run_seeded(13, |k:u64| sqrt_mod(k, (13<<20) + 1)),
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
    // Wide division
    {
        let n = U1024::from_str(PQ256).unwrap();
        Bench::new("1000x mod(u256, 65537)")
        .with_samples(5_000)
        .run_seeded((n, 65537), |(n, p)| for _ in 0..1000 { let _ = n % p as u64; })
    },
    {
        let n = U1024::from_str(PQ256).unwrap();
        let d = arith::Dividers::new(65537);
        Bench::new("1000x mod const(u256, 65537)")
        .with_samples(50_000)
        .run_seeded((&n, &d), |(n, d)| for _ in 0..1000 { d.divmod_uint(n).1; })
    }
}
