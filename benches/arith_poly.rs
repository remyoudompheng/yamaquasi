use std::str::FromStr;

use yamaquasi::arith_montgomery::{MInt, ZmodN};
use yamaquasi::arith_poly::{Poly, PolyRing};
use yamaquasi::Uint;

fn main() {
    fn bench_poly(zn: &ZmodN) {
        for degree in [
            30, 60, 120, 240, 480, 960, 1920, 4032, 8192, 16384, 32768, 65536, 131072, 262144,
        ] {
            let zr = PolyRing::new(zn, std::cmp::max(256, degree));
            let p1: Vec<MInt> = (1..degree as u64)
                .map(|x: u64| zn.from_int(Uint::from(x * x * 12345 + x * 1234 + 123)))
                .collect();
            let p2: Vec<MInt> = (1..degree as u64)
                .map(|x: u64| zn.from_int(Uint::from(x * x * 56789 + x * 6789 + 789)))
                .collect();
            let pol1 = Poly::new(&zr, p1);
            let pol2 = Poly::new(&zr, p2);

            let start = std::time::Instant::now();
            let res_f = Poly::mul_fft(&pol1, &pol2);
            eprintln!(
                "mFFT degree {degree} in {:.prec$}s",
                start.elapsed().as_secs_f64(),
                prec = if degree <= 2048 { 6 } else { 3 }
            );

            if degree < 131072 {
                let start = std::time::Instant::now();
                let res_k = Poly::mul_karatsuba(&pol1, &pol2);
                eprintln!(
                    "mulK degree {degree} in {:.prec$}s",
                    start.elapsed().as_secs_f64(),
                    prec = if degree <= 2048 { 6 } else { 3 }
                );

                for i in 0..2 * degree as usize - 3 {
                    assert_eq!(res_k.c[i], res_f.c[i], "fail idx={i}");
                }
            }

            if degree <= 8192 {
                let start = std::time::Instant::now();
                let res_b = Poly::mul_basic(&pol1, &pol2);
                eprintln!(
                    "mulB degree {degree} in {:.prec$}s",
                    start.elapsed().as_secs_f64(),
                    prec = if degree <= 2048 { 6 } else { 3 }
                );

                for i in 0..2 * degree as usize - 3 {
                    assert_eq!(res_b.c[i], res_f.c[i], "fail idx={i}");
                }
            }
        }
    }

    eprintln!("Polynomial arithmetic timings (mod 256-bit)");
    let n = Uint::from_str(
        "107910248100432407082438802565921895527548119627537727229429245116458288637047",
    )
    .unwrap();
    let zn = ZmodN::new(n);
    bench_poly(&zn);

    eprintln!("Polynomial arithmetic timings (mod 480-bit)");
    let n = Uint::from_str(
        "1814274712676087950344811991522598371991048724422784825007845656050800905627423692122807639509275259938192211611976651772022623688843091923010451",
    )
    .unwrap();
    let zn480 = ZmodN::new(n);
    bench_poly(&zn480);

    for degree in [
        30, 60, 120, 240, 480, 960, 1920, 4032, 8192, 16384, 32768, 65536, 131072,
    ] {
        let zr = PolyRing::new(&zn, degree);
        let p1: Vec<MInt> = (1..degree as u64)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 12345 + x * 1234 + 123)))
            .collect();
        let p2: Vec<MInt> = (1..degree as u64)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 56789 + x * 6789 + 789)))
            .collect();
        let pol1 = Poly::new(&zr, p1);
        let pol2 = Poly::new(&zr, p2);

        let start = std::time::Instant::now();
        let _ = Poly::div_mod_xn(&pol1, &pol2);
        eprintln!(
            "divN degree {degree} in {:.4}s",
            start.elapsed().as_secs_f64()
        );
    }
    // Product tree from roots
    // Multi evaluation of P (degree 10D) at D points.
    for degree in [
        30, 60, 120, 240, 480, 960, 1920, 4032, 8064, 16128, 32768, 65536, 131072,
    ] {
        let zr = PolyRing::new(&zn, degree);
        let roots: Vec<MInt> = (1..=10 * degree as u64)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 12345 + x * 1234 + 123)))
            .collect();
        let roots2: Vec<MInt> = (1..degree as u64)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 56789 + x * 6789 + 789)))
            .collect();
        let start = std::time::Instant::now();
        let pol = Poly::from_roots(&zr, &roots2);
        eprintln!(
            "product tree {degree} in {:.4}s",
            start.elapsed().as_secs_f64()
        );

        for i in 0..64 {
            let r = roots2[(i * roots2.len()) / 64];
            assert_eq!(pol.eval(r), zn.zero());
        }

        // Benchmark blockwise remainder tree.
        let start = std::time::Instant::now();
        let pol2 = Poly::from_roots(&zr, &roots2);
        let vals = pol2.multi_eval(&roots);
        eprintln!(
            "block multieval {degree} over {} in {:.4}s",
            roots.len(),
            start.elapsed().as_secs_f64()
        );

        for i in 0..64 {
            let idx = (i * roots.len()) / 64;
            assert_eq!(pol2.eval(roots[idx]), vals[idx]);
        }

        // Benchmark reduction modulo smaller polynomial
        // + remainder tree.
        let start = std::time::Instant::now();
        let vals = Poly::roots_eval(&zn, &roots, &roots2);
        eprintln!(
            "mulmod+multieval {} over {degree} in {:.4}s",
            roots.len(),
            start.elapsed().as_secs_f64()
        );

        if roots.len() <= 512 * 1024 {
            let zr = PolyRing::new(&zn, 5 * degree);
            let pol1 = Poly::from_roots(&zr, &roots);
            for i in 0..64 {
                let idx = (i * roots2.len()) / 64;
                assert_eq!(pol1.eval(roots2[idx]), vals[idx]);
            }
        }
    }
}
