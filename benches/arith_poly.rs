use std::str::FromStr;

use yamaquasi::arith_montgomery::{MInt, ZmodN};
use yamaquasi::arith_poly::Poly;
use yamaquasi::Uint;

fn main() {
    eprintln!("Polynomial arithmetic timings");
    let n = Uint::from_str(
        "107910248100432407082438802565921895527548119627537727229429245116458288637047",
    )
    .unwrap();
    let zn = ZmodN::new(n);
    for degree in [30, 60, 120, 240, 480, 960, 1920, 4032, 8064, 16128] {
        let p1: Vec<MInt> = (1..degree)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 12345 + x * 1234 + 123)))
            .collect();
        let p2: Vec<MInt> = (1..degree)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 56789 + x * 6789 + 789)))
            .collect();
        let pol1 = Poly::new(&zn, p1);
        let pol2 = Poly::new(&zn, p2);

        if degree < 4000 {
            let start = std::time::Instant::now();
            let _ = Poly::mul_basic(&pol1, &pol2);
            eprintln!(
                "mulB degree {degree} in {:.4}s",
                start.elapsed().as_secs_f64()
            );
        }

        let start = std::time::Instant::now();
        let _ = Poly::mul(&pol1, &pol2);
        eprintln!(
            "mulK degree {degree} in {:.4}s",
            start.elapsed().as_secs_f64()
        );

        let start = std::time::Instant::now();
        let _ = Poly::div_mod_xn(&pol1, &pol2);
        eprintln!(
            "divN degree {degree} in {:.4}s",
            start.elapsed().as_secs_f64()
        );
    }
    // Product tree from roots
    for degree in [30, 60, 120, 240, 480, 960, 1920, 4032, 8064, 16128] {
        let roots: Vec<MInt> = (1..=degree)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 12345 + x * 1234 + 123)))
            .collect();
        let roots2: Vec<MInt> = (1..degree - 4)
            .map(|x: u64| zn.from_int(Uint::from(x * x * 56789 + x * 6789 + 789)))
            .collect();
        let start = std::time::Instant::now();
        let _ = Poly::from_roots(&zn, roots.clone());
        eprintln!(
            "product tree {degree} in {:.4}s",
            start.elapsed().as_secs_f64()
        );

        let start = std::time::Instant::now();
        let pol = Poly::from_roots(&zn, roots2);
        pol.multi_eval(roots);
        eprintln!(
            "multieval {degree} in {:.4}s",
            start.elapsed().as_secs_f64()
        );
    }
}
