use brunch::Bench;
use std::str::FromStr;
use yamaquasi::poly::select_polys;
use yamaquasi::Uint;

const PQ256: &str =
    "104567211693678450173299212092863908236097914668062065364632502155864426186497";

brunch::benches! {
    // Polynomial selection
    {
        let n = Uint::from_str(PQ256).unwrap();
        Bench::new("select_polys(256-bit n) = Some(...)")
        .run_seeded(n, |n| select_polys(25, 0, n))
    },
}
