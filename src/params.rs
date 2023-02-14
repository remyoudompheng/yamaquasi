use crate::Uint;

// Fits L1 cache size per core on most reasonable CPUs.
pub const BLOCK_SIZE: usize = 32 * 1024;

pub fn factor_base_size(n: &Uint) -> u32 {
    let sz = n.bits();
    if sz < 160 {
        // Factor bases from Silverman's article
        // 100 primes for 80 bits
        // 200 primes for 100 bits
        // 400 primes for 120 bits
        return ((20 + (sz % 20)) << (sz / 20)) / 3;
    } else {
        // 500 for 192 bits, grow as exp(C sqrt(sz))
        let rt = ((256 * sz) as f64).sqrt() as u32;
        let (a, b) = (rt / 12, rt % 12);
        return (12 + b) << (a - 10);
    }
}

pub fn large_prime_factor(n: &Uint) -> u64 {
    // Allow large cofactors up to FACTOR * largest prime
    if n.bits() < 50 {
        // Even larger cofactors for extremely small numbers
        // 100 => 200
        100 + 2 * n.bits() as u64
    } else if n.bits() < 100 {
        // 200..100
        300 - 2 * n.bits() as u64
    } else {
        n.bits() as u64
    }
}

/// ECM/P-1 suitable parameters according to values of B2.
/// d1: size of "giant steps", such that φ(d1) is small and close to 2^k
/// d2: number of giant steps, a power of 2 close to φ(d1)/2,
///     multiplied by a number of blocks (2, 3, 4)
///
/// For small values where multipoint evaluation is not used,
/// the power of 2 constraint can be relaxed and d2 should be very close
/// to φ(d1)/2 for optimal cost.
const STAGE2_PARAMS: &[(f64, u64, u64)] = &[
    // B2, d1, d2
    // Using quadratic method, d2=φ(d1)/2, cost d2^2
    (7.7e3, 240, 32),
    (13.2e3, 330, 40),
    (20e3, 420, 48),
    (33e3, 510, 64),
    (53e3, 660, 80),
    (81e3, 840, 96),
    (126e3, 1050, 120),
    (181e3, 1260, 144),
    (323e3, 1680, 192),
    (554e3, 2310, 240),
    (786e3, 2730, 288),
    (1.37e6, 3570, 384),
    // Use polyeval starting from here
    (2.3e6, 4620, 512),       // φ/2=480
    (4.7e6, 4620, 1024),      // φ/2=480
    (7.1e6, 4620, 1536),      // φ/2=480
    (9.5e6, 4620, 2048),      // φ/2=480
    (19e6, 9240, 2048),       // φ/2=960
    (28e6, 9240, 3072),       // φ/2=960
    (38e6, 9240, 4096),       // φ/2=960
    (78e6, 19110, 4096),      // φ/2=2016
    (117e6, 19110, 6144),     // φ/2=2016
    (156e6, 19110, 8192),     // φ/2=2016
    (322e6, 39270, 8192),     // φ/2=3840
    (643e6, 39270, 16384),    // φ/2=3840
    (1.3e9, 79170, 16384),    // φ/2=8064
    (2.6e9, 79170, 32768),    // φ/2=8064
    (5.2e9, 159390, 32768),   // φ/2=15840
    (10.5e9, 159390, 65536),  // φ/2=15840
    (21.6e9, 330330, 65536),  // φ/2=31680
    (32.5e9, 330330, 98304),  // φ/2=31680
    (43e9, 330330, 131072),   // φ/2=31680
    (136e9, 690690, 196608),  // φ/2=63360
    (362e9, 1381380, 262144), // φ/2=126720
    (543e9, 1381380, 393216), // φ/2=126720
];

pub fn stage2_params(b2: f64) -> (f64, u64, u64) {
    *STAGE2_PARAMS
        .iter()
        .min_by(|x, y| (x.0 - b2).abs().total_cmp(&(y.0 - b2).abs()))
        .unwrap()
}

#[test]
fn test_factor_base_size() {
    let b80 = factor_base_size(&(Uint::from(1u64) << 80));
    let b100 = factor_base_size(&(Uint::from(1u64) << 100));
    let b120 = factor_base_size(&(Uint::from(1u64) << 120));
    eprintln!("Bound 80-bit: {}", b80);
    eprintln!("Bound 100-bit: {}", b100);
    eprintln!("Bound 120-bit: {}", b120);
    assert!(80 <= b80 && b80 <= 120);
    assert!(160 <= b100 && b100 <= 240);
    assert!(320 <= b120 && b120 <= 480);

    let b160 = factor_base_size(&(Uint::from(1u64) << 160));
    let b192 = factor_base_size(&(Uint::from(1u64) << 192));
    let b224 = factor_base_size(&(Uint::from(1u64) << 224));
    let b256 = factor_base_size(&(Uint::from(1u64) << 256));
    eprintln!("Bound 160-bit: {}", b160);
    eprintln!("Bound 192-bit: {}", b192);
    eprintln!("Bound 224-bit: {}", b224);
    eprintln!("Bound 256-bit: {}", b256);

    assert!(1200 <= b160 && b160 <= 2000);
    assert!(2500 <= b192 && b192 <= 5000);
    assert!(7000 <= b224 && b224 <= 15000);
    assert!(30_000 <= b256 && b256 <= 50_000);
}
