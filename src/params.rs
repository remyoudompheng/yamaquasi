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
