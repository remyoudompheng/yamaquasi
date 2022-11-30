use crate::Uint;

// Fits cache size per core on most reasonable CPUs.
pub const BLOCK_SIZE: usize = 512 * 1024;

pub fn factor_base_size(n: Uint) -> u32 {
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
    n.bits() as u64 / 4
}

pub fn qs_blocksize(n: &Uint) -> usize {
    let sz = n.bits();
    match sz {
        // QS converges quickly for small inputs
        0..=59 => 1 << 13,
        60..=84 => 1 << (2 + sz / 5), // 14..18
        _ => BLOCK_SIZE,
    }
}

pub fn mpqs_interval_logsize(n: &Uint) -> u32 {
    // 90 bits => 512k
    // 120 bits => 1M
    // 160 bits => 2M
    // 200 bits => 4M
    // 250 bits => 8M
    let sz = n.bits();
    match sz {
        // Small numbers don't have enough polynomials,
        // sieve large intervals
        0..=79 => 17,
        60..=99 => 16,
        // Ordinary growth
        100..=119 => 16 + sz / 30, // 16..19
        120..=239 => 17 + sz / 40, // 20..22
        _ => 23,
    }
}

pub fn target_size(n: &Uint) -> u32 {
    // Target smooth factor size during sieve
    // When smooths are abundant, don't try too much
    // to find large factors.
    let sz = n.bits();
    match sz {
        // Large factor can be 10-16 bits
        0..=127 => sz / 2 + 4,
        // Large factor is about 20 bits
        // Target lower than expected sz/2+8
        128..=192 => sz / 2,
        // Lower target for large sizes
        _ => sz / 2 + 10 - sz / 20,
    }
}

#[test]
fn test_factor_base_size() {
    let b80 = factor_base_size(Uint::from(1u64) << 80);
    let b100 = factor_base_size(Uint::from(1u64) << 100);
    let b120 = factor_base_size(Uint::from(1u64) << 120);
    eprintln!("Bound 80-bit: {}", b80);
    eprintln!("Bound 100-bit: {}", b100);
    eprintln!("Bound 120-bit: {}", b120);
    assert!(80 <= b80 && b80 <= 120);
    assert!(160 <= b100 && b100 <= 240);
    assert!(320 <= b120 && b120 <= 480);

    let b160 = factor_base_size(Uint::from(1u64) << 160);
    let b192 = factor_base_size(Uint::from(1u64) << 192);
    let b224 = factor_base_size(Uint::from(1u64) << 224);
    let b256 = factor_base_size(Uint::from(1u64) << 256);
    eprintln!("Bound 160-bit: {}", b160);
    eprintln!("Bound 192-bit: {}", b192);
    eprintln!("Bound 224-bit: {}", b224);
    eprintln!("Bound 256-bit: {}", b256);

    assert!(1200 <= b160 && b160 <= 2000);
    assert!(2500 <= b192 && b192 <= 5000);
    assert!(7000 <= b224 && b224 <= 15000);
    assert!(30_000 <= b256 && b256 <= 50_000);
}
