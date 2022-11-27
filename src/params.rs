use crate::Uint;

// Fits cache size per core on most reasonable CPUs.
pub const BLOCK_SIZE: usize = 1024 * 1024;

pub fn smooth_bound(n: Uint) -> u32 {
    // x => sqrt(2)/4 * sqrt(x * log(2) * log(x * log(2)))
    let bits = n.bits() as f64;
    let x = bits.sqrt() * bits.ln().sqrt();
    (0.4 * x - 1.4).exp() as u32
}

pub fn sieve_interval_logsize(n: Uint) -> u32 {
    // 200 bits => 1M
    // 250 bits => 32M
    // 300 bits => 1B
    let sz = n.bits() / 11;
    if sz < 20 {
        20
    } else {
        sz
    }
}

#[test]
fn test_smooth_bound() {
    // msieve uses 22000-25000 we want about 20-30k
    let b160 = smooth_bound(Uint::from(1u64) << 160);
    eprintln!("Bound 160-bit: {}", b160);

    // msieve uses 48000 we want about 80k
    let b192 = smooth_bound(Uint::from(1u64) << 192);
    eprintln!("Bound 192-bit: {}", b192);

    // msieve uses 130k
    let b216 = smooth_bound(Uint::from(1u64) << 216);
    eprintln!("Bound 216-bit: {}", b216);

    // msieve uses 800k-900k, we want about 1.1M
    let b256 = smooth_bound(Uint::from(1u64) << 256);
    eprintln!("Bound 256-bit: {}", b256);

    assert!(20_000 <= b160 && b160 <= 30_000);
    assert!(70_000 <= b192 && b192 <= 100_000);
    assert!(200_000 <= b216 && b216 <= 500_000);
    assert!(800_000 <= b256 && b256 <= 1_300_000);
}
