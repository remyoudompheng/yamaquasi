//! SIMD-friendly trial division
//!
//! For u128 inputs, it takes about 3 Zen4 cycles per prime.
//! - 1800 Zen4 cycles for primes < 4096
//! - 10000 Zen4 cycles for primes < 32768

#![allow(clippy::needless_range_loop)]
#![allow(clippy::assign_op_pattern)]

use std::sync::LazyLock;

use bnum::cast::CastFrom;

use crate::arith_montgomery::mg_2adic_inv;
use crate::fbase;
use crate::Uint;

pub fn factor4096(n: &mut Uint, facs: &mut Vec<Uint>) {
    if n.bits() <= 64 {
        let nred = _factor_u64(u64::cast_from(*n), facs, &BLOCKS_4096);
        *n = nred.into()
    } else if n.bits() <= 128 {
        let nred = _factor_u128(u128::cast_from(*n), facs, &BLOCKS_4096);
        *n = nred.into()
    } else {
        _factor(n, facs, &BLOCKS_4096)
    }
}

pub fn factor16384(n: &mut Uint, facs: &mut Vec<Uint>) {
    if n.bits() <= 64 {
        let nred = _factor_u64(u64::cast_from(*n), facs, &BLOCKS_16384);
        *n = nred.into()
    } else if n.bits() <= 128 {
        let nred = _factor_u128(u128::cast_from(*n), facs, &BLOCKS_16384);
        *n = nred.into()
    } else {
        _factor(n, facs, &BLOCKS_16384)
    }
}

pub fn factor128k(n: &mut Uint, facs: &mut Vec<Uint>) {
    if n.bits() <= 64 {
        let nred = _factor_u64(u64::cast_from(*n), facs, &BLOCKS_128K);
        *n = nred.into()
    } else if n.bits() <= 128 {
        let nred = _factor_u128(u128::cast_from(*n), facs, &BLOCKS_128K);
        *n = nred.into()
    } else {
        _factor(n, facs, &BLOCKS_128K)
    }
}

// Returns a guaranteed exact factorization of n.
pub fn factor_u32(n: u32, facs: &mut Vec<Uint>) {
    assert!(n & 1 == 1);

    let mut n = n;
    let mut fs = Vec::with_capacity(16);
    for b in &BLOCKS_U16[..] {
        if b.p[0] * b.p[0] > n {
            break;
        }
        b.factor32(n, &mut fs);
        if !fs.is_empty() {
            for &f in &fs {
                while n % f == 0 {
                    n = n / f;
                    facs.push(f.into());
                }
            }
            fs.clear();
        }
    }
    if n != 1 {
        facs.push(n.into());
    }
}

fn _factor_u64(n: u64, facs: &mut Vec<Uint>, primes: &[Block16]) -> u64 {
    let mut n = n;
    let mut fs = Vec::with_capacity(16);
    for b in primes {
        b.factor64(n, &mut fs);
        if !fs.is_empty() {
            for &f in &fs {
                while n % f as u64 == 0 {
                    n = n / f as u64;
                    facs.push(f.into());
                }
            }
            fs.clear();
        }
    }
    n
}

fn _factor_u128(n: u128, facs: &mut Vec<Uint>, primes: &[Block16]) -> u128 {
    let mut n = n;
    let mut fs = Vec::with_capacity(16);
    for b in primes {
        b.factor128(n, &mut fs);
        if !fs.is_empty() {
            for &f in &fs {
                while n % f as u128 == 0 {
                    n = n / f as u128;
                    facs.push(f.into());
                }
            }
            fs.clear();
        }
    }
    n
}

fn _factor(n: &mut Uint, facs: &mut Vec<Uint>, primes: &[Block16]) {
    assert!(u64::cast_from(*n) & 1 == 1);
    let mut k = n.bits().div_ceil(64);
    let mut n128 = 0;
    if n.bits() <= 128 {
        n128 = u128::cast_from(*n);
    }
    let mut fs = Vec::with_capacity(16);
    for b in primes {
        if n128 as u64 != 0 {
            b.factor128(n128, &mut fs);
        } else {
            b.factor(n, k as usize, &mut fs);
        }
        if !fs.is_empty() {
            for &f in &fs {
                while *n % f as u64 == 0 {
                    *n = *n / f as u64;
                    facs.push(f.into());
                }
                k = n.bits().div_ceil(64);
                if n.bits() <= 128 {
                    n128 = u128::cast_from(*n);
                }
            }
            fs.clear();
        }
    }
}

static BLOCKS_4096: LazyLock<Box<[Block16]>> = LazyLock::new(|| {
    let primes = fbase::primes(600);
    assert!(primes[primes.len() - 1] > 4096);
    let mut blocks = vec![];
    for b in primes[1..].chunks_exact(16) {
        // Only use 15-bit primes to avoid overflows.
        if b[0] > 4096 {
            break;
        }
        blocks.push(Block16::new(std::array::from_fn(|i| b[i])));
    }
    blocks.into_boxed_slice()
});

static BLOCKS_16384: LazyLock<Box<[Block16]>> = LazyLock::new(|| {
    let primes = fbase::primes(2000);
    let mut blocks = vec![];
    for b in primes[1..].chunks_exact(16) {
        // Only use 15-bit primes to avoid overflows.
        if b[0] > 16384 {
            break;
        }
        blocks.push(Block16::new(std::array::from_fn(|i| b[i])));
    }
    blocks.into_boxed_slice()
});

static BLOCKS_128K: LazyLock<Box<[Block16]>> = LazyLock::new(|| {
    let mut primes = fbase::primes(13000);
    assert!(primes[primes.len() - 1] > 131072);
    primes.retain(|&p| (2 < p && p < 11300) || (p < 131072 && p % 12 == 11));
    assert_eq!(primes.len(), 4096);
    let mut blocks = vec![];
    for i in 0..4096 / 16 {
        blocks.push(Block16::new([
            primes[4 * i],
            primes[1024 + 12 * i],
            primes[1024 + 12 * i + 1],
            primes[1024 + 12 * i + 2],
            primes[4 * i + 1],
            primes[1024 + 12 * i + 3],
            primes[1024 + 12 * i + 4],
            primes[1024 + 12 * i + 5],
            primes[4 * i + 2],
            primes[1024 + 12 * i + 6],
            primes[1024 + 12 * i + 7],
            primes[1024 + 12 * i + 8],
            primes[4 * i + 3],
            primes[1024 + 12 * i + 6],
            primes[1024 + 12 * i + 7],
            primes[1024 + 12 * i + 8],
        ]));
    }
    blocks.into_boxed_slice()
});

/// A divisibility tree for a block of 16 primes representable by u16.
#[repr(align(32))]
struct Block16 {
    p: [u32; 16],
    p4: [u64; 4],    // Tree level 2
    pinv4: [u64; 4], // Inverse of p4
    pinv: [u64; 16], // Inverse of p
    bound: [u64; 16],
}

impl Block16 {
    fn new(p: [u32; 16]) -> Self {
        let p4 = std::array::from_fn(|idx| {
            (p[4 * idx] as u64)
                .checked_mul(p[4 * idx + 1] as u64)
                .expect("overflow")
                .checked_mul(p[4 * idx + 2] as u64)
                .expect("overflow")
                .checked_mul(p[4 * idx + 3] as u64)
                .expect("overflow")
        });
        let pinv = std::array::from_fn(|idx| inv64(p[idx] as u64));
        let pinv4 = std::array::from_fn(|idx| mg_2adic_inv(p4[idx]));
        let bound = std::array::from_fn(|idx| u64::MAX / (p[idx] as u64));

        Block16 {
            p,
            p4,
            pinv4,
            pinv,
            bound,
        }
    }

    fn factor64(&self, n: u64, facs: &mut Vec<u32>) {
        let r: [u64; 16] = std::array::from_fn(|i| self.pinv[i].wrapping_mul(n));
        let ok: [bool; 16] = std::array::from_fn(|i| r[i] <= self.bound[i]);
        if ok.iter().any(|&x| x) {
            for i in 0..ok.len() {
                if ok[i] {
                    facs.push(self.p[i]);
                }
            }
        }
    }

    fn factor128(&self, n: u128, facs: &mut Vec<u32>) {
        let nw = [n as u64, (n >> 64) as u64];
        let r4: [u64; 4] = std::array::from_fn(|i| redc64(self.p4[i], self.pinv4[i], &nw));
        let r: [u64; 16] = std::array::from_fn(|i| self.pinv[i].wrapping_mul(r4[i / 4]));
        let ok: [bool; 16] = std::array::from_fn(|i| r[i] <= self.bound[i]);
        if ok.iter().any(|&x| x) {
            for i in 0..ok.len() {
                if ok[i] {
                    facs.push(self.p[i]);
                }
            }
        }
    }

    fn factor(&self, n: &Uint, words: usize, facs: &mut Vec<u32>) {
        let nw = &n.digits()[..words];
        let r4: [u64; 4] = std::array::from_fn(|i| redc64(self.p4[i], self.pinv4[i], nw));
        let r: [u64; 16] = std::array::from_fn(|i| self.pinv[i].wrapping_mul(r4[i / 4]));
        let ok: [bool; 16] = std::array::from_fn(|i| r[i] <= self.bound[i]);
        if ok.iter().any(|&x| x) {
            for i in 0..ok.len() {
                if ok[i] {
                    facs.push(self.p[i]);
                }
            }
        }
    }
}

static BLOCKS_U16: LazyLock<Box<[BlockU32x16]>> = LazyLock::new(|| {
    // All 16-bit odd primes
    let mut blocks = vec![];
    for b in fbase::U16_PRIMES[1..].chunks(16) {
        blocks.push(BlockU32x16::new(std::array::from_fn(|i| {
            if i < b.len() {
                b[i]
            } else {
                1
            }
        })));
    }
    // 6541 primes = 409 blocks
    assert_eq!(blocks.len(), 409);
    blocks.into_boxed_slice()
});

#[repr(align(32))]
struct BlockU32x16 {
    p: [u32; 16],
    pinv: [u32; 16], // Inverse of p
    bound: [u32; 16],
}

impl BlockU32x16 {
    fn new(p: [u32; 16]) -> Self {
        let pinv = std::array::from_fn(|idx| inv32(p[idx]));
        let bound = std::array::from_fn(|idx| if p[idx] == 1 { 0 } else { u32::MAX / p[idx] });

        BlockU32x16 { p, pinv, bound }
    }

    fn factor32(&self, n: u32, facs: &mut Vec<u32>) {
        let r: [u32; 16] = std::array::from_fn(|i| self.pinv[i].wrapping_mul(n));
        let ok: [bool; 16] = std::array::from_fn(|i| r[i] <= self.bound[i]);
        if ok.iter().any(|&x| x) {
            for i in 0..ok.len() {
                if ok[i] {
                    facs.push(self.p[i]);
                }
            }
        }
    }
}

fn inv64(x: u64) -> u64 {
    let mut y = x;
    for _ in 0..5 {
        y = y.wrapping_mul(2_u64.wrapping_sub(x.wrapping_mul(y)));
    }
    y
}

fn inv32(x: u32) -> u32 {
    let mut y = x;
    for _ in 0..4 {
        y = y.wrapping_mul(2_u32.wrapping_sub(x.wrapping_mul(y)));
    }
    y
}

/// Compute a pseudo-remainder r ~= n % m such gcd(r,m)=gcd(n,m)
fn redc64(m: u64, minv: u64, n: &[u64]) -> u64 {
    // Apply REDC reduction to n
    let mut acc = 0; // acc < m
    for &ni in n {
        // Invariant: acc is the reduction of n[:i] and acc <= m
        // Beware: overflow can happen even if acc<m and ni<m
        let macc = m - acc;
        if ni >= macc {
            acc = ni - macc;
        } else {
            acc = ni + acc; // ni+acc <= m
        }
        if acc != 0 {
            let mn = acc.wrapping_mul(minv);
            debug_assert_eq!(acc.wrapping_add(mn.wrapping_mul(m)), 0);
            acc = 1 + ((mn as u128 * m as u128) >> 64) as u64;
        }
    }
    acc
}

// Compute r such that gcd(r,m)=gcd(n,m)
#[inline]
#[allow(unused)]
fn redc32(m: u32, minv: u32, n: u64) -> u32 {
    let mn = (n as u32).wrapping_mul(minv);
    let res = n + (mn as u64 * m as u64);
    (res >> 32) as u32
}

#[cfg(test)]
use num_integer::Integer;

#[test]
fn test_redc64() {
    // m = 6426834311640878753
    let m = 17 * 37 * 59 * 61 * 67 * 101 * 127 * 131 * 151 * 167;
    let minv = mg_2adic_inv(m);
    let x: u128 = 80976961557859824481803872805632063005;
    let r = redc64(m, minv, &[x as u64, (x >> 64) as u64]);
    assert_eq!(r.gcd(&m), 37 * 127 * 167);
}

#[test]
fn test_redc32() {
    let m: u32 = 37 * 59 * 61 * 67 * 101;
    let minv = mg_2adic_inv(m as u64) as u32;
    let x: u64 = 690885575;
    let r = redc32(m, minv, x);
    assert_eq!(r.gcd(&m), 59 * 67);
}

#[test]
fn test_inv32() {
    for x in 0..32768 {
        let x = 2 * x + 1;
        let y = inv32(x);
        assert_eq!(x.wrapping_mul(y), 1);
    }
}

#[test]
fn test_factor() {
    use std::str::FromStr;

    let b = Block16::new([
        17, 23, 37, 59, 61, 67, 101, 127, 131, 151, 167, 197, 229, 239, 271, 277,
    ]);
    let mut facs = vec![];
    b.factor128(80976961557859824481803872805632063005, &mut facs);
    assert_eq!(&facs, &[37, 127, 167]);

    let mut facs = vec![];
    let x = 3553516504924158683018250089368740_u128;
    for b in &*BLOCKS_4096 {
        b.factor128(x, &mut facs);
    }
    assert_eq!(&facs, &[3, 5, 11, 23, 67, 83, 199, 541]);

    let x = Uint::from_str("133238636072612598477702109711586687001916037112939054138101340113")
        .unwrap();
    let mut facs = vec![];
    for b in &*BLOCKS_4096 {
        b.factor(&x, 4, &mut facs);
    }
    assert_eq!(&facs, &[23, 67, 83, 199, 541]);
}

#[test]
fn test_factor_u32() {
    let n = 5 * 5;
    let mut facs = vec![];
    factor_u32(n, &mut facs);
    assert_eq!(&facs, &[Uint::from(5u32), Uint::from(5_u32)]);

    let n = 65521 * 65537;
    let mut facs = vec![];
    factor_u32(n, &mut facs);
    assert_eq!(&facs, &[Uint::from(65521u32), Uint::from(65537_u32)]);

    let n = 5 * 4337 * 32089;
    let mut facs = vec![];
    factor_u32(n, &mut facs);
    assert_eq!(
        facs,
        [5, 4337, 32089]
            .into_iter()
            .map(Uint::from_digit)
            .collect::<Vec<_>>()
    );

    let n = 5 * 5 * 7 * 20936329;
    let mut facs = vec![];
    factor_u32(n, &mut facs);
    assert_eq!(
        facs,
        [5, 5, 7, 20936329]
            .into_iter()
            .map(Uint::from_digit)
            .collect::<Vec<_>>()
    );

    let n = 5 * 5 * 13 * 101 * 367;
    let mut facs = vec![];
    factor_u32(n, &mut facs);
    assert_eq!(
        facs,
        [5, 5, 13, 101, 367]
            .into_iter()
            .map(Uint::from_digit)
            .collect::<Vec<_>>()
    );
}
