// Copyright 2022, 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::borrow::Borrow;

use crate::arith;
use crate::Uint;

// Montgomery form arithmetic for 64-bit moduli.

// Returns ninv such that n*ninv = -1
pub fn mg_2adic_inv(n: u64) -> u64 {
    // Invariant: nx = 1 + 2^k s, k increasing
    let mut x = 1u64;
    loop {
        let rem = n.wrapping_mul(x) - 1;
        if rem == 0 {
            break;
        }
        x += 1 << rem.trailing_zeros();
    }
    assert!(n.wrapping_mul(x) == 1);
    1 + !x
}

#[inline(always)]
pub fn mg_mul(n: u64, ninv: u64, x: u64, y: u64) -> u64 {
    mg_redc(n, ninv, (x as u128) * (y as u128))
}

#[inline(always)]
pub fn mg_redc(n: u64, ninv: u64, x: u128) -> u64 {
    // Montgomery reduction (x/R mod n).
    // compute -x/N mod R
    let mul: u64 = (x as u64).wrapping_mul(ninv);
    // reduce
    let m = mul as u128 * n as u128;
    let res = ((x + m) >> 64) as u64;
    if res >= n {
        res - n
    } else {
        res
    }
}

// Montgomery form arithmetic for large (512-bit) integers

#[derive(Clone)]
pub struct ZmodN {
    pub n: Uint,
    // Minus n^-1 mod R
    ninv: MInt,
    // Auxiliary base R=2^64k
    k: u32,
    // R mod n
    r: MInt,
    // R^2 mod n
    r2: MInt,
}

// Store values as 512-bit integers.
const MINT_WORDS: usize = 8;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub struct MInt(pub [u64; MINT_WORDS]);

impl From<MInt> for Uint {
    fn from(m: MInt) -> Self {
        let mut digits = [0u64; Uint::BITS as usize / 64];
        digits[..MINT_WORDS].copy_from_slice(&m.0);
        Uint::from_digits(digits)
    }
}

impl MInt {
    fn from_uint(n: Uint) -> Self {
        let mut m = MInt::default();
        m.0[..].copy_from_slice(&n.digits()[..MINT_WORDS]);
        m
    }
}

impl ZmodN {
    pub fn new(n: Uint) -> Self {
        assert!(n.bits() < Uint::BITS / 2);
        let k = (n.bits() + 63) / 64;
        assert!(n.bits() <= 64 * k);
        let rsqrt = Uint::ONE << (32 * k);
        let r = (rsqrt * rsqrt) % n;
        let r2 = (r * r) % n;
        let ninv = {
            // Invariant: nx = 1 + 2^k s, k increasing
            let mut x = Uint::ONE;
            loop {
                let rem = n.wrapping_mul(x) - Uint::ONE;
                if rem == Uint::ZERO {
                    break;
                }
                x += Uint::ONE << rem.trailing_zeros();
            }
            assert!((n.wrapping_mul(x) - Uint::ONE).trailing_zeros() >= 64 * k);
            // Now compute R-x
            if 64 * k == Uint::BITS {
                !x + Uint::ONE
            } else {
                // clear top bits
                let x = x - ((x >> (64 * k)) << (64 * k));
                (Uint::ONE << (64 * k)) - x
            }
        };
        ZmodN {
            n,
            ninv: MInt::from_uint(ninv),
            k,
            r: MInt::from_uint(r),
            r2: MInt::from_uint(r2),
        }
    }

    pub fn zero(&self) -> MInt {
        MInt::default()
    }

    pub fn one(&self) -> MInt {
        self.r
    }

    pub fn from_int(&self, x: Uint) -> MInt {
        self.mul(MInt::from_uint(x), self.r2)
    }

    pub fn to_int(&self, x: MInt) -> Uint {
        let mut xx = [0u64; 2 * MINT_WORDS];
        xx[..MINT_WORDS].copy_from_slice(&x.0);
        self.redc(&xx).into()
    }

    pub fn mul<M: Borrow<MInt>>(&self, x: M, y: M) -> MInt {
        // all bit lengths MUST be < 512
        debug_assert!(Uint::from(*x.borrow()) < self.n);
        debug_assert!(Uint::from(*y.borrow()) < self.n);
        self.redc(&uint_mul(&x.borrow().0, &y.borrow().0, self.k))
    }

    pub fn inv(&self, x: MInt) -> Option<MInt> {
        // No optimization, use ordinary modular inversion.
        Some(self.from_int(arith::inv_mod(self.to_int(x), self.n)?))
    }

    pub fn add<M: Borrow<MInt>>(&self, x: M, y: M) -> MInt {
        debug_assert!(Uint::from(*x.borrow()) < self.n);
        debug_assert!(Uint::from(*y.borrow()) < self.n);
        let sum = uint_addmod(&x.borrow().0, &y.borrow().0, &self.n, self.k);
        debug_assert!(Uint::from(MInt(sum)) < self.n);
        MInt(sum)
    }

    pub fn sub<M: Borrow<MInt>>(&self, x: M, y: M) -> MInt {
        debug_assert!(Uint::from(*x.borrow()) < self.n);
        debug_assert!(Uint::from(*y.borrow()) < self.n);
        let sub = uint_submod(&x.borrow().0, &y.borrow().0, &self.n, self.k);
        debug_assert!(Uint::from(MInt(sub)) < self.n);
        MInt(sub)
    }

    fn redc(&self, x: &[u64; 2 * MINT_WORDS]) -> MInt {
        debug_assert!(Uint::from_digits(*x) < (self.n << (64 * self.k)));
        // Montgomery reduction (x/R mod n).
        // compute -x/N mod R
        let mul = uint_lowmul(&x[..], &self.ninv.0, self.k);
        // multiply by N again
        let mut m = uint_mul(&mul, self.n.digits(), self.k);
        // now x+mul*N is divisible by M
        // it is enough to add the high words
        // the carry is always equal to 1 iff x_low != 0
        let k = self.k as usize;
        if x[..k].iter().any(|&w| w != 0) {
            for i in k..2 * k {
                if m[i] == !0 {
                    m[i] = 0;
                } else {
                    m[i] += 1;
                    break;
                }
            }
        }
        let res = MInt(uint_addmod(&m[k..], &x[k..], &self.n, self.k));
        debug_assert!(Uint::from(res) < self.n);
        res
    }
}

fn uint_addmod(x: &[u64], y: &[u64], n: &Uint, sz: u32) -> [u64; MINT_WORDS] {
    // Add 2 integers spanning at most sz words.
    let nd = n.digits();
    let sz = sz as usize;
    // Store x+y
    let mut z = [0_u64; MINT_WORDS];
    // Store x+y-n
    let mut z2 = [0_u64; MINT_WORDS];
    let mut carry = 0_u64;
    let mut subcarry = 1_u64;
    for i in 0..sz {
        unsafe {
            let xi = *x.get_unchecked(i) as u128;
            let yi = *y.get_unchecked(i) as u128;
            let ni = *nd.get_unchecked(i);
            let sum = xi + yi;
            let zw = sum + (carry as u128);
            *z.get_unchecked_mut(i) = zw as u64;
            carry = (zw >> 64) as u64;
            // Add 1<<64w - n = !n + 1 to subtract n
            let z2w = sum + ((!ni) as u128) + (subcarry as u128);
            *z2.get_unchecked_mut(i) = z2w as u64;
            subcarry = (z2w >> 64) as u64;
        }
    }
    if subcarry == 0 {
        // x+y-n < 0
        assert!(carry == 0);
        debug_assert!(
            Uint::from(MInt(z.try_into().unwrap()))
                == Uint::from(MInt(x[..MINT_WORDS].try_into().unwrap()))
                    + Uint::from(MInt(y[..MINT_WORDS].try_into().unwrap()))
        );
        z
    } else {
        debug_assert!(
            Uint::from(MInt(z2.try_into().unwrap()))
                == Uint::from(MInt(x[..MINT_WORDS].try_into().unwrap()))
                    + Uint::from(MInt(y[..MINT_WORDS].try_into().unwrap()))
                    - n
        );
        z2
    }
}

fn uint_submod(
    x: &[u64; MINT_WORDS],
    y: &[u64; MINT_WORDS],
    n: &Uint,
    sz: u32,
) -> [u64; MINT_WORDS] {
    // Subtract 2 integers spanning at most sz words.
    let nd = n.digits();
    let sz = sz as usize;
    // Store x-y (actually x + NOT y + 1)
    let mut z = [0_u64; MINT_WORDS];
    // Store x-y+n (actually x + NOT y + n + 1)
    let mut z2 = [0_u64; MINT_WORDS];
    let mut carry1 = 1_u64;
    let mut carry2 = 1_u64;
    for i in 0..sz {
        unsafe {
            let xi = *x.get_unchecked(i) as u128;
            let yi = *y.get_unchecked(i);
            let ni = *nd.get_unchecked(i) as u128;
            let sum = xi + ((!yi) as u128);
            let zw = sum + (carry1 as u128);
            *z.get_unchecked_mut(i) = zw as u64;
            carry1 = (zw >> 64) as u64;
            let z2w = sum + ni + (carry2 as u128);
            *z2.get_unchecked_mut(i) = z2w as u64;
            carry2 = (z2w >> 64) as u64;
        }
    }
    if carry1 == 0 {
        // x-y < 0
        debug_assert!(Uint::from(MInt(z2)) == Uint::from(MInt(*x)) + n - Uint::from(MInt(*y)));
        z2
    } else {
        debug_assert!(Uint::from(MInt(z)) == Uint::from(MInt(*x)) - Uint::from(MInt(*y)));
        z
    }
}

fn uint_lowmul(x: &[u64], y: &[u64; MINT_WORDS], sz: u32) -> [u64; MINT_WORDS] {
    // Lower words of product.
    let sz = sz as usize;
    debug_assert!(sz <= MINT_WORDS);
    let mut z = [0_u64; MINT_WORDS];
    for i in 0..sz {
        let mut carry = 0_u64;
        let xi = unsafe { *x.get_unchecked(i) };
        if xi == 0 {
            continue;
        }
        for j in 0..sz - i {
            unsafe {
                let xi = xi as u128;
                let yj = *y.get_unchecked(j) as u128;
                let xy = xi * yj + (carry as u128);
                let zlo = xy as u64;
                let zhi = (xy >> 64) as u64;
                let zij = z[..].get_unchecked_mut(i + j);
                let (zlo2, c) = zij.overflowing_add(zlo);
                *zij = zlo2;
                carry = zhi + (if c { 1 } else { 0 });
            }
        }
    }
    z
}

fn uint_mul(x: &[u64], y: &[u64], sz: u32) -> [u64; 2 * MINT_WORDS] {
    let sz = sz as usize;
    debug_assert!(sz <= MINT_WORDS);
    let mut z = [0_u64; 2 * MINT_WORDS];
    for i in 0..sz {
        let mut carry = 0_u64;
        let xi = unsafe { *x.get_unchecked(i) };
        if xi == 0 {
            continue;
        }
        for j in 0..sz {
            unsafe {
                let xi = xi as u128;
                let yj = *y.get_unchecked(j) as u128;
                let xy = xi * yj + (carry as u128);
                let zlo = xy as u64;
                let zhi = (xy >> 64) as u64;
                let zij = z[..].get_unchecked_mut(i + j);
                let (zlo2, c) = zij.overflowing_add(zlo);
                *zij = zlo2;
                carry = zhi + (if c { 1 } else { 0 });
            }
        }
        let (zlo2, c) = z[i + sz].overflowing_add(carry);
        z[i + sz] = zlo2;
        if c && i + sz + 1 < z.len() {
            z[i + sz + 1] += 1;
        }
    }
    z
}

#[test]
fn test_montgomery() {
    use crate::arith::Num;
    use std::str::FromStr;

    let n = Uint::from_str("2953951639731214343967989360202131868064542471002037986749").unwrap();
    let p = Uint::from_str("17917317351877").unwrap();
    let pinv = Uint::from_str("42403041586861144438126400473690086613066961901031711489").unwrap();
    let zn = ZmodN::new(n);
    let x = zn.from_int(p);
    let y = zn.from_int(pinv);
    let one = zn.from_int(Uint::ONE);
    assert_eq!(zn.to_int(x), p);
    assert_eq!(zn.to_int(y), pinv);
    assert_eq!(zn.to_int(one), Uint::ONE);
    assert_eq!(zn.mul(x, y), one);
    assert_eq!(zn.inv(x), Some(y));
    assert_eq!(zn.inv(y), Some(x));

    // n = 107910248100432407082438802565921895527548119627537727229429245116458288637047
    // n is very close to 2^256
    // 551/901 mod n = 38924340324795263376018435997696577188072296203051899389083800957656985357426
    let n = Uint::from_str(
        "107910248100432407082438802565921895527548119627537727229429245116458288637047",
    )
    .unwrap();
    let zn = ZmodN::new(n);
    let x = zn.from_int(Uint::from(551_u64));
    assert_eq!(zn.to_int(x).to_u64(), Some(551));
    let y = zn.from_int(Uint::from(901_u64));
    assert_eq!(zn.to_int(y).to_u64(), Some(901));
    assert_eq!(
        zn.to_int(zn.inv(y).unwrap()),
        Uint::from_str(
            "84675411106554619097984720770373784836821887432485208824868453160195349685230"
        )
        .unwrap()
    );
    assert_eq!(zn.mul(y, zn.inv(y).unwrap()), zn.one());
    let x_y = zn.mul(x, zn.inv(y).unwrap());
    let expect = Uint::from_str(
        "38924340324795263376018435997696577188072296203051899389083800957656985357426",
    )
    .unwrap();
    assert_eq!(zn.to_int(x_y), expect);
}
