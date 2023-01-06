// Copyright 2022, 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
    ninv: Uint,
    // Auxiliary base R=2^64k
    k: u32,
    // R mod n
    r: Uint,
    // R^2 mod n
    r2: Uint,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct MInt(pub Uint);

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
        ZmodN { n, ninv, k, r, r2 }
    }

    pub fn zero(&self) -> MInt {
        MInt(Uint::ZERO)
    }

    pub fn one(&self) -> MInt {
        MInt(self.r)
    }

    pub fn from_int(&self, x: Uint) -> MInt {
        self.mul(MInt(x), MInt(self.r2))
    }

    pub fn to_int(&self, x: MInt) -> Uint {
        self.redc(x.0).0
    }

    pub fn mul(&self, x: MInt, y: MInt) -> MInt {
        // all bit lengths MUST be < 512
        debug_assert!(x.0 < self.n);
        debug_assert!(y.0 < self.n);
        self.redc(uint_mul(&x.0, &y.0, self.k))
    }

    pub fn inv(&self, x: MInt) -> Option<MInt> {
        // No optimization, use ordinary modular inversion.
        Some(self.from_int(arith::inv_mod(self.to_int(x), self.n)?))
    }

    pub fn add(&self, x: MInt, y: MInt) -> MInt {
        let mut sum = x.0 + y.0;
        while sum >= self.n {
            sum -= self.n;
        }
        MInt(sum)
    }

    pub fn sub(&self, x: MInt, y: MInt) -> MInt {
        debug_assert!(
            y.0.bits() < (x.0 + self.n).bits() + 2,
            "x.bits={} y.bits={} n.bits={}",
            x.0.bits(),
            y.0.bits(),
            self.n.bits()
        );
        let mut x = x.0;
        while y.0 > x {
            x += self.n;
        }
        debug_assert!(x - y.0 < self.n);
        MInt(x - y.0)
    }

    fn redc(&self, x: Uint) -> MInt {
        debug_assert!(x < (self.n << (64 * self.k)));
        // Montgomery reduction (x/R mod n).
        // compute -x/N mod R
        // Half precision is enough.
        let mul = uint_mul(&x, &self.ninv, self.k);
        // Manually clear upper words
        let mut mul_digits = mul.digits().clone();
        for i in self.k as usize..mul_digits.len() {
            mul_digits[i] = 0
        }
        let mul = Uint::from_digits(mul_digits);
        // reduce, mul <= R
        let m = uint_mul(&mul, &self.n, self.k);
        let x_plus_m = x + m;
        let xmd = x_plus_m.digits();
        // Shift right by 64k bits (x+m can overflow by one bit)
        let mut res = [0_u64; Uint::BITS as usize / 64];
        for i in 0..=self.k as usize {
            res[i] = xmd[i + self.k as usize];
        }
        let mut res = Uint::from_digits(res);
        if res >= self.n {
            res -= self.n
        }
        debug_assert!(res < self.n);
        MInt(res)
    }
}

fn uint_mul(x: &Uint, y: &Uint, sz: u32) -> Uint {
    // Desperate attempt to be faster than bnum multiplication.
    // We should definitely not be doing this but otherwise performance
    // is abysmal.
    debug_assert!(sz <= Uint::BITS / 64 / 2);
    let xd = x.digits();
    let yd = y.digits();
    let sz = sz as usize;
    let mut z = [0_u64; Uint::BITS as usize / 64];
    for i in 0..sz {
        let mut carry = 0_u64;
        let xi = unsafe { *xd.get_unchecked(i) };
        if xi == 0 {
            continue;
        }
        for j in 0..sz {
            unsafe {
                let xi = xi as u128;
                let yj = *yd.get_unchecked(j) as u128;
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
    Uint::from_digits(z)
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
