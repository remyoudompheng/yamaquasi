// Copyright 2022, 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Montgomery form arithmetic for 64-bit moduli.
//!
//! Reference:
//! Peter L. Montgomery, Modular Multiplication Without Trial Division
//! <https://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777282-X/S0025-5718-1985-0777282-X.pdf>

use std::borrow::Borrow;

use bnum::cast::CastFrom;
use bnum::BUint;

use crate::arith;
use crate::arith_gcd;
use crate::Uint;

/// Returns ninv such that n*ninv = -1
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

/// Montgomery reduction (x/R mod n), only if x < nR
#[inline(always)]
pub fn mg_redc(n: u64, ninv: u64, x: u128) -> u64 {
    if x as u64 == 0 {
        return (x >> 64) as u64;
    }
    // compute -x/N mod R
    let mul: u64 = (x as u64).wrapping_mul(ninv);
    // reduce
    let m = mul as u128 * n as u128;
    debug_assert!((x as u64).wrapping_add(m as u64) == 0);
    // The low words always produce a carry.
    // The high words are guaranteed to be < n.
    // We want xhi+mhi+1 mod n.
    let (xhi, mhi) = ((x >> 64) as u64, (m >> 64) as u64);
    let d = n - mhi - 1;
    if xhi >= d {
        xhi - d
    } else {
        xhi + mhi + 1
    }
}

/// Inverse of Montgomery representation:
/// input is xR mod p, output is R/x mod p
pub fn mg_inv(n: u64, ninv: u64, r2: u64, x: u64) -> Option<u64> {
    // mm = m/R
    let mm = mg_redc(n, ninv, x as u128);
    let mminv = arith::inv_mod64(mm, n)?;
    // minv = R/mm = R^2/m
    Some(mg_mul(n, ninv, mminv, r2))
}

/// Context for modular Montgomery arithmetic for large (512-bit) moduli
#[derive(Clone)]
pub struct ZmodN {
    pub n: Uint,
    // Minus n^-1 mod 2^64
    ninv64: u64,
    // Auxiliary base R=2^64k
    k: u32,
    // R mod n
    r: MInt,
    // R^2 mod n
    r2: MInt,
}

// Store values as 512-bit integers.
pub(crate) const MINT_WORDS: usize = 8;

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
    pub(crate) fn from_uint(n: Uint) -> Self {
        let mut m = MInt::default();
        m.0[..].copy_from_slice(&n.digits()[..MINT_WORDS]);
        m
    }
}

impl ZmodN {
    pub fn new(n: Uint) -> Self {
        assert!(n.bit(0));
        assert!(n.bits() <= 64 * MINT_WORDS as u32);
        let k = (n.bits() + 63) / 64;
        assert!(n.bits() <= 64 * k);
        let rsqrt = Uint::ONE << (32 * k);
        let r = (rsqrt * rsqrt) % n;
        let r2 = (r * r) % n;
        let ninv64 = {
            // Invariant: nx = 1 + 2^k s, k increasing
            let mut x = 1_u64;
            let n64 = n.digits()[0];
            loop {
                let rem = n64.wrapping_mul(x) - 1;
                if rem == 0 {
                    break;
                }
                x += 1 << rem.trailing_zeros();
            }
            assert!(n64.wrapping_mul(x) == 1);
            // Now compute R-x
            !x + 1
        };
        ZmodN {
            n,
            ninv64,
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

    pub fn words(&self) -> usize {
        self.k as usize
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
        let mut m = MInt::default();
        mint_mulmod(&self, &mut m.0, &x.borrow().0, &y.borrow().0, self.k);
        // FIXME: remove
        let nd = self.n.digits();
        if !mint_lt(&m.0, nd, self.k) {
            mint_sub(&mut m.0, nd, self.k);
        }
        debug_assert!(Uint::from(m) < self.n);
        m
    }

    pub fn inv(&self, x: MInt) -> Option<MInt> {
        // No optimization, use ordinary modular inversion.
        // Input is xR we want R/x
        let i = arith_gcd::inv_mod(&BUint::from_digits(x.0), &BUint::cast_from(self.n)).ok()?;
        // Multiply by R twice
        let i = self.from_int(Uint::cast_from(i));
        Some(self.mul(&i, &self.r2))
    }

    pub fn gcd(&self, x: &MInt) -> Uint {
        // No optimization, use ordinary modular inversion.
        Uint::cast_from(arith_gcd::big_gcd(
            &BUint::cast_from(self.n),
            &BUint::from_digits(x.0),
        ))
    }

    pub fn add<M: Borrow<MInt>>(&self, x: M, y: M) -> MInt {
        debug_assert!(Uint::from(*x.borrow()) < self.n);
        debug_assert!(Uint::from(*y.borrow()) < self.n);
        let mut m = *x.borrow();
        mint_add(&mut m.0, &y.borrow().0, self.k);
        if !mint_lt(&m.0, self.n.digits(), self.k) {
            mint_sub(&mut m.0, self.n.digits(), self.k);
        }
        debug_assert!(Uint::from(m) < self.n);
        m
    }

    pub fn sub<M: Borrow<MInt>>(&self, x: M, y: M) -> MInt {
        debug_assert!(Uint::from(*x.borrow()) < self.n);
        debug_assert!(Uint::from(*y.borrow()) < self.n);
        let yp = &y.borrow().0;
        let mut s = *x.borrow();
        if mint_lt(&s.0, yp, self.k) {
            mint_add(&mut s.0, self.n.digits(), self.k)
        };
        mint_sub(&mut s.0, yp, self.k);
        debug_assert!(Uint::from(s) < self.n);
        s
    }

    /// Multiprecision Montgomery reduction (x/R mod n).
    #[doc(hidden)]
    pub fn redc(&self, x: &[u64; 2 * MINT_WORDS]) -> MInt {
        debug_assert!(Uint::from_digits(*x) < (self.n << (64 * self.k)));
        // This is N times a division by 2^64.
        // Repeatedly add N * (-x[0]/N mod 2^64) and divide by 2^64
        let mut m = x.clone();
        let n = self.n.digits();
        let sz = self.k as usize;
        let ninv64 = self.ninv64;
        for i in 0..sz {
            let m_ninv = m[i].wrapping_mul(ninv64);
            let mut carryn = 0;
            for j in 0..sz {
                unsafe {
                    let nj = *n.get_unchecked(j) as u128;
                    let mij = m[..].get_unchecked_mut(i + j);
                    let mut mn = (m_ninv as u128) * nj;
                    mn += *mij as u128;
                    mn += carryn as u128;
                    *mij = mn as u64;
                    carryn = (mn >> 64) as u64;
                }
            }
            let (mi, c) = m[i + sz].overflowing_add(carryn);
            m[i + sz] = mi;
            if c {
                assert!(i + sz + 1 < m.len());
                // FIXME: overflow
                m[i + sz + 1] += u64::from(c);
            }
        }
        let mut m: [u64; MINT_WORDS] = m[sz..sz + MINT_WORDS].try_into().unwrap();
        if !mint_lt(&m, self.n.digits(), self.k) {
            mint_sub(&mut m, self.n.digits(), self.k);
        }
        MInt(m)
    }

    /// Like redc but for numbers possibly exceeding n<<64k
    /// It is assumed that x.len() <= 3 * MINT_WORDS (24)
    /// which is enough to accomodate FFT convolution outputs.
    pub fn redc_large(&self, x: &[u64]) -> MInt {
        debug_assert!(x.len() < 3 * MINT_WORDS);
        let mut xlo = [0_u64; 2 * MINT_WORDS];
        let mut xhi = [0_u64; 2 * MINT_WORDS];
        let k = self.k as usize;
        // Divide upper part by R.
        for i in 0..k {
            xlo[i] = x[i]
        }
        for i in k..x.len() {
            xhi[i - k] = x[i];
        }
        let m = self.redc(&xlo);
        let mhi = self.redc(&xhi);
        self.add(m, self.mul(mhi, self.r2))
    }
}

// Returns whether x < n, where x is stored on at most sz+1 words.
fn mint_lt(x: &[u64], n: &[u64], sz: u32) -> bool {
    let sz = sz as usize;
    debug_assert!((sz + 1..x.len()).all(|idx| x[idx] == 0));
    debug_assert!((sz..n.len()).all(|idx| n[idx] == 0));
    if x.len() > sz && x[sz] > 0 {
        return false;
    }
    for i in 0..sz {
        let idx = sz - 1 - i;
        let (xi, ni) = unsafe { (*x.get_unchecked(idx), *n.get_unchecked(idx)) };
        if xi == ni {
            continue;
        }
        return xi < ni;
    }
    return false;
}

// Combined multiplication and multiprecision REDC.
// <https://www.microsoft.com/en-us/research/wp-content/uploads/1996/01/j37acmon.pdf>
fn mint_mulmod(zn: &ZmodN, res: &mut [u64], x: &[u64], y: &[u64], sz: u32) {
    match sz {
        // Dispatch to unrolled implementations
        1 => _mint_mulmod::<1>(zn, res, x, y),
        2 => _mint_mulmod::<2>(zn, res, x, y),
        3 => _mint_mulmod::<3>(zn, res, x, y),
        4 => _mint_mulmod::<4>(zn, res, x, y),
        5 => _mint_mulmod::<5>(zn, res, x, y),
        6 => _mint_mulmod::<6>(zn, res, x, y),
        7 => _mint_mulmod::<7>(zn, res, x, y),
        8 => _mint_mulmod::<8>(zn, res, x, y),
        _ => unreachable!("impossible"),
    }
}

fn _mint_mulmod<const SIZE: usize>(zn: &ZmodN, res: &mut [u64], x: &[u64], y: &[u64]) {
    // SIZE times, multiply by a word of y,
    // multiply by lower*ninv64*N to cancel 1 word.
    let ninv64 = zn.ninv64;
    let n = &zn.n.digits();
    debug_assert!(SIZE <= MINT_WORDS);
    let mut z = [0_u64; 2 * MINT_WORDS];
    let mut overflow = false;
    for i in 0..SIZE {
        // Accumulate x[i] * y in z.
        let mut carry = 0_u64;
        let xi = unsafe { *x.get_unchecked(i) };
        for j in 0..SIZE {
            unsafe {
                let xi = xi as u128;
                let yj = *y.get_unchecked(j) as u128;
                let zij = z[..].get_unchecked_mut(i + j);
                let mut xy = xi * yj;
                xy += *zij as u128;
                xy += carry as u128;
                *zij = xy as u64;
                carry = (xy >> 64) as u64;
            }
        }
        // Add m * N to cancel the lower word.
        let m = z[i].wrapping_mul(ninv64);
        let mut carryn = 0;
        for j in 0..SIZE {
            unsafe {
                let nj = *n.get_unchecked(j) as u128;
                let zij = z[..].get_unchecked_mut(i + j);
                let mut xy = (m as u128) * nj;
                xy += *zij as u128;
                xy += carryn as u128;
                *zij = xy as u64;
                carryn = (xy >> 64) as u64;
            }
        }
        debug_assert!(z[i] == 0);
        let (zi, c1) = z[i + SIZE].overflowing_add(carry);
        let (zin, c2) = zi.overflowing_add(carryn);
        z[i + SIZE] = zin;
        if c1 || c2 {
            if i + 1 < SIZE {
                z[i + SIZE + 1] = u64::from(c1) + u64::from(c2);
            } else {
                overflow = true;
            }
        }
    }
    for i in 0..SIZE {
        res[i] = z[i + SIZE]
    }
    if overflow {
        // Add 2^64W - n = not(n)+1
        let mut carry = 1;
        for i in 0..SIZE {
            let s = res[i] as u128 + !n[i] as u128 + carry as u128;
            res[i] = s as u64;
            carry = (s >> 64) as u64;
        }
        if carry > 0 {
            // FIXME: can it happen?
            res[SIZE] = 1
        }
    }
}

fn mint_add(x: &mut [u64], y: &[u64], sz: u32) {
    // Add 2 integers spanning at most sz words.
    let sz = sz as usize;
    let mut carry = 0_u64;
    for i in 0..sz {
        unsafe {
            let xi = x.get_unchecked_mut(i);
            let yi = *y.get_unchecked(i) as u128;
            let sum = (*xi as u128) + yi;
            let zw = sum + (carry as u128);
            *xi = zw as u64;
            carry = (zw >> 64) as u64;
        }
    }
    if sz < x.len() {
        x[sz] += carry;
    } else {
        debug_assert!(carry == 0);
    }
}

/// Subtract 2 integers x > y spanning at most sz words.
fn mint_sub(x: &mut [u64; MINT_WORDS], y: &[u64], sz: u32) {
    debug_assert!(Uint::from(MInt(*x)) >= Uint::from(MInt(y[..MINT_WORDS].try_into().unwrap())));
    // Store x-y (actually x + NOT y + 1)
    // The carry must propagate until the end unless x[sz]==0.
    let sz = sz as usize;
    let mut carry = 1_u64;
    for i in 0..sz {
        unsafe {
            let xi = x.get_unchecked_mut(i);
            let yi = *y.get_unchecked(i);
            let sum = (*xi as u128) + ((!yi) as u128);
            let zw = sum + (carry as u128);
            *xi = zw as u64;
            carry = (zw >> 64) as u64;
        }
    }
    if sz < MINT_WORDS {
        debug_assert!(x[sz] == 1 - carry);
        x[sz] = 0;
    } else {
        debug_assert!(carry == 1);
    }
}

/// Determine factors from a sequence of values mod n
///
/// The sequence of values is supposed to have increasing GCD with n,
/// i.e. for i <= j, gcd(n, vals[i]) divides gcd(n, vals[j]).
///
/// The product of returned factors is gcd(n, last)/gcd(n, first).
///
/// This is satisfied if vals represents iterated powers of
/// a group element (ECM and P±1 stage 1) or a cumulative product
/// (ECM and P±1 stage 2).
pub fn gcd_factors(n: &Uint, vals: &[MInt]) -> (Vec<Uint>, Uint) {
    // Recursively split input slice and check whether
    // gcd(first, n) = gcd(last, n).
    // If they differ by a single prime factor, stop recursion.
    // gcd1 = gcd(n, vals[0])
    // gcd2 = gcd(n, vals.last())
    fn find_factors(n: &Uint, vals: &[MInt], gcd1: &Uint, gcd2: &Uint, factors: &mut Vec<Uint>) {
        if gcd1 == gcd2 {
            return;
        }
        let p = gcd2 / gcd1;
        debug_assert!(gcd2 > gcd1 && *gcd2 == p * gcd1);
        if crate::pseudoprime(p) || vals.len() <= 2 {
            factors.push(p);
            return;
        }
        let mid = vals.len() / 2;
        let d = arith_gcd::big_gcd(n, &Uint::from(vals[mid]));
        find_factors(n, &vals[..=mid], gcd1, &d, factors);
        find_factors(n, &vals[mid..], &d, gcd2, factors);
    }
    let mut facs = vec![];
    let d1 = arith_gcd::big_gcd(n, &Uint::from(vals[0]));
    let d2 = arith_gcd::big_gcd(n, &Uint::from(vals[vals.len() - 1]));
    find_factors(n, vals, &d1, &d2, &mut facs);
    let mut n = *n;
    for f in &facs {
        n /= f;
    }
    (facs, n)
}

#[test]
fn test_montgomery64() {
    // Check that 64-bit moduli do not trigger overflow.
    let n = 0xeb67d1ff62bd9f49;
    let ninv = mg_2adic_inv(n);
    let r = ((1_u128 << 64) % (n as u128)) as u64;
    let r2 = ((r as u128 * r as u128) % (n as u128)) as u64;
    let x = mg_mul(n, ninv, 12345, r2);
    let y = mg_mul(n, ninv, 56789, r2);
    let xy = mg_mul(n, ninv, 12345 * 56789, r2);
    assert_eq!(mg_mul(n, ninv, x, y), xy);
}

#[test]
fn test_montgomery() {
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
    assert_eq!(zn.to_int(x).try_into().ok(), Some(551_u64));
    let y = zn.from_int(Uint::from(901_u64));
    assert_eq!(zn.to_int(y).try_into().ok(), Some(901_u64));
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

#[test]
fn test_gcd_factors() {
    use num_integer::Integer;

    let n = Uint::from_digit(13 * 17 * 19 * 31 * 37);
    assert_eq!(Integer::gcd(&Uint::ZERO, &n), n);

    let vals64 = [
        79,
        77 * 13,
        7 * 13,
        9 * 13,
        97 * 13 * 37,
        5 * 13 * 37,
        13 * 17 * 37,
        13 * 17 * 37 * 8,
    ];
    let zn = ZmodN::new(n);
    let vals: Vec<MInt> = vals64
        .into_iter()
        .map(|x| zn.from_int(Uint::from_digit(x)))
        .collect();
    let (facs, nn) = gcd_factors(&n, &vals);
    eprintln!("factors={facs:?}");
    assert_eq!(
        facs,
        vec![
            Uint::from_digit(13),
            Uint::from_digit(37),
            Uint::from_digit(17)
        ]
    );
    assert_eq!(nn, Uint::from_digit(19 * 31));
}
