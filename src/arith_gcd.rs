// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Implementation of faster multiprecision GCD.
//!
//! As described in https://cr.yp.to/lineartime/multapps-20080515.pdf
//! the best asymptotic complexity is O(n (log n)^2+ε), however the
//! same idea can be used to reduce 32 bits at a time, which
//! is good enough for small multiword integers.

use std::cmp::max;

use bnum::cast::CastFrom;
use bnum::{BInt, BUint};
use num_integer::Integer;

use crate::arith::Num;

pub fn big_gcd<const N: usize>(n: &BUint<N>, p: &BUint<N>) -> BUint<N> {
    if p.is_zero() {
        return *n;
    }
    if n.is_zero() {
        return *p;
    }
    gcd_internal::<N, false>(n, p).0
}

/// Modular inverse of n modulo p.
///
/// Returns Ok(x) if x is a modular inverse, Err(gcd) if gcd > 1
pub fn inv_mod<const N: usize>(n: &BUint<N>, p: &BUint<N>) -> Result<BUint<N>, BUint<N>> {
    assert!(!p.is_zero());
    if n.is_zero() {
        return Err(*p);
    }
    let (d, u, _) = gcd_internal::<N, true>(n, p);
    if d != BUint::ONE {
        return Err(d);
    }
    if u.is_negative() {
        Ok(p - u.abs().to_bits() % p)
    } else {
        Ok(u.to_bits() % p)
    }
}

/// Fast extended GCD.
///
/// The extended GCD is determined by a lattice reduction algorithm:
/// the result is an invertible integer matrix M (det M = ±1)
/// such that M(n, p) = (gcd, 0)
///
/// To reduce the number of operations, the matrix is determined
/// by computing partial reductions using 64-bit arithmetic.
/// This hopefully requires N/32 multiword products instead of N.
pub fn gcd_internal<const N: usize, const EXT: bool>(
    n: &BUint<N>,
    p: &BUint<N>,
) -> (BUint<N>, BInt<N>, BInt<N>) {
    // A matrix such that x = biga*n + bigb*p, y = bigc*n + bigd*p
    let (mut biga, mut bigb) = (BInt::<N>::ONE, BInt::<N>::ZERO);
    let (mut bigc, mut bigd) = (bigb, biga);
    // Faster than generic Euclid algorithm.
    // Invariants:
    // x >= y
    // (x,y) generate the same ideal as original (n,p).
    let mut x = *n;
    let mut y = *p;
    // Now gcd(x,y) is odd: x,y can never be simultaneously even.
    loop {
        //eprintln!("x={x}");
        //eprintln!("y={y}");
        // Make sure x > y.
        if y >= x {
            (biga, bigb, bigc, bigd) = (bigc, bigd, biga, bigb);
            (x, y) = (y, x)
        }
        let lx = x.bits();
        let ly = y.bits();
        if lx == 0 {
            return (y, bigc, bigd);
        }
        if ly == 0 {
            return (x, biga, bigb);
        }
        if lx < 64 && ly < 64 {
            let (x0, y0) = (x.digits()[0] as i64, y.digits()[0] as i64);
            if EXT {
                // Extended gcd.
                let e = Integer::extended_gcd(&x0, &y0);
                let u: BInt<N> = BInt::from(e.x) * biga + BInt::from(e.y) * bigc;
                let v: BInt<N> = BInt::from(e.x) * bigb + BInt::from(e.y) * bigd;
                return (BUint::from_digit(e.gcd as u64), u, v);
            } else {
                let d = Integer::gcd(&x0, &y0) as u64;
                return (BUint::from_digit(d), BInt::ZERO, BInt::ZERO);
            }
        }
        // To reduce by 32 bits we need:
        // a matrix M such that M is less than 32 bits
        // M (xtop, ytop) is less than 32 bits.
        let bits = max(lx, ly);
        // xtop and ytop have sizes differing by < 32 bits
        let xtop = top64(x.digits(), bits);
        let ytop = top64(y.digits(), bits);
        // If ytop is too small, the reduction matrix cannot be small.
        // If x is too large we will have trouble multiplying.
        // Use multiprecision quotient.
        if lx + 36 >= N as u32 * 64 || ly + 36 >= N as u32 * 64 || ytop < 1 << 32 {
            if EXT {
                let q = x / y;
                let r = x - q * y;
                let (c, d) = if (r << 1) > y {
                    // Use (q+1)y - x for smaller y.
                    (x, y) = (y, y - r);
                    let q = BInt::<N>::cast_from(q) + BInt::<N>::ONE;
                    (q * bigc - biga, q * bigd - bigb)
                } else {
                    let q = BInt::<N>::cast_from(q);
                    (x, y) = (y, r);
                    (biga - q * bigc, bigb - q * bigd)
                };
                (biga, bigb) = (bigc, bigd);
                (bigc, bigd) = (c, d);
            } else {
                (x, y) = (y, x % y);
            }
            continue;
        }
        // Now xtop and ytop have similar sizes.
        // Reduce vector: invariant u=ax+by, v=cx+dy
        let (a, b, c, d) = reduce64(xtop, ytop);
        // By definition, ab, cd have opposite signs.
        // Also a,b,c,d are guaranteed to be less than 40 bits
        // so that the multiplication does not overflow.
        let size = (bits as usize + 63) / 64;
        let (ax_by, negx) = dot_product(size, a, &x, b, &y);
        let (cx_dy, negy) = dot_product(size, c, &x, d, &y);
        // x, y are now smaller.
        (x, y) = (ax_by, cx_dy);
        if EXT {
            let (ai, bi) = (BInt::<N>::from(a), BInt::<N>::from(b));
            let (ci, di) = (BInt::<N>::from(c), BInt::<N>::from(d));
            let (aa, bb) = (ai * biga + bi * bigc, ai * bigb + bi * bigd);
            let (cc, dd) = (ci * biga + di * bigc, ci * bigb + di * bigd);
            (biga, bigb, bigc, bigd) = (aa, bb, cc, dd);
            if negx {
                (biga, bigb) = (-biga, -bigb);
            }
            if negy {
                (bigc, bigd) = (-bigc, -bigd);
            }
        }
    }
}

/// Extract bits from indices (bits-64..bits)
fn top64(digs: &[u64], bits: u32) -> u64 {
    let w = bits as usize / 64;
    debug_assert!(bits >= 64);
    if bits % 64 == 0 {
        return digs[w - 1];
    }
    let xtop1 = digs[w - 1];
    let xtop2 = digs[w];
    (xtop2 << (64 - bits % 64)) | (xtop1 >> (bits % 64))
}

fn mulword<const N: usize>(w: u64, sz: usize, n: &BUint<N>) -> BUint<N> {
    let mut nd = *n.digits();
    let mut carry = 0;
    for i in 0..sz {
        let nw = nd[i] as u128 * w as u128 + carry as u128;
        nd[i] = nw as u64;
        carry = (nw >> 64) as u64;
    }
    if carry > 0 {
        nd[sz] = carry;
    }
    BUint::from_digits(nd)
}

/// Compute abs(ax + by) and whether the sign was inverted.
fn dot_product<const N: usize>(
    sz: usize,
    a: i64,
    x: &BUint<N>,
    b: i64,
    y: &BUint<N>,
) -> (BUint<N>, bool) {
    let (au, bu) = (a.unsigned_abs(), b.unsigned_abs());
    if a.signum() * b.signum() < 0 {
        let ax = mulword(au, sz, x);
        let by = mulword(bu, sz, y);
        let neg = (ax > by && a < 0) || (ax < by && b < 0);
        (ax.abs_diff(by), neg)
    } else {
        (mulword(au, sz, x) + mulword(bu, sz, y), a < 0 || b < 0)
    }
}

/// Determine a matrix of 64-bit signed integers such that (ax+by, cx+dy)
/// are "small" (less than (x,y) and preferably less than 32 bits),
/// but (a,b,c,d) are also small, guaranteed to be less than 36 bits.
///
/// In the worst case, (ax+by, cx+dy) correspond to a single
/// iteration of Euclid algorithm (y, x % y) if x >= y.
/// The reduction is performed using Gauss reduction.
fn reduce64(x: u64, y: u64) -> (i64, i64, i64, i64) {
    // Reduce vector: invariant u=ax+by, v=cx+dy
    let (mut a, mut b, mut c, mut d) = (1_i64, 0_i64, 0_i64, 1_i64);
    let (mut u, mut v) = (x, y);
    // Loop until (u,v) are small enough.
    while u >> 24 > 0 && v >> 24 > 0 {
        if u < v {
            (a, b, c, d, u, v) = (c, d, a, b, v, u);
        } else {
            let (q, r) = ((u / v) as i64, u % v);
            // But stop if the matrix is too large
            // (cannot happen at first iteration since y >= 1<<32)
            if ((q + 1) as u64).bits() + (max(c.abs(), d.abs()) as u64).bits() > 36 {
                break;
            }
            // Ensure that r < |v/2| to reduce matrix size and speed up iteration.
            if r > v / 2 {
                (a, b, c, d, u, v) = (c, d, (q + 1) * c - a, (q + 1) * d - b, v, v - r);
            } else {
                (a, b, c, d, u, v) = (c, d, a - q * c, b - q * d, v, r);
            }
        }
        debug_assert!(a as i128 * x as i128 + b as i128 * y as i128 == u as i128);
        debug_assert!(c as i128 * x as i128 + d as i128 * y as i128 == v as i128);
    }
    debug_assert!(a.abs() <= 1 << 36 && b.abs() <= 1 << 36);
    debug_assert!(c.abs() <= 1 << 36 && d.abs() <= 1 << 36);
    (a, b, c, d)
}

#[test]
fn test_gcd() {
    use bnum::types::{I2048, U1024};
    use rand::{self, Rng};

    let mut rng = rand::thread_rng();
    for k in 1..1000u64 {
        let mut x = [0; 16];
        rng.try_fill(&mut x[..15]).unwrap();
        let a = U1024::from(k * k + 1) * U1024::from_digits(x);
        // b can be shorter.
        rng.try_fill(&mut x[..15]).unwrap();
        x[k as usize % 8 + 4..].fill(0);
        let b = U1024::from(k * k + 1) * U1024::from_digits(x);
        assert_eq!(a.gcd(&b), big_gcd(&a, &b));

        // Check Bézout identity.
        let (d, u, v) = gcd_internal::<16, true>(&a, &b);
        assert_eq!(
            I2048::cast_from(d),
            I2048::cast_from(u) * I2048::cast_from(a) + I2048::cast_from(v) * I2048::cast_from(b)
        );
    }
}

#[test]
fn test_invmod() {
    use bnum::types::U1024;
    use std::str::FromStr;

    let n = U1024::from_str("2953951639731214343967989360202131868064542471002037986749").unwrap();
    for k in 1..100u64 {
        let k = U1024::from(k);
        let kinv = inv_mod(&k, &n).unwrap();
        assert_eq!((kinv * k) % n, U1024::ONE);
    }

    // Observed regressions
    let n = U1024::from_str("26984400680641981219").unwrap();
    let (a, b) = (73427761, 30894741361);
    let binv = inv_mod(&U1024::from_digit(b), &n).unwrap();
    assert_eq!(binv, U1024::from_str("18748949926630253258").unwrap());
    assert_eq!(
        (U1024::from_digit(a) * binv) % n,
        U1024::from_str("22160499496729207058").unwrap()
    );
}
