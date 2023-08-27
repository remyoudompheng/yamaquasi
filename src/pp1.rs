// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Implementation of Williams P+1 algorithm
//!
//! The P+1 algorithm is an algebraic group factoring method using
//! conics (quadratic twist of the multiplicative group, which has
//! order p+1 over GF(p)).
//!
//! The standard conic xy=1 maps to v^2 = u^2 - 4 through
//! (u,v)=(x+y,x-y) and computing using only the u-coordinate
//! will use the quadratic twist iff u^2-4 is not a square modulo p.
//!
//! The conic supports:
//! - doubling of u(2P) = u(P)^2 - 2
//! - differential addition u(P+Q) + u(P-Q) = u(P) u(Q)
//!
//! Scalar multiplication corresponds to Chebyshev polynomials.
//!
//! The usual recommendation (of GMP-ECM and the literature) is to try
//! 3 seeds to have a reasonably high probability of encounting the correct conic.
//!
//! # Performance of stage 1
//!
//! The standard method is to use Lucas chains to apply exponents.
//! This implementation nuses the non-optimal binary chains, instead
//! of the better approaches using continued fractions or the golden ratio.
//!
//! This requires about 2 multiplications per bit (2.9 B1) instead
//! of 1.2 per bit for P-1 (about 1.7 B1) for stage 1.
//!
//! # Performance of stage 2
//!
//! Stage 2 is much slower than P-1 due to using remainder trees instead
//! of the chirp-z transform (same speed as ECM).
//!
//! Since multiple iterations of P+1 are recommended, this means that the
//! current implementation offers little benefit compared to ECM.

use num_integer::Integer;

use crate::arith_montgomery::{gcd_factors, MInt, ZmodN};
use crate::arith_poly::Poly;
use crate::fbase;
use crate::params;
use crate::{Uint, Verbosity};

/// Run the P+1 algorithm with selected seed and smoothness bounds B1 and B2.
pub fn pp1(
    n: Uint,
    seed: u64,
    b1: u64,
    b2: f64,
    verbosity: Verbosity,
) -> Option<(Vec<Uint>, Uint)> {
    let mut factors = vec![];
    let start1 = std::time::Instant::now();
    let (b2real, d1, d2) = params::stage2_params(b2);
    if verbosity >= Verbosity::Info {
        eprintln!("Attempting P+1 with B1={b1} B2={b2real:e}");
    }
    assert!(b1 > 3);
    // The modulus/ring can shrink as we find factors.
    let mut nred = n;
    let mut zn = ZmodN::new(n);
    let mut sieve = fbase::PrimeSieve::new();
    let mut block = sieve.next();
    // Stage 1
    let mut g = zn.from_int(Uint::from(seed));
    let mut p_prev: u32 = 1;
    let mut gpows = vec![zn.one()];
    let two = zn.add(&zn.one(), &zn.one());
    loop {
        for &p in block {
            p_prev = p as u32;
            let p = p as u64;
            let mut pow = p;
            while pow * p < b1 {
                pow *= p;
            }
            // With binary chains, the cost is 2 BITS - 3
            // so it is better to NOT combine primes into large exponents.
            // About 1.7-1.8 multiplication per bit.
            if p > b1 {
                break;
            }
            g = chebyshev_modn(&zn, &g, pow);
            gpows.push(zn.sub(&g, &two));
            if g == zn.one() {
                // We can reach 1 if φ(n) is B1-smooth, no need to go further.
                break;
            }
        }
        // Check GCD after each prime block (several thousands primes)
        let logstage = Some(1).filter(|_| verbosity >= Verbosity::Verbose);
        if check_gcd_factors(&n, &mut factors, &mut nred, &mut gpows, logstage) {
            return if factors.is_empty() {
                None
            } else {
                Some((factors, nred))
            };
        }
        if zn.n != nred {
            // Shrink ring
            let gint = zn.to_int(g);
            zn = ZmodN::new(nred);
            g = zn.from_int(gint % nred);
        }
        if p_prev > b1 as u32 {
            break;
        }
        block = sieve.next();
    }
    drop(gpows);
    let elapsed1 = start1.elapsed();

    let start2 = std::time::Instant::now();
    let logtime = || {
        if verbosity >= Verbosity::Verbose {
            let elapsed2 = start2.elapsed();
            if elapsed2.as_secs_f64() < 0.01 {
                eprintln!(
                    "P+1 stage1={:.6}s stage2={:.6}s",
                    elapsed1.as_secs_f64(),
                    elapsed2.as_secs_f64()
                );
            } else {
                eprintln!(
                    "P+1 stage1={:.3}s stage2={:.3}s",
                    elapsed1.as_secs_f64(),
                    elapsed2.as_secs_f64()
                );
            }
        }
    };

    // Stage 2
    // To evaluate a Lucas sequence for exponents in an arithmetic progression
    // we need 1 multiplication by element using L(n+k) = L(n) L(k) - L(n-k)
    // We always use the FFT continuation for simplicity.
    //
    // It is enough to find L(i d1) == L(j) where gcd(d1,j) = 1.
    // By symmetry of g^k + g^(-k) is is enough to consider
    // j < d1/2
    assert!(d1 % 6 == 0);
    // Compute baby steps.
    let bsteps = {
        // L(-1) == L(1)
        let mut v = Vec::with_capacity(d1 as usize / 2);
        let g2 = chebyshev_modn(&zn, &g, 2);
        let mut b = g;
        let mut bprev = g;
        v.push(g.clone());
        let mut exp = 1;
        debug_assert!(b == chebyshev_modn(&zn, &g, exp));
        while exp + 2 < d1 / 2 {
            exp += 2;
            (bprev, b) = (b, zn.sub(&zn.mul(&b, &g2), &bprev));
            if exp % 3 != 0 && Integer::gcd(&exp, &d1) == 1 {
                v.push(b);
            }
        }
        debug_assert!(b == chebyshev_modn(&zn, &g, exp));
        v
    };
    // Compute the giant steps i*d1 for i in 2..2+d2
    let d2 = d2 as usize;
    let gsteps = {
        let mut steps = Vec::with_capacity(d2);
        let mut dgprev = two;
        let mut dg = chebyshev_modn(&zn, &g, d1);
        let step = dg;
        steps.push(two);
        steps.push(dg);
        for _ in 2..d2 {
            let dgnext = zn.sub(&zn.mul(&dg, &step), &dgprev);
            steps.push(dgnext);
            (dgprev, dg) = (dg, dgnext);
        }
        steps
    };
    debug_assert!(gsteps.len() == d2);

    let vals = Poly::roots_eval(&zn, &gsteps, &bsteps);
    // Compute cumulative product
    let mut prods = Vec::with_capacity(vals.len() + 1);
    prods.push(zn.one());
    for v in vals {
        prods.push(zn.mul(prods.last().unwrap(), &v));
    }
    let logstage = Some(2).filter(|_| verbosity >= Verbosity::Verbose);
    check_gcd_factors(&n, &mut factors, &mut nred, &mut prods, logstage);
    logtime();
    if factors.is_empty() {
        None
    } else {
        Some((factors, nred))
    }
}

/// Extract factors by computing GCD and update arrays.
/// Values array will be truncated to keep only the last element.
fn check_gcd_factors(
    n: &Uint,
    factors: &mut Vec<Uint>,
    nred: &mut Uint,
    values: &mut Vec<MInt>,
    stage: Option<usize>,
) -> bool {
    let (mut fs, nred_) = gcd_factors(nred, &values[..]);
    if fs.contains(n) {
        return true;
    }
    if !fs.is_empty() {
        if let Some(stage) = stage {
            for &f in &fs {
                eprintln!("Found factor {f} during stage {stage}");
            }
        };
        factors.append(&mut fs);
        *nred = nred_;
        if *nred == Uint::ONE || crate::pseudoprime(*nred) {
            return true;
        }
    }
    let last = values[values.len() - 1];
    values.clear();
    values.push(last);
    false
}

fn chebyshev_modn(zn: &ZmodN, g: &MInt, exp: u64) -> MInt {
    if exp == 0 {
        return zn.one();
    }
    // Use the simple binary Lucas chain [Montgomery]
    // 2n P is obtained by doubling nP
    // 2n+1 P is obtained by adding nP and n+1 P
    // This requires 2 modular multiplications per exponent bit.
    // TODO: use a better method.

    // Compute k P and k+1 P where k = n >> (length - i).
    // u(0P) = 2, u(1P) = g.
    let two = zn.add(&zn.one(), &zn.one());
    let mut p_k = two;
    let mut p_kp1 = *g;
    let expbits = u64::BITS - u64::leading_zeros(exp);
    for i in 1..expbits {
        // For i=1, k=1
        // For i=2, k=2 or 3.
        let k = exp >> (expbits - i);
        if k % 2 == 0 {
            // (k,k+1) => (2k,2k+1)
            (p_k, p_kp1) = (
                zn.sub(&zn.mul(&p_k, &p_k), &two),
                zn.sub(&zn.mul(&p_k, &p_kp1), g),
            );
        } else {
            // (k,k+1) => (2k+1,2k+2)
            (p_k, p_kp1) = (
                zn.sub(&zn.mul(&p_k, &p_kp1), g),
                zn.sub(&zn.mul(&p_kp1, &p_kp1), &two),
            );
        }
    }
    // For the last step, no need to compute exp+1
    if exp % 2 == 0 {
        zn.sub(&zn.mul(&p_k, &p_k), &two)
    } else {
        zn.sub(&zn.mul(&p_k, &p_kp1), g)
    }
}

#[test]
fn test_pp1() {
    use std::str::FromStr;
    let v = Verbosity::Silent;

    // A rather strong prime. Neither p-1 or p+1 are smooth.
    let p128 = Uint::from_str("192361420203955321314102766284003105319").unwrap();

    // p+1 is smooth (max factor 1303)
    // 9^2-4 is not a square mod p
    let p = Uint::from_digit(4106365409);
    let Some((fs, q)) = pp1(p * p128, 9, 1500, 30e3, v) else {
        panic!("failed Pollard P-1")
    };
    assert_eq!(fs, vec![p]);
    assert_eq!(q, p128);
    assert!(pp1(p * p128, 5, 1500, 30e3, v).is_none());

    // p+1 = 2^2 * 3 * 7 * 11 * 139 * 77893 * 25293451
    // 3^2-4 is not a square mod p
    // 25293451 % 9240 = 3571
    let p = Uint::from_digit(253042395370635947);
    let Some((fs, q)) = pp1(p * p128, 3, 80_000, 28e6, v) else {
        panic!("failed Pollard P+1")
    };
    assert_eq!(fs, vec![p]);
    assert_eq!(q, p128);

    // p+1 = 2^3 * 3 * 7 * 89 * 523 * 1279 * 21323177
    // 3^2-4 is not a square mod p
    // 21323177 % 9240 = 6497
    let p = Uint::from_digit(213266888931348167);
    let Some((fs, q)) = pp1(p * p128, 3, 2000, 28e6, v) else {
        panic!("failed Pollard P+1")
    };
    assert_eq!(fs, vec![p]);
    assert_eq!(q, p128);
}

#[test]
fn test_chebyshev_modn() {
    // 2^22 3^15 - 1 is prime.
    // 5=3^2-4 is not a quadratic residue
    let p = 60183678025727;
    let zn = ZmodN::new(p.into());
    // g must have order p+1
    let g = zn.from_int(Uint::from_digit(3));
    let gk = chebyshev_modn(&zn, &g, p + 1);
    assert_eq!(gk, zn.from_int(Uint::from_digit(2)));
    // Order of g is not p-1
    let gk = chebyshev_modn(&zn, &g, p - 1);
    assert!(gk != zn.from_int(Uint::from_digit(2)));
}
