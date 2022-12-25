// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Polynomial selection for Multiple Polynomial Quadratic Sieve
//!
//! Bibliography:
//! Robert D. Silverman, The multiple polynomial quadratic sieve
//! Math. Comp. 48, 1987, https://doi.org/10.1090/S0025-5718-1987-0866119-8

use std::sync::RwLock;

use bnum::cast::CastFrom;
use num_traits::One;
use rayon::prelude::*;

use crate::arith::{self, inv_mod, isqrt, pow_mod, Num, U256};
use crate::fbase::{self, FBase};
use crate::params::{self, BLOCK_SIZE};
use crate::relations::{Relation, RelationSet};
use crate::sieve;
use crate::{Int, Uint, DEBUG};

pub fn mpqs(
    n: Uint,
    prefs: &params::Preferences,
    tpool: Option<&rayon::ThreadPool>,
) -> Vec<Relation> {
    // Choose factor base. Sieve twice the number of primes
    // (n will be a quadratic residue for only half of them)
    let fb = prefs.fb_size.unwrap_or(params::factor_base_size(&n));
    let fbase = FBase::new(n, fb);
    eprintln!("Smoothness bound {}", fbase.bound());
    eprintln!("Factor base size {} ({:?})", fbase.len(), fbase.smalls());

    let mut target = fbase.len() * 8 / 10;
    let fb = fbase.len();
    let mlog = mpqs_interval_logsize(&n);
    if mlog >= 20 {
        eprintln!("Sieving interval size {}M", 2 << (mlog - 20));
    } else {
        eprintln!("Sieving interval size {}k", 2 << (mlog - 10));
    }
    // Precompute starting point for polynomials
    // See [Silverman, Section 3]
    // Look for A=D^2 such that (2A M/2)^2 ~= N/2
    // D is less than 64-bit for a 256-bit n
    let mut polybase: Uint = isqrt(n >> 1) >> mlog;
    polybase = isqrt(polybase);
    let maxprime = fbase.bound() as u64;
    if polybase < Uint::from(maxprime) {
        polybase = Uint::from(maxprime + 1000);
    }
    // Generate multiple polynomials at a time.
    // For small numbers (90-140 bits) usually less than
    // a hundred polynomials will provide enough relations.
    // We multiply by 1.4 log2(n) which is the expected gap between
    // 2 solutions.
    let polystride = if n.bits() < 120 {
        20 * 20 / 7 * polybase.bits()
    } else if n.bits() < 130 && tpool.is_some() {
        20 * 20 / 7 * polybase.bits()
    } else {
        100 * 20 / 7 * polybase.bits()
    };
    let mut polys = select_polys(polybase, polystride as usize, &n);
    let mut polyidx = 0;
    let mut polys_done = 0;
    let use_double = n.bits() > 256;
    eprintln!("Generated {} polynomials", polys.len());
    let maxlarge: u64 = maxprime
        * prefs
            .large_factor
            .unwrap_or(large_prime_factor(&n, use_double));
    let rels = RwLock::new(RelationSet::new(n, maxlarge));
    eprintln!("Max cofactor {}", maxlarge);
    loop {
        // Pop next polynomial.
        if polyidx == polys.len() {
            let rels = rels.read().unwrap();
            let gap = rels.gap();
            if gap == 0 {
                eprintln!("Found enough relations");
                break;
            }
            rels.log_progress(format!(
                "Sieved {}M {} polys",
                ((polys_done) << (mlog + 1 - 10)) >> 10,
                polys_done,
            ));
            polybase += Uint::from(polystride);
            polys = select_polys(polybase, polystride as usize, &n);
            polyidx = 0;
            eprintln!("Generated {} polynomials", polys.len());
        }
        if let Some(pool) = tpool {
            // Parallel sieving: do all polynomials at once.
            let v = pool.install(|| {
                (&polys[polyidx..])
                    .par_iter()
                    .map(|p| mpqs_poly(p, n, &fbase, maxlarge, use_double, &rels))
                    .collect()
            });
            polys_done += polys.len() - polyidx;
            polyidx = polys.len();
            v
        } else {
            // Sequential sieving
            let pol = &polys[polyidx];
            polyidx += 1;
            polys_done += 1;
            mpqs_poly(pol, n, &fbase, maxlarge, use_double, &rels);
        }
        let rels = rels.read().unwrap();
        if rels.len() >= target {
            let gap = rels.gap();
            if gap == 0 {
                eprintln!("Found enough relations");
                break;
            } else {
                eprintln!("Need {} additional relations", gap);
                target += gap + std::cmp::min(10, fb as usize / 4);
            }
        }
    }
    {
        let r = rels.read().unwrap();
        r.log_progress(format!(
            "Sieved {}M {} polys",
            ((polys_done) << (mlog + 1 - 10)) >> 10,
            polys_done,
        ));
    }
    rels.into_inner().unwrap().into_inner()
}

/// A polynomial is an omitted quadratic Ax^2 + Bx + C
/// such that (ax+b)^2-n = d^2(Ax^2 + Bx + C) and A=d^2
/// and the polynomial values are small.
///
/// The polynomial values are divisible by p iff
/// ax+b is a square root of n modulo p
#[derive(Debug)]
pub struct Poly {
    pub a: U256,
    pub b: U256,
    pub d: U256,
}

impl Poly {
    pub fn prepare_prime(
        &self,
        p: u32,
        r: u32,
        div: &arith::Dividers,
        offset: i32,
    ) -> sieve::SievePrime {
        let off: u32 = div.div31.modi32(offset);
        let shift = |r: u32| -> u32 {
            if r < off as u32 {
                r + p - off
            } else {
                r - off as u32
            }
        };

        // Determine roots r1, r2 such that P(offset+r)==0 mod p.
        let offsets = if p == 2 {
            // We don't really know what will happen.
            [Some(0), Some(1)]
        } else {
            // Transform roots as: r -> (r - B) / A
            let a = div.divmod_uint(&self.a).1;
            let b = div.divmod_uint(&self.b).1;
            let ainv = div.inv(a).unwrap();
            [
                Some(shift(
                    div.divmod64((p as u64 + r as u64 - b) * ainv).1 as u32,
                )),
                Some(shift(
                    div.divmod64((2 * p as u64 - r as u64 - b) * ainv).1 as u32,
                )),
            ]
        };
        sieve::SievePrime { p: p, offsets }
    }
}

#[test]
fn test_poly_prime() {
    use crate::arith;
    use std::str::FromStr;

    let p = 10223;
    let r = 4526;
    let div = arith::Dividers::new(10223);
    let poly = Poly {
        a: U256::from_str("13628964805482736048449433716121").unwrap(),
        b: U256::from_str("2255304218805619815720698662795").unwrap(),
        d: U256::from(3691742787015739u64),
    };
    for rt in poly.prepare_prime(p, r, &div, 0).offsets {
        let Some(rt) = rt else { continue };
        let x1: Uint = Uint::cast_from(poly.a) * Uint::from(rt) + Uint::cast_from(poly.b);
        let x1p: u64 = (x1 % Uint::from(p)).to_u64().unwrap();
        assert_eq!(pow_mod(x1p, 2, p as u64), pow_mod(r as u64, 2, p as u64));
    }
}

/// Returns a set of polynomials suitable for sieving across ±2^sievebits
/// The base offset is a seed for prime generation.
pub fn select_polys(base: Uint, width: usize, n: &Uint) -> Vec<Poly> {
    sieve_for_polys(base, width, &n)
        .into_iter()
        .map(|(d, r)| make_poly(d, r, &n))
        .collect()
}

pub fn sieve_for_polys(bmin: Uint, width: usize, n: &Uint) -> Vec<(Uint, Uint)> {
    let mut composites = vec![false; width as usize];
    for &p in fbase::SMALL_PRIMES {
        let off = bmin % (p as u64);
        let mut idx = -(off as isize);
        while idx < composites.len() as isize {
            if idx >= 0 {
                composites[idx as usize] = true
            }
            idx += p as isize
        }
    }
    let base4 = bmin.low_u64() % 4;
    let mut result = vec![];
    for i in 0..width {
        if !composites[i] && (base4 + i as u64) % 4 == 3 {
            // No small factor, 3 mod 4
            let p = bmin + Uint::from(i);
            let r = pow_mod(*n, (p >> 2) + Uint::one(), p);
            if (r * r) % p == n % p {
                result.push((p, r));
            }
        }
    }
    result
}

fn make_poly(d: Uint, r: Uint, n: &Uint) -> Poly {
    // Lift square root mod D^2
    // Since D*D < N, computations can be done using the same integer width.
    let h1 = r;
    let c = ((n - h1 * h1) / d) % d;
    let h2 = (c * inv_mod(h1 << 1, d).unwrap()) % d;
    // (h1 + h2*D)**2 = n mod D^2
    let mut b = (h1 + h2 * d) % (d * d);

    // If kn = 1 mod 4:
    // A = D^2, B = sqrt(n) mod D^2, C = (B^2 - kn) / 4A
    // (2Ax + B)^2 - kn = 4A (Ax^2 + B x + C)
    // Ax^2 + Bx + C = ((2Ax + B)/2D)^2 mod n
    //
    // otherwise:
    // A = D^2, B = sqrt(n) mod D^2, C = (4B^2 - kn) / A
    // (Ax+2B)^2 - kn = A (Ax^2 + 2Bx + C)
    // Ax^2 + 2Bx + C = ((Ax+2B)/D)^2 mod n
    if n.low_u64() % 4 == 1 {
        // want an odd b
        if b.low_u64() % 2 == 0 {
            b = d * d - b;
        }
        Poly {
            a: U256::cast_from(d * d << 1),
            b: U256::cast_from(b),
            d: U256::cast_from(d << 1),
        }
    } else {
        // want even b
        if b.low_u64() % 2 == 1 {
            b = d * d - b;
        }
        Poly {
            a: U256::cast_from(d * d),
            b: U256::cast_from(b),
            d: U256::cast_from(d),
        }
    }
}

#[test]
fn test_select_poly() {
    use crate::arith::{isqrt, sqrt_mod};
    use crate::Int;
    use std::str::FromStr;

    let n = Uint::from_str(
        "104567211693678450173299212092863908236097914668062065364632502155864426186497",
    )
    .unwrap();
    let mut polybase: Uint = isqrt(n >> 1) >> 24;
    polybase = isqrt(polybase);
    let Poly { a, b, d } = select_polys(polybase, 512, &n)[0];
    let (a, b, d) = (Uint::cast_from(a), Uint::cast_from(b), Uint::cast_from(d));
    // D = 3 mod 4, 2D = 6 mod 8
    assert_eq!(d.low_u64() % 8, 6);
    // N is a square modulo D
    assert!(sqrt_mod(n, d >> 1).is_some());
    // A = D^2, 2A = (2D)^2
    assert_eq!(a << 1, d * d);
    // B^2 = N mod 4D^2
    assert_eq!(pow_mod(b, Uint::from(2u64), d * d), n % (d * d));
    println!("D={} A={} B={}", d, a, b);

    let c = (n - (b * b)) / (a << 2);

    // Check that:
    // Ax²+Bx+C is small
    // 4A(Ax²+Bx+C) = (2Ax+B)^2 mod N
    let x = Uint::from(1_234_567u64);
    let num = ((a << 1) * x + b) % n;
    let den = inv_mod(d << 1, n).unwrap();
    println!("1/2D={}", den);
    let q = (num * den) % n;
    println!("{}", q);
    let q2 = pow_mod(q, Uint::from(2u64), n);
    println!("{}", q2);
    let px = Int::from_bits(a * x * x + b * x) - Int::from_bits(c);
    println!("Ax²+Bx+C = {}", px);
    assert!(px.abs().bits() <= 128 + 24);
}

// One MPQS unit of work, identified by an integer 'idx'.
fn mpqs_poly(
    pol: &Poly,
    n: Uint,
    fbase: &FBase,
    maxlarge: u64,
    use_double: bool,
    rels: &RwLock<RelationSet>,
) {
    let mlog = mpqs_interval_logsize(&n);
    let nblocks = (2 << mlog) / BLOCK_SIZE;
    if DEBUG {
        eprintln!(
            "Sieving polynomial A={} B={} M=2^{} blocks={}",
            pol.a, pol.b, mlog, nblocks
        );
    }
    // Precompute inverse of D^2
    let d = Uint::cast_from(pol.d);
    let d2inv = inv_mod(d * d, n).unwrap();
    // Precompute inverse of D
    let dinv = inv_mod(d, n).unwrap();

    // Sieve from -M to M
    let sieve = SieveMPQS {
        n,
        fbase,
        pol,
        dinv,
        d2inv,
        maxlarge,
        use_double,
        rels: rels,
    };

    let start_offset = -(1 << mlog);
    let end_offset = 1 << mlog;
    let pfunc = |pidx| {
        let p = fbase.p(pidx);
        let r = fbase.r(pidx);
        let div = fbase.div(pidx);
        pol.prepare_prime(p, r, div, start_offset as i32)
    };
    let mut state = sieve::Sieve::new(start_offset, nblocks, fbase, &pfunc);
    if nblocks == 0 {
        sieve_block_poly(&sieve, &mut state);
    }
    while state.offset < end_offset {
        sieve_block_poly(&sieve, &mut state);
        state.next_block();
    }
}

fn mpqs_interval_logsize(n: &Uint) -> u32 {
    let sz = n.bits();
    match sz {
        // Small numbers don't have enough polynomials,
        // but we compensate already with large cofactors.
        0..=119 => 15,
        // MPQS has a high polynomial switch cost, the interval
        // must be large enough for very large n.
        120..=279 => 13 + sz / 40, // 16..20
        280..=295 => 21,
        _ => 22,
    }
}

fn large_prime_factor(n: &Uint, doubles: bool) -> u64 {
    // Allow large cofactors up to FACTOR * largest prime
    let sz = n.bits();
    match sz {
        0..=49 => {
            // Large cofactors for extremely small numbers
            // to compensate small intervals
            100 + 2 * n.bits() as u64
        }
        50..=100 =>
        // Polynomials are scarce, we need many relations:
        {
            300 - 2 * n.bits() as u64 // 200..100
        }
        _ => {
            // Less large primes if we use double large primes.
            if doubles {
                n.bits() as u64 / 16
            } else {
                n.bits() as u64
            }
        }
    }
}

struct SieveMPQS<'a> {
    n: Uint,
    fbase: &'a FBase,
    pol: &'a Poly,
    dinv: Uint,
    d2inv: Uint,
    maxlarge: u64,
    use_double: bool,
    rels: &'a RwLock<RelationSet>,
}

// Sieve using a selected polynomial
fn sieve_block_poly(s: &SieveMPQS, st: &mut sieve::Sieve) {
    st.sieve_block();

    let offset = st.offset;
    let maxlarge = s.maxlarge;
    assert!(maxlarge == (maxlarge as u32) as u64);
    let max_cofactor: u64 = if s.use_double {
        maxlarge * maxlarge
    } else {
        maxlarge
    };
    let target = s.n.bits() / 2 + mpqs_interval_logsize(&s.n) - max_cofactor.bits();
    let (pol, n, dinv, d2inv) = (s.pol, &s.n, &s.dinv, &s.d2inv);
    let (idxs, facss) = st.smooths(target as u8);
    for (i, facs) in idxs.into_iter().zip(facss) {
        let mut factors: Vec<(i64, u64)> = Vec::with_capacity(20);
        // Evaluate polynomial
        let x: Int = Int::cast_from(pol.a) * Int::from(offset + (i as i64));
        let x = x + Int::cast_from(pol.b);
        let candidate: Int = x * x - Int::from_bits(*n);
        let cabs = (candidate.abs().to_bits() * d2inv) % n;
        if candidate.is_negative() {
            factors.push((-1, 1));
        }
        let mut cofactor: Uint = cabs;
        for pidx in facs {
            let mut exp = 0;
            let div = s.fbase.div(pidx);
            loop {
                let (q, r) = div.divmod_uint(&cofactor);
                if r == 0 {
                    cofactor = q;
                    exp += 1;
                } else {
                    break;
                }
            }
            factors.push((s.fbase.p(pidx) as i64, exp));
        }
        let Some(cofactor) = cofactor.to_u64() else { continue };
        if cofactor > max_cofactor {
            continue;
        }
        let pq = fbase::try_factor64(cofactor);
        if pq.is_none() && cofactor > maxlarge {
            continue;
        }

        let sabs = (x.abs().to_bits() * dinv) % n;
        let xrel = if candidate.is_negative() {
            n - sabs
        } else {
            sabs
        };
        if DEBUG {
            eprintln!("i={} smooth {} cofactor {}", i, cabs, cofactor);
        }
        let rel = Relation {
            x: xrel,
            cofactor,
            factors,
            cyclelen: 1,
        };
        debug_assert!(rel.verify(&n));
        s.rels.write().unwrap().add(rel, pq);
    }
}
