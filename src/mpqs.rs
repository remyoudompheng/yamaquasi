// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Polynomial selection for Multiple Polynomial Quadratic Sieve
//!
//! Bibliography:
//! Robert D. Silverman, The multiple polynomial quadratic sieve
//! Math. Comp. 48, 1987, https://doi.org/10.1090/S0025-5718-1987-0866119-8

use std::collections::HashMap;

use bnum::cast::CastFrom;
use num_traits::One;
use rayon::prelude::*;

use crate::arith::{inv_mod, isqrt, pow_mod, Num, U256};
use crate::fbase::{self, Prime};
use crate::params::{self, BLOCK_SIZE};
use crate::relations::{combine_large_relation, relation_gap, Relation};
use crate::sieve;
use crate::{Int, Uint, DEBUG};

pub fn mpqs(n: Uint, primes: &[Prime], tpool: Option<&rayon::ThreadPool>) -> Vec<Relation> {
    let mut target = primes.len() * 8 / 10;
    let mut relations = vec![];
    let mut larges = HashMap::<u64, Relation>::new();
    let mut extras = 0;
    let fb = primes.len();
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
    let maxprime = primes.last().unwrap().p;
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
    eprintln!("Generated {} polynomials", polys.len());
    let maxlarge: u64 = maxprime * large_prime_factor(&n);
    eprintln!("Max cofactor {}", maxlarge);
    loop {
        // Pop next polynomial.
        if polyidx == polys.len() {
            let gap = relation_gap(&relations);
            if gap == 0 {
                eprintln!("Found enough relations");
                break;
            }
            eprintln!(
                "Sieved {}M {} polys found {} smooths (cofactors: {} combined, {} pending)",
                ((polys_done) << (mlog + 1 - 10)) >> 10,
                polys_done,
                relations.len(),
                extras,
                larges.len(),
            );
            polybase += Uint::from(polystride);
            polys = select_polys(polybase, polystride as usize, &n);
            polyidx = 0;
            eprintln!("Generated {} polynomials", polys.len());
        }
        let mut results: Vec<(Vec<_>, Vec<_>)> = if let Some(pool) = tpool {
            // Parallel sieving: do all polynomials at once.
            let v = pool.install(|| {
                (&polys[polyidx..])
                    .par_iter()
                    .map(|p| mpqs_poly(p, n, &primes))
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
            vec![mpqs_poly(pol, n, &primes)]
        };
        for (ref mut found, foundlarge) in &mut results {
            relations.append(found);
            for r in foundlarge {
                if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                    relations.push(rr);
                    extras += 1;
                }
            }
        }
        if relations.len() >= target {
            let gap = relation_gap(&relations);
            if gap == 0 {
                eprintln!("Found enough relations");
                break;
            } else {
                eprintln!("Need {} additional relations", gap);
                target += gap + std::cmp::min(10, fb as usize / 4);
            }
        }
    }
    eprintln!(
        "Sieved {}M {} polys found {} smooths (cofactors: {} combined, {} pending)",
        ((polys_done) << (mlog + 1 - 10)) >> 10,
        polys_done,
        relations.len(),
        extras,
        larges.len(),
    );
    relations
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
    pub fn prepare_prime(&self, p: &Prime, offset: i32) -> [Option<u32>; 2] {
        let off: u32 = p.div.modi32(offset);
        let shift = |r: u32| -> u32 {
            if r < off as u32 {
                r + p.p as u32 - off
            } else {
                r - off as u32
            }
        };

        // Determine roots r1, r2 such that P(offset+r)==0 mod p.
        if p.p == 2 {
            // We don't really know what will happen.
            [Some(0), Some(1)]
        } else {
            // Transform roots as: r -> (r - B) / A
            let a = p.div.divmod_uint(&self.a).1;
            let b = p.div.divmod_uint(&self.b).1;
            let ainv = p.div.inv(a).unwrap();
            [
                Some(shift(p.div.divmod64((p.p + p.r - b) * ainv).1 as u32)),
                Some(shift(p.div.divmod64((2 * p.p - p.r - b) * ainv).1 as u32)),
            ]
        }
    }
}

#[test]
fn test_poly_prime() {
    use crate::arith;
    use std::str::FromStr;

    let p = Prime {
        p: 10223,
        r: 4526,
        div: arith::Dividers::new(10223),
    };
    let poly = Poly {
        a: U256::from_str("13628964805482736048449433716121").unwrap(),
        b: U256::from_str("2255304218805619815720698662795").unwrap(),
        d: U256::from(3691742787015739u64),
    };
    for r in poly.prepare_prime(&p, 0) {
        let Some(r) = r else { continue };
        let x1: Uint = Uint::cast_from(poly.a) * Uint::from(r) + Uint::cast_from(poly.b);
        let x1p: u64 = (x1 % Uint::from(p.p)).to_u64().unwrap();
        assert_eq!(pow_mod(x1p, 2, p.p), pow_mod(p.r, 2, p.p));
    }
}

/// Returns a polynomial suitable for sieving across ±2^sievebits
/// The offset is a seed for prime generation.
pub fn select_poly(base: Uint, offset: u64, n: Uint) -> Poly {
    // Select an appropriate pseudoprime. It is enough to be able
    // to compute a modular square root of n.
    let (d, r) = sieve_poly(base, offset, n);
    make_poly(d, r, &n)
}

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

pub const POLY_STRIDE: usize = 32768;

fn sieve_at(base: Uint, offset: u64) -> [u64; 128] {
    if base.bits() > 1024 {
        panic!("not expecting to sieve {}-bit primes", base.bits())
    }
    let mut composites = [0u8; POLY_STRIDE];
    for &p in fbase::SMALL_PRIMES {
        let off = (base + Uint::from(offset)) % p;
        let mut idx = -(off as isize);
        while idx < composites.len() as isize {
            if idx >= 0 {
                composites[idx as usize] = 1
            }
            idx += p as isize
        }
    }
    let mut res = [0u64; 128];
    let mut idx = 0;
    let base4 = base.low_u64() % 4;
    for i in 0..POLY_STRIDE {
        if composites[i] == 0 && (base4 + i as u64) % 4 == 3 {
            res[idx] = offset + i as u64;
            idx += 1;
        }
        if idx == 128 {
            break;
        }
    }
    res
}

fn sieve_poly(base: Uint, offset: u64, n: Uint) -> (Uint, Uint) {
    let offs = sieve_at(base, offset);
    for o in offs {
        if o == 0 {
            continue;
        }
        let base = base + Uint::from(o);
        if base.low_u64() % 4 == 3 {
            // Compute pow(n, (d+1)/4, d)
            let r = pow_mod(n, (base >> 2) + Uint::one(), base);
            if (r * r) % base == n % base {
                return (base, r);
            }
        }
    }
    panic!(
        "impossible! failed to find a pseudoprime {} {}=>{:?}",
        base, offset, offs
    )
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
    let Poly { a, b, d } = select_poly(polybase, 0, n);
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
fn mpqs_poly(pol: &Poly, n: Uint, primes: &[Prime]) -> (Vec<Relation>, Vec<Relation>) {
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
        primes,
        pol,
        dinv,
        d2inv,
    };

    let start_offset = -(1 << mlog);
    let end_offset = 1 << mlog;
    let mut state = sieve::Sieve::new(start_offset, nblocks, primes, |_, p, offset| {
        pol.prepare_prime(p, offset as i32)
    });
    if nblocks == 0 {
        return sieve_block_poly(&sieve, &mut state);
    }
    let mut result: Vec<Relation> = vec![];
    let mut extras: Vec<Relation> = vec![];
    while state.offset < end_offset {
        let (mut x, mut y) = sieve_block_poly(&sieve, &mut state);
        result.append(&mut x);
        extras.append(&mut y);
        state.next_block();
    }
    (result, extras)
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

fn large_prime_factor(n: &Uint) -> u64 {
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
        _ => n.bits() as u64,
    }
}

struct SieveMPQS<'a> {
    n: Uint,
    primes: &'a [Prime],
    pol: &'a Poly,
    dinv: Uint,
    d2inv: Uint,
}

// Sieve using a selected polynomial
fn sieve_block_poly(s: &SieveMPQS, st: &mut sieve::Sieve) -> (Vec<Relation>, Vec<Relation>) {
    st.sieve_block();

    let offset = st.offset;
    let maxprime = s.primes.last().unwrap().p;
    let maxlarge = maxprime * large_prime_factor(&s.n);
    let mut result = vec![];
    let mut extras = vec![];

    let target = s.n.bits() / 2 + mpqs_interval_logsize(&s.n) - maxlarge.bits();
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
        for p in facs {
            let mut exp = 0;
            loop {
                let (q, r) = p.div.divmod_uint(&cofactor);
                if r == 0 {
                    cofactor = q;
                    exp += 1;
                } else {
                    break;
                }
            }
            factors.push((p.p as i64, exp));
        }
        let Some(cofactor) = cofactor.to_u64() else { continue };
        if cofactor > maxlarge {
            continue;
        }
        let sabs = (x.abs().to_bits() * dinv) % n;
        let xrel = if candidate.is_negative() {
            n - sabs
        } else {
            sabs
        };
        if cofactor == 1 {
            if DEBUG {
                eprintln!("i={} smooth {}", i, cabs);
            }
            result.push(Relation {
                x: xrel,
                cofactor: 1,
                factors,
            });
        } else {
            if DEBUG {
                eprintln!("i={} smooth {} cofactor {}", i, cabs, cofactor);
            }
            extras.push(Relation {
                x: xrel,
                cofactor: cofactor,
                factors,
            });
        }
    }
    (result, extras)
}
