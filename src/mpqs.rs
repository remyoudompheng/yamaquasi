// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Multiple Polynomial Quadratic Sieve with large prime A
//!
//! Bibliography:
//! Robert D. Silverman, The multiple polynomial quadratic sieve
//! Math. Comp. 48, 1987, <https://doi.org/10.1090/S0025-5718-1987-0866119-8>

use std::cmp::{max, min};
use std::sync::RwLock;

use bnum::cast::CastFrom;
use rayon::prelude::*;

use crate::arith::{self, isqrt, pow_mod, Num, I256, U256};
use crate::arith_gcd::inv_mod;
use crate::fbase::{self, FBase};
use crate::params;
use crate::relations::{self, Relation, RelationSet};
use crate::sieve::{self, BLOCK_SIZE};
use crate::{Int, Preferences, Uint, Verbosity};

/// Run MPQS to factor `n` with multiplier k
pub fn mpqs(n: Uint, k: u32, prefs: &Preferences, tpool: Option<&rayon::ThreadPool>) -> Vec<Uint> {
    let (norig, n) = (n, n * Uint::from(k));
    // We cannot use MPQS for numbers above 448 bits
    // (or at least it is extremely unreasonable and cost days of CPU).
    // This is so that sqrt(n) * interval size always fits in 256 bits.
    // In particular the D values chosen for polynomials cannot exceed 128 bits.
    if n.bits() > 448 {
        if prefs.verbose(Verbosity::Info) {
            eprintln!("Number {n} too large for classical quadratic sieve!");
        }
        return vec![];
    }

    let use_double = prefs.use_double.unwrap_or(n.bits() > 256);
    // Choose factor base. Sieve twice the number of primes
    // (n will be a quadratic residue for only half of them)
    let fb = prefs.fb_size.unwrap_or(fb_size(&n, use_double));
    let fbase = FBase::new(n, fb);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Smoothness bound {}", fbase.bound());
        eprintln!("Factor base size {} ({:?})", fbase.len(), fbase.smalls());
    }
    let mut target = fbase.len() * 8 / 10;
    let fb = fbase.len();
    let mm = prefs.interval_size.unwrap_or(mpqs_interval_size(&n) as u32);
    if prefs.verbose(Verbosity::Info) {
        if mm > 2 << 20 {
            eprintln!("Sieving interval size {}M", mm >> 20);
        } else {
            eprintln!("Sieving interval size {}k", mm >> 10);
        }
    }

    // Precompute starting point for polynomials
    // See [Silverman, Section 3]
    // Look for A=D^2 such that (2A M/2)^2 ~= N/2
    // D is less than 64-bit for a 256-bit n
    let a_target: Uint = if n % 4 == 1 {
        isqrt(n >> 1) / Uint::from(mm as u64 / 2)
    } else {
        isqrt(n << 1) / Uint::from(mm as u64 / 2)
    };
    let d_target = max(Uint::from_digit(3), isqrt(a_target));
    assert!(d_target.bits() < 127);
    let maxprime = fbase.bound() as u64;

    // Precompute inverters: preparation of polynomials is almost entirely spent
    // in modular inversion.
    let inverters: Vec<_> = (0..fb)
        .map(|idx| arith::Inverter::new(fbase.p(idx)))
        .collect();

    // Generate multiple polynomials at a time.
    // For small numbers (90-140 bits) usually less than
    // a hundred polynomials will provide enough relations.
    // We multiply by 1.4 log2(n) which is the expected gap between
    // 2 solutions.
    let polystride = match n.bits() {
        // Interval size is larger than sqrt(n)
        0..=32 => 200,
        // Small integer, process enough number at a time.
        33..=64 => 50 * 20 / 7 * d_target.bits(),
        65..=130 => 20 * 20 / 7 * d_target.bits(),
        // Enough for efficient parallelism
        131..=240 => 100 * 20 / 7 * d_target.bits(),
        241.. => 200 * 20 / 7 * d_target.bits(),
    };

    // Start slightly before the ideal values to get a few nice polynomials.
    let mut polybase = u128::cast_from(d_target);
    if polybase >= 20 {
        polybase -= min(polybase / 10, polystride.into());
    }

    let mut d_r_values = sieve_for_polys(&fbase, &n, polybase, polystride as usize);
    let mut polyidx = 0;
    let mut polys_done: u64 = 0;
    if prefs.verbose(Verbosity::Verbose) {
        eprintln!(
            "Generated {} polynomials D={}..{} optimal={d_target}",
            d_r_values.len(),
            d_r_values[0].0,
            d_r_values.last().unwrap().0
        );
    }
    let maxlarge: u64 = maxprime * prefs.large_factor.unwrap_or(large_prime_factor(&n));
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Max large prime {maxlarge}");
        if use_double {
            eprintln!(
                "Max double large prime {}",
                maxlarge * fbase.bound() as u64 * 2
            );
        }
    }
    let rels = RwLock::new(RelationSet::new(n, fbase.len(), maxlarge));

    // Prepare shared context.
    let s = SieveMPQS {
        n,
        fbase: &fbase,
        inverters: &inverters,
        maxlarge,
        use_double,
        interval_size: mm as i64,
        rels: &rels,
        prefs,
    };

    loop {
        // Pop next polynomial.
        if polyidx == d_r_values.len() {
            let rels = rels.read().unwrap();
            let gap = rels.gap(s.fbase);
            if gap == 0 {
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Found enough relations");
                }
                break;
            }
            if prefs.verbose(Verbosity::Verbose) {
                rels.log_progress(format!(
                    "Sieved {}M {polys_done} polys",
                    (polys_done * s.interval_size as u64) >> 20,
                ));
            }
            polybase += polystride as u128;
            d_r_values = sieve_for_polys(&fbase, &n, polybase, polystride as usize);
            polyidx = 0;
            if prefs.verbose(Verbosity::Info) {
                eprintln!(
                    "Generated {} polynomials D={}..{} optimal={d_target}",
                    d_r_values.len(),
                    d_r_values[0].0,
                    d_r_values.last().unwrap().0
                );
            }
        }
        if let Some(pool) = tpool {
            // Parallel sieving: do all polynomials at once.
            // We don't have many polynomials but optimistically assume
            // that splitting in chunks will enable memory recycling
            // while still being efficient for parallelism.
            pool.install(|| {
                d_r_values[polyidx..].par_chunks(8).for_each(|chunk| {
                    let mut wks = Workspace::default();
                    for (a, r) in chunk {
                        mpqs_poly(&s, a, r, &mut wks);
                    }
                })
            });
            polys_done += (d_r_values.len() - polyidx) as u64;
            polyidx = d_r_values.len();
        } else {
            // Sequential sieving
            let (a, r) = &d_r_values[polyidx];
            let mut wks = Workspace::default();
            polyidx += 1;
            polys_done += 1;
            mpqs_poly(&s, a, r, &mut wks);
        }
        let rels = rels.read().unwrap();
        if rels.len() >= target {
            let gap = rels.gap(s.fbase);
            if gap == 0 {
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Found enough relations");
                }
                break;
            } else {
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Need {} additional relations", gap);
                }
                target += gap + std::cmp::min(10, fb / 4);
            }
        }
    }
    if prefs.verbose(Verbosity::Info) {
        let r = rels.read().unwrap();
        r.log_progress(format!(
            "Sieved {}M {polys_done} polys",
            (polys_done * s.interval_size as u64) >> 20
        ));
    }
    let mut rels = rels.into_inner().unwrap();
    if rels.len() > fbase.len() + relations::MIN_KERNEL_SIZE {
        rels.truncate(fbase.len() + relations::MIN_KERNEL_SIZE)
    }
    let rels = rels.into_inner();
    if rels.len() == 0 {
        return vec![];
    }
    relations::final_step(&norig, &fbase, &rels, prefs.verbosity)
}

/// A polynomial is a quadratic Ax^2 + Bx + C
/// such that (Ax+BB)^2-n = D^2 (Ax^2 + Bx + C) and A=D^2
/// and the polynomial values are small.
///
/// The polynomial values are divisible by p iff
/// ax+b is a square root of n modulo p
#[derive(Debug)]
pub struct Poly {
    // The polynomial is ax^2+bx+c
    pub a: U256,
    pub b: U256,
    c: I256,
    // (ax+bb)/d is the modular square root of ax^2+bx+c
    bb: Uint,
    pub d: U256,
    dinv: Uint,
}

impl Poly {
    /// Evaluate polynomial at the specified point. The absolute value
    /// of x must not exceed 2^32.
    ///
    /// The polynomial value P(x) in D^2 P(x)=y^2 is always small,
    /// and we return (P(x), y/D mod n)
    pub fn eval(&self, x: i64) -> (I256, Uint) {
        let ax = I256::cast_from(self.a) * I256::cast_from(x);
        let v = (ax + I256::cast_from(self.b)) * I256::cast_from(x) + self.c;
        let y = Uint::cast_from((Int::cast_from(ax) + Int::cast_from(self.bb)).abs()) * self.dinv;
        (v, y)
    }

    pub fn prepare_prime(
        &self,
        p: u32,
        r: u32,
        div: &arith::Dividers,
        inv: &arith::Inverter,
        offset: i32,
    ) -> (u32, u32) {
        let off: u32 = div.div31.modi32(offset);
        let shift = |r: u32| -> u32 {
            if r < off {
                r + p - off
            } else {
                r - off
            }
        };

        // Determine roots r1, r2 such that P(offset+r)==0 mod p.
        if p == 2 {
            // We don't really know what will happen.
            (0, 1)
        } else {
            // Transform roots as:
            // if n % 4 == 1 (b is odd), r -> (r - B) / 2A
            // if n % 4 != 1 (b is even), r -> (r - B/2) / A
            let (a, b) = if self.b.bit(0) {
                (2 * div.divmod_uint(&self.a).1, div.divmod_uint(&self.b).1)
            } else {
                (div.divmod_uint(&self.a).1, div.divmod_uint(&self.bb).1)
            };
            if a == 0 {
                // For very small integers, we may select D inside the factor base.
                // In this case the roots are the roots of Bx-abs(C) (C < 0)
                let b = div.divmod_uint(&self.b).1;
                let binv = inv.invert(b as u32, &div.div64) as u64;
                debug_assert!(self.c.is_negative());
                let c = div.divmod_uint(&self.c.abs().to_bits()).1;
                let r = shift(div.divmod64(c * binv).1 as u32);
                (r, r)
            } else {
                let ainv = inv.invert(a as u32, &div.div64) as u64;
                let r1 = shift(div.divmod64((p as u64 + r as u64 - b) * ainv).1 as u32);
                (
                    r1,
                    if r == 0 {
                        r1
                    } else {
                        shift(div.divmod64((2 * p as u64 - r as u64 - b) * ainv).1 as u32)
                    },
                )
            }
        }
    }
}

#[doc(hidden)]
pub fn sieve_for_polys(fb: &FBase, n: &Uint, bmin: u128, width: usize) -> Vec<(Uint, Uint)> {
    let mut composites = vec![false; width];
    for &p in &fbase::SMALL_PRIMES {
        let off = bmin % p as u128;
        let mut idx = if bmin > p as u128 {
            -(off as isize)
        } else {
            2 * p as isize - bmin as isize // offset of 2p
        };
        while idx < composites.len() as isize {
            if idx >= 0 {
                composites[idx as usize] = true
            }
            idx += p as isize
        }
    }
    let base4 = bmin % 4;
    let mut result = vec![];
    'nextsieve: for i in 0..width {
        if !composites[i] && (base4 + i as u128) % 4 == 3 {
            // No small factor, 3 mod 4
            let p = bmin + i as u128;
            let p256 = U256::cast_from(p);
            let nmodp = U256::cast_from(n % Uint::from(p));
            let r = pow_mod(nmodp, U256::from((p + 1) / 4), p256);
            if (r * r) % p256 == nmodp {
                // Beware, D may (exceptionally) not be prime.
                // Perform trial division by the factor base.
                for idx in 0..fb.len() {
                    // If it is a member of the factor base, it is prime.
                    if p == fb.p(idx) as u128 {
                        break;
                    }
                    if fb.div(idx).mod_uint(&p256) == 0 {
                        continue 'nextsieve;
                    }
                }
                if r.is_zero() {
                    // FIXME: use this factor to answer.
                    eprintln!("WARNING: unexpectedly found a factor of N!");
                    eprintln!("D={}", p);
                    continue 'nextsieve;
                }
                result.push((p.into(), Uint::cast_from(r)));
            }
        }
    }
    result
}

/// Construct a MPQS polynomial from modulus N, a prime D and a modular square root D.
#[doc(hidden)]
pub fn make_poly(n: &Uint, d: &Uint, r: &Uint) -> Poly {
    debug_assert!((r * r) % d == n % d);
    // Lift square root mod D^2
    // Since D*D < N, computations can be done using the same integer width.
    let h1 = r;
    let c = ((n - h1 * h1) / d) % d;
    let h2 = (c * inv_mod(&(h1 << 1), &d).unwrap()) % d;
    // (h1 + h2*D)**2 = n mod D^2
    let mut b = h1 + h2 * d;

    // Precompute inverse of D
    let dinv = inv_mod(&d, &n).unwrap();
    assert!(d.bits() < 128);
    assert!(b.bits() < 256);
    debug_assert!((b * b) % (d * d) == n % (d * d));

    // If n = 1 mod 4:
    // A = D^2, B = sqrt(n) mod D^2, C = (B^2 - n) / 4A
    // (2Ax + B)^2 - n = 4A (Ax^2 + B x + C)
    // D^2 (Ax^2 + Bx + C) = (Ax + B/2)^2 mod n
    //
    // otherwise:
    // A = D^2, B = sqrt(n) mod D^2, C = (B^2 - n) / A
    // (Ax+B)^2 - n = A (Ax^2 + 2Bx + C)
    // D^2 (Ax^2 + 2Bx + C) = (Ax + B)^2 mod n
    if n.low_u64() % 4 == 1 {
        // want an odd b
        if !b.bit(0) {
            b = d * d - b;
        }
        debug_assert!((b * b) % (d * d << 2) == n % (d * d << 2));
        let c = (Int::cast_from(b * b) - Int::cast_from(*n)) / Int::cast_from(d * d << 2);
        assert!(c.abs().bits() < 256);
        Poly {
            a: U256::cast_from(d * d),
            b: U256::cast_from(b),
            c: I256::cast_from(c),
            bb: (n + b) >> 1,
            d: U256::cast_from(*d),

            dinv,
        }
    } else {
        // want even b
        if b.bit(0) {
            b = d * d - b;
        }
        let c = (Int::cast_from(b * b) - Int::cast_from(*n)) / Int::cast_from(d * d);
        assert!(c.abs().bits() < 256);
        Poly {
            a: U256::cast_from(d * d),
            b: U256::cast_from(b << 1),
            c: I256::cast_from(c),
            bb: b,
            d: U256::cast_from(*d),

            dinv,
        }
    }
}

#[test]
fn test_select_poly() {
    use crate::arith::{isqrt, sqrt_mod};
    use std::str::FromStr;

    // n = 1 mod 4
    let n = Uint::from_str(
        "104567211693678450173299212092863908236097914668062065364632502155864426186497",
    )
    .unwrap();
    let fb = fbase::FBase::new(n, 1000);
    let mlog: u32 = 24;
    let polybase: Uint = isqrt(n >> 1) >> mlog;
    let polybase = u128::cast_from(isqrt(polybase));
    let (d, r) = sieve_for_polys(&fb, &n, polybase, 512)[0];
    let pol = make_poly(&n, &d, &r);
    let &Poly { a, b, d, .. } = &pol;
    let (a, b, d) = (Uint::cast_from(a), Uint::cast_from(b), Uint::cast_from(d));
    // D = 3 mod 4
    assert_eq!(d.low_u64() % 4, 3);
    // N is a square modulo D
    assert!(sqrt_mod(n, d).is_some());
    // A = D^2
    assert_eq!(a, d * d);
    // B^2 = N mod 4D^2
    assert_eq!(pow_mod(b, Uint::from(2u64), d * d), n % (d * d));
    eprintln!("D={d} A={a} B={b}");
    // C = (N - B^2)/4D^2
    let c: Uint = (n - (b * b)) / (a << 1);
    eprintln!("n = {n}");
    eprintln!("P = {}*x^2+{}*x-{}", a >> 1, b, c);

    // Check that:
    // Ax²+Bx+C is small and balanced
    // min,max = ±sqrt(2N)*M
    let pmin = pol.eval(0).0;
    let pmax = pol.eval(1 << mlog).0;
    let target = U256::cast_from(isqrt(n >> 1) << (mlog - 1));
    let sz = target.bits();
    eprintln!("min(P) = {pmin}");
    eprintln!("max(P) = {pmax}");
    assert!(pmin.is_negative() && pmin.abs().to_bits() >> (sz - 12) == target >> (sz - 12));
    assert!(pmax.is_positive() && pmax.to_bits() >> (sz - 12) == target >> (sz - 12));

    // n = 3 mod 4
    let n = Uint::from_str("1290017141416619832024483521723784417815009599").unwrap();
    let mlog: u32 = 16;
    let polybase: Uint = isqrt(n << 1) >> mlog;
    let polybase = u128::cast_from(isqrt(polybase));
    let (d, r) = sieve_for_polys(&fb, &n, polybase, 512)[0];
    let Poly { a, b, d, .. } = make_poly(&n, &d, &r);
    let (a, b, _) = (Uint::cast_from(a), Uint::cast_from(b), Uint::cast_from(d));

    let target: Uint = isqrt(n << 1) << (mlog - 1);
    let c: Uint = (n - (b * b)) / a;
    eprintln!("n = {n}");
    eprintln!("P = {a}*x^2+{b}*x-{c}");
    let xmax = (a << mlog) + b;
    // ((2A M + B)^2 - n) / 4D^2
    let pmax: Uint = ((xmax * xmax) - n) / a;
    // must match target with good accuracy
    let sz = target.bits();
    eprintln!("min(P) = -{c}");
    eprintln!("max(P) = {pmax}");
    assert_eq!(c >> (sz - 12), target >> (sz - 12));
    assert_eq!(pmax >> (sz - 12), target >> (sz - 12));

    // Sample failures in polynomial selection:
    // n=5*545739830203115604058837931639003
    // D=1064363 composite
    // n=3*936196470328602335308479219639141053
    // D=4255903 composite (n^(D+1)/4 is a valid square root)
    // n=6*1026830418586472562456155798159521
    // D=1302451 base 2 pseudoprime
}

/// A structure holding memory allocations for MPQS.
#[derive(Default)]
struct Workspace {
    roots1: Vec<u32>,
    roots2: Vec<u32>,
    recycled: Option<sieve::SieveRecycle>,
}

/// Process a single MPQS unit of work, corresponding to a polynomial.
fn mpqs_poly(s: &SieveMPQS, a: &Uint, r: &Uint, wks: &mut Workspace) {
    let n = &s.n;
    let pol = make_poly(n, a, r);
    let nblocks = s.interval_size as usize / BLOCK_SIZE;
    if s.prefs.verbose(Verbosity::Debug) {
        let pmin = pol.eval(0).0;
        let pmax = max(
            pol.eval(-s.interval_size / 2).0,
            pol.eval(s.interval_size / 2).0,
        );
        eprintln!(
            "Sieving polynomial A={} B={} M={} blocks={nblocks} min={pmin} max={pmax}",
            pol.a,
            pol.b,
            s.interval_size / 2
        );
    }

    let start_offset = -s.interval_size / 2;
    let end_offset = s.interval_size / 2;
    let fbase = s.fbase;
    wks.roots1.resize(fbase.len(), 0);
    wks.roots2.resize(fbase.len(), 0);
    let roots1 = &mut wks.roots1[..];
    let roots2 = &mut wks.roots2[..];
    for i in 0..fbase.len() {
        let p = fbase.p(i);
        let r = fbase.r(i);
        let div = fbase.div(i);
        let inv = &s.inverters[i];
        let (r1, r2) = pol.prepare_prime(p, r, div, inv, start_offset as i32);
        roots1[i] = r1;
        roots2[i] = r2;
    }
    let roots12 = [roots1.as_ref(), roots2.as_ref()];
    let mut state = sieve::Sieve::new(start_offset, nblocks, fbase, roots12, wks.recycled.take());
    if nblocks == 0 {
        sieve_block_poly(s, &pol, roots12, &mut state);
    }
    while state.offset < end_offset {
        sieve_block_poly(s, &pol, roots12, &mut state);
        state.next_block();
    }
    wks.recycled = Some(state.recycle());
}

// MPQS can use the same factor bases as SIQS but since the polynomial
// initalization cost is higher, we can include a penalty for the larger
// intervals.
fn fb_size(n: &Uint, use_double: bool) -> u32 {
    // When using type 2 polynomials, values will be twice smaller
    // as if the size of n was down by 2 bits.
    let nshift = if n.low_u64() % 8 == 1 { n >> 2 } else { *n };
    let penalty = n.bits() / 32;
    let mut sz = params::factor_base_size(&(nshift << penalty));
    // Reduce factor base size when using large double primes
    // since they will cover the large prime space.
    if use_double {
        sz /= 2;
    }
    sz
}

fn mpqs_interval_size(n: &Uint) -> i64 {
    let sz = n.bits();
    let nblocks = match sz {
        0..=100 => 1,
        // The cost of switching polynomials grows very fast, and it needs to be a small ratio
        // of CPU cost for optimal efficiency.
        // So the interval size needs to grow along with factor base size.
        //
        // For 200 bits n, we need at least 30 blocks.
        // For 240 bits n, we need at least 50 blocks
        // For 320 bits n, we need at least 300 blocks
        101..=128 => sz / 30,           // 3..4
        129..=256 => (sz * sz) / 1000,  // 16..64
        257.. => (sz * sz) / 100 - 550, // 75..500
    };
    nblocks as i64 * sieve::BLOCK_SIZE as i64
}

fn large_prime_factor(n: &Uint) -> u64 {
    // Allow large cofactors up to FACTOR * largest prime
    // Because MPQS needs larger intervals and polynomials are costly,
    // large primes are useful even for 96-bit integers.
    let sz = n.bits();
    match sz {
        0..=92 => 1,
        93..=128 => n.bits() as u64 / 32, // 2..4
        129..=250 => n.bits() as u64 - 100,
        251.. => {
            // Bound large primes to avoid exceeding 32 bits.
            50 + n.bits() as u64 / 2
        }
    }
}

struct SieveMPQS<'a> {
    n: Uint,
    fbase: &'a FBase,
    inverters: &'a [arith::Inverter],
    maxlarge: u64,
    use_double: bool,
    interval_size: i64,
    rels: &'a RwLock<RelationSet>,
    prefs: &'a Preferences,
}

// Sieve using a selected polynomial
fn sieve_block_poly(s: &SieveMPQS, pol: &Poly, roots: [&[u32]; 2], st: &mut sieve::Sieve) {
    st.sieve_block();

    let maxprime = s.fbase.bound() as u64;
    let maxlarge = s.maxlarge;
    assert!(maxlarge == (maxlarge as u32) as u64);
    let max_cofactor: u64 = if s.use_double {
        // We don't want double large prime to reach maxlarge^2
        // See siqs.rs
        maxlarge * maxprime * 2
    } else if maxlarge > maxprime {
        // Use the large prime variation
        maxlarge
    } else {
        // No large prime at all (actually 1 but increase to have some tolerance)
        1
    };
    let target = s.n.bits() / 2 + (s.interval_size as u64 / 2).bits() - max_cofactor.bits();
    let n = &s.n;
    let root = s.interval_size * 7 / 20; // sqrt(2)/4
    let (idxs, facss) = st.smooths(target as u8, Some(root as u32), roots);
    for (i, facs) in idxs.into_iter().zip(facss) {
        // Evaluate polynomial
        let (v, mut x) = pol.eval(st.offset + i as i64);
        debug_assert!((x * x) % n == Uint::cast_from(Int::cast_from(*n) + Int::cast_from(v)) % n);
        let Some(((p, q), factors)) = fbase::cofactor(
            s.fbase, &v, &facs,
            maxlarge, s.use_double)
            else { continue };
        let pq = if q > 1 { Some((p, q)) } else { None };
        let cofactor = p * q;
        if p > 1 {
            debug_assert!(!fbase::certainly_composite(p), "p={p} {pol:?}");
        }
        if &x > n {
            x %= n; // should not happen?
        }
        if s.prefs.verbose(Verbosity::Debug) {
            eprintln!(
                "x={} sqrt={x} smooth {v} cofactor {cofactor}",
                st.offset + i as i64
            );
        }
        let rel = Relation {
            x,
            cofactor,
            factors,
            cyclelen: 1,
        };
        debug_assert!(rel.verify(n));
        s.rels.write().unwrap().add(rel, pq);
    }
}
