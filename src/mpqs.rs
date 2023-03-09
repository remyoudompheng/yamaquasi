// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Multiple Polynomial Quadratic Sieve with large prime A
//!
//! Bibliography:
//! Robert D. Silverman, The multiple polynomial quadratic sieve
//! Math. Comp. 48, 1987, <https://doi.org/10.1090/S0025-5718-1987-0866119-8>

use std::cmp::{max, min};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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
            eprintln!("Number {n} too large for quadratic sieve!");
        }
        return vec![];
    }

    // Use double large primes sooner than SIQS:
    // it allows smaller factor bases, speeding up roots computation.
    let use_double = prefs.use_double.unwrap_or(n.bits() > 224);
    // Choose factor base. Sieve twice the number of primes
    // (n will be a quadratic residue for only half of them)
    let fb = prefs
        .fb_size
        .unwrap_or(params::mpqs_fb_size(norig.bits(), use_double));
    let fbase = FBase::new(n, fb);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Smoothness bound {}", fbase.bound());
        eprintln!("Factor base size {} ({:?})", fbase.len(), fbase.smalls());
    }
    let fb = fbase.len();
    let mm = prefs.interval_size.unwrap_or(mpqs_interval_size(&n) as u32);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Sieving interval size {}k", mm >> 10);
    }

    // Precompute inverters: preparation of polynomials is almost entirely spent
    // in modular inversion.
    let inverters: Vec<_> = (0..fb)
        .map(|idx| arith::Inverter::new(fbase.p(idx)))
        .collect();

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
    // Generate multiple polynomials at a time.
    // For small numbers (90-140 bits) usually less than
    // a hundred polynomials will provide enough relations.
    // We multiply by 1.4 log2(n) which is the expected gap between
    // 2 solutions.
    let polystride = match n.bits() {
        // Interval size is larger than sqrt(n)
        0..=32 => 200,
        // Arrange so that each block contains at least a prime p=4k+3
        // hoping that we don't hit a huge prime gap.
        33..=256 => 50 * 20 / 7 * d_target.bits(),
        257.. => 200 * 20 / 7 * d_target.bits(),
    };

    // Start slightly before the ideal values to get a few nice polynomials.
    let d_target = u128::cast_from(d_target);
    let mut polybase = d_target;
    if polybase >= 20 {
        polybase -= min(polybase / 10, polystride.into());
    }
    // Polynomials will be processed in blocks of size polystride
    // (50 primes on average). In multi-threaded mode, threads will process
    // blocks in a round-robin fashion.

    let maxprime = fbase.bound() as u64;
    let mut maxlarge: u64 = maxprime * prefs.large_factor.unwrap_or(large_prime_factor(&n));
    if maxlarge > u32::MAX as u64 {
        maxlarge = u32::MAX as u64;
    }
    if use_double && maxlarge < 2 * maxprime {
        // Double large prime implies large prime
        maxlarge = 2 * maxprime
    }
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Max large prime B2={maxlarge} ({} B1)", maxlarge / maxprime);
        if use_double {
            let maxdouble = maxprime * maxprime * double_large_factor(&n);
            eprintln!(
                "Max double large prime {maxdouble} ({} B1²)",
                maxdouble / maxprime / maxprime
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
        d_target,
        rels: &rels,
        prefs,
        polys_done: AtomicUsize::new(0),
        target: AtomicUsize::new(fb * 8 / 10),
        done: AtomicBool::new(false),
    };

    fn process_poly_block(s: &SieveMPQS, wks: &mut Workspace, dbase: u128, dstride: usize) {
        let d_r_values = sieve_for_polys(&s.n, dbase, dstride);
        if s.prefs.verbose(Verbosity::Verbose) {
            eprintln!(
                "Generated {} polynomials D={}..{} optimal={}",
                d_r_values.len(),
                d_r_values[0].0,
                d_r_values.last().unwrap().0,
                s.d_target
            );
        }
        for chunk in d_r_values.chunks(16) {
            wks.batch_inversion(&s, chunk.iter().map(|&(d, _)| d).collect());
            for (idx, (d, r)) in chunk.iter().enumerate() {
                mpqs_poly(&s, idx, *d, r, wks);
                if s.finished() {
                    return;
                }
            }
        }
        if s.prefs.verbose(Verbosity::Info) {
            let polys_done = s.polys_done.load(Ordering::SeqCst) as u64;
            s.rels.read().unwrap().log_progress(format!(
                "Sieved {}M {polys_done} polys",
                (polys_done * s.interval_size as u64) >> 20,
            ));
        }
    }

    if let Some(pool) = tpool {
        // FIXME: We shouldn't have such a value: iterate and stop when finished.
        let maxblocks = if n.bits() < 256 { 100_000 } else { 1_000_000 };
        pool.install(|| {
            (0..maxblocks).into_par_iter().for_each(|blkno| {
                if s.finished() || prefs.abort() {
                    return;
                }
                let dbase = polybase + blkno as u128 * polystride as u128;
                // Workspace is reused during the entire block.
                let mut wks = Workspace::default();
                process_poly_block(&s, &mut wks, dbase, polystride as usize);
            })
        })
    } else {
        let mut wks = Workspace::default();
        for blkno in 0.. {
            let dbase = polybase + blkno as u128 * polystride as u128;
            process_poly_block(&s, &mut wks, dbase, polystride as usize);
            if prefs.abort() || s.finished() {
                break;
            }
        }
    }
    if prefs.abort() {
        return vec![];
    }
    let polys_done = s.polys_done.load(Ordering::SeqCst) as u64;
    let mut rels = rels.into_inner().unwrap();
    if prefs.verbose(Verbosity::Info) {
        rels.log_progress(format!(
            "Sieved {}M {polys_done} polys",
            (polys_done * mm as u64) >> 20
        ));
    }
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
    pub d: u128,
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
        // The modular inverse of self.d mod p.
        dinv: u32,
        offset: i32,
    ) -> (u32, u32) {
        // Could be precomputed/handled as 32-bit.
        let off: u32 = div.modi64(offset as i64) as u32;
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
            // A=D^2 so it is faster to reduce D rather than A.
            if dinv == 0 {
                // For very small integers, we may select D inside the factor base.
                // In this case the roots are the roots of Bx-abs(C) (C < 0)
                let b = div.mod_uint(&self.b);
                let binv = inv.invert(b as u32, &div) as u64;
                debug_assert!(self.c.is_negative());
                let c = div.mod_uint(&self.c.abs().to_bits());
                let r = shift(div.divmod64(c * binv).1 as u32);
                (r, r)
            } else {
                let d2inv = div.modu63(dinv as u64 * dinv as u64);
                let (ainv, b) = if self.b.bit(0) {
                    // We need 1/2D^2
                    (
                        if d2inv & 1 == 0 {
                            d2inv >> 1
                        } else {
                            (d2inv + p as u64) >> 1
                        },
                        div.mod_uint(&self.b),
                    )
                } else {
                    (d2inv, div.mod_uint(&self.bb))
                };
                let r1 = shift(div.modu63((p as u64 + r as u64 - b) * ainv) as u32);
                let r2 = shift(div.modu63((2 * p as u64 - r as u64 - b) * ainv) as u32);
                (r1, r2)
            }
        }
    }
}

/// Find appropriate pseudoprime D such that A=D² is a good polynomial coefficient.
/// The goal is to be able to compute a modular square root of n mod D.
#[doc(hidden)]
pub fn sieve_for_polys(n: &Uint, bmin: u128, width: usize) -> Vec<(u128, Uint)> {
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
            if let Err(gcd) = inv_mod(&nmodp, &p256) {
                // FIXME: use this factor to answer.
                eprintln!("WARNING: unexpectedly found a factor of N!");
                eprintln!("D={p} gcd(D,n)={gcd}");
                continue 'nextsieve;
            }
            if (r * r) % p256 == nmodp {
                // Beware, D may (exceptionally) not be prime, but this is fine
                // as long as we found a square root of n.
                // It usually happens for small n (100-120 bits), with low probability.
                // It can also be a member of the factor base, which is also fine.
                result.push((p, Uint::cast_from(r)));
                if crate::DEBUG && !crate::pseudoprime(p.into()) {
                    eprintln!("D={p} is not prime");
                }
            }
        }
    }
    result
}

/// Construct a MPQS polynomial from modulus N, a prime D and a modular square root D.
#[doc(hidden)]
pub fn make_poly(n: &Uint, d: u128, r: &Uint) -> Poly {
    let d128 = d;
    let d = Uint::cast_from(d);
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
            d: d128,

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
            d: d128,

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
    let mlog: u32 = 24;
    let polybase: Uint = isqrt(n >> 1) >> mlog;
    let polybase = u128::cast_from(isqrt(polybase));
    let (d, r) = sieve_for_polys(&n, polybase, 512)[0];
    let pol = make_poly(&n, d, &r);
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
    let (d, r) = sieve_for_polys(&n, polybase, 512)[0];
    let Poly { a, b, d, .. } = make_poly(&n, d, &r);
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
}

/// A structure holding memory allocations for MPQS.
#[derive(Default)]
struct Workspace {
    roots1: Vec<u32>,
    roots2: Vec<u32>,
    recycled: Option<sieve::SieveRecycle>,
    /// Precomputed inverses of D modulo the factor base.
    /// This is done for several D's at a time to benefit
    /// from batch inversion.
    /// If D is divisible by p, the value is zero.
    dinv_modp: Vec<Box<[u32]>>,
}

impl Workspace {
    fn batch_inversion(&mut self, s: &SieveMPQS, ds: Vec<u128>) {
        assert!(!ds.is_empty());
        let len = s.fbase.len();
        for i in 0..ds.len() {
            if i >= self.dinv_modp.len() {
                self.dinv_modp.push(vec![0; len].into_boxed_slice());
            } else {
                assert!(self.dinv_modp[i].len() == len);
            }
        }
        let mut prods = vec![0; ds.len()];
        let mut dmod = vec![0; ds.len()];
        for i in 0..len {
            // The overhead of batch inversion is rather large
            // (3 modular multiplications and vector allocations)
            let div = s.fbase.div(i);
            let inv = &s.inverters[i];
            // Compute cumulative products (excluding zeros).
            let mut prod = 1;
            unsafe {
                // This is a hot spot, skip bound checks.
                for j in 0..ds.len() {
                    let dm = div.mod_u128(*ds.get_unchecked(j));
                    *dmod.get_unchecked_mut(j) = dm;
                    if dm != 0 {
                        prod = div.modu63(prod * dm);
                    }
                    *prods.get_unchecked_mut(j) = prod;
                }
                let invprod = inv.invert(prod as u32, &div);
                // 1/d[i] = product(j < i, d[j]) * product(j > i, d[j]) * invprod
                let mut prodrev = invprod as u64;
                for j in 0..ds.len() {
                    let jrev = ds.len() - 1 - j;
                    let dm = *dmod.get_unchecked(jrev);
                    let out = self.dinv_modp.get_unchecked_mut(jrev).get_unchecked_mut(i);
                    if dm == 0 {
                        *out = 0;
                        continue;
                    }
                    if jrev > 0 {
                        // prodrev = product(j > i, d[j]) * invprod
                        *out = div.modu63(prodrev * *prods.get_unchecked(jrev - 1)) as u32;
                        prodrev = div.modu63(prodrev * dm);
                    } else {
                        *out = prodrev as u32;
                    };
                    debug_assert!(
                        div.modu63(*out as u64 * dm) == 1,
                        "dmod={dmod:?} p={} j={jrev} d={dm} dinv={}",
                        s.fbase.p(i),
                        *out
                    );
                }
            }
        }
    }
}

/// Process a single MPQS unit of work, corresponding to a polynomial.
///
/// The idx is the location of the precomputed inverses of D.
fn mpqs_poly(s: &SieveMPQS, idx: usize, d: u128, r: &Uint, wks: &mut Workspace) {
    let n = &s.n;
    let pol = make_poly(n, d, r);
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
    let dinvs = wks.dinv_modp[idx].as_ref();
    for i in 0..fbase.len() {
        let p = fbase.p(i);
        let r = fbase.r(i);
        let div = fbase.div(i);
        let inv = &s.inverters[i];
        let dinv = dinvs[i];
        let (r1, r2) = pol.prepare_prime(p, r, div, inv, dinv, start_offset as i32);
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
    s.polys_done.fetch_add(1, Ordering::SeqCst);
    wks.recycled = Some(state.recycle());
}

/// Interval size used during MPQS.
fn mpqs_interval_size(n: &Uint) -> i64 {
    let sz = n.bits();
    let nblocks = match sz {
        0..=100 => 1,
        // The cost of switching polynomials grows with the factor base,
        // but small intervals have a higher probability to yield smooth numbers.
        // A good balance is obtained when the CPU cost of polynomial roots is
        // between 10% and 30%.
        101..=129 => 1,
        130..=189 => 2 + (sz - 130) / 10, // 64k + 32k every 10 bits
        190..=219 => 8 + 8 * (sz - 190) / 30, // 256k..512k
        220..=259 => 16 + 2 * (sz - 220) / 10, // 512k..768k (transition to double large primes)
        260..=289 => 24 + 8 * (sz - 260) / 10, // 768k then 8 blocks every 10 bits.
        290..=319 => 48 + 16 * (sz - 290) / 10, // 1536k, then 16 blocks every 10 bits.
        320.. => 96 + 32 * (sz - 320) / 10, // 3M, then 32 blocks every 10 bits.
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

// Parameter such that double large primes are bounded by D*B1^2
// where B1 is the factor base bound.
fn double_large_factor(n: &Uint) -> u64 {
    let sz = n.bits() as u64;
    match sz {
        0..=128 => 2,
        129..=255 => 2 + (sz - 128) / 2,
        256.. => sz - 192,
    }
}

struct SieveMPQS<'a> {
    n: Uint,
    fbase: &'a FBase,
    inverters: &'a [arith::Inverter],
    maxlarge: u64,
    use_double: bool,
    interval_size: i64,
    d_target: u128,
    rels: &'a RwLock<RelationSet>,
    prefs: &'a Preferences,
    polys_done: AtomicUsize,
    target: AtomicUsize,
    done: AtomicBool,
}

impl SieveMPQS<'_> {
    fn finished(&self) -> bool {
        // The relaxed memory ordering is fine, it's okay to do
        // some extra work if threads don't fully synchronize.
        if self.done.load(Ordering::Relaxed) {
            return true;
        }
        let relcount = { self.rels.read().unwrap().len() };
        if relcount >= self.target.load(Ordering::Relaxed) {
            let gap = { self.rels.read().unwrap().gap(self.fbase) };
            if gap == 0 {
                if self.prefs.verbose(Verbosity::Info) {
                    eprintln!("Found enough relations");
                }
                self.done.store(true, Ordering::Relaxed);
                return true;
            } else {
                if self.prefs.verbose(Verbosity::Info) {
                    eprintln!("Need {} additional relations", gap);
                }
                self.target.store(
                    relcount + gap + std::cmp::min(10, self.fbase.len() / 4),
                    Ordering::Relaxed,
                );
            }
        }
        false
    }
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
        maxprime * maxprime * double_large_factor(&s.n)
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
