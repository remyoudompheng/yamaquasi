// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Class group computation using Jacobson's quadratic sieve method.
//!
//! The computations of class groups uses the same computation as
//! factorization, applied to a negative number D instead of a positive N.
//!
//! The differences with factorization are:
//! - we do not allow a multiplier
//! - for each relation, the exact prime decomposition of the binary
//!   quadratic form which produced it, corresponding to the choice of signs
//!   in square root modulo A
//! - the binary forms are positive definite so the range of values
//!   is [sqrt(D) M, 2 sqrt(D) M] and they are never close to zero
//! - the linear algebra is over the integers so relations must be exact
//!   and the factor base must remain small
//! - it is not enough to have more relations than the factor base size
//!   We usually need extra relations to obtain the entire lattice.
//!
//! Bibliography:
//! Michael Jacobson, Applying sieving to the computation of class groups
//! Math. Comp. 68 (226), 1999, 859-867
//! <https://www.ams.org/journals/mcom/1999-68-226/S0025-5718-99-01003-0/S0025-5718-99-01003-0.pdf>

use std::cmp::{max, min};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::RwLock;

use rayon::prelude::*;

use crate::arith::Num;
use crate::fbase::{self, FBase};
use crate::params::clsgrp_fb_size;
use crate::relationcls::{CRelation, CRelationSet};
use crate::sieve::{self, BLOCK_SIZE};
use crate::siqs::{self, prepare_a, select_a, select_siqs_factors, Factors, Poly, PolyType, A};
use crate::{Int, Preferences, Uint, Verbosity};

pub fn ideal_relations(d: &Int, prefs: &Preferences, tpool: Option<&rayon::ThreadPool>) -> () {
    let dabs = d.unsigned_abs();
    let use_double = prefs.use_double.unwrap_or(dabs.bits() > 180);
    // Choose factor base. Sieve twice the number of primes
    // (n will be a quadratic residue for only half of them)
    let fb = prefs
        .fb_size
        .unwrap_or(clsgrp_fb_size(dabs.bits(), use_double));
    // WARNING: SIQS doesn't use 4D if D=3 mod 4
    // This creates some redundant *2 /2 operations.
    let dred = if dabs.low_u64() & 3 == 0 { *d >> 2 } else { *d };
    let fbase = FBase::new(dred, fb);
    // It is fine to have a divisor of D in the factor base.
    let mm = prefs.interval_size.unwrap_or(interval_size(d, use_double));
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Smoothness bound B1={}", fbase.bound());
        eprintln!("Factor base size {} ({:?})", fbase.len(), fbase.smalls(),);
        eprintln!("Sieving interval size {}k", mm >> 10);
    }

    // Generate all values of A now.
    let nfacs = siqs::nfactors(&dabs) as usize;
    let factors = select_siqs_factors(&fbase, &dred, nfacs, mm as usize, prefs.verbosity);
    let a_count = a_value_count(&dabs);
    let a_ints = select_a(&factors, a_count, prefs.verbosity);
    let polys_per_a = 1 << (nfacs - 1);
    if prefs.verbose(Verbosity::Info) {
        eprintln!(
        "Generated {} values of A with {} factors in {}..{} ({} polynomials each, spread={:.2}%)",
        a_ints.len(),
        nfacs,
        factors.factors[0].p,
        factors.factors.last().unwrap().p,
        polys_per_a,
        siqs::a_quality(&a_ints) * 100.0
    );
    }

    let maxprime = fbase.bound() as u64;
    let maxlarge: u64 = maxprime * prefs.large_factor.unwrap_or(large_prime_factor(&d));
    // Don't allow maxlarge to exceed 32 bits (it would not be very useful anyway).
    let maxlarge = min(maxlarge, (1 << 32) - 1);
    let maxdouble = if use_double {
        maxprime * maxprime * large_prime_factor(&d)
    } else {
        0
    };
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Max large prime B2={maxlarge} ({} B1)", maxlarge / maxprime);
        if use_double {
            eprintln!(
                "Max double large prime {maxdouble} ({} B1^2)",
                maxdouble / maxprime / maxprime
            );
        }
    }
    // WARNING: use reduced D again.
    let qs = siqs::SieveSIQS::new(dred, &fbase, maxlarge, maxdouble, mm as usize, prefs);
    let target_rels = qs.fbase.len() * max(1, dabs.bits() as usize / 30) + 64;
    let s = ClSieve {
        d: *d,
        qs,
        prefs,
        rels: RwLock::new(CRelationSet::new(*d, target_rels, maxlarge as u32)),
        done: AtomicBool::new(false),
        polys_done: AtomicUsize::new(0),
    };
    // When using multiple threads, each thread will sieve a different A
    // to avoid breaking parallelism during 'prepare_a'.
    //
    // To avoid wasting CPU on very small inputs, completion is checked after
    // each polynomial to terminate the loop early.

    if let Some(pool) = tpool.as_ref() {
        pool.install(|| {
            a_ints.par_iter().for_each(|&a_int| {
                if s.done.load(Ordering::Relaxed) || prefs.abort() {
                    return;
                }
                sieve_a(&s, &a_int, &factors);
            });
        });
    } else {
        for a_int in a_ints {
            sieve_a(&s, &a_int, &factors);
            if s.done.load(Ordering::Relaxed) || prefs.abort() {
                break;
            }
        }
    }
    if prefs.abort() {
        return;
    }
    // FIXME: No linear algebra?
    let pdone = s.polys_done.load(Ordering::Relaxed);
    let mm = s.qs.interval_size;
    let rels = s.rels.read().unwrap();
    if s.prefs.verbose(Verbosity::Info) {
        rels.log_progress(format!(
            "Sieved {}M {pdone} polys",
            (pdone as u64 * mm as u64) >> 20,
        ));
    }
}

struct ClSieve<'a> {
    // A negative discriminant
    d: Int,
    qs: siqs::SieveSIQS<'a>,
    // A signal for threads to stop sieving.
    done: AtomicBool,
    rels: RwLock<CRelationSet>,
    // Progress trackers
    polys_done: AtomicUsize,
    prefs: &'a Preferences,
}

// We need more values of A for class field computations
// than for factoring.
fn a_value_count(n: &Uint) -> usize {
    // Many polynomials are required to accomodate small intervals.
    // When sz=180 we need more than 5k polynomials
    // When sz=200 we need more than 20k polynomials
    // When sz=280 we need more than 1M polynomials
    // When sz=360 we need more than 20M polynomials
    let sz = n.bits() as usize;
    match sz {
        // Even one A value (2-4 polynomials) will give enough smooth values.
        0..=48 => 8,
        49..=71 => 12,
        // We are using small intervals until 200 bits, many As are needed.
        72..=150 => 10 * (sz - 71),    // 10..800
        151..=199 => 40 * (sz - 131),  // 800..2800
        200..=255 => 300 * (sz - 190), // 3000..20000 (sz=256)
        256..=400 => 800 * (sz - 225), // 20000.. 60000 (sz=300)
        _ => unreachable!("impossible"),
    }
}

// Interval size is similar to SIQS.
// Factor base is smaller, shrink
fn interval_size(n: &Int, use_double: bool) -> u32 {
    let sz = n.unsigned_abs().bits();
    let nblocks = match sz {
        0..=180 => 4,
        181..=255 => (sz - 141) / 20,
        256..=340 => {
            if use_double {
                (sz - 176) / 20
            } else {
                (sz - 176) / 20 + 1
            }
        }
        341.. => 8 + sz / 100,
    };
    nblocks * 32768
}

fn large_prime_factor(n: &Int) -> u64 {
    let sz = n.unsigned_abs().bits() as u64;
    match sz {
        0..=160 => sz,
        161.. => 2 * sz - 160,
    }
}

fn sieve_a(s: &ClSieve, a_int: &Uint, factors: &Factors) {
    let mm = s.qs.interval_size;
    let a = &prepare_a(factors, a_int, s.qs.fbase, -(mm as i64) / 2);
    if s.prefs.verbose(Verbosity::Debug) {
        eprintln!("Sieving A={}", a.description());
    }
    let nfacs = a.len();
    let polys_per_a = 1 << (nfacs - 1);
    let mut pol = Poly::first(&s.qs, a);
    // Storage for recycled resources.
    let mut recycled = None;
    for idx in 0..polys_per_a {
        if s.done.load(Ordering::Relaxed) {
            // Interrupt early.
            return;
        }
        if idx > 0 {
            pol.next(&s.qs, a);
        }
        //assert!(pol.idx == idx);
        recycled = Some(siqs_sieve_poly(s, a, &pol, recycled));
        // Check status.
        s.polys_done.fetch_add(1, Ordering::SeqCst);
        if s.rels.read().unwrap().done() {
            s.done.store(true, Ordering::Relaxed);
        }
    }
    let pdone = s.polys_done.load(Ordering::Relaxed);
    if s.prefs.verbose(Verbosity::Info) {
        let rels = s.rels.read().unwrap();
        rels.log_progress(format!(
            "Sieved {}M {} polys",
            (pdone as u64 * mm as u64) >> 20,
            pdone,
        ));
    }
}

fn siqs_sieve_poly(
    s: &ClSieve,
    a: &A,
    pol: &Poly,
    rec: Option<sieve::SieveRecycle>,
) -> sieve::SieveRecycle {
    let mm = s.qs.interval_size;
    let nblocks: usize = mm / BLOCK_SIZE;
    if s.prefs.verbose(Verbosity::Debug) {
        eprintln!(
            "Sieving polynomial {} M={}k blocks={}",
            pol.description(),
            mm / 2048,
            nblocks
        );
    }
    // Construct initial state.
    let start_offset: i64 = -(mm as i64) / 2;
    let end_offset: i64 = (mm as i64) / 2;
    let r1p = &pol.r1p[..];
    let r2p = &pol.r2p[..];
    let mut state = sieve::Sieve::new(start_offset, nblocks, s.qs.fbase, [r1p, r2p], rec);
    if nblocks == 0 {
        sieve_block_poly(s, pol, a, &mut state);
    }
    while state.offset < end_offset {
        sieve_block_poly(s, pol, a, &mut state);
        state.next_block();
    }
    state.recycle()
}

// Sieve using a selected polynomial
fn sieve_block_poly(s: &ClSieve, pol: &Poly, a: &A, st: &mut sieve::Sieve) {
    st.sieve_block();
    // print!("{:?}", st.blk);
    let qs = &s.qs;
    let maxprime = qs.fbase.bound() as u64;
    let maxlarge = qs.maxlarge;
    assert!(maxlarge == (maxlarge as u32) as u64);
    let max_cofactor: u64 = if qs.maxdouble > maxprime * maxprime {
        qs.maxdouble
    } else if maxlarge > maxprime {
        maxlarge
    } else {
        1 // Do not use the large prime variation.
    };
    // Polynomial values range from [-m sqrt(2n), m sqrt(2n)] so they have variable size.
    // The smallest values are about A which is sqrt(2n) / m
    // If the target is too low, the sieve will be slow.
    let msize = if pol.kind == siqs::PolyType::Type1 {
        qs.interval_size as u64 / 2
    } else {
        // Values are smaller (M/2 sqrt(n)) for type 2.
        qs.interval_size as u64 / 4
    };
    let target = s.d.unsigned_abs().bits() / 2 + msize.bits() - max_cofactor.bits();

    let (idx, facss) = st.smooths(target as u8, None, [&pol.r1p, &pol.r2p]);
    let qfacs = pol.factors(a);
    for (i, intfacs) in idx.into_iter().zip(facss) {
        let x = st.offset + (i as i64);
        let (v, bx) = pol.eval(x);
        debug_assert!(v.is_positive());
        // If a binary form q=(A, B, C) represents v, then
        // (Bx=2AX+B)^2 - 4AV = -D and q is equivalent to (V, -Bx, A)
        // and (A, B, C) * (V, Bx, A) == 1 in the class group
        //
        // If the discriminant is 4D, the quadratic form is
        // q = (A, 2B, C) where B^2-AC=D and q(x)=V
        // then (2Ax+2B)^2-4AV = 4D and q is equivalent to
        // (V, -2Ax-2B, A)
        let Some(((p, q), intfacs)) = fbase::cofactor(
            qs.fbase,
            &v,
            &intfacs,
            maxlarge,
            qs.maxdouble > maxprime * maxprime,
        ) else {
            continue;
        };
        // We only accept large primes under 32 bits.
        if p >> 32 > 0 || q >> 32 > 0 {
            continue;
        }
        // Convert integer factors to ideal factors
        // product(ai^±1) = product(pi^ei)
        //
        // This is done by comparing bx % p
        // with the reference square root of D mod p
        // (with the same parity as D).

        let mut factors = vec![];
        for (p, e) in intfacs {
            if p == 2 {
                // Positive exponent if b % 4 = 1
                // Note that if D=4D', ideal class [2] has order 2.
                if !bx.bit(1) {
                    factors.push((2, e as i32))
                } else {
                    factors.push((2, -(e as i32)))
                }
                continue;
            }
            let pr = qs.fbase.prime(qs.fbase.idx(p as u32).unwrap());
            debug_assert!(pr.p == p as u64);
            let ref_bp = pr.b_plus(pol.kind == PolyType::Type1);
            let mut bp = pr.div.mod_uint(&bx.unsigned_abs());
            if bx.is_negative() && bp > 0 {
                bp = p as u64 - bp;
            }
            let e = if bp == ref_bp {
                e as i32
            } else {
                debug_assert!(bp == p as u64 - ref_bp);
                -(e as i32)
            };
            factors.push((p as u32, e));
        }
        for &(f, e) in &qfacs {
            if let Some(idx) = factors.iter().position(|&(p, _)| p as u64 == f) {
                factors[idx].1 += e as i32;
            } else {
                factors.push((f as u32, e as i32));
            }
        }
        // Also compute exponents for large primes
        let parity = if pol.kind == PolyType::Type1 { 0 } else { 1 };
        let large1 = if p > 1 {
            let mut bp = bx.unsigned_abs() % p;
            if bx.is_negative() && bp != 0 {
                bp = p - bp;
            }
            if bp % 2 == parity {
                Some((p as u32, if q == p { 2 } else { 1 }))
            } else {
                Some((p as u32, if q == p { -2 } else { -1 }))
            }
        } else {
            None
        };
        let large2 = if q > 1 && q != p {
            let mut bq = bx.unsigned_abs() % q;
            if bx.is_negative() && bq != 0 {
                bq = q - bq;
            }
            if bq % 2 == parity {
                Some((q as u32, 1))
            } else {
                Some((q as u32, -1))
            }
        } else {
            None
        };
        let rel = CRelation {
            factors,
            large1,
            large2,
        };
        s.rels.write().unwrap().add(rel);
    }
}
