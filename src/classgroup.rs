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
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::RwLock;

use num_traits::ToPrimitive;
use rayon::prelude::*;

use crate::arith::{Dividers, Num};
use crate::fbase::{self, FBase};
use crate::params::clsgrp_fb_size;
use crate::relationcls::{CRelation, CRelationSet, ClassGroup};
use crate::sieve::{self, BLOCK_SIZE};
use crate::siqs::{self, prepare_a, select_a, select_siqs_factors, Factors, Poly, PolyType, A};
use crate::{Int, Preferences, Uint, Verbosity};

pub fn classgroup(
    d: &Int,
    prefs: &Preferences,
    tpool: Option<&rayon::ThreadPool>,
) -> Option<ClassGroup> {
    let (hmin, hmax) = estimate(d);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Estimate by class number formula {hmin:.5e}-{hmax:.5e}")
    }
    if let Some(outdir) = prefs.outdir.as_ref() {
        std::fs::create_dir_all(outdir).unwrap();
        let mut json: Vec<u8> = vec![];
        writeln!(&mut json, "{{").unwrap();
        writeln!(&mut json, r#"  "d": "{d}","#).unwrap();
        writeln!(&mut json, r#"  "h_estimate_min": {hmin},"#).unwrap();
        writeln!(&mut json, r#"  "h_estimate_max": {hmax}"#).unwrap();
        writeln!(&mut json, "}}").unwrap();
        std::fs::write(PathBuf::from(outdir).join("args.json"), json).unwrap();
    }

    let dabs = d.unsigned_abs();
    // Choose factor base. Estimate the smoothness bias to increase/decrease
    // the factor base accordingly.
    let bias = smoothness_bias(d);
    let adjsize = max(1, dabs.bits() as i64 - (2.5 * bias).round() as i64) as u32;
    let use_double = prefs.use_double.unwrap_or(adjsize > 180);
    let fb = prefs.fb_size.unwrap_or(clsgrp_fb_size(adjsize, use_double));
    // WARNING: SIQS doesn't use 4D if D=3 mod 4
    // This creates some redundant *2 /2 operations.
    let dred = if dabs.low_u64() & 3 == 0 { *d >> 2 } else { *d };
    let fbase = FBase::new(dred, fb);
    let mut conductor_primes = vec![];
    for idx in 0..fbase.len() {
        let pr = fbase.prime(idx);
        if pr.r == 0 && dred.unsigned_abs() % (pr.p * pr.p) == 0 {
            if prefs.verbose(Verbosity::Info) {
                eprintln!(
                    "WARNING: D is not fundamental, {} divides the conductor",
                    pr.p
                );
            }
            conductor_primes.push(pr.p);
        }
    }
    // It is fine to have a divisor of D in the factor base.
    let mm = prefs.interval_size.unwrap_or(interval_size(adjsize));
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Smoothness bias {bias:.3} using parameters for {adjsize} bits");
        eprintln!("Smoothness bound B1={}", fbase.bound());
        eprintln!("Factor base size {} ({:?})", fbase.len(), fbase.smalls(),);
        eprintln!("Sieving interval size {}k", mm >> 10);
    }

    // Generate all values of A now.
    let (a_count, nfacs) = a_params(adjsize);
    let factors = select_siqs_factors(&fbase, &dred, nfacs as usize, mm as usize, prefs.verbosity);
    let a_ints = select_a(&factors, a_count as usize, prefs.verbosity);
    let polys_per_a = if nfacs > 1 { 1 << (nfacs - 1) } else { 1 };
    if prefs.verbose(Verbosity::Info) && nfacs > 0 {
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
    assert!(a_ints.len() >= a_count as usize);

    let maxprime = fbase.bound() as u64;
    let maxlarge: u64 = maxprime * prefs.large_factor.unwrap_or(large_prime_factor(adjsize));
    // Don't allow maxlarge to exceed 32 bits (it would not be very useful anyway).
    let maxlarge = min(maxlarge, (1 << 32) - 1);
    let maxdouble = if use_double {
        maxprime * maxprime * double_large_factor(&d)
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
    let target_rels = if qs.fbase.len() < 100 {
        qs.fbase.len() + 80
    } else {
        qs.fbase.len() * max(2, (dabs.bits() as isize - 100) / 20) as usize
    };
    let relfilepath = prefs
        .outdir
        .as_ref()
        .map(|p| PathBuf::from(p).join("relations.sieve"));
    let s = ClSieve {
        d: *d,
        qs,
        conductor_primes,
        prefs,
        rels: RwLock::new(CRelationSet::new(
            *d,
            target_rels,
            maxlarge as u32,
            relfilepath,
        )),
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
        return None;
    }
    let pdone = s.polys_done.load(Ordering::Relaxed);
    let mm = s.qs.interval_size;
    let rels = s.rels.read().unwrap();
    if s.prefs.verbose(Verbosity::Info) {
        rels.log_progress(format!(
            "Sieved {}M {pdone} polys",
            (pdone as u64 * mm as u64) >> 20,
        ));
    }
    if rels.len() < rels.target {
        panic!("not enough polynomials to sieve")
    }
    drop(rels);
    let use_sparse = Some(hmax.log2() >= 128.0 || fbase.len() > SPARSE_LINALG_THRESHOLD);
    let crels = s.result();
    use crate::relationcls;
    let outdir = prefs.outdir.as_ref().map(PathBuf::from);
    relationcls::group_structure(
        crels,
        use_sparse,
        (hmin, hmax),
        prefs.verbosity,
        outdir,
        tpool,
    )
}

const SPARSE_LINALG_THRESHOLD: usize = 800;

struct ClSieve<'a> {
    // A negative discriminant
    d: Int,
    qs: siqs::SieveSIQS<'a>,
    // Relations involving conductor primes will be rejected.
    conductor_primes: Vec<u64>,
    // A signal for threads to stop sieving.
    done: AtomicBool,
    rels: RwLock<CRelationSet>,
    // Progress trackers
    polys_done: AtomicUsize,
    prefs: &'a Preferences,
}

impl<'a> ClSieve<'a> {
    fn result(self) -> Vec<CRelation> {
        self.rels.into_inner().unwrap().to_vec()
    }
}

// Determine the number of A values and the number of factors
// necessary for the class group computation.
// The total number of ideals will be values * 2^(factors-1)
fn a_params(sz: u32) -> (u32, u32) {
    match sz {
        // sqrt(|D|)/M is less than 1
        // The optimal polynomial is the unit binary form.
        0..=32 => (1, 0),
        // We can use non-unit ideals if D is large enough
        33..=64 => (8, 2),
        65..=80 => (2 * (sz - 60), 3),
        81..=99 => (10 * (sz - 77), 3),
        100..=119 => (40 * (sz - 95), 4),   // 200..1000 x8
        120..=149 => (50 * (sz - 100), 5),  // 1000..2500 x16
        150..=169 => (60 * (sz - 130), 6),  // 1200..2400 x32
        170..=199 => (200 * (sz - 160), 7), // 2000..8000 x64
        200..=209 => (400 * (sz - 190), 8), // 4000..8000 x128
        210..=219 => (400 * (sz - 190), 9),
        220..=229 => (400 * (sz - 190), 10),
        230..=255 => (400 * (sz - 190), (sz - 10) / 20),
        256.. => (800 * (sz - 225), (sz - 10) / 20),
    }
}

// Interval size is similar to SIQS.
// Factor base is much smaller, so interval size must be much smaller too.
fn interval_size(sz: u32) -> u32 {
    let nblocks = match sz {
        // We use a single unit polynomial.
        // Sieving will stop after enough blocks are sieved.
        0..=32 => 16,
        // We have multiple polynomials
        33..=64 => 2,
        65..=128 => 3,
        129..=180 => 4,
        181..=255 => (sz - 141) / 20,
        256..=300 => (sz - 150) / 25,
        301..=330 => 8,
        331..=360 => 9,
        _ => 10,
    };
    nblocks * 32768
}

fn large_prime_factor(sz: u32) -> u64 {
    let sz = sz as u64;
    match sz {
        0..=32 => 2,
        33..=128 => 4,
        129..=160 => sz - 120,
        161.. => 2 * sz - 160,
    }
}

// The bound for double large primes, as a ratio of B².
// It must be larger than 1 and smaller that L² where L
// is the large prime factor.
fn double_large_factor(n: &Int) -> u64 {
    let sz = n.unsigned_abs().bits() as u64;
    match sz {
        0..=160 => sz,
        161..=250 => 2 * sz - 160,
        // For larger sizes, use double large primes
        // more aggressively, we get less than 1 relation
        // per polynomial so there is no risk of factoring cost.
        // L=400 at 280 bits
        // L=500 at 330 bits
        251.. => 10 * sz - 2000,
    }
}

fn sieve_a(s: &ClSieve, a_int: &Uint, factors: &Factors) {
    let mm = s.qs.interval_size;
    let start_offset = if a_int.is_one() { 0 } else { -(mm as i64) / 2 };
    let a = &prepare_a(factors, a_int, s.qs.fbase, start_offset);
    if s.prefs.verbose(Verbosity::Debug) {
        eprintln!("Sieving A={}", a.description());
    }
    let nfacs = a.len();
    let polys_per_a = if nfacs > 1 { 1 << (nfacs - 1) } else { 1 };
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
    if s.prefs.verbose(Verbosity::Debug)
        || (s.prefs.verbose(Verbosity::Info) && a_int.low_u64() % 17 <= 1)
    {
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
    let mut start_offset: i64 = -(mm as i64) / 2;
    let mut end_offset: i64 = (mm as i64) / 2;
    if pol.is_unit() {
        (start_offset, end_offset) = (0, mm as i64);
    }
    let r1p = &pol.r1p[..];
    let r2p = &pol.r2p[..];
    let mut state = sieve::Sieve::new(start_offset, nblocks, s.qs.fbase, [r1p, r2p], rec);
    if nblocks == 0 {
        sieve_block_poly(s, pol, a, &mut state);
    }
    while state.offset < end_offset {
        if s.rels.read().unwrap().done() {
            // Exit early if finished.
            break;
        }
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
    'smoothloop: for (i, intfacs) in idx.into_iter().zip(facss) {
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
            if s.conductor_primes.contains(&(p as u64)) {
                if s.prefs.verbose(Verbosity::Debug) {
                    eprintln!("WARNING: rejecting relation with conductor prime {p}");
                }
                continue 'smoothloop;
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
        let mut rels = s.rels.write().unwrap();
        rels.add(rel);
        if rels.done() {
            break;
        }
    }
}

// Estimates and approximations.

/// Compute the amount of extra smoothness expected for a given number.
/// The result is a floating-point number usually in the -5..5 range.
fn smoothness_bias(d: &Int) -> f64 {
    // The contribution of 2 is:
    // +1 if D % 8 == 1
    // -0.5 for the 4D case
    let mut bias: f64 = match d.unsigned_abs().low_u64() & 7 {
        7 => 1.0,
        3 => 0.0,
        0 | 4 => -0.5,
        _ => panic!("impossible"),
    };
    // The contribution of a prime is (1 + (D|p)) log(p)/p
    // Small primes are enough to compute an estimate.
    let dabs = d.unsigned_abs();
    for p in fbase::SMALL_PRIMES {
        if p == 2 {
            continue;
        }
        let mut l = legendre(&dabs, p as u32);
        if p % 4 == 3 {
            l = -l;
        }
        bias += (l as f64) * (p as f64).log2() / (p as f64);
    }
    bias
}

/// Compute an estimate of the class number.
pub fn estimate(d: &Int) -> (f64, f64) {
    // The class number formula is:
    // h(-D) = sqrt(D)/pi * prod(1/(1 - (D|p)/p) for prime p)
    // For p=2 the factor is 1/(1-1/2)=2 if D % 8 = 1
    // otherwise 1/(1+1/2) = 2/3
    //
    // Numerical evaluation takes
    // ~0.1s for bound 10^7
    // ~1s for bound 10^8
    // ~5s for bound 10^9

    let dabs = d.unsigned_abs();
    // When the group of units is non trivial, the formula is different.
    if dabs.bits() <= 3 {
        if dabs.low_u64() == 4 {
            return (1.0, 1.0);
        } else if dabs.low_u64() == 3 {
            return (1.0, 1.0);
        }
    }
    let fbsize = clsgrp_fb_size(d.unsigned_abs().bits(), true);
    // enough to get 4 decimal digits
    let bound = std::cmp::min(100_000_000, fbsize * fbsize);
    let mut logprod = 0f64;
    let mut logmin = f64::MAX;
    let mut logmax = f64::MIN;
    let mut compute_prime = |p: u32, bound: u32| {
        if p == 2 {
            return; // handled separately.
        }
        // legendre(-d,p) = legendre(d,p) * (-1)^(p-1)/2
        let mut l = legendre(&dabs, p);
        if p % 4 == 3 {
            l = -l;
        }
        logprod += -(-l as f64 / p as f64).ln_1p();
        if p > bound / 2 {
            // Compute lower/upper bounds over a window
            logmin = logmin.min(logprod);
            logmax = logmax.max(logprod);
        }
    };
    // Faster paths for small bounds
    if bound < 8000 {
        let ps = fbase::primes(200 + bound / 8);
        let bound = ps[ps.len() - 1];
        for p in ps {
            compute_prime(p, bound);
        }
    } else {
        let mut s = fbase::PrimeSieve::new();
        'primeloop: loop {
            let block = s.next();
            for &p in block {
                if p > bound {
                    break 'primeloop;
                }
                compute_prime(p, bound);
            }
        }
    }
    drop(compute_prime);
    let h = d.to_f64().unwrap().abs().sqrt() / std::f64::consts::PI;
    let h = match d.unsigned_abs().low_u64() & 7 {
        // Only values 7, 4, 3 are valid for fundamental discriminants.
        5 | 7 => h * 2.0,
        0 | 2 | 4 | 6 => h,
        1 | 3 => h * 2.0 / 3.0,
        _ => unreachable!(),
    };
    (h * logmin.exp(), h * logmax.exp())
}

fn legendre(d: &Uint, p: u32) -> i32 {
    let div = Dividers::new(p);
    let dmodp = div.mod_uint(d) as u32;
    let mut k = p / 2;
    let mut pow = 1u64;
    let mut sq = dmodp as u64;
    while k > 0 {
        if k & 1 == 1 {
            pow = div.modu63(pow * sq);
        }
        sq = div.modu63(sq * sq);
        k = k >> 1;
    }
    if pow > 1 {
        debug_assert!(pow == p as u64 - 1);
        pow as i32 - p as i32
    } else {
        debug_assert!(pow <= 1);
        pow as i32
    }
}

#[test]
fn test_estimate() {
    use std::str::FromStr;

    let (h1, h2) = estimate(&Int::from(-4));
    assert!(0.9 <= h1 && h2 <= 1.1, "h1={h1} h2={h2}");
    let (h1, h2) = estimate(&Int::from(-3));
    assert!(0.9 <= h1 && h2 <= 1.1, "h1={h1} h2={h2}");
    let (h1, h2) = estimate(&Int::from(-163));
    assert!(0.9 <= h1 && h2 <= 1.1);

    // D = 1-8k
    let d =
        Int::from_str("-1139325066844575699589813265217200398493708241839938355464231").unwrap();
    // h=964415698883565364637432450736
    let (h1, h2) = estimate(&d);
    assert!(h1 <= 9.644157e29 && 9.644157e29 <= h2 * 1.001);

    // D = 5-8k
    let d = Int::from_str("-12239807779826253214859975412431303497371919444169932188160735019")
        .unwrap();
    // h=109997901313565058259819609742265
    let (h1, h2) = estimate(&d);
    assert!(h1 <= 1.099979e32 && 1.099979e32 <= h2);

    // D = 4-8k
    let d = Int::from_str("-40000000000000000000000000000000000000000000000000000000000000004")
        .unwrap();
    // h=178397819605839608466892693850112
    let (h1, h2) = estimate(&d);
    eprintln!("{h1} {h2}");
    assert!(h1 <= 1.783978e32 && 1.783978e32 <= h2);
}

#[cfg(test)]
const fn parse_int(s: &'static str) -> Int {
    Int::parse_str_radix(s, 10)
}

#[test]
fn test_classgroup() {
    let prefs = Preferences::default();
    // A few failing tests for non necessarily difficult numbers.

    // Coordinates should be non zero (indicates bug in Smith normal form).
    let d = Int::from(-103142932);
    let g = classgroup(&d, &prefs, None).unwrap();
    for (_, c) in &g.gens {
        assert!(c.iter().any(|&x| x != 0));
    }

    // May be difficult to choose between h=60 (correct) and h=66
    let d = Int::from(-10148);
    classgroup(&d, &prefs, None).unwrap();

    // Extremely small and smooth class number (64)
    // Relations are very rare and may not generate the full lattice.
    let d = Int::from(-424708);
    let g = classgroup(&d, &prefs, None).unwrap();
    assert_eq!(g.h, 64_u64.into());
    // Other very unlucky numbers
    let d = Int::from(-1411012);
    let g = classgroup(&d, &prefs, None).unwrap();
    assert_eq!(g.h, 124_u64.into());
    let d = Int::from(-2402548);
    let g = classgroup(&d, &prefs, None).unwrap();
    assert_eq!(g.h, 176_u64.into());

    // Affected by incorrect Smith normal form.
    let d = parse_int("-131675478501979154852");
    classgroup(&d, &prefs, None).unwrap();
    // Can trigger an incorrect Smith normal form due to very smooth h.
    let d = parse_int("-4133106580052");
    let g = classgroup(&d, &prefs, None).unwrap();
    assert_eq!(g.h, 615040_u64.into());

    // Close to 128 bits: edge cases for 64-bit overflow.
    let d = parse_int("-277747586393177609383447877774824905287");
    classgroup(&d, &prefs, None).unwrap();
    let d = parse_int("-1547792612939506766277963208426820605844");
    classgroup(&d, &prefs, None).unwrap();

    // Affected by spurious relations p^2 = product(pi^ei)
    // (unstable test)
    //let d = parse_int("-10607235129657707");
    //classgroup(&d, &prefs, None).unwrap();
}

#[allow(unused)]
#[cfg(test)]
fn failing_test_classgroup() {
    let prefs = Preferences::default();
    // Worst case inputs with various sizes.
    let d = parse_int("-5625246009237013252");
    classgroup(&d, &prefs, None);

    let d = parse_int("-17442319661992626809332");
    classgroup(&d, &prefs, None);

    let d = parse_int("-1100921531608271618166868");
    classgroup(&d, &prefs, None);

    let d = parse_int("-4197708731399051763400135492");
    classgroup(&d, &prefs, None);

    let d = parse_int("-333684818975420457430375646788");
    classgroup(&d, &prefs, None);
}
