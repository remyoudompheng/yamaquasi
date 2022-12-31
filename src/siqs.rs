// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Self Initializing Quadratic Sieve
//!
//! Bibliography:
//! Alford, Pomerance, Implementing the self-initializing quadratic sieve
//! https://math.dartmouth.edu/~carlp/implementing.pdf
//!
//! This method accelerates polynomial switch costs by computing polynomials
//! with easy roots.
//! Assuming a factor base {p} such that N is a quadratic residue mod p has been
//! computed, numbers A = p1 * ... * pk with factors in this factor base
//! are such that N has 2^k modular square roots modulo N
//! which can be computed from the factor base data through the Chinese remainder
//! theorem.

use std::cmp::max;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::RwLock;

use bnum::cast::CastFrom;
use num_traits::One;
use rayon::prelude::*;

use crate::arith::{self, Num, U256};
use crate::fbase::{self, FBase, Prime};
use crate::params::{self, BLOCK_SIZE};
use crate::pollard_pm1;
use crate::relations::{self, Relation, RelationSet};
use crate::sieve;
use crate::{Int, Uint, DEBUG};

pub fn siqs(
    n: &Uint,
    prefs: &crate::Preferences,
    tpool: Option<&rayon::ThreadPool>,
) -> Vec<Relation> {
    // Choose factor base. Sieve twice the number of primes
    // (n will be a quadratic residue for only half of them)
    let fb = prefs.fb_size.unwrap_or(params::factor_base_size(&n));
    // Reduce factor base size when using large double primes
    // since they will cover the large prime space.
    let use_double = prefs.use_double.unwrap_or(n.bits() > 256);
    let fb = if use_double { fb / 2 } else { fb };
    let fbase = FBase::new(*n, fb);
    eprintln!("Smoothness bound {}", fbase.bound());
    eprintln!("Factor base size {} ({:?})", fbase.len(), fbase.smalls(),);

    let mm = interval_size(n);
    eprintln!("Sieving interval size {}k", mm >> 10);

    // Generate all values of A now.
    let nfacs = nfactors(n) as usize;
    let factors = select_siqs_factors(&fbase, n, nfacs, mm as usize);
    let a_ints = select_a(&factors, a_value_count(n));
    let polys_per_a = 1 << (nfacs - 1);
    eprintln!(
        "Generated {} values of A with {} factors in {}..{} ({} polynomials each, spread={:.2}%)",
        a_ints.len(),
        nfacs,
        factors.factors[0].p,
        factors.factors.last().unwrap().p,
        polys_per_a,
        a_quality(&a_ints) * 100.0
    );

    let done = AtomicBool::new(false);

    let maxprime = fbase.bound() as u64;
    let maxlarge: u64 = maxprime * prefs.large_factor.unwrap_or(large_prime_factor(n));
    eprintln!("Max large prime {}", maxlarge);
    if use_double {
        eprintln!("Max double large prime {}", maxlarge * maxprime * 2);
    }

    // Prepare packed auxiliary numbers (the Prime structure is > 64 bytes large)
    // They are constant over the whole process.
    // We need for each p the in factor base:
    // * interval start offset mod p (24 bits)
    // * prime number p (24 bits)
    // * multiplier and shift for Divider31 (32 + 8 bits)
    // We pack them as (u64, u32)
    //
    let start_offset: i64 = -(mm as i64) / 2;
    let s = SieveSIQS::new(n, &fbase, maxlarge, use_double, mm as usize);

    let polys_done = AtomicUsize::new(0);
    let gap = AtomicUsize::new(fbase.len());
    let target = AtomicUsize::new(fbase.len() * 8 / 10);

    let handle_result = || -> bool {
        let rlen = {
            let rels = s.rels.read().unwrap();
            rels.len()
        };

        polys_done.fetch_add(1, Ordering::SeqCst);

        if rlen >= target.load(Ordering::Relaxed) {
            // unlikely
            let rgap = {
                let rels = s.rels.read().unwrap();
                rels.gap()
            };
            gap.store(rgap, Ordering::Relaxed);
            if rgap == 0 {
                eprintln!("Found enough relations");
                done.store(true, Ordering::Relaxed);
                return true;
            } else {
                eprintln!("Need {} additional relations", rgap);
                target.store(
                    rlen + rgap + std::cmp::min(10, fb as usize / 4),
                    Ordering::SeqCst,
                );
            }
        }
        false
    };

    let poly_idxs: Vec<usize> = (0..polys_per_a).collect();

    for a_int in a_ints.iter() {
        let a = &prepare_a(&factors, a_int, &fbase, start_offset);
        eprintln!(
            "Sieving A={} (factors {})",
            a.a,
            a.factors
                .iter()
                .map(|item| item.p.to_string())
                .collect::<Vec<_>>()[..]
                .join("*")
        );

        if let Some(pool) = tpool.as_ref() {
            // Multi-threaded
            pool.install(|| {
                poly_idxs.par_iter().for_each(|&idx| {
                    if done.load(Ordering::Relaxed) {
                        return;
                    }
                    let pol = make_polynomial(&s, n, a, idx);
                    siqs_sieve_poly(&s, a, &pol);
                    handle_result();
                });
            })
        } else {
            // Single-threaded
            for idx in 0..polys_per_a {
                let pol = make_polynomial(&s, n, a, idx);
                siqs_sieve_poly(&s, a, &pol);
                let enough = handle_result();
                if enough {
                    break;
                }
            }
        }
        let pdone = polys_done.load(Ordering::Relaxed);
        let rels = s.rels.read().unwrap();
        rels.log_progress(format!(
            "Sieved {}M {} polys",
            (pdone as u64 * mm as u64) >> 20,
            pdone,
        ));
        if gap.load(Ordering::Relaxed) == 0 {
            break;
        }
    }
    if gap.load(Ordering::Relaxed) != 0 {
        panic!("Internal error: not enough smooth numbers with selected parameters");
    }
    let mut rels = s.rels.into_inner().unwrap();
    if rels.len() > fbase.len() + relations::MIN_KERNEL_SIZE {
        rels.truncate(fbase.len() + relations::MIN_KERNEL_SIZE)
    }
    rels.into_inner()
}

/// Helper function to study performance impact of parameters
pub fn siqs_calibrate(n: Uint, threads: Option<usize>) {
    // Prepare central parameters and A values.
    let (k, score) = fbase::select_multiplier(n);
    eprintln!("Using fixed multiplier {} (score {:.2}/8)", k, score);
    let n = &(n * Uint::from(k));

    let fb0 = params::factor_base_size(n);
    let fb0 = if n.bits() > 256 { fb0 / 2 } else { fb0 };
    let lf0 = large_prime_factor(n);
    let use_double = n.bits() > 256;
    // This factor base is only to select A
    let fbase0 = FBase::new(*n, fb0);
    let mm0 = interval_size(n);
    let blocks0 = mm0 as i64 / BLOCK_SIZE as i64;

    let nfacs = nfactors(n) as usize;
    let polys_per_a = 1 << (nfacs - 1);

    let tpool = threads.map(|t| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("cannot create thread pool")
    });

    eprintln!("Base params fb={fb0} lf={lf0}");
    // The value of a varies according to interval size.
    let mut factorss = vec![];
    let mut mms = vec![];
    let mut a_s = vec![];
    let dblk = max(1, blocks0 / 6);
    for blks in [
        blocks0 - 2 * dblk,
        blocks0 - dblk,
        blocks0,
        blocks0 + dblk,
        blocks0 + 2 * dblk,
    ] {
        if blks > 0 {
            // FIXME: functions have their own parameters.
            let mm = BLOCK_SIZE * (blks as usize);
            let factors = select_siqs_factors(&fbase0, n, nfacs, mm);
            let a_ints = select_a(&factors, a_value_count(n));
            let a0 = a_ints[0];
            let polys_per_a = 1 << (nfacs - 1);
            eprintln!("Test set M={}k A={} npolys={}", mm / 2048, a0, polys_per_a);
            // sample polynomial
            let a = &prepare_a(&factors, &a0, &fbase0, -(mm as i64) / 2);
            let s = SieveSIQS::new(n, &fbase0, 0, use_double, mm);
            let pol = make_polynomial(&s, n, a, 0);
            eprintln!("min(P) ~ {}", pol.c);
            eprintln!(
                "max(P) ~ {}",
                Int::cast_from(pol.a) * Int::from(mm * mm / 4)
                    + Int::cast_from(pol.b) * Int::from(mm)
                    + pol.c
            );
            mms.push(mm);
            factorss.push(factors);
            a_s.push(a0);
        }
    }

    // Iterate on parameters
    for fb in [4 * fb0 / 5, fb0, 5 * fb0 / 4] {
        // Print separator: results have different meanings
        eprintln!("===");
        let fbase = FBase::new(*n, fb);
        for lf in [2 * lf0 / 3, lf0, 3 * lf0 / 2] {
            for use_double in [false, true] {
                for (idx, &mm) in mms.iter().enumerate() {
                    let aint = a_s[idx];
                    let factors = &factorss[idx];
                    let maxprime = fbase.bound() as u64;
                    let maxlarge: u64 = maxprime * lf as u64;
                    let s = SieveSIQS::new(n, &fbase, maxlarge, use_double, mm);
                    // Measure metrics
                    let t0 = std::time::Instant::now();
                    let a = &prepare_a(factors, &aint, &fbase, -(mm as i64) / 2);
                    if let Some(pool) = tpool.as_ref() {
                        let poly_idxs: Vec<usize> = (0..polys_per_a).collect();
                        pool.install(|| {
                            poly_idxs.par_iter().for_each(|&idx| {
                                let pol = make_polynomial(&s, n, a, idx);
                                siqs_sieve_poly(&s, a, &pol);
                            });
                        })
                    } else {
                        for idx in 0..polys_per_a {
                            let pol = make_polynomial(&s, n, a, idx);
                            siqs_sieve_poly(&s, a, &pol);
                        }
                    }
                    let dt = t0.elapsed().as_secs_f64();
                    let rels = s.rels.read().unwrap();
                    eprintln!(
                        "fb={fb} B1={lf} B2={} M={}k dt={dt:2.3}s {:.2}ns/i c={} ({:.1}/s) p={} ({:.1}/s) pp={} ({:.1}/s)",
                        if use_double { 2 } else { 0 },
                        mm / 2048,
                        dt * 1.0e9 / (mm as f64) / (polys_per_a as f64),
                        rels.len(),
                        rels.len() as f64 / dt,
                        rels.n_partials,
                        rels.n_partials as f64/ dt,
                        rels.n_doubles,
                        rels.n_doubles as f64/dt,
                    )
                }
            }
        }
    }
}

// Parameters:
// m = number of needed A values
// k = number of factors in each A
// A is around sqrt(2N)/M
//
// Number of polynomials: m * 2^(k-1)
// 120..160 bits => k=7 (factors 8-10 bits)
// 160..200 bits => k=8 (factors 10-11 bits)
// 200..250 bits => k=9 (factors 11-13 bits)
// 250..300 bits => k=10 (factors 13-15 bits)

fn nfactors(n: &Uint) -> u32 {
    match n.bits() {
        0..=69 => 2,
        70..=79 => 3,
        80..=89 => 4,
        90..=149 => 5,
        150..=169 => 6,
        170..=189 => 7,
        190..=209 => 8,
        210..=239 => 9,
        240..=269 => 10,
        270..=299 => 11,
        _ => 12,
    }
}

fn a_value_count(n: &Uint) -> usize {
    // Many polynomials are required to accomodate small intervals.
    // When sz=180 we need more than 5k polynomials
    // When sz=200 we need more than 20k polynomials
    // When sz=280 we need more than 1M polynomials
    let sz = n.bits() as usize;
    match sz {
        0..=129 => 8 + sz / 10,        // 8..20
        130..=169 => sz - 60,          // 20..100
        170..=199 => 50 * (sz - 168),  // 100..1000
        200..=249 => 100 * (sz - 190), // 1000..5000
        _ => 20 * sz,                  // 5000..
    }
}

// Returns d such that we select A up to distance 1/d-th of the optimal value.
fn a_tolerance_divisor(n: &Uint) -> usize {
    match n.bits() {
        0..=50 => 3,
        51..=70 => 5,
        71..=90 => 6,
        91..=110 => 20,
        111..=140 => 40,
        141..=160 => 80,
        _ => 100,
    }
}

fn interval_size(n: &Uint) -> u32 {
    // Choose very small intervals since the cost of switching
    // polynomials is very small (less than 1ms).
    // Large intervals also hurt memory locality during sieve.
    // We want an integral amount of 32k blocks:
    // 1 block under 100 bits
    // 4 blocks for 120-180 bits
    // ~10 blocks for 240 bits
    // ~32 blocks for 300 bits
    let sz = n.bits();
    let nblocks = match sz {
        0..=100 => 1,
        101..=130 => 2,
        131..=160 => 3,
        161..=190 => 5,
        191..=260 => (sz - 140) / 10,    // 5..12
        261..=300 => (sz - 200) / 5,     // 12..20
        301..=350 => (2 * sz - 500) / 5, // 20..40
        _ => 40,
    };
    nblocks * sieve::BLOCK_SIZE as u32
}

fn large_prime_factor(n: &Uint) -> u64 {
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
        101..=250 => {
            // More large primes to compensate fewer relations
            n.bits() as u64
        }
        251.. => {
            // Bound large primes to avoid exceeding 32 bits.
            128 + n.bits() as u64 / 2
        }
    }
}

// Polynomial selection

pub struct Factors<'a> {
    pub n: &'a Uint,
    pub target: U256,
    pub nfacs: usize,
    // A sorted list of factors
    pub factors: Vec<Prime<'a>>,
    // inverses[i][j] = pi^-1 mod pj
    pub inverses: Vec<Vec<u32>>,
}

// Select factors of generated A values. It is enough to select about
// twice the number of expected factors in A, because the number of
// combinations is large enough to generate values close to the target.
pub fn select_siqs_factors<'a>(fb: &'a FBase, n: &'a Uint, nfacs: usize, mm: usize) -> Factors<'a> {
    // For interval [-M,M] the target is sqrt(2N) / M, see [Pomerance].
    // Don't go below 2000 for extremely small numbers.
    let target = max(
        Uint::from(2000u64),
        arith::isqrt(n << 1) / Uint::from(mm as u64 / 2),
    );
    let idx = fb
        .primes
        .partition_point(|&p| Uint::from(p as u64).pow(nfacs as u32) < target);
    let selected_idx = if idx + nfacs >= fb.len() {
        eprintln!("WARNING: suboptimal choice of A factors");
        fb.len() - 2 * nfacs..fb.len()
    } else if idx > 4 * nfacs && idx + 4 * nfacs < fb.len() {
        idx - 2 * nfacs..idx + 2 * nfacs
    } else {
        idx - nfacs..idx + max(nfacs, 6)
    };
    // Make sure that selected factors don't divide n.
    let selection: Vec<Prime> = selected_idx
        .filter(|&i| i < fb.len() && fb.div(i).mod_uint(n) != 0)
        .map(|i| fb.prime(i))
        .collect();
    // Precompute inverses
    let mut inverses = vec![];
    for p in &selection {
        let mut row = vec![];
        for q in &selection {
            let pinvq = if p.p == q.p {
                0
            } else {
                arith::inv_mod64(p.p as u64, q.p as u64).unwrap()
            };
            row.push(pinvq as u32);
        }
        inverses.push(row);
    }
    assert!(target.bits() < 256);
    Factors {
        n,
        target: U256::cast_from(target),
        nfacs,
        factors: selection.into_iter().collect(),
        inverses,
    }
}

/// Precomputed information to compute all square roots of N modulo A.
/// The square roots are sum(CRT[j] b[j]) with 2 choices for each b[j].
/// Precompute CRT[j] b[j] modulo the factor base (F vectors)
/// then combine them for each polynomial (faster than computing 2^F vectors).
pub struct A<'a> {
    a: Uint,
    factors: Vec<Prime<'a>>,
    factors_idx: Box<[usize]>,
    // Base data for polynomials:
    // Precomputed bj = CRT[j] * sqrt(n) mod pj
    // bj^2 = n mod pj, 0 mod pi (i != j)
    roots: Vec<[Uint; 2]>,
    // To prepare polynomial roots faster, precompute sets of roots mod p
    // in a "radix-16" way. We need (-B±r)/A - offset
    //
    // roots_mod_p[16i+j] is the combination of minus roots[4i+j1][j2] mod p
    // where (j1,j2) goes through the bit mask (j << (4*i)).
    // The roots are pre-multiplied by ainv to make preparation easier.
    //
    // If i==0 we also add -rp - start_offset % p
    // to avoid doing it repeatedly.
    //
    // Since the number of factors is at most 12, at most 3 vectors
    // out of 48 will be added to compute the final result for a given polynomial.
    roots_mod_p: Box<[Box<[u32]>]>,
    // Precomputed p.r / a mod p
    rp: Vec<u32>,
}

fn a_quality(a_s: &[Uint]) -> f64 {
    let (amin, amax) = (a_s.first().unwrap(), a_s.last().unwrap());
    let a_diff = amax - amin;
    let a_mid: Uint = (amin + amax) >> 1;
    if a_mid.bits() < 64 {
        a_diff.low_u64() as f64 / a_mid.low_u64() as f64
    } else {
        let shift = a_diff.bits() - 10;
        (a_diff >> shift).low_u64() as f64 / (a_s.last().unwrap() >> shift).low_u64() as f64
    }
}

/// Find smooth numbers around the target that are products of
/// distinct elements of the factor base.
/// The factor base is assumed to be an array of primes with similar
/// sizes.
pub fn select_a(f: &Factors, want: usize) -> Vec<Uint> {
    // Sample deterministically products of W primes
    // closest to target and select best candidates.
    // We usually don't need more than 1000 values.
    //
    // We are going to select ~2^W best products of W primes among 2W

    let mut div = a_tolerance_divisor(f.n);
    let mut amin = f.target - f.target / div as u64;
    let mut amax = f.target + f.target / div as u64;

    let mut rng: u64 = 0xcafebeefcafebeef;
    let fb = f.factors.len();
    let mut gen = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng % fb as u64
    };
    let mut candidates = vec![];
    let mut iters = 0;
    while iters < 1000 * want || candidates.len() < want {
        iters += 1;
        if iters % (100 * want) == 0 && candidates.len() < want {
            eprintln!("WARNING: unable to find suitable A, increasing tolerance");
            div = max(div, 1) - 1;
            if div == 0 {
                amin = f.target >> 2;
                amax = f.target << 2;
            } else {
                amin = f.target - f.target / div as u64;
                amax = f.target + f.target / div as u64;
            }
        }
        // A is smaller than sqrt(n) so 256-bit arithmetic is enough.
        let mut product = U256::one();
        let mut mask = 0u64;
        while mask.count_ones() < f.nfacs as u32 - 1 {
            let g = gen();
            if mask & (1 << g) == 0 {
                mask |= 1 << g;
                product *= U256::from(f.factors[g as usize].p);
            }
        }
        let t = (f.target / product).to_u64().unwrap();
        let idx = (0usize..fb)
            .filter(|g| mask & (1 << g) == 0)
            .min_by_key(|&idx| (f.factors[idx].p as i64 - t as i64).abs())
            .unwrap();
        product *= U256::from(f.factors[idx].p);
        if amin < product && product < amax {
            candidates.push(Uint::cast_from(product));
        }
        if candidates.len() > want && iters % 10 == 0 {
            candidates.sort();
            candidates.dedup();
            let idx = candidates.partition_point(|&c| U256::cast_from(c) < f.target);
            if idx > want && idx + want < candidates.len() {
                return candidates[idx - want / 2..idx + want / 2].to_vec();
            }
        }
    }
    // Should not happen? return what we found so far
    candidates.sort();
    candidates.dedup();
    candidates
}

pub fn prepare_a<'a>(f: &Factors<'a>, a: &Uint, fbase: &FBase, start_offset: i64) -> A<'a> {
    let afactors: Vec<(usize, &Prime)> = f
        .factors
        .iter()
        .enumerate()
        .filter(|(_, p)| p.div.mod_uint(a) == 0)
        .collect();
    let mut crt = vec![];
    // Each CRT coefficient is the product of pj (pj^-1 mod pi)
    for &(idx, p) in afactors.iter() {
        let mut c = Uint::one();
        for &(jdx, q) in afactors.iter() {
            if jdx != idx {
                c *= Uint::from(q.p * (f.inverses[jdx][idx] as u64));
                debug_assert!(c % q.p == 0);
                debug_assert!(c % p.p == 1);
            }
        }
        crt.push(c % a);
    }
    // Compute modular inverses of A or 1 for prime factors of A.
    let mut ainv = vec![];
    let mut factors_idx = vec![];
    for pidx in 0..fbase.len() {
        let p = fbase.p(pidx);
        let div = fbase.div(pidx);
        let amod = div.mod_uint(a);
        if amod == 0 {
            factors_idx.push(pidx);
        }
        ainv.push(arith::inv_mod64(amod, p as u64).unwrap_or(1) as u32);
    }
    // Compute sqrt(n)/A mod p
    let mut rp = vec![0u32; (fbase.len() + 15) & !15];
    assert!(rp.len() % 16 == 0);
    for pidx in 0..fbase.len() {
        let r = fbase.r(pidx);
        let pdiv = fbase.div(pidx);
        rp[pidx] = pdiv.divmod64(ainv[pidx] as u64 * r as u64).1 as u32;
    }
    // Compute basic roots
    let mut roots = vec![];
    for i in 0..afactors.len() {
        let r1 = afactors[i].1.r;
        let r2 = afactors[i].1.p - afactors[i].1.r;
        roots.push([(crt[i] * Uint::from(r1)) % a, (crt[i] * Uint::from(r2)) % a]);
    }
    // Compute roots mod p.
    // We need 16(l/4) + (2^(l%4)) vectors.
    let n_parts = 16 * (afactors.len() / 4)
        + if afactors.len() % 4 > 0 {
            1 << (afactors.len() % 4)
        } else {
            0
        };
    let mut roots_mod_p = Box::from_iter(
        (0..n_parts).map(|_| vec![0u32; (fbase.len() + 15) & !15].into_boxed_slice()),
    );
    for idx in 0..n_parts {
        let (i, j) = (idx / 16, idx % 16);
        // Compute the partial B corresponding to mask (j << 4i)
        let mut b = Uint::ZERO;
        for k in 0..4 {
            if 4 * i + k >= afactors.len() {
                continue;
            }
            b += roots[4 * i + k][(j >> k) & 1];
        }
        let b = arith::U256::cast_from(b);
        assert!(b.bits() < 250);
        let v = &mut roots_mod_p[idx];
        assert!(v.len() % 16 == 0);
        for pidx in 0..fbase.len() {
            let p = fbase.p(pidx) as i32;
            let pdiv = fbase.div(pidx);
            let b_over_a = b * arith::U256::from(ainv[pidx]);
            // If i = 0, subtract rp and offset
            let mut val = p - pdiv.mod_uint(&b_over_a) as i32;
            if i == 0 {
                val = val - rp[pidx] as i32 - start_offset as i32;
            }
            v[pidx] = pdiv.div31.modi32(val) as u32;
        }
    }
    A {
        a: *a,
        factors: afactors.into_iter().map(|(_, p)| p).cloned().collect(),
        factors_idx: factors_idx.into_boxed_slice(),
        roots,
        roots_mod_p,
        rp,
    }
}

#[derive(Debug)]
pub struct Poly {
    a: Uint,
    b: Uint,
    c: Int,
    // Precomputed roots
    r1p: Box<[u32]>,
    r2p: Box<[u32]>,
}

/// Given coefficients A and B, compute all roots (Ax+B)^2=N
/// modulo the factor base, computing (r - B)/A mod p
pub fn make_polynomial(s: &SieveSIQS, n: &Uint, a: &A, pol_idx: usize) -> Poly {
    let pol_idx = pol_idx << 1;
    // Combine roots: don't reduce modulo n.
    // This allows combining the roots modulo the factor base,
    // with the issue that (Ax+B) is no longer minimal for x=0
    // but for round(-B/A) which is extremely small.
    let mut b = Uint::ZERO;
    for i in 0..a.factors.len() {
        b += a.roots[i][(pol_idx >> i) & 1];
    }
    // Also combine roots mod p
    let mut bmodp = vec![0u32; a.roots_mod_p[0].len()].into_boxed_slice();
    for i in 0..(a.factors.len() + 3) / 4 {
        let v = &a.roots_mod_p[16 * i + ((pol_idx >> (4 * i)) % 16)];
        assert_eq!(v.len() % 8, 0);
        // first += v
        let mut idx = 0;
        while idx < v.len() {
            unsafe {
                // (possibly) unaligned pointers
                let v8 = (v.get_unchecked(idx) as *const u32) as *const [u32; 8];
                let r8 = (bmodp.get_unchecked_mut(idx) as *const u32) as *mut [u32; 8];
                let v8w = wide::u32x8::new(*v8);
                let r8w = wide::u32x8::new(*r8);
                *r8 = (r8w + v8w).to_array();
            }
            idx += 8;
        }
    }
    // Compute final polynomial roots
    // (±r - B)/A = (-B/A) ± (r/p) solutions of (Ax+B)^2 = N
    //
    // In prepare_a, everything has been done so that bmodp = (-r - B)/A - offset
    // We only need to add 2r for the second root, and reduce mod p.
    let mut r1p = bmodp;
    let mut r2p = vec![0u32; r1p.len()].into_boxed_slice();
    let mut idx = 0;
    let primes = &s.fbase.primes[..];
    assert_eq!(r1p.len(), r2p.len());
    assert!(r1p.len() >= primes.len());
    assert_eq!(primes.len() % 8, 0);
    while idx < primes.len() {
        unsafe {
            // (possibly) unaligned pointers
            let r18 = (r1p.get_unchecked_mut(idx) as *mut u32) as *mut [u32; 8];
            let r28 = (r2p.get_unchecked_mut(idx) as *mut u32) as *mut [u32; 8];
            let rs = (a.rp.get_unchecked(idx) as *const u32) as *const [u32; 8];
            let ps = (primes.get_unchecked(idx) as *const u32) as *const [u32; 8];
            let mut r1w = wide::u32x8::new(*r18);
            let r8 = wide::u32x8::new(*rs);
            let p8 = wide::u32x8::new(*ps);

            // Reduce r1 mod p.
            // To properly benefit from SIMD vectorization we need to avoid
            // multiplication and division.
            // We use the fact that individual terms are already reduced modulo p
            // so that the reduction can be written as conditional subtractions by p.

            // r1 < 3p so at most 2 subtractions are required.
            // Repeatedly mask and subtract p to reduce mod p.
            // FIXME: there is a bug in wide 0.7.5
            // where u32x8::cmp_lt is mapped to cmp_eq_mask_i32_m256i
            // We use (r2 > p-1) as mask instead of (not r2 < p)
            let p8_m1 = p8 - wide::u32x8::ONE;
            r1w -= p8 & r1w.cmp_gt(p8_m1);
            r1w -= p8 & r1w.cmp_gt(p8_m1);
            *r18 = r1w.to_array();
            debug_assert!(r1w >> 24 == wide::u32x8::ZERO);
            debug_assert!(r1w.cmp_gt(p8_m1) == wide::u32x8::ZERO);
            // Compute the second root by adding 2*rp
            let mut r2w: wide::u32x8 = r1w + (r8 << 1);
            // The result is < 3p, subtract at most 2 times p.
            // FIXME: don't use u32x8::cmp_lt in wide 0.7.5
            r2w -= p8 & r2w.cmp_gt(p8_m1);
            r2w -= p8 & r2w.cmp_gt(p8_m1);
            *r28 = r2w.to_array();
            debug_assert!(r2w >> 24 == wide::u32x8::ZERO);
            debug_assert!(r2w.cmp_gt(p8_m1) == wide::u32x8::ZERO);
        }
        idx += 8;
    }

    debug_assert!((b * b) % a.a == n % a.a);
    // Compute c such that b^2 - ac = N
    // (Ax+B)^2 - n = A(Ax^2 + 2Bx + C)
    let c = (Int::from_bits(b * b) - Int::from_bits(*n)) / Int::from_bits(a.a);
    debug_assert!(Int::from_bits(*n) == Int::from_bits(b * b) - c * Int::from_bits(a.a));

    // Special case for divisors of A.
    // poly % p = 2Bx + C, root is -C/2B
    for &pidx in a.factors_idx.iter() {
        let pidx = pidx as usize;
        let div = &s.fbase.div(pidx);
        let p = s.fbase.p(pidx);
        let bp = div.mod_uint(&b);
        let mut cp = div.mod_uint(&c.abs().to_bits());
        if !c.is_negative() {
            cp = p as u64 - cp;
        }
        let r = div
            .divmod64(cp * arith::inv_mod64(2 * bp, p as u64).unwrap())
            .1 as u32;
        let off = s.offset_modp[pidx];
        let r = div.div31.modu31(r + p as u32 - off);
        r1p[pidx] = r;
        r2p[pidx] = r;
    }
    // Each r1p,r2p[idx] is a root of P(x+offset) modulo p[idx]

    // n has at most 512 bits, and b < sqrt(n)
    assert!(b.bits() < 256);
    Poly {
        a: a.a,
        b,
        c,
        r1p,
        r2p,
    }
}

// Sieving process

fn siqs_sieve_poly(s: &SieveSIQS, a: &A, pol: &Poly) {
    let mm = s.interval_size;
    let nblocks: usize = mm / BLOCK_SIZE;
    if DEBUG {
        eprintln!(
            "Sieving polynomial A={} B={} M={}k blocks={}",
            pol.a,
            pol.b,
            mm / 2048,
            nblocks
        );
    }

    // Construct initial state.
    let start_offset: i64 = -(mm as i64) / 2;
    let end_offset: i64 = (mm as i64) / 2;
    let primes = &s.fbase.primes[..];
    let r1p = &pol.r1p[..];
    let r2p = &pol.r2p[..];
    let pfunc = move |pidx| -> sieve::SievePrime {
        unsafe {
            let p = *primes.get_unchecked(pidx);
            let r1 = *r1p.get_unchecked(pidx);
            let r2 = *r2p.get_unchecked(pidx);
            let offsets = [Some(r1), if r1 == r2 { None } else { Some(r2) }];
            sieve::SievePrime { p, offsets }
        }
    };
    let mut state = sieve::Sieve::new(start_offset, nblocks, s.fbase, &pfunc);
    if nblocks == 0 {
        sieve_block_poly(s, pol, a, &mut state);
    }
    while state.offset < end_offset {
        sieve_block_poly(s, pol, a, &mut state);
        state.next_block();
    }
}

pub struct SieveSIQS<'a> {
    pub n: &'a Uint,
    pub interval_size: usize,
    pub fbase: &'a FBase,
    pub maxlarge: u64,
    pub use_double: bool,
    pub rels: RwLock<RelationSet>,
    pub offset_modp: Box<[u32]>,
    pub pm1_base: Option<pollard_pm1::PM1Base>,
}

impl<'a> SieveSIQS<'a> {
    pub fn new(
        n: &'a Uint,
        fb: &'a FBase,
        maxlarge: u64,
        use_double: bool,
        interval_size: usize,
    ) -> Self {
        let start_offset: i64 = -(interval_size as i64) / 2;
        let mut offsets = vec![0u32; (fb.len() + 15) & !15].into_boxed_slice();
        assert_eq!(offsets.len() % 16, 0);
        for idx in 0..fb.len() {
            let off = fb.div(idx).modi64(start_offset) as u32;
            offsets[idx] = off;
        }
        SieveSIQS {
            n,
            interval_size,
            fbase: fb,
            rels: RwLock::new(RelationSet::new(*n, maxlarge)),
            maxlarge,
            use_double,
            offset_modp: offsets,
            pm1_base: if use_double {
                Some(pollard_pm1::PM1Base::new())
            } else {
                None
            },
        }
    }
}

// Sieve using a selected polynomial
fn sieve_block_poly(s: &SieveSIQS, pol: &Poly, a: &A, st: &mut sieve::Sieve) {
    st.sieve_block();

    let maxprime = s.fbase.bound() as u64;
    let maxlarge = s.maxlarge;
    assert!(maxlarge == (maxlarge as u32) as u64);
    let max_cofactor: u64 = if s.use_double {
        // We don't want double large prime to reach maxlarge^2
        // because p-relations are much less dense in the large prime area
        // and this creates extra factoring pressure.
        // We require that at least one prime is "small" so multiply
        // the lower and upper end of the large prime range is a good
        // midpoint.
        // See [Lentra-Manasse]
        maxlarge * maxprime * 2
    } else {
        maxlarge
    };
    let interval_m = interval_size(s.n) as u64 / 2;
    let target = s.n.bits() / 2 + (interval_m).bits() - max_cofactor.bits();
    let n = s.n;
    let (idx, facss) = st.smooths(target as u8);
    for (i, facs) in idx.into_iter().zip(facss) {
        let mut factors: Vec<(i64, u64)> = Vec::with_capacity(20);
        // Evaluate polynomial Ax^2 + 2Bx+ C
        let x = Int::from(st.offset + (i as i64));
        let ax_b = Int::from_bits(pol.a) * x + Int::from_bits(pol.b);
        let v = (ax_b + Int::from_bits(pol.b)) * x + pol.c;
        // xrel^2 = (Ax+B)^2 = A * v mod n
        // v is never divisible by A
        if v.is_negative() {
            factors.push((-1, 1));
        }
        let mut cofactor: Uint = v.abs().to_bits();
        for pidx in facs {
            let p = s.fbase.p(pidx);
            let div = s.fbase.div(pidx);
            let mut exp = 0;
            loop {
                let (q, r) = div.divmod_uint(&cofactor);
                if r == 0 {
                    cofactor = q;
                    exp += 1;
                } else {
                    break;
                }
            }
            // FIXME: we should have exp > 0
            if exp > 0 {
                factors.push((p as i64, exp));
            }
        }
        let Some(cofactor) = cofactor.to_u64() else { continue };
        if cofactor > max_cofactor {
            continue;
        }
        let pq = if cofactor > maxprime * maxprime {
            // Possibly a double large prime
            let pq = fbase::try_factor64(s.pm1_base.as_ref(), cofactor);
            match pq {
                Some((p, q)) if p > maxlarge || q > maxlarge => continue,
                None if cofactor > maxlarge => continue,
                _ => pq,
            }
        } else {
            // Must be prime
            debug_assert!(!fbase::certainly_composite(cofactor));
            if cofactor > maxlarge {
                continue;
            }
            None
        };
        // Complete with factors of A
        for f in &a.factors {
            if let Some(idx) = factors.iter().position(|&(p, _)| p as u64 == f.p) {
                factors[idx].1 += 1;
            } else {
                factors.push((f.p as i64, 1));
            }
        }
        let xrel = ax_b.abs().to_bits() % n;
        let xrel = if v.is_negative() { n - xrel } else { xrel };
        if DEBUG {
            eprintln!("x={} smooth {} cofactor {}", x, v, cofactor);
        }
        assert!(
            cofactor == 1 || cofactor > maxprime as u64,
            "invalid cofactor {}",
            cofactor
        );
        let rel = Relation {
            x: xrel,
            cofactor,
            factors,
            cyclelen: 1,
        };
        debug_assert!(rel.verify(s.n));
        s.rels.write().unwrap().add(rel, pq);
    }
}

#[test]
fn test_poly_a() {
    use crate::fbase;
    use std::str::FromStr;

    // Check that we obtain enough A values for various input sizes.
    const N100: &str = "1037510308142021112704792564947";
    const N120: &str = "966900989857874724182183960752602697";
    const N140: &str = "628343462775940766740025939587872832856351";
    const N160: &str = "924749938828041082847054913126284372335960469233";
    const N180: &str = "609717477947510609865834953348014542054334353064133851";
    const N200: &str = "1499802708882526909122644146289721370711415544090318473858983";
    const N220: &str = "960924940937451908640130999851435804366317230146233463701332018699";
    const N240: &str = "1563849171863495214507949103370077342033765608728382665100245282240408041";
    const N260: &str =
        "960683424930949054059211750923998830300819600123951352711285164932272044258249";
    const N280: &str =
        "768443048336808679050045973609532661331806099418806842517773942579627976951969897517";
    const N300: &str = "1441811698044810694576317861317873120840140792275400485937339204632253015892676440922745819";
    const N320: &str = "1420239033309714984094415212531751615454438351340772432135724366126533462488286213242822097387831";

    fn compute(n: &Uint, nfacs: u8, want: usize) {
        // Divide by 20 to obtain faster tests.
        let want = if want > 50 { want / 20 } else { want };

        let fb_size = params::factor_base_size(n);
        let fb = fbase::FBase::new(*n, fb_size);

        let facs = select_siqs_factors(&fb, n, nfacs as usize, 1 << 20);
        let target = Uint::cast_from(facs.target);

        let a_vals = select_a(&facs, want);
        let _10000 = Uint::from(10000u64);
        let a_first = a_vals.first().unwrap();
        let a_last = a_vals.last().unwrap();
        let d1: f64 = (target * _10000 / a_first).low_u64() as f64 / 100.0;
        let d2: f64 = (a_last * _10000 / target).low_u64() as f64 / 100.0;
        eprintln!(
            "N {} bits, {} values [{}..{}], quality -{:.2}%..{:.2}%",
            n.bits(),
            a_vals.len(),
            a_first,
            a_last,
            d1 - 100.0,
            d2 - 100.0
        );
        if n.bits() < 150 {
            // 1% tolerance
            assert!(d1 - 100.0 < 1.0);
            assert!(d2 - 100.0 < 1.0);
            assert!(a_vals.len() >= want);
        } else {
            assert!(d1 - 100.0 < 0.5);
            assert!(d2 - 100.0 < 0.5);
            assert!(a_vals.len() >= want / 2);
        }
    }
    // Need very few polynomials
    compute(&Uint::from_str(N100).unwrap(), 4, 4); // 100
    compute(&Uint::from_str(N120).unwrap(), 4, 6); // 120
    compute(&Uint::from_str(N140).unwrap(), 5, 10);
    compute(&Uint::from_str(N160).unwrap(), 6, 10);
    // Need 1000-5000 polynomials
    compute(&Uint::from_str(N180).unwrap(), 7, 100);
    compute(&Uint::from_str(N200).unwrap(), 8, 100);
    // Need >10k-50k polynomials
    compute(&Uint::from_str(N220).unwrap(), 9, 200);
    compute(&Uint::from_str(N240).unwrap(), 9, 500);
    // Takes a long time.
    compute(&Uint::from_str(N260).unwrap(), 9, 1000);
    compute(&Uint::from_str(N280).unwrap(), 10, 1000);
    compute(&Uint::from_str(N300).unwrap(), 10, 2000);
    compute(&Uint::from_str(N320).unwrap(), 11, 2000);
}

#[test]
fn test_poly_prepare() {
    use crate::fbase;
    use std::str::FromStr;

    const N240: &str = "1563849171863495214507949103370077342033765608728382665100245282240408041";
    let n = Uint::from_str(N240).unwrap();
    let fb = fbase::FBase::new(n, 10000);
    let s = SieveSIQS::new(&n, &fb, fb.bound() as u64, false, 0);
    // Prepare A values
    // Only test 10 A values and 35 polynomials per A.
    let f = select_siqs_factors(&fb, &n, 9, 1 << 20);
    let a_ints = select_a(&f, 10);
    for a_int in &a_ints {
        let a = prepare_a(&f, a_int, &fb, 0);
        // Check CRT coefficients.
        assert_eq!(a.a, a.factors.iter().map(|x| Uint::from(x.p)).product());
        for (i, r) in a.roots.iter().enumerate() {
            for (j, p) in a.factors.iter().enumerate() {
                let &[r1, r2] = r;
                if i == j {
                    assert_eq!((r1 * r1) % p.p, n % p.p);
                    assert_eq!((r2 * r2) % p.p, n % p.p);
                } else {
                    assert_eq!(r1 % p.p, 0);
                    assert_eq!(r2 % p.p, 0);
                }
            }
        }
        for idx in 0..35 {
            let idx = 7 * idx;
            // Generate and check each polynomial.
            let pol = make_polynomial(&s, &n, &a, idx);
            let (pa, pb, pc) = (pol.a, pol.b, pol.c);
            // B is a square root of N modulo A.
            assert_eq!((pb * pb) % pa, n % pa);
            // Check that (Ax+B)^2 - n = A(Ax^2+2Bx+C)
            for x in [1u64, 100, 50000] {
                let u = pa * Uint::from(x) + pb;
                let u = Int::from_bits(u * u) - Int::from_bits(n);
                let v = pa * Uint::from(x * x) + (pb << 1) * Uint::from(x);
                let v = Int::from_bits(v) + pc;
                assert_eq!(u, Int::from_bits(pa) * v);
            }
            for pidx in 0..fb.len() {
                // Roots are roots of Ax^2+2Bx+C modulo p.
                for r in [pol.r1p[pidx], pol.r2p[pidx]] {
                    let r = r as u64;
                    let v = pa * Uint::from(r * r) + pb * Uint::from(2 * r);
                    let v = Int::from_bits(v) + pc;
                    assert_eq!(
                        v.abs().to_bits() % (fb.p(pidx) as u64),
                        0,
                        "p={} r={} P(r) mod p={}",
                        fb.p(pidx),
                        r,
                        v.abs().to_bits() % (fb.p(pidx) as u64)
                    );
                }
            }
        }
    }
}
