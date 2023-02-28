// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Self Initializing Quadratic Sieve
//!
//! Bibliography:
//! Alford, Pomerance, Implementing the self-initializing quadratic sieve
//! <https://math.dartmouth.edu/~carlp/implementing.pdf>
//!
//! This method accelerates polynomial switch costs by computing polynomials
//! with easy roots.
//! Assuming a factor base {p} such that N is a quadratic residue mod p has been
//! computed, numbers A = p1 * ... * pk with factors in this factor base
//! are such that N has 2^k modular square roots modulo N
//! which can be computed from the factor base data through the Chinese remainder
//! theorem.
//!
//! Just like MPQS (see Silverman's article and [crate::mpqs])
//! there are two types of polynomials:
//! type 1 (Ax^2+2Bx+C) = (Ax+B)^2/A mod N where B^2-AC=N
//! => always a valid choice, A=sqrt(2N)/M max value sqrt(N)M
//!
//! type 2 (Ax^2+Bx+C) = (2Ax+B)^2/4A mod N where B^2-4AC=N
//! => only valid if N=1 mod 4, A=sqrt(N/2)/M max value sqrt(N)M/2
//! If N=5 mod 8 all polynomial values will be odd.
//! If N=1 mod 8 all polynomial values will be even.

use std::cmp::{max, min};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::RwLock;

use bnum::cast::CastFrom;
use num_traits::One;
use rayon::prelude::*;

use crate::arith::{self, Num, I256, U256};
use crate::fbase::{self, FBase, Prime};
use crate::params;
use crate::relations::{self, Relation, RelationSet};
use crate::sieve::{self, BLOCK_SIZE};
use crate::{Int, Preferences, Uint, UnexpectedFactor, Verbosity};

pub fn siqs(
    n: &Uint,
    k: u32, // multiplier
    prefs: &Preferences,
    tpool: Option<&rayon::ThreadPool>,
) -> Result<Vec<Uint>, UnexpectedFactor> {
    let (norig, n) = (n, n * Uint::from(k));
    let use_double = prefs.use_double.unwrap_or(n.bits() > 256);
    // Choose factor base. Sieve twice the number of primes
    // (n will be a quadratic residue for only half of them)
    let fb = prefs.fb_size.unwrap_or(fb_size(&n, use_double));
    let fbase = FBase::new(n, fb);
    if let Err(e) = fbase.check_divisors() {
        if prefs.verbose(Verbosity::Info) {
            eprintln!("Unexpected divisor {} in factor base", e.0);
        }
        return Err(e);
    }
    let mm = interval_size(&n);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Smoothness bound B1={}", fbase.bound());
        eprintln!("Factor base size {} ({:?})", fbase.len(), fbase.smalls(),);
        eprintln!("Sieving interval size {}k", mm >> 10);
    }

    // Generate all values of A now.
    let nfacs = nfactors(&n) as usize;
    let factors = select_siqs_factors(&fbase, &n, nfacs, mm as usize, prefs.verbosity);
    let a_ints = select_a(&factors, a_value_count(&n), prefs.verbosity);
    let polys_per_a = 1 << (nfacs - 1);
    if prefs.verbose(Verbosity::Info) {
        eprintln!(
        "Generated {} values of A with {} factors in {}..{} ({} polynomials each, spread={:.2}%)",
        a_ints.len(),
        nfacs,
        factors.factors[0].p,
        factors.factors.last().unwrap().p,
        polys_per_a,
        a_quality(&a_ints) * 100.0
    );
    }

    let maxprime = fbase.bound() as u64;
    let maxlarge: u64 = maxprime * prefs.large_factor.unwrap_or(large_prime_factor(&n));
    // Don't allow maxlarge to exceed 32 bits (it would not be very useful anyway).
    let maxlarge = min(maxlarge, (1 << 32) - 1);
    let maxdouble = if use_double {
        maxprime * maxprime * double_large_factor(&n)
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

    let s = SieveSIQS::new(&n, &fbase, maxlarge, maxdouble, mm as usize, prefs);

    // When using multiple threads, each thread will sieve a different A
    // to avoid breaking parallelism during 'prepare_a'.
    //
    // To avoid wasting CPU on very small inputs, completion is checked after
    // each polynomial to terminate the loop early.

    if let Some(pool) = tpool.as_ref() {
        pool.install(|| {
            a_ints.par_iter().for_each(|&a_int| {
                if s.gap.load(Ordering::Relaxed) == 0 {
                    return;
                }
                if s.done.load(Ordering::Relaxed) || prefs.abort() {
                    return;
                }
                sieve_a(&s, &a_int, &factors);
            });
        });
    } else {
        for a_int in a_ints {
            sieve_a(&s, &a_int, &factors);
            if s.gap.load(Ordering::Relaxed) == 0 {
                break;
            }
            if s.done.load(Ordering::Relaxed) || prefs.abort() {
                break;
            }
        }
    }
    if prefs.abort() {
        return Ok(vec![]);
    }
    let mut rels = s.rels.into_inner().unwrap();
    // Log final progress
    let pdone = s.polys_done.load(Ordering::Relaxed);
    if prefs.verbose(Verbosity::Info) {
        rels.log_progress(format!(
            "Sieved {}M {} polys",
            (pdone as u64 * mm as u64) >> 20,
            pdone,
        ));
    }
    if rels.len() > fbase.len() + relations::MIN_KERNEL_SIZE {
        rels.truncate(fbase.len() + relations::MIN_KERNEL_SIZE)
    }
    if s.gap.load(Ordering::Relaxed) != 0 && rels.len() <= fbase.len() {
        panic!("Internal error: not enough smooth numbers with selected parameters (n={n})");
    }
    let rels = rels.into_inner();
    if rels.len() == 0 {
        return Ok(vec![]);
    }
    Ok(relations::final_step(norig, &fbase, &rels, prefs.verbosity))
}

fn sieve_a(s: &SieveSIQS, a_int: &Uint, factors: &Factors) {
    let mm = s.interval_size;
    let a = &prepare_a(factors, a_int, s.fbase, -(mm as i64) / 2);
    if s.prefs.verbose(Verbosity::Debug) {
        eprintln!(
            "Sieving A={} (factors {})",
            a.a,
            a.factors
                .iter()
                .map(|item| item.p.to_string())
                .collect::<Vec<_>>()[..]
                .join("*")
        );
    }
    let nfacs = a.factors.len();
    let polys_per_a = 1 << (nfacs - 1);
    let mut pol = Poly::first(s, a);
    // Storage for recycled resources.
    let mut recycled = None;
    for idx in 0..polys_per_a {
        if s.done.load(Ordering::Relaxed) {
            // Interrupt early.
            return;
        }
        if idx > 0 {
            pol.next(s, a);
        }
        assert!(pol.idx == idx);
        recycled = Some(siqs_sieve_poly(s, a, &pol, recycled));
        // Check status.
        let rlen = {
            let rels = s.rels.read().unwrap();
            rels.len()
        };

        s.polys_done.fetch_add(1, Ordering::SeqCst);

        if rlen >= s.target.load(Ordering::Relaxed) {
            // unlikely: are we done yet?
            let rgap = {
                let rels = s.rels.read().unwrap();
                rels.gap(s.fbase)
            };
            s.gap.store(rgap, Ordering::Relaxed);
            if rgap == 0 {
                if s.prefs.verbose(Verbosity::Info) {
                    eprintln!("Found enough relations");
                }
                s.done.store(true, Ordering::Relaxed);
                return;
            } else {
                if s.prefs.verbose(Verbosity::Info) {
                    eprintln!("Need {rgap} additional relations");
                }
                s.target.store(
                    rlen + rgap + std::cmp::min(10, s.fbase.len() / 4),
                    Ordering::SeqCst,
                );
            }
        }
    }
    let pdone = s.polys_done.load(Ordering::Relaxed);
    let rels = s.rels.read().unwrap();
    if s.prefs.verbose(Verbosity::Info) {
        if !s.prefs.verbose(Verbosity::Verbose) && rels.len() % factors.nfacs != 0 {
            // Only log every nfacs A, should be enough,
            // and avoid threads colliding.
            return;
        }
        rels.log_progress(format!(
            "Sieved {}M {pdone} polys",
            (pdone as u64 * mm as u64) >> 20,
        ));
    }
}

/// Helper function to study performance impact of parameters
pub fn siqs_calibrate(n: Uint) {
    // Prepare central parameters and A values.
    let (k, score) = fbase::select_multiplier(n);
    let prefs = Preferences::default();
    eprintln!("Using fixed multiplier {k} (score {score:.2}/10)");
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

    eprintln!("Base params B1={} fb={fb0} lf={lf0}", fbase0.bound());
    // The value of a varies according to interval size.
    let mut factorss = vec![];
    let mut mms = vec![];
    let mut a_s = vec![];
    let dblk = max(1, blocks0 / 6);
    for blks in [
        blocks0 - 3 * dblk,
        blocks0 - 2 * dblk,
        blocks0 - dblk,
        blocks0,
        blocks0 + dblk,
        blocks0 + 2 * dblk,
    ] {
        if blks > 0 {
            // FIXME: functions have their own parameters.
            let mm = BLOCK_SIZE * (blks as usize);
            let factors = select_siqs_factors(&fbase0, n, nfacs, mm, prefs.verbosity);
            let a_ints = select_a(&factors, a_value_count(n), prefs.verbosity);
            let a0 = a_ints[0];
            let polys_per_a = 1 << (nfacs - 1);
            eprintln!("Test set M={}k A={} npolys={}", mm / 2048, a0, polys_per_a);
            // sample polynomial
            let a = &prepare_a(&factors, &a0, &fbase0, -(mm as i64) / 2);
            let b1 = fbase0.bound() as u64;
            let maxdouble = if use_double {
                b1 * b1 * double_large_factor(&n)
            } else {
                0
            };
            let s = SieveSIQS::new(n, &fbase0, 0, maxdouble, mm, &prefs);
            let pol = Poly::first(&s, a);
            eprintln!("min(P) ~ {}", pol.eval(0).0);
            eprintln!("max(P) ~ {}", pol.eval(mm as i64 / 2).0);
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
        for lf in [lf0] {
            for use_double in [false, true] {
                for (idx, &mm) in mms.iter().enumerate() {
                    let aint = a_s[idx];
                    let factors = &factorss[idx];
                    let maxprime = fbase.bound() as u64;
                    let maxlarge: u64 = maxprime * lf;
                    let maxdouble = if use_double {
                        maxprime * maxprime * double_large_factor(&n)
                    } else {
                        0
                    };
                    let df = maxdouble / maxprime / maxprime;
                    let s = SieveSIQS::new(n, &fbase, maxlarge, maxdouble, mm, &prefs);
                    // Measure metrics
                    let t0 = std::time::Instant::now();
                    let a = &prepare_a(factors, &aint, &fbase, -(mm as i64) / 2);
                    let mut pol = Poly::first(&s, a);
                    let mut recycled = None;
                    for idx in 0..polys_per_a {
                        if idx > 0 {
                            pol.next(&s, &a);
                        }
                        assert!(pol.idx == idx);
                        recycled = Some(siqs_sieve_poly(&s, a, &pol, recycled));
                    }
                    let dt = t0.elapsed().as_secs_f64();
                    let rels = s.rels.read().unwrap();
                    eprintln!(
                        "fb={fb} B2/B1={lf} D/B1²={df} M={}k dt={dt:2.3}s {:.2}ns/i c={} ({:.1}/s) p={} ({:.1}/s) pp={} ({:.1}/s)",
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

fn fb_size(n: &Uint, use_double: bool) -> u32 {
    // When using type 2 polynomials, values will be twice smaller
    // as if the size of n was down by 2 bits.
    let mut sz = params::factor_base_size(&if n.low_u64() % 8 == 1 { n >> 2 } else { *n });
    // Reduce factor base size when using large double primes
    // since they will cover the large prime space.
    if use_double {
        sz /= 2;
    }
    sz
}

// Number of polynomials: m * 2^(k-1)

// The number of factors of A is chosen to avoid very small primes
// (hurting ability to obtain a product in the correct range)
// but it also needs to fit the factor base size.
//
// 64..80 bits need 3 factors (A is 16-24 bits, maxprime < 1000)
//
// For inputs over 80 bits, max prime is above 1000
// k factors are fine for input size (15..20)*k + 30 or better (18..22)k + 30
// for factors between 200..1000 (A = sqrt(N) / M)
fn nfactors(n: &Uint) -> u32 {
    match n.bits() {
        0..=64 => 2,
        65..=89 => 3,
        90..=119 => 4,
        120..=149 => 5,
        150..=169 => 6,
        170..=199 => 7,
        // 13 factors for size 330
        // 14 factors for size 360
        200.. => n.bits() / 25,
    }
}

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
        72..=129 => sz / 5,           // 14..25
        130..=169 => sz - 60,         // 70..100
        170..=199 => 50 * (sz - 168), // 100..1000
        200.. => 100 * (sz - 190),    // 1000..5000 (sz=256).. 17000 (sz=360)
        _ => unreachable!("impossible"),
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

// Interval size and large prime bound determine the sieve performance:
//
// Factor base size (smoothness bound B)
// Determines the output levels of the sieve (exponentially low
// if B is below the ideal levels).
//
// When using single large prime (p < B*B1)
// the detection threshold is lowered to sqrt(n/2)/M/B1
// Single large primes disappear quickly (B1 > 400 doesn't yield
// more p-relations, B1 = 100 already gives a majority of p-relations)
//
// When using double large primes (p < B*B1, pq < B²*B1*B2 where B2 < B1)
// the detection threshold is lowered to sqrt(n/2)/M/(B*B1*B2)
// - all single large primes will be detected
// - the extra cost of smooth candidates is fully determined by B1*B2
// For a fixed product B1*B2, choosing a large B1 yields more p-relations,
// only limited by available memory. Experimentally double large primes
// are quickly depleted even with small B2.
//
// Since B is bounded by 2^24 and we want cofactors to fit in u64,
// B1*B2 must not exceed 2^16.

fn interval_size(n: &Uint) -> u32 {
    // Choose very small intervals since the cost of switching
    // polynomials is very small (less than 1ms).
    //
    // Due to fixed costs being proportional to factor base size,
    // choose interval size accordingly.
    let sz = n.bits();
    let nblocks = match sz {
        // For small integers, A values are scarce, choose a large
        // interval to avoid running out of values.
        0..=48 => 3,
        49..=130 => 2,
        131..=160 => 3,
        161..=190 => 4,
        191..=250 => (sz - 141) / 10, // 5..10
        251..=310 => (sz - 171) / 7,  // 11..19
        311..=370 => (sz - 210) / 5,  // 20..32
        _ => 33,
    };
    nblocks * sieve::BLOCK_SIZE as u32
}

fn large_prime_factor(n: &Uint) -> u64 {
    let sz = n.bits() as u64;
    match sz {
        // Smoothness density is very high for small sizes, do not use large primes.
        0..=96 => 1,
        97..=128 => 2,
        // More large primes to compensate fewer relations
        129..=250 => (sz * (sz - 100)) / 100, // 7..225
        251.. => {
            // For these input sizes, smooth numbers are so sparse
            // that we need very large cofactors, so any value of B1 is fine.
            //
            // Since cycles appear with frequency O(density^2) there is almost
            // no benefit for B1 > 100. Reducing B1 avoids storing too many
            // uninteresting pp-relations.
            //
            // Choose the same value as double_large_factor.
            2 * sz - 384
        }
    }
}

// Parameter D such that double large primes are bounded by D*B1^2
// where B1 is the factor base bound.
fn double_large_factor(n: &Uint) -> u64 {
    // We don't want double large prime to reach maxlarge^2
    // because p-relations are much less dense in the large prime area
    // and this creates extra factoring pressure.
    // See [Lentra-Manasse]
    //
    // In practice, the density of double primes decreases quickly and the number
    // of cycles is O(density^4) so all interesting double large primes are
    // quickly depleted after 100-200 B1^2
    //
    // For small values of n, it is enough to require than
    // double large primes are smaller than (10 B1)^2
    let sz = n.bits() as u64;
    match sz {
        0..=255 => sz / 2,
        256.. => 2 * sz - 384,
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

fn polytype(n: &Uint) -> PolyType {
    // Make sure to not use unoptimized BUint.mod
    if n.low_u64() % 4 == 1 {
        PolyType::Type2
    } else {
        PolyType::Type1
    }
}

/// Select factors of generated A values.
///
/// It is enough to select about twice the number of expected factors in
/// A, because the number of combinations is large enough to generate
/// values close to the target.
pub fn select_siqs_factors<'a>(
    fb: &'a FBase,
    n: &'a Uint,
    nfacs: usize,
    mm: usize,
    v: Verbosity,
) -> Factors<'a> {
    // For interval [-M,M] the target is sqrt(2N) / M, see [Pomerance].
    // Note that if N=1 mod 4, the target can be sqrt(N/2)/M
    // giving smaller numbers.
    // Don't go below 2000 for extremely small numbers.
    let target = max(
        Uint::from(2000u64),
        if polytype(n) == PolyType::Type2 {
            // Type 2
            arith::isqrt(n >> 1) / Uint::from(mm as u64 / 2)
        } else {
            // Type 1
            arith::isqrt(n << 1) / Uint::from(mm as u64 / 2)
        },
    );
    let idx = fb
        .primes
        .partition_point(|&p| Uint::from(p as u64).pow(nfacs as u32) < target);
    let selected_idx = if idx + nfacs >= fb.len() {
        if v >= Verbosity::Info {
            eprintln!("WARNING: suboptimal choice of A factors");
        }
        fb.len() - 2 * nfacs..fb.len()
    } else if idx > 4 * nfacs && idx + 4 * nfacs < fb.len() {
        idx - 2 * nfacs..idx + 2 * nfacs
    } else {
        idx - nfacs..idx + max(nfacs, 6)
    };
    // Make sure that selected factors don't divide n.
    // Also never take the first prime (usually 2).
    let selection: Vec<Prime> = selected_idx
        .filter(|&i| i > 0 && i < fb.len() && fb.div(i).mod_uint(n) != 0)
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
                arith::inv_mod64(p.p, q.p).unwrap()
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
///
/// The square roots are `sum(CRT[j] b[j])` with 2 choices for each `b[j]`.
/// Precompute `CRT[j] b[j]` modulo the factor base (F vectors)
/// then combine them for each polynomial (faster than computing 2^F vectors).
///
/// Beware: for type 2 polynomials (Ax^2+Bx+C) the roots are `(±r - B) / 2A`
/// and it is required that B is odd (B is a square root of n modulo 4A)
pub struct A<'a> {
    a: Uint,
    factors: Vec<Prime<'a>>,
    factors_idx: Box<[usize]>,
    // Base data for polynomials:
    // Precomputed bj = CRT[j] * sqrt(n) mod pj
    // bj^2 = n mod pj, 0 mod pi (i != j)
    //
    // To control the parity of B=sum(bj) it is required that
    // roots[0] are odd and roots[i>0] are even
    // (a multiple of A can be added if necessary).
    roots: Vec<[I256; 2]>,
    // Prepare root differences for the Gray code iteration.
    // We require that roots[i][0] <= roots[i][1]
    //
    // For each i, store deltas_mod_p[i][j] = -1/A (root1[i] - root0[i]) % p
    deltas_mod_p: Vec<Vec<u32>>,
    // To initialize polynomials, compute:
    // B0 mod pi - rp[i] - start_offset % pi
    root0_mod_p: Vec<u32>,
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
pub fn select_a(f: &Factors, want: usize, v: Verbosity) -> Vec<Uint> {
    // Sample deterministically products of W primes
    // closest to target and select best candidates.
    // We usually don't need more than 1000 values.
    //
    // We are going to select ~2^W best products of W primes among 2W

    let mut div = a_tolerance_divisor(f.n);
    assert!(div >= 3);
    let mut amin = f.target - f.target / div as u64;
    let mut amax = f.target + f.target / div as u64;

    if f.nfacs <= 3 {
        // Generate all products (quicker than random sampling)
        assert!(f.target.bits() < 60);
        let mut candidates: Vec<u64> = vec![];
        for f1 in &f.factors {
            let p1 = f1.p;
            for f2 in &f.factors {
                let p2 = f2.p;
                if p2 >= p1 {
                    break;
                }
                if f.nfacs == 2 {
                    candidates.push(p1 * p2);
                } else {
                    for f3 in &f.factors {
                        let p3 = f3.p;
                        if p3 >= p2 {
                            break;
                        }
                        candidates.push(p1 * p2 * p3);
                    }
                }
            }
        }
        let target = f.target.low_u64();
        candidates.sort_by_key(|&c| c.abs_diff(target));
        candidates.dedup();
        if candidates.len() > want {
            candidates.truncate(want);
        }
        candidates.sort();
        return candidates.into_iter().map(Uint::from).collect();
    }

    let mut rng: u64 = 0xcafebeefcafebeef;
    let fb = f.factors.len();
    let mut gen = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng % fb as u64
    };
    let mut candidates = BTreeSet::new();
    let mut iters = 0;
    while iters < 1000 * want || candidates.len() < want {
        iters += 1;
        if iters % (100 * want) == 0 && candidates.len() < want {
            if v >= Verbosity::Info {
                eprintln!("WARNING: unable to find suitable A, increasing tolerance");
            }
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
            candidates.insert(Uint::cast_from(product));
        }
        if candidates.len() > 2 * want && iters % 10 == 0 {
            let mut candidates = Vec::from_iter(candidates.into_iter());
            candidates.sort_by_key(|&c| U256::cast_from(c).abs_diff(f.target));
            candidates.truncate(want);
            candidates.sort();
            return candidates;
        }
    }
    // Should not happen? return what we found so far including farther values.
    let mut candidates = Vec::from_iter(candidates.into_iter());
    candidates.sort_by_key(|&c| U256::cast_from(c).abs_diff(f.target));
    candidates.dedup();
    if candidates.len() > want {
        candidates.truncate(want)
    }
    candidates.sort();
    candidates
}

pub fn prepare_a<'a>(f: &Factors<'a>, a: &Uint, fbase: &FBase, start_offset: i64) -> A<'a> {
    assert!(a.bits() < 255);
    let afactors: Vec<(usize, &Prime)> = f
        .factors
        .iter()
        .enumerate()
        .filter(|(_, p)| p.div.mod_uint(a) == 0)
        .collect();
    // Compute A/pi = product(pj for j != i)
    // And also pinv[i] = product(pj^1 for j != i) mod pi
    let mut p_over_pi = vec![];
    let mut pinv = vec![];
    for &(idx, p) in afactors.iter() {
        let mut c = Uint::one();
        let mut inv = 1;
        for &(jdx, q) in afactors.iter() {
            if jdx != idx {
                c *= Uint::from(q.p);
                inv = (inv * f.inverses[jdx][idx] as u64) % p.p;
            }
        }
        p_over_pi.push(c);
        pinv.push(inv);
    }
    // Compute modular inverses of A or 1 for prime factors of A.
    let mut ainv = vec![];
    let mut factors_idx = vec![];
    let a2a = if polytype(f.n) == PolyType::Type1 {
        U256::cast_from(*a)
    } else {
        U256::cast_from(*a) << 1
    };
    for pidx in 0..fbase.len() {
        let p = fbase.p(pidx);
        let div = fbase.div(pidx);
        let amod = div.mod_uint(&a2a);
        // We have made sure that 2 never divides A.
        if amod == 0 && p != 2 {
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
    // For type 2 polynomials we require that B is odd!
    // To achieve that, choose roots all having the same parity (odd for i=0, even for i>0)
    let mut roots = vec![];
    let mut deltas_mod_p = vec![];
    let mut root0 = Uint::ZERO;
    for (i, (_, f)) in afactors.iter().enumerate() {
        // The CRT basis is pinv[i]*p_over_pi[i]
        let k: u64 = (f.r * pinv[i]) % f.p;
        let r = Uint::from(k) * p_over_pi[i];
        // Parity is 1 if r is odd (reverse if i == 0)
        let parity = r.bit(0) ^ (i == 0);
        let (r0, r1) = if parity {
            (a - r, a + r)
        } else {
            (r, (a << 1) - r)
        };
        debug_assert!(r0 <= r1);
        roots.push([I256::cast_from(r0), I256::cast_from(r1)]);
        root0 += r0;
        let mut deltas = Vec::with_capacity(fbase.len());
        let d = r1 - r0;
        for pidx in 0..fbase.len() {
            let p = fbase.p(pidx);
            let pdiv = fbase.div(pidx);
            // Compute -(r1-r0)/a
            let d_over_a = {
                let dmod = pdiv.mod_uint(&d);
                pdiv.divmod64(dmod * ainv[pidx] as u64).1 as u32
            };
            deltas.push(if d_over_a == 0 { 0 } else { p - d_over_a });
        }
        deltas_mod_p.push(deltas);
    }
    debug_assert!((root0 * root0) % a == f.n % a);
    // Compute root0 - rp[i] - start_offset
    let mut root0_mod_p = Vec::with_capacity(fbase.len());
    for pidx in 0..fbase.len() {
        let pdiv = fbase.div(pidx);
        let r0 = pdiv.mod_uint(&root0);
        let r0_over_a = pdiv.divmod64(r0 * ainv[pidx] as u64).1 as i32;
        let val = -r0_over_a - rp[pidx] as i32 - start_offset as i32;
        root0_mod_p.push(pdiv.div31.modi32(val) as u32);
    }
    A {
        a: *a,
        factors: afactors.into_iter().map(|(_, p)| p).cloned().collect(),
        factors_idx: factors_idx.into_boxed_slice(),
        roots,
        deltas_mod_p,
        root0_mod_p,
        rp,
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum PolyType {
    Type1,
    Type2,
}

#[derive(Debug)]
pub struct Poly {
    idx: usize,
    kind: PolyType,
    a: I256,
    b: I256,
    c: I256,
    // Rounded root of the polynomial
    root: u32,
    // Precomputed roots
    r1p: Box<[u32]>,
    r2p: Box<[u32]>,
    n: Uint,
}

impl Poly {
    // Returns v, y such that:
    // P(x) = v
    // y^2 = A P(x) mod n
    // For type 1, y = Ax + B
    // For type 2, y = (Ax + B/2)
    //
    // v is always small (candidate to be smooth)
    // y can be large (due to halving modulo n)
    fn eval(&self, x: i64) -> (I256, Int) {
        let x = I256::from(x);
        if self.kind == PolyType::Type1 {
            // Evaluate polynomial Ax^2 + 2Bx+ C
            let ax_b = self.a * x + self.b;
            let v = (ax_b + self.b) * x + self.c;
            (v, Int::cast_from(ax_b))
        } else {
            // Evaluate polynomial Ax^2 + Bx+ C
            let ax = self.a * x;
            let v = (ax + self.b) * x + self.c;
            // Build ax + (n+b)//2 (b is odd)
            let b_half = (Int::cast_from(self.b) + Int::cast_from(self.n)) >> 1;
            let y = Int::cast_from(ax) + b_half;
            (v, y)
        }
    }

    /// First polynomial in the family determined by A.
    #[doc(hidden)]
    pub fn first(s: &SieveSIQS, a: &A) -> Poly {
        let mut b = I256::ZERO;
        for i in 0..a.factors.len() {
            b += a.roots[i][0];
        }
        let typ = polytype(s.n);
        if typ == PolyType::Type2 {
            assert!(b.bit(0));
        }
        // Compute polynomial roots
        // (±r - B)/A = (-B/A) ± (r/A) solutions of (Ax+B)^2 = N
        //
        // root0 contains precomputed: (-b-r)/a - offset
        let r1p = a.root0_mod_p.clone();
        let mut r2p = r1p.clone();
        for i in 0..s.fbase.len() {
            let p = s.fbase.p(i);
            r2p[i] += 2 * a.rp[i];
            while r2p[i] >= p {
                r2p[i] -= p;
            }
        }
        let mut pol = Poly {
            idx: 0,
            kind: typ,
            a: I256::cast_from(a.a),
            b: I256::cast_from(b),
            c: I256::ZERO,
            root: 0,
            r1p: r1p.into_boxed_slice(),
            r2p: r2p.into_boxed_slice(),
            n: *s.n,
        };
        _finish_polynomial(s, a, &mut pol);
        pol
    }
    /// Given coefficients A and B, compute all roots (Ax+B)^2=N
    /// modulo the factor base, computing (r - B)/A mod p
    ///
    /// We iterate only up to 2*(nfacs-1) so that one bit is constant.
    #[doc(hidden)]
    pub fn next(&mut self, s: &SieveSIQS, a: &A) {
        // Advance to next polynomial.
        // This is done by adding a single delta.
        let prev_idx = self.idx;
        let next_idx = prev_idx + 1;
        self.idx = next_idx;
        let prev_gray = prev_idx ^ (prev_idx >> 1);
        let next_gray = next_idx ^ (next_idx >> 1);
        let bit = (prev_gray ^ next_gray).trailing_zeros() as usize;
        assert!(next_gray == prev_gray ^ (1 << bit));

        // Flip requested bit/root
        let delta_mod = &a.deltas_mod_p[bit][..];
        let primes = &s.fbase.primes[..];
        if (prev_gray >> bit) & 1 == 0 {
            // Flip bit from 0 to 1
            self.b = self.b + a.roots[bit][1] - a.roots[bit][0];
            // This loop should be vectorized by LLVM.
            // SSE2/AVX2/NEON have native packed min/max
            for i in 0..delta_mod.len() {
                let (d, p) = (delta_mod[i], primes[i]);
                let (r1, r2) = (self.r1p[i] + d, self.r2p[i] + d);
                self.r1p[i] = min(r1, r1.wrapping_sub(p));
                self.r2p[i] = min(r2, r2.wrapping_sub(p));
            }
        } else {
            // Flip bit from 1 to 0
            self.b = self.b + a.roots[bit][0] - a.roots[bit][1];
            // This loop should be vectorized by LLVM.
            for i in 0..delta_mod.len() {
                let (d, p) = (delta_mod[i], primes[i]);
                let r1 = self.r1p[i].wrapping_sub(d);
                let r2 = self.r2p[i].wrapping_sub(d);
                self.r1p[i] = min(r1, r1.wrapping_add(p));
                self.r2p[i] = min(r2, r2.wrapping_add(p));
            }
            if self.kind == PolyType::Type2 {
                assert!(self.b.bit(0));
            }
        }
        _finish_polynomial(s, a, self);
    }
}

// Complete polynomial information from a modular root B and computed
// roots mod p.
//
// Fields c, root and r1p/r2p for factors of A are populated.
fn _finish_polynomial(s: &SieveSIQS, a: &A, pol: &mut Poly) {
    let n = pol.n;
    let polytype = pol.kind;
    // Compute c such that b^2 - ac = N
    // (Ax+B)^2 - n = A(Ax^2 + 2Bx + C)
    assert!(pol.b.is_positive());
    let b = Uint::cast_from(pol.b);
    let c = if polytype == PolyType::Type1 {
        debug_assert!((b * b) % a.a == n % a.a);
        (Int::cast_from(b * b) - Int::from_bits(n)) / Int::from_bits(a.a)
    } else {
        debug_assert!((b * b) % (a.a << 2) == n % (a.a << 2));
        (Int::cast_from(b * b) - Int::from_bits(n)) / Int::from_bits(a.a << 2)
    };

    // Manually repair roots modulo 2 for Type 2
    if polytype == PolyType::Type2 && s.fbase.p(0) == 2 {
        // A and B are odd. If C is even, 0 and 1 are roots.
        if c.abs().to_bits() % 2_u64 == 0 {
            pol.r1p[0] = 0;
            pol.r2p[0] = 1;
        }
    }

    // Special case for divisors of A.
    // poly % p = 2Bx + C, root is -C/2B
    // type2:
    // poly % p = Bx + C, root is -C/B
    for &pidx in a.factors_idx.iter() {
        let div = &s.fbase.div(pidx);
        let p = s.fbase.p(pidx);
        let bp = div.mod_uint(&b);
        let mut cp = div.mod_uint(&c.abs().to_bits());
        if !c.is_negative() {
            cp = p as u64 - cp;
        }
        let multiplier = if polytype == PolyType::Type1 { 2 } else { 1 };
        let Some(binv) = arith::inv_mod64(multiplier * bp, p as u64)
            else { unreachable!("no inverse of b={bp} mod p={p}") };
        let r = div.divmod64(cp * binv).1 as u32;
        let off = s.offset_modp[pidx];
        let r = div.div31.modu31(r + p - off);
        pol.r1p[pidx] = r;
        pol.r2p[pidx] = r;
    }
    // Each r1p,r2p[idx] is a root of P(x+offset) modulo p[idx]

    // The rounded root may be wrong by an offset |B/A| which is bounded
    // by 2 * #factors(A)
    // This is because we don't reduce B modulo A.
    let root = if polytype == PolyType::Type1 {
        (s.nsqrt / a.a).low_u64()
    } else {
        (s.nsqrt / (a.a << 1) as Uint).low_u64()
    };
    if a.factors.len() >= 5 {
        // For small n, A factors are below 200 and can have huge gaps.
        // Optimal A's break down for very small n.
        // We assume than 5 factors correspond to nice factor ranges
        // where we can never be far away from the optimal value.
        assert!(
            (root as usize) < s.interval_size / 2,
            "A={} root={}",
            a.a,
            root
        );
    }

    // All polynomial values are assumed to fit in 256 bits:
    // A has the size of sqrt(n)/M
    // B has the size of A and is always positive
    // C has the size of sqrt(n)M
    let mlog = usize::BITS - usize::leading_zeros(s.interval_size);
    assert!(a.a.bits() + 2 * mlog < 255);
    assert!(pol.b.bits() + mlog < 255);
    assert!(pol.c.abs().bits() < 255);
    pol.c = I256::cast_from(c);
    pol.root = root as u32;
}

// Sieving process

fn siqs_sieve_poly(
    s: &SieveSIQS,
    a: &A,
    pol: &Poly,
    rec: Option<sieve::SieveRecycle>,
) -> sieve::SieveRecycle {
    let mm = s.interval_size;
    let nblocks: usize = mm / BLOCK_SIZE;
    if s.prefs.verbose(Verbosity::Debug) {
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
    let r1p = &pol.r1p[..];
    let r2p = &pol.r2p[..];
    let mut state = sieve::Sieve::new(start_offset, nblocks, s.fbase, [r1p, r2p], rec);
    if nblocks == 0 {
        sieve_block_poly(s, pol, a, &mut state);
    }
    while state.offset < end_offset {
        sieve_block_poly(s, pol, a, &mut state);
        state.next_block();
    }
    state.recycle()
}

pub struct SieveSIQS<'a> {
    pub n: &'a Uint,
    nsqrt: Uint,
    pub interval_size: usize,
    pub fbase: &'a FBase,
    pub maxlarge: u64,
    pub maxdouble: u64,
    pub rels: RwLock<RelationSet>,
    pub offset_modp: Box<[u32]>,
    // A signal for threads to stop sieving.
    pub done: AtomicBool,
    // Progress trackers
    polys_done: AtomicUsize,
    gap: AtomicUsize,
    target: AtomicUsize,
    prefs: &'a Preferences,
}

impl<'a> SieveSIQS<'a> {
    pub fn new(
        n: &'a Uint,
        fb: &'a FBase,
        maxlarge: u64,
        maxdouble: u64,
        interval_size: usize,
        prefs: &'a Preferences,
    ) -> Self {
        let start_offset = -(interval_size as i64 / 2);
        let mut offsets = vec![0u32; (fb.len() + 15) & !15].into_boxed_slice();
        assert_eq!(offsets.len() % 16, 0);
        for idx in 0..fb.len() {
            let off = fb.div(idx).modi64(start_offset) as u32;
            offsets[idx] = off;
        }
        let fb_size = fb.len();
        SieveSIQS {
            n,
            nsqrt: arith::isqrt(*n),
            interval_size,
            fbase: fb,
            rels: RwLock::new(RelationSet::new(*n, fb_size, maxlarge)),
            maxlarge,
            maxdouble,
            offset_modp: offsets,
            done: AtomicBool::new(false),
            polys_done: AtomicUsize::new(0),
            gap: AtomicUsize::new(fb_size),
            target: AtomicUsize::new(fb_size * 8 / 10),
            prefs,
        }
    }
}

// Sieve using a selected polynomial
fn sieve_block_poly(s: &SieveSIQS, pol: &Poly, a: &A, st: &mut sieve::Sieve) {
    st.sieve_block();

    let maxprime = s.fbase.bound() as u64;
    let maxlarge = s.maxlarge;
    assert!(maxlarge == (maxlarge as u32) as u64);
    let max_cofactor: u64 = if s.maxdouble > maxprime * maxprime {
        s.maxdouble
    } else if maxlarge > maxprime {
        maxlarge
    } else {
        1 // Do not use the large prime variation.
    };
    // Polynomial values range from [-m sqrt(2n), m sqrt(2n)] so they have variable size.
    // The smallest values are about A which is sqrt(2n) / m
    // If the target is too low, the sieve will be slow.
    let msize = if pol.kind == PolyType::Type1 {
        s.interval_size as u64 / 2
    } else {
        // Values are smaller (M/2 sqrt(n)) for type 2.
        s.interval_size as u64 / 4
    };
    let target = s.n.bits() / 2 + msize.bits() - max_cofactor.bits();

    let n = s.n;
    let (idx, facss) = st.smooths(target as u8, Some(pol.root), [&pol.r1p, &pol.r2p]);
    for (i, facs) in idx.into_iter().zip(facss) {
        let (v, y) = pol.eval(st.offset + (i as i64));
        // xrel^2 = (Ax+B)^2 = A * v mod n
        // v is never divisible by A
        let Some(((p, q), mut factors)) = fbase::cofactor(
            s.fbase, &v, &facs,
            maxlarge, s.maxdouble > maxprime * maxprime)
            else { continue };
        let pq = if q > 1 { Some((p, q)) } else { None };
        let cofactor = p * q;
        // Complete with factors of A
        for f in &a.factors {
            if let Some(idx) = factors.iter().position(|&(p, _)| p as u64 == f.p) {
                factors[idx].1 += 1;
            } else {
                factors.push((f.p as i64, 1));
            }
        }
        let mut x = y.abs().to_bits();
        if &x > n {
            x %= n; // should not happen?
        }
        if s.prefs.verbose(Verbosity::Debug) {
            eprintln!("x={x} smooth {v} cofactor {cofactor}");
        }
        assert!(
            cofactor == 1 || cofactor > maxprime,
            "invalid cofactor {}",
            cofactor
        );
        let rel = Relation {
            x,
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

        let facs = select_siqs_factors(&fb, n, nfacs as usize, 256 << 10, Verbosity::Info);
        let target = Uint::cast_from(facs.target);

        let a_vals = select_a(&facs, want, Verbosity::Info);
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
            // 2% tolerance
            assert!(d1 - 100.0 < 2.0);
            assert!(d2 - 100.0 < 2.0);
            assert!(a_vals.len() >= want);
        } else {
            assert!(d1 - 100.0 < 1.0);
            assert!(d2 - 100.0 < 1.0);
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
    let mm = 1_usize << 20;
    let prefs = Preferences::default();
    let s = SieveSIQS::new(&n, &fb, fb.bound() as u64, 0, mm, &prefs);
    // Prepare A values
    // Only test 10 A values and 35 polynomials per A.
    let f = select_siqs_factors(&fb, &n, 9, mm, prefs.verbosity);
    let start_offset = -(mm as i64) / 2;
    let a_ints = select_a(&f, 10, prefs.verbosity);
    for a_int in &a_ints {
        let a = prepare_a(&f, a_int, &fb, start_offset);
        // Check CRT coefficients.
        assert_eq!(a.a, a.factors.iter().map(|x| Uint::from(x.p)).product());
        for (i, r) in a.roots.iter().enumerate() {
            for (j, p) in a.factors.iter().enumerate() {
                let &[r1, r2] = r;
                let (r1, r2) = (Uint::cast_from(r1), Uint::cast_from(r2));
                if i == j {
                    assert_eq!((r1 * r1) % p.p, n % p.p);
                    assert_eq!((r2 * r2) % p.p, n % p.p);
                } else {
                    assert_eq!(r1 % p.p, 0);
                    assert_eq!(r2 % p.p, 0);
                }
            }
        }
        let mut pol = Poly::first(&s, &a);
        for idx in 0..7 * 35 {
            // Generate and check several polynomial.
            if idx > 0 {
                pol.next(&s, &a);
            }
            assert!(pol.idx == idx);
            if idx % 7 != 0 {
                continue;
            }
            let (pa, pb) = (pol.a, pol.b);
            let pa = Int::cast_from(pa);
            let pb = Int::cast_from(pb);
            // B is a square root of N modulo A.
            assert_eq!((pb * pb) % pa, Int::from_bits(n) % pa);
            // Check that (Ax+B)^2 - n = A(Ax^2+2Bx+C)
            for x in [1u64, 100, 50000] {
                let (v, y) = pol.eval(x as i64);
                let u = Int::cast_from(y) * Int::cast_from(y);
                let v = Int::cast_from(v);
                if pol.kind == PolyType::Type1 {
                    assert_eq!(u, pa * v);
                } else {
                    // Type2: (Ax+B/2)^2 - n = A(Ax^2 + Bx + (B^2-4n)/4A)
                    assert_eq!(
                        u % Int::from_bits(n),
                        (pa * v).rem_euclid(Int::from_bits(n))
                    );
                }
            }
            for pidx in 0..fb.len() {
                // To prevent test runtime blowup, only test first 100 primes
                // and then every 100th prime.
                if pidx >= 100 && pidx % 100 != 0 {
                    continue;
                }
                // Roots are roots of Ax^2+2Bx+C modulo p.
                for r in [pol.r1p[pidx], pol.r2p[pidx]] {
                    let v = pol.eval(start_offset + r as i64).0;
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
