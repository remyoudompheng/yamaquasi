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

use std::cmp::{max, min};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::RwLock;

use bnum::cast::CastFrom;
use num_traits::One;
use rayon::prelude::*;

use crate::arith::{self, Num};
use crate::fbase::{self, FBase, Prime};
use crate::params::{self, BLOCK_SIZE};
use crate::relations::{Relation, RelationSet};
use crate::sieve;
use crate::{Int, Uint, DEBUG};

pub fn siqs(
    n: &Uint,
    prefs: &params::Preferences,
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

    let mlog = interval_logsize(&n);
    if mlog >= 20 {
        eprintln!("Sieving interval size {}M", 2 << (mlog - 20));
    } else {
        eprintln!("Sieving interval size {}k", 2 << (mlog - 10));
    }

    // Generate all values of A now.
    let nfacs = nfactors(n) as usize;
    let factors = select_siqs_factors(&fbase, n, nfacs);
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
    let maxlarge: u64 = maxprime * prefs.large_factor.unwrap_or(large_prime_factor(&n));
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
    let start_offset: i64 = -(1 << mlog);
    let mut pdata = vec![];
    for idx in 0..fbase.len() {
        let p = fbase.prime(idx);
        let off = p.div.modi64(start_offset) as u64;
        let arith::Divider31 { p, m31, s31 } = p.div.div31;
        pdata.push(SieveSIQS::pack_pdata(off, p, s31, m31));
    }

    let s = SieveSIQS {
        n,
        fbase: &fbase,
        rels: RwLock::new(RelationSet::new(*n, maxlarge)),
        maxlarge,
        use_double,
        pdata,
    };

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
        let a = &prepare_a(&factors, a_int, &fbase);
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
                    let pol = make_polynomial(n, a, idx);
                    siqs_sieve_poly(&s, n, a, &pol);
                    handle_result();
                });
            })
        } else {
            // Single-threaded
            for idx in 0..polys_per_a {
                let pol = make_polynomial(n, a, idx);
                siqs_sieve_poly(&s, n, a, &pol);
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
            ((pdone) << (mlog + 1 - 10)) >> 10,
            pdone,
        ));
        if gap.load(Ordering::Relaxed) == 0 {
            break;
        }
    }
    if gap.load(Ordering::Relaxed) != 0 {
        panic!("Internal error: not enough smooth numbers with selected parameters");
    }
    s.rels.into_inner().unwrap().into_inner()
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
        70..=99 => 3,
        100..=129 => 4,
        130..=149 => 5,
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

fn interval_logsize(n: &Uint) -> u32 {
    // Choose very small intervals since the cost of switching
    // polynomials is very small (less than 1ms).
    // Large intervals also hurt memory locality during sieve.
    let sz = n.bits();
    match sz {
        0..=119 => 15,
        // 270 bits => we want mlog=18
        // 300 bits => we want mlog=19
        120..=330 => 14 + sz / 60, // 16..19
        _ => 20,
    }
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
    pub target: Uint,
    pub nfacs: usize,
    // A sorted list of factors
    pub factors: Vec<Prime<'a>>,
    // inverses[i][j] = pi^-1 mod pj
    pub inverses: Vec<Vec<u32>>,
}

// Select factors of generated A values. It is enough to select about
// twice the number of expected factors in A, because the number of
// combinations is large enough to generate values close to the target.
pub fn select_siqs_factors<'a>(fb: &'a FBase, n: &'a Uint, nfacs: usize) -> Factors<'a> {
    let mlog = interval_logsize(n);
    // For interval [-M,M] the target is sqrt(2N) / M, see [Pomerance].
    // Don't go below 2000 for extremely small numbers.
    let target = max(Uint::from(2000u64), arith::isqrt(n << 1) >> mlog);
    let idx = fb
        .primes
        .partition_point(|&p| Uint::from(p as u64).pow(nfacs as u32) < target);
    // This may fail for very small n.
    assert!(idx > nfacs && idx + nfacs < fb.len());
    let selected_idx = if idx > 4 * nfacs && idx + 4 * nfacs < fb.len() {
        idx - 2 * nfacs as usize..idx + 2 * nfacs as usize
    } else {
        idx - nfacs as usize..idx + max(nfacs, 6) as usize
    };
    let selection: Vec<Prime> = selected_idx.map(|i| fb.prime(i)).collect();
    // Precompute inverses
    let mut inverses = vec![];
    for p in &selection {
        let mut row = vec![];
        for q in &selection {
            let pinvq = if p.p == q.p {
                0
            } else {
                q.div.inv(p.p).unwrap()
            };
            row.push(pinvq as u32);
        }
        inverses.push(row);
    }
    Factors {
        n,
        target,
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
    factor_min: u32,
    factor_max: u32,
    // Base data for polynomials:
    // Precomputed bj = CRT[j] * sqrt(n) mod pj
    // bj^2 = n mod pj, 0 mod pi (i != j)
    roots: Vec<[Uint; 2]>,
    // roots_mod_j[2i+1] is roots[j][i] mod factor base
    // The roots are pre-multiplied by ainv to make preparation easier.
    roots_mod_p: Vec<Vec<u32>>,
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

    let div = a_tolerance_divisor(f.n);
    let amin = f.target - f.target / div as u64;
    let amax = f.target + f.target / div as u64;

    let mut rng: u64 = 0xcafebeefcafebeef;
    let fb = f.factors.len();
    let mut gen = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng % fb as u64
    };
    let mut candidates = vec![];
    for i in 0..100 * want {
        let mut product = Uint::one();
        let mut mask = 0u64;
        while mask.count_ones() < f.nfacs as u32 - 1 {
            let g = gen();
            if mask & (1 << g) == 0 {
                mask |= 1 << g;
                product *= Uint::from(f.factors[g as usize].p);
            }
        }
        let t = (f.target / product).to_u64().unwrap();
        let idx = (0usize..fb)
            .filter(|g| mask & (1 << g) == 0)
            .min_by_key(|&idx| (f.factors[idx].p as i64 - t as i64).abs())
            .unwrap();
        product *= Uint::from(f.factors[idx].p);
        if amin < product && product < amax {
            candidates.push(product);
        }
        if candidates.len() > want && i % 10 == 0 {
            candidates.sort();
            candidates.dedup();
            let idx = candidates.partition_point(|c| c < &f.target);
            if idx > want && idx + want < candidates.len() {
                return candidates[idx - want / 2..idx + want / 2].to_vec();
            }
        }
    }
    // Should not happen?
    candidates.sort();
    candidates.dedup();
    let idx = candidates.partition_point(|c| c < &f.target);
    candidates[idx - min(idx, want / 2)..min(candidates.len(), idx + want / 2)].to_vec()
}

pub fn prepare_a<'a>(f: &Factors<'a>, a: &Uint, fbase: &FBase) -> A<'a> {
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
                c *= Uint::from((q.p as u64) * (f.inverses[jdx][idx] as u64));
                debug_assert!(c % q.p == 0);
                debug_assert!(c % p.p == 1);
            }
        }
        crt.push(c % a);
    }
    // Compute modular inverses of A or 1 for prime factors of A.
    let mut ainv = vec![];
    for pidx in 0..fbase.len() {
        let div = fbase.div(pidx);
        let amod = div.mod_uint(a);
        ainv.push(div.inv(amod).unwrap_or(1) as u32);
    }
    // Compute basic roots
    let mut roots = vec![];
    for i in 0..afactors.len() {
        let r1 = afactors[i].1.r;
        let r2 = afactors[i].1.p - afactors[i].1.r;
        roots.push([(crt[i] * Uint::from(r1)) % a, (crt[i] * Uint::from(r2)) % a]);
    }
    // Compute roots mod p.
    let mut roots_mod_p = vec![];
    for i in 0..afactors.len() {
        for j in 0..2 {
            let b = arith::U256::cast_from(roots[i][j]);
            let mut v = vec![0u32; (fbase.len() + 15) & !15];
            assert!(v.len() % 16 == 0);
            for pidx in 0..fbase.len() {
                let pdiv = fbase.div(pidx);
                let b_over_a = b * arith::U256::from(ainv[pidx]);
                v[pidx] = pdiv.mod_uint(&b_over_a) as u32;
            }
            roots_mod_p.push(v);
        }
    }
    // Compute sqrt(n)/A mod p
    let mut rp = vec![];
    for pidx in 0..fbase.len() {
        let r = fbase.r(pidx);
        let pdiv = fbase.div(pidx);
        rp.push(pdiv.divmod64(ainv[pidx] as u64 * r as u64).1 as u32);
    }
    let factor_min = afactors[0].1.p as u32;
    let factor_max = afactors.last().unwrap().1.p as u32;
    A {
        a: *a,
        factors: afactors.into_iter().map(|(_, p)| p).cloned().collect(),
        factor_min,
        factor_max,
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
    // Precomputed B/A mod p
    bmodp: Vec<u32>,
}

impl Poly {
    #[inline]
    pub fn prepare_prime(
        &self,
        idx: usize,
        fbase: &FBase,
        div31: &arith::Divider31,
        off: u32,
        a: &A,
    ) -> sieve::SievePrime {
        let p32 = div31.p;
        let shift = |r: u32| -> u32 {
            if r < off as u32 {
                r + p32 - off
            } else {
                r - off
            }
        };

        // Compute polynomial roots.
        let offsets = if div31.p == 2 {
            // A x^2 + C
            let c2 = self.c.low_u64() & 1;
            [Some(shift(c2 as u32)), None]
        } else if p32 < a.factor_min
            || p32 > a.factor_max
            || !a.factors.iter().any(|q| q.p == p32 as u64)
        {
            // Compute (±r - B)/A = (-B/A) ± (r/p) solutions of (Ax+B)^2 = N
            let b_div_a = p32 - div31.modi32(self.bmodp[idx] as i32) as u32;
            let rp = a.rp[idx];
            [
                Some(shift({
                    let x = b_div_a + rp;
                    if x >= p32 {
                        x - p32
                    } else {
                        x
                    }
                })),
                Some(shift({
                    if rp <= b_div_a {
                        b_div_a - rp
                    } else {
                        b_div_a + p32 - rp
                    }
                })),
            ]
        } else {
            // p is a factor of A.
            // 2Bx + C, root is -C/2B
            let bmodp = self.bmodp[idx] as u64;
            let div = &fbase.divs[idx];
            let mut cp = div.mod_uint(&self.c.abs().to_bits());
            if !self.c.is_negative() {
                cp = p32 as u64 - cp;
            }
            let r = div.divmod64(cp * div.inv(2 * bmodp as u64).unwrap()).1 as u32;
            [Some(shift(r)), None]
        };
        sieve::SievePrime { p: p32, offsets }
    }
}

/// Given coefficients A and B, compute all roots (Ax+B)^2=N
/// modulo the factor base, computing (r - B)/A mod p
pub fn make_polynomial(n: &Uint, a: &A, idx: usize) -> Poly {
    use wide::u32x8;

    let idx = idx << 1;
    // Combine roots: don't reduce modulo n.
    // This allows combining the roots modulo the factor base,
    // with the issue that (Ax+B) is no longer minimal for x=0
    // but for round(-B/A) which is extremely small.
    let mut b = Uint::ZERO;
    for i in 0..a.factors.len() {
        b += a.roots[i][(idx >> i) & 1];
    }
    // Also combine roots mod p
    let first = idx & 1;
    let mut bmodp = a.roots_mod_p[first].clone();
    for i in 1..a.factors.len() {
        let v = &a.roots_mod_p[2 * i + ((idx >> i) & 1)];
        assert_eq!(v.len() % 8, 0);
        // first += v
        let mut idx = 0;
        while idx < v.len() {
            unsafe {
                // (possibly) unaligned pointers
                let v8 = (v.get_unchecked(idx) as *const u32) as *const u32x8;
                let r8 = (bmodp.get_unchecked_mut(idx) as *const u32) as *mut u32x8;
                *r8 += *v8;
            }
            idx += 8;
        }
    }

    debug_assert!((b * b) % a.a == n % a.a);
    // Compute c such that b^2 - ac = N
    // (Ax+B)^2 - n = A(Ax^2 + 2Bx + C)
    let c = (Int::from_bits(b * b) - Int::from_bits(*n)) / Int::from_bits(a.a);
    debug_assert!(Int::from_bits(*n) == Int::from_bits(b * b) - c * Int::from_bits(a.a));
    // n has at most 512 bits, and b < sqrt(n)
    assert!(b.bits() < 256);
    Poly {
        a: a.a,
        b,
        c,
        bmodp,
    }
}

// Sieving process

fn siqs_sieve_poly(s: &SieveSIQS, n: &Uint, a: &A, pol: &Poly) -> () {
    let mlog = interval_logsize(&n);
    let nblocks: usize = (2 << mlog) / BLOCK_SIZE;
    if DEBUG {
        eprintln!(
            "Sieving polynomial A={} B={} M=2^{} blocks={}",
            pol.a, pol.b, mlog, nblocks
        );
    }

    // Construct initial state.
    let start_offset: i64 = -(1 << mlog);
    let end_offset: i64 = 1 << mlog;
    let pfunc = move |pidx| {
        let (off, div31) = SieveSIQS::unpack_pdata(s.pdata[pidx]);
        pol.prepare_prime(pidx, s.fbase, &div31, off as u32, a)
    };
    let mut state = sieve::Sieve::new(start_offset, nblocks, s.fbase, &pfunc);
    if nblocks == 0 {
        sieve_block_poly(s, &pol, a, &mut state);
    }
    while state.offset < end_offset {
        sieve_block_poly(s, &pol, a, &mut state);
        state.next_block();
    }
}

struct SieveSIQS<'a> {
    n: &'a Uint,
    fbase: &'a FBase,
    maxlarge: u64,
    use_double: bool,
    rels: RwLock<RelationSet>,
    pdata: Vec<(u64, u32)>,
}

impl<'a> SieveSIQS<'a> {
    #[inline]
    fn pack_pdata(off: u64, p: u32, s31: u32, m31: u32) -> (u64, u32) {
        (
            (off as u64) << 32 | (p as u64) << 8 | s31 as u64,
            m31 as u32,
        )
    }

    #[inline]
    fn unpack_pdata(t: (u64, u32)) -> (u64, arith::Divider31) {
        let (x, m31) = t;
        let off = x >> 32;
        let p = (x as u32) >> 8;
        let s31 = x as u8;
        (
            off,
            arith::Divider31 {
                p,
                m31,
                s31: s31 as u32,
            },
        )
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

    let target = s.n.bits() / 2 + interval_logsize(&s.n) - max_cofactor.bits();
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
        let pq = fbase::try_factor64(cofactor);
        match pq {
            Some((p, q)) if p > maxlarge || q > maxlarge => continue,
            None if cofactor > maxlarge => continue,
            _ => {}
        }
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
        assert!(cofactor == 1 || cofactor > maxprime as u64);
        let rel = Relation {
            x: xrel,
            cofactor,
            factors,
            pp: false,
        };
        debug_assert!(rel.verify(&s.n));
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

        let facs = select_siqs_factors(&fb, n, nfacs as usize);
        let target = facs.target;

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
    // Prepare A values
    // Only test 10 A values and 35 polynomials per A.
    let f = select_siqs_factors(&fb, &n, 9);
    let a_ints = select_a(&f, 10);
    for a_int in &a_ints {
        let a = prepare_a(&f, a_int, &fb);
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
            let pol = make_polynomial(&n, &a, idx);
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
                for r in pol
                    .prepare_prime(pidx, &fb, &fb.div(pidx).div31, 0, &a)
                    .offsets
                {
                    let Some(r) = r else { continue };
                    let r = r as u64;
                    let v = pa * Uint::from(r * r) + pb * Uint::from(2 * r);
                    let v = Int::from_bits(v) + pc;
                    assert_eq!(v.abs().to_bits() % (fb.p(pidx) as u64), 0);
                }
            }
        }
    }
}
