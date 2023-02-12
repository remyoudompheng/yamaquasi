// Copyright 2022,2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! A basic/naive implementation of ECM using Edwards curves.
//!
//! References:
//! <https://gitlab.inria.fr/zimmerma/ecm>
//! <https://eecm.cr.yp.to/index.html>
//!
//! It includes a selection of Edwards "good curves" from
//! <https://eecm.cr.yp.to/goodcurves.html>
//!
//! An Edwards curve ax²+y²=1+dx²y² can be converted to Weierstrass
//! form (y²=x³+ a2 x²+ a4) with a2 = (a+d)/2 and a4=(a-d)²/16
//!
//! After good curves (Q-torsion Z/12 or Z/2 x Z/8) it iterates over
//! a simple infinite family of curves with rational Z/2 x Z/4 torsion.
//!
//! It implements the "baby step giant step" optimization for stage 2
//! as described in section 5.2 of <https://eecm.cr.yp.to/eecm-20111008.pdf>
//! and the FFT continuation giving complexity O(sqrt(B2) log B2)
//!
//! Like the rest of Yamaquasi, it only supports numbers under 512 bits.
//!
//! Curves are enumerated in a deterministic order.
//!
//! # Performance of stage 1
//!
//! Using batch modular inversion, running ECM on a given curve computes
//! a bounded amount of inversions, and scalar multiplication using 1024-bit
//! integers costs 1 doubling and 0.16 point addition per exponent bit in stage 1,
//!
//! # Extraction of almost equal factors
//!
//! When prime factors are extremely close to each other, they may be caught
//! in the same exponent block, preventing them from being resolved individually.
//! The problem is mitigated by binary searching for the most precise common
//! divisor and using smaller blocks for the few first primes.
//!
//! If a curve is exceptionally unlucky, another one can modify curve orders
//! to avoid sharing factors.
//!
//! TODO: merge exponent base with Pollard P-1

use std::cmp::{max, min};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use bnum::types::U1024;
use num_integer::Integer;
use rayon::prelude::*;

use crate::arith_gcd;
use crate::arith_montgomery::{gcd_factors, MInt, ZmodN};
use crate::arith_poly::Poly;
use crate::fbase;
use crate::params::stage2_params;
use crate::{Preferences, Uint, UnexpectedFactor, Verbosity};

/// Run ECM with automatically selected small parameters.
///
/// The goal of this function is not to completely factor numbers, but
/// to detect cases where a number has a relatively small prime factor
/// (about size(n) / 5)
pub fn ecm_auto(
    n: Uint,
    prefs: &Preferences,
    tpool: Option<&rayon::ThreadPool>,
) -> Option<(Uint, Uint)> {
    // The CPU budget here is only a few seconds (at most 1% of SIQS time).
    // So we intentionally use small parameters hoping to be very lucky.
    // Best D values have D/phi(D) > 4.3
    //
    // Sample parameters can be found at https://eecm.cr.yp.to/performance.html
    match n.bits() {
        0..=160 => {
            // Very quick run.
            ecm(n, 4, 120, 10e3, prefs, tpool)
        }
        161..=190 => {
            // Will quite often find a 30-32 bit factor (budget 10-20ms)
            ecm(n, 16, 120, 40e3, prefs, tpool)
        }
        191..=220 => {
            // Will quite often find a 40-45 bit factor (budget <100ms)
            ecm(n, 16, 800, 80e3, prefs, tpool)
        }
        221..=250 => {
            // Will quite often find a factor of size 50-55 bits (budget 0.1-0.5s)
            ecm(n, 30, 3000, 300e3, prefs, tpool)
        }
        251..=280 => {
            // Will quite often find a factor of size 60-65 bits (budget 2-3s)
            ecm(n, 100, 8000, 1.2e6, prefs, tpool)
        }
        281..=310 => {
            // Will often find a factor of size 70-75 bits (budget 5-10s)
            ecm(n, 100, 30_000, 4.7e6, prefs, tpool)
        }
        311..=340 => {
            // Will often find a factor of size 75-80 bits (budget 20-30s)
            ecm(n, 100, 60_000, 20e6, prefs, tpool)
        }
        341..=370 => {
            // Try to find a factor of size 80-85 bits (budget 1min)
            ecm(n, 200, 100_000, 70e6, prefs, tpool)
        }
        // For very large numbers, we don't expect quadratic sieve to complete
        // in reasonable time, so all hope is on ECM.
        371..=450 => {
            // May find a 100-110 bit factor. Budget is more than 10 minutes
            ecm(n, 400, 500_000, 640e6, prefs, tpool)
        }
        451.. => {
            // Budget is virtually unlimited (several seconds per curve)
            ecm(n, 500, 5_000_000, 21e9, prefs, tpool)
        }
    }
}

/// Factor number using purely ECM. This may never end, or fail.
pub fn ecm_only(
    n: Uint,
    prefs: &Preferences,
    tpool: Option<&rayon::ThreadPool>,
) -> Option<(Uint, Uint)> {
    // B1 values should be such that step 1 takes about as much time as step 2.
    // D values are only such that phi(D) is a bit less than a power of 2.

    // Only try as many curves as required by a reference factor size.
    // There is no use running too many curves for given B1/B2 choices.

    // Target 32 bit primes
    ecm(n, 10, 200, 10e3, prefs, tpool)
        // Or 48-bit primes (20x longer)
        .or_else(|| ecm(n, 30, 2000, 130e3, prefs, tpool))
        // Or 64-bit primes (15x longer)
        .or_else(|| ecm(n, 60, 10_000, 1.8e6, prefs, tpool))
        // Or 80-bit primes (10-15x longer)
        .or_else(|| ecm(n, 100, 100_000, 28e6, prefs, tpool))
        // Or 96-bit primes (10x longer)
        .or_else(|| ecm(n, 200, 500_000, 156e6, prefs, tpool))
        // Or 128-bit primes (50x longer, several CPU hours)
        .or_else(|| ecm(n, 1000, 5_000_000, 21e9, prefs, tpool))
        // Or 160-bit primes (~30x longer?, several CPU days)
        .or_else(|| ecm(n, 3000, 50_000_000, 543e9, prefs, tpool))
}

// Run ECM for a given number of curves and bounds B1, B2.
pub fn ecm(
    n: Uint,
    curves: usize,
    b1: usize,
    b2: f64,
    prefs: &Preferences,
    tpool: Option<&rayon::ThreadPool>,
) -> Option<(Uint, Uint)> {
    let (b2, _, _) = stage2_params(b2);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Attempting ECM with {curves} curves B1={b1} B2={b2:e}",);
    }
    let start = std::time::Instant::now();
    let zn = ZmodN::new(n);
    let sb = SmoothBase::new(b1);
    // Keep track of how many curves were examined.
    let iters = AtomicUsize::new(0);
    // Try good curves first. They have large torsion (extra factor 3 or 4)
    // so their order is more probably smooth.
    let done = AtomicBool::new(false);
    let do_good_curve = |&(x1, x2, y1, y2)| {
        if done.load(Ordering::Relaxed) || prefs.abort() {
            return None;
        }
        let iter = iters.fetch_add(1, Ordering::SeqCst) + 1;
        if prefs.verbose(Verbosity::Verbose) {
            eprintln!("Trying good Edwards curve G=({x1}/{x2},{y1}/{y2})");
        }
        let c = match Curve::from_fractional_point(zn.clone(), x1, x2, y1, y2) {
            Ok(c) => c,
            Err(UnexpectedFactor(p)) => {
                if Uint::from(p) == n {
                    // Not an actual factor, we have truly divided by zero.
                    return None;
                }
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Unexpected factor {p}");
                }
                done.store(true, Ordering::Relaxed);
                return Some((Uint::from(p), n / Uint::from(p)));
            }
        };
        if let res @ Some((p, _)) = ecm_curve(&sb, &zn, &c, b2, prefs.verbosity) {
            if prefs.verbose(Verbosity::Info) {
                eprintln!("ECM success {iter}/{curves} for special Edwards curve G=({x1}/{x2},{y1}/{y2}) p={p} elapsed={:.3}s",
                start.elapsed().as_secs_f64());
            }
            done.store(true, Ordering::Relaxed);
            return res;
        }
        None
    };
    // Only if they were not tried before.
    if !prefs.eecm_done.load(Ordering::Relaxed) {
        if let Some(pool) = tpool {
            let cs = &GOOD_CURVES[..min(curves, GOOD_CURVES.len())];
            let results: Vec<Option<_>> =
                pool.install(|| cs.par_iter().map(|t| do_good_curve(t)).collect());
            for r in results {
                if r.is_some() {
                    return r;
                }
            }
        } else {
            for (idx, gen) in GOOD_CURVES.iter().enumerate() {
                if idx >= curves {
                    return None;
                }
                if let Some(res) = do_good_curve(gen) {
                    return Some(res);
                }
            }
        }
        // Special Edwards curves failed, do not try them again.
        prefs.eecm_done.store(true, Ordering::Relaxed);
    }

    // Choose curves with torsion Z/2 x Z/4. There is a very easy infinite supply
    // of such curves because (3k+5)² + (4k+5)² = 1 + (5k+7)².
    // They are slightly more smooth than general Edwards curves
    // (exponent of 2 is 4.33 on average instead of 3.66).
    //
    // Pseudo-randomize curves to avoid using the same ones when recursing through factors.
    let (seed1, seed2) = {
        let n0 = n.digits()[0];
        (n0 as u32 % 65536, n0 as u32 >> 16)
    };
    let curves_k: Vec<_> = (0..curves - iters.load(Ordering::SeqCst))
        .map(|k| k as u32 * seed1 + seed2)
        .collect();
    let do_curve = |k: u32| {
        if done.load(Ordering::Relaxed) || prefs.abort() {
            return None;
        }
        let k = k as u64;
        let (gx, gy) = (3 * k + 8, 4 * k + 9);
        let iter = iters.fetch_add(1, Ordering::SeqCst) + 1;
        if prefs.verbose(Verbosity::Verbose) {
            eprintln!(
                "Trying Edwards curve with (2,4)-torsion G=({gx},{gy}) d=({}/GxGy)²",
                5 * k + 7,
            );
        }
        let c = match Curve::from_point(zn.clone(), gx, gy) {
            Ok(c) => c,
            Err(UnexpectedFactor(p)) => {
                if Uint::from(p) == n {
                    // Not an actual factor, we have truly divided by zero.
                    return None;
                }
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Unexpected factor {p}");
                }
                done.store(true, Ordering::Relaxed);
                return Some((Uint::from(p), n / Uint::from(p)));
            }
        };
        if let res @ Some((p, _)) = ecm_curve(&sb, &zn, &c, b2, prefs.verbosity) {
            if prefs.verbose(Verbosity::Info) {
                eprintln!(
                    "ECM success {iter}/{curves} for Edwards curve G=({gx},{gy}) d=({}/GxGy)² p={p} elapsed={:.3}s",
                    5 * k + 7,
                    start.elapsed().as_secs_f64()
                    );
            }
            done.store(true, Ordering::Relaxed);
            return res;
        }
        None
    };
    if let Some(pool) = tpool {
        let results: Vec<Option<_>> =
            pool.install(|| curves_k.par_iter().map(|&k| do_curve(k)).collect());
        for r in results {
            if r.is_some() {
                return r;
            }
        }
    } else {
        for k in curves_k {
            if let Some(res) = do_curve(k) {
                return Some(res);
            }
        }
    }
    if prefs.verbose(Verbosity::Info) {
        eprintln!("ECM failure after {:.3}s", start.elapsed().as_secs_f64());
    }
    None
}

fn ecm_curve(
    sb: &SmoothBase,
    zn: &ZmodN,
    c: &Curve,
    b2: f64,
    verbosity: Verbosity,
) -> Option<(Uint, Uint)> {
    let n = &zn.n;
    let (_, d1, d2) = stage2_params(b2);
    // ECM stage 1
    let start1 = std::time::Instant::now();
    let mut g = c.gen().clone();
    const GCD_INTERVAL: usize = 1000;
    let mut gxs = Vec::with_capacity(GCD_INTERVAL);
    for block in sb.factors.chunks(GCD_INTERVAL) {
        // Once g.0 is divisible by p, it will be for the rest of the loop.
        gxs.clear();
        gxs.push(g.0);
        for &f in block {
            g = c.scalar64_chainmul(f, &g);
            gxs.push(g.0);
        }
        if let Some(d) = check_gcd_factor(n, &gxs) {
            return Some((d, n / d));
        }
    }
    gxs.clear();
    gxs.push(g.0);
    for f in sb.larges.iter() {
        g = c.scalar1024_chainmul(f, &g);
        gxs.push(g.0);
    }
    if let Some(d) = check_gcd_factor(n, &gxs) {
        return Some((d, n / d));
    }
    drop(gxs);
    assert!(
        c.is_valid(&g),
        "invalid point G=[{}:{}:{}] for d={} mod {}",
        c.zn.to_int(g.0),
        c.zn.to_int(g.1),
        c.zn.to_int(g.2),
        c.zn.to_int(c.d),
        c.zn.n
    );
    let elapsed1 = start1.elapsed();

    // ECM stage 2
    // The order of G (hopefully) no longer has small prime factors.
    // Look for a "large" prime order. Instead of computing [l]G
    // for all primes l, use the baby step giant step method:
    //
    // [l]G = 0 translates to [ad+b]G = 0 or [ad]G = [-b]G
    // whre d² <= B2, a <= d, b in [-d/2, d/2] is coprime to d.
    //
    // Since the opposite of (x,y) is (-x,y) this means that
    // [ad]G and [abs(b)]G have the same coordinate y.
    //
    // After stage 1 we know that [ad]G and [abs(b)]G are never zero.
    let start2 = std::time::Instant::now();

    // Prepare values of abs(b): there are phi(d1)/2 < d1/4 such values.
    let mut bs = Vec::with_capacity(d1 as usize / 4);
    for b in 1..d1 / 2 {
        if Integer::gcd(&b, &d1) == 1 {
            bs.push(b);
        }
    }
    let g2 = c.double(&g);
    let g4 = c.double(&g2);
    let mut gaps = vec![c.to_extended(&g2), c.to_extended(&g4)];
    // Baby/giant steps in a single vector.
    let mut steps = Vec::with_capacity(d1 as usize / 4 + d2 as usize);
    // Compute the baby steps
    let mut bg = c.to_extended(&g);
    let mut bexp = 1;
    assert_eq!(bs[0], 1);
    steps.push(g.clone());
    let mut n_bsteps = 1 as usize;
    for &b in &bs[1..] {
        let gap = b - bexp;
        while gaps.len() < gap as usize / 2 {
            let gap2 = c.addext(&gaps[0], &gaps[gaps.len() - 1]);
            gaps.push(gap2);
        }
        bg = c.addext(&bg, &gaps[gap as usize / 2 - 1]);
        steps.push(bg.to_proj());
        n_bsteps += 1;
        bexp = b;
    }
    // Compute the giant steps
    // WARNING: extended coordinate addition must not be used
    // to compute the first step.
    let dg = c.scalar64_chainmul(d1, &g);
    let dg2 = c.double(&dg);
    let dgext = c.to_extended(&dg);
    let mut gg = c.to_extended(&dg2);
    steps.push(dg);
    steps.push(dg2);
    for _ in 2..d2 {
        gg = c.addext(&gg, &dgext);
        steps.push(gg.to_proj());
    }
    // Normalize, 1 modular inversion using batch inversion.
    batch_normalize(zn, &mut steps);

    let stage2_roots_time = start2.elapsed().as_secs_f64();

    let bsteps = &steps[..n_bsteps];
    let gsteps = &steps[n_bsteps..];
    let mut result = None;
    if d1 < 4000 {
        // Compute O(d*phi(d)) products
        let mut buffer = zn.one();
        let mut prods = Vec::with_capacity(gsteps.len());
        prods.push(buffer);
        for pg in gsteps {
            // Compute the gcd after each row for finer granularity.
            for pb in bsteps {
                // y(G) - y(B)
                let delta_y = zn.sub(&pg.1, &pb.1);
                buffer = zn.mul(&buffer, &delta_y);
            }
            prods.push(buffer);
        }
        if let Some(d) = check_gcd_factor(n, &prods) {
            result = Some((d, n / d));
        }
    } else {
        // Usually D > 4 phi(D) (phi(2*3*5*7) < N/4)
        // And there are only phi(D)/2 baby steps
        //
        // If we split gsteps into blocks, each of them multieval'ed on bsteps.
        // the complexity will be O(gsteps/bsteps * bsteps log(bsteps)^alpha)
        //
        // However, the multipoint evaluation is ~10x the complexity of
        // a polynomial product.
        // It is more efficient to compute the (large) polynomial
        // PG=Product(X - pgs[i]) and reduce it modulo PB=Product(X - pbs[j])
        // and compute a single remainder tree.
        // (See "20 years of ECM" article)
        //
        // Efficiency is maximal if #bsteps is very close to a power of 2
        // and if PG/PB can be computed efficiently.
        let pbs: Vec<MInt> = bsteps.iter().map(|p| p.1).collect();
        let pgs: Vec<MInt> = gsteps.iter().map(|p| p.1).collect();
        let mut vals = Poly::roots_eval(zn, &pgs, &pbs);
        let mut prod = zn.one();
        // Replace array by cumulative products.
        for i in 0..vals.len() {
            let v = vals[i];
            vals[i] = prod;
            prod = zn.mul(&prod, &v);
        }
        vals.push(prod);
        if let Some(d) = check_gcd_factor(n, &vals) {
            result = Some((d, n / d));
        }
    }
    if verbosity >= Verbosity::Verbose {
        let stage1 = elapsed1.as_secs_f64();
        let stage2 = start2.elapsed().as_secs_f64();
        if stage2 < 0.01 {
            eprintln!("ECM stage1={stage1:.6}s stage2={stage2:.6}s (stage2 roots {stage2_roots_time:.6}s)");
        } else {
            eprintln!("ECM stage1={stage1:.3}s stage2={stage2:.3}s (stage2 roots {stage2_roots_time:.3}s)");
        }
    }
    result
}

/// Extract a single factor using GCD.
fn check_gcd_factor(n: &Uint, values: &[MInt]) -> Option<Uint> {
    // FIXME: we should return all factors. We just
    // return the largest one.
    let fs = gcd_factors(n, &values[..]).0;
    fs.iter().filter(|f| f != &n).max().copied()
}

// Normalize projective coordinates (z=1) for multiple points.
fn batch_normalize(zn: &ZmodN, pts: &mut [Point]) -> Option<()> {
    // Use Montgomery's batch inversion.
    // Compute cumulative products.
    let mut prod = zn.one();
    let mut prods = Vec::with_capacity(pts.len());
    for p in pts.iter() {
        prod = zn.mul(prod, p.2);
        prods.push(prod);
    }
    // Now prod is the product z0*...*z[n-1]
    let mut prodinv = zn.inv(prod)?;
    let one = zn.one();
    // Loop backwards and multiply by inverses.
    for i in 0..pts.len() {
        let j = pts.len() - 1 - i;
        // Invariant: prodinv is the product 1/z0 ... z[j]
        let zinv = if j > 0 {
            zn.mul(prodinv, prods[j - 1])
        } else {
            prodinv
        };
        let Point(x, y, z) = pts[j];
        pts[j] = Point(zn.mul(x, zinv), zn.mul(y, zinv), one);
        // Cancel z[j]
        prodinv = zn.mul(prodinv, z);
    }
    Some(())
}

/// An exponent base for ECM.
pub struct SmoothBase {
    /// Chunks of primes multiplied into u64 integers.
    factors: Box<[u64]>,
    /// Large chunks of primes (1024 bits)
    larges: Box<[Uint]>,
}

impl SmoothBase {
    pub fn new(b1: usize) -> Self {
        const LARGE_THRESHOLD: u64 = 4096;
        let primes = if b1 < 65_536 {
            fbase::primes(b1 as u32 / 2)
        } else {
            let mut s = fbase::PrimeSieve::new();
            let mut primes = vec![];
            loop {
                let b = s.next();
                primes.extend_from_slice(b);
                if b[b.len() - 1] > b1 as u32 {
                    break;
                }
            }
            primes
        };
        let mut factors = vec![];
        let mut factors_lg = vec![];
        let mut buffer = 1_u64;
        let mut buffer_lg = U1024::ONE;
        for p in primes {
            // Small primes are raised to some power (until B1).
            if p >= b1 as u32 {
                break;
            }
            let p = p as u64;
            let mut pow = p;
            while pow * p < b1 as u64 {
                pow *= p;
            }
            // Curve order has extra 2 and 3 factors.
            if p == 2 {
                pow *= 16;
            }
            if p == 3 {
                pow *= 3;
            }
            // Avoid too many primes in exponent buffer.
            // Small primes must make smaller blocks.
            if p < 256 && buffer > 1 << 32 {
                factors.push(buffer);
                buffer = 1;
            }
            if 1 << buffer.leading_zeros() <= pow {
                if p < LARGE_THRESHOLD {
                    factors.push(buffer);
                    buffer = 1;
                } else {
                    buffer_lg *= Uint::from_digit(buffer);
                    buffer = 1;
                }
            }
            if buffer_lg.bits() > 1024 - 64 {
                factors_lg.push(buffer_lg);
                buffer_lg = Uint::ONE;
            }
            buffer *= pow;
        }
        if buffer > 1 {
            if (b1 as u64) < LARGE_THRESHOLD {
                factors.push(buffer);
            } else {
                buffer_lg *= Uint::from_digit(buffer);
            }
        }
        if buffer_lg > Uint::ONE {
            factors_lg.push(buffer_lg)
        }
        SmoothBase {
            factors: factors.into_boxed_slice(),
            larges: factors_lg.into_boxed_slice(),
        }
    }
}

/// An elliptic curve in (twisted) Edwards form ax²+y² = 1 + dx²y²
/// where a = ±1
pub struct Curve {
    /// The base ring
    zn: ZmodN,
    /// Whether the curve has a=-1
    twisted: bool,
    d: MInt,
    // Coordinates of a "non-torsion" base point.
    g: Point,
}

/// A point in the projective plane using homogeneous coordinates.
#[derive(Clone, Debug)]
pub struct Point(MInt, MInt, MInt);

/// An Edwards curve point in extended coordinates (x,y,z,t)
/// located in the quadric xy=zt.
///
/// The equation of the Edwards curve in these coordinates is:
/// a x^2 + y^2 = z^2 + d t^2
///
/// Addition in this representation costs 8 multiplications
/// instead of 12, making it ideal for stage 2.
#[derive(Clone, Debug)]
pub struct ExtPoint(MInt, MInt, MInt, MInt);

impl ExtPoint {
    fn to_proj(&self) -> Point {
        Point(self.0, self.1, self.2)
    }
}

#[derive(Debug)]
#[doc(hidden)]
pub struct UnexpectedLargeFactor(Uint);

impl UnexpectedLargeFactor {
    fn new(zn: &ZmodN, x: &MInt) -> Self {
        let d = Integer::gcd(&zn.n, &Uint::from(*x));
        assert!(d != Uint::ONE);
        Self(d)
    }
}

fn zn_divide(zn: &ZmodN, a: &MInt, b: &MInt) -> Result<MInt, UnexpectedLargeFactor> {
    match zn.inv(*b) {
        Some(binv) => Ok(zn.mul(a, &binv)),
        None => Err(UnexpectedLargeFactor::new(zn, b)),
    }
}

// Special curves from https://eecm.cr.yp.to/goodcurves.html
// We only need the coordinates of the generator (d can be computed).
// We select only those with prime factors < 200
const GOOD_CURVES: &[(i64, i64, i64, i64)] = &[
    // 8 curves with Z/2 x Z/8 torsion
    (13, 7, 289, 49),       // d=25921/83521
    (319, 403, 551, 901),   // d=1681/707281
    (943, 979, 1271, 2329), // d=2307361/2825761
    (623, 103, 979, 589),   // d=23804641/62742241
    (125, 91, 841, 791),    // d=418079809/442050625
    (1025, 158, 697, 25),   // d=44182609/1766100625
    (1025, 1032, 41, 265),  // d=779135569/1766100625
    // 8 curves with Z/12Z torsion
    (5, 23, -1, 7),         // d=-24167/25
    (81, 5699, -901, 2501), // d=-895973/27
    (11, 589, -17, 107),    // d=-13391879/121
    (8, 17, -20, 19),       // d=1375/1024
    (5, 4, -25, 89),        // d=81289/15625
    (45, 19, -3, 1),        // d=4913/18225
    (35, 109, -49, 1),      // d=1140625/117649
    (49, 101, -21, 19),     // d=560947/352947
    (11, 53, -121, 4),      // d=41083561/1771561
];

impl Curve {
    fn fraction_modn(zn: &ZmodN, a: i64, b: i64) -> Result<MInt, UnexpectedFactor> {
        fn to_uint(n: Uint, x: i64) -> Uint {
            if x >= 0 {
                Uint::from(x as u64) % n
            } else {
                n - Uint::from((-x) as u64) % n
            }
        }
        let binv = match arith_gcd::inv_mod(&to_uint(zn.n, b), &zn.n) {
            Ok(inv) => inv,
            Err(d) => return Err(UnexpectedFactor(d.digits()[0])),
        };
        Ok(zn.mul(zn.from_int(to_uint(zn.n, a)), zn.from_int(binv)))
    }

    // Construct a curve from a Q-rational point (x1/x2, y1/y2).
    fn from_fractional_point(
        zn: ZmodN,
        x1: i64,
        x2: i64,
        y1: i64,
        y2: i64,
    ) -> Result<Curve, UnexpectedFactor> {
        assert!(max(x1.abs(), x2.abs()) < 65536 && max(y1.abs(), y2.abs()) < 65536);
        // This assumes that the denominators are invertible mod n.
        // Small primes must have been eliminated beforehand.
        let gx = Self::fraction_modn(&zn, x1, x2)?;
        let gy = Self::fraction_modn(&zn, y1, y2)?;
        // Compute d = (x1²y2²+x2²y1²-x2²y2²) / (x1²y1²)
        let dnum: i64 = x1 * x1 * y2 * y2 + x2 * x2 * y1 * y1 - x2 * x2 * y2 * y2;
        let dden: i64 = x1 * x1 * y1 * y1;
        let d = Self::fraction_modn(&zn, dnum, dden)?;
        let one = zn.one();
        let c = Curve {
            zn,
            twisted: false,
            d,
            g: Point(gx, gy, one),
        };
        assert!(
            c.is_valid(&c.g),
            "invalid point G=({x1}/{x2}={},{y1}/{y2}={}) for d={dnum}/{dden}={} mod {}",
            c.zn.to_int(gx),
            c.zn.to_int(gy),
            c.zn.to_int(d),
            c.zn.n
        );
        Ok(c)
    }

    // Construct the unique curve through a point with nonzero
    // coordinates.
    pub fn from_point(zn: ZmodN, x: u64, y: u64) -> Result<Curve, UnexpectedFactor> {
        assert!(x < 1 << 31 && y < 1 << 31);
        // Compute d = (x²+y²-1) / x²y²
        let gx = zn.from_int(Uint::from(x) % zn.n);
        let gy = zn.from_int(Uint::from(y) % zn.n);
        // gx*gx + gy*gy - 1
        let dn = Self::fraction_modn(&zn, (x * x + y * y - 1) as i64, 1)?;
        let dd = Self::fraction_modn(&zn, 1, (x * y) as i64)?;
        let d = zn.mul(zn.mul(dn, dd), dd);
        let g = Point(gx, gy, zn.one());
        Ok(Curve {
            zn,
            twisted: false,
            d,
            g,
        })
    }

    pub fn twisted_from_point(zn: ZmodN, g: Point) -> Result<Curve, UnexpectedLargeFactor> {
        // (-x²+y²)z² = z^4 + dx²y²
        // d = (y^2 - x^2 - z^2)z^2 / x^2 y^2
        let x2 = zn.mul(&g.0, &g.0);
        let y2 = zn.mul(&g.1, &g.1);
        let z2 = zn.mul(&g.2, &g.2);
        let dn = zn.mul(&z2, &zn.sub(&zn.sub(&y2, &x2), &z2));
        let dd = zn.mul(&x2, &y2);
        let d = zn_divide(&zn, &dn, &dd)?;
        Ok(Curve {
            zn,
            twisted: true,
            d,
            g,
        })
    }

    pub fn gen(&self) -> &Point {
        &self.g
    }

    // Addition formula following add-2007-bl (a=1) and add-2008-bbjlp (a=-1)
    // https://hyperelliptic.org/EFD/g1p/auto-edwards-projective.html#addition-add-2007-bl
    //
    // Both formulas are strongly unified and work even for P=Q.
    fn add(&self, p: &Point, q: &Point) -> Point {
        // 12 multiplications are required.
        let zn = &self.zn;
        // Handle z
        let a = zn.mul(&p.2, &q.2);
        let b = zn.mul(&a, &a);
        // Karatsuba-like product
        let c = zn.mul(&p.0, &q.0);
        let d = zn.mul(&p.1, &q.1);
        let cd = zn.mul(&zn.add(&p.0, &p.1), &zn.add(&q.0, &q.1));
        let c_plus_d = zn.add(&c, &d);
        let cross = zn.sub(&cd, &c_plus_d);

        let e = zn.mul(&self.d, &zn.mul(&c, &d));
        let f = zn.sub(&b, &e);
        let g = zn.add(&b, &e);

        // See formula add-2008-bbjlp for the a=-1 case.
        let x = zn.mul(&zn.mul(&a, &f), &cross);
        let y = zn.mul(
            zn.mul(&a, &g),
            if self.twisted {
                c_plus_d
            } else {
                zn.sub(&d, &c)
            },
        );
        let z = zn.mul(&f, &g);
        Point(x, y, z)
    }

    /// Addition formmula for extended coordinates on twisted Edwards curves.
    ///
    /// For a=-1, it requires 8 modular multiplications
    /// https://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-4
    /// For a=1, it requires 9 modular multiplications
    /// https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#addition-add-2008-hwcd-2
    ///
    /// Warning: these formulas are not "strongly unified" and will return
    /// zero coordinates if P=Q. We expect this case to only happen by
    /// accident (finding a factor during stage 2 steps) and not
    /// by design (doubling a point).
    fn addext(&self, p: &ExtPoint, q: &ExtPoint) -> ExtPoint {
        let zn = &self.zn;
        let (a, b, c, d) = if self.twisted {
            // Formula add-2008-hwcd-4 in Explicit Formula Database
            // (yP - xP)(yQ + xQ), (yP + xP)(yQ - xQ)
            let a = zn.mul(&zn.sub(&p.1, &p.0), &zn.add(q.1, q.0));
            let b = zn.mul(&zn.add(&p.1, &p.0), &zn.sub(q.1, q.0));
            // 2 zP tQ, 2 zQ tP
            let c = zn.mul(&p.2, &q.3);
            let c = zn.add(&c, &c);
            let d = zn.mul(&p.3, &q.2);
            let d = zn.add(&d, &d);
            (a, b, c, d)
        } else {
            // Formula add-2008-hwcd-2 in Explicit Formula Database
            (
                zn.mul(&p.0, &q.0),
                zn.mul(&p.1, &q.1),
                zn.mul(&p.2, &q.3),
                zn.mul(&p.3, &q.2),
            )
        };
        let e = zn.add(&d, &c);
        let f = if self.twisted {
            zn.sub(&b, &a)
        } else {
            // xP yQ - xQ yP = (xP-yP)*(xQ+yQ) +yPyQ -xPxQ
            zn.add(
                &zn.mul(&zn.sub(p.0, p.1), &zn.add(q.0, q.1)),
                &zn.sub(&b, &a),
            )
        };
        let g = zn.add(&b, &a);
        let h = zn.sub(&d, &c);
        // xy(P+Q) = zt(P+Q) = efgh
        ExtPoint(
            zn.mul(&e, &f),
            zn.mul(&g, &h),
            zn.mul(&f, &g),
            zn.mul(&e, &h),
        )
    }

    fn sub(&self, p: &Point, q: &Point) -> Point {
        let q = Point(self.zn.sub(&self.zn.zero(), &q.0), q.1, q.2);
        self.add(p, &q)
    }

    // Doubling formula following dbl-2007-bl
    // https://hyperelliptic.org/EFD/g1p/auto-edwards-projective.html#doubling-dbl-2007-bl
    fn double(&self, p: &Point) -> Point {
        // 7 multiplications are required.
        let zn = &self.zn;
        // Handle z
        let x_plus_y = zn.add(&p.0, &p.1);
        let b = zn.mul(&x_plus_y, &x_plus_y);
        let c = zn.mul(&p.0, &p.0);
        let d = zn.mul(&p.1, &p.1);
        if self.twisted {
            // Formula dbl-2008-bbjlp
            let c_plus_d = zn.add(&c, &d);
            let f = zn.sub(&d, &c);
            let h = zn.mul(&p.2, &p.2);
            let j = zn.sub(&zn.sub(&f, &h), &h);
            let x = zn.mul(&zn.sub(&b, &c_plus_d), &j);
            let y = zn.mul(&f, &zn.sub(&zn.zero(), &c_plus_d));
            let z = zn.mul(&f, &j);
            Point(x, y, z)
        } else {
            let e = zn.add(&c, &d);
            let h = zn.mul(&p.2, &p.2);
            let j = zn.sub(&zn.sub(&e, &h), &h);
            // Final result
            let x = zn.mul(&zn.sub(&b, &e), &j);
            let y = zn.mul(&e, &zn.sub(&c, &d));
            let z = zn.mul(&e, &j);
            Point(x, y, z)
        }
    }

    /// Naïve double-and-add scalar multiplication.
    /// It must not be used in ordinary code, and exists only for testing purposes.
    pub fn scalar64_mul_dbladd(&self, k: u64, p: &Point) -> Point {
        let zn = &self.zn;
        let mut res = Point(zn.zero(), zn.one(), zn.one());
        let mut sq: Point = p.clone();
        let mut k = k;
        while k > 0 {
            if k & 1 == 1 {
                res = self.add(&res, &sq);
            }
            sq = self.double(&sq);
            k >>= 1;
        }
        res
    }

    pub fn scalar64_chainmul(&self, k: u64, p: &Point) -> Point {
        if k == 0 {
            let zn = &self.zn;
            return Point(zn.zero(), zn.one(), zn.one());
        }
        // Compute an addition chain for k as in
        // https://eprint.iacr.org/2007/455.pdf
        // We find that m=7 is optimal for 64-bit blocks (~14 adds instead of 28 for ~56-bit blocks)
        // For 32-bit blocks, the optimal value is m=5 (7 adds instead of 12 for ~22-bit blocks)
        let p2 = self.double(p);
        let p3 = self.add(p, &p2);
        let p5 = self.add(&p3, &p2);
        let p7 = self.add(&p5, &p2);
        let gaps = [p, &p3, &p5, &p7];
        // Encode the chain as:
        // 0 (doubling)
        // ±k (add/sub kP)
        let mut c = [0_i8; 128];
        let l = Self::make_addition_chain(&mut c, k);
        // Get initial element (chain[l-1] = 1 or 3 or 5 or 7)
        let mut q = gaps[c[l - 1] as usize / 2].clone();
        for idx in 1..l {
            let op = c[l - 1 - idx];
            if op == 0 {
                q = self.double(&q);
            } else if op == 2 {
                q = self.add(&q, &p2);
            } else if op > 0 {
                debug_assert!(op & 1 == 1);
                q = self.add(&q, gaps[op as usize / 2]);
            } else if op < 0 {
                debug_assert!(op & 1 == 1);
                q = self.sub(&q, gaps[(-op) as usize / 2]);
            }
        }
        q
    }

    fn make_addition_chain(chain: &mut [i8; 128], k: u64) -> usize {
        // Build an addition chain as a reversed list of opcodes:
        // - first opcode retrieves xP for x in (1, 3, 5, 7)
        // - opcode 0 means double
        // - opcode y means add yP
        if k == 0 {
            chain[0] = 0;
            return 1;
        }
        const M: u64 = 7;
        let mut l = 0;
        let mut kk = k;
        loop {
            if kk % 2 == 0 {
                // make K/2, double
                chain[l] = 0;
                kk /= 2;
                l += 1;
            } else if kk <= M {
                chain[l] = kk as i8;
                return l + 1;
            } else if kk == M + 2 {
                // make M, add 2
                chain[l] = 2;
                kk = M;
                l += 1;
            } else if M + 4 <= kk && kk < 3 * M {
                // make 2x+1, double, add k-(4x+2) (<= M)
                let x = kk / 6;
                chain[l] = (kk - 4 * x - 2) as i8;
                chain[l + 1] = 0;
                l += 2;
                kk = 2 * x + 1;
            } else {
                // select in window
                let mut best = 1;
                let mut best_tz = (kk - 1).trailing_zeros();
                for x in [-7, -5, -3, -1, 1_i64, 3, 5, 7] {
                    let tz = (kk as i64 - x).trailing_zeros();
                    if tz > best_tz {
                        best = x;
                        best_tz = tz
                    }
                }
                chain[l] = best as i8;
                kk = ((kk as i64) - best) as u64;
                l += 1;
            }
        }
    }

    pub fn scalar1024_chainmul(&self, k: &U1024, p: &Point) -> Point {
        if k.is_zero() {
            let zn = &self.zn;
            return Point(zn.zero(), zn.one(), zn.one());
        }
        let p2 = self.double(p);
        // Store p, 3p ... 63p
        let mut gaps = Vec::with_capacity(32);
        gaps.push(p.clone());
        for i in 1..32 {
            let pi = self.add(&gaps[i - 1], &p2);
            gaps.push(pi);
        }
        // Construct chain
        let mut c = [0_i8; 384];
        let l = Self::make_addition_chain_long(&mut c, k);
        // Get initial element (chain[l-1] = 1 or 3 or 5 or 7)
        let mut q = gaps[c[l - 1] as usize / 2].clone();
        for idx in 1..l {
            let op = c[l - 1 - idx];
            if op % 2 == 0 {
                for _ in 0..op / 2 {
                    q = self.double(&q);
                }
            } else if op > 0 {
                debug_assert!(op & 1 == 1);
                q = self.add(&q, &gaps[op as usize / 2]);
            } else if op < 0 {
                debug_assert!(op & 1 == 1);
                q = self.sub(&q, &gaps[(-op) as usize / 2]);
            }
        }
        q
    }

    fn make_addition_chain_long(chain: &mut [i8; 384], n: &U1024) -> usize {
        // We intentionally assert that Uint is 1024 bits.
        // The add/sub will use odd integers from 1 to 63 (6 bits)
        // After each add/sub we expect to perform >= 6 doubles
        // The chain length cannot be more than 340
        // (there are at most 1024/6 blocks)
        //
        // This function does not implement the special edge cases
        // that save 1/2 adds in `make_addition_chain`
        //
        // Encoding:
        // 2k => double, repeated k times (k <= 63)
        // +k => add kP (k is odd)
        // -k => sub kP (k is odd)
        let nd: &[u64; 16] = n.digits();
        // Start from LSB and work with 2 words at a time.
        let mut exp = nd[0] as u128;
        let mut nextword = 1;
        let mut idx = 0;
        // How many bits were processed
        let mut bits = 0;
        // How many bits are stored in exp.
        let mut curbits = if n.bits() >= 64 {
            64
        } else {
            u128::BITS - u128::leading_zeros(exp)
        };
        let nbits = n.bits();
        let lastword = (nbits as usize - 1) / 64;
        while bits < nbits || exp > 0 {
            if curbits <= 32 && nextword <= lastword {
                exp += (nd[nextword] as u128) << curbits;
                nextword += 1;
                curbits += 64;
            }
            if exp & 1 == 0 {
                // We cannot shift right by 128 and we cannot store
                // large numbers in the chain, so limit tz to 60.
                let tz = min(60, min(exp.trailing_zeros(), curbits));
                exp >>= tz;
                bits += tz;
                curbits -= tz;
                chain[idx] = 2 * tz as i8;
                idx += 1;
            } else {
                let low = exp & 127;
                debug_assert!(low % 2 == 1);
                if low < 64 {
                    chain[idx] = low as i8;
                    idx += 1;
                    exp -= low;
                    if exp == 0 && nbits - bits <= 6 {
                        // Finished!
                        break;
                    }
                } else {
                    chain[idx] = -((128 - low) as i8);
                    idx += 1;
                    // Note: exp may become larger than 1 << curbits.
                    exp += 128 - low;
                }
            }
        }
        idx
    }

    #[cfg(test)]
    fn scalar_mul(&self, k: Uint, p: &Point) -> Point {
        let zn = &self.zn;
        let mut res = Point(zn.zero(), zn.one(), zn.one());
        let mut sq: Point = p.clone();
        let mut k = k;
        while k > Uint::ZERO {
            if k.bit(0) {
                res = self.add(&res, &sq);
            }
            sq = self.double(&sq);
            k >>= 1;
        }
        res
    }

    /// A projective plane point [x:y:z] defines an extended point by
    /// the Segre embedding of ([x:z], [y:z])
    fn to_extended(&self, p: &Point) -> ExtPoint {
        let zn = &self.zn;
        ExtPoint(
            zn.mul(&p.0, &p.2),
            zn.mul(&p.1, &p.2),
            zn.mul(&p.2, &p.2),
            zn.mul(&p.0, &p.1),
        )
    }

    fn is_valid(&self, p: &Point) -> bool {
        // (±x²+y²)z² = z^4 + dx²y²
        let zn = &self.zn;
        let x2 = zn.mul(p.0, p.0);
        let y2 = zn.mul(p.1, p.1);
        let z2 = zn.mul(p.2, p.2);
        let lhs = if self.twisted {
            zn.mul(zn.sub(y2, x2), z2)
        } else {
            zn.mul(zn.add(x2, y2), z2)
        };
        let rhs = zn.add(zn.mul(z2, z2), zn.mul(self.d, zn.mul(x2, y2)));
        lhs == rhs
    }

    #[cfg(test)]
    fn is_validext(&self, p: &ExtPoint) -> bool {
        // ax²+ y² = z² + dt²
        let zn = &self.zn;
        let x2 = zn.mul(p.0, p.0);
        let y2 = zn.mul(p.1, p.1);
        let z2 = zn.mul(p.2, p.2);
        let t2 = zn.mul(p.3, p.3);
        let lhs = if self.twisted {
            zn.sub(&y2, &x2)
        } else {
            zn.add(&y2, &x2)
        };
        let rhs = zn.add(&z2, &zn.mul(&self.d, &t2));
        lhs == rhs
    }

    fn is_2_torsion(&self, p: &Point) -> Option<Uint> {
        let d = self.zn.gcd(&p.0);
        if d == Uint::ONE {
            None
        } else {
            Some(d)
        }
    }

    #[cfg(test)]
    fn equal(&self, p: &Point, q: &Point) -> bool {
        let zn = &self.zn;
        zn.mul(p.0, q.1) == zn.mul(p.1, q.0)
            && zn.mul(p.1, q.2) == zn.mul(p.2, q.1)
            && zn.mul(p.2, q.0) == zn.mul(p.0, q.2)
    }
}

#[cfg(test)]
use std::str::FromStr;

#[cfg(test)]
const MODULUS256: &'static str =
    "107910248100432407082438802565921895527548119627537727229429245116458288637047";

#[test]
fn test_curve() {
    use std::str::FromStr;
    let n = Uint::from_str("2953951639731214343967989360202131868064542471002037986749").unwrap();
    let c = Curve::from_point(ZmodN::new(n), 2, 3).unwrap();
    eprintln!("d={}", c.zn.to_int(c.d));
    let g = c.gen();
    assert!(c.is_valid(&g));
    assert!(c.is_valid(&c.add(&g, &g)));
    assert!(c.is_valid(&c.double(&g)));

    // Edwards curve(d) is equivalent to Y² = X³ + (1 − 2c)X² + c^2 X where c = (1-d)/4
    // This curve has orders 59528557881166853791031641960 and 49622429047046610830438562488
    // modulo the factors of n (49622429047046386146923837183 * 59528557881167220232630894403)
    let ord1 = Uint::from_str("49622429047046610830438562488").unwrap();
    let ord2 = Uint::from_str("59528557881166853791031641960").unwrap();
    let g1 = c.scalar_mul(ord1, &g);
    assert!(c.is_2_torsion(&g1).is_some());
    eprintln!("factor {}", c.is_2_torsion(&g1).unwrap());
    assert_eq!(
        c.is_2_torsion(&g1),
        Uint::from_str("49622429047046386146923837183").ok()
    );
    let g2 = c.scalar_mul(ord2, &g);
    assert!(c.is_2_torsion(&g2).is_some());
    eprintln!("factor {}", c.is_2_torsion(&g2).unwrap());
    assert_eq!(
        c.is_2_torsion(&g2),
        Uint::from_str("59528557881167220232630894403").ok()
    );

    // Test extended coordinates.
    let g1 = c.scalar64_chainmul(123_456_789, &g);
    let g2 = c.scalar64_chainmul(987_654_321, &g);
    let g1ext = c.to_extended(&g1);
    let g2ext = c.to_extended(&g2);
    assert!(c.is_validext(&g1ext));
    assert!(c.is_validext(&g2ext));
    let g12 = c.add(&g1, &g2);
    let g12ext = c.addext(&g1ext, &g2ext);
    assert!(c.is_validext(&g12ext));
    assert!(c.equal(&g12, &g12ext.to_proj()));
}

#[test]
fn test_twisted_curve() {
    let p = Uint::from(602768606663711_u64);
    let q = Uint::from(957629686686973_u64);
    let n = p * q;
    let zn = ZmodN::new(n);
    // The σ=11 twisted Edwards curve.
    // Its order is:
    // mod p: 602768647071432 = 2^3 * 3^2 * 17 * 251 * 4679 * 419317
    // mod q: 957629727109848 = 2^3 * 3 * 13 * 359 * 8549654731
    let gx = Curve::fraction_modn(&zn, -11, 60).unwrap();
    let gy = Curve::fraction_modn(&zn, 11529, 12860).unwrap();
    let g = Point(gx, gy, zn.one());
    let c = Curve::twisted_from_point(zn.clone(), g).unwrap();
    let g = c.gen();
    assert!(c.is_valid(&g));

    let g1 = c.scalar64_chainmul(602768647071432, &g);
    assert!(c.is_valid(&g1));
    let g2 = c.scalar64_chainmul(957629727109848, &g1);
    assert!(c.is_valid(&g2));

    assert!(c.is_2_torsion(&g1) == Some(p));
    assert!(&g2.0 == &zn.zero());

    // Test extended coordinates.
    let g1 = c.scalar64_chainmul(123_456_789, &g);
    let g2 = c.scalar64_chainmul(987_654_321, &g);
    let g1ext = c.to_extended(&g1);
    let g2ext = c.to_extended(&g2);
    assert!(c.is_validext(&g1ext));
    assert!(c.is_validext(&g2ext));
    let g12 = c.add(&g1, &g2);
    let g12ext = c.addext(&g1ext, &g2ext);
    assert!(c.is_validext(&g12ext));
    assert!(c.equal(&g12, &g12ext.to_proj()));
}

#[test]
fn test_ecm_curve() {
    let p = Uint::from(602768606663711_u64);
    let q = Uint::from(957629686686973_u64);
    let n = p * q;
    let zn = ZmodN::new(n);
    // This curve has smooth order for prime 602768606663711
    // order: 2^2 * 7 * 19 * 29 * 347 * 503 * 223843
    let c = Curve::from_point(zn.clone(), 2, 10).unwrap();
    let sb = SmoothBase::new(1000);
    let res = ecm_curve(&sb, &zn, &c, 500e3, Verbosity::Silent);
    eprintln!("{:?}", res);
    assert_eq!(res, Some((p, q)));
}

#[test]
fn test_ecm_curve2() {
    use std::str::FromStr;
    let p = Uint::from_str("1174273970803390465747303").unwrap();
    let q = Uint::from_str("607700066377545220515437").unwrap();
    let n = p * q;
    let zn = ZmodN::new(n);
    // This curve has smooth order for prime 1174273970803390465747303
    // Order has largest prime factors 11329 and 802979
    let c = Curve::from_point(zn.clone(), 2, 132).unwrap();
    let sb = SmoothBase::new(15000);
    let res = ecm_curve(&sb, &zn, &c, 1e6, Verbosity::Silent);
    eprintln!("{:?}", res);
    assert_eq!(res, Some((p, q)));
}

#[test]
fn test_addition_chain() {
    fn eval_chain(c: &[i8]) -> u64 {
        let mut k = c[c.len() - 1] as u64;
        for idx in 1..c.len() {
            let op = c[c.len() - 1 - idx];
            if op == 0 {
                k = 2 * k;
            } else {
                k = ((k as i64) + (op as i64)) as u64;
            }
        }
        k
    }

    for k in 1..1000_u64 {
        let mut c = [0i8; 128];
        let l = Curve::make_addition_chain(&mut c, k);
        assert_eq!(k, eval_chain(&c[..l]), "chain={:?}", &c[..l]);
    }
    {
        let k = 602768647071432_u64;
        let mut c = [0i8; 128];
        let l = Curve::make_addition_chain(&mut c, k);
        assert_eq!(k, eval_chain(&c[..l]), "chain={:?}", &c[..l]);
    }
    let mut adds = 0;
    for i in 1..=1000_u64 {
        let mut c = [0i8; 128];
        let k = i.wrapping_mul(1_234_567_123_456_789);
        let l = Curve::make_addition_chain(&mut c, k);
        assert_eq!(k, eval_chain(&c[..l]), "chain={:?}", &c[..l]);
        adds += 3;
        for &op in &c[..l - 1] {
            if op != 0 {
                adds += 1;
            }
        }
    }
    eprintln!(
        "average additions {:.2} for 64-bit integers",
        adds as f64 / 1000.0
    );

    let n = Uint::from_str(MODULUS256).unwrap();
    let zn = ZmodN::new(n);
    let c = Curve::from_point(zn, 2, 132).unwrap();
    let k: u64 = 1511 * 1523 * 1531 * 1543 * 1549 * 1553;
    let p1 = c.scalar64_mul_dbladd(k, &c.gen());
    let p2 = c.scalar64_chainmul(k, &c.gen());
    assert!(c.equal(&p1, &p2));
    for k in 0..2000 {
        let p1 = c.scalar64_mul_dbladd(k, &c.gen());
        let p2 = c.scalar64_chainmul(k, &c.gen());
        assert!(c.equal(&p1, &p2), "failure for k={k}");
    }
}

#[test]
fn test_addition_chain_long() {
    use bnum::cast::CastFrom;
    use bnum::types::U2048;
    use rand::Rng;

    fn eval_chain(c: &[i8]) -> U1024 {
        let mut k = U2048::from_digit(c[c.len() - 1] as u64);
        for idx in 1..c.len() {
            let op = c[c.len() - 1 - idx];
            if op % 2 == 0 {
                k <<= (op / 2) as u32;
            } else if op > 0 {
                k += U2048::from_digit(op as u64);
            } else {
                k -= U2048::from_digit((-op) as u64);
            }
        }
        U1024::cast_from(k)
    }

    // Some sparse numbers
    for i in 900..1020 {
        let j = i - 400;
        let k = U1024::power_of_two(i) | U1024::power_of_two(j);
        let mut c = [0i8; 384];
        let l = Curve::make_addition_chain_long(&mut c, &k);
        assert_eq!(k, eval_chain(&c[..l]), "chain={:?}", &c[..l]);
        // Test negative steps
        let j1 = i - 300;
        let j2 = i / 2;
        let k = U1024::power_of_two(i) + U1024::power_of_two(j1) - U1024::power_of_two(j2);
        let l = Curve::make_addition_chain_long(&mut c, &k);
        assert_eq!(k, eval_chain(&c[..l]), "chain={:?}", &c[..l]);
    }

    // Random numbers: assume that 10000 is enough to express
    // all 8-bit patterns.
    let mut adds = 0;
    let mut maxsize = 0;
    for _ in 0..10000 {
        let mut rng = rand::thread_rng();
        let mut digits = [0; 16];
        let mut c = [0i8; 384];
        rng.fill(&mut digits);
        let k = U1024::from_digits(digits);
        let l = Curve::make_addition_chain_long(&mut c, &k);
        assert_eq!(k, eval_chain(&c[..l]), "chain={:?}", &c[..l]);
        adds += c[..l - 1].iter().filter(|&x| x & 1 == 1).count();
        maxsize = max(maxsize, l);
    }
    eprintln!(
        "average additions {:.2} for 1024-bit integers",
        adds as f64 / 10000.0
    );
    eprintln!("max chain size {maxsize}");

    let n = Uint::from_str(MODULUS256).unwrap();
    let zn = ZmodN::new(n);
    let c = Curve::from_point(zn, 2, 132).unwrap();
    let k: u64 = 1511 * 1523 * 1531 * 1543 * 1549 * 1553;
    let p1 = c.scalar64_mul_dbladd(k, &c.gen());
    let p2 = c.scalar1024_chainmul(&(k.into()), &c.gen());
    assert!(c.equal(&p1, &p2), "failure for k={k}");
    for k in 0..100 {
        let k = k * 12345678_123456789;
        let k2 = Uint::from(k) * Uint::from(k);
        let p1 = c.scalar64_mul_dbladd(k, &c.scalar64_mul_dbladd(k, &c.gen()));
        let p2 = c.scalar1024_chainmul(&k2, &c.gen());
        assert!(c.equal(&p1, &p2), "failure for k={k}^2");
    }
}
