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
//! After good curves (Q-torsion Z/12 or Z/2 x Z/8) it iterates over
//! a simple infinite family of curves with rational Z/2 x Z/4 torsion.
//!
//! It implements the "baby step giant step" optimization for stage 2
//! as described in section 5.2 of <https://eecm.cr.yp.to/eecm-20111008.pdf>
//! This is about 3x faster than the prime-by-prime approach.
//!
//! Due to slow big integer arithmetic, we use projective coordinates without
//! normalization, causing extra multiplications during stage 2.
//!
//! It does not try to be particularly efficient (it is at least 5x slower than GMP-ECM).
//! Like the rest of Yamaquasi, it only supports numbers under 512 bits.
//!
//! Curves are enumerated in a deterministic order.
//!
//! TODO: merge exponent base with Pollard P-1

use std::cmp::{max, min};
use std::sync::atomic::{AtomicBool, Ordering};

use num_integer::Integer;
use rayon::prelude::*;

use crate::arith;
use crate::arith_montgomery::{MInt, ZmodN};
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
        0..=190 => {
            // Will quite often find a 30-32 bit factor (budget 10-20ms)
            ecm(n, 16, 120, 40e3, prefs, tpool)
        }
        191..=220 => {
            // Will quite often find a 36 bit factor (budget <100ms)
            ecm(n, 16, 120, 80e3, prefs, tpool)
        }
        221..=250 => {
            // Will quite often find a factor of size 42-46 bits (budget 0.1-0.5s)
            ecm(n, 30, 500, 300e3, prefs, tpool)
        }
        251..=280 => {
            // Will quite often find a factor of size 52-56 bits (budget 2-3s)
            ecm(n, 80, 2_000, 1e6, prefs, tpool)
        }
        281..=310 => {
            // Will often find a factor of size 58-62 bits (budget 5-10s)
            ecm(n, 40, 8_000, 4e6, prefs, tpool)
        }
        311..=340 => {
            // Will often find a factor of size 64-70 bits (budget 20-30s)
            ecm(n, 40, 25_000, 20e6, prefs, tpool)
        }
        341..=370 => {
            // Try to find a factor of size 68-76 bits (budget 1min)
            ecm(n, 100, 75_000, 70e6, prefs, tpool)
        }
        // For very large numbers, we don't expect quadratic sieve to complete
        // in reasonable time, so all hope is on ECM.
        371..=450 => {
            // Budget is more than 10 minutes
            ecm(n, 200, 200_000, 400e6, prefs, tpool)
        }
        451.. => {
            // Budget is virtually unlimited (hours)
            ecm(n, 500, 500_000, 1e9, prefs, tpool)
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
    //
    // Since we are using Karatsuba, we can make B1 grow as O(D^1.58)
    match n.bits() {
        // The following parameters work well even for balanced semiprimes.
        0..=64 => ecm(n, 100, 128, 40e3, prefs, tpool),
        65..=80 => ecm(n, 100, 300, 40e3, prefs, tpool),
        81..=96 => ecm(n, 300, 1000, 200e3, prefs, tpool),
        97..=119 => ecm(n, 1000, 3_000, 1e6, prefs, tpool),
        // May require 100-300 curves for 72-bit factors
        120..=144 => ecm(n, 1000, 10_000, 4e6, prefs, tpool),
        145..=168 => {
            // Can find a 80 bit factor after a few dozen curves.
            ecm(n, 1000, 30_000, 20e6, prefs, tpool)
        }
        169..=192 => {
            // Can find a 90 bit factor after a few hundreds curves.
            ecm(n, 2000, 100_000, 70e6, prefs, tpool)
        }
        193..=224 => {
            // Should be able to find 100 bit factors after
            // a few hundred curves
            ecm(n, 5000, 300_000, 400e6, prefs, tpool)
        }
        225..=256 => {
            // May find 100-120 bit factors after ~1000 curves
            // Similar to GMP-ECM recommended for 35 digit factors.
            ecm(n, 20, 100_000, 80e3, prefs, tpool)
                .or_else(|| ecm(n, 15000, 1_000_000, 1e9, prefs, tpool))
        }
        257.. => {
            // May find 120-140 bit factors after a few thousand curves.
            // B2 is about 5.8 billion.
            // Similar to GMP-ECM recommended for 40 digit factors.
            ecm(n, 50, 300_000, 400e3, prefs, tpool)
                .or_else(|| ecm(n, 40000, 3_000_000, 4e9, prefs, tpool))
        }
    }
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
    // Try good curves first. They have large torsion (extra factor 3 or 4)
    // so their order is more probably smooth.
    let done = AtomicBool::new(false);
    let do_good_curve = |(idx, &(x1, x2, y1, y2))| {
        if done.load(Ordering::Relaxed) {
            return None;
        }
        if prefs.verbose(Verbosity::Verbose) {
            eprintln!("Trying good Edwards curve G=({x1}/{x2},{y1}/{y2})");
        }
        let c = match Curve::from_fractional_point(zn.clone(), x1, x2, y1, y2) {
            Ok(c) => c,
            Err(UnexpectedFactor(p)) => {
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Unexpected factor {p}");
                }
                done.store(true, Ordering::Relaxed);
                return Some((Uint::from(p), n / Uint::from(p)));
            }
        };
        if let res @ Some((p, _)) = ecm_curve(&sb, &zn, &c, b2, prefs.verbosity) {
            if prefs.verbose(Verbosity::Info) {
                eprintln!("ECM success {}/{curves} for special Edwards curve G=({x1}/{x2},{y1}/{y2}) p={p} elapsed={:.3}s",
                idx + 1,
                start.elapsed().as_secs_f64());
            }
            done.store(true, Ordering::Relaxed);
            return res;
        }
        None
    };
    if let Some(pool) = tpool {
        let cs: Vec<_> = GOOD_CURVES[..min(curves, GOOD_CURVES.len())]
            .iter()
            .enumerate()
            .collect();
        let results: Vec<Option<_>> =
            pool.install(|| cs.par_iter().map(|&t| do_good_curve(t)).collect());
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
            if let Some(res) = do_good_curve((idx, gen)) {
                return Some(res);
            }
        }
    }
    // Choose curves with torsion Z/2 x Z/4. There is a very easy infinite supply
    // of such curves because (3k+5)² + (4k+5)² = 1 + (5k+7)².
    // They are slightly more smooth than general Edwards curves
    // (exponent of 2 is 4.33 on average instead of 3.66).
    let curves_k: Vec<_> = (GOOD_CURVES.len() as u32..curves as u32).collect();
    let do_curve = |k: u32| {
        if done.load(Ordering::Relaxed) {
            return None;
        }
        let k = k as u64;
        let (gx, gy) = (3 * k + 8, 4 * k + 9);
        if prefs.verbose(Verbosity::Debug) {
            eprintln!(
                "Trying Edwards curve with (2,4)-torsion d=({}/{})² G=({},{})",
                5 * k + 7,
                gx * gy,
                gx,
                gy,
            );
        }
        let c = match Curve::from_point(zn.clone(), gx, gy) {
            Ok(c) => c,
            Err(UnexpectedFactor(p)) => {
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
                    "ECM success {}/{curves} for Edwards curve d=({}/{})² G=({},{}) p={p} elapsed={:.3}s",
                    k as usize + GOOD_CURVES.len() + 1,
                    5 * k + 7,
                    gx*gy, gx, gy,
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
    let mut g = c.gen();
    for block in sb.factors.chunks(4) {
        for &f in block {
            g = c.scalar64_mul(f, &g);
        }
        if let Some(d) = c.is_2_torsion(&g) {
            if d > Uint::ONE && d < *n {
                return Some((d, n / d));
            }
        }
    }
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
    let logtime = || {
        if verbosity >= Verbosity::Debug {
            let elapsed2 = start2.elapsed();
            if elapsed2.as_secs_f64() < 0.01 {
                eprintln!(
                    "ECM stage1={:.6}s stage2={:.6}s",
                    elapsed1.as_secs_f64(),
                    elapsed2.as_secs_f64()
                );
            } else {
                eprintln!(
                    "ECM stage1={:.3}s stage2={:.3}s",
                    elapsed1.as_secs_f64(),
                    elapsed2.as_secs_f64()
                );
            }
        }
    };

    // Prepare values of abs(b): there are phi(d1)/2 < d1/4 such values.
    let mut bs = Vec::with_capacity(d1 as usize / 4);
    for b in 1..d1 / 2 {
        if Integer::gcd(&b, &d1) == 1 {
            bs.push(b);
        }
    }
    let g2 = c.double(&g);
    let g4 = c.double(&g2);
    let mut gaps = vec![g2.clone(), g4];
    // Baby/giant steps in a single vector.
    let mut steps = Vec::with_capacity(d1 as usize / 4 + d2 as usize);
    // Compute the baby steps
    let mut bg = g.clone();
    let mut bexp = 1;
    assert_eq!(bs[0], 1);
    steps.push(bg.clone());
    let mut n_bsteps = 1 as usize;
    for &b in &bs[1..] {
        let gap = b - bexp;
        while gaps.len() < gap as usize / 2 {
            let gap2 = c.add(&g2, &gaps[gaps.len() - 1]);
            gaps.push(gap2);
        }
        bg = c.add(&bg, &gaps[gap as usize / 2 - 1]);
        steps.push(bg.clone());
        n_bsteps += 1;
        bexp = b;
    }
    // Compute the giant steps
    let dg = c.scalar64_mul(d1, &g);
    let mut gg = dg.clone();
    for _ in 0..d2 {
        steps.push(gg.clone());
        gg = c.add(&gg, &dg);
    }
    // Normalize, 1 modular inversion using batch inversion.
    batch_normalize(&zn, &mut steps);
    let bsteps = &steps[..n_bsteps];
    let gsteps = &steps[n_bsteps..];
    if d1 < 4000 {
        // Compute O(d*phi(d)) products
        let mut buffer = zn.one();
        for (idx, pg) in gsteps.iter().enumerate() {
            // Compute the gcd after each row for finer granularity.
            for pb in bsteps {
                // y(G) - y(B)
                let delta_y = zn.sub(&pg.1, &pb.1);
                buffer = zn.mul(&buffer, &delta_y);
            }
            if idx % 8 == 0 || idx == gsteps.len() - 1 {
                let d = Integer::gcd(n, &Uint::from(buffer));
                if d > Uint::ONE && d < *n {
                    logtime();
                    return Some((d, n / d));
                }
            }
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
        let vals = Poly::roots_eval(zn, &pgs, &pbs);
        let mut buffer = zn.one();
        for (idx, &v) in vals.iter().enumerate() {
            // Compute the gcd every few rows for finer granularity.
            buffer = zn.mul(buffer, v);
            if idx % 8 == 0 || idx == vals.len() - 1 {
                let d = Integer::gcd(n, &Uint::from(buffer));
                if d > Uint::ONE && d < *n {
                    logtime();
                    return Some((d, n / d));
                }
            }
        }
    }
    logtime();
    None
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
}

impl SmoothBase {
    pub fn new(b1: usize) -> Self {
        let primes = fbase::primes(b1 as u32 / 2);
        let mut factors = vec![];
        let mut buffer = 1_u64;
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
            if p == 2 {
                pow *= 16;
            }
            if p == 3 {
                pow *= 3;
            }
            if 1 << buffer.leading_zeros() <= pow {
                factors.push(buffer);
                buffer = 1;
            }
            buffer *= pow;
        }
        if buffer > 1 {
            factors.push(buffer)
        }
        SmoothBase {
            factors: factors.into_boxed_slice(),
        }
    }
}

// Edwards curves
// x²+y² = 1 + dx²y²
pub struct Curve {
    // The base ring
    zn: ZmodN,
    // Parameter
    d: MInt,
    // Coordinates of a "non-torsion" point.
    gx: MInt,
    gy: MInt,
}

// A point in projective coordinates.
#[derive(Clone, Debug)]
pub struct Point(MInt, MInt, MInt);

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
        let Some(binv) = arith::inv_mod(to_uint(zn.n, b), zn.n)
            else {
                let babs = b.abs() as u64;
                return Err(UnexpectedFactor(Integer::gcd(&(zn.n % babs), &babs)));
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
        let c = Curve { zn, d, gx, gy };
        assert!(
            c.is_valid(&Point(gx, gy, one)),
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
        let gx = zn.from_int(Uint::from(x));
        let gy = zn.from_int(Uint::from(y));
        // gx*gx + gy*gy - 1
        let dn = zn.from_int(Uint::from(x * x + y * y - 1));
        let dd = Self::fraction_modn(&zn, 1, (x * y) as i64)?;
        let d = zn.mul(zn.mul(dn, dd), dd);
        Ok(Curve { zn, d, gx, gy })
    }

    pub fn gen(&self) -> Point {
        Point(self.gx, self.gy, self.zn.from_int(Uint::ONE))
    }

    // Addition formula following add-2007-bl
    // https://hyperelliptic.org/EFD/g1p/auto-edwards-projective.html#addition-add-2007-bl
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
        let cross = zn.sub(&cd, &zn.add(&c, &d));

        let e = zn.mul(&self.d, &zn.mul(&c, &d));
        let f = zn.sub(&b, &e);
        let g = zn.add(&b, &e);

        let x = zn.mul(&zn.mul(&a, &f), &cross);
        let y = zn.mul(zn.mul(&a, &g), zn.sub(&d, &c));
        let z = zn.mul(&f, &g);
        Point(x, y, z)
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
        let e = zn.add(&c, &d);
        let h = zn.mul(&p.2, &p.2);
        let j = zn.sub(&zn.sub(&e, &h), &h);
        // Final result
        let x = zn.mul(&zn.sub(&b, &e), &j);
        let y = zn.mul(&e, &zn.sub(&c, &d));
        let z = zn.mul(&e, &j);
        Point(x, y, z)
    }

    pub fn scalar64_mul(&self, k: u64, p: &Point) -> Point {
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
        // Compute an addition chain for k as in
        // https://eprint.iacr.org/2007/455.pdf
        // We find that m=7 is optimal for 64-bit blocks (~14 adds instead of 28 for ~56-bit blocks)
        // For 32-bit blocks, the optimal value is m=5 (7 adds instead of 12 for ~22-bit blocks)
        let p2 = self.double(&p);
        let p3 = self.add(&p, &p2);
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
            } else if op > 0 {
                q = self.add(&q, &gaps[op as usize / 2]);
            } else if op < 0 {
                q = self.sub(&q, &gaps[(-op) as usize / 2]);
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

    fn is_valid(&self, p: &Point) -> bool {
        // (x²+y²)z² = z^4 + dx²y²
        let zn = &self.zn;
        let x2 = zn.mul(p.0, p.0);
        let y2 = zn.mul(p.1, p.1);
        let z2 = zn.mul(p.2, p.2);
        let lhs = zn.mul(zn.add(x2, y2), z2);
        let rhs = zn.add(zn.mul(z2, z2), zn.mul(self.d, zn.mul(x2, y2)));
        lhs == rhs
    }

    fn is_2_torsion(&self, p: &Point) -> Option<Uint> {
        let d = Integer::gcd(&Uint::from(p.0), &self.zn.n);
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
    let p1 = c.scalar64_mul(k, &c.gen());
    let p2 = c.scalar64_chainmul(k, &c.gen());
    assert!(c.equal(&p1, &p2));
}

#[cfg(test)]
const MODULUS256: &'static str =
    "107910248100432407082438802565921895527548119627537727229429245116458288637047";
