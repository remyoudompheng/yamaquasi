// Copyright 2022,2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! A basic/naive implementation of ECM using Edwards curves.
//!
//! References:
//! https://gitlab.inria.fr/zimmerma/ecm
//! https://eecm.cr.yp.to/index.html
//!
//! It includes a selection of Edwards "good curves" from
//! https://eecm.cr.yp.to/goodcurves.html
//!
//! After good curves (Q-torsion Z/12 or Z/2 x Z/8) it iterates over
//! a simple infinite family of curves with rational Z/2 x Z/4 torsion.
//!
//! It implements the "baby step giant step" optimization for stage 2
//! as described in section 5.2 of https://eecm.cr.yp.to/eecm-20111008.pdf
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
//! TODO: merge Montgomery arithmetic with arith library

use std::cmp::min;
use std::sync::atomic::{AtomicBool, Ordering};

use num_integer::Integer;
use rayon::prelude::*;

use crate::arith;
use crate::arith_montgomery::{MInt, ZmodN};
use crate::fbase;
use crate::Uint;

// Run ECM with automatically selected parameters. The goal of this function
// is not to completely factor numbers, but to detect cases where a number
// has a relatively small prime factor (about size(n) / 5)
pub fn ecm_auto(n: Uint, tpool: Option<&rayon::ThreadPool>) -> Option<(Uint, Uint)> {
    // The CPU budget here is only a few seconds (at most 1% of SIQS time).
    // So we intentionally use small parameters hoping to be very lucky.
    // Best D values have D/phi(D) > 4.3
    //
    // Sample parameters can be found at https://eecm.cr.yp.to/performance.html
    match n.bits() {
        0..=190 => {
            // Will quite often find a 30-32 bit factor (budget 10-20ms)
            // B2 = D² = 44100
            ecm(n, 16, 120, 210, 1, tpool)
        }
        191..=220 => {
            // Will quite often find a 36 bit factor (budget <100ms)
            // B2 = D² = 78400
            ecm(n, 16, 120, 280, 1, tpool)
        }
        221..=250 => {
            // Will quite often find a factor of size 42-46 bits (budget 0.1-0.5s)
            // B2 = D² = 176400
            ecm(n, 50, 500, 420, 1, tpool)
        }
        251..=280 => {
            // Will quite often find a factor of size 52-56 bits (budget 2-3s)
            // B2 = D² = 700k
            ecm(n, 120, 2_000, 840, 1, tpool)
        }
        281..=310 => {
            // Will often find a factor of size 58-62 bits (budget 5-10s)
            // B2 = D² ~= 1.6M
            ecm(n, 150, 4_000, 1260, 1, tpool)
        }
        311..=340 => {
            // Will often find a factor of size 64-70 bits (budget 20-30s)
            // B2 = D² = 4M
            ecm(n, 150, 12_000, 2730, 1, tpool)
        }
        341.. => {
            // Try to find a factor of size 68-76 bits (budget 1min)
            // B2 = D² = 11.3M
            ecm(n, 200, 30_000, 3570, 1, tpool)
        }
    }
}

// Factor number using purely ECM. This may never end, or fail.
pub fn ecm_only(n: Uint, tpool: Option<&rayon::ThreadPool>) -> Option<(Uint, Uint)> {
    match n.bits() {
        0..=64 => {
            // This is probably guaranteed to work.
            ecm(n, 100, 128, 210, 2, tpool)
        }
        65..=80 => {
            // Try to find a factor of size 36 bits
            ecm(n, 300, 400, 420, 2, tpool)
        }
        81..=96 => {
            // Try to find a factor of size 48 bits
            ecm(n, 800, 1700, 840, 2, tpool)
        }
        97..=128 => {
            // Try to find a factor of size 52 bits
            ecm(n, 1000, 2500, 1050, 2, tpool)
        }
        129..=160 => {
            // Try to find a factor of size 60 bits
            // 1000 curves are often not enough for 80-bit factors.
            ecm(n, 1000, 6000, 1470, 2, tpool)
        }
        161..=192 => {
            // Try to find a factor of size >70 bits
            // This will probably take a very long time.
            // TODO: try smaller parameters first.
            ecm(n, 5000, 15_000, 11550, 2, tpool)
        }
        193..=256 => {
            // Try to find a factor of size ~80?? bits
            // It may find factors of ~70 bits but not much more.
            // It will take several seconds per curve.
            // TODO: try smaller parameters first.
            ecm(n, 15000, 25_000, 30030, 2, tpool)
        }
        257.. => {
            // This is mostly to fill the table, but what we do?
            // TODO: try smaller parameters first.
            ecm(n, 40000, 40_000, 60060, 2, tpool)
        }
    }
}

// Run ECM for a given number of curves and bounds B1, B2.
pub fn ecm(
    n: Uint,
    curves: usize,
    b1: usize,
    d: usize,
    verbose: usize,
    tpool: Option<&rayon::ThreadPool>,
) -> Option<(Uint, Uint)> {
    if verbose > 0 {
        eprintln!(
            "Attempting ECM with {curves} curves B1={b1} D={d} (B2={})",
            d * d
        );
    }
    let start = std::time::Instant::now();
    let zn = ZmodN::new(n);
    let sb = SmoothBase::new(b1, d);
    // Try good curves first. They have large torsion (extra factor 3 or 4)
    // so their order is more probably smooth.
    let done = AtomicBool::new(false);
    let do_good_curve = |(idx, &(x1, x2, y1, y2))| {
        if done.load(Ordering::Relaxed) {
            return None;
        }
        if verbose > 1 {
            eprintln!("Trying good Edwards curve G=({x1}/{x2},{y1}/{y2})");
        }
        let c = Curve::from_fractional_point(zn.clone(), x1, x2, y1, y2);
        if let res @ Some((p, _)) = ecm_curve(&sb, &zn, &c) {
            if verbose > 0 {
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
        if verbose > 1 {
            eprintln!(
                "Trying Edwards curve with (2,4)-torsion d=({}/{})² G=({},{})",
                5 * k + 7,
                (3 * k + 8) * (4 * k + 9),
                3 * k + 8,
                4 * k + 9
            );
        }
        let c = Curve::from_point(zn.clone(), Uint::from(3 * k + 8), Uint::from(4 * k + 9));
        if let res @ Some((p, _)) = ecm_curve(&sb, &zn, &c) {
            if verbose > 0 {
                eprintln!(
                    "ECM success {}/{curves} for Edwards curve d=({}/{})² G=({},{}) p={p} elapsed={:.3}s",
                    k as usize + GOOD_CURVES.len() + 1,
                    5 * k + 7,
                    (3 * k + 8) * (4 * k + 9),
                    3 * k + 8,
                    4 * k + 9,
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
    if verbose > 0 {
        eprintln!("ECM failure after {:.3}s", start.elapsed().as_secs_f64());
    }
    None
}

fn ecm_curve(sb: &SmoothBase, zn: &ZmodN, c: &Curve) -> Option<(Uint, Uint)> {
    let n = &zn.n;
    // ECM stage 1
    let mut g = c.gen();
    for block in sb.factors.chunks(8) {
        for &f in block {
            g = c.scalar32_mul(f, &g);
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

    // Prepare values of abs(b): there are phi(d)/2 < d/4 such values.
    let mut bs = Vec::with_capacity(sb.d / 4);
    for b in 1..sb.d / 2 {
        if Integer::gcd(&b, &sb.d) == 1 {
            bs.push(b);
        }
    }
    // For our values of D the gap between successive values of b is less than 16,
    // or up to 22 in the case of D > 10000
    let g2 = c.double(&g);
    let g4 = c.double(&g2);
    let g6 = c.add(&g2, &g4);
    let g8 = c.double(&g4);
    let g10 = c.add(&g4, &g6);
    let g12 = c.double(&g6);
    let g14 = c.add(&g6, &g8);
    let g16 = c.double(&g8);
    let g18 = c.add(&g6, &g10);
    let g20 = c.double(&g10);
    let g22 = c.add(&g10, &g12);
    let gaps = [g2, g4, g6, g8, g10, g12, g14, g16, g18, g20, g22];
    // Compute the baby steps
    let mut bsteps = Vec::with_capacity(sb.d / 4);
    let mut bg = g.clone();
    let mut bexp = 1;
    assert_eq!(bs[0], 1);
    bsteps.push(bg.clone());
    for &b in &bs[1..] {
        let gap = b - bexp;
        bg = c.add(&bg, &gaps[gap / 2 - 1]);
        bsteps.push(bg.clone());
        bexp = b;
    }
    // Compute the giant steps
    let mut gsteps = Vec::with_capacity(sb.d);
    let dg = c.scalar32_mul(sb.d as u32, &g);
    let mut gg = dg.clone();
    for _ in 0..sb.d {
        gsteps.push(gg.clone());
        gg = c.add(&gg, &dg);
    }
    // Normalize, 2 modular inversion using batch inversion.
    batch_normalize(&zn, &mut bsteps);
    batch_normalize(&zn, &mut gsteps);
    // Compute O(d²) products
    let mut buffer = zn.one();
    for (idx, pg) in gsteps.iter().enumerate() {
        // Compute the gcd after each row for finer granularity.
        for pb in &bsteps {
            // y(G) - y(B)
            let delta_y = zn.sub(&pg.1, &pb.1);
            buffer = zn.mul(&buffer, &delta_y);
        }
        if idx % 8 == 0 || idx == gsteps.len() - 1 {
            let d = Integer::gcd(n, &Uint::from(buffer));
            if d > Uint::ONE && d < *n {
                return Some((d, n / d));
            }
        }
    }
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
/// Chunks of primes are multiplied into u32.
pub struct SmoothBase {
    factors: Box<[u32]>,
    // A number d such that d² ~ B2
    // In BSGS to check that a point has order B2, we look for
    // q in [0,d) r in [-d/2,d/2] such that [qd]P = [-r]P
    d: usize,
}

impl SmoothBase {
    pub fn new(b1: usize, d: usize) -> Self {
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
            if buffer * pow >= 1 << 32 {
                factors.push(buffer as u32);
                buffer = 1;
            }
            buffer *= pow;
        }
        if buffer > 1 {
            factors.push(buffer as u32)
        }
        SmoothBase {
            factors: factors.into_boxed_slice(),
            d,
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
#[derive(Clone)]
struct Point(MInt, MInt, MInt);

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
    // Construct a curve from a Q-rational point (x1/x2, y1/y2).
    fn from_fractional_point(zn: ZmodN, x1: i64, x2: i64, y1: i64, y2: i64) -> Curve {
        // This assumes that the denominators are invertible mod n.
        // Small primes must have been eliminated beforehand.
        fn to_uint(n: Uint, x: i64) -> Uint {
            if x >= 0 {
                Uint::from(x as u64) % n
            } else {
                n - Uint::from((-x) as u64) % n
            }
        }
        let x2inv = zn.from_int(arith::inv_mod(to_uint(zn.n, x2), zn.n).unwrap());
        let y2inv = zn.from_int(arith::inv_mod(to_uint(zn.n, y2), zn.n).unwrap());
        let gx = zn.mul(zn.from_int(to_uint(zn.n, x1)), x2inv);
        let gy = zn.mul(zn.from_int(to_uint(zn.n, y1)), y2inv);
        // Compute d = (x1²y2²+x2²y1²-x2²y2²) / (x1²y1²)
        let dnum: i64 = x1 * x1 * y2 * y2 + x2 * x2 * y1 * y1 - x2 * x2 * y2 * y2;
        let dden: i64 = x1 * x1 * y1 * y1;
        let d = zn.mul(
            zn.from_int(to_uint(zn.n, dnum)),
            zn.from_int(arith::inv_mod(to_uint(zn.n, dden), zn.n).unwrap()),
        );
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
        c
    }

    // Construct the unique curve through a point with nonzero
    // coordinates.
    fn from_point(zn: ZmodN, x: Uint, y: Uint) -> Curve {
        // Compute d = (x²+y²-1) / x²y²
        let xinv = zn.from_int(arith::inv_mod(x, zn.n).unwrap());
        let yinv = zn.from_int(arith::inv_mod(y, zn.n).unwrap());
        let gx = zn.from_int(x);
        let gy = zn.from_int(y);
        // gx*gx + gy*gy - 1
        let dn = zn.sub(zn.add(zn.mul(gx, gx), zn.mul(gy, gy)), zn.one());
        let dd = zn.mul(zn.mul(xinv, xinv), zn.mul(yinv, yinv));
        let d = zn.mul(dn, dd);
        Curve { zn, d, gx, gy }
    }

    fn gen(&self) -> Point {
        Point(self.gx, self.gy, self.zn.from_int(Uint::ONE))
    }

    // Addition formula following add-2007-bl
    // https://hyperelliptic.org/EFD/g1p/auto-edwards-projective.html#addition-add-2007-bl
    fn add(&self, p: &Point, q: &Point) -> Point {
        // 12 multiplications are required.
        let zn = &self.zn;
        // Handle z
        let a = zn.mul(p.2, q.2);
        let b = zn.mul(a, a);
        // Karatsuba-like product
        let c = zn.mul(p.0, q.0);
        let d = zn.mul(p.1, q.1);
        let cd = zn.mul(zn.add(p.0, p.1), zn.add(q.0, q.1));
        let cross = zn.sub(cd, zn.add(c, d));

        let e = zn.mul(self.d, zn.mul(c, d));
        let f = zn.sub(b, e);
        let g = zn.add(b, e);

        let x = zn.mul(zn.mul(a, f), cross);
        let y = zn.mul(zn.mul(a, g), zn.sub(d, c));
        let z = zn.mul(f, g);
        Point(x, y, z)
    }

    // Doubling formula following dbl-2007-bl
    // https://hyperelliptic.org/EFD/g1p/auto-edwards-projective.html#doubling-dbl-2007-bl
    fn double(&self, p: &Point) -> Point {
        // 7 multiplications are required.
        let zn = &self.zn;
        // Handle z
        let x_plus_y = zn.add(p.0, p.1);
        let b = zn.mul(x_plus_y, x_plus_y);
        let c = zn.mul(p.0, p.0);
        let d = zn.mul(p.1, p.1);
        let e = zn.add(c, d);
        let h = zn.mul(p.2, p.2);
        let j = zn.sub(zn.sub(e, h), h);
        // Final result
        let x = zn.mul(zn.sub(b, e), j);
        let y = zn.mul(e, zn.sub(c, d));
        let z = zn.mul(e, j);
        Point(x, y, z)
    }

    fn scalar32_mul(&self, k: u32, p: &Point) -> Point {
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
}

#[test]
fn test_curve() {
    use std::str::FromStr;
    let n = Uint::from_str("2953951639731214343967989360202131868064542471002037986749").unwrap();
    let c = Curve::from_point(ZmodN::new(n), Uint::from(2_u64), Uint::from(3_u64));
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
    let c = Curve::from_point(zn.clone(), Uint::from(2_u64), Uint::from(10_u64));
    let sb = SmoothBase::new(1000, 630);
    let res = ecm_curve(&sb, &zn, &c);
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
    let c = Curve::from_point(zn.clone(), Uint::from(2_u64), Uint::from(132_u64));
    let sb = SmoothBase::new(15000, 1050);
    let res = ecm_curve(&sb, &zn, &c);
    eprintln!("{:?}", res);
    assert_eq!(res, Some((p, q)));
}
