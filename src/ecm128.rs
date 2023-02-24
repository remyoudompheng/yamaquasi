// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/// A specialized implementation of ECM for 128-bit integers
///
/// It is meant to only run twisted Edwards curves for input integers
/// under 128 bits, for a limited number of iterations and,
/// without the FFT extension, hoping to find factors faster than quadratic sieve.
///
/// It usually runs for less than 1ms and is appropriate for 60-80 bit numbers.
///
/// Bibliography
/// Bos, Kleinjung, "ECM at Work", https://eprint.iacr.org/2012/089
use bnum::cast::CastFrom;
use num_integer::Integer;

use crate::arith_montgomery::{self, ZmodN};
use crate::ecm;
use crate::params::stage2_params;
use crate::{Preferences, Uint, Verbosity};

pub fn ecm128(n: Uint, try_harder: bool, prefs: &Preferences) -> Option<(Uint, Uint)> {
    // The CPU budget here is only a few seconds (at most 1% of SIQS time).
    // So we intentionally use small parameters hoping to be very lucky.
    // This means that the parameters should be tuned to find factors
    // smaller than about n^1/4.
    let n128 = u128::cast_from(n);
    let multiplier = if try_harder { 1000 } else { 1 };
    let (p, q) = match n.bits() {
        // Target factors of size sqrt(n): use 2x the number of expected curves.
        0..=40 => ecm(n128, 4 * multiplier, 50, 1080., prefs),
        41..=48 => ecm(n128, 8 * multiplier, 50, 1920., prefs),
        49..=56 => ecm(n128, 8 * multiplier, 100, 3e3, prefs),
        57..=64 => ecm(n128, 10 * multiplier, 180, 7.7e3, prefs),
        65..=72 => ecm(n128, 12 * multiplier, 350, 13.2e3, prefs),
        73..=80 => ecm(n128, 30 * multiplier, 600, 20e3, prefs),
        // ECM is no longer optimal, reduce the number of curves (1/10th of required curves)
        81..=88 => ecm(n128, 2 * multiplier, 1000, 53e3, prefs),
        89..=96 => ecm(n128, 3 * multiplier, 1500, 81e3, prefs),
        97..=112 => ecm(n128, 7 * multiplier, 3600, 181e3, prefs),
        113..=128 => ecm(n128, 8 * multiplier, 10000, 554e3, prefs),
        // Only numbers below 128 bits are supported.
        129.. => None,
    }?;
    Some((p.into(), q.into()))
}

fn ecm(n: u128, curves: usize, b1: u64, b2: f64, prefs: &Preferences) -> Option<(u128, u128)> {
    let zn = ZmodN::new(Uint::from(n));
    let suyama = ecm::Suyama11::new(&zn).unwrap();
    let sb = ecm::SmoothBase::new(b1 as usize, false);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Attempting small ECM with {curves} curves B1={b1} B2={b2:e}",);
    }
    let start = std::time::Instant::now();
    for seed in 1..=curves {
        let g = suyama
            .element(seed as u32 + 1)
            .and_then(|p| suyama.params_point(&p));
        let g = match g {
            Ok(g) => g,
            Err(ecm::UnexpectedLargeFactor(p)) => {
                let p = u128::cast_from(p);
                if p < n {
                    if prefs.verbose(Verbosity::Info) {
                        eprintln!("Unexpected factor {p} while selecting curve");
                    }
                    return Some((p, n / p));
                }
                continue;
            }
        };

        // Convert point to u128.
        let (gx, gy, gz) = g.xyz();
        let g128 = Point(
            M128(u128::cast_from(Uint::from(gx))),
            M128(u128::cast_from(Uint::from(gy))),
            M128(u128::cast_from(Uint::from(gz))),
        );
        let c = Curve::from_point(n, g128);
        if let res @ Some((p, _)) = ecm_curve(&c, &sb, b2, prefs.verbosity) {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if prefs.verbose(Verbosity::Info) {
                eprintln!(
                    "Small ECM found factor {p} at curve {seed}/{curves} elapsed={elapsed:.2}ms"
                );
            }
            return res;
        }
    }
    None
}

/// Runs ECM for a single curve.
///
/// This implementation is a simplified version of `ecm::ecm_curve`.
fn ecm_curve(
    c: &Curve,
    sb: &ecm::SmoothBase,
    b2: f64,
    verbosity: Verbosity,
) -> Option<(u128, u128)> {
    let n = c.n;
    let (_, d1, d2) = stage2_params(b2);
    // ECM stage 1
    let start1 = std::time::Instant::now();
    let mut g = c.gen().clone();
    let mut gxs = Vec::with_capacity(sb.factors.len());
    gxs.push(c.one);
    for &f in sb.factors.iter() {
        g = c.scalar64_mul(f, &g);
        gxs.push(g.0);
    }
    // FIXME: gcd factors
    let d = Integer::gcd(&g.0 .0, &n);
    if d > n && d < n {
        return Some((d, n / d));
    }
    drop(gxs);
    assert!(c.is_valid(&c.ext(&g)));
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

    let sub = |x, y| M128::sub(c.n, x, y);
    let mul = |x, y| M128::mul(c.n, c.ninv, x, y);
    // Prepare values of abs(b): there are phi(d1)/2 < d1/4 such values.
    let mut bs = Vec::with_capacity(d1 as usize / 4);
    for b in 1..d1 / 2 {
        if Integer::gcd(&b, &d1) == 1 {
            bs.push(b);
        }
    }
    let g2 = c.dblext(&g);
    let g4 = c.dblext(&g2.proj());
    let mut gaps: Vec<ExtPoint> = vec![g2, g4];
    // Baby/giant steps in a single vector.
    let mut steps: Vec<Point> = Vec::with_capacity(d1 as usize / 4 + d2 as usize);
    // Compute the baby steps
    let mut bg = c.ext(&g);
    let mut bexp = 1;
    assert_eq!(bs[0], 1);
    steps.push(g.clone());
    let mut n_bsteps = 1 as usize;
    for &b in &bs[1..] {
        let gap = b - bexp;
        while gaps.len() < gap as usize / 2 {
            let gap2 = c.add(&gaps[0], &gaps[gaps.len() - 1]);
            gaps.push(gap2);
        }
        bg = c.add(&bg, &gaps[gap as usize / 2 - 1]);
        steps.push(bg.proj());
        n_bsteps += 1;
        bexp = b;
    }
    // Compute the giant steps
    // WARNING: extended coordinate addition must not be used
    // to compute the first step.
    let dg = c.scalar64_mul(d1, &g);
    let dg2 = c.dblext(&dg);
    let dgext = c.ext(&dg);
    let mut gg = dg2.clone();
    steps.push(dg);
    steps.push(dg2.proj());
    for _ in 2..d2 {
        gg = c.add(&gg, &dgext);
        steps.push(gg.proj());
    }
    // Normalize Y coordinates, 4 multiplications per point:
    // replace y[i] by y[i]/z[i] * product(z[j])
    // Compute u[i] = p[0] * ... * p[i-1]
    // Compute v[i] = p[i+1] * ... * p[d]
    // Then product(z[j]) / z[i] = u[i] * v[i]
    {
        let l = steps.len();
        let mut u = steps[0].2;
        for i in 1..l {
            steps[i].1 = mul(steps[i].1, u);
            u = mul(u, steps[i].2);
        }
        u = steps[l - 1].2;
        for i in 2..=l {
            steps[l - i].1 = mul(steps[l - i].1, u);
            u = mul(u, steps[l - i].2);
        }
    }

    let bsteps = &steps[..n_bsteps];
    let gsteps = &steps[n_bsteps..];
    // Compute O(d*phi(d)) products
    let mut buffer = c.one;
    let mut prods = Vec::with_capacity(gsteps.len());
    prods.push(buffer);
    for pg in gsteps {
        // Compute the gcd after each row for finer granularity.
        for pb in bsteps {
            // y(G) - y(B)
            let delta_y = sub(pg.1, pb.1);
            buffer = mul(buffer, delta_y);
        }
        prods.push(buffer);
    }
    if verbosity >= Verbosity::Verbose {
        let stage1 = elapsed1.as_secs_f64();
        let stage2 = start2.elapsed().as_secs_f64();
        eprintln!("ECM128 stage1={stage1:.6}s stage2={stage2:.6}s");
    }
    // FIXME: gcd factors
    let d = Integer::gcd(&buffer.0, &c.n);
    if d > 1 && d < n {
        return Some((d, n / d));
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct M128(u128);

// Montgomery modular arithmetic for 128-bit moduli.
// If the modulus fits in 64 bits, operations are defined with
// multiplier 2^64 instead of 2^128.
impl M128 {
    fn inv_2adic(n: u128) -> u128 {
        debug_assert!(n % 2 == 1);
        // Use 64-bit inverse as starting point.
        let mut x = arith_montgomery::mg_2adic_inv(n as u64) as u128;
        if n >> 64 == 0 {
            return x;
        }
        loop {
            let rem = n.wrapping_mul(x) - 1;
            if rem == 0 {
                break;
            }
            x += 1 << rem.trailing_zeros();
        }
        assert!(n.wrapping_mul(x) == 1);
        1 + !x
    }

    // Compute 2^128 and 2^256 modulo n.
    fn r_r2(n: u128, ninv: u128) -> (M128, M128) {
        // (2^128 - n) % n
        let r = M128(0_u128.wrapping_sub(n) % n);
        if n >> 64 == 0 {
            return (M128((1 << 64) % n), r);
        }
        // 2^256 % n
        // This is the Montgomery representation of 2^128.
        let two = M128::add(n, r, r);
        let mut r2 = two;
        for _ in 0..7 {
            r2 = Self::mul(n, ninv, r2, r2);
        }
        (r, r2)
    }

    fn add(n: u128, x: M128, y: M128) -> M128 {
        let (x, y) = (x.0, y.0);
        let my = n - y;
        if x >= my {
            M128(x - my)
        } else {
            M128(x + y)
        }
    }

    fn sub(n: u128, x: M128, y: M128) -> M128 {
        let (x, y) = (x.0, y.0);
        if x >= y {
            M128(x - y)
        } else {
            M128(x + (n - y))
        }
    }

    fn mul(n: u128, ninv: u128, x: M128, y: M128) -> M128 {
        if n >> 64 == 0 {
            return M128(
                arith_montgomery::mg_mul(n as u64, ninv as u64, x.0 as u64, y.0 as u64) as u128,
            );
        }
        fn mul256(x: u128, y: u128) -> (u128, u128) {
            // Compute the 256-bit product xy.
            let (x0, x1) = (x as u64, (x >> 64) as u64);
            let (y0, y1) = (y as u64, (y >> 64) as u64);
            let mut xy0 = x0 as u128 * y0 as u128;
            let mut xy1 = x1 as u128 * y1 as u128;
            let (mid, mut c) = (x0 as u128 * y1 as u128).overflowing_add(x1 as u128 * y0 as u128);
            if c {
                xy1 += 1 << 64;
            }
            (xy0, c) = xy0.overflowing_add(mid.wrapping_shl(64));
            xy1 += mid >> 64;
            if c {
                xy1 += 1;
            }
            (xy0, xy1)
        }
        // Apply the REDC algorithm.
        let (xy0, xy1) = mul256(x.0, y.0);
        if xy0 == 0 {
            return M128(xy1);
        }
        // mul cannot be zero.
        let m = xy0.wrapping_mul(ninv);
        // Return xy1 + 1 + highword(mul * n).
        let mn1 = mul256(m, n).1;
        let d = n - mn1 - 1;
        if xy1 >= d {
            M128(xy1 - d)
        } else {
            M128(xy1 + mn1 + 1)
        }
    }
}

/// An elliptic curve in twisted Edwards form -ax²+y² = 1 + dx²y²
/// Coefficient d is not stored and implicitly contained in the generator coordinates.
pub struct Curve {
    /// The modulus.
    n: u128,
    /// Helper constants for modular arithmetic.
    ninv: u128, // inverse of n modulo 2^128
    one: M128,
    // Coordinates of a "non-torsion" base point.
    g: Point,
}

// A curve point in projective coordinates.
#[derive(Clone, Debug)]
pub struct Point(M128, M128, M128);

// A curve point in extended coordinates (on quadric XY=ZT).
#[derive(Clone, Debug)]
pub struct ExtPoint(M128, M128, M128, M128);

impl ExtPoint {
    fn proj(&self) -> Point {
        Point(self.0, self.1, self.2)
    }
}

// Conversion from multiprecision curve to 128-bit curve.
impl From<&ecm::Curve> for Curve {
    fn from(c: &ecm::Curve) -> Curve {
        assert!(c.is_twisted128());
        let n = u128::cast_from(*c.n());
        let ninv = M128::inv_2adic(n);
        let (r, _) = M128::r_r2(n, ninv);
        let g = c.gen().xyz();
        Curve {
            n,
            ninv,
            one: r,
            g: Point(
                M128(u128::cast_from(Uint::from(g.0))),
                M128(u128::cast_from(Uint::from(g.1))),
                M128(u128::cast_from(Uint::from(g.2))),
            ),
        }
    }
}

impl Curve {
    pub fn from_fractional_point(n: u128, x1: i64, x2: i64, y1: i64, y2: i64) -> Self {
        // Point x1/x2 y1/y2 has extended coordinates:
        // (x1 y2, x2 y1, y1 y2, x1 x2)
        let ninv = M128::inv_2adic(n);
        let (r, r2) = M128::r_r2(n, ninv);
        let mul = |a: i64, b: i64| {
            let m = if a.signum() * b.signum() < 0 {
                M128(n - (a.abs() as u128 * b.abs() as u128))
            } else {
                M128(a.abs() as u128 * b.abs() as u128)
            };
            M128::mul(n, ninv, m, r2)
        };
        let g = Point(mul(x1, y2), mul(y1, x2), mul(x2, y2));
        Curve { n, ninv, one: r, g }
    }

    pub fn from_point(n: u128, p: Point) -> Self {
        // Don't recompute each time?
        let ninv = M128::inv_2adic(n);
        let (r, _) = M128::r_r2(n, ninv);
        Curve {
            n,
            ninv,
            one: r,
            g: p,
        }
    }

    pub fn gen(&self) -> &Point {
        &self.g
    }

    pub fn ext(&self, p: &Point) -> ExtPoint {
        let mul = |x, y| M128::mul(self.n, self.ninv, x, y);
        ExtPoint(mul(p.0, p.2), mul(p.1, p.2), mul(p.2, p.2), mul(p.0, p.1))
    }

    // Computes 2P + Q using 8+7 multiplications.
    fn dbladd(&self, p: &Point, q: &ExtPoint) -> Point {
        let add = |x, y| M128::add(self.n, x, y);
        let sub = |x, y| M128::sub(self.n, x, y);
        let mul = |x, y| M128::mul(self.n, self.ninv, x, y);

        let p = self.dblext(p);
        // Copy-paste add(p,q) but omit one coordinate.
        // (yP - xP)(yQ + xQ), (yP + xP)(yQ - xQ)
        let a = mul(sub(p.1, p.0), add(q.1, q.0));
        let b = mul(add(p.1, p.0), sub(q.1, q.0));
        // 2 zP tQ, 2 zQ tP
        let c = mul(p.2, q.3);
        let c = add(c, c);
        let d = mul(p.3, q.2);
        let d = add(d, d);
        let e = add(d, c);
        let f = sub(b, a);
        let g = add(b, a);
        let h = sub(d, c);
        // xy(P+Q) = zt(P+Q) = efgh
        Point(mul(e, f), mul(g, h), mul(f, g))
    }

    // Formula add-2008-hwcd-4 in Explicit Formula Database
    fn add(&self, p: &ExtPoint, q: &ExtPoint) -> ExtPoint {
        let add = |x, y| M128::add(self.n, x, y);
        let sub = |x, y| M128::sub(self.n, x, y);
        let mul = |x, y| M128::mul(self.n, self.ninv, x, y);
        // (yP - xP)(yQ + xQ), (yP + xP)(yQ - xQ)
        let a = mul(sub(p.1, p.0), add(q.1, q.0));
        let b = mul(add(p.1, p.0), sub(q.1, q.0));
        // 2 zP tQ, 2 zQ tP
        let c = mul(p.2, q.3);
        let c = add(c, c);
        let d = mul(p.3, q.2);
        let d = add(d, d);
        let e = add(d, c);
        let f = sub(b, a);
        let g = add(b, a);
        let h = sub(d, c);
        // xy(P+Q) = zt(P+Q) = efgh
        ExtPoint(mul(e, f), mul(g, h), mul(f, g), mul(e, h))
    }

    // Doubling formula following dbl-2008-bbjlp (7 multiplications)
    // https://hyperelliptic.org/EFD/g1p/auto-twisted-projective.html#doubling-dbl-2008-bbjlp
    fn double(&self, p: &Point) -> Point {
        let add = |x, y| M128::add(self.n, x, y);
        let sub = |x, y| M128::sub(self.n, x, y);
        let mul = |x, y| M128::mul(self.n, self.ninv, x, y);
        // Similar to the begining of dblext()
        let x_plus_y = add(p.0, p.1);
        let b = mul(x_plus_y, x_plus_y);
        let c = mul(p.0, p.0);
        let d = mul(p.1, p.1);
        let c_plus_d = add(c, d);
        let f = sub(d, c);
        let h = mul(p.2, p.2);
        let j = sub(sub(f, h), h);
        let x = mul(sub(b, c_plus_d), j);
        let y = mul(f, sub(M128(0), c_plus_d));
        let z = mul(f, j);
        Point(x, y, z)
    }

    // Formula is dbl-2008-hwcd (8 multiplications)
    // <https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#doubling-dbl-2008-hwcd>
    fn dblext(&self, p: &Point) -> ExtPoint {
        let add = |x, y| M128::add(self.n, x, y);
        let sub = |x, y| M128::sub(self.n, x, y);
        let mul = |x, y| M128::mul(self.n, self.ninv, x, y);
        let x_y = add(p.0, p.1);
        let x_y2 = mul(x_y, x_y);
        let a = mul(p.0, p.0);
        let b = mul(p.1, p.1);
        let c = mul(p.2, p.2);
        let c = add(c, c);
        let e = sub(sub(x_y2, a), b);
        // D=-A, G=B-A, H=-B-A, F=G-C
        let g = sub(b, a);
        let f = sub(g, c);
        let h = sub(M128(0), add(a, b));
        ExtPoint(mul(e, f), mul(g, h), mul(f, g), mul(e, h))
    }

    pub fn scalar64_mul(&self, k: u64, p: &Point) -> Point {
        // Prepare small steps.
        let pext = self.ext(p);
        let p2 = self.dblext(p);
        let p3 = self.add(&pext, &p2);
        let p5 = self.add(&p3, &p2);
        let p7 = self.add(&p5, &p2);
        let gaps = [&pext, &p3, &p5, &p7];
        // Encode the chain as:
        // 2l (doubling l times)
        // ±k (add/sub kP where k is odd)
        let mut c = [0_i8; 32];
        let l = ecm::Curve::make_addition_chain(&mut c, k);
        // Get initial element (chain[l-1] = 1 or 3 or 5 or 7)
        let mut q = gaps[c[l - 1] as usize / 2].proj();
        for idx in 1..l {
            let op = c[l - 1 - idx];
            if op % 2 == 0 {
                for _ in 0..op / 2 {
                    // FIXME: 1 MUL wasted
                    q = self.double(&q);
                }
            } else {
                if op > 0 {
                    q = self.dbladd(&q, gaps[op as usize / 2]);
                } else {
                    let mut gap = gaps[(-op) as usize / 2].clone();
                    gap.0 = M128(self.n - gap.0 .0);
                    gap.3 = M128(self.n - gap.3 .0);
                    q = self.dbladd(&q, &gap);
                }
            }
        }
        q
    }

    fn is_valid(&self, p: &ExtPoint) -> bool {
        // A point is valid if it lies on the same quadric as the generator.
        // The equation is -x^2 + y^2 = z^2 + d t^2
        let add = |x, y| M128::add(self.n, x, y);
        let sub = |x, y| M128::sub(self.n, x, y);
        let mul = |x, y| M128::mul(self.n, self.ninv, x, y);

        let g = self.ext(&self.g);
        // Compute y^2 - x^2 - z^2
        let zg = sub(mul(g.1, g.1), add(mul(g.0, g.0), mul(g.2, g.2)));
        let zp = sub(mul(p.1, p.1), add(mul(p.0, p.0), mul(p.2, p.2)));
        mul(zg, mul(p.3, p.3)) == mul(zp, mul(g.3, g.3))
    }
}

#[test]
fn test_mod128() {
    // 128-bit modular arithmetic
    let n = 242117442329613169289925222087543973661;
    let ninv = M128::inv_2adic(n);
    let (r, r2) = M128::r_r2(n, ninv);
    eprintln!("N={n}");
    eprintln!("R={r:?}");
    eprintln!("R²={r2:?}");
    assert_eq!(r, M128::mul(n, ninv, r, r));
    let x = M128::mul(n, ninv, M128(123456789), r2);
    let y = M128::mul(n, ninv, M128(987654321), r2);
    let xy = M128::mul(n, ninv, M128(123456789 * 987654321), r2);
    eprintln!("x={x:?}");
    eprintln!("y={y:?}");
    assert_eq!(xy, M128::mul(n, ninv, x, y));
}

#[test]
fn test_curve() {
    let p = 602768606663711;
    let q = 957629686686973;
    let n: u128 = p * q;
    // The σ=11 twisted Edwards curve.
    // Its order is:
    // mod p: 602768647071432 = 2^3 * 3^2 * 17 * 251 * 4679 * 419317
    // mod q: 957629727109848 = 2^3 * 3 * 13 * 359 * 8549654731
    let c = Curve::from_fractional_point(n, -11, 60, 11529, 12860);
    let g = c.gen();
    assert!(c.is_valid(&c.ext(&g)));

    let g1 = c.scalar64_mul(602768647071432, g);
    assert!(c.is_valid(&c.ext(&g1)));
    let g2 = c.scalar64_mul(957629727109848, &g1);
    assert!(c.is_valid(&c.ext(&g2)));

    eprintln!("g1 = {g1:?}");
    assert!(g1.0 .0 % p == 0);
    assert!(g2.0 .0 == 0);

    // Also test 64-bit.
    let c = Curve::from_fractional_point(p, -11, 60, 11529, 12860);
    let g = c.gen();
    assert!(c.is_valid(&c.ext(&g)));

    let g1 = c.scalar64_mul(602768647071432, g);
    assert!(g1.0 .0 == 0);
}

#[test]
fn test_ecm_curve() {
    // Test duplicated from ecm module (same example as above).
    let p = 602768606663711;
    let q = 957629686686973;
    let n: u128 = p * q;
    // This curve has smooth order for prime 602768606663711
    // 602768647071432 = 2 2 2 3 3 17 251 4679 419317
    let c = Curve::from_fractional_point(n, -11, 60, 11529, 12860);
    let sb = ecm::SmoothBase::new(5000, false);
    let res = ecm_curve(&c, &sb, 554e3, Verbosity::Verbose);
    eprintln!("{res:?}");
    assert_eq!(res, Some((p, q)));
}
