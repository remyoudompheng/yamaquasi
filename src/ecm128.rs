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

use crate::arith_montgomery::{Mod128, Mod192, Mod64, Zmod, ZmodN, M128};
use crate::ecm;
use crate::params::stage2_params;
use crate::{Preferences, Uint, Verbosity};

pub fn ecm128(n: Uint, try_harder: bool, prefs: &Preferences) -> Option<(Uint, Uint)> {
    let n128 = u128::cast_from(n);
    // If we want to completely factor the input number, multiply the average
    // required number of curves by the size of n.
    let multiplier = if try_harder { n.bits() as usize } else { 1 };
    let v = prefs.verbosity;
    let (p, q) = match n.bits() {
        // Target factors of size sqrt(n): use 2x the number of expected curves.
        0..=32 => ecm(n128, 6 * multiplier, 16, 660., v),
        33..=40 => ecm(n128, 6 * multiplier, 40, 1080., v),
        41..=48 => ecm(n128, 8 * multiplier, 50, 1920., v),
        // For smallest inputs, boost the number of curves to avoid fallback
        // to slower Pollard's Rho or SIQS (2x expected curves for success)
        49..=56 => ecm(n128, 8 * 2 * multiplier, 100, 3e3, v),
        57..=64 => ecm(n128, 10 * 2 * multiplier, 180, 7.7e3, v),
        65..=72 => ecm(n128, 12 * 2 * multiplier, 350, 13.2e3, v),
        73..=80 => ecm(n128, 30 * 2 * multiplier, 600, 20e3, v),
        // ECM is no longer optimal, reduce the number of curves (1/10th of required curves)
        // and keep small parameters.
        81..=128 if !try_harder => ecm(n128, 3 * multiplier, 600, 20e3, v),
        81..=88 if try_harder => ecm(n128, 20 * multiplier, 1000, 53e3, v),
        89..=96 if try_harder => ecm(n128, 30 * multiplier, 1500, 81e3, v),
        97..=112 if try_harder => ecm(n128, 70 * multiplier, 3600, 181e3, v),
        113..=128 if try_harder => ecm(n128, 80 * multiplier, 10000, 554e3, v),
        // Only numbers below 128 bits are supported.
        _ => None,
    }?;
    Some((p.into(), q.into()))
}

/// Factor a number known to be a double large prime from quadratic sieve
///
/// Assumes that the smallest factor has size 25-31 bits.
pub fn ecm_semiprime(n: u64) -> Option<(u64, u64)> {
    let (p, q) = if n >> 52 == 0 {
        // Assume a 25-bit factor
        ecm(n as u128, 40, 60, 1920., Verbosity::Silent)?
    } else if n >> 58 == 0 {
        // Assume a 28-bit factor
        ecm(n as u128, 80, 100, 3000., Verbosity::Silent)?
    } else {
        // Assume a 30-bit factor
        ecm(n as u128, 60, 180, 7700., Verbosity::Silent)?
    };
    Some((p as u64, q as u64))
}

#[doc(hidden)]
pub fn ecm(n: u128, curves: usize, b1: u64, b2: f64, v: Verbosity) -> Option<(u128, u128)> {
    assert!(n & 1 == 1 && n > 3, "n={n}");
    let zn = ZmodN::new(Uint::from(n));
    let suyama = ecm::Suyama11::new(&zn).unwrap();
    let sb = ecm::SmoothBase::new(b1 as usize, false);
    if v >= Verbosity::Info {
        eprintln!("Attempting small ECM with {curves} curves B1={b1} B2={b2:e}",);
    }
    let start = std::time::Instant::now();
    for seed in 1..=curves {
        let g = suyama
            .element(seed as u32 + 1)
            .and_then(|p| ecm::Suyama11::params_point(&zn, &p));
        let g = match g {
            Ok(g) => g,
            Err(ecm::UnexpectedLargeFactor(p)) => {
                let p = u128::cast_from(p);
                if p < n {
                    if v >= Verbosity::Info {
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
        if let res @ Some((p, _)) = ecm_curve(&c, &sb, b2, v) {
            if v >= Verbosity::Info {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "Small ECM found factor {p} at curve {seed}/{curves} elapsed={elapsed:.2}ms"
                );
            }
            return res;
        }
    }
    if v >= Verbosity::Info {
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!("Small ECM failure after {elapsed:.2}ms");
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
    // Assume that factors will not be found simultaneously
    for &f in sb.factors.iter() {
        let fg = c.scalar64_mul(f, &g);
        if fg.0 .0 == 0 {
            // All factors are found, maybe the previous point
            // was the solution? Otherwise abort.
            let d = Integer::gcd(&g.0 .0, &n);
            if d > 1 && d < n {
                return Some((d, n / d));
            }
            return None;
        }
        g = fg;
    }
    // FIXME: gcd factors
    let d = Integer::gcd(&g.0 .0, &n);
    if d > 1 && d < n {
        return Some((d, n / d));
    }
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
    let mut n_bsteps = 1;
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
        let stage1 = elapsed1.as_secs_f64() * 1000.0;
        let stage2 = start2.elapsed().as_secs_f64() * 1000.0;
        eprintln!("ECM128 stage1={stage1:.3}ms stage2={stage2:.3}ms");
    }
    // FIXME: gcd factors
    let d = Integer::gcd(&buffer.0, &c.n);
    if d > 1 && d < n {
        return Some((d, n / d));
    }
    None
}

// Run ECM for a number assumed to be smooth (returns multiple factors).
#[doc(hidden)]
pub fn ecm_curve_smooth(n: &Uint, curveseed: u32, sb: &ecm::SmoothBase, b2: f64) -> Vec<Uint> {
    assert!(n.bits() <= 192);

    let mut facs = vec![];

    // Build curve.
    let zn = ZmodN::new(*n);
    let g = if (curveseed as usize) < ecm::SUYAMA_POINTS.len() {
        let p = ecm::SUYAMA_POINTS[curveseed as usize];
        // Transform as [3x-z, 3y, 3z]
        let p = ecm::Point(
            zn.from_i64(3 * p.0 - p.2),
            zn.from_i64(3 * p.1),
            zn.from_i64(3 * p.2),
        );
        match ecm::Suyama11::params_point(&zn, &p) {
            Ok(g) => g,
            Err(ecm::UnexpectedLargeFactor(p)) => {
                facs.push(p.into());
                return facs;
            }
        }
    } else {
        let suyama = ecm::Suyama11::new(&zn).unwrap();
        match suyama
            .element(curveseed + 1)
            .and_then(|p| ecm::Suyama11::params_point(&zn, &p))
        {
            Ok(g) => g,
            Err(ecm::UnexpectedLargeFactor(p)) => {
                facs.push(p);
                return facs;
            }
        }
    };

    let (gx, gy, gz) = g.xyz();
    drop(g);
    let (nred, mut g1) = if n.bits() <= 64 {
        let zn = Mod64::new64(u64::cast_from(*n));
        let g = (u64::from(gx), u64::from(gy), u64::from(gz));
        ecm_curve_stage1(ECurve(zn), g, sb, &mut facs)
    } else if n.bits() <= 128 {
        let zn = Mod128::new128(u128::cast_from(*n));
        let g = (u128::from(gx), u128::from(gy), u128::from(gz));
        ecm_curve_stage1(ECurve(zn), g, sb, &mut facs)
    } else if n.bits() <= 192 {
        let zn = Mod192::new(*n);
        let g = (
            <Mod192 as Zmod>::Val::cast_from(Uint::from(gx)),
            <Mod192 as Zmod>::Val::cast_from(Uint::from(gy)),
            <Mod192 as Zmod>::Val::cast_from(Uint::from(gz)),
        );
        ecm_curve_stage1(ECurve(zn), g, sb, &mut facs)
    } else {
        unreachable!()
    };

    // Stage 2
    if &nred < n {
        g1.0 %= nred;
        g1.1 %= nred;
        g1.2 %= nred;
    }
    if nred.bits() <= 64 {
        //eprintln!("ECM stage2 64");
        let zn = Mod64::new64(u64::cast_from(nred));
        let g = (
            u64::cast_from(g1.0),
            u64::cast_from(g1.1),
            u64::cast_from(g1.2),
        );
        ecm_curve_stage2(ECurve(zn), g, b2, &mut facs)
    } else if nred.bits() <= 128 {
        //eprintln!("ECM stage2 128");
        let zn = Mod128::new128(u128::cast_from(nred));
        let g = (
            u128::cast_from(g1.0),
            u128::cast_from(g1.1),
            u128::cast_from(g1.2),
        );
        ecm_curve_stage2(ECurve(zn), g, b2, &mut facs)
    } else if n.bits() <= 192 {
        let zn = Mod192::new(*n);
        let g = (
            <Mod192 as Zmod>::Val::cast_from(Uint::from(gx)),
            <Mod192 as Zmod>::Val::cast_from(Uint::from(gy)),
            <Mod192 as Zmod>::Val::cast_from(Uint::from(gz)),
        );
        ecm_curve_stage2(ECurve(zn), g, b2, &mut facs)
    } else {
        unreachable!()
    };

    facs
}

fn ecm_curve_stage1<Z>(
    c: ECurve<Z>,
    g: (Z::Val, Z::Val, Z::Val),
    sb: &ecm::SmoothBase,
    facs: &mut Vec<Uint>,
) -> (Uint, (Uint, Uint, Uint))
where
    Z: Zmod + Copy,
    Z::Val: Copy,
    Uint: CastFrom<Z::Val>,
{
    let mut g = g;
    let mut gs = vec![g.0];
    // Assume that factors will not be found simultaneously
    for &f in sb.factors.iter() {
        g = c.scalar64_mul(f, g);
        gs.push(g.0);
        //if g.0 == 0 { break; }
    }
    let nred = c.0.gcd_factors(&gs, facs);
    drop(gs);
    /*
    if crate::pseudoprime(nred.into()) {
        //eprintln!("finished {nred}!");
        return facs;
    }
    */
    (
        Uint::cast_from(nred),
        (
            Uint::cast_from(g.0),
            Uint::cast_from(g.1),
            Uint::cast_from(g.2),
        ),
    )
}

fn ecm_curve_stage2<Z, V>(c: ECurve<Z>, g: (V, V, V), b2: f64, facs: &mut Vec<Uint>)
where
    Z: Zmod<Val = V> + Copy,
    Z::Val: Copy,
{
    let (_, d1, d2) = stage2_params(b2);
    let zn = c.0;
    //assert!(c.is_valid(&c.ext(g)));
    fn proj<T>(p: (T, T, T, T)) -> (T, T, T) {
        (p.0, p.1, p.2)
    }

    // Prepare values of abs(b): there are phi(d1)/2 < d1/4 such values.
    let mut bs = Vec::with_capacity(d1 as usize / 4);
    for b in 1..d1 / 2 {
        if Integer::gcd(&b, &d1) == 1 {
            bs.push(b);
        }
    }
    let g2 = c.dblext(g);
    let g4 = c.dblext(proj(g2));
    let mut gaps: Vec<(_, _, _, _)> = vec![g2, g4];
    // Baby/giant steps in a single vector.
    let mut steps: Vec<(_, _, _)> = Vec::with_capacity(d1 as usize / 4 + d2 as usize);
    // Compute the baby steps
    let mut bg = c.ext(g);
    let mut bexp = 1;
    assert_eq!(bs[0], 1);
    steps.push(g.clone());
    let mut n_bsteps = 1 as usize;
    for &b in &bs[1..] {
        let gap = b - bexp;
        while gaps.len() < gap as usize / 2 {
            let gap2 = c.add(gaps[0], gaps[gaps.len() - 1]);
            gaps.push(gap2);
        }
        bg = c.add(bg, gaps[gap as usize / 2 - 1]);
        steps.push(proj(bg));
        n_bsteps += 1;
        bexp = b;
    }
    // Compute the giant steps
    // WARNING: extended coordinate addition must not be used
    // to compute the first step.
    let dg = c.scalar64_mul(d1, g);
    let dg2 = c.dblext(dg);
    let dgext = c.ext(dg);
    let mut gg = dg2.clone();
    steps.push(dg);
    steps.push(proj(dg2));
    for _ in 2..d2 {
        gg = c.add(gg, dgext);
        steps.push(proj(gg));
    }
    // Clear denominators
    {
        let l = steps.len();
        let mut u = steps[0].2;
        for i in 1..l {
            steps[i].1 = zn.mul(steps[i].1, u);
            u = zn.mul(u, steps[i].2);
        }
        u = steps[l - 1].2;
        for i in 2..=l {
            steps[l - i].1 = zn.mul(steps[l - i].1, u);
            u = zn.mul(u, steps[l - i].2);
        }
    }

    let bsteps = &steps[..n_bsteps];
    let gsteps = &steps[n_bsteps..];
    // Compute O(d*phi(d)) products
    let mut buffer = zn.one();
    let mut prods = Vec::with_capacity(gsteps.len());
    prods.push(buffer);
    for pg in gsteps {
        // Compute the gcd after each row for finer granularity.
        for pb in bsteps {
            // y(G) - y(B)
            let delta_y = zn.abs_diff(pg.1, pb.1);
            buffer = zn.mul(buffer, delta_y);
        }
        prods.push(buffer);
    }
    zn.gcd_factors(&prods, facs);
}

/// An elliptic curve in twisted Edwards form -ax²+y² = 1 + dx²y²
/// Coefficient d is not stored and implicitly contained in the generator coordinates.
pub struct Curve {
    zn: Mod128,
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
#[doc(hidden)]
pub struct Point(pub M128, pub M128, pub M128);

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
            zn: Mod128::new128(n),
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
        let zn = Mod128::new128(n);
        Curve {
            zn,
            n,
            ninv,
            one: r,
            g,
        }
    }

    pub fn from_point(n: u128, p: Point) -> Self {
        // Don't recompute each time?
        let ninv = M128::inv_2adic(n);
        let (r, _) = M128::r_r2(n, ninv);
        Curve {
            zn: Mod128::new128(n),
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

    // Formula add-2008-hwcd-4 in Explicit Formula Database
    fn add(&self, p: &ExtPoint, q: &ExtPoint) -> ExtPoint {
        let r = ECurve(self.zn).add(
            (p.0 .0, p.1 .0, p.2 .0, p.3 .0),
            (q.0 .0, q.1 .0, q.2 .0, q.3 .0),
        );
        ExtPoint(M128(r.0), M128(r.1), M128(r.2), M128(r.3))
    }

    // Formula is dbl-2008-hwcd (8 multiplications)
    // <https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#doubling-dbl-2008-hwcd>
    fn dblext(&self, p: &Point) -> ExtPoint {
        let r = ECurve(self.zn).dblext((p.0 .0, p.1 .0, p.2 .0));
        ExtPoint(M128(r.0), M128(r.1), M128(r.2), M128(r.3))
    }

    pub fn scalar64_mul(&self, k: u64, p: &Point) -> Point {
        let p = ECurve(self.zn).scalar64_mul(k, (p.0 .0, p.1 .0, p.2 .0));
        Point(M128(p.0), M128(p.1), M128(p.2))
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

// Curve arithmetic in fixed precision.
struct ECurve<Z>(Z);

impl<V, Z: Zmod<Val = V>> ECurve<Z>
where
    V: Copy,
    Z: Copy,
{
    pub fn ext(&self, p: (V, V, V)) -> (V, V, V, V) {
        let (x, y, z) = p;
        let zn = self.0;
        (zn.mul(x, z), zn.mul(y, z), zn.mul(z, z), zn.mul(x, y))
    }

    // Computes 2P + Q using 8+7 multiplications.
    fn dbladd(&self, p: (V, V, V), q: (V, V, V, V)) -> (V, V, V) {
        let zn = self.0;

        let p = self.dblext(p);
        // Copy-paste add(p,q) but omit one coordinate.
        // (yP - xP)(yQ + xQ), (yP + xP)(yQ - xQ)
        let a = zn.mul(zn.sub(p.1, p.0), zn.add(q.1, q.0));
        let b = zn.mul(zn.add(p.1, p.0), zn.sub(q.1, q.0));
        // 2 zP tQ, 2 zQ tP
        let c = zn.mul(p.2, q.3);
        let c = zn.add(c, c);
        let d = zn.mul(p.3, q.2);
        let d = zn.add(d, d);
        let e = zn.add(d, c);
        let f = zn.sub(b, a);
        let g = zn.add(b, a);
        let h = zn.sub(d, c);
        // xy(P+Q) = zt(P+Q) = efgh
        (zn.mul(e, f), zn.mul(g, h), zn.mul(f, g))
    }

    // Formula add-2008-hwcd-4 in Explicit Formula Database
    fn add(&self, p: (V, V, V, V), q: (V, V, V, V)) -> (V, V, V, V) {
        let zn = self.0;
        // (yP - xP)(yQ + xQ), (yP + xP)(yQ - xQ)
        let a = zn.mul(zn.sub(p.1, p.0), zn.add(q.1, q.0));
        let b = zn.mul(zn.add(p.1, p.0), zn.sub(q.1, q.0));
        // 2 zP tQ, 2 zQ tP
        let c = zn.mul(p.2, q.3);
        let c = zn.add(c, c);
        let d = zn.mul(p.3, q.2);
        let d = zn.add(d, d);
        let e = zn.add(d, c);
        let f = zn.sub(b, a);
        let g = zn.add(b, a);
        let h = zn.sub(d, c);
        // xy(P+Q) = zt(P+Q) = efgh
        (zn.mul(e, f), zn.mul(g, h), zn.mul(f, g), zn.mul(e, h))
    }

    // Doubling formula following dbl-2008-bbjlp (7 multiplications)
    // https://hyperelliptic.org/EFD/g1p/auto-twisted-projective.html#doubling-dbl-2008-bbjlp
    fn double(&self, p: (V, V, V)) -> (V, V, V) {
        let zn = self.0;
        // Similar to the begining of dblext()
        let x_plus_y = zn.add(p.0, p.1);
        let b = zn.mul(x_plus_y, x_plus_y);
        let c = zn.mul(p.0, p.0);
        let d = zn.mul(p.1, p.1);
        let c_plus_d = zn.add(c, d);
        let f = zn.sub(d, c);
        let h = zn.mul(p.2, p.2);
        let j = zn.sub(zn.sub(f, h), h);
        let x = zn.mul(zn.sub(b, c_plus_d), j);
        let y = zn.mul(f, zn.sub(zn.cast(0), c_plus_d));
        let z = zn.mul(f, j);
        (x, y, z)
    }

    // Formula is dbl-2008-hwcd (8 multiplications)
    // <https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html#doubling-dbl-2008-hwcd>
    fn dblext(&self, p: (V, V, V)) -> (V, V, V, V) {
        let zn = self.0;
        let x_y = zn.add(p.0, p.1);
        let x_y2 = zn.mul(x_y, x_y);
        let a = zn.mul(p.0, p.0);
        let b = zn.mul(p.1, p.1);
        let c = zn.mul(p.2, p.2);
        let c = zn.add(c, c);
        let e = zn.sub(zn.sub(x_y2, a), b);
        // D=-A, G=B-A, H=-B-A, F=G-C
        let g = zn.sub(b, a);
        let f = zn.sub(g, c);
        let h = zn.sub(zn.cast(0), zn.add(a, b));
        (zn.mul(e, f), zn.mul(g, h), zn.mul(f, g), zn.mul(e, h))
    }

    pub fn scalar64_mul(&self, k: u64, p: (V, V, V)) -> (V, V, V) {
        let zn = self.0;
        // Prepare small steps.
        let pext = self.ext(p);
        let p2 = self.dblext(p);
        let p3 = self.add(pext, p2);
        let p5 = self.add(p3, p2);
        let p7 = self.add(p5, p2);
        let gaps = [&pext, &p3, &p5, &p7];
        // Encode the chain as:
        // 2l (doubling l times)
        // ±k (add/sub kP where k is odd)
        let mut c = [0_i8; 32];
        let l = ecm::Curve::make_addition_chain(&mut c, k);
        // Get initial element (chain[l-1] = 1 or 3 or 5 or 7)
        let q = gaps[c[l - 1] as usize / 2];
        let mut q = (q.0, q.1, q.2);
        for idx in 1..l {
            let op = c[l - 1 - idx];
            if op % 2 == 0 {
                for _ in 0..op / 2 {
                    // FIXME: 1 MUL wasted
                    q = self.double(q);
                }
            } else {
                if op > 0 {
                    q = self.dbladd(q, *gaps[op as usize / 2]);
                } else {
                    let mut gap = gaps[(-op) as usize / 2].clone();
                    gap.0 = zn.sub(zn.cast(0), gap.0);
                    gap.3 = zn.sub(zn.cast(0), gap.3);
                    q = self.dbladd(q, gap);
                }
            }
        }
        q
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

#[test]
fn test_ecm_small() {
    // Test that tiny integers can be factored (during stage 1, both factors
    // can be found very closely).
    // Examples were found during random testing for various
    // input sizes.
    let ns = &[
        211 * 401,
        269 * 479,
        263 * 839,
        347 * 859,
        769 * 797,
        919 * 1889,
        1187 * 1693,
        1319 * 2663,
        3203 * 6449,
        8297 * 12473,
    ];
    let mut prefs = Preferences::default();
    prefs.verbosity = Verbosity::Info;
    for &n in ns {
        if let Some((p, q)) = ecm128(Uint::from(n), false, &prefs) {
            assert_eq!(p * q, Uint::from_digit(n));
        } else {
            panic!("failure for {n}");
        }
    }
}

#[test]
fn test_ecm_smooth() {
    // 17227965250634428706527504247997363: 3 13 701 5399 10979 11423 47149 1005287 19635073
    // Most prime factors are not P-1 smooth.
    let n: u128 = 17227965250634428706527504247997363 / 39;
    let sb = ecm::SmoothBase::new(60, false);
    let facs = ecm_curve_smooth(&Uint::from(n), 1, &sb, 660.0);
    assert!(facs.len() >= 3);
    let facs = ecm_curve_smooth(&Uint::from(n), 2, &sb, 660.0);
    assert!(facs.len() >= 3);
    let facs = ecm_curve_smooth(&Uint::from(n), 3, &sb, 660.0);
    assert!(facs.len() >= 3);

    let sb = ecm::SmoothBase::new(125, false);
    let facs = ecm_curve_smooth(&Uint::from(n), 1, &sb, 1920.0);
    assert!(facs.len() >= 4);
    let facs = ecm_curve_smooth(&Uint::from(n), 2, &sb, 1920.0);
    assert!(facs.len() >= 4, "{facs:?}");
}
