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

use num_integer::Integer;

use crate::arith;
use crate::fbase;
use crate::Uint;

// Run ECM with automatically selected parameters. The goal of this function
// is not to completely factor numbers, but to detect cases where a number
// has a relatively small prime factor (about size(n) / 5)
pub fn ecm_auto(n: Uint) -> Option<(Uint, Uint)> {
    // The CPU budget here is only a few seconds (less than 1% of SIQS time).
    // So we intentionally use suboptimal parameters hoping to be very lucky.
    //
    // Sample parameters can be found at https://eecm.cr.yp.to/performance.html
    match n.bits() {
        0..=220 => {
            // Try to find a factor of size ~40 bits (takes <100ms)
            // B2 = D² = 44100
            ecm(n, 10, 800, 210)
        }
        221..=250 => {
            // Try to find a factor of size ~48 bits (takes <1s)
            // B2 = D² = 176400
            ecm(n, 10, 1_000, 420)
        }
        251..=280 => {
            // Try to find a factor of size ~56 bits (takes ~1s)
            // B2 = D² = 396900
            ecm(n, 10, 3_000, 630)
        }
        281..=310 => {
            // Try to find a factor of size ~64 bits (takes ~2s)
            // B2 = D² ~= 1 million
            ecm(n, 20, 5_000, 1050)
        }
        311..=340 => {
            // Try to find a factor of size ~72 bits (takes ~10s)
            // B2 = D² = 5.3M
            ecm(n, 20, 15_000, 2310)
        }
        341.. => {
            // Try to find a factor of size ~80 bits (takes ~20s)
            // B2 = D² = 7.5M
            ecm(n, 20, 25_000, 2730)
        }
    }
}

// Run ECM for a given number of curves and bounds B1, B2.
pub fn ecm(n: Uint, curves: usize, b1: usize, d: usize) -> Option<(Uint, Uint)> {
    eprintln!(
        "Attempting ECM with {curves} curves B1={b1} D={d} (B2={})",
        d * d
    );
    let start = std::time::Instant::now();
    let zn = ZmodN::new(n);
    let sb = SmoothBase::new(b1, d);
    // Try good curves first. They have large torsion (extra factor 3 or 4)
    // so their order is more probably smooth.
    for &(x1, x2, y1, y2) in GOOD_CURVES {
        let c = Curve::from_fractional_point(zn.clone(), x1, x2, y1, y2);
        if let res @ Some((p, _)) = ecm_curve(&sb, &zn, &c) {
            eprintln!("ECM success for special Edwards curve G=({x1}/{x2},{y1}/{y2}) p={p} elapsed={:.3}s",
                start.elapsed().as_secs_f64());
            return res;
        }
    }
    // Choose curves through point (2,k > 1) such that d=(3+k²)/4k²
    for k in 0..curves - GOOD_CURVES.len() {
        let c = Curve::from_point(zn.clone(), Uint::from(2_u64), Uint::from(2 + k as u64));
        if let res @ Some((p, _)) = ecm_curve(&sb, &zn, &c) {
            eprintln!(
                "ECM success for Edwards curve d={}/{} G=(2,{}) p={p} elapsed={:.3}s",
                3 + (2 + k) * (2 + k),
                4 * k * k,
                2 + k,
                start.elapsed().as_secs_f64()
            );
            return res;
        }
    }
    eprintln!("ECM failure after {:.3}s", start.elapsed().as_secs_f64());
    None
}

fn ecm_curve(sb: &SmoothBase, zn: &ZmodN, c: &Curve) -> Option<(Uint, Uint)> {
    let n = &zn.n;
    // ECM stage 1
    let mut g = c.gen();
    for block in sb.factors.chunks(4) {
        for &f in block {
            g = c.scalar32_mul(f, &g);
        }
        if let Some(d) = c.is_2_torsion(&g) {
            if d > Uint::ONE && d < *n {
                return Some((d, n / d));
            }
        }
    }
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
    // For our values of D the gap between successive values of b is less than 16.
    let g2 = c.double(&g);
    let g4 = c.double(&g2);
    let g6 = c.add(&g2, &g4);
    let g8 = c.double(&g4);
    let g10 = c.add(&g4, &g6);
    let g12 = c.double(&g6);
    let g14 = c.add(&g6, &g8);
    let g16 = c.double(&g8);
    let gaps = [g2, g4, g6, g8, g10, g12, g14, g16];
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
    // Normalize, O(d) modular inversions
    // FIXME: it is theoretically useful to normalize coordinates
    // but our inversion is too slow.
    //for pg in gsteps.iter_mut() {
    //    c.normalize(pg);
    //}
    //for pb in bsteps.iter_mut() {
    //    c.normalize(pb);
    //}
    // Compute cross products, O(d²) products
    let mut buffer = zn.one();
    for (idx, pg) in gsteps.iter().enumerate() {
        // Compute the gcd after each row for finer granularity.
        for pb in &bsteps {
            // y(G) z(B) - y(B) z(G)
            //let delta_y = zn.sub(pg.1, pb.1);
            // When coordinates are not normalized, we have 2 extra multiplications.
            let delta_y = zn.sub(zn.mul(pg.1, pb.2), zn.mul(pb.1, pg.2));
            buffer = zn.mul(buffer, delta_y);
        }
        if idx % 4 == 0 || idx == gsteps.len() - 1 {
            let d = Integer::gcd(n, &buffer.0);
            if d > Uint::ONE && d < *n {
                return Some((d, n / d));
            }
        }
    }
    None
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
        // Don't use large B2 here!
        assert!(d < 5000);
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
// We select only those with denominators < 200
const GOOD_CURVES: &[(i64, i64, i64, i64)] = &[
    // Z/2 x Z/8 torsion
    (13, 7, 289, 49),     // d 25921/83521
    (1025, 158, 697, 25), // d 44182609/1766100625
    // Z/12Z torsion
    (5, 23, -1, 7),     // d -24167/25
    (8, 17, -20, 19),   // d 1375/1024
    (5, 4, -25, 89),    // d 81289/15625
    (45, 19, -3, 1),    // d 4913/18225
    (35, 109, -49, 1),  // d 1140625/117649
    (49, 101, -21, 19), // d 560947/352947
    (11, 53, -121, 4),  // d 41083561/1771561
];

impl Curve {
    // Construct a curve from a Q-rational point (x1/x2, y1/y2).
    fn from_fractional_point(zn: ZmodN, x1: i64, x2: i64, y1: i64, y2: i64) -> Curve {
        // This assumes that the denominators are invertible mod n.
        // Small primes must have been eliminated beforehand.
        fn to_uint(n: Uint, x: i64) -> Uint {
            if x >= 0 {
                Uint::from(x as u64)
            } else {
                n - Uint::from((-x) as u64)
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
        assert!(c.is_valid(&Point(gx, gy, one)));
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
        let minus_1 = zn.from_int(zn.n - Uint::ONE);
        let dn = zn.mul(gx, gx) + zn.mul(gy, gy) + minus_1;
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
        let d = Integer::gcd(&p.0 .0, &self.zn.n);
        if d == Uint::ONE {
            None
        } else {
            Some(d)
        }
    }
}

// Montgomery form arithmetic

#[derive(Clone)]
pub struct ZmodN {
    n: Uint,
    // Minus n^-1 mod R
    ninv: Uint,
    // Auxiliary base R=2^64k
    k: u32,
    // R mod n
    r: Uint,
    // R^2 mod n
    r2: Uint,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
struct MInt(Uint);

impl std::ops::Add<MInt> for MInt {
    type Output = MInt;
    fn add(self, rhs: MInt) -> MInt {
        MInt(self.0 + rhs.0)
    }
}

impl ZmodN {
    fn new(n: Uint) -> Self {
        assert!(n.bits() < Uint::BITS / 2);
        let k = (n.bits() + 63) / 64;
        let rsqrt = Uint::ONE << (32 * k);
        let r = (rsqrt * rsqrt) % n;
        let r2 = (r * r) % n;
        let ninv = {
            // Invariant: nx = 1 + 2^k s, k increasing
            let mut x = Uint::ONE;
            loop {
                let rem = n.wrapping_mul(x) - Uint::ONE;
                if rem == Uint::ZERO {
                    break;
                }
                x += Uint::ONE << rem.trailing_zeros();
            }
            assert!((n.wrapping_mul(x) - Uint::ONE).trailing_zeros() >= 64 * k);
            // Now compute R-x
            if 64 * k == Uint::BITS {
                !x + Uint::ONE
            } else {
                // clear top bits
                let x = x - ((x >> (64 * k)) << (64 * k));
                (Uint::ONE << (64 * k)) - x
            }
        };
        ZmodN { n, ninv, k, r, r2 }
    }

    fn zero(&self) -> MInt {
        MInt(Uint::ZERO)
    }

    fn one(&self) -> MInt {
        MInt(self.r)
    }

    fn from_int(&self, x: Uint) -> MInt {
        self.mul(MInt(x), MInt(self.r2))
    }

    #[cfg(test)]
    fn to_int(&self, x: MInt) -> Uint {
        self.redc(x.0).0
    }

    fn mul(&self, x: MInt, y: MInt) -> MInt {
        // all bit lengths MUST be < 512
        self.redc(uint_mul(&x.0, &y.0, self.k))
    }

    fn add(&self, x: MInt, y: MInt) -> MInt {
        let mut sum = x.0 + y.0;
        while sum > self.n {
            sum -= self.n;
        }
        MInt(sum)
    }

    fn sub(&self, x: MInt, y: MInt) -> MInt {
        debug_assert!(
            y.0.bits() < (x.0 + self.n).bits() + 2,
            "x.bits={} y.bits={} n.bits={}",
            x.0.bits(),
            y.0.bits(),
            self.n.bits()
        );
        let mut x = x.0;
        while y.0 > x {
            x += self.n;
        }
        MInt(x - y.0)
    }

    fn redc(&self, x: Uint) -> MInt {
        assert!(x < (self.n << (64 * self.k)));
        // Montgomery reduction (x/R mod n).
        // compute -x/N mod R
        // Half precision is enough.
        let mul = uint_mul(&x, &self.ninv, self.k);
        // Manually clear upper words
        let mut mul_digits = mul.digits().clone();
        for i in self.k as usize..mul_digits.len() {
            mul_digits[i] = 0
        }
        let mul = Uint::from_digits(mul_digits);
        // reduce
        let m = uint_mul(&mul, &self.n, self.k);
        let mut res = (x + m).digits().clone();
        // Shift right by 64k bits
        for i in 0..res.len() {
            res[i] = if i + (self.k as usize) < res.len() {
                res[i + self.k as usize]
            } else {
                0
            };
        }
        let mut res = Uint::from_digits(res);
        if res >= self.n {
            res -= self.n
        }
        MInt(res)
    }
}

fn uint_mul(x: &Uint, y: &Uint, sz: u32) -> Uint {
    // Desperate attempt to be faster than bnum multiplication.
    // We should definitely not be doing this but otherwise performance
    // is abysmal.
    debug_assert!(sz <= Uint::BITS / 64 / 2);
    let xd = x.digits();
    let yd = y.digits();
    let sz = sz as usize;
    let mut z = [0_u64; Uint::BITS as usize / 64];
    for i in 0..sz {
        let mut carry = 0_u64;
        for j in 0..sz {
            unsafe {
                let xi = *xd.get_unchecked(i) as u128;
                let yj = *yd.get_unchecked(j) as u128;
                let xy = xi * yj + (carry as u128);
                let zlo = xy as u64;
                let zhi = (xy >> 64) as u64;
                let zij = z[..].get_unchecked_mut(i + j);
                let (zlo2, c) = zij.overflowing_add(zlo);
                *zij = zlo2;
                carry = zhi + (if c { 1 } else { 0 });
            }
        }
        let (zlo2, c) = z[i + sz].overflowing_add(carry);
        z[i + sz] = zlo2;
        if c && i + sz + 1 < z.len() {
            z[i + sz + 1] += 1;
        }
    }
    Uint::from_digits(z)
}

#[test]
fn test_montgomery() {
    use std::str::FromStr;
    let n = Uint::from_str("2953951639731214343967989360202131868064542471002037986749").unwrap();
    let p = Uint::from_str("17917317351877").unwrap();
    let pinv = Uint::from_str("42403041586861144438126400473690086613066961901031711489").unwrap();
    let zn = ZmodN::new(n);
    let x = zn.from_int(p);
    let y = zn.from_int(pinv);
    let one = zn.from_int(Uint::ONE);
    assert_eq!(zn.to_int(x), p);
    assert_eq!(zn.to_int(y), pinv);
    assert_eq!(zn.to_int(one), Uint::ONE);
    assert_eq!(zn.mul(x, y), one);
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
