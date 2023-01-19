// Copyright 2022, 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Implementation of Pollard P-1
//!
//! # Small integers (64-bit)
//!
//! A specialized version of Pollard P-1 for double large primes is provided
//! by [PM1Base].
//!
//! The target is to factor numbers of size 40-52 bits which are products
//! of primes above the factor base, slightly biased so that one factor
//! remains small. The factor base is precomputed and can be reused.
//!
//! For example, while factoring RSA-100 half of numbers to be factored
//! have a factor such that p-1 is 5000-smooth.
//! The expected cost of SQUFOF is O(N^1/4) divisions so Pollard P-1
//! is interesting if it works in less than ~10000 multiplications.
//!
//! The "2-step" Pollard rho makes it more efficient:
//! in the (2^22, 2^27) range, the largest factors of p-1 are:
//! => 1st largest: a median of ~2^(N/2+1), 75% quantile around 10x-20x that value
//! => 2nd largest: almost always smaller than 500, in >90% of cases
//!
//! We can thus apply the following strategy:
//! Stage 1: multiply by small primes up to bound B=500 (~1000 multiplications)
//! Stage 2: test g^k-1 for k in large primes (500..400000)
//! where g^k are computed recursively using "gaps" (prime gaps range from 2 to 112).
//! (up to ~32000 multiplications)
//!
//! Stages 1 and 2 can be shrinked according to available CPU budget.
//!
//! Most 64-bit semiprimes that are not products of 2 strong primes can be factored in this way.
//!
//! # Large integers
//!
//! A generic Pollard P-1 implementation for possibly large integers
//! is provided by functions [pm1_only] and [pm1_quick]
//!
//! It uses the multipoint evaluation model for large D=sqrt(B2)
//! where D is such that D ~= 5 φ(D).

use num_integer::Integer;

use crate::arith_montgomery::{mg_2adic_inv, mg_mul, mg_redc, MInt, ZmodN};
use crate::arith_poly::Poly;
use crate::fbase;
use crate::Uint;

/// A factor base for Pollard P-1.
/// Pairs of factors are multiplied into u32.
pub struct PM1Base {
    // Compact blocks of factors, up to bound 500 (95 primes)
    factors: Box<[u32]>,
    // 16384 primes from 500 to ~180000
    larges: Box<[u32]>,
}

impl PM1Base {
    pub fn new() -> Self {
        let primes = fbase::primes(70000);
        let mut factors = vec![];
        let mut larges = vec![];
        let mut buffer = 1_u64;
        for p in primes {
            // Small primes are raised to some power.
            if p < 500 {
                let p = p as u64;
                let mut pow = p;
                while pow * p < 1024 {
                    pow *= p;
                }
                if buffer * pow >= 1 << 32 {
                    factors.push(buffer as u32);
                    buffer = 1;
                }
                buffer *= pow;
            } else {
                if larges.len() < 64 * 1024 {
                    larges.push(p)
                }
            }
        }
        if buffer > 1 {
            factors.push(buffer as u32)
        }
        PM1Base {
            factors: factors.into_boxed_slice(),
            larges: larges.into_boxed_slice(),
        }
    }

    // Tentatively factor number n with a budget of squarings.
    // budget=40000 will factor about 75% of 50-bit semiprimes
    // for about 6x-8x less time than SQUFOF.
    pub fn factor(&self, n: u64, budget: usize) -> Option<(u64, u64)> {
        assert!(n % 2 == 1);
        // We have a lot of modular reductions to compute,
        // so we use Montgomery forms.
        // Precompute opposite inverse of n mod R (R=2^64)
        let ninv = mg_2adic_inv(n);

        // Compute 2^K-1 mod n where K bit length is <= budget
        // 2R mod N,
        let one_r = ((1u128 << 64) % (n as u128)) as u64;
        let minus_one_r = n - one_r;
        let mut xr = (2 * one_r) % n;
        debug_assert!(mg_redc(n, ninv, xr as u128) == 2);
        // Small primes is assumed to have a cost of 1024 (95 primes).
        let fmax = std::cmp::min(self.factors.len(), budget * self.factors.len() / 1024);
        for block in self.factors[..fmax].chunks(8) {
            for &f in block {
                // Compute x^f
                let mut res = one_r;
                let mut sq = xr;
                let mut exp = f;
                while exp > 0 {
                    if exp & 1 == 1 {
                        res = mg_mul(n, ninv, res, sq);
                    }
                    sq = mg_mul(n, ninv, sq, sq);
                    exp /= 2;
                }
                xr = res;
            }
            // Maybe we have finished?
            // No need to reduce out of Montgomery form, subtract R
            // to get 2^K R - R = (2^K-1)R
            let d = Integer::gcd(&n, &(xr + minus_one_r));
            if d > 1 && d < n {
                return Some((d, n / d));
            }
        }
        let d = Integer::gcd(&n, &(xr + minus_one_r));
        if d > 1 && d < n {
            return Some((d, n / d));
        }
        // Start stage 2.
        // We still have not factored but maybe the order of xr is a small prime
        // since we have eliminated small factors.
        if budget < 1001 {
            return None;
        }
        let pmax = std::cmp::min(self.larges.len(), budget - 1000);
        // Compute xr^2k for 2k = 2 ... 86
        let xr2 = mg_mul(n, ninv, xr, xr);
        // jumps[k] = xr^(2k+2)
        let mut jumps = [0u64; 64];
        let mut j = xr2;
        for k in 1..=jumps.len() {
            jumps[k - 1] = j;
            j = mg_mul(n, ninv, j, xr2);
        }
        // The first large prime is 503 = 120 * 4 + 22 + 1
        let xr240 = mg_mul(n, ninv, jumps[120 / 2 - 1], jumps[120 / 2 - 1]);
        let xr480 = mg_mul(n, ninv, xr240, xr240);
        let xr502 = mg_mul(n, ninv, xr480, jumps[22 / 2 - 1]);
        let mut h = mg_mul(n, ninv, xr502, xr);
        let mut product = h + minus_one_r;
        let mut exp = 503;
        debug_assert!(self.larges[0] == 503);
        for (idx, &p) in self.larges[1..pmax].iter().enumerate() {
            if idx % 64 == 0 {
                // Factoring will fail if both factors have similar
                // largest primes.
                let d = Integer::gcd(&n, &product);
                if d > 1 && d < n {
                    return Some((d, n / d));
                }
            }
            // Accumulate the product of (h^p - 1) for primes p
            let gap = (p - exp) as usize;
            h = mg_mul(n, ninv, h, jumps[gap / 2 - 1]);
            product = mg_mul(n, ninv, product, h + minus_one_r);
            exp = p;
        }
        let d = Integer::gcd(&n, &product);
        if d > 1 && d < n {
            return Some((d, n / d));
        } else {
            None
        }
    }
}

// Similarly to ECM implementation, run Pollard P-1 with small parameters
// to possibly detect a small factor. The cost is 2 multiplications per large
// prime and it should use a CPU budget similar to 1 ECM run.
pub fn pm1_quick(n: Uint) -> Option<(Uint, Uint)> {
    match n.bits() {
        // Catches many 24-bit factors in 1-5ms.
        0..=190 => pm1_impl(n, 256, 210),
        // Takes less than a few ms
        191..=220 => pm1_impl(n, 512, 462),
        // Takes less than 0.01 second
        221..=250 => pm1_impl(n, 1024, 840),
        // Takes less than 0.1 second
        251..=280 => pm1_impl(n, 8192, 2310),
        // Takes less than 0.5 second
        281..=310 => pm1_impl(n, 256 << 10, 9240),
        // Takes about 1 second
        311..=340 => pm1_impl(n, 1 << 20, 19110),
        // Takes about 2-5 seconds
        341..=370 => pm1_impl(n, 4 << 20, 39270),
        // Above this size, quadratic sieve will be extremely
        // long so allow a lot of CPU budget into Pollard P-1.
        371..=420 => pm1_impl(n, 16 << 20, 79170),
        421..=470 => pm1_impl(n, 64 << 20, 159390),
        471.. => pm1_impl(n, 256 << 20, 324870),
    }
}

/// Perform Pollard P-1 with a large CPU budget.
/// This is however unlikely to produce interesting results
/// but should be faster than ECM with similar B1/B2.
pub fn pm1_only(n: Uint) -> Option<(Uint, Uint)> {
    match n.bits() {
        0..=80 => pm1_impl(n, 16 << 10, 462),
        81..=120 => pm1_impl(n, 64 << 10, 1050),
        121..=160 => pm1_impl(n, 256 << 10, 2310),
        161..=200 => pm1_impl(n, 1 << 20, 4620),
        201..=240 => pm1_impl(n, 4 << 20, 9240),
        241..=280 => pm1_impl(n, 32 << 20, 19110),
        281..=320 => pm1_impl(n, 64 << 20, 39270),
        321..=370 => pm1_impl(n, 128 << 20, 79170),
        // Below parameters can take several minutes.
        371..=420 => pm1_impl(n, 256 << 20, 159390),
        421..=470 => pm1_impl(n, 512 << 20, 324870),
        471.. => pm1_impl(n, 1 << 30, 649740),
    }
}

const MULTIEVAL_THRESHOLD: u64 = 2000;

#[doc(hidden)]
pub fn pm1_impl(n: Uint, b1: u64, sqrtb2: u64) -> Option<(Uint, Uint)> {
    let b2 = sqrtb2 * sqrtb2;
    assert!(b1 > 3);
    let zn = ZmodN::new(n);
    let mut sieve = fbase::PrimeSieve::new();
    let mut block = sieve.next();
    let mut expblock = 1u64;
    // Stage 1
    let mut g = zn.from_int(Uint::from(2_u64));
    let mut p_prev: u32 = 3;
    'stage1: loop {
        for &p in block {
            if p > b1 as u32 {
                break 'stage1;
            }
            let p = p as u64;
            let mut pow = p;
            while pow * p < 1024 {
                pow *= p;
            }
            if 1 << expblock.leading_zeros() <= pow {
                // process exponent block
                g = exp_modn(&zn, &g, expblock);
                expblock = 1;
            }
            expblock *= pow;
            p_prev = p as u32;
        }
        // Check GCD after each prime block
        let d = Integer::gcd(&n, &Uint::from(zn.sub(&g, &zn.one())));
        if d > Uint::ONE && d < n {
            return Some((d, n / d));
        }
        block = sieve.next();
    }

    // Stage 2
    // g is 2^(product of small primes) mod n
    // gaps[i] = g^2i+2
    // The vector is grown dynamically during iteration.
    if sqrtb2 > MULTIEVAL_THRESHOLD {
        return pm1_stage2_polyeval(&zn, sqrtb2 as usize, g);
    }
    let g2 = zn.mul(&g, &g);
    let mut gaps = vec![g2];

    // Compute g^pprev
    let mut x = exp_modn(&zn, &g, p_prev as u64);
    // Accumulate product of (g^p-1)
    let one = zn.one();
    let mut product = zn.sub(&x, &one);
    'stage2: loop {
        for &p in block {
            if p > b2 as u32 {
                break 'stage2;
            }
            if p <= p_prev {
                continue;
            }
            let gap = (p - p_prev) as usize;
            // Extend gaps if needed.
            while gaps.len() <= gap / 2 {
                gaps.push(zn.mul(gaps[gaps.len() - 1], g2));
            }
            assert!(gap > 0 && gap % 2 == 0, "gap={gap} p_prev={p_prev} p={p}");
            x = zn.mul(x, gaps[gap / 2 - 1]);
            product = zn.mul(&product, &zn.sub(&x, &one));
            p_prev = p;
        }
        block = sieve.next();
        // Check GCD after each prime block
        let d = Integer::gcd(&n, &Uint::from(product));
        if d > Uint::ONE && d < n {
            return Some((d, n / d));
        }
    }
    let d = Integer::gcd(&n, &Uint::from(product));
    if d > Uint::ONE && d < n {
        return Some((d, n / d));
    }
    None
}

fn exp_modn(zn: &ZmodN, g: &MInt, exp: u64) -> MInt {
    let mut res = zn.one();
    let mut sq = g.clone();
    let mut exp = exp;
    while exp > 0 {
        if exp & 1 == 1 {
            res = zn.mul(&res, &sq);
        }
        sq = zn.mul(&sq, &sq);
        exp /= 2;
    }
    res
}

fn pm1_stage2_polyeval(zn: &ZmodN, sqrtb2: usize, g: MInt) -> Option<(Uint, Uint)> {
    // Instead of computing g^p for all primes p in [b1, b2]
    // write the unknown p as qD - r where r < D and gcd(r,D)=1
    // and look for (g^D)^q == g^r modulo some unknown factor.
    let n = &zn.n;
    let g2 = zn.mul(&g, &g);
    let mut gaps = vec![g2];

    let mut bs = Vec::with_capacity(sqrtb2 / 2);
    for b in 1..sqrtb2 {
        if Integer::gcd(&b, &sqrtb2) == 1 {
            bs.push(b);
        }
    }
    // Compute the baby steps
    let mut bsteps = Vec::with_capacity(sqrtb2 / 2);
    let mut bg = g.clone();
    let mut bexp = 1;
    assert_eq!(bs[0], 1);
    bsteps.push(bg.clone());
    for &b in &bs[1..] {
        let gap = b - bexp;
        while gaps.len() <= gap / 2 {
            gaps.push(zn.mul(gaps[gaps.len() - 1], g2));
        }
        bg = zn.mul(&bg, &gaps[gap / 2 - 1]);
        bsteps.push(bg.clone());
        bexp = b;
    }
    // Compute the giant steps
    let mut gsteps = Vec::with_capacity(sqrtb2);
    let dg = exp_modn(zn, &g, sqrtb2 as u64);
    let mut gg = dg.clone();
    for _ in 0..sqrtb2 {
        gsteps.push(gg.clone());
        gg = zn.mul(&gg, &dg);
    }

    // Compute:
    // P = product(X - bg[i])
    // product P(gg[j])
    let vals = Poly::roots_eval(zn, &gsteps, &bsteps);
    let mut buffer = zn.one();
    for (idx, &v) in vals.iter().enumerate() {
        // Compute the gcd every few rows for finer granularity.
        buffer = zn.mul(buffer, v);
        if idx % 8 == 0 || idx == vals.len() - 1 {
            let d = Integer::gcd(n, &Uint::from(buffer));
            if d > Uint::ONE && d < *n {
                return Some((d, n / d));
            }
        }
    }
    None
}

#[test]
fn test_pm1_basic() {
    let ns: &[u64] = &[
        235075827453629, // max divisor 29129
        166130059616737, // max divisor 467
        159247921097933, // max divisor 3251
        224077614412439, // max divisor 1759
        219669028971857, // max divisor 193
    ];
    let pb = PM1Base::new();
    'nextn: for &n in ns {
        for budget in [500, 1000, 1500, 2000, 4000, 7000, 10000, 20000] {
            if let Some((p, q)) = pb.factor(n, budget) {
                eprintln!("factored {n} with budget {budget} => {p}*{q}");
                assert_eq!(p * q, n);
                continue 'nextn;
            }
        }
        panic!("failed to factor {n}");
    }
}

#[test]
fn test_pm1_random() {
    let pb = PM1Base::new();
    for bits in [20, 22, 24, 26, 28] {
        for budget in [500, 2000, 5000, 10000, 20000, 35000, 65000] {
            let mut seed = 1234567_u32;
            let mut primes = vec![];
            let mut ok = 0;
            let mut attempts = 0;
            for _ in 0..1000 {
                seed = seed.wrapping_mul(123456789);
                let p = seed % (1 << bits);
                if !fbase::certainly_composite(p as u64) {
                    primes.push(p);
                }
                if primes.len() == 2 {
                    let p = primes[0];
                    let q = primes[1];
                    attempts += 1;
                    if let Some((x, y)) = pb.factor(p as u64 * q as u64, budget) {
                        assert_eq!(x * y, p as u64 * q as u64);
                        ok += 1;
                    }
                    primes.clear();
                }
            }
            eprintln!(
                "{} bits, budget={budget} factored {ok}/{attempts} semiprimes",
                2 * bits
            );
        }
    }
}

#[test]
fn test_pm1_uint() {
    use std::str::FromStr;

    // A 128-bit prime such that p-1 is smooth:
    // p-1 = 2 * 5 * 29 * 89 * 211 * 433 * 823 * 1669 * 4013 * 7717 * 416873
    let p128 = Uint::from_str("41815371135748981224332082131").unwrap();
    // A 256-bit strong prime
    let p256 = Uint::from_str(
        "92504863121296400653652753711376140294298584431452956354291724864471735145079",
    )
    .unwrap();
    let n = p128 * p256;
    let Some((p, q)) = pm1_impl(n, 30000, 1_050)
        else { panic!("failed Pollard P-1") };
    assert_eq!(p, p128);
    assert_eq!(q, p256);

    // 2^3 * 3 * 7 * 11 * 31 * 131 * 1109 * 1699 * 8317 * 5984903
    let p = Uint::from_str("703855808397033138741049").unwrap();
    let n = p * p256;
    let Some((p, q)) = pm1_impl(n, 30000, 4_200)
        else { panic!("failed Pollard P-1") };
    assert_eq!(p, p);
    assert_eq!(q, p256);
}
