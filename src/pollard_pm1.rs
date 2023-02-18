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
//!
//! When looking for a large prime order (p=qd+r) using the baby steps giant steps
//! method, the classical method to test only φ(d)/2 values of r is to use
//! an even polynomial (x^2): such a large prime exists if g^(qd^2) = g^(r^2).
//! because (qd^2 - r^2) = (qd-r)(qd+r), allowing to test only r < d/2.
//!
//! The Bluestein (chirp-z) algorithm can compute a multipoint evaluation faster when
//! the base points are a geometric progression.
//!
//! FFT extension for algebraic-group factorization algorithms
//! Richard P. Brent, Alexander Kruppa, Paul Zimmermann
//! https://hal.inria.fr/hal-01630907/document

use bnum::types::U1024;
use num_integer::Integer;

use crate::arith_fft::convolve_modn_ntt;
use crate::arith_montgomery::{gcd_factors, mg_2adic_inv, mg_mul, mg_redc, MInt, ZmodN};
use crate::arith_poly::{Poly, PolyRing};
use crate::fbase;
use crate::{Uint, Verbosity};

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
pub fn pm1_quick(n: &Uint, v: Verbosity) -> Option<(Vec<Uint>, Uint)> {
    // Choose B1 so that stage 1 is a large part of CPU time.
    match n.bits() {
        // Ignore single word integers: the multiprecision implementation
        // is too costly.
        0..=64 => None,
        // Extremely quick run
        65..=128 => pm1_impl(n, 400, 15e3, v),
        // Catches many 24-bit factors in 1-5ms.
        129..=190 => pm1_impl(n, 600, 40e3, v),
        // Takes less than a few ms
        191..=220 => pm1_impl(n, 10_000, 270e3, v),
        // Takes less than 0.01 second
        221..=250 => pm1_impl(n, 50_000, 8e6, v),
        // Takes less than 0.1 second
        251..=280 => pm1_impl(n, 500_000, 300e6, v),
        // Takes less than 0.5 second
        281..=310 => pm1_impl(n, 1_000_000, 1.2e9, v),
        // Takes about 1 second
        311..=340 => pm1_impl(n, 2_000_000, 5e9, v),
        // Takes about 2-5 seconds
        341..=370 => pm1_impl(n, 7_000_000, 18e9, v),
        // Above this size, quadratic sieve will be extremely
        // long so allow a lot of CPU budget into Pollard P-1.
        371..=420 => pm1_impl(n, 16 << 20, 150e9, v),
        421..=470 => pm1_impl(n, 45_000_000, 2.5e12, v),
        471.. => pm1_impl(n, 160_000_000, 22e12, v),
    }
}

/// Perform Pollard P-1 with a large CPU budget.
/// This is however unlikely to produce interesting results
/// but should be faster than ECM with similar B1/B2.
pub fn pm1_only(n: &Uint, v: Verbosity) -> Option<(Vec<Uint>, Uint)> {
    match n.bits() {
        0..=80 => pm1_impl(n, 16 << 10, 450e3, v),
        81..=120 => pm1_impl(n, 64 << 10, 8e6, v),
        121..=160 => pm1_impl(n, 256 << 10, 300e6, v),
        161..=200 => pm1_impl(n, 1 << 20, 1.2e9, v),
        201..=240 => pm1_impl(n, 4 << 20, 8e9, v),
        241..=280 => pm1_impl(n, 32 << 20, 640e9, v),
        281..=320 => pm1_impl(n, 64 << 20, 1.4e12, v),
        321..=370 => pm1_impl(n, 128 << 20, 2.5e12, v),
        // Below parameters can take several minutes.
        371..=420 => pm1_impl(n, 200_000_000, 5e12, v),
        421..=470 => pm1_impl(n, 300_000_000, 10e12, v),
        471.. => pm1_impl(n, 500_000_000, 22.5e12, v),
    }
}

const MULTIEVAL_THRESHOLD: f64 = 80e3;

#[doc(hidden)]
pub fn pm1_impl(n: &Uint, b1: u64, b2: f64, verbosity: Verbosity) -> Option<(Vec<Uint>, Uint)> {
    let mut factors = vec![];
    let start1 = std::time::Instant::now();
    let (b2real, _, _) = stage2_params(b2);
    if verbosity >= Verbosity::Info {
        eprintln!("Attempting P-1 with B1={b1} B2={b2real:e}");
    }
    assert!(b1 > 3);
    // The modulus/ring can shrink as we find factors.
    let mut nred = *n;
    let mut zn = ZmodN::new(*n);
    let mut sieve = fbase::PrimeSieve::new();
    let mut block = sieve.next();
    let mut expblock = 1u64;
    let mut expblock_lg = U1024::ONE;
    let largeblocks = b1 >= 65536;
    // Stage 1
    let mut g = zn.from_int(Uint::from(2_u64));
    let mut p_prev: u32 = 1;
    let mut gpows = vec![zn.one()];
    loop {
        for &p in block {
            let p = p as u64;
            let mut pow = p;
            while pow * p < b1 {
                pow *= p;
            }
            let stop = p > b1;
            // process exponent block
            if stop || 1 << expblock.leading_zeros() <= pow {
                if !largeblocks {
                    g = exp_modn(&zn, &g, expblock);
                    gpows.push(zn.sub(&g, &zn.one()));
                    expblock = 1;
                } else {
                    expblock_lg *= U1024::from_digit(expblock);
                    expblock = 1;
                }
            }
            if stop || expblock_lg.bits() > 1024 - 32 {
                g = exp_modn_large(&zn, &g, &expblock_lg);
                gpows.push(zn.sub(&g, &zn.one()));
                expblock_lg = U1024::ONE;
            }
            p_prev = p as u32;
            if stop {
                break;
            }
            expblock *= pow;
            if g == zn.one() {
                // We can reach 1 if φ(n) is B1-smooth, no need to go further.
                break;
            }
        }
        // Check GCD after each prime block (several thousands primes)
        let logstage = Some(1).filter(|_| verbosity >= Verbosity::Verbose);
        if check_gcd_factors(&n, &mut factors, &mut nred, &mut gpows, logstage) {
            return if factors.is_empty() {
                None
            } else {
                Some((factors, nred))
            };
        }
        if zn.n != nred {
            // Shrink ring
            let gint = zn.to_int(g);
            zn = ZmodN::new(nred);
            g = zn.from_int(gint % nred);
        }
        if p_prev > b1 as u32 {
            break;
        }
        block = sieve.next();
    }
    drop(gpows);
    let elapsed1 = start1.elapsed();

    // Stage 2
    // g is 2^(product of small primes) mod n
    // gaps[i] = g^2i+2
    // The vector is grown dynamically during iteration.
    let start2 = std::time::Instant::now();
    let logtime = || {
        if verbosity >= Verbosity::Verbose {
            let elapsed2 = start2.elapsed();
            if elapsed2.as_secs_f64() < 0.01 {
                eprintln!(
                    "PM1 stage1={:.6}s stage2={:.6}s",
                    elapsed1.as_secs_f64(),
                    elapsed2.as_secs_f64()
                );
            } else {
                eprintln!(
                    "PM1 stage1={:.3}s stage2={:.3}s",
                    elapsed1.as_secs_f64(),
                    elapsed2.as_secs_f64()
                );
            }
        }
    };
    if b2 > MULTIEVAL_THRESHOLD {
        let (mut f2, n2) = pm1_stage2_polyeval(&zn, b2, g);
        factors.append(&mut f2);
        nred = n2;
        logtime();
        if !factors.is_empty() {
            return Some((factors, nred));
        }
        return None;
    }
    let g2 = zn.mul(&g, &g);
    let mut gaps = vec![g2];

    // Compute g^pprev
    let mut x = exp_modn(&zn, &g, p_prev as u64);
    // Accumulate product of (g^p-1)
    let one = zn.one();
    let mut products = Vec::with_capacity(block.len());
    products.push(one);
    let mut product = zn.sub(&x, &one);
    loop {
        for &p in block {
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
            products.push(product);
            p_prev = p;
            if p > b2 as u32 {
                break;
            }
        }
        // Check GCD after each prime block (several thousands primes)
        let logstage = Some(2).filter(|_| verbosity >= Verbosity::Verbose);
        if check_gcd_factors(&n, &mut factors, &mut nred, &mut products, logstage) {
            return if factors.is_empty() {
                None
            } else {
                Some((factors, nred))
            };
        }
        if p_prev > b2 as u32 {
            break;
        }
        block = sieve.next();
    }
    logtime();
    if factors.len() > 0 {
        return Some((factors, nred));
    }
    None
}

/// Extract factors by computing GCD and update arrays.
fn check_gcd_factors(
    n: &Uint,
    factors: &mut Vec<Uint>,
    nred: &mut Uint,
    values: &mut Vec<MInt>,
    stage: Option<usize>,
) -> bool {
    let (mut fs, nred_) = gcd_factors(nred, &values[..]);
    if fs.contains(&n) {
        return true;
    }
    if !fs.is_empty() {
        if let Some(stage) = stage {
            for &f in &fs {
                eprintln!("Found factor {f} during stage {stage}");
            }
        };
        factors.append(&mut fs);
        *nred = nred_;
        if *nred == Uint::ONE || crate::pseudoprime(*nred) {
            return true;
        }
    }
    let last = values[values.len() - 1];
    values.clear();
    values.push(last);
    false
}

fn exp_modn(zn: &ZmodN, g: &MInt, exp: u64) -> MInt {
    if exp == 0 {
        return zn.one();
    }
    // Start with MSB and consume blocks of 3 bits (optimal for 64 bits).
    // Note that 1, 3, 5, 7 have symmetrical bits.
    let mut exprev = exp.reverse_bits();
    let g2 = zn.mul(g, g);
    let g3 = zn.mul(g, &g2);
    let g5 = zn.mul(&g3, &g2);
    let g7 = zn.mul(&g5, &g2);
    let mut i = exprev.trailing_zeros();
    exprev >>= i;
    let (mut res, mut consumed) = {
        // Exprev must start with bit "1"
        match exprev & 7 {
            1 => {
                // g or g^2
                if i > 60 {
                    // Consume a single bit, we may have reached the end.
                    (*g, 1)
                } else {
                    (g2, 2)
                }
            }
            3 => (g3, 2),
            5 => (g5, 3),
            7 => (g7, 3),
            _ => unreachable!("impossible"),
        }
    };
    exprev >>= consumed;
    i += consumed;
    while i < 64 {
        consumed = if exprev & 1 == 0 {
            res = zn.mul(&res, &res);
            1
        } else {
            match exprev & 7 {
                1 => {
                    // x => x^2 g
                    res = zn.mul(&res, &res);
                    res = zn.mul(&res, g);
                    1
                }
                3 => {
                    // x => x^4 g^3
                    res = zn.mul(&res, &res);
                    res = zn.mul(&res, &res);
                    res = zn.mul(&res, &g3);
                    2
                }
                5 => {
                    // x => x^8 g^5
                    res = zn.mul(&res, &res);
                    res = zn.mul(&res, &res);
                    res = zn.mul(&res, &res);
                    res = zn.mul(&res, &g5);
                    3
                }
                7 => {
                    // x => x^8 g^7
                    res = zn.mul(&res, &res);
                    res = zn.mul(&res, &res);
                    res = zn.mul(&res, &res);
                    res = zn.mul(&res, &g7);
                    3
                }
                _ => unreachable!("impossible"),
            }
        };
        exprev >>= consumed;
        i += consumed;
    }
    res
}

type LargeExpType = U1024;

#[inline(never)]
fn exp_modn_large(zn: &ZmodN, g: &MInt, exp: &LargeExpType) -> MInt {
    // The optimal strategy for 1024-bit exponent is to use 6-bit blocks.
    // It requires 32 precomputed multiplications (exponents 1, 3, ... 63)
    // and at most 1024/6 multiplications in addition to squarings.
    let bitlen = match exp.bits() {
        0 => return zn.one(),
        1 => return *g,
        n if n <= 64 => return exp_modn(zn, g, exp.digits()[0]),
        n => n,
    };
    // Consume bits starting with MSB.
    let g2 = zn.mul(g, g);
    let mut gk = g.clone();
    // Powers g, g^3 .. g^63.
    let g_smalls = {
        let v = std::mem::MaybeUninit::<[MInt; 32]>::uninit();
        let mut v = unsafe { v.assume_init() };
        v[0] = *g;
        for i in 1..32 {
            gk = zn.mul(&gk, &g2);
            v[i] = gk;
        }
        v
    };
    // Extract 6 bits (exp >> offset & 63)
    fn expblock(exp: &LargeExpType, offset: u32) -> u64 {
        let digs = exp.digits();
        if offset >= LargeExpType::BITS - 64 {
            let w = digs[offset as usize / 64];
            (w >> (offset - (LargeExpType::BITS - 64))) & 63
        } else if offset % 64 <= 64 - 6 {
            let w = digs[offset as usize / 64];
            (w >> (offset % 64)) & 63
        } else {
            let w0 = digs[offset as usize / 64] as u128;
            let w1 = digs[offset as usize / 64 + 1] as u128;
            let w = (w1 << 64) | w0;
            (w >> (offset % 64)) as u64 & 63
        }
    }
    // Exponent is guaranteed to have more than 64 bits.
    let mut rem_bits = bitlen;
    // First block.
    let blk = expblock(exp, rem_bits - 6);
    let tz = blk.trailing_zeros();
    gk = g_smalls[(blk as usize) >> (tz + 1)];
    for _ in 0..tz {
        gk = zn.mul(&gk, &gk);
    }
    rem_bits -= 6;
    loop {
        if rem_bits == 0 {
            return gk;
        }
        if !exp.bit(rem_bits - 1) {
            gk = zn.mul(&gk, &gk);
            rem_bits -= 1;
            continue;
        }
        // Next bit is 1, fetch a block.
        if rem_bits >= 6 {
            let blk = expblock(exp, rem_bits - 6);
            let tz = blk.trailing_zeros();
            for _ in tz..6 {
                gk = zn.mul(&gk, &gk);
            }
            gk = zn.mul(&gk, &g_smalls[(blk as usize) >> (tz + 1)]);
            for _ in 0..tz {
                gk = zn.mul(&gk, &gk);
            }
            rem_bits -= 6
        } else {
            gk = zn.mul(&gk, &gk);
            gk = zn.mul(&gk, &g);
            rem_bits -= 1;
        }
    }
}

fn pm1_stage2_polyeval(zn: &ZmodN, b2: f64, g: MInt) -> (Vec<Uint>, Uint) {
    let (_, d1, d2) = stage2_params(b2);
    // Instead of computing g^p for all primes p in [b1, b2]
    // write the unknown p as qD - r where r < D and gcd(r,D)=1
    // and look for (g^D)^q == g^r modulo some unknown factor.
    //
    // Define the polynomial P(x) = product(x - g^r).
    // We need to compute P(g^D), P(g^2D) etc.
    //
    // Define
    // Pg = [P[i] g^(-i² D/2)]
    // Q = [1, g^D/2, g^4D/2 ... g^(D/2 k²)]
    //
    // FIXME: start Q at g^(D/2 i^2) where iD > B1
    //
    // Then the convolution (Pg * Q)[k] is:
    //     sum(i+j=k, P[i] g^(i² D/2) g^(K-j² D/2))
    //   = g^? * sum(i+j=k, P[i] (g^kD)^i)
    //   = g^? P(g^kD)
    // because K+(i²-j²)D/2 = K+k(2i-k)D/2 = (K-k²D/2) + ikD
    //
    // If Q has size 2^n we recover 2^n - deg P evaluations of P.
    assert!(d1 % 6 == 0);
    // Compute baby steps.
    let bsteps = {
        let mut v = Vec::with_capacity(d1 as usize / 2);
        let mut b = 1;
        let g2 = zn.mul(&g, &g);
        let mut gaps = vec![g2];
        let mut bg = g.clone();
        let mut bexp = 1;
        v.push(bg.clone());
        while b < d1 {
            b += 2;
            if b % 3 == 0 || Integer::gcd(&b, &d1) != 1 {
                continue;
            }
            let gap = b - bexp;
            while gaps.len() as u64 <= gap / 2 {
                gaps.push(zn.mul(gaps[gaps.len() - 1], g2));
            }
            bg = zn.mul(&bg, &gaps[gap as usize / 2 - 1]);
            v.push(bg.clone());
            bexp = b;
        }
        debug_assert!(bg == exp_modn(zn, &g, bexp));
        v
    };
    // Compute the giant steps i²D/2 for i in 0..k
    // but also the negated (k²-i²)D/2
    let d2 = d2 as usize;
    let (gsteps, negsteps) = {
        let mut gaps = Vec::with_capacity(d2);
        let mut steps = Vec::with_capacity(d2);
        let mut gexp = zn.one();
        let mut dg = exp_modn(zn, &g, d1 / 2);
        let ddg = zn.mul(&dg, &dg);
        for _ in 0..d2 {
            // gexp = exp(g, i² D/2)
            steps.push(gexp);
            gexp = zn.mul(&gexp, &dg);
            // dg = exp(g, (2k+1)D/2)
            gaps.push(dg);
            dg = zn.mul(&dg, &ddg)
        }
        debug_assert!(gexp == exp_modn(zn, &g, (d2 as u64 * d2 as u64 * d1) / 2));
        // sum(gaps) = d2² D/2
        assert!(gaps.len() == d2);
        // Apply gaps in reverse order to build (k²-i²)D/2
        let mut negsteps = Vec::with_capacity(d2);
        gexp = zn.one();
        for i in 0..d2 {
            gexp = zn.mul(&gexp, &gaps[d2 - 1 - i]);
            negsteps.push(gexp);
        }
        (steps, negsteps)
    };
    assert!(d2 & (d2 - 1) == 0);
    debug_assert!(gsteps.len() == d2);
    debug_assert!(negsteps.len() == d2);

    // Compute the convolution. Only coefficients [deg P .. d2]
    // d2/2 is always above FFT_THRESHOLD.
    let znx = PolyRing::new(zn, d2 / 2);
    let mzp = znx.mzp().unwrap();
    let mut p = Poly::from_roots(&znx, &bsteps).c;
    for i in 0..p.len() {
        p[i] = zn.mul(&p[i], &negsteps[i]);
    }
    let q = gsteps;
    let mut z = vec![MInt::default(); d2];
    convolve_modn_ntt(mzp, d2, &p, &q, &mut z, 0);
    let vals = &mut z[p.len() - 2..];
    // Compute cumulative product
    // The first interesting value is z[deg p = len(p)-1].
    vals[0] = zn.one();
    for i in 1..vals.len() {
        vals[i] = zn.mul(&vals[i - 1], &vals[i]);
    }
    gcd_factors(&zn.n, vals)
}

fn stage2_params(b2: f64) -> (f64, u64, u64) {
    *STAGE2_PARAMS
        .iter()
        .min_by(|x, y| (x.0 - b2).abs().total_cmp(&(y.0 - b2).abs()))
        .unwrap()
}

const STAGE2_PARAMS: &[(f64, u64, u64)] = &[
    // B2, d1, d2
    // Cost assumption: a product tree costs about 8 times a convolution.
    // Parameters are chosen so that d2 is about 8x or 16x φ(d1)
    // B2 = d1*(d2 - φ(d1))
    (30e3, 120, 256),
    (60e3, 120, 512),
    (100e3, 240, 512),
    (200e3, 240, 1024),
    (450e3, 510, 1024),
    (980e3, 510, 2048),
    (1.9e6, 1050, 2048),
    (4e6, 1050, 4096),
    (8.3e6, 2310, 4096),
    (18e6, 2310, 8192),
    (33e6, 4620, 8192),
    (71e6, 4620, 16384),
    (133e6, 9240, 16384),
    (285e6, 9240, 32768),
    (550e6, 19110, 32768),
    (1.2e9, 19110, 65536),
    (2.3e9, 39270, 65536),
    (4.8e9, 39270, 131072),
    (7.9e9, 79170, 131072),
    (18e9, 79170, 262144),
    (37e9, 159390, 262144),
    (78e9, 159390, 524288),
    (150e9, 330330, 524288), // φ=63360
    (320e9, 330330, 1048576),
    (640e9, 690690, 1048576), // φ=126720
    (1360e9, 690690, 2097152),
    (2500e9, 1381380, 2097152), // φ=253440
    (5400e9, 1381380, 4194304),
    (10.5e12, 2852850, 4194304), // φ=518400
    (22.5e12, 2852850, 8388608),
];

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
        let mut seed = 1234567_u32;
        let mut samples = vec![];
        let mut primes = vec![];
        while samples.len() < 300 {
            seed = seed.wrapping_mul(123456789);
            let p = seed % (1 << bits);
            if !fbase::certainly_composite(p as u64) {
                primes.push(p);
            }
            if primes.len() == 2 {
                let p = primes[0];
                let q = primes[1];
                samples.push(p as u64 * q as u64);
                primes.clear();
            }
        }

        let size = 2 * bits;
        let total = samples.len();
        for budget in [500, 2000, 5000, 10000, 20000, 35000, 65000] {
            let mut ok = 0;
            let start = std::time::Instant::now();
            for &n in &samples {
                if let Some((x, y)) = pb.factor(n, budget) {
                    assert_eq!(x * y, n);
                    ok += 1;
                }
            }
            let elapsed = start.elapsed().as_secs_f64() * 1000.;
            eprintln!(
                "{size} bits, budget={budget} factored {ok}/{total} semiprimes in {elapsed:.2}ms"
            );
        }
    }
}

#[test]
fn test_pm1_uint() {
    use std::str::FromStr;

    let v = Verbosity::Silent;
    // A 128-bit prime such that p-1 is smooth:
    // p-1 = 2 * 5 * 29 * 89 * 211 * 433 * 823 * 1669 * 4013 * 7717 * 416873
    let p128 = Uint::from_str("41815371135748981224332082131").unwrap();
    // A 256-bit strong prime
    let p256 = Uint::from_str(
        "92504863121296400653652753711376140294298584431452956354291724864471735145079",
    )
    .unwrap();
    let n = p128 * p256;
    let Some((p, q)) = pm1_impl(&n, 30000, 1e6, v)
        else { panic!("failed Pollard P-1") };
    assert_eq!(p, vec![p128]);
    assert_eq!(q, p256);

    // p-1 = 2*59*35509
    // Checks that blocks of primes are properly iterated (35509 is in the first block).
    let n = Uint::from_digit(2 * 59 * 35509 + 1) * p256;
    let Some((ps, q)) = pm1_impl(&n, 200, 40e3, v)
        else { panic!("failed Pollard P-1") };
    assert_eq!(ps, vec![Uint::from_digit(2 * 59 * 35509 + 1)]);
    assert_eq!(q, p256);

    // 2^3 * 3 * 7 * 11 * 31 * 131 * 1109 * 1699 * 8317 * 5984903
    let p = Uint::from_str("703855808397033138741049").unwrap();
    let n = p * p256;
    let Some((ps, q)) = pm1_impl(&n, 30000, 18e6, v)
        else { panic!("failed Pollard P-1") };
    assert_eq!(ps, vec![p]);
    assert_eq!(q, p256);

    // p-1 = 2^2 * 3^2 * 11 * 41 * 71 * 379 * 613 * 1051 * 13195979
    // 13195979 % 9240 = 1259 is small, positive
    let p = Uint::from_str("3714337881767344119949").unwrap();
    let n = p * p256;
    let Some((ps, q)) = pm1_impl(&n, 2000, 19e6, v)
        else { panic!("failed Pollard P-1") };
    assert_eq!(ps, vec![p]);
    assert_eq!(q, p256);

    // All factors are smooth:
    // 271750259454572315341 = 91033 * 6472621 * 461201737
    let n = Uint::from_str("271750259454572315341").unwrap();
    let Some((ps, q)) = pm1_impl(&n, 16384, 40e3, v)
        else { panic!("failed Pollard P-1") };
    assert!(q == Uint::ONE && ps.len() == 3);
    assert_eq!(ps[0] * ps[1] * ps[2], n);

    // Has factor p=25362180101
    // p-1 = 2*2*5*5*53*53*90289 where 53*53 > 1024
    // 90289 == 240 * 377 - 191
    let n = Uint::from_str("11006826704494670034453871933878113282264711716157472884058231906746817631612072806760744802006592942296873294117168841323692902345209717573").unwrap();
    let Some((p, q)) = pm1_impl(&n, 16384, 100e3, v)
        else { panic!("failed Pollard P-1") };
    assert_eq!(p, vec![Uint::from_digit(25_362_180_101)]);
    assert_eq!(p[0] * q, n);

    // Has factor p=285355513 such that p-1 = 8*9*13*304867
    // 304867 % 6 = 1
    let n = Uint::from_str("71269410553363907234778342302207120711196110927").unwrap();
    let Some((p, q)) = pm1_impl(&n, 20, 450e3, v)
        else { panic!("failed Pollard P-1") };
    assert_eq!(p, vec![Uint::from_digit(285_355_513)]);
    assert_eq!(p[0] * q, n);
}

#[test]
fn test_exp_modn() {
    use bnum::cast::CastFrom;
    use std::str::FromStr;
    // Test Fermat theorem.
    let p63 = Uint::from_digit(4347206819778221123);
    let p480 = Uint::from_str("801643889160962459503567529599420993581193510766215918385643775834136080985009029500140562854896402036056836567241446409601881132259487327233447").unwrap();

    let two = Uint::from_digit(2);
    let zn = ZmodN::new(p63);
    assert_eq!(
        zn.one(),
        exp_modn(&zn, &zn.from_int(two), p63.digits()[0] - 1)
    );

    let zn = ZmodN::new(p480);
    assert_eq!(
        zn.one(),
        exp_modn_large(&zn, &zn.from_int(two), &U1024::cast_from(p480 - Uint::ONE))
    );
}
