// Copyright 2022, 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! The classical quadratic sieve (using polynomial (x + Nsqrt)^2 - N).
//!
//! Due to its simplicity it can compete with MPQS/SIQS (even faster
//! than ECM actually) to factor integers between 64 and 100 bits.
//! However, since it uses a single interval, it cannot find a very
//! large ratio of smooth numbers.
//!
//! Bibliography:
//! J. Gerver, Factoring Large Numbers with a Quadratic Sieve
//! <https://www.jstor.org/stable/2007781>

use std::cmp::min;
use std::sync::RwLock;

use bnum::cast::CastFrom;

use crate::arith::{isqrt, Num, I256};
use crate::fbase::{self, FBase, Prime};
use crate::params;
use crate::relations::{self, Relation, RelationSet};
use crate::sieve::{Sieve, BLOCK_SIZE};
use crate::{Int, Preferences, Uint, Verbosity};

pub fn qsieve(
    n: Uint,
    k: u32,
    prefs: &Preferences,
    tpool: Option<&rayon::ThreadPool>,
) -> Vec<Uint> {
    let (norig, n) = (n, n * Uint::from(k));
    // We cannot use quadratic sieve for numbers above 400 bits
    // (or at least it is extremely unreasonable and cost days of CPU).
    // This is so that sqrt(n) * interval size always fits in 256 bits.
    if n.bits() > 400 {
        if prefs.verbose(Verbosity::Info) {
            eprintln!("Number {n} too large for classical quadratic sieve!");
        }
        return vec![];
    }
    let use_double = prefs.use_double.unwrap_or(n.bits() > 200);

    // Choose factor base among twice the number of needed primes
    // (n will be a quadratic residue for only half of them)
    //
    // Compared to MPQS, classical quadratic sieve uses a single huge interval
    // so resulting numbers (2M sqrt(n)) can be larger by 10-20 bits.
    // Choose factor base size as if n was larger (by a factor O(M^2)).
    //
    // 160-bit input => 20 bit penalty (interval size 300M-500M)
    // 180-bit input => 22 bit penalty (interval size 1G-3G)
    // 200-bit input => 25 bit penalty (interval size 3G-10G)
    let shift = n.bits() / 8;
    let fb = prefs
        .fb_size
        .unwrap_or(params::factor_base_size(&(n << shift)) / (if use_double { 2 } else { 1 }));
    // When using double large primes, use a smaller factor base.
    let fbase = FBase::new(n, fb);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Smoothness bound {}", fbase.bound());
        eprintln!("Factor base size {} ({:?})", fbase.len(), fbase.smalls());
    }
    // Naïve quadratic sieve with polynomial x²-n (x=-M..M)
    // Max value is X = sqrt(n) * M
    // Smooth bound Y = exp(1/2 sqrt(log X log log X))
    // If input number is 512 bits, generated values are less then 300 bits

    // There are at most 10 prime factors => 8-bit precision is enough
    // Blocks optimized to fit 512kB cache memory per core
    // Work unit is 16 blocks

    // We run 2 sieves:
    // Forward: polynomial (R+x)^2 - N where r = isqrt(N)
    // => roots are sqrt(N) - R
    // Backward: polynomial (R-1-x)^2 - N where r = isqrt(N)
    // => roots are sqrt(N) + R-1
    //
    // The sieve can be split into blocks: this is equivalent to using
    // multiple polynomials (R+x+kB)^2 - N where B is the block size.

    let mut target = fbase.len() * 8 / 10;

    let maxlarge: u64 = fbase.bound() as u64 * prefs.large_factor.unwrap_or(large_prime_factor(&n));
    let qs = SieveQS::new(n, &fbase, maxlarge, use_double);
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Max large prime {}", qs.maxlarge);
        if use_double {
            eprintln!(
                "Max double large prime {}",
                maxlarge * fbase.bound() as u64 * 2
            );
        }
        if qs.only_odds {
            eprintln!("N is 1 mod 4, only odd numbers will be sieved");
        }
    }

    // Precompute the amount by which roots must be offset after each large block.
    let large_block_size = qs.nblocks() * BLOCK_SIZE;
    let large_blksz_modp: Vec<u32> = (0..fbase.len())
        .map(|pidx| {
            let div = fbase.div(pidx);
            div.div31.modi32(large_block_size as i32)
        })
        .collect();
    // A vectorization-friendly way to shift roots for next block.
    // On shifted interval x+B the roots are r-B mod p
    let primes = &fbase.primes[..];
    let next_lgblock = |roots1: &mut [u32], roots2: &mut [u32]| {
        assert_eq!(roots1.len(), large_blksz_modp.len());
        assert_eq!(roots2.len(), large_blksz_modp.len());
        assert_eq!(roots2.len(), primes.len());
        for i in 0..roots1.len() {
            let o = large_blksz_modp[i];
            let p = primes[i];
            let (r1, r2) = (roots1[i].wrapping_sub(o), roots2[i].wrapping_sub(o));
            roots1[i] = min(r1, r1.wrapping_add(p));
            roots2[i] = min(r2, r2.wrapping_add(p));
        }
    };

    // These counters are actually not accessed concurrently.
    // Construct 2 initial states, forward and backwards.
    let mut roots_fwd1 = vec![0; fbase.len()];
    let mut roots_fwd2 = vec![0; fbase.len()];
    let mut roots_bck1 = vec![0; fbase.len()];
    let mut roots_bck2 = vec![0; fbase.len()];
    for pidx in 0..fbase.len() {
        let (f1, f2) = qs.prepare_prime_fwd(pidx);
        roots_fwd1[pidx] = f1;
        roots_fwd2[pidx] = f2;
        let (b1, b2) = qs.prepare_prime_bck(pidx);
        roots_bck1[pidx] = b1;
        roots_bck2[pidx] = b2;
    }
    // We need to both mutate the roots array and use it as read-only
    // argument to functions, hence the clumsy syntax.
    let mut s_fwd = Sieve::new(
        0,
        qs.nblocks(),
        qs.fbase,
        [&roots_fwd1[..], &roots_fwd2[..]],
        None,
    );
    let mut s_bck = Sieve::new(
        0,
        qs.nblocks(),
        qs.fbase,
        [&roots_bck1[..], &roots_bck2[..]],
        None,
    );
    for large_blk_idx in 1.. {
        // The unit of work is an entire large block (blocks * BLOCK_SIZE)
        // The size of a large block should be similar to the SIQS interval size.
        // Forward sieve
        let mut do_sieve_fwd = || {
            if s_fwd.blk_no == qs.nblocks() {
                next_lgblock(&mut roots_fwd1, &mut roots_fwd2);
                s_fwd.rehash([&roots_fwd1[..], &roots_fwd2[..]]);
            }
            for _ in 0..qs.nblocks() {
                sieve_block(&qs, &mut s_fwd, [&roots_fwd1[..], &roots_fwd2[..]], false);
                s_fwd.next_block();
            }
        };
        // Backward sieve
        let mut do_sieve_bck = || {
            if s_bck.blk_no == qs.nblocks() {
                next_lgblock(&mut roots_bck1, &mut roots_bck2);
                s_bck.rehash([&roots_bck1[..], &roots_bck2[..]]);
            }
            for _ in 0..qs.nblocks() {
                sieve_block(&qs, &mut s_bck, [&roots_bck1[..], &roots_bck2[..]], true);
                s_bck.next_block();
            }
        };
        if let Some(pool) = tpool {
            pool.install(|| rayon::join(do_sieve_fwd, do_sieve_bck));
            assert_eq!(s_fwd.blk_no, s_bck.blk_no);
        } else {
            do_sieve_fwd();
            do_sieve_bck();
            assert_eq!(s_fwd.blk_no, s_bck.blk_no);
        }

        let sieved = s_fwd.offset + s_bck.offset;
        let do_print = prefs.verbose(Verbosity::Info)
            && if n.bits() > 260 {
                large_blk_idx % 1000 == 0
            } else if n.bits() > 180 {
                large_blk_idx % 500 == 0
            } else if n.bits() > 150 {
                large_blk_idx % 50 == 0
            } else {
                large_blk_idx % 10 == 0
            };
        let rels = qs.rels.read().unwrap();
        if do_print {
            rels.log_progress(format!("Sieved {}M", sieved >> 20,));
        }
        // For small n the sieve must stop quickly:
        // test whether we already have enough relations.
        if n.bits() < 64 || rels.len() >= target {
            let rels = qs.rels.read().unwrap();
            let gap = rels.gap(&fbase);
            if gap == 0 {
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Found enough relations");
                }
                break;
            } else {
                if prefs.verbose(Verbosity::Info) {
                    eprintln!("Need {} additional relations", gap);
                }
                target = rels.len() + gap;
            }
        }
    }
    let sieved = s_fwd.offset + s_bck.offset;
    let mut rels = qs.rels.into_inner().unwrap();
    if prefs.verbose(Verbosity::Info) {
        rels.log_progress(format!(
            "Sieved {:.1}M",
            (sieved as f64) / ((1 << 20) as f64)
        ));
    }
    if rels.len() > fbase.len() + relations::MIN_KERNEL_SIZE {
        rels.truncate(fbase.len() + relations::MIN_KERNEL_SIZE)
    }
    if rels.len() == 0 {
        return vec![];
    }
    relations::final_step(&norig, &fbase, &rels.into_inner(), prefs.verbosity)
}

/// Large factor multiplier for classical QS.
pub fn large_prime_factor(n: &Uint) -> u64 {
    // Allow large cofactors up to FACTOR * largest prime
    let sz = n.bits() as u64;
    match sz {
        // Small inputs have very high smoothness already.
        // Unlike its multiple polynomial variants, there is no shortage
        // of numbers to sieve even for very small inputs. So it is best
        // to avoid the large prime variation completely (to reduce sieve cost).
        0..=72 => 1,
        // However, in the case of classical QS, the interval grows very quickly
        // and if we don't enable the large prime variation, the algorithm
        // takes longer.
        // The factor should grow gently to avoid creating a huge amount
        // of cofactors when input is 72-96 bit large.
        // (most of the density is below 10B anyway).
        73.. => sz - 70,
    }
}

pub struct SieveQS<'a> {
    n: Uint,
    // The polynomial is (nsqrt + x)^2 - n = x^2 + 2 nsqrt x + (nsqrt^2 - n)
    nsqrt: I256,
    nsqrt2_minus_n: I256,
    // Precomputed nsqrt_mods modulo the factor base.
    nsqrt_mods: Vec<u32>,
    fbase: &'a FBase,
    only_odds: bool,

    maxlarge: u64,
    use_double: bool,
    rels: RwLock<RelationSet>,
}

impl<'a> SieveQS<'a> {
    pub fn new(n: Uint, fbase: &'a FBase, maxlarge: u64, use_double: bool) -> Self {
        let mut nsqrt = isqrt(n);
        let only_odds = if n.low_u64() % 8 == 1 {
            // If n == 1 mod 8, only use odd numbers.
            // It could also be done for n = 1 mod 4, but it does not statistically
            // increase the yield and results in more complicated code.
            // Make sqrt odd.
            nsqrt += Uint::from(1 - nsqrt % 2_u64);
            true
        } else {
            false
        };
        let nsqrt2_minus_n = Int::cast_from(nsqrt * nsqrt) - Int::cast_from(n);
        assert!(nsqrt.bits() < 255);
        assert!(nsqrt2_minus_n.abs().bits() < 255);
        let nsqrt_mods: Vec<u32> = fbase
            .divs
            .iter()
            .map(|div| div.mod_uint(&nsqrt) as u32)
            .collect();
        // Prepare sieve
        SieveQS {
            n,
            nsqrt: I256::cast_from(nsqrt),
            nsqrt2_minus_n: I256::cast_from(nsqrt2_minus_n),
            nsqrt_mods,
            fbase,
            only_odds,
            maxlarge,
            use_double,
            rels: RwLock::new(RelationSet::new(n, fbase.len(), maxlarge)),
        }
    }

    // Size of large sieve blocks (requiring rehash).
    // The cost of switching to the next large block is similar
    // to SIQS so we can use small blocks if it improves cache locality,
    // but there is no number-theoretical improvement doing so.
    fn nblocks(&self) -> usize {
        // Note that QS factor bases are larger than SIQS.
        let sz = self.n.bits() as usize;
        match sz {
            0..=80 => 1,
            81..=110 => 2,
            111..=120 => 3,
            // Grow number of blocks, since we know the interval
            // is becoming quickly large.
            121..=138 => sz - 118,
            // We need to sieve several million integers:
            // use a large but reasonable size.
            _ => 20,
        }
    }

    fn prepare_prime_fwd(&self, pidx: usize) -> (u32, u32) {
        // Return r such that nsqrt + r is a root of n.
        let Prime { p, r, .. } = self.fbase.prime(pidx);
        let base = self.nsqrt_mods[pidx] as u64;
        let mut s1 = 2 * p + r - base;
        let mut s2 = 2 * p - r - base;
        // If we are in "only odds" mode, the polynomial is
        // not (R + x)^2 - n but (R + 2x)^2 - n so the roots
        // must be divided by 2.
        if self.only_odds {
            if p == 2 {
                // The polynomial is x^2 + x + (R^2-n)/4
                // It has no roots if n % 8 == 5.
                // Return a superset of actual roots.
                return (0, 1);
            }
            if s1 % 2 == 0 {
                // s2 % 2 == 0 as well
                s1 /= 2;
                s2 /= 2;
            } else {
                s1 = (s1 + p) / 2;
                s2 = (s2 + p) / 2;
            }
        }
        while s1 >= p {
            s1 -= p
        }
        while s2 >= p {
            s2 -= p
        }
        (s1 as u32, s2 as u32)
    }

    fn prepare_prime_bck(&self, pidx: usize) -> (u32, u32) {
        let Prime { p, r, .. } = self.fbase.prime(pidx);
        // Return r such that nsqrt - 2(r + 1) is a root of n.
        let base = self.nsqrt_mods[pidx] as u64;
        let mut s1 = 2 * p + base - r;
        let mut s2 = 2 * p + base + r;
        // If we are in "only odds" mode, the polynomial is
        // not (R + x)^2 - n but (R + 2x)^2 - n so the roots
        // must be divided by 2.
        if self.only_odds {
            if p == 2 {
                // See above.
                return (0, 1);
            }
            if s1 % 2 == 0 {
                // s2 % 2 == 0 as well
                s1 /= 2;
                s2 /= 2;
            } else {
                s1 = (s1 + p) / 2;
                s2 = (s2 + p) / 2;
            }
        }
        s1 += p - 1;
        s2 += p - 1;
        while s1 >= p {
            s1 -= p
        }
        while s2 >= p {
            s2 -= p
        }
        (s1 as u32, s2 as u32)
    }

    /// An unsafe constructor for the sieve object, only for tests.
    pub fn init_sieve_for_test(&'a self) -> (Sieve<'a>, [Vec<u32>; 2]) {
        let fbase = &self.fbase;
        let mut roots_fwd1 = vec![0; fbase.len()];
        let mut roots_fwd2 = vec![0; fbase.len()];
        for pidx in 0..fbase.len() {
            let (f1, f2) = self.prepare_prime_fwd(pidx);
            roots_fwd1[pidx] = f1;
            roots_fwd2[pidx] = f2;
        }
        let s = Sieve::new(
            0,
            self.nblocks(),
            self.fbase,
            [&roots_fwd1, &roots_fwd2],
            None,
        );
        (s, [roots_fwd1, roots_fwd2])
    }
}

fn sieve_block(s: &SieveQS, st: &mut Sieve, roots: [&[u32]; 2], backward: bool) {
    st.sieve_block();

    let len: usize = BLOCK_SIZE;
    let offset = st.offset;
    let maxprime = s.fbase.bound() as u64;
    let maxlarge = s.maxlarge;
    let max_cofactor = if s.use_double {
        // We don't want double large prime to reach maxlarge^2
        // See siqs.rs
        maxlarge * maxprime * 2
    } else if maxlarge > maxprime {
        // Use the single large prime variation
        maxlarge
    } else {
        // Use the plain quadratic sieve (truly smooth values)
        1
    };
    let magnitude = u64::BITS
        - u64::leading_zeros(std::cmp::max(st.offset.abs() as u64, len as u64))
        + (if s.only_odds { 1 } else { 0 });
    let target = s.n.bits() / 2 + magnitude - max_cofactor.bits();
    assert!(target < 256);
    let (idxs, facss) = st.smooths(target as u8, None, roots);
    let maybe_two: i64 = if s.only_odds { 2 } else { 1 };
    for (i, facs) in idxs.into_iter().zip(facss) {
        // In classical quadratic sieve, the interval size can exceed 2^32.
        let x = if !backward {
            maybe_two * (offset + i as i64)
        } else {
            -(maybe_two * (offset + i as i64 + 1))
        };
        let xplus = s.nsqrt + I256::from(x);
        // Evaluate polynomial (x + nsqrt)^2 - n
        let candidate =
            I256::from(x as i128 * x as i128) + I256::from(2 * x) * s.nsqrt + s.nsqrt2_minus_n;
        let Some(((p, q), factors)) = fbase::cofactor(
            s.fbase, &candidate, &facs,
            maxlarge, s.use_double)
            else { continue };
        let pq = if q > 1 { Some((p, q)) } else { None };
        let cofactor = p * q;
        //println!("i={} smooth {} cofactor {}", i, cabs, cofactor);
        let rel = Relation {
            x: Uint::cast_from(xplus.abs()),
            cofactor,
            cyclelen: 1,
            factors,
        };
        debug_assert!(
            rel.verify(&s.n),
            "INTERNAL ERROR: failed relation check {:?}",
            &rel
        );
        s.rels.write().unwrap().add(rel, pq);
    }
}
