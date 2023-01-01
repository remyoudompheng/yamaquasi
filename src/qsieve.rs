// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! The classical quadratic sieve (using polynomial (x + Nsqrt)^2 - N).
//!
//! Bibliography:
//! J. Gerver, Factoring Large Numbers with a Quadratic Sieve
//! https://www.jstor.org/stable/2007781

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::RwLock;

use crate::arith::{isqrt, Num};
use crate::fbase::{self, FBase, Prime};
use crate::params::{self, large_prime_factor, BLOCK_SIZE};
use crate::relations::{self, Relation, RelationSet};
use crate::sieve::{Sieve, SievePrime};
use crate::{Int, Uint};

pub fn qsieve(
    n: Uint,
    prefs: &crate::Preferences,
    tpool: Option<&rayon::ThreadPool>,
) -> Vec<Relation> {
    let use_double = prefs.use_double.unwrap_or(n.bits() > 200);

    // Choose factor base among twice the number of needed primes
    // (n will be a quadratic residue for only half of them)
    //
    // Compared to MPQS, classical quadratic sieve uses a single huge interval
    // so resulting numbers (2M sqrt(n)) can be larger by 10-20 bits.
    // Choose factor base size as if n was larger (by a factor O(M^2)).
    let shift = if n.bits() > 100 {
        // 160-bit input => 20 bit penalty (interval size 300M-500M)
        // 180-bit input => 22 bit penalty (interval size 1G-3G)
        // 200-bit input => 25 bit penalty (interval size 3G-10G)
        n.bits() / 8
    } else {
        0
    };
    let fb = prefs
        .fb_size
        .unwrap_or(params::factor_base_size(&(n << shift)));
    // When using double large primes, use a smaller factor base.
    let fb = if use_double { fb / 2 } else { fb };
    let fbase = FBase::new(n, fb);
    eprintln!("Smoothness bound {}", fbase.bound());
    eprintln!("Factor base size {} ({:?})", fbase.len(), fbase.smalls());

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

    let mut target = fbase.len() * 8 / 10;

    let maxlarge: u64 = fbase.bound() as u64 * prefs.large_factor.unwrap_or(large_prime_factor(&n));
    let qs = SieveQS::new(n, &fbase, maxlarge, use_double);
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

    // These counters are actually not accessed concurrently.
    let fwd_offset = AtomicI64::new(0i64);
    let bck_offset = AtomicI64::new(0i64);
    // Construct 2 initial states, forward and backwards.
    let pfunc1 = |pidx| qs.prepare_prime_fwd(pidx, fwd_offset.load(Ordering::SeqCst));
    let pfunc2 = |pidx| qs.prepare_prime_bck(pidx, bck_offset.load(Ordering::SeqCst));
    let mut s_fwd = Sieve::new(0, qs.nblocks(), qs.fbase, &pfunc1);
    let mut s_bck = Sieve::new(0, qs.nblocks(), qs.fbase, &pfunc2);
    loop {
        if let Some(pool) = tpool {
            pool.install(|| {
                rayon::join(
                    || sieve_block(&qs, &mut s_fwd, false),
                    || sieve_block(&qs, &mut s_bck, true),
                )
            });
        } else {
            sieve_block(&qs, &mut s_fwd, false);
            sieve_block(&qs, &mut s_bck, true);
        }

        // Next block
        s_fwd.next_block();
        s_bck.next_block();
        if s_fwd.blk_no == qs.nblocks() {
            assert_eq!(s_bck.blk_no, qs.nblocks());
            fwd_offset.store(s_fwd.offset, Ordering::SeqCst);
            bck_offset.store(s_bck.offset, Ordering::SeqCst);
            s_fwd.rehash();
            s_bck.rehash();
        }

        let sieved = s_fwd.offset + s_bck.offset;
        let do_print = if sieved > (5 << 30) {
            sieved % (500 << 20) == 0
        } else if sieved > (500 << 20) {
            sieved % (50 << 20) == 0
        } else {
            sieved % (10 << 20) == 0
        };
        let rels = qs.rels.read().unwrap();
        if do_print {
            rels.log_progress(format!("Sieved {}M", sieved >> 20,));
        }
        // For small n the sieve must stop quickly:
        // test whether we already have enough relations.
        if n.bits() < 64 || rels.len() >= target {
            let rels = qs.rels.read().unwrap();
            let gap = rels.gap();
            if gap == 0 {
                eprintln!("Found enough relations");
                break;
            } else {
                eprintln!("Need {} additional relations", gap);
                target = rels.len() + gap;
            }
        }
    }
    let sieved = s_fwd.offset + s_bck.offset;
    let mut rels = qs.rels.into_inner().unwrap();
    rels.log_progress(format!(
        "Sieved {:.1}M",
        (sieved as f64) / ((1 << 20) as f64)
    ));
    if rels.len() > fbase.len() + relations::MIN_KERNEL_SIZE {
        rels.truncate(fbase.len() + relations::MIN_KERNEL_SIZE)
    }
    rels.into_inner()
}

pub struct SieveQS<'a> {
    n: Uint,
    nsqrt: Uint,
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
        let nsqrt_mods: Vec<u32> = fbase
            .divs
            .iter()
            .map(|div| div.mod_uint(&nsqrt) as u32)
            .collect();
        // Prepare sieve
        SieveQS {
            n,
            nsqrt,
            nsqrt_mods,
            fbase,
            only_odds,
            maxlarge,
            use_double,
            rels: RwLock::new(RelationSet::new(n, maxlarge)),
        }
    }

    // Rehash large primes every xx blocks
    fn nblocks(&self) -> usize {
        // mpqs_interval_size / BLOCK_SIZE
        let sz = self.n.bits();
        match sz {
            0..=119 => 8,                // doesn't matter
            120..=330 => 8 << (sz / 80), // 8..128
            _ => 128,
        }
    }

    fn prepare_prime_fwd(&self, pidx: usize, offset: i64) -> SievePrime {
        // Return r such that nsqrt + r is a root of n.
        let Prime { p, r, div } = self.fbase.prime(pidx);
        let base = self.nsqrt_mods[pidx] as u64;
        let off: u64 = div.modi64(offset);
        let mut s1 = 2 * p + r - base;
        let mut s2 = 2 * p - r - base;
        // If we are in "only odds" mode, the polynomial is
        // not (R + x)^2 - n but (R + 2x)^2 - n so the roots
        // must be divided by 2.
        if self.only_odds {
            if p == 2 {
                if base == self.n.low_u64() % 2 {
                    return SievePrime {
                        p: 2,
                        offsets: [Some(0), Some(1)],
                    };
                } else {
                    return SievePrime {
                        p: 2,
                        offsets: [None, None],
                    };
                }
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
        s1 += p - off;
        s2 += p - off;
        while s1 >= p {
            s1 -= p
        }
        while s2 >= p {
            s2 -= p
        }
        SievePrime {
            p: p as u32,
            offsets: if s1 != s2 {
                [Some(s1 as u32), Some(s2 as u32)]
            } else {
                [Some(s1 as u32), None]
            },
        }
    }

    fn prepare_prime_bck(&self, pidx: usize, offset: i64) -> SievePrime {
        let Prime { p, r, div } = self.fbase.prime(pidx);
        // Return r such that nsqrt - 2(r + 1) is a root of n.
        let base = self.nsqrt_mods[pidx] as u64;
        let off: u64 = div.modi64(offset);
        let mut s1 = 2 * p + base - r;
        let mut s2 = 2 * p + base + r;
        // If we are in "only odds" mode, the polynomial is
        // not (R + x)^2 - n but (R + 2x)^2 - n so the roots
        // must be divided by 2.
        if self.only_odds {
            if p == 2 {
                if base == self.n.low_u64() % 2 {
                    return SievePrime {
                        p: 2,
                        offsets: [Some(0), Some(1)],
                    };
                } else {
                    return SievePrime {
                        p: 2,
                        offsets: [None, None],
                    };
                }
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
        s1 += p - off - 1;
        s2 += p - off - 1;
        while s1 >= p {
            s1 -= p
        }
        while s2 >= p {
            s2 -= p
        }
        SievePrime {
            p: p as u32,
            offsets: if s1 != s2 {
                [Some(s1 as u32), Some(s2 as u32)]
            } else {
                [Some(s1 as u32), None]
            },
        }
    }

    /// An unsafe constructor for the sieve object, only for tests.
    pub fn init_sieve_for_test(&'a self) -> Sieve<'a> {
        let pfunc = |pidx| self.prepare_prime_fwd(pidx, 0);
        let unsafe_pfunc = Box::leak(Box::new(pfunc));
        Sieve::new(0, self.nblocks(), self.fbase, unsafe_pfunc)
    }
}

fn sieve_block(s: &SieveQS, st: &mut Sieve, backward: bool) {
    st.sieve_block();

    let len: usize = BLOCK_SIZE;
    let offset = st.offset;
    let maxprime = s.fbase.bound() as u64;
    let maxlarge = s.maxlarge;
    let max_cofactor = if s.use_double {
        // We don't want double large prime to reach maxlarge^2
        // See siqs.rs
        maxlarge * maxprime * 2
    } else {
        maxlarge
    };
    let magnitude = u64::BITS
        - u64::leading_zeros(std::cmp::max(st.offset.abs() as u64, len as u64))
        + (if s.only_odds { 1 } else { 0 });
    let target = s.n.bits() / 2 + magnitude - max_cofactor.bits();
    assert!(target < 256);
    let n = &s.n;
    let (idxs, facss) = st.smooths(target as u8, None);
    let maybe_two: i64 = if s.only_odds { 2 } else { 1 };
    for (i, facs) in idxs.into_iter().zip(facss) {
        let x = if !backward {
            Int::from_bits(s.nsqrt) + Int::from(maybe_two * (offset as i64 + i as i64))
        } else {
            Int::from_bits(s.nsqrt) - Int::from(maybe_two * (offset as i64 + i as i64 + 1))
        };
        let candidate: Int = x * x - Int::from_bits(*n);
        let Some(((p, q), factors)) = fbase::cofactor(
            s.fbase, &candidate, &facs,
            maxlarge, max_cofactor, None)
            else { continue };
        let pq = if q > 1 { Some((p, q)) } else { None };
        let cofactor = p * q;
        //println!("i={} smooth {} cofactor {}", i, cabs, cofactor);
        let rel = Relation {
            x: x.abs().to_bits(),
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
