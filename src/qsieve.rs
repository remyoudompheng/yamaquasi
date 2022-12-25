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
use crate::relations::{Relation, RelationSet};
use crate::sieve::{Sieve, SievePrime};
use crate::{Int, Uint};

pub fn qsieve(
    n: Uint,
    prefs: &params::Preferences,
    tpool: Option<&rayon::ThreadPool>,
) -> Vec<Relation> {
    let use_double = prefs.use_double.unwrap_or(n.bits() > 200);

    // Choose factor base among twice the number of needed primes
    // (n will be a quadratic residue for only half of them)
    //
    // Compared to MPQS, classical quadratic sieve uses a single huge interval
    // so resulting numbers can be larger by 15-30 bits. Choose factor base size
    // as if n was larger than it really is.
    let shift = if n.bits() > 100 {
        (n.bits() - 100) / 10
    } else {
        0
    };
    let fb = prefs
        .fb_size
        .unwrap_or(params::factor_base_size(&(n << shift)));
    // When using double large primes, use a smaller factor base.
    let fb = if use_double { fb * 3 / 4 } else { fb };
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

    let mut maxlarge: u64 =
        fbase.bound() as u64 * prefs.large_factor.unwrap_or(large_prime_factor(&n));
    if use_double {
        // When using double large primes, use smaller larges.
        maxlarge /= 2
    }
    let qs = SieveQS::new(n, &fbase, maxlarge, use_double);
    eprintln!("Max large prime {}", qs.maxlarge);
    // These counters are actually not accessed concurrently.
    let fwd_offset = AtomicI64::new(0i64);
    let bck_offset = AtomicI64::new(0i64);
    // Construct 2 initial states, forward and backwards.
    let pfunc1 = |pidx| qs.prepare_prime_fwd(pidx, (&fwd_offset).load(Ordering::SeqCst));
    let pfunc2 = |pidx| qs.prepare_prime_bck(pidx, (&bck_offset).load(Ordering::SeqCst));
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
        let mut rels = qs.rels.write().unwrap();
        if rels.len() > fbase.len() + 32 {
            // Too many relations! May happen for very small inputs.
            rels.truncate(fbase.len() + 32);
            let gap = rels.gap();
            if gap == 0 {
                eprintln!("Found enough relations");
                break;
            } else {
                eprintln!("Need {} additional relations", gap);
                target = rels.len() + gap + 16;
            }
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
        if do_print {
            rels.log_progress(format!("Sieved {}M", sieved >> 20,));
        }
        // For small n the sieve must stop quickly:
        // test whether we already have enough relations.
        if n.bits() < 64 || rels.len() >= target {
            let gap = rels.gap();
            if gap == 0 {
                eprintln!("Found enough relations");
                break;
            } else {
                eprintln!("Need {} additional relations", gap);
                target = rels.len() + gap + 10;
            }
        }
    }
    let sieved = s_fwd.offset + s_bck.offset;
    let rels = qs.rels.into_inner().unwrap();
    rels.log_progress(format!(
        "Sieved {:.1}M",
        (sieved as f64) / ((1 << 20) as f64)
    ));
    rels.into_inner()
}

pub struct SieveQS<'a> {
    n: Uint,
    nsqrt: Uint,
    // Precomputed nsqrt_mods modulo the factor base.
    nsqrt_mods: Vec<u32>,
    fbase: &'a FBase,

    maxlarge: u64,
    use_double: bool,
    rels: RwLock<RelationSet>,
}

impl<'a> SieveQS<'a> {
    pub fn new(n: Uint, fbase: &'a FBase, maxlarge: u64, use_double: bool) -> Self {
        let nsqrt = isqrt(n);
        let nsqrt_mods: Vec<u32> = fbase
            .divs
            .iter()
            .map(|div| div.mod_uint(&nsqrt) as u32)
            .collect();
        // Prepare sieve
        SieveQS {
            n,
            nsqrt,
            fbase,
            nsqrt_mods,
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
        let mut s1 = r + 2 * p - off - base;
        let mut s2 = 3 * p - r - off - base;
        while s1 >= p {
            s1 -= p
        }
        while s2 >= p {
            s2 -= p
        }
        SievePrime {
            p: p as u32,
            offsets: [Some(s1 as u32), Some(s2 as u32)],
        }
    }

    fn prepare_prime_bck(&self, pidx: usize, offset: i64) -> SievePrime {
        let Prime { p, r, div } = self.fbase.prime(pidx);
        // Return r such that nsqrt - (r + 1) is a root of n.
        let base = self.nsqrt_mods[pidx] as u64;
        let off: u64 = div.modi64(offset);
        let mut s1 = 2 * p + base - r - off - 1;
        let mut s2 = p + base + r - off - 1;
        while s1 >= p {
            s1 -= p
        }
        while s2 >= p {
            s2 -= p
        }
        SievePrime {
            p: p as u32,
            offsets: [Some(s1 as u32), Some(s2 as u32)],
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
    let maxlarge = s.maxlarge;
    let max_cofactor = if s.use_double {
        maxlarge * maxlarge
    } else {
        maxlarge
    };
    let magnitude =
        u64::BITS - u64::leading_zeros(std::cmp::max(st.offset.abs() as u64, len as u64));
    let target = s.n.bits() / 2 + magnitude - max_cofactor.bits();
    assert!(target < 256);
    let n = &s.n;
    let (idxs, facss) = st.smooths(target as u8);
    for (i, facs) in idxs.into_iter().zip(facss) {
        let x = if !backward {
            Int::from_bits(s.nsqrt) + Int::from(offset as i64 + i as i64)
        } else {
            Int::from_bits(s.nsqrt) - Int::from(offset as i64 + i as i64 + 1)
        };
        let candidate: Int = x * x - Int::from_bits(*n);
        let mut factors: Vec<(i64, u64)> = vec![];
        if candidate.is_negative() {
            factors.push((-1, 1));
        }
        let cabs = candidate.abs().to_bits();
        let mut cofactor: Uint = cabs;
        for pidx in facs {
            let p = s.fbase.p(pidx);
            let div = s.fbase.div(pidx);
            let mut exp = 0;
            loop {
                let (q, r) = div.divmod_uint(&cofactor);
                if r == 0 {
                    cofactor = q;
                    exp += 1;
                } else {
                    break;
                }
            }
            factors.push((p as i64, exp));
        }
        let Some(cofactor) = cofactor.to_u64() else { continue };
        if cofactor > max_cofactor {
            continue;
        }
        let pq = fbase::try_factor64(cofactor);
        if pq.is_none() && cofactor > maxlarge {
            continue;
        }

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
