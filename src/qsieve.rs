// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! The classical quadratic sieve (using polynomial (x + Nsqrt)^2 - N).
//!
//! Bibliography:
//! J. Gerver, Factoring Large Numbers with a Quadratic Sieve
//! https://www.jstor.org/stable/2007781

use std::sync::RwLock;

use crate::arith::{isqrt, Num};
use crate::fbase::{self, Prime};
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
    let primes = fbase::primes(2 * fb);
    eprintln!("Smoothness bound {}", primes.last().unwrap());
    let primes: Vec<Prime> = fbase::prepare_factor_base(&n, &primes);
    let primes = &primes[..];
    eprintln!("All primes {}", primes.len());
    // Prepare factor base
    let smallprimes: Vec<u64> = primes.iter().map(|f| f.p).take(10).collect();
    eprintln!("Factor base size {} ({:?})", primes.len(), smallprimes);

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

    let mut target = primes.len() * 8 / 10;

    let mut maxlarge: u64 =
        primes.last().unwrap().p * prefs.large_factor.unwrap_or(large_prime_factor(&n));
    if use_double {
        // When using double large primes, use smaller larges.
        maxlarge /= 2
    }
    let qs = SieveQS::new(n, primes, maxlarge, use_double);
    eprintln!("Max large prime {}", qs.maxlarge);
    // Construct 2 initial states, forward and backwards.
    let (mut s_fwd, mut s_bck) = qs.init(None);
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
        if rels.len() > primes.len() + 16 {
            // Too many relations! May happen for very small inputs.
            rels.complete.truncate(primes.len() + 16);
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
            (s_fwd, s_bck) = qs.init(Some((s_fwd, s_bck)));
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
    eprintln!(
        "Sieved {:.1}M found {} smooths (cofactors: {} combined, {} pending)",
        (sieved as f64) / ((1 << 20) as f64),
        rels.len(),
        rels.n_combined,
        rels.n_partials,
    );
    rels.into_inner()
}

pub struct SieveQS<'a> {
    n: Uint,
    nsqrt: Uint,
    // Precomputed nsqrt_mods modulo the factor base.
    nsqrt_mods: Vec<u32>,
    primes: &'a [Prime],

    maxlarge: u64,
    use_double: bool,
    rels: RwLock<RelationSet>,
}

impl<'a> SieveQS<'a> {
    pub fn new(n: Uint, primes: &'a [Prime], maxlarge: u64, use_double: bool) -> Self {
        let nsqrt = isqrt(n);
        let nsqrt_mods: Vec<u32> = primes
            .iter()
            .map(|p| p.div.mod_uint(&nsqrt) as u32)
            .collect();
        // Prepare sieve
        SieveQS {
            n,
            nsqrt,
            primes,
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

    pub fn init(&self, sieves: Option<(Sieve<'a>, Sieve<'a>)>) -> (Sieve<'a>, Sieve<'a>) {
        let offset = if let Some((s1, _)) = sieves.as_ref() {
            s1.offset
        } else {
            0
        };
        let f1 = |pidx, p| {
            // Return r such that nsqrt + r is a root of n.
            let p: &'a Prime = p;
            let base = self.nsqrt_mods[pidx] as u64;
            let off: u64 = p.div.modi64(offset);
            let mut s1 = p.r + 2 * p.p - off - base;
            let mut s2 = 3 * p.p - p.r - off - base;
            while s1 >= p.p {
                s1 -= p.p
            }
            while s2 >= p.p {
                s2 -= p.p
            }
            SievePrime {
                p: p.p as u32,
                offsets: [Some(s1 as u32), Some(s2 as u32)],
            }
        };
        let f2 = |pidx, p| {
            // Return r such that nsqrt - (r + 1) is a root of n.
            let p: &'a Prime = p;
            let base = self.nsqrt_mods[pidx] as u64;
            let off: u64 = p.div.modi64(offset);
            let mut s1 = 2 * p.p + base - p.r - off - 1;
            let mut s2 = p.p + base + p.r - off - 1;
            while s1 >= p.p {
                s1 -= p.p
            }
            while s2 >= p.p {
                s2 -= p.p
            }
            SievePrime {
                p: p.p as u32,
                offsets: [Some(s1 as u32), Some(s2 as u32)],
            }
        };

        if let Some((mut s1, mut s2)) = sieves {
            s1.rehash(f1);
            s2.rehash(f2);
            (s1, s2)
        } else {
            let s1 = Sieve::new(0, self.nblocks(), self.primes, f1);
            let s2 = Sieve::new(0, self.nblocks(), self.primes, f2);
            (s1, s2)
        }
    }
}

fn sieve_block(s: &SieveQS, st: &mut Sieve, backward: bool) {
    st.sieve_block();

    let len: usize = BLOCK_SIZE;
    let offset = st.offset;
    let maxprime = st.primes.last().unwrap().p;
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
        for p in facs {
            let mut exp = 0;
            loop {
                let (q, r) = p.div.divmod_uint(&cofactor);
                if r == 0 {
                    cofactor = q;
                    exp += 1;
                } else {
                    break;
                }
            }
            factors.push((p.p as i64, exp));
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
            pp: false,
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
