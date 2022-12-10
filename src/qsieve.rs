// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! The classical quadratic sieve (using polynomial (x + Nsqrt)^2 - N).
//!
//! Bibliography:
//! J. Gerver, Factoring Large Numbers with a Quadratic Sieve
//! https://www.jstor.org/stable/2007781

use std::collections::HashMap;

use crate::arith::{isqrt, Num};
use crate::fbase::Prime;
use crate::params::{large_prime_factor, BLOCK_SIZE};
use crate::relations::{combine_large_relation, relation_gap, Relation};
use crate::sieve;
use crate::{Int, Uint};

pub fn qsieve(n: Uint, primes: &[Prime]) -> Vec<Relation> {
    // Prepare sieve
    let nsqrt = isqrt(n);
    let maxlarge: u64 = primes.last().unwrap().p * large_prime_factor(&n);
    eprintln!("Max cofactor {}", maxlarge);

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

    let mut relations = vec![];
    let mut target = primes.len() * 8 / 10;
    let mut larges = HashMap::<u64, Relation>::new();
    let mut extras = 0;
    let sieve = SieveQS { n, nsqrt };
    // Construct 2 initial states, forward and backwards.
    let (mut s_fwd, mut s_bck) = init_sieves(primes, nsqrt);
    loop {
        let (mut found, foundlarge) = sieve_block(&sieve, &mut s_fwd, false);
        if found.len() > primes.len() + 16 {
            // Too many relations! May happen for very small inputs.
            relations.extend_from_slice(&mut found[..primes.len() + 16]);
            let gap = relation_gap(&relations);
            if gap == 0 {
                println!("Found enough relations");
                break;
            } else {
                println!("Need {} additional relations", gap);
                target = relations.len() + gap + 16;
            }
        }
        s_fwd.next_block();
        relations.append(&mut found);
        for r in foundlarge {
            if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                relations.push(rr);
                extras += 1;
            }
        }
        let (mut found, foundlarge) = sieve_block(&sieve, &mut s_bck, true);
        s_bck.next_block();
        relations.append(&mut found);
        for r in foundlarge {
            if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                relations.push(rr);
                extras += 1;
            }
        }
        let sieved = s_fwd.offset + s_bck.offset;
        if sieved % (10 << 20) == 0 {
            println!(
                "Sieved {}M found {} smooths (cofactors: {} combined, {} pending)",
                sieved >> 20,
                relations.len(),
                extras,
                larges.len(),
            );
        }
        // For small n the sieve must stop quickly:
        // test whether we already have enough relations.
        if n.bits() < 64 || relations.len() >= target {
            let gap = relation_gap(&relations);
            if gap == 0 {
                println!("Found enough relations");
                break;
            } else {
                println!("Need {} additional relations", gap);
                target = relations.len() + gap + 10;
            }
        }
    }
    let sieved = s_fwd.offset + s_bck.offset;
    println!(
        "Sieved {:.1}M found {} smooths (cofactors: {} combined, {} pending)",
        (sieved as f64) / ((1 << 20) as f64),
        relations.len(),
        extras,
        larges.len(),
    );
    relations
}

pub fn init_sieves(fb: &[Prime], nsqrt: Uint) -> (sieve::Sieve, sieve::Sieve) {
    let mut st_primes = vec![];
    let mut st_logs = vec![];
    let mut st_hi = vec![];
    let mut st_lo = vec![];
    let mut st_hi2 = vec![];
    let mut st_lo2 = vec![];
    for p in fb.iter() {
        assert_eq!(p.p >> 24, 0);
        let rp = p.div.mod_uint(&nsqrt);
        let logp = (32 - u32::leading_zeros(p.p as u32)) as u8;
        st_primes.push(p);
        st_logs.push(logp);
        let s1 = p.div.divmod64(p.r + p.p - rp).1;
        let s2 = p.div.divmod64(p.p - p.r + p.p - rp).1;
        st_hi.push((s1 / BLOCK_SIZE as u64) as u8);
        st_lo.push((s1 % BLOCK_SIZE as u64) as u16);
        if p.r != 0 {
            st_primes.push(p);
            st_logs.push(logp);
            st_hi.push((s2 / BLOCK_SIZE as u64) as u8);
            st_lo.push((s2 % BLOCK_SIZE as u64) as u16);
        }
        // Backward sieve
        let rp = if rp == 0 { p.p - 1 } else { rp - 1 };
        let s1 = p.div.divmod64(p.r + rp).1;
        let s2 = p.div.divmod64(p.p - p.r + rp).1;
        st_hi2.push((s1 / BLOCK_SIZE as u64) as u8);
        st_lo2.push((s1 % BLOCK_SIZE as u64) as u16);
        if p.r != 0 {
            // 2 roots
            st_hi2.push((s2 / BLOCK_SIZE as u64) as u8);
            st_lo2.push((s2 % BLOCK_SIZE as u64) as u16);
        }
    }

    let idx15 = st_primes
        .iter()
        .position(|&p| p.p > BLOCK_SIZE as u64)
        .unwrap_or(st_primes.len());

    let s1 = sieve::Sieve {
        offset: 0,
        idx15,
        primes: st_primes.clone(),
        logs: st_logs.clone(),
        hi: st_hi,
        lo: st_lo,
        starts: vec![],
        blk: [0u8; BLOCK_SIZE],
    };
    let s2 = sieve::Sieve {
        offset: 0,
        idx15,
        primes: st_primes,
        logs: st_logs,
        hi: st_hi2,
        lo: st_lo2,
        starts: vec![],
        blk: [0u8; BLOCK_SIZE],
    };
    (s1, s2)
}

struct SieveQS {
    n: Uint,
    nsqrt: Uint,
}

fn sieve_block(
    s: &SieveQS,
    st: &mut sieve::Sieve,
    backward: bool,
) -> (Vec<Relation>, Vec<Relation>) {
    st.sieve_block();

    let len: usize = BLOCK_SIZE;
    let offset = st.offset;
    let maxprime = st.primes.last().unwrap().p;
    let maxlarge = maxprime * large_prime_factor(&s.n);
    let mut result = vec![];
    let mut extras = vec![];
    let magnitude = u64::BITS - u64::leading_zeros(std::cmp::max(offset, len as u64));
    let target = s.n.bits() / 2 + magnitude - maxlarge.bits();
    assert!(target < 256);
    let n = &s.n;
    for i in st.smooths(target as u8) {
        let x = if !backward {
            Int::from_bits(s.nsqrt) + Int::from(offset as i64 + i as i64)
        } else {
            Int::from_bits(s.nsqrt) - Int::from(offset as i64 + i as i64 + 1)
        };
        let candidate: Int = x * x - Int::from_bits(*n);
        let (cofactor, factors) = st.cofactor(i, &candidate);
        let Some(cofactor) = cofactor.to_u64() else { continue };
        //println!("i={} smooth {} cofactor {}", i, cabs, cofactor);
        let rel = Relation {
            x: x.abs().to_bits(),
            cofactor,
            factors,
        };
        debug_assert!(
            rel.verify(&s.n),
            "INTERNAL ERROR: failed relation check {:?}",
            rel
        );
        if cofactor == 1 {
            result.push(rel)
        } else {
            extras.push(rel)
        }
    }
    (result, extras)
}
