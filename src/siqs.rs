// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Self Initializing Quadratic Sieve
//!
//! Bibliography:
//! Alford, Pomerance, Implementing the self-initializing quadratic sieve
//! https://math.dartmouth.edu/~carlp/implementing.pdf
//!
//! This method accelerates polynomial switch costs by computing polynomials
//! with easy roots.
//! Assuming a factor base {p} such that N is a quadratic residue mod p has been
//! computed, numbers A = p1 * ... * pk with factors in this factor base
//! are such that N has 2^k modular square roots modulo N
//! which can be computed from the factor base data through the Chinese remainder
//! theorem.

use num_traits::One;
use std::cmp::{max, min};
use std::collections::HashMap;

use crate::arith::{self, Num};
use crate::fbase::{Prime, SievePrime};
use crate::params::{self, BLOCK_SIZE};
use crate::relations::{combine_large_relation, relation_gap, Relation};
use crate::{Int, Uint, DEBUG};

pub fn siqs(n: &Uint, primes: &[Prime]) -> Vec<Relation> {
    let mut target = primes.len() * 8 / 10;
    let mut relations = vec![];
    let mut larges = HashMap::<u64, Relation>::new();
    let mut extras = 0;
    let fb = primes.len();
    let mlog = interval_logsize(&n);
    if mlog >= 20 {
        eprintln!("Sieving interval size {}M", 1 << (mlog - 20));
    } else {
        eprintln!("Sieving interval size {}k", 1 << (mlog - 10));
    }

    // Generate all values of A now.
    let nfacs = nfactors(n) as usize;
    let factors = select_siqs_factors(primes, n, nfacs);
    let a_s = prepare_as(&factors, primes, a_value_count(n));
    let a_diff = a_s.last().unwrap().a - a_s.first().unwrap().a;
    let a_quality = (a_diff >> (a_diff.bits() - 10)).low_u64() as f64
        / (a_s.last().unwrap().a >> (a_diff.bits() - 10)).low_u64() as f64;
    let polys_per_a = 1 << (nfacs - 1);
    eprintln!(
        "Generated {} values of A with {} factors in {}..{} ({} polynomials each, spread={:.2}%)",
        a_s.len(),
        nfacs,
        factors.factors[0].p,
        factors.factors.last().unwrap().p,
        polys_per_a,
        a_quality * 100.0
    );

    let maxprime = primes.last().unwrap().p;
    let maxlarge: u64 = maxprime * params::large_prime_factor(&n);
    eprintln!("Max cofactor {}", maxlarge);
    let mut gap = fb;
    let mut polys_done = 0;
    for a in a_s.iter() {
        eprintln!(
            "Sieving A={} (factors {})",
            a.a,
            a.factors
                .iter()
                .map(|item| item.p.to_string())
                .collect::<Vec<_>>()[..]
                .join("*")
        );

        for idx in 0..polys_per_a {
            let (pol, sprimes) = make_polynomial(n, primes, a, idx);
            let (mut found, foundlarge) = siqs_sieve_poly(n, a, &pol, primes, &sprimes);
            relations.append(&mut found);
            for r in foundlarge {
                if let Some(rr) = combine_large_relation(&mut larges, &r, &n) {
                    if rr.factors.iter().all(|(p, exp)| exp % 2 == 0) {
                        // FIXME: Poor choice of A's can lead to duplicate relations.
                        eprintln!("FIXME: ignoring trivial relation");
                        //eprintln!("{:?}", rr.factors);
                    } else {
                        relations.push(rr);
                        extras += 1;
                    }
                }
            }
            polys_done += 1;

            if relations.len() >= target {
                gap = relation_gap(*n, &relations);
                if gap == 0 {
                    eprintln!("Found enough relations");
                    break;
                } else {
                    eprintln!("Need {} additional relations", gap);
                    target += gap + std::cmp::min(10, fb as usize / 4);
                }
            }
        }
        eprintln!(
            "Sieved {}M {} polys found {} smooths (cofactors: {} combined, {} pending)",
            ((polys_done) << (mlog + 1 - 10)) >> 10,
            polys_done,
            relations.len(),
            extras,
            larges.len(),
        );
        if gap == 0 {
            break;
        }
    }
    if gap != 0 {
        panic!("Internal error: not enough smooth numbers with selected parameters");
    }
    relations
}

// Parameters:
// m = number of needed A values
// k = number of factors in each A
// A is around sqrt(2N)/M
//
// Number of polynomials: m * 2^(k-1)
// 120..160 bits => k=7 (factors 8-10 bits)
// 160..200 bits => k=8 (factors 10-11 bits)
// 200..250 bits => k=9 (factors 11-13 bits)
// 250..300 bits => k=10 (factors 13-15 bits)

fn nfactors(n: &Uint) -> u32 {
    match n.bits() {
        0..=69 => 2,
        70..=99 => 3,
        100..=129 => 4,
        130..=149 => 5,
        150..=169 => 6,
        170..=189 => 7,
        190..=209 => 8,
        210..=269 => 9,
        270..=309 => 10,
        _ => 11,
    }
}

fn a_value_count(n: &Uint) -> usize {
    let sz = n.bits() as usize;
    match sz {
        0..=129 => 8 + sz / 10,      // 8..20
        130..=169 => sz - 110,       // 20..60
        170..=219 => (sz - 160) * 5, // 50..250
        _ => sz,                     // 220..
    }
}

fn interval_logsize(n: &Uint) -> u32 {
    // Choose very small intervals since the cost of switching
    // polynomials is very small (less than 1ms).
    let sz = n.bits();
    match sz {
        0..=39 => 13,
        40..=59 => 14,
        60..=119 => 15,
        120..=330 => 13 + sz / 40, // 16..21
        _ => 21,
    }
}

// Polynomial selection

pub struct Factors<'a> {
    pub target: Uint,
    pub nfacs: usize,
    pub factors: Vec<&'a Prime>,
    // inverses[i][j] = pi^-1 mod pj
    pub inverses: Vec<Vec<u32>>,
}

// Select factors of generated A values. It is enough to select about
// twice the number of expected factors in A, because the number of
// combinations is large enough to generate values close to the target.
pub fn select_siqs_factors<'a>(fb: &'a [Prime], n: &'a Uint, nfacs: usize) -> Factors<'a> {
    let mlog = interval_logsize(n);
    // The target is sqrt(2N) / 2M. Don't go below 2000 for extremely small numbers.
    let target = max(Uint::from(2000u64), arith::isqrt(n >> 1) >> mlog);
    let idx = fb.partition_point(|p| Uint::from(p.p).pow(nfacs as u32) < target);
    // This may fail for very small n.
    assert!(idx > nfacs && idx + nfacs < fb.len());
    let selection = &fb[idx - nfacs as usize..idx + max(nfacs, 6) as usize];
    // Precompute inverses
    let mut inverses = vec![];
    for p in selection {
        let mut row = vec![];
        for q in selection {
            let pinvq = if p.p == q.p {
                0
            } else {
                q.div.inv(p.p).unwrap()
            };
            row.push(pinvq as u32);
        }
        inverses.push(row);
    }
    Factors {
        target,
        nfacs,
        factors: selection.into_iter().collect(),
        inverses,
    }
}

/// Precomputed information to compute all square roots of N modulo A.
/// We need crt coefficients and the inverse of A modulo the factor base.
pub struct A<'a> {
    a: Uint,
    factors: Vec<&'a Prime>,
    // crt[i] = (a/pi ^1 mod pi) * (a/pi)
    crt: Vec<Uint>,
    ainv: Vec<u32>,
}

pub fn prepare_as<'a>(f: &'a Factors, fb: &[Prime], want: usize) -> Vec<A<'a>> {
    let a_s = select_a(f, want);
    a_s.into_iter().map(|a| prepare_a(f, &a, fb)).collect()
}

/// Find smooth numbers around the target that are products of
/// distinct elements of the factor base.
/// The factor base is assumed to be an array of primes with similar
/// sizes.
fn select_a(f: &Factors, want: usize) -> Vec<Uint> {
    // Sample deterministically products of W primes
    // closest to target and select best candidates.
    // We usually don't need more than 1000 values.
    //
    // We are going to select ~2^W best products of W primes among 2W

    let mut rng: u64 = 0xcafebeefcafebeef;
    let fb = f.factors.len();
    let mut gen = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng % fb as u64
    };
    let mut candidates = vec![];
    for _ in 0..30 {
        for _ in 0..(min(100, f.nfacs << f.nfacs) * want) {
            let mut product = Uint::one();
            let mut mask = 0u64;
            while mask.count_ones() < f.nfacs as u32 - 1 {
                let g = gen();
                if mask & (1 << g) == 0 {
                    mask |= 1 << g;
                    product *= Uint::from(f.factors[g as usize].p);
                }
            }
            let t = (f.target / product).to_u64().unwrap();
            let idx = (0usize..fb)
                .filter(|g| mask & (1 << g) == 0)
                .min_by_key(|&idx| (f.factors[idx].p as i64 - t as i64).abs())
                .unwrap();
            product *= Uint::from(f.factors[idx].p);
            candidates.push(product);
        }
        candidates.sort();
        candidates.dedup();
        let idx = candidates.partition_point(|c| c < &f.target);
        if idx > want && idx + want < candidates.len() {
            return candidates[idx - want / 2..idx + want / 2].to_vec();
        }
    }
    // Should not happen?
    let idx = candidates.partition_point(|c| c < &f.target);
    candidates[idx - min(idx, want / 2)..min(candidates.len(), idx + want / 2)].to_vec()
}

fn prepare_a<'a>(f: &Factors<'a>, a: &Uint, fbase: &[Prime]) -> A<'a> {
    let afactors: Vec<(usize, &Prime)> = f
        .factors
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| p.div.mod_uint(a) == 0)
        .collect();
    let mut crt = vec![];
    // Each CRT coefficient is the product of pj (pj^-1 mod pi)
    for &(idx, p) in afactors.iter() {
        let mut c = Uint::one();
        for &(jdx, q) in afactors.iter() {
            if jdx != idx {
                c *= Uint::from((q.p as u64) * (f.inverses[jdx][idx] as u64));
                debug_assert!(c % q.p == 0);
                debug_assert!(c % p.p == 1);
            }
        }
        crt.push(c % a);
    }
    let mut ainv = vec![];
    for p in fbase {
        let amod = p.div.mod_uint(a);
        ainv.push(p.div.inv(amod).unwrap_or(0) as u32);
    }
    A {
        a: *a,
        factors: afactors.iter().map(|(_, p)| p).copied().collect(),
        crt,
        ainv,
    }
}

#[derive(Debug)]
pub struct Poly {
    a: Uint,
    b: Uint,
    c: Int,
}

/// Given coefficients A and B, compute all roots (Ax+B)^2=N
/// modulo the factor base, computing (r - B)/A mod p
pub fn make_polynomial(n: &Uint, fb: &[Prime], a: &A, idx: usize) -> (Poly, Vec<SievePrime>) {
    let idx = idx << 1;
    // Combine roots using CRT coefficients.
    let mut b = Uint::ZERO;
    for i in 0..a.factors.len() {
        let r = if idx & (1 << i) == 0 {
            a.factors[i].r
        } else {
            a.factors[i].p - a.factors[i].r
        };
        b += a.crt[i] * Uint::from(r);
    }
    debug_assert!((b * b) % a.a == n % a.a);
    // Compute c such that b^2 - ac = N
    // (Ax+B)^2 - n = A(Ax^2 + 2Bx + C)
    let c = (n - b * b) / a.a;
    let pol = Poly {
        a: a.a,
        b,
        c: -Int::from_bits(c),
    };
    // Compute polynomial roots.
    // Ignore factors of A.
    let mut sprimes = vec![];
    for (idx, p) in fb.iter().enumerate() {
        if p.p == 2 {
            // A x^2 + C
            let c2 = c.low_u64() & 1;
            sprimes.push(SievePrime {
                p: p.p,
                roots: [c2, c2],
            });
        } else if !a.factors.iter().any(|q| q.p == p.p) {
            // A x + B = sqrt(n)
            let ainv = a.ainv[idx];
            let bp: u64 = p.div.mod_uint(&b);
            sprimes.push(SievePrime {
                p: p.p,
                roots: [
                    p.div.divmod64((p.p + p.r - bp) * ainv as u64).1,
                    p.div.divmod64((p.p - p.r + p.p - bp) * ainv as u64).1,
                ],
            });
        } else {
            // 2Bx + C, root is -C/2B
            let bp: u64 = p.div.mod_uint(&(b << 1));
            let cp: u64 = p.div.mod_uint(&c); // == -pol.c
            let r = p.div.divmod64(cp * p.div.inv(bp).unwrap()).1;
            sprimes.push(SievePrime {
                p: p.p,
                roots: [r, r],
            });
        }
    }
    assert_eq!(fb.len(), sprimes.len());
    (pol, sprimes)
}

// Sieving process

fn siqs_sieve_poly(
    n: &Uint,
    a: &A,
    pol: &Poly,
    primes: &[Prime],
    sprimes: &[SievePrime],
) -> (Vec<Relation>, Vec<Relation>) {
    let mlog = interval_logsize(&n);
    let nblocks = (1u64 << (mlog - 10)) / (BLOCK_SIZE as u64 / 1024);
    if DEBUG {
        eprintln!(
            "Sieving polynomial A={} B={} M=2^{} blocks={}",
            pol.a, pol.b, mlog, nblocks
        );
    }

    // Sieve from -M to M
    let sieve = SieveSIQS {
        n,
        primes,
        sprimes: &sprimes,
        factors: &a.factors,
        pol,
    };
    // Construct initial state.
    let mut st_primes = vec![];
    let mut st_logs = vec![];
    let mut st_hi = vec![];
    let mut st_lo = vec![];
    let start_offset = -(1 << mlog);
    for (p, sp) in primes.iter().zip(sprimes) {
        assert_eq!(p.p >> 24, 0);
        let [s1, s2] = sieve_starts(p, sp, start_offset);
        let logp = (32 - u32::leading_zeros(p.p as u32)) as u8;
        st_primes.push(p.p as u32);
        st_logs.push(logp);
        st_hi.push((s1 / BLOCK_SIZE as u64) as u8);
        st_lo.push((s1 % BLOCK_SIZE as u64) as u16);
        if s1 != s2 {
            // 2 roots
            st_primes.push(p.p as u32);
            st_logs.push(logp);
            st_hi.push((s2 / BLOCK_SIZE as u64) as u8);
            st_lo.push((s2 % BLOCK_SIZE as u64) as u16);
        }
    }
    // Make st_hi size a multiple of 16.
    st_hi.reserve(16 - st_hi.len() % 16);
    let mut state = StateSIQS {
        idx15: st_primes
            .iter()
            .position(|&p| p > BLOCK_SIZE as u32)
            .unwrap_or(st_primes.len()),
        primes: st_primes,
        logs: st_logs,
        hi: st_hi,
        lo: st_lo,
    };
    if nblocks == 0 {
        return sieve_block_poly(&sieve, -(1 << mlog), 2 << mlog, &mut state);
    }
    let mut result: Vec<Relation> = vec![];
    let mut extras: Vec<Relation> = vec![];
    let nblocks = nblocks as i64;
    for i in -nblocks..nblocks {
        let (mut x, mut y) =
            sieve_block_poly(&sieve, i * BLOCK_SIZE as i64, BLOCK_SIZE, &mut state);
        result.append(&mut x);
        extras.append(&mut y);
        // Decrement MSB by 1.
        let mut idx: usize = 0;
        while idx < state.hi.len() {
            unsafe {
                let p = (&mut state.hi[idx]) as *mut u8 as *mut wide::u8x16;
                *p = (*p).min(*p - 1);
            }
            idx += 16;
        }
    }
    (result, extras)
}

struct SieveSIQS<'a> {
    n: &'a Uint,
    primes: &'a [Prime],
    sprimes: &'a [SievePrime],
    factors: &'a [&'a Prime],
    pol: &'a Poly,
}

struct StateSIQS {
    idx15: usize, // Offset of prime > 32768
    primes: Vec<u32>,
    logs: Vec<u8>,
    // The MSB of the offset for each cursor.
    hi: Vec<u8>,
    // The LSB of the offset for each cursor.
    lo: Vec<u16>,
}

// Compute sieving offsets i,j such that offset+i,j are roots
fn sieve_starts(p: &Prime, sp: &SievePrime, offset: i64) -> [u64; 2] {
    let [r1, r2] = sp.roots;
    let off: u64 = if offset < 0 {
        sp.p - p.div.divmod64((-offset) as u64).1
    } else {
        p.div.divmod64(offset as u64).1
    };
    [
        if r1 < off { r1 + p.p - off } else { r1 - off },
        if r2 < off { r2 + p.p - off } else { r2 - off },
    ]
}

// Sieve using a selected polynomial
fn sieve_block_poly(
    s: &SieveSIQS,
    offset: i64,
    len: usize,
    st: &mut StateSIQS,
) -> (Vec<Relation>, Vec<Relation>) {
    let mut blk = vec![0u8; len];
    unsafe {
        for i in 0..st.idx15 {
            let i = i as usize;
            let p = *st.primes.get_unchecked(i);
            // Small primes always have a hit.
            debug_assert!(st.hi[i] == 0);
            let mut off: usize = *st.lo.get_unchecked(i) as usize;
            let size = *st.logs.get_unchecked(i);
            if p < 1024 {
                let ll = len - 4 * p as usize;
                while off < ll {
                    *blk.get_unchecked_mut(off) += size;
                    off += p as usize;
                    *blk.get_unchecked_mut(off) += size;
                    off += p as usize;
                    *blk.get_unchecked_mut(off) += size;
                    off += p as usize;
                    *blk.get_unchecked_mut(off) += size;
                    off += p as usize;
                }
            }
            while off < len {
                *blk.get_unchecked_mut(off) += size;
                off += p as usize;
            }
            // Update state. No need to set hi=1.
            st.lo[i] = (off % BLOCK_SIZE) as u16;
        }
    }
    for i in st.idx15..st.primes.len() {
        // Large primes have at most 1 hit.
        if st.hi[i] != 0 {
            continue;
        }
        let i = i as usize;
        let p = st.primes[i];
        blk[st.lo[i] as usize] += st.logs[i];
        let off = st.lo[i] as usize + p as usize;
        debug_assert!(off > BLOCK_SIZE);
        st.hi[i] = (off / BLOCK_SIZE) as u8;
        st.lo[i] = (off % BLOCK_SIZE) as u16;
    }

    sieve_result(s, st, offset, len, &blk)
}

fn sieve_result(
    s: &SieveSIQS,
    st: &StateSIQS,
    offset: i64,
    len: usize,
    blk: &[u8],
) -> (Vec<Relation>, Vec<Relation>) {
    let maxprime = s.primes.last().unwrap().p;
    let maxlarge = maxprime * params::large_prime_factor(&s.n);
    let mut result = vec![];
    let mut extras = vec![];

    let target = s.n.bits() / 2 + params::mpqs_interval_logsize(&s.n) - maxlarge.bits();
    let (a, b, c, n) = (s.pol.a, s.pol.b, s.pol.c, s.n);
    for i in 0..len {
        if blk[i] as u32 >= target {
            let mut factors: Vec<(i64, u64)> = Vec::with_capacity(20);
            // Evaluate polynomial Ax^2 + 2Bx+ C
            let x = Int::from(offset + (i as i64));
            let ax_b = Int::from_bits(a) * x + Int::from_bits(b);
            let v = (ax_b + Int::from_bits(b)) * x + c;
            // xrel^2 = (Ax+B)^2 = A * v mod n
            // v is never divisible by A
            let cabs = v.abs().to_bits() % n;
            if v.is_negative() {
                factors.push((-1, 1));
            }
            let mut cofactor: Uint = cabs;
            let arg = offset + i as i64;
            for (idx, item) in s.primes.iter().enumerate() {
                if s.sprimes[idx].roots.contains(&item.div.modi64(arg)) {
                    let mut exp = 0;
                    loop {
                        let (q, r) = item.div.divmod_uint(&cofactor);
                        if r == 0 {
                            cofactor = q;
                            exp += 1;
                        } else {
                            break;
                        }
                    }
                    factors.push((item.p as i64, exp));
                }
            }
            let Some(cofactor) = cofactor.to_u64() else { continue };
            if cofactor > maxlarge {
                continue;
            }
            // Complete with factors of A
            for f in s.factors {
                if let Some(idx) = factors.iter().position(|&(p, _)| p as u64 == f.p) {
                    factors[idx].1 += 1;
                } else {
                    factors.push((f.p as i64, 1));
                }
            }
            let xrel = ax_b.abs().to_bits() % n;
            let xrel = if v.is_negative() { n - xrel } else { xrel };
            if cofactor == 1 {
                if DEBUG {
                    eprintln!("i={} smooth {}", i, v);
                }
                let rel = Relation {
                    x: xrel,
                    cofactor: 1,
                    factors,
                };
                debug_assert!(rel.verify(&s.n));
                result.push(rel);
            } else {
                if DEBUG {
                    eprintln!("x={} smooth {} cofactor {}", x, v, cofactor);
                }
                let rel = Relation {
                    x: xrel,
                    cofactor: cofactor,
                    factors,
                };
                debug_assert!(rel.verify(s.n));
                extras.push(rel);
            }
        }
    }
    (result, extras)
}

#[test]
fn test_poly_a() {
    use crate::fbase;
    use std::str::FromStr;

    // Check that we obtain enough A values for various input sizes.
    const N100: &str = "1037510308142021112704792564947";
    const N120: &str = "966900989857874724182183960752602697";
    const N140: &str = "628343462775940766740025939587872832856351";
    const N160: &str = "924749938828041082847054913126284372335960469233";
    const N180: &str = "609717477947510609865834953348014542054334353064133851";
    const N200: &str = "1499802708882526909122644146289721370711415544090318473858983";
    const N220: &str = "960924940937451908640130999851435804366317230146233463701332018699";
    const N240: &str = "1563849171863495214507949103370077342033765608728382665100245282240408041";
    const N260: &str =
        "960683424930949054059211750923998830300819600123951352711285164932272044258249";
    const N280: &str =
        "768443048336808679050045973609532661331806099418806842517773942579627976951969897517";
    const N300: &str = "1441811698044810694576317861317873120840140792275400485937339204632253015892676440922745819";
    const N320: &str = "1420239033309714984094415212531751615454438351340772432135724366126533462488286213242822097387831";

    fn compute(n: &Uint, nfacs: u8, want: usize) {
        // Divide by 20 to obtain faster tests.
        let want = if want > 50 { want / 20 } else { want };

        let fb_size = params::factor_base_size(*n);
        let ps = fbase::primes(fb_size);
        let fb = fbase::prepare_factor_base(n, &ps);

        let facs = select_siqs_factors(&fb, n, nfacs as usize);
        let target = facs.target;

        let a_vals = select_a(&facs, want);
        let _10000 = Uint::from(10000u64);
        let a_first = a_vals.first().unwrap();
        let a_last = a_vals.last().unwrap();
        let d1: f64 = (target * _10000 / a_first).low_u64() as f64 / 100.0;
        let d2: f64 = (a_last * _10000 / target).low_u64() as f64 / 100.0;
        eprintln!(
            "N {} bits, {} values [{}..{}], quality -{:.2}%..{:.2}%",
            n.bits(),
            a_vals.len(),
            a_first,
            a_last,
            d1 - 100.0,
            d2 - 100.0
        );
        if n.bits() < 150 {
            // 1% tolerance
            assert!(d1 - 100.0 < 1.0);
            assert!(d2 - 100.0 < 1.0);
            assert!(a_vals.len() >= want);
        } else {
            assert!(d1 - 100.0 < 0.5);
            assert!(d2 - 100.0 < 0.5);
            assert!(a_vals.len() >= want / 2);
        }
    }
    // Need very few polynomials
    compute(&Uint::from_str(N100).unwrap(), 4, 4); // 100
    compute(&Uint::from_str(N120).unwrap(), 4, 6); // 120
    compute(&Uint::from_str(N140).unwrap(), 5, 10);
    compute(&Uint::from_str(N160).unwrap(), 6, 10);
    // Need 1000-5000 polynomials
    compute(&Uint::from_str(N180).unwrap(), 7, 100);
    compute(&Uint::from_str(N200).unwrap(), 8, 100);
    // Need >10k-50k polynomials
    compute(&Uint::from_str(N220).unwrap(), 9, 200);
    compute(&Uint::from_str(N240).unwrap(), 9, 500);
    // Takes a long time.
    compute(&Uint::from_str(N260).unwrap(), 9, 1000);
    compute(&Uint::from_str(N280).unwrap(), 10, 1000);
    compute(&Uint::from_str(N300).unwrap(), 10, 2000);
    compute(&Uint::from_str(N320).unwrap(), 11, 2000);
}

#[test]
fn test_poly_prepare() {
    use crate::fbase;
    use std::str::FromStr;

    const N240: &str = "1563849171863495214507949103370077342033765608728382665100245282240408041";
    let n = &Uint::from_str(N240).unwrap();
    let primes = fbase::primes(10000);
    let fb = fbase::prepare_factor_base(&n, &primes[..]);
    // Prepare A values
    // Only test 10 A values and 35 polynomials per A.
    let f = select_siqs_factors(&fb[..], &n, 9);
    let a_s = prepare_as(&f, &fb, 10);
    for a in a_s.iter() {
        // Check CRT coefficients.
        assert_eq!(a.a, a.factors.iter().map(|x| Uint::from(x.p)).product());
        for (i, c) in a.crt.iter().enumerate() {
            for (j, p) in a.factors.iter().enumerate() {
                if i == j {
                    assert_eq!(*c % p.p, 1);
                } else {
                    assert_eq!(*c % p.p, 0);
                }
            }
        }
        for idx in 0..35 {
            let idx = 7 * idx;
            // Generate and check each polynomial.
            let (pol, sprimes) = make_polynomial(n, &fb, a, idx);
            let (a, b, c) = (pol.a, pol.b, pol.c);
            // B is a square root of N modulo A.
            assert_eq!((b * b) % a, n % a);
            // Check that (Ax+B)^2 - n = A(Ax^2+2Bx+C)
            for x in [1u64, 100, 50000] {
                let u = a * Uint::from(x) + b;
                let u = Int::from_bits(u * u) - Int::from_bits(*n);
                let v = a * Uint::from(x * x) + (b << 1) * Uint::from(x);
                let v = Int::from_bits(v) + c;
                assert_eq!(u, Int::from_bits(a) * v);
            }
            for sp in sprimes {
                // Roots are roots of Ax^2+2Bx+C modulo p.
                let p = sp.p;
                let [r1, r2] = sp.roots;
                let v = a * Uint::from(r1 * r1) + b * Uint::from(2 * r1);
                let v = Int::from_bits(v) + c;
                assert_eq!(v.abs().to_bits() % p, 0);
                let v = a * Uint::from(r2 * r2) + b * Uint::from(2 * r2);
                let v = Int::from_bits(v) + c;
                assert_eq!(v.abs().to_bits() % p, 0);
            }
        }
    }
}
