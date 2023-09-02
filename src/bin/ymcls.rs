// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::str::FromStr;

use num_traits::cast::ToPrimitive;

use yamaquasi::arith::{Dividers, Num};
use yamaquasi::classgroup;
use yamaquasi::{fbase, params};
use yamaquasi::{Int, Preferences, Uint, Verbosity};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.get::<bool>("help").is_some() || arg.orphans.len() != 1 {
        eprintln!("Usage: ymcls [OPTIONS] [-]DISCRIMINANT");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --help                    show this help");
        eprintln!("  --verbose silent|info|verbose|debug");
        eprintln!("  --threads N:              enable up to N computation threads");
        eprintln!("  --fb F:                   override automatic factor base size");
        eprintln!("  --large B1:               multiplier for large primes");
        eprintln!("  --use-double true|false:  use double large prime");
        return;
    }
    let threads = arg.get::<usize>("threads");
    let fb_user = arg.get::<u32>("fb");
    let large = arg.get::<u64>("large");
    let double = arg.get::<bool>("use-double");
    let v = arg.get::<String>("verbose").unwrap_or("info".into());
    let number = &arg.orphans[0];
    let mut d = Int::from_str(number).expect("could not read input number");
    if d.is_positive() {
        d = -d;
    }
    const MAXBITS: u32 = 512;
    if d.unsigned_abs().bits() > MAXBITS {
        panic!(
            "Number size ({} bits) exceeds {MAXBITS} bits limit",
            d.unsigned_abs().bits(),
        )
    }
    if d.bit(1) {
        panic!("Discriminant must be 0 or 1 mod 4");
    }

    let mut prefs = Preferences::default();
    prefs.fb_size = fb_user;
    prefs.large_factor = large;
    prefs.use_double = double;
    prefs.threads = threads;
    prefs.verbosity = Verbosity::from_str(&v).unwrap();
    if prefs.verbose(Verbosity::Info) {
        eprintln!("Computing class group of discriminant {d}");
        let (hmin, hmax) = estimate(&d);
        eprintln!("Estimate by class number formula {hmin:.5e}-{hmax:.5e}")
    }

    // Create thread pool
    let tpool: Option<rayon::ThreadPool> = match prefs.threads {
        None | Some(1) => None,
        Some(t) => {
            if prefs.verbose(Verbosity::Verbose) {
                eprintln!("Using a pool of {t} threads");
            }
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .expect("cannot create thread pool"),
            )
        }
    };
    let tpool = tpool.as_ref();
    classgroup::ideal_relations(&d, &prefs, tpool);
}

/// Compute an estimate of the class number.
fn estimate(d: &Int) -> (f64, f64) {
    // The class number formula is:
    // h(-D) = sqrt(D)/pi * prod(1/(1 - (D|p)/p) for prime p)
    // For p=2 the factor is 1/(1-1/2)=2 if D % 8 = 1
    // otherwise 1/(1+1/2) = 2/3
    //
    // Numerical evaluation takes
    // ~0.1s for bound 10^7
    // ~1s for bound 10^8
    // ~5s for bound 10^9
    let fbsize = params::clsgrp_fb_size(d.unsigned_abs().bits(), true);
    // enough to get 4 decimal digits
    let bound = std::cmp::min(100_000_000, fbsize * fbsize);
    let mut logprod = 0f64;
    let mut logmin = f64::MAX;
    let mut logmax = f64::MIN;
    let mut s = fbase::PrimeSieve::new();
    let dabs = d.unsigned_abs();
    'primeloop: loop {
        let block = s.next();
        for &p in block {
            if p == 2 {
                continue;
            }
            // legendre(-d,p) = legendre(d,p) * (-1)^(p-1)/2
            let mut l = legendre(&dabs, p);
            if p % 4 == 3 {
                l = -l;
            }
            logprod += -(-l as f64 / p as f64).ln_1p();
            if p > bound {
                break 'primeloop;
            }
            if p > bound / 2 {
                // Compute loweR/upper bounds over a window
                logmin = logmin.min(logprod);
                logmax = logmax.max(logprod);
            }
        }
    }
    let h = d.to_f64().unwrap().abs().sqrt() / std::f64::consts::PI;
    let h = match d.unsigned_abs().low_u64() & 7 {
        // Only values 7, 4, 3 are valid for fundamental discriminants.
        5 | 7 => h * 2.0,
        0 | 2 | 4 | 6 => h,
        1 | 3 => h * 2.0 / 3.0,
        _ => unreachable!(),
    };
    (h * logmin.exp(), h * logmax.exp())
}

fn legendre(d: &Uint, p: u32) -> i32 {
    let div = Dividers::new(p);
    let dmodp = div.mod_uint(d) as u32;
    let mut k = p / 2;
    let mut pow = 1u64;
    let mut sq = dmodp as u64;
    while k > 0 {
        if k & 1 == 1 {
            pow = div.modu63(pow * sq);
        }
        sq = div.modu63(sq * sq);
        k = k >> 1;
    }
    if pow > 1 {
        debug_assert!(pow == p as u64 - 1);
        pow as i32 - p as i32
    } else {
        debug_assert!(pow <= 1);
        pow as i32
    }
}

#[test]
fn test_estimate() {
    // D = 1-8k
    let d =
        Int::from_str("-1139325066844575699589813265217200398493708241839938355464231").unwrap();
    // h=964415698883565364637432450736
    let (h1, h2) = estimate(&d);
    assert!(h1 <= 9.644157e29 && 9.644157e29 <= h2 * 1.001);

    // D = 5-8k
    let d = Int::from_str("-12239807779826253214859975412431303497371919444169932188160735019")
        .unwrap();
    // h=109997901313565058259819609742265
    let (h1, h2) = estimate(&d);
    assert!(h1 <= 1.099979e32 && 1.099979e32 <= h2);

    // D = 4-8k
    let d = Int::from_str("-40000000000000000000000000000000000000000000000000000000000000004")
        .unwrap();
    // h=178397819605839608466892693850112
    let (h1, h2) = estimate(&d);
    eprintln!("{h1} {h2}");
    assert!(h1 <= 1.783978e32 && 1.783978e32 <= h2);
}
