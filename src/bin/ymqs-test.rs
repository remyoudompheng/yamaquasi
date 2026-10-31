// Copyright 2022, 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Random YMQS testing.
//!
//! This program generates random semiprimes for the requested bit length
//! and runs yamaquasi with selected algorithm for test and benchmark purposes.

use std::str::FromStr;
use std::time::Instant;

use bnum::cast::CastFrom;
use rand::{self, Rng};

use yamaquasi::arith::U256;
use yamaquasi::fbase;
use yamaquasi::{factor, isprime64, pseudoprime, Algo, Preferences, Verbosity};
use yamaquasi::{Int, Uint};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.get::<bool>("help").is_some() {
        eprintln!("Usage: ymqs [OPTIONS]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --help                    show this help");
        eprintln!("  --sampling semiprime|uniform");
        eprintln!("  --mode ecm|ecm128|rho|qs|mpqs|siqs: force algorithm selection");
        eprintln!("  --bits B:                 input length");
        return;
    }
    let mode = arg.get::<String>("mode").unwrap_or("auto".into());
    let smpl = arg.get::<String>("sampling").unwrap_or("semiprime".into());
    let bits = arg.get::<u32>("bits").unwrap_or(64);
    assert!(matches!(smpl.as_str(), "semiprime" | "uniform"));
    // Prepare a factor base for trial division
    let fbase = fbase::FBase::new(Int::ZERO, 2 * bits);
    let mut rng = rand::thread_rng();
    let mut i = 0;
    let mut inputs = vec![];
    if smpl == "semiprime" {
        for _ in 0..0x4000 {
            let mut words = [0u64; 4];
            rng.try_fill(&mut words).unwrap();
            let p0 = U256::from_digits(words) | U256::power_of_two(255);
            rng.try_fill(&mut words).unwrap();
            let q0 = U256::from_digits(words) | U256::power_of_two(255);
            let p = Uint::cast_from(nextprime(&fbase, p0 >> (256 - bits / 2)));
            let q = Uint::cast_from(nextprime(&fbase, q0 >> (256 - bits + bits / 2)));
            eprintln!("p={p} q={q} => n={}", p * q);
            inputs.push(p * q);
        }
    } else {
        for _ in 0..0x4000 {
            let mut words = [0u64; 4];
            rng.try_fill(&mut words).unwrap();
            let p0 = U256::from_digits(words) | U256::power_of_two(255);
            let p = p0 >> (256 - bits);
            inputs.push(Uint::cast_from(p));
        }
    }
    let mut prefs = Preferences::default();
    prefs.verbosity = Verbosity::Silent;
    let alg = Algo::from_str(&mode).unwrap();
    let t0 = Instant::now();
    loop {
        let n = inputs[i & 0x3fff];
        // Factor
        let pq = factor(n, alg, &prefs).unwrap();
        if smpl == "semiprime" {
            if pq.len() != 2 {
                eprintln!("ERROR failed to factor {n}");
                std::process::exit(1);
            }
            assert!(pq[0] * pq[1] == n);
            assert!(pq[0] > Uint::ONE && pq[1] > Uint::ONE);
        } else {
            assert_eq!(n, pq.iter().product::<Uint>());
        }
        i += 1;
        if bits < 64 && i % 10000 == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let avg = elapsed / (i as f64) * 1e6;
            eprintln!("Processed {i} numbers in {elapsed:.3}s (average {avg:.3}us)");
        }
        if bits >= 64 && i % 100 == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let avg = elapsed / (i as f64) * 1000.;
            eprintln!("Processed {i} numbers in {elapsed:.3}s (average {avg:.3}ms)");
        }
    }
}

fn nextprime64(fb: &fbase::FBase, base: u64) -> u64 {
    'nextcandidate: for i in 0..8000 {
        let p = base + i;
        for pidx in 0..fb.len() {
            if fb.div(pidx).divmod64(p).1 == 0 {
                if p == fb.p(pidx) as u64 {
                    return p;
                }
                continue 'nextcandidate;
            }
        }
        // Naive Miller test
        if isprime64(p) {
            return p;
        }
    }
    unreachable!("impossible");
}

fn nextprime(fb: &fbase::FBase, base: U256) -> U256 {
    if base.bits() < 64 {
        return nextprime64(fb, base.digits()[0]).into();
    }
    'nextcandidate: for i in 0..8000 {
        let p = base + U256::from(i as u64);
        for pidx in 0..fb.len() {
            if fb.div(pidx).mod_uint(&p) == 0 {
                if Ok(fb.p(pidx) as u64) == p.try_into() {
                    return p;
                }
                continue 'nextcandidate;
            }
        }
        if pseudoprime(Uint::cast_from(p)) {
            return p;
        }
    }
    unreachable!("impossible");
}
