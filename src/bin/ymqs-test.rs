// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Random YMQS testing

use std::str::FromStr;
use std::time::Instant;

use bnum::cast::CastFrom;
use rand::{self, Rng};

use yamaquasi::arith::{Num, U256};
use yamaquasi::fbase;
use yamaquasi::Uint;
use yamaquasi::{factor, pseudoprime, Algo, Preferences};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.get::<bool>("help").is_some() {
        eprintln!("Usage: ymqs [OPTIONS]");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --help                    show this help");
        eprintln!("  --mode ecm|qs|qs64|mpqs|siqs: force algorithm selection");
        eprintln!("  --bits B:                 input length");
        return;
    }
    let mode = arg.get::<String>("mode").unwrap_or("auto".into());
    let bits = arg.get::<u32>("bits").unwrap_or(64);
    // Prepare a factor base for trial division
    let fbase = fbase::FBase::new(Uint::from(0u64), 1000);
    let mut rng = rand::thread_rng();
    let mut i = 0;
    let t0 = Instant::now();
    loop {
        let mut words = [0u64; 4];
        rng.try_fill(&mut words).unwrap();
        let p0 = U256::from_digits(words);
        rng.try_fill(&mut words).unwrap();
        let q0 = U256::from_digits(words);
        let p = Uint::cast_from(nextprime(&fbase, p0 >> (256 - bits / 2)));
        let q = Uint::cast_from(nextprime(&fbase, q0 >> (256 - bits / 2)));
        eprintln!("p={p} q={q} => n={}", p * q);
        // Factor
        let n = p * q;
        let prefs = Preferences {
            fb_size: None,
            large_factor: None,
            use_double: None,
            threads: None,
            verbose: false,
        };
        let alg = Algo::from_str(&mode).unwrap();
        let pq = factor(n, alg, &prefs);
        if pq.len() != 2 {
            eprintln!("ERROR failed to factor {n}={p}*{q}");
            std::process::exit(1);
        }
        assert!(&pq == &[p, q] || &pq == &[q, p]);
        i += 1;
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!(
            "Processed {i} numbers in {:.3}s (average {:.3}s)",
            elapsed,
            elapsed / (i as f64)
        );
    }
}

fn nextprime(fb: &fbase::FBase, base: U256) -> U256 {
    'nextcandidate: for i in 0..8000 {
        let p = base + U256::from(i as u64);
        for pidx in 0..fb.len() {
            if Some(fb.p(pidx) as u64) == p.to_u64() {
                return p;
            }
            if fb.div(pidx).mod_uint(&p) == 0 {
                continue 'nextcandidate;
            }
        }
        // Naive Miller test
        if !pseudoprime(Uint::cast_from(p)) {
            continue 'nextcandidate;
        }
        return p;
    }
    unreachable!("impossible");
}
