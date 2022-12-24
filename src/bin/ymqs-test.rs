// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Random YMQS testing

use std::time::Instant;

use bnum::cast::CastFrom;
use rand::{self, Rng};

use yamaquasi::arith::{self, Num, U256};
use yamaquasi::Uint;
use yamaquasi::{fbase, params};
use yamaquasi::{mpqs, qsieve, relations, siqs};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.get::<bool>("help").is_some() {
        eprintln!("Usage: ymqs [OPTIONS]");
        eprintln!("");
        eprintln!("Options:");
        eprintln!("  --help                    show this help");
        eprintln!("  --mode qs|qs64|mpqs|siqs: force algorithm selection");
        eprintln!("  --bits B:                 input length");
        return;
    }
    let mode = arg.get::<String>("mode").unwrap_or("siqs".into());
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
        eprintln!("p={} q={} => n={}", p, q, p * q);
        // Factor
        let n = p * q;
        let (k, _) = fbase::select_multiplier(n);
        eprintln!("multiplier k={}", k);
        let nk = n * Uint::from(k);
        let prefs = params::Preferences {
            fb_size: None,
            large_factor: None,
            use_double: None,
        };
        let rels = match &mode[..] {
            "qs" => qsieve::qsieve(nk, &prefs, None),
            "mpqs" => mpqs::mpqs(nk, &prefs, None),
            "siqs" => siqs::siqs(&nk, &prefs, None),
            _ => {
                eprintln!("Invalid operation mode {:?}", mode);
                return;
            }
        };
        let pq = relations::final_step(&n, &rels, true);
        if pq.is_none() {
            eprintln!("ERROR failed to factor {}={}*{}", n, p, q);
            std::process::exit(1);
        }
        assert!(pq == Some((p, q)) || pq == Some((q, p)));
        i += 1;
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!(
            "Processed {} numbers in {:.3}s (average {:.3}s)",
            i,
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
        let s = (p.low_u64() - 1).trailing_zeros();
        for &b in fbase::SMALL_PRIMES {
            let mut pow =
                arith::pow_mod(Uint::from(b), Uint::cast_from(p) >> s, Uint::cast_from(p));
            let p = Uint::cast_from(p);
            let pm1 = p - Uint::from(1u64);
            let mut ok = pow.to_u64() == Some(1) || pow == pm1;
            for _ in 0..s {
                pow = (pow * pow) % p;
                if pow == pm1 {
                    ok = true;
                    break;
                } else if pow.to_u64() == Some(1) {
                    break;
                }
            }
            if !ok {
                continue 'nextcandidate;
            }
        }
        return p;
    }
    unreachable!("impossible");
}
