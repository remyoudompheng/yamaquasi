// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: 2026 Rémy Oudompheng

//! ECM parameter sweep
//!
//! This program generates random semiprimes for the requested bit length
//! and runs ECM with various parameters.

use std::time::Instant;

use bnum::cast::CastFrom;
use rand::{self, Rng};

use yamaquasi::arith::U256;
use yamaquasi::fbase;
use yamaquasi::{ecm, isprime64, pseudoprime, Preferences, Verbosity};
use yamaquasi::{Int, Uint};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.get::<bool>("help").is_some() {
        eprintln!("Usage: ecm_test [OPTIONS]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --help                    show this help");
        eprintln!("  --samples N               number of samples");
        eprintln!("  --bits B:                 input length");
        return;
    }
    let samples = arg.get::<usize>("samples").unwrap_or(1000);
    let bits = arg.get::<u32>("bits").unwrap();
    // Prepare a factor base for trial division
    let fbase = fbase::FBase::new(Int::ZERO, 2 * bits);
    let mut rng = rand::thread_rng();
    let mut inputs = vec![];
    // p has B bits, q has 96 bits
    let qbits = bits + std::cmp::max(32, bits / 2);
    eprintln!("Using semiprimes {bits}b × {qbits}b");
    for _ in 0..10 * samples {
        let mut words = [0u64; 4];
        rng.try_fill(&mut words).unwrap();
        let p0 = U256::from_digits(words) | U256::power_of_two(255);
        rng.try_fill(&mut words).unwrap();
        let q0 = U256::from_digits(words) | U256::power_of_two(255);
        let p = Uint::cast_from(nextprime(&fbase, p0 >> (256 - bits)));
        let q = Uint::cast_from(nextprime(&fbase, q0 >> (256 - qbits)));
        inputs.push(p * q);
    }

    let params = [
        // (minsize, maxsize, cost class, B1, B2)
        (0, 20, 300, 10, 480.),
        (0, 20, 300, 12, 252.),
        (0, 24, 500, 10, 660.),
        (0, 24, 700, 30, 660.),
        (0, 24, 700, 10, 1080.),
        (0, 32, 1000, 10, 5040.),
        (0, 32, 1000, 40, 1920.),
        (0, 32, 1000, 50, 660.),
        (0, 40, 2000, 40, 7700.),
        (0, 40, 2000, 80, 5040.),
        (0, 40, 2000, 100, 1920.),
        (0, 48, 4000, 60, 20000.),
        (0, 48, 4000, 120, 13200.),
        (0, 48, 4000, 200, 7700.),
        (0, 56, 8000, 200, 33000.),
        (0, 56, 8000, 300, 20000.),
        (0, 56, 8000, 400, 13200.),
        (16, 64, 16000, 400, 81000.),
        (16, 64, 16000, 800, 33000.),
        (16, 64, 16000, 1000, 20000.),
        (16, 64, 32000, 800, 180_000.),
        (16, 64, 32000, 1600, 81_000.),
        (16, 64, 32000, 2500, 53_000.),
        (24, 64, 50000, 1500, 320_000.),
        (24, 64, 50000, 3000, 126_000.),
        (24, 64, 50000, 4000, 81_000.),
        (32, 72, 100000, 3000, 550_000.),
        (32, 72, 100000, 5000, 320_000.),
        (32, 72, 100000, 7000, 180_000.),
        (32, 72, 200000, 5000, 1.4e6),
        (32, 72, 200000, 10000, 800e3),
        (32, 72, 200000, 15000, 320e3),
        // Cost = 13.7 B1 + 20 sqrt(B2) log(B2)
        (32, 80, 500000, 9000, 7e6),
        (32, 80, 500000, 15000, 4.5e6),
        (32, 80, 500000, 27000, 1.4e6),
        (48, 96, 1000000, 15000, 30e6),
        (48, 96, 1000000, 25000, 20e6),
        (48, 96, 1000000, 50000, 4.7e6),
        // Use ecm_calibrate.py for larger params.
    ];

    for (minsize, maxsize, cost, b1, b2) in params {
        if bits < minsize || bits > maxsize {
            continue;
        }
        let t0 = Instant::now();
        let mut ok = 0;
        let ns = if cost < 100000 {
            &inputs
        } else {
            &inputs[..samples]
        };
        for &n in ns {
            // Reset ecm seed to zero.
            let mut prefs = Preferences::default();
            prefs.verbosity = Verbosity::Silent;
            let pq = ecm::ecm(n, 1, b1, b2, &prefs, None);
            if let Some((p, q)) = pq {
                assert!(p.bits() > 1 && q.bits() > 1);
                ok += 1;
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let avg = elapsed / (ns.len() as f64);
        let rate = ok as f64 / ns.len() as f64;
        let eta = avg / rate;
        eprintln!(
            "Processed {} numbers in {elapsed:.3}s ({cost}, {b1}, {b2}) rate={rate:.4} eta={eta:.6}s",
            ns.len()
        );
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
