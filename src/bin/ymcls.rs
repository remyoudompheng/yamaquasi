// Copyright 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;

use yamaquasi::classgroup;
use yamaquasi::{Int, Preferences, Verbosity};

fn main() {
    let arg = arguments::parse(std::env::args()).unwrap();
    if arg.get::<bool>("help").is_some() || !matches!(arg.orphans.len(), 1 | 2) {
        eprintln!("Usage: ymcls [OPTIONS] [-]DISCRIMINANT [OUTPUTDIR]");
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
    let outdir = if arg.orphans.len() > 1 {
        Some(&arg.orphans[1])
    } else {
        None
    };
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
    }
    prefs.outdir = outdir.map(PathBuf::from);

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
    let g = classgroup::classgroup(&d, &prefs, tpool);
    let Some(g) = g else {
        return;
    };
    let mut buf = vec![];
    // Group invariants
    write!(&mut buf, "G").unwrap();
    for d in &g.invariants {
        write!(&mut buf, " {d}").unwrap();
    }
    buf.push(b'\n');
    // Coordinates
    for (p, v) in g.gens {
        write!(&mut buf, "{p}").unwrap();
        for &x in &v {
            write!(&mut buf, " {x}").unwrap();
        }
        buf.push(b'\n');
    }
    if let Some(outdir) = prefs.outdir {
        let mut w = fs::File::create(outdir.join("group.structure")).unwrap();
        w.write(&buf[..]).unwrap();
    }
    std::io::stdout().write(&buf[..]).unwrap();
}
