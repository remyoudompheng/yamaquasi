//! Benchmark for linear algebra of classgroup computations
//!
//! Requires a data directory from ymcls.

use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::PathBuf;

use yamaquasi::relationcls;
use yamaquasi::Verbosity;

fn main() -> io::Result<()> {
    let dir = std::env::args().nth(1).unwrap();
    let threads = || -> Option<usize> {
        let s = std::env::args().nth(2)?;
        str::parse::<usize>(&s).ok()
    }();
    let dirpath = PathBuf::from(dir);
    // Read parameters
    let f = fs::File::open(dirpath.join("args.json"))?;
    let mut buf = BufReader::new(f);
    let mut line = String::new();
    let (mut hmin, mut hmax) = (0.0, 0.0);
    while buf.read_line(&mut line).is_ok() && !line.is_empty() {
        if line.contains("h_estimate_min") {
            let idx1 = line.find(':').unwrap();
            let idx2 = line.find(',').unwrap();
            let s = (&line[idx1 + 1..idx2]).trim();
            hmin = s.parse::<f64>().unwrap();
        }
        if line.contains("h_estimate_max") {
            let idx1 = line.find(':').unwrap();
            let s = (&line[idx1 + 1..]).trim();
            hmax = s.parse::<f64>().unwrap();
        }
        line.clear();
    }
    eprintln!("Class number bounds {hmin}..{hmax}");
    // Read relations
    let f = fs::File::open(dirpath.join("relations.sieve"))?;
    let mut buf = BufReader::new(f);
    let mut line = String::new();
    let mut rels = vec![];
    while buf.read_line(&mut line).is_ok() {
        if line.is_empty() {
            // EOF
            break;
        }
        let mut factors: Vec<(u32, i32)> = vec![];
        'words: for word in line.split_ascii_whitespace() {
            let psign = str::parse::<i32>(word).unwrap();
            let p = psign.unsigned_abs();
            let e = psign.signum();
            for pe in factors.iter_mut() {
                if pe.0 == p {
                    pe.1 += e;
                    continue 'words;
                }
            }
            factors.push((p, e));
        }
        let rel = relationcls::CRelation {
            factors,
            large1: None,
            large2: None,
        };
        rels.push(rel);
        line.clear();
    }
    eprintln!("{} relations loaded", rels.len());
    // Use thread pool if requested.
    let tpool: Option<rayon::ThreadPool> = threads.map(|t| {
        eprintln!("Using a pool of {t} threads");
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .expect("cannot create thread pool")
    });
    let tpool = tpool.as_ref();

    relationcls::group_structure(rels, None, (hmin, hmax), Verbosity::Debug, None, tpool);
    Ok(())
}
