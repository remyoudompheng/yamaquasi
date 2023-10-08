// Copyright 2022 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! An artificial benchmark for matrix algebra.
//!
//! It tries to imitate the shape of quadratic sieve matrices by packing
//! coefficients towards first rows.
//!
//! However the current choices fail to reproduce the actual performance
//! of linear algebra during actual factorization.
//!
//! For size 20000 we expect Gauss elimination to be at least 10x slower
//! than block Lanczos (case of 270-bit integers in SIQS with double large primes).

///
use std::time::Instant;

use bitvec_simd::BitVec;
use rand::{self, Rng};

use yamaquasi::matrix::gf2 as matrix;
use yamaquasi::Verbosity;

fn main() {
    for size in [
        100, 200, 500, 1000, 2000, 3000, 4000, 5000, 10_000, 20_000, 50_000, 80_000, 100_000,
        200_000,
    ] {
        // Density is 10 + log N
        let density = 42 - u32::leading_zeros(size as u32);
        let mat = qs_like_matrix(size, 30, density as usize);
        if size > 250 {
            let start = Instant::now();
            let ker = matrix::kernel_lanczos(&mat, Verbosity::Info);
            eprintln!(
                "size={} Lanczos {:.3}s",
                size,
                start.elapsed().as_secs_f64()
            );
            assert!(ker.len() > 0);
        }
        if size < 25_000 {
            let mat = to_bitvec(&mat);
            let start = Instant::now();
            let ker = matrix::kernel_gauss(mat);
            assert!(ker.len() > 0);
            eprintln!("size={} Gauss {:.3}s", size, start.elapsed().as_secs_f64());
        }
    }
}

/// Randomly generate a sparse matrix similar to a quadratic sieve result.
///
/// Sample distributions looks like:
/// 180 bits
/// 2847x2860 (24.3 entries/col, p50=33 p80=396 p90=1083 p95=1698 p99=2490)
/// 2719x2733 (23.8 entries/col, p50=32 p80=377 p90=1038 p95=1623 p99=2385)
///
/// 200 bits
/// 5339x5351 (25.4 entries/col, p50=35 p80=646 p90=1928 p95=3118 p99=4653)
/// 5584x5598 (26.0 entries/col, p50=35 p80=636 p90=1983 p95=3251 p99=4870)
///
/// 220 bits
/// 9788x9809 (27.8 entries/col, p50=37 p80=921 p90=3215 p95=5471 p99=8460)
/// 9708x9746 (28.0 entries/col, p50=37 p80=824 p90=3046 p95=5320 p99=8359)
/// (60% in first size 64 block)
///
/// 240 bits
/// 17419x17482 (27.6 entries/col, p50=38 p80=1575 p90=5586 p95=9639 p99=15010)
///
/// 260 bits
/// 35522x35550 (45.1 entries/col, p50=80 p80=4459 p90=12377 p95=19796 p99=30125)
///
/// Larger matrices have 40-45% of coefficients in first block.
///
/// where the CDF looks like a power law (x^0.1)
fn qs_like_matrix(size: usize, extra: usize, density: usize) -> matrix::SparseMat {
    let mut stats = vec![0u32; size];
    let mut rng = rand::thread_rng();
    let mut cols = vec![];
    for _ in 0..size + extra {
        let mut col = vec![];
        while col.len() < density {
            // Use the power law slightly combined with a uniform law.
            let x: f64 = rng.gen_range(0.0..2.0);
            let x = if x < 1.0 {
                0.02 * x + 0.98 * x.powf(10.0)
            } else {
                // Concentrate half of coefficients in the dense part.
                // It must fit in about 64 columns, let's select an exponential
                // law with parameter log2(size) so that large matrices
                // have less than half of coefficients in first block.
                (1.0_f64).min(-(x - 0.99999).ln() * (size as f64).log2() / size as f64)
            };
            let idx = (size as f64 * x) as usize;
            if idx < size && !col.contains(&idx) {
                col.push(idx);
                stats[idx] += 1;
            }
        }
        cols.push(col);
    }
    // Each row must have at least 2 coefficients (as QS filtered matrices).
    // This also avoids small rank for transposed matrix, which is harmful
    // to Block Lanczos.
    let mut extra_fill = 0;
    for i in 0..size {
        if stats[i] < 2 {
            for j in i..i + extra {
                if !cols[j].contains(&i) {
                    cols[j].push(i);
                    stats[i] += 1;
                    extra_fill += 1;
                }
                if stats[i] >= 2 {
                    break;
                }
            }
        }
    }
    let (mut p50, mut p80, mut p90, mut p95, mut p99) = (0, 0, 0, 0, 0);
    let total: u32 = stats.iter().sum();
    let mut sum = 0u32;
    for (idx, &k) in stats.iter().enumerate() {
        sum += k;
        if sum * 100 <= total * 50 {
            p50 = idx
        }
        if sum * 100 <= total * 80 {
            p80 = idx
        }
        if sum * 100 <= total * 90 {
            p90 = idx
        }
        if sum * 100 <= total * 95 {
            p95 = idx
        }
        if sum * 100 <= total * 99 {
            p99 = idx
        }
    }
    let densepart: u32 = stats[..64].iter().sum();
    eprintln!(
        "Generated matrix of size {} (dense block={:.2}% p50={} p80={} p90={} p95={} p99={}) filled={}",
        size, 100.0*densepart as f64 / total as f64, p50, p80, p90, p95, p99, extra_fill
    );
    matrix::SparseMat { k: size, cols }
}

fn to_bitvec(mat: &matrix::SparseMat) -> Vec<BitVec> {
    let mut dense = vec![];
    for col in mat.cols.iter() {
        let mut v = BitVec::zeros(mat.k);
        for &idx in col {
            v.set(idx, true);
        }
        dense.push(v);
    }
    dense
}
