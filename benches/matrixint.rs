//! Benchmarks for determinant of large matrices over GF(p).
//!
//! Dense and sparse linear algebra benchmarks both use the same matrices
//! as the GF(2) tests, reinterpreted over a different ring.

use std::time::Instant;

use yamaquasi::matrix::gf2::make_test_sparsemat;
use yamaquasi::matrix::intdense::GFpEchelonBuilder;
use yamaquasi::matrix::intsparse::SparseMat;

fn main() {
    let p: u64 = 1_000_000_000_000_037;
    let p2: u64 = 1_000_000_000_000_091;
    let p3: u64 = 1_000_000_000_000_159;
    let p4: u64 = 1_000_000_000_000_187;
    for size in [30, 100, 200, 500, 1000, 2000, 5000, 10000, 20000] {
        let logsize = u32::BITS - u32::leading_zeros(size);
        let density = 10 + 2 * logsize;

        let mat = make_sparse_matrix(size as usize, density as usize);
        let start = Instant::now();
        let det = mat.detp4([p, p2, p3, p4]);
        eprintln!(
            "size={size} density={density} sparse det {:.3}s = {det:?}",
            start.elapsed().as_secs_f64()
        );

        if size > 3000 {
            continue;
        }
        // Compute dense determinant
        let mut m = GFpEchelonBuilder::new(p);
        let mat = make_test_sparsemat(size as usize, 0, density as usize);
        let start = Instant::now();
        let mut row = vec![0; size as usize];
        // We use a transposed matrix, it's fine.
        for c in &mat.cols {
            row.fill(0);
            for &j in c {
                row[j] = if j % 12 == 1 {
                    2
                } else if j % 2 == 0 {
                    1
                } else {
                    -1
                };
            }
            m.add(&row);
        }
        let det = m.det();
        eprintln!(
            "size={size} density={density} dense det {:.3}s = {det}",
            start.elapsed().as_secs_f64()
        );
    }
}

pub fn make_sparse_matrix(n: usize, weight: usize) -> SparseMat {
    let mat = make_test_sparsemat(n, 0, weight);
    let mut rows = vec![];
    for c in &mat.cols {
        let mut c = c.clone();
        c.sort();
        c.dedup();
        let mut row = vec![];
        for j in c {
            let e = if j % 12 == 1 {
                2
            } else if j % 2 == 0 {
                1
            } else {
                -1
            };
            row.push((j as u32, e));
        }
        rows.push(row);
    }
    SparseMat::new(rows)
}
