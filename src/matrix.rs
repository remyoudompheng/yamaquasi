//! Gauss reduction and kernels of matrices modulo 2
//!
//! Matrix are represented as vectors of dense bit vectors
//! A size 20000 matrix will use 50MB of memory.
//! A size 40000 matrix will use 200MB of memory.
//! A size 100000 matrix will use 1.25GB of memory.

// Note that crate bitvec 1.0 generates slow code for our purpose.

use bitvec_simd::BitVec;
use std::num::Wrapping;

/// Kernel of a matrix of sparse vectors.
pub fn kernel_sparse(vecs: Vec<Vec<usize>>) -> Vec<BitVec> {
    panic!("")
}

/// Given a list of m columns of n bits, return a list
/// of bit vectors (size m) generating the kernel of the matrix.
///
/// The matrix is supposed to be pre-filtered.
pub fn kernel(columns: Vec<BitVec>) -> Vec<BitVec> {
    let size = columns[0].len();
    let ncols = columns.len();
    assert!(columns.iter().all(|v| v.len() == size));
    // Auxiliary matrix
    let mut coefs = vec![];
    for i in 0..columns.len() {
        let mut r = BitVec::zeros(columns.len());
        r.set(i, true);
        coefs.push(r);
    }
    let mut zeros: Vec<usize> = columns.iter().map(|c| c.leading_zeros()).collect();
    // Make matrix triangular.
    let mut done: usize = 0;
    let mut cols = columns;
    let mut tmpc = cols[0].clone();
    let mut tmpr = coefs[0].clone();
    while done < ncols {
        assert!(
            &zeros[..done].iter().max().unwrap_or(&0)
                <= &zeros[done..].iter().min().unwrap_or(&size)
        );
        // Find longest columns
        // Invariant: zeros[done..] >= zeros[..done]
        let i = (done..ncols).min_by_key(|&j| zeros[j]).unwrap();
        if zeros[i] == size {
            return (coefs[i..]).to_vec();
        }
        // Move first
        if i > done {
            zeros.swap(i, done);
            coefs.swap(i, done);
            cols.swap(i, done);
        }
        // Eliminate
        let (cols_1, cols_2) = cols.split_at_mut(done + 1);
        let (coefs_1, coefs_2) = coefs.split_at_mut(done + 1);
        for i in done + 1..ncols {
            if zeros[i] == zeros[done] {
                cols_2[i - done - 1].xor_inplace(&cols_1[done]);
                coefs_2[i - done - 1].xor_inplace(&coefs_1[done]);
                zeros[i] = cols_2[i - done - 1].leading_zeros();
            }
        }
        done += 1;
    }
    if size > 0 && zeros[size - 1] == size {
        return vec![coefs.swap_remove(size - 1)];
    }
    // Reached the end but no vector is null.
    vec![]
}

fn make_bitvec(slice: &[u8]) -> BitVec {
    BitVec::from(slice.iter().map(|&n| n != 0))
}

#[test]
fn test_kernel_small() {
    // Rank 4
    let v = kernel(vec![
        make_bitvec(&[1, 0, 0, 1]),
        make_bitvec(&[0, 1, 0, 1]),
        make_bitvec(&[0, 1, 0, 0]),
        make_bitvec(&[1, 1, 1, 0]),
    ]);
    assert_eq!(v, Vec::<BitVec>::new());
    // Rank 3
    let v = kernel(vec![
        make_bitvec(&[1, 0, 0, 1]),
        make_bitvec(&[1, 0, 1, 0]),
        make_bitvec(&[1, 1, 1, 0]),
        make_bitvec(&[1, 1, 0, 1]),
    ]);
    assert_eq!(v, vec![make_bitvec(&[1, 1, 1, 1])]);
}

pub fn make_test_matrix(n: usize) -> (Vec<BitVec>, BitVec) {
    // Generate 2 polynomials
    // P = random bits (degree n)
    // Q = P + x^i (degree n)
    let mut seed: Wrapping<u32> = Wrapping(0xcafe1337 + n as u32);
    let mut p = BitVec::zeros(n);
    for i in 0..n {
        seed *= 0x12345;
        seed += 0x1337;
        p.set(i, (seed >> 31).0 == 1);
    }
    p.set(n - 1, true);
    let mut q = p.clone();
    q.set(50, !p[50]); // flip a bit

    let mut vecs = vec![];
    for i in 0..n {
        let mut v = BitVec::zeros(2 * n - 1);
        for j in 0..n {
            v.set(i + j, p[j]);
        }
        vecs.push(v);
    }
    for i in 0..n {
        let mut v = BitVec::zeros(2 * n - 1);
        for j in 0..n {
            v.set(i + j, q[j]);
        }
        vecs.push(v);
    }
    assert_eq!(vecs.len(), 2 * n);
    // Resulting matrix has rank 1 (size 2n x 2n-1)
    // Augmented sylvester matrix:
    // [P, .. P << n, Q, ... Q << n]
    // the kernel is exactly [Q || P]
    let mut ker = BitVec::zeros(2 * n);
    for i in 0..n {
        ker.set(i, q[i]);
        ker.set(n + i, p[i]);
    }
    (vecs, ker)
}

pub fn make_test_matrix_sparse(n: usize, k: usize, ones: usize) -> Vec<BitVec> {
    let mut seed: Wrapping<u32> = Wrapping(0xcafe1337 + n as u32);
    let mut matrix = vec![];
    for i in 0..n + k {
        let mut p = BitVec::zeros(n);
        for j in 0..ones {
            seed *= 0x12345;
            seed ^= 0x1337;
            p.set((seed.0 >> 8) as usize % n, true);
        }
        matrix.push(p);
    }
    matrix
}

#[test]
fn test_kernel() {
    let n = 100;
    let (mat, ker) = make_test_matrix(n);
    let k = kernel(mat);
    assert_eq!(k.len(), 1);
    let k: BitVec = k.into_iter().next().unwrap();
    assert_eq!(k, ker);

    let n = 500;
    let (mat, ker) = make_test_matrix(n);
    let k = kernel(mat);
    assert_eq!(k.len(), 1);
    let k: BitVec = k.into_iter().next().unwrap();
    assert_eq!(k, ker);

    let n = 5000;
    let mat = make_test_matrix_sparse(n, 2, 16);
    let ker = kernel(mat);
    assert!(ker.len() >= 2);
}
