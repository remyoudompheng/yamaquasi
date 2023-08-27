// Copyright 2022, 2023 Rémy Oudompheng. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Linear algebra algorithms for kernels of matrices modulo 2.
//!
//! The purpose of this module is to compute kernels of large matrices over GF(2).
//! These matrices represent the parity of exponents in the factorization of
//! pseudorandom integers so their coefficients are not evenly distributed:
//! a typical matrix coming from the quadratic sieve has extremely high density
//! on the first columns (corresponding to smallest primes or factors of A
//! in the case of SIQS) and decreasing density for larger primes.
//!
//! Numerical simulations show that more than half of coefficients
//! (or 40% for input integers over 256 bits) is concentrated on the densest
//! 64 columns (even for the largest matrices of size ~400000).
//!
//! In this module the kernel is meant as a right kernel, so the matrix
//! density "gradient" follows the rows (downward) rather than the columns.
//!
//! The block Lanczos implementation uses system randomness and is thus
//! non-deterministic.
//!
//! Relevant bibliography
//! A Block Lanczos Algorithm for Finding Dependencies over GF(2)
//! Peter L. Montgomery, <https://doi.org/10.1007/3-540-49264-X_9>
//!
//! Charles Bouillaguet, Paul Zimmermann.
//! Parallel Structured Gaussian Elimination for the Number Field Sieve.
//! Mathematical Cryptology, 2021, 1, pp.22-39.
//! https://hal.inria.fr/hal-02098114v2/document

// Note that crate bitvec 1.0 generates slow code for our purpose.

use std::default::Default;
use std::num::Wrapping;
use std::ops::Mul;

use bitvec_simd::BitVec;
use rand::{self, Fill};
use wide;

use crate::Verbosity;

/// Gauss reduction and kernels of matrices modulo 2
///
/// Matrices are represented as vectors of dense bit vectors
/// A size 20000 matrix will use 50MB of memory.
/// A size 40000 matrix will use 200MB of memory.
/// A size 100000 matrix will use 1.25GB of memory.
///
/// Given a list of m columns of n bits, return a list
/// of bit vectors (size m) generating the kernel of the matrix.
pub fn kernel_gauss(columns: Vec<BitVec>) -> Vec<BitVec> {
    let size = columns[0].len();
    let ncols = columns.len();
    assert!(columns.iter().all(|v| v.len() == size));
    // Auxiliary matrix
    let mut coefs = vec![];
    for i in 0..ncols {
        let mut r = BitVec::zeros(ncols);
        r.set(i, true);
        coefs.push(r);
    }
    let mut zeros: Vec<usize> = columns.iter().map(|c| c.leading_zeros()).collect();
    // Make matrix triangular.
    let mut done: usize = 0;
    let mut cols = columns;
    while done < ncols {
        debug_assert!(
            &zeros[..done].iter().max().unwrap_or(&0)
                <= &zeros[done..].iter().min().unwrap_or(&size)
        );
        debug_assert!(zeros[done] == cols[done].leading_zeros());
        // Find longest columns
        // Invariant: zeros[done..] >= zeros[..done]
        let i = (done..ncols).min_by_key(|&j| zeros[j]).unwrap();
        if zeros[i] == size {
            return (coefs[done..]).to_vec();
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
    if ncols > 0 && zeros[ncols - 1] == size {
        return vec![coefs.swap_remove(ncols - 1)];
    }
    // Reached the end but no vector is null.
    vec![]
}

/// Block Lanczos algorithm for kernel of sparse matrices.
///
/// Computation involves:
/// - Dot product of blocks of vectors (a few ms)
/// - Product of a small matrix by a block (a few ms)
/// - Product of a sparse matrix by a block (<1ms)
/// - Inverse of a small matrix
pub fn kernel_lanczos(b: &SparseMat, verbose: Verbosity) -> Vec<BitVec> {
    // Consider the quadratic form defined by A = b^T b
    // Decompose the entire space in blocks such that
    // A is block-diagonal over these blocks.
    // Compute kernel by X = preimage(AY) => A(X-Y) = 0

    // Optimize matrix for sparse multiply.
    let b = &qs_optimize(b);
    let mul_aab = mul_aab_opt;

    // At each step:
    // Wk = B0 + ... + Bk
    // A W[k-1] inside W[k]
    // A W[k] inside Wk + ABk
    // => use A Bk for the next block

    let mut vs: Vec<Block> = vec![];
    // W[i] a masked subblock of V[i]
    let mut ws: Vec<Block> = vec![];
    // Inverse Gram matrix of W[i]
    let mut invgs: Vec<SmallMat> = vec![];
    // How many vectors from vs[i] are kept at current step.
    let mut masks: Vec<Lane> = vec![];

    // Generate a random block with full rank (no need to mask).
    let mut y = genblock(b);
    // First block, use AY
    let ay = mul_aab(b, &y);
    {
        let bay = b * &ay;
        let g = &bay * &bay;
        let ginv = g.inverse().unwrap();
        // Project on block 1.
        // Use block to reduce Y => Y - W (W^TAW)^-1 W^T (AY)
        let coef = &ginv * &(&ay * &ay);
        y.muladd(&coef, &ay);

        vs.push(ay.clone());
        ws.push(ay.clone());
        invgs.push(ginv);
        masks.push(!0);
    }
    if verbose >= Verbosity::Info {
        eprintln!("[Lanczos] Selected block 1 with rank {}", LSIZE);
    }
    loop {
        assert_eq!(vs.len(), ws.len());
        // Use A * previous W + previous V
        let mut next = mul_aab(b, ws.last().unwrap());
        let prev = vs.last().unwrap();
        for i in 0..next.0.len() {
            next.0[i] ^= prev.0[i];
        }
        // Make it orthogonal to all previous blocks: [Montgomery, Section 5, 6]
        // We are computing V[i+1] using AW[i] + V[i]
        // Subtract W[j] with coefficient <Wj|A|Wj>^-1 (<Wj|A^2|Wi> + <Wj|A|Vi>)
        // By construction V[i] is orthogonal to W[j < i]
        // <Wj|A^2|Wi> = <AWj|A|Wi> = 0 if AWj has been consumed in Vj for j < i
        //
        // Compute <next, b> / <b, b> for each previous block
        let av = mul_aab(b, &next);
        let mut projs = 0;
        for j in 0..ws.len() {
            if ws[j].0.is_empty() {
                continue;
            }
            let mut mask = !0;
            for k in j + 2..vs.len() {
                mask &= masks[k - 1];
            }
            if mask == 0 {
                // AWj has been consumed in V[j+1..i-1] no need to compute.
                // Free memory: the block will no longer be used.
                debug_assert!(&ws[j] * &av == SmallMat::default());
                vs[j].0 = vec![];
                ws[j].0 = vec![];
                continue;
            }
            //eprintln!("[Lanczos] Block {} projection to block {}", vs.len()+1, j+1);
            let w = &ws[j];
            let coef = &invgs[j] * &(w * &av);
            next.muladd(&coef, w);
            debug_assert!(&mul_aab(b, w) * &next == SmallMat::default());
            projs += 1;
        }
        // Compute a non-degenerate subblock.
        let bv = b * &next;
        let gram = &bv * &bv;
        // For every 2nd block we compute rank in reverse
        // to avoid masking the same vector twice in a row.
        let reverse = vs.len() % 2 == 1;
        let (rk, mask) = if !reverse {
            gram.rank()
        } else {
            gram.rank_reverse()
        };
        if rk == 0 {
            // Lanczos iterations are finished, return kernel.
            if verbose >= Verbosity::Info {
                eprintln!(
                    "[Lanczos] Space is exhausted after {} blocks of size {}",
                    vs.len(),
                    LSIZE
                );
            }
            break;
        } else {
            vs.push(next.clone());
            if verbose >= Verbosity::Verbose && vs.len() % 16 == 0 {
                eprintln!(
                    "[Lanczos] Found block {} rank {} ({} projections)",
                    vs.len(),
                    rk,
                    projs
                );
            }
        }
        for v in &mut next.0 {
            *v &= mask;
        }
        let w = next;
        let ginv = gram.mask(mask).pseudoinverse();
        debug_assert!(ginv.rank() == (rk, mask));
        // Use block to reduce Y => Y - W (W^TAW)^-1 W^T (AY)
        let coef = &ginv * &(&w * &ay);
        y.muladd(&coef, &w);
        debug_assert!(&w * &mul_aab(b, &y) == SmallMat::default());
        ws.push(w);
        invgs.push(ginv);
        masks.push(!mask);
    }
    // Check that Y is orthogonal to all blocks
    for w in &ws {
        // Only if block has not been purged
        if w.0.len() > 0 {
            debug_assert!(w * &mul_aab(b, &y) == SmallMat::default());
        }
    }
    // Y is orthogonal to all blocks, its image is contained
    // in the (small, possibly null) final block.
    //
    // Compute an actual kernel: compute the kernel of BY as K, return YK.
    let by = b * &y;
    let mut by_bits = vec![];
    for _ in 0..LSIZE {
        by_bits.push(BitVec::zeros(by.0.len()));
    }
    for (i, l) in by.0.iter().enumerate() {
        for j in 0..LSIZE {
            if (l >> j) & 1 != 0 {
                by_bits[j].set(i, true);
            }
        }
    }
    let ker = kernel_gauss(by_bits);
    let dimker = ker.len();
    if verbose >= Verbosity::Info {
        eprintln!("[Lanczos] found kernel subspace of rank <= {}", ker.len());
    }
    // For each actual kernel basis, turn into a bit vector.
    // Beware, if a basis element is in the kernel of Y,
    // it will be a null column of YK.
    let mut basis = vec![];
    for _ in 0..dimker {
        basis.push(BitVec::zeros(y.0.len()));
    }
    // basis[i,j] = sum y[i,_] * ker[_,j]
    for (i, l) in y.0.into_iter().enumerate() {
        for (j, k) in ker.iter().enumerate() {
            debug_assert!(k.len() == LSIZE);
            let k: *const wide::u64x4 = k.as_ptr();
            let lk: Lane = l & unsafe { *(k as *const Lane) };
            basis[j].set(i, lk.count_ones() % 2 == 1);
        }
    }
    // Pop any resulting null vector.
    for i in 0..dimker {
        let i = dimker - 1 - i;
        if basis[i].none() {
            basis.swap_remove(i);
        }
    }
    if verbose >= Verbosity::Info {
        eprintln!("[Lanczos] final kernel rank <= {}", basis.len());
    }
    basis
}

/// Compute A^T A B, which is a block of the same
/// shape as b.
pub fn mul_aab(a: &SparseMat, b: &Block) -> Block {
    assert_eq!(a.cols.len(), b.0.len());
    // Output[k,*] = sum a[i,k] a[i,j] b[j,*]
    let tmp: Block = a * b;
    let mut out = Block::new(b.0.len());
    for (k, col) in a.cols.iter().enumerate() {
        for &i in col {
            out.0[k] ^= tmp.0[i];
        }
    }
    out
}

pub fn mul_aab_opt(a: &SparseMatOpt, b: &Block) -> Block {
    // Output[k,*] = sum a[i,k] a[i,j] b[j,*]
    let tmp: Block = a * b;
    // Dense part (i < LSIZE)
    // The contribution is a.block * tmp[:LSIZE]
    let tmp_dense = SmallMat(tmp.0[..LSIZE].try_into().unwrap());
    let mut out = &a.block * &tmp_dense;
    // Sparse part
    // This is a.xy.transposed * tmp[LSIZE:]
    for &(i, k) in &a.xy {
        out.0[k as usize] ^= tmp.0[i as usize];
    }
    out
}

pub fn genblock(b: &SparseMatOpt) -> Block {
    // Input matrix B.
    // Generate block Y such that Gram(B A Y) has full rank
    // This is to avoid null rows in the Gram matrix of AY
    let mut rng = rand::thread_rng();
    let mut y = Block::new(b.ny);
    loop {
        y.try_fill(&mut rng).unwrap();
        let ay: Block = mul_aab_opt(b, &y);
        let bay = b * &ay;
        let gram: SmallMat = &bay * &bay;
        if gram.rank().0 == LSIZE {
            return y;
        }
    }
}

type Lane = u64;
const LSIZE: usize = Lane::BITS as usize;

/// A square (often symmetric) matrix of size BxB (B=256)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmallMat([Lane; LSIZE]);

impl Default for SmallMat {
    fn default() -> Self {
        SmallMat([Default::default(); LSIZE])
    }
}

/// A matrix of size K rows x N columns
pub struct SparseMat {
    pub k: usize,
    pub cols: Vec<Vec<usize>>,
}

/// Sparse GF(2) matrix represented as a coordinate list and a dense part.
pub struct SparseMatOpt {
    pub nx: usize,
    pub ny: usize,
    /// A dense block corresponding to rows 0..LSIZE
    pub block: Block,
    /// List of (row, col) coordinates.
    pub xy: Vec<(u32, u32)>,
}

// Sort a sparse matrix into a list of coordinates adapted
// for efficient multiplication.
// Since QSieve matrices are concentrated in low row indices
// sort by blocks in row major order:
//
//  1    2    3 ...
//  4    5    6 ...
pub fn qs_optimize(mat: &SparseMat) -> SparseMatOpt {
    const BLOCK_SIZE: u64 = 64;
    let mut coords = vec![];
    let mut dense = Block::new(mat.cols.len());
    for (j, col) in mat.cols.iter().enumerate() {
        for &i in col {
            if i < LSIZE {
                dense.0[j] |= 1 << i;
            } else {
                coords.push((i as u32, j as u32));
            }
        }
    }
    coords
        .sort_unstable_by_key(|&(i, j)| ((i as u64 / BLOCK_SIZE) << 32) + (j as u64 / BLOCK_SIZE));
    SparseMatOpt {
        nx: mat.k,
        ny: mat.cols.len(),
        block: dense,
        xy: coords,
    }
}

/// A block of B vectors of length N
#[derive(Clone, PartialEq, Eq)]
pub struct Block(Vec<Lane>);

impl Block {
    pub fn new(n: usize) -> Self {
        Block(vec![Default::default(); n])
    }
}

#[inline]
fn randlane<R: rand::Rng + ?Sized>(rng: &mut R) -> Result<Lane, rand::Error> {
    let mut w = [0 as Lane; 1];
    rng.try_fill(&mut w)?;
    Ok(w[0])
}

impl rand::Fill for SmallMat {
    fn try_fill<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> Result<(), rand::Error> {
        for i in 0..LSIZE {
            self.0[i] = randlane(rng)?;
        }
        Ok(())
    }
}

impl rand::Fill for Block {
    fn try_fill<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) -> Result<(), rand::Error> {
        for i in 0..self.0.len() {
            self.0[i] = randlane(rng)?;
        }
        Ok(())
    }
}

impl Mul<&SmallMat> for &Block {
    type Output = Block;

    fn mul(self, rhs: &SmallMat) -> Block {
        // output[i,k] = sum self[i,j] * rhs[j,k]
        let mut output = Block::new(self.0.len());
        muladd(&mut output.0, &self.0, rhs);
        output
    }
}

impl Mul<&SmallMat> for &SmallMat {
    type Output = SmallMat;

    fn mul(self, rhs: &SmallMat) -> SmallMat {
        // output[i,k] = sum self[i,j] * rhs[j,k]
        let mut output = SmallMat::default();
        muladd(&mut output.0, &self.0, rhs);
        output
    }
}

/// Dot product of blocks of vectors.
impl Mul<&Block> for &Block {
    type Output = SmallMat;

    fn mul(self, rhs: &Block) -> SmallMat {
        assert_eq!(self.0.len(), rhs.0.len());
        // output[i,j] = sum self[i,k] * rhs[j,k]
        // Use the rotate trick [Montgomery, section 9]:
        // output[i, r(i)] = sum lhs[i,*] * rhs[r(i), *]
        let mut mat = SmallMat::default();
        let m = &mut mat.0;
        for r in 0..LSIZE as u32 {
            // NOTE: this loop can be auto-vectorized by LLVM when AVX2 is on.
            for i in 0..self.0.len() {
                unsafe {
                    let x = *self.0.get_unchecked(i);
                    let y = *rhs.0.get_unchecked(i);
                    let yrot = y.rotate_right(r);
                    m[r as usize] ^= x & yrot;
                }
            }
        }
        // Now m[64*r1+r2, i] = output[i, rot(r1, r2, i)]
        let mut mat = mat.transpose();
        // Now m[i, 64*r1+r2] = output[i, rot(r1, r2, i)]
        let m = &mut mat.0;
        for r in 0..LSIZE as u32 {
            // Blocks 0
            let v = m[r as usize];
            m[r as usize] = v.rotate_left(r);
        }
        mat
    }
}

impl Block {
    // Computes self -= m*b.
    fn muladd(&mut self, m: &SmallMat, b: &Block) {
        assert_eq!(self.0.len(), b.0.len());
        muladd(&mut self.0, &b.0, m);
    }
}

fn muladd(output: &mut [Lane], b: &[Lane], m: &SmallMat) {
    assert_eq!(output.len(), b.len());
    // Compute output[k,i] += sum b[k,j] * m[j,i]
    //
    // Use the rotate method again: [Montgomery, section 9]:
    // output[k,i] += sum b[k,j+(r1,r2)] * m[j+(r1,r2),i]
    // (we are using SIMD vectors so it's a "2-level" rotation)
    //
    // We need to multiply rotated b with "diagonals" of m.
    let mut diags = m.transpose();
    let p = &mut diags.0;
    for r in 0..LSIZE as u32 {
        let v = p[r as usize];
        p[r as usize] = v.rotate_right(r);
    }
    // Now diags[i,j] = m[j+i,i]
    // For a fixed j, diags[*,j] * b[k,j+*] gives a contribution to output[k,i]
    let diags = diags.transpose();
    let p = &diags.0;
    let dim = output.len();
    for j in 0..LSIZE {
        let m0 = p[j];
        // NOTE: this loop can be auto-vectorized by LLVM when AVX2 is on.
        for k in 0..dim {
            let xk = unsafe { output.get_unchecked_mut(k) };
            let bk = unsafe { *b.get_unchecked(k) };
            let bk_rot = bk.rotate_right(j as u32);
            *xk ^= m0 & bk_rot;
        }
    }
}

impl Mul<&Block> for &SparseMat {
    type Output = Block;

    fn mul(self, rhs: &Block) -> Block {
        assert_eq!(self.cols.len(), rhs.0.len());
        // output[i] = sum self[i,j] * rhs[j]
        let mut out = Block::new(self.k);
        for (j, col) in self.cols.iter().enumerate() {
            let row = &rhs.0[j];
            for &i in col {
                out.0[i] ^= row;
            }
        }
        out
    }
}

impl Mul<&Block> for &SparseMatOpt {
    type Output = Block;

    fn mul(self, rhs: &Block) -> Block {
        assert_eq!(self.ny, rhs.0.len());
        // output[i] = sum self[i,j] * rhs[j]
        let mut out = Block::new(self.nx);
        // dense part
        let dense = &self.block * rhs;
        for i in 0..LSIZE {
            out.0[i] = dense.0[i];
        }
        // sparse part
        for &(i, j) in &self.xy {
            let row = &rhs.0[j as usize];
            out.0[i as usize] ^= row;
        }
        out
    }
}

fn lz(w: Lane) -> usize {
    w.trailing_zeros() as usize
}

fn reverse_lane(l: Lane) -> Lane {
    l.reverse_bits()
}

impl SmallMat {
    fn identity() -> Self {
        let mut m = SmallMat::default();
        for i in 0..LSIZE {
            m.0[i] = 1 << i;
        }
        m
    }

    fn symmetric(&self) -> bool {
        for i in 0..LSIZE {
            for j in 0..i {
                let mij = (self.0[i] >> j) as usize & 1;
                let mji = (self.0[j] >> i) as usize & 1;
                if mij != mji {
                    return false;
                }
            }
        }
        true
    }

    fn transpose(&self) -> Self {
        let mut m = SmallMat::default();
        for i in 0..LSIZE {
            let mut row = Lane::default();
            for j in 0..LSIZE {
                let mji = (self.0[j] >> i) & 1;
                if mji == 1 {
                    row |= 1 << j;
                }
            }
            m.0[i] = Lane::from(row);
        }
        m
    }

    fn mask(&self, mask: Lane) -> SmallMat {
        let mut m = self.clone();
        for i in 0..LSIZE {
            if (mask >> i) & 1 == 0 {
                m.0[i] = 0;
            } else {
                m.0[i] &= mask;
            }
        }
        debug_assert!(!self.symmetric() || m.symmetric());
        m
    }

    // Find a submatrix M[I,I] such that rank = #I
    // See [Montgomery, Section 8]
    pub fn submatrix(&self) -> SmallMat {
        debug_assert!(self.symmetric());
        let (r, mask) = self.rank();
        let m = self.mask(mask);
        debug_assert!(m.symmetric());
        debug_assert!(m.rank() == (r, mask));
        m
    }

    // Returns the rank of self and a mask of linearly
    // independent rows.
    pub fn rank(&self) -> (usize, Lane) {
        let mut m: SmallMat = self.clone();
        let mut mask = 0 as Lane;
        // Run Gauss elimination
        let mut rank = 0;
        let mut orig_idx = [0u8; LSIZE];
        for i in 0..LSIZE {
            orig_idx[i] = i as u8;
        }
        for i in 0..LSIZE {
            let Some(j) = m.0.iter().position(|&v| lz(v) == i) else {
                continue;
            };
            let idx = orig_idx[j] as usize;
            mask |= 1 << idx;
            m.0.swap(rank, j);
            orig_idx.swap(rank, j);
            for j in (rank + 1)..LSIZE {
                if lz(m.0[j]) == i {
                    m.0[j] ^= m.0[rank];
                }
            }
            rank += 1;
        }
        debug_assert!(rank == mask.count_ones() as usize);
        (rank, mask)
    }

    /// Pseudo inverse of a small symmetric matrix.
    ///
    /// If M is a square matrix with null coefficients
    /// outside of set of indices I, return matrix N
    /// such that `(MN)[i,i] = 1` for i in I
    fn pseudoinverse(&self) -> SmallMat {
        let (rk, mask) = self.rank();
        // Augment with an identity matrix
        let mut m: SmallMat = self.clone();
        let mut minv = Self::identity();
        for i in 0..LSIZE {
            minv.0[i] &= mask;
        }
        debug_assert!(minv.rank() == self.rank());
        let mut idx = [0u8; 256];
        let mut i = 0;
        for j in 0..LSIZE {
            if (mask >> j) & 1 == 1 {
                idx[i] = j as u8;
                i += 1;
            }
        }
        // Run Gauss elimination
        for &i in &idx[..rk] {
            let i = i as usize;
            let j = m.0.iter().position(|&v| lz(v) == i).unwrap();
            m.0.swap(i, j);
            minv.0.swap(i, j);
            for j in (i + 1)..LSIZE {
                if lz(m.0[j]) == i {
                    m.0[j] ^= m.0[i];
                    minv.0[j] ^= minv.0[i];
                } else {
                    debug_assert!(lz(m.0[j]) > i);
                }
            }
        }
        // Solve triangular inverse
        for idx1 in 0..rk {
            let i = idx[rk - 1 - idx1] as usize;
            let r = m.0[i];
            debug_assert!(lz(m.0[i]) == i);
            for idx2 in 0..idx1 {
                let j = idx[rk - 1 - idx2] as usize;
                debug_assert!(i < j);
                if (r >> j) & 1 != 0 {
                    m.0[i] ^= m.0[j];
                    minv.0[i] ^= minv.0[j];
                }
            }
            let r = m.0[i];
            debug_assert!(r == 1 << i);
        }
        debug_assert!(minv.rank() == self.rank());
        minv
    }

    fn reverse(&self) -> SmallMat {
        let mut rev = SmallMat::default();
        for i in 0..LSIZE {
            rev.0[i] = reverse_lane(self.0[LSIZE - 1 - i]);
        }
        rev
    }

    fn rank_reverse(&self) -> (usize, Lane) {
        let (rk, mask) = self.reverse().rank();
        let mask = reverse_lane(mask);
        (rk, mask.into())
    }

    pub fn inverse(&self) -> Option<SmallMat> {
        // Augment with an identity matrix
        let mut m: SmallMat = self.clone();
        let mut minv = Self::identity();
        // Run Gauss elimination
        for i in 0..LSIZE {
            let Some(j) = m.0.iter().position(|&v| lz(v) == i) else {
                return None;
            };
            m.0.swap(i, j);
            minv.0.swap(i, j);
            for j in (i + 1)..LSIZE {
                if lz(m.0[j]) == i {
                    m.0[j] ^= m.0[i];
                    minv.0[j] ^= minv.0[i];
                } else {
                    debug_assert!(lz(m.0[j]) > i);
                }
            }
        }
        // Solve triangular inverse
        for i in 0..LSIZE {
            let i = LSIZE - 1 - i;
            let r = m.0[i];
            for j in (i + 1)..LSIZE {
                if (r >> j) & 1 == 1 {
                    m.0[i] ^= m.0[j];
                    minv.0[i] ^= minv.0[j];
                }
            }
            debug_assert!(m.0[i] == 1 << i);
        }
        Some(minv)
    }
}

#[cfg(test)]
fn make_bitvec(slice: &[u8]) -> BitVec {
    BitVec::from(slice.iter().map(|&n| n != 0))
}

#[test]
fn test_kernel_small() {
    // Rank 4
    let v = kernel_gauss(vec![
        make_bitvec(&[1, 0, 0, 1]),
        make_bitvec(&[0, 1, 0, 1]),
        make_bitvec(&[0, 1, 0, 0]),
        make_bitvec(&[1, 1, 1, 0]),
    ]);
    assert_eq!(v, Vec::<BitVec>::new());
    // Rank 3
    let v = kernel_gauss(vec![
        make_bitvec(&[1, 0, 0, 1]),
        make_bitvec(&[1, 0, 1, 0]),
        make_bitvec(&[1, 1, 1, 0]),
        make_bitvec(&[1, 1, 0, 1]),
    ]);
    assert_eq!(v, vec![make_bitvec(&[1, 1, 1, 1])]);
}

#[doc(hidden)]
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

#[allow(dead_code)]
#[doc(hidden)]
pub fn make_test_sparsemat(k: usize, corank: usize, ones: usize) -> SparseMat {
    let mut seed: Wrapping<u32> = Wrapping(0xcafe1337 + k as u32);
    let mut cols = vec![];
    for _ in 0..k + corank {
        let mut col = vec![];
        for _ in 0..ones {
            seed *= 0x12345;
            seed ^= 0x1337;
            col.push((seed.0 >> 8) as usize % k);
        }
        cols.push(col);
    }
    SparseMat { k, cols }
}

#[allow(dead_code)]
#[doc(hidden)]
pub fn make_test_matrix_sparse(n: usize, k: usize, ones: usize) -> Vec<BitVec> {
    let mut seed: Wrapping<u32> = Wrapping(0xcafe1337 + n as u32);
    let mut matrix = vec![];
    for _ in 0..n + k {
        let mut p = BitVec::zeros(n);
        for _ in 0..ones {
            seed *= 0x12345;
            seed ^= 0x1337;
            p.set((seed.0 >> 8) as usize % n, true);
        }
        matrix.push(p);
    }
    matrix
}

#[test]
fn test_kernel_gauss() {
    let n = 100;
    let (mat, ker) = make_test_matrix(n);
    let k = kernel_gauss(mat);
    assert_eq!(k.len(), 1);
    let k: BitVec = k.into_iter().next().unwrap();
    assert_eq!(k, ker);

    let n = 500;
    let (mat, ker) = make_test_matrix(n);
    let k = kernel_gauss(mat);
    assert_eq!(k.len(), 1);
    let k: BitVec = k.into_iter().next().unwrap();
    assert_eq!(k, ker);

    let n = 2000;
    let mat = make_test_matrix_sparse(n, 2, 16);
    let ker = kernel_gauss(mat);
    assert!(ker.len() >= 2);
}

#[test]
fn test_smallmat() {
    let mut rng = rand::thread_rng();
    // Basic ops
    let mut mat = SmallMat::default();
    mat.try_fill(&mut rng).unwrap();
    assert_eq!(mat, mat.transpose().transpose());
    // M + M^T
    let mut sym = mat.clone();
    let mat_t = mat.transpose();
    for i in 0..LSIZE {
        sym.0[i] ^= mat_t.0[i];
    }
    assert!(sym.symmetric());
    // Inverse
    let mut inverts = 0;
    for _ in 0..30 {
        let mut x = Block::new(300);
        x.try_fill(&mut rng).unwrap();
        // Small symmetric matrix
        let m: SmallMat = &x * &x;
        assert!(m.symmetric());
        let (rk, _) = m.rank();
        if let Some(minv) = m.inverse() {
            assert_eq!(&m * &minv, SmallMat::identity());
            assert_eq!(&(&m * &minv) * &m, m);
            assert_eq!(&(&m * &minv) * &minv, minv);
            inverts += 1;
        } else {
            assert!(rk < LSIZE);
            let msub = m.submatrix();
            assert_eq!(m.rank(), msub.rank());
            let minv = msub.pseudoinverse();
            assert!(minv.symmetric());
            assert_eq!(&msub * &minv, &minv * &msub);
            assert_eq!(&(&msub * &minv) * &msub, msub);
            assert_eq!(&(&msub * &minv) * &minv, minv);

            // Reverse computation.
            let (invrk, invmask) = m.rank_reverse();
            assert_eq!(rk, invrk);
            let msub = m.mask(invmask);
            assert_eq!(msub.rank(), (invrk, invmask));
            let minv = msub.pseudoinverse();
            assert!(minv.symmetric());
            assert_eq!(&msub * &minv, &minv * &msub);
            assert_eq!(&(&msub * &minv) * &msub, msub);
            assert_eq!(&(&msub * &minv) * &minv, minv);
        }
    }
    // 1/4 of matrices are invertible.
    assert!(inverts > 5);
}

#[test]
fn test_sparsemat() {
    // Test that X·A^T A X == (AX)^T (AX)
    for _ in 0..10 {
        let mat = make_test_sparsemat(1000, 10, 20);
        let mut x = Block::new(1010);
        let mut rng = rand::thread_rng();
        x.try_fill(&mut rng).unwrap();
        let ax = &mat * &x;
        assert_eq!(&ax * &ax, &x * &mul_aab(&mat, &x));
    }
}

#[test]
fn test_sparsemat_opt() {
    // Test that X·A^T A X == (AX)^T (AX)
    for _ in 0..10 {
        let mat = make_test_sparsemat(1000, 10, 20);
        let mat = qs_optimize(&mat);
        let mut x = Block::new(1010);
        let mut rng = rand::thread_rng();
        x.try_fill(&mut rng).unwrap();
        let ax = &mat * &x;
        assert_eq!(&ax * &ax, &x * &mul_aab_opt(&mat, &x));
    }
}

#[test]
fn test_projection() {
    for _ in 0..10 {
        let mut rng = rand::thread_rng();
        let mut x = Block::new(1000);
        x.try_fill(&mut rng).unwrap();
        // Generate block, mask extra vectors
        let mut y = Block::new(1000);
        y.try_fill(&mut rng).unwrap();
        let g = &y * &y;
        let (_, mask) = g.rank();
        for v in &mut y.0 {
            *v &= mask;
        }
        assert_eq!(g.submatrix(), &y * &y);
        // Compute projection
        let g = g.submatrix();
        let ginv = g.pseudoinverse();
        // <x - y (yy)^-1 (yx), y>
        let c = &ginv * &(&y * &x);
        x.muladd(&c, &y);
        assert_eq!(&x * &y, SmallMat::default());
    }
}

#[test]
fn test_lanczos() {
    // Avoid long running time in non-release mode.
    const N: usize = 1_000;
    let mat = make_test_sparsemat(N, 10, 20);
    eprintln!("Matrix size {}x{}", mat.k, mat.cols.len());
    let ker = kernel_lanczos(&mat, Verbosity::Silent);
    eprintln!("Kernel rank {}", ker.len());
    for (i, v) in ker.into_iter().enumerate() {
        // Vector is non zero and in kernel
        assert!(v.any());
        let mut w = BitVec::zeros(N);
        for &i in v.into_usizes().iter() {
            for &j in mat.cols[i].iter() {
                w.set(j, !w.get_unchecked(j));
            }
        }
        if !w.none() {
            eprintln!("Kernel element {i} not in kernel!");
        }
    }
}
