//! Relations describe an equation:
//! x^2 = product(pi^ki) mod n
//!
//! where pi = -1 or a prime in the factor base
//!
//!

use std::collections::HashMap;

use bitvec_simd::BitVec;
use num_integer::Integer;
use num_traits::One;

use crate::arith::pow_mod;
use crate::matrix;
use crate::{Int, Uint};

#[derive(Clone, Debug)]
pub struct Relation {
    pub x: Uint,
    pub cofactor: u64,
    pub factors: Vec<(i64, u64)>, // -1 for the sign
}

impl Relation {
    /// Combine 2 relations sharing a cofactor.
    pub fn combine(&self, rhs: &Relation, n: &Uint) -> Relation {
        assert_eq!(self.cofactor, rhs.cofactor);
        let mut exps = HashMap::<i64, u64>::new();
        for (p, k) in &self.factors {
            let e = exps.get(&p).unwrap_or(&0);
            exps.insert(*p, e + k);
        }
        for (p, k) in &rhs.factors {
            let e = exps.get(&p).unwrap_or(&0);
            exps.insert(*p, e + k);
        }
        exps.insert(self.cofactor as i64, 2);
        Relation {
            x: (self.x * rhs.x) % n,
            cofactor: 1,
            factors: exps.into_iter().collect(),
        }
    }

    pub fn verify(&self, n: Uint) -> bool {
        let mut prod = Uint::one();
        for &(p, k) in self.factors.iter() {
            if p == -1 && k % 2 == 1 {
                prod = n - prod;
            } else {
                prod = (prod * pow_mod(Uint::from(p as u64), Uint::from(k), n)) % n;
            }
        }
        (self.x * self.x) % n == prod
    }
}

pub fn combine_large_relation(
    h: &mut HashMap<u64, Relation>,
    r: &Relation,
    n: &Uint,
) -> Option<Relation> {
    let c = r.cofactor;
    if let Some(r0) = h.get(&c) {
        Some(r.combine(r0, &n))
    } else {
        h.insert(c, r.clone());
        None
    }
}

pub fn relation_gap(n: Uint, rels: &[Relation]) -> usize {
    let mut occs = HashMap::<i64, u64>::new();
    for r in rels {
        for (f, k) in r.factors.iter() {
            if k % 2 == 1 {
                let c = occs.get(&f).unwrap_or(&0);
                occs.insert(*f, c + 1);
            }
        }
    }
    if occs.len() > rels.len() {
        occs.len() - rels.len()
    } else {
        0
    }
}

pub fn final_step(n: Uint, rels: &[Relation]) {
    for r in rels {
        debug_assert!(r.verify(n));
    }
    // Collect occurrences
    let mut occs = HashMap::<i64, u64>::new();
    for r in rels {
        for (f, k) in r.factors.iter() {
            if k % 2 == 1 {
                let c = occs.get(&f).unwrap_or(&0);
                occs.insert(*f, c + 1);
            }
        }
    }
    eprintln!("Input {} relations {} factors", rels.len(), occs.len());
    // Sort factors by increasing occurrences
    let mut occs: Vec<(i64, u64)> = occs.into_iter().filter(|&(_, k)| k > 1).collect();
    occs.sort_by_key(|&(_, k)| k);
    let nfactors = occs.len();
    let mut idxs = HashMap::<i64, usize>::new();
    for (idx, (f, _)) in occs.into_iter().enumerate() {
        idxs.insert(f, idx);
    }
    // Build vectors
    // ridx[i] = j if rels[j] is the i-th vector in the matrix
    let mut filt_rels: Vec<&Relation> = vec![];
    let mut matrix = vec![];
    let size = idxs.len();
    let mut coeffs = 0;
    'skiprel: for r in rels.iter() {
        let mut v = vec![];
        for (f, k) in r.factors.iter() {
            if k % 2 == 0 {
                continue;
            }
            if let Some(&idx) = idxs.get(&f) {
                v.push(idx);
                coeffs += 1;
            } else {
                continue 'skiprel;
            }
        }
        filt_rels.push(r);
        matrix.push(v);
    }
    eprintln!(
        "Filtered {} relations {} factors",
        filt_rels.len(),
        nfactors
    );
    eprintln!(
        "Build matrix {}x{} ({:.1} entries/col)",
        size,
        matrix.len(),
        coeffs as f64 / size as f64
    );
    let k = if size > 5000 {
        // Block Lanczos
        let mat = matrix::SparseMat {
            k: size,
            cols: matrix,
        };
        matrix::kernel_lanczos(&mat, true)
    } else {
        let mut dense = vec![];
        for col in matrix {
            let mut v = BitVec::zeros(size);
            for idx in col {
                v.set(idx, true);
            }
            dense.push(v);
        }
        matrix::kernel_gauss(dense)
    };
    eprintln!("Found kernel of dimension {}", k.len());
    for eq in k {
        let mut rs = vec![];
        for i in eq.into_usizes().into_iter() {
            rs.push(filt_rels[i].clone());
        }
        eprintln!("Combine {} relations...", rs.len());
        let (a, b) = combine(n, &rs);
        eprintln!("Same square mod N: {} {}", a, b);
        if let Some((p, q)) = try_factor(n, a, b) {
            eprintln!("Found factors!\n{}\n{}", p, q);
            break;
        }
    }
}

/// Combine relations into an identity a^2 = b^2
fn combine(n: Uint, rels: &[Relation]) -> (Uint, Uint) {
    // Check that the product is a square
    let mut a = Uint::one();
    for r in rels {
        a = (a * r.x) % n;
    }
    // Collect exponents
    let mut exps = HashMap::<i64, u64>::new();
    for r in rels {
        for (p, k) in &r.factors {
            let e = exps.get(&p).unwrap_or(&0);
            exps.insert(*p, e + k);
        }
    }
    let mut b = Uint::one();
    for (p, k) in exps.into_iter() {
        assert_eq!(k % 2, 0);
        if p == -1 {
            continue;
        }
        b = (b * pow_mod(Uint::from(p as u64), Uint::from(k / 2), n)) % n;
    }
    assert_eq!((a * a) % n, (b * b) % n);
    (a, b)
}

/// Using a^2 = b^2 mod n, try to factor n
fn try_factor(n: Uint, a: Uint, b: Uint) -> Option<(Uint, Uint)> {
    if a == b || a + b == n {
        // Trivial square relation
        return None;
    }
    let e = Integer::extended_gcd(&Int::from_bits(n), &Int::from_bits(a + b));
    if e.gcd > Int::one() {
        let p = e.gcd.to_bits();
        let q = n / p;
        assert!(p * q == n);
        assert!(p.bits() > 1 && q.bits() > 1);
        return Some((p, q));
    }
    let e = Integer::extended_gcd(&Int::from_bits(n), &Int::from_bits(n + a - b));
    if e.gcd > Int::one() {
        let p = e.gcd.to_bits();
        let q = n / p;
        assert!(p * q == n);
        assert!(p.bits() > 1 && q.bits() > 1);
        return Some((p, q));
    }
    None
}
