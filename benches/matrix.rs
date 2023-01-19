use brunch::Bench;
use rand::{self, Fill};
use yamaquasi::{matrix, Verbosity};

fn main() {
    let mut rng = rand::thread_rng();
    let mut b1 = matrix::Block::new(8000);
    let mut b2 = matrix::Block::new(8000);
    b1.try_fill(&mut rng).unwrap();
    b2.try_fill(&mut rng).unwrap();
    let mut mat = matrix::SmallMat::default();
    loop {
        mat.try_fill(&mut rng).unwrap();
        if mat.inverse().is_some() {
            break;
        }
    }
    let mut mat2 = matrix::SmallMat::default();
    mat2.try_fill(&mut rng).unwrap();

    brunch::benches! {
        inline:
        // Block Lanczos subroutines
        {
            Bench::new("(block 64x8000) x (block 8000x64)")
            .run_seeded((&b1, &b2), |(b1, b2)| (b1 * b2) as matrix::SmallMat)
        },
        {
            Bench::new("(block 8000x64) x (matrix 64x64)")
            .run_seeded((&b1, &mat), |(b, m)| (b * m) as matrix::Block)
        },
        {
            let mat = matrix::make_test_sparsemat(8000, 10, 20);
            let mut b = matrix::Block::new(8010);
            b.try_fill(&mut rng).unwrap();
            Bench::new("(sparse 8000x8010) x (block 8010)")
            .run_seeded((&mat, &b), |(m, b)| (m * b) as matrix::Block)
        },
        {
            let mat = matrix::make_test_sparsemat(8000, 10, 20);
            let mut b = matrix::Block::new(8010);
            b.try_fill(&mut rng).unwrap();
            Bench::new("(sparse 8000x8010)^2 x (block 8010)")
            .run_seeded((&mat, &b), |(m, b)| matrix::mul_aab(m, b) as matrix::Block)
        },
        {
            Bench::new("rank (64x64)")
            .run_seeded(&mat, |m| m.rank())
        },
        {
            Bench::new("inverse (64x64)")
            .run_seeded(&mat, |m| m.inverse().unwrap() as matrix::SmallMat)
        },
        {
            Bench::new("(64x64) x (64x64)")
            .run_seeded((&mat, &mat2), |(m1, m2)| (m1 * m2) as matrix::SmallMat)
        },
        {
            let mat = matrix::make_test_sparsemat(8000, 10, 20);
            Bench::new("gen lanczos block(sparse 8000x8010)")
            .run_seeded(&mat, |m| matrix::genblock(m) as matrix::Block)
        },
        // Block Lanczos
              {
            let mat = matrix::make_test_sparsemat(500, 10, 20);
            Bench::new("lanczos(size 500, 20/row)")
            .run_seeded(&mat, |mat| matrix::kernel_lanczos(mat, Verbosity::Silent).pop().unwrap())
        },
        {
            let mat = matrix::make_test_sparsemat(1000, 10, 20);
            Bench::new("lanczos(size 1000, 20/row)")
            .run_seeded(&mat, |mat| matrix::kernel_lanczos(mat, Verbosity::Silent).pop().unwrap())
        },
        // Gauss elimination
        {
            let mat = matrix::make_test_matrix_sparse(500, 10, 20);
            Bench::new("kernel(matrix 1000x1000)")
            .run_seeded(mat, |mat| matrix::kernel_gauss(mat).pop().unwrap())
        },
        {
            let mat = matrix::make_test_matrix_sparse(1000, 10, 20);
            Bench::new("kernel(sparse size 1000, 16 per vec)")
            .run_seeded(mat, |mat| matrix::kernel_gauss(mat).pop().unwrap())
        },
    }
}
