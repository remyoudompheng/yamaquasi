use brunch::Bench;
use yamaquasi::matrix::{kernel, make_test_matrix, make_test_matrix_sparse};

brunch::benches! {
    {
        let (mat, _) = make_test_matrix(500);
        Bench::new("kernel(matrix 1000x1000)")
        .run_seeded(mat, |mat| kernel(mat).pop().unwrap())
    },
    {
        let (mat, _) = make_test_matrix(2000);
        Bench::new("kernel(matrix 4000x4000)")
        .run_seeded(mat, |mat| kernel(mat).pop().unwrap())
    },
    {
        let mat = make_test_matrix_sparse(1000, 10, 16);
        Bench::new("kernel(sparse size 1000, 16 per vec)")
        .run_seeded(mat, |mat| kernel(mat).pop().unwrap())
    },
}
