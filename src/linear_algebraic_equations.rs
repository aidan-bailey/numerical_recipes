use ndarray::Array2;
use num_traits::Float;

/// Perform Gauss-Jordan elimination on the given augmented matrix.
///
/// # Arguments
///
/// * `a` - coefficient matrix
/// * `b` - solution matrix
///
/// # Returns
/// A tuple containing the matrix inverse of `a` and the corresponding set of solution vectors `b`.
pub fn gaussj<T: Float>(mut a: Array2<T>, mut b: Array2<T>) -> (Array2<T>, Array2<T>) {
    let n = a.nrows();
    let m = b.ncols();

    let mut indxc: Vec<usize> = vec![0; n];
    let mut indxr: Vec<usize> = vec![0; n];
    let mut ipiv: Vec<usize> = vec![0; n];

    let mut icol = 0;
    let mut irow = 0;

    for i in 0..n {
        let mut big = T::zero();

        for j in 0..n {
            if ipiv[j] == 1 {
                continue;
            }

            for k in 0..n {
                if ipiv[k] != 0 {
                    continue;
                }

                let absval = a[(j, k)].abs();

                if (absval) >= big {
                    big = absval;
                    irow = j;
                    icol = k;
                }
            }
        }

        ipiv[icol] += 1;

        if irow != icol {
            for l in 0..n {
                a.swap((irow, l), (icol, l))
            }
            for l in 0..m {
                b.swap((irow, l), (icol, l))
            }
        }

        indxr[i] = irow;
        indxc[i] = icol;

        if a[(icol, icol)].is_zero() {
            panic!("gaussj: Singular Matrix");
        }

        let pivinv: T = T::one() / a[(icol, icol)];

        a[(icol, icol)].set_one();

        for l in 0..n {
            a[(icol, l)] = a[(icol, l)] * pivinv;
        }

        for l in 0..m {
            b[(icol, l)] = b[(icol, l)] * pivinv;
        }

        for ll in 0..n {
            if ll == icol {
                continue;
            }

            let dum: T = a[(ll, icol)];
            a[(ll, icol)].set_zero();

            for l in 0..n {
                a[(ll, l)] = a[(ll, l)] - a[(icol, l)] * dum;
            }

            for l in 0..m {
                b[(ll, l)] = b[(ll, l)] - b[(icol, l)] * dum;
            }
        }

        for l in (0..n).rev() {
            if indxr[l] == indxc[l] {
                continue;
            }

            for k in 0..n {
                a.swap((k, indxr[l]), (k, indxc[l]))
            }
        }
    }

    (a, b)
}
