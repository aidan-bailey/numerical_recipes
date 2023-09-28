pub mod linear_algebraic_equations;

#[cfg(test)]
mod linear_algebraic_equations_tests {

    use crate::linear_algebraic_equations::gaussj;
    use ndarray::arr2;

    #[test]
    fn gaussj_test() {
        let a = arr2(&[[2., 1., 3.], [15., 2., 0.], [1., 3., 1.]]);
        let b = arr2(&[[10.], [5.], [3.]]);
        let expa = arr2(&[
            [4. / 59., 1. / 59., -3. / 59.],
            [-1. / 118., -15. / 118., 45. / 118.],
            [-5. / 118., 43. / 118., -11. / 118.],
        ]);
        //let expb = arr2(&[[4 / 59], [1.]]);
        let (a, _) = gaussj(a, b);
        assert!(a.relative_eq(&expa, f64::EPSILON, 1.));
    }
}
