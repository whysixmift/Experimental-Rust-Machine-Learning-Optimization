use crate::tensor::Tensor;

pub fn dataset() -> Vec<(Tensor, f64)> {
    vec![
        (Tensor::new(vec![1.0, 2.0]), 5.0),
        (Tensor::new(vec![2.0, 1.0]), 4.0),
        (Tensor::new(vec![3.0, 3.0]), 9.0),
    ]
}
