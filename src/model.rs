use crate::tensor::Tensor;

pub struct LinearModel {
    pub weights: Tensor,
}

impl LinearModel {
    pub fn new(dim: usize) -> Self {
        LinearModel {
            weights: Tensor::new(vec![0.0; dim]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> f64 {
        self.weights
            .data
            .iter()
            .zip(&x.data)
            .map(|(w, xi)| w * xi)
            .sum()
    }

    pub fn grad(&self, x: &Tensor, y: f64) -> Tensor {
        let pred = self.forward(x);
        let error = pred - y;
        Tensor::new(x.data.iter().map(|xi| 2.0 * error * xi).collect())
    }
}
