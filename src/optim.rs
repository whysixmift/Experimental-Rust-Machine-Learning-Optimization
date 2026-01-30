use crate::tensor::Tensor;

pub struct SGD {
    pub lr: f64,
}

impl SGD {
    pub fn step(&self, params: &mut Tensor, grads: &Tensor) {
        let update = grads.mul_scalar(self.lr);
        *params = params.sub(&update);
    }
}
