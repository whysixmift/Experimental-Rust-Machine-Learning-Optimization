#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn new(data: Vec<f64>) -> Self {
        Tensor { data }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        Tensor {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        Tensor {
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    pub fn mul_scalar(&self, lr: f64) -> Tensor {
        Tensor {
            data: self.data.iter().map(|x| x * lr).collect(),
        }
    }
}
