mod data;
mod model;
mod optim;
mod tensor;

use data::dataset;
use model::LinearModel;
use optim::SGD;

fn main() {
    let mut model = LinearModel::new(2);
    let opt = SGD { lr: 0.01 };
    let data = dataset();

    for epoch in 0..1000 {
        let mut loss = 0.0;
        let mut grad_sum = vec![0.0, 0.0];

        for (x, y) in &data {
            let pred = model.forward(x);
            loss += (pred - y).powi(2);

            let grad = model.grad(x, *y);
            for i in 0..2 {
                grad_sum[i] += grad.data[i];
            }
        }

        let grad_tensor = tensor::Tensor::new(grad_sum);
        opt.step(&mut model.weights, &grad_tensor);

        if epoch % 100 == 0 {
            println!("epoch {} loss {}", epoch, loss);
        }
    }

    println!("trained weights = {:?}", model.weights.data);
}