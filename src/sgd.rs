use ndarray::LinalgScalar;
use crate::tensor::Tensor;

//stochastic gradient decent
pub struct SGD {
    parameters: Vec<Tensor>,
    learning_rate: f32,
}

impl SGD {
    pub fn new(parameters: Vec<Tensor>, learning_rate: f32) -> Self {
        Self { parameters, learning_rate }
    }
    pub fn step(&mut self) {
        for p in &self.parameters {
            let grad = p.gradient.lock().unwrap();
            let mut data = p.data.lock().unwrap();

            // Update rule: data = data - lr * grad
            data.scaled_add(-self.learning_rate, &*grad);
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.parameters {
            p.gradient.lock().unwrap().fill(0.0);
        }
    }
}
