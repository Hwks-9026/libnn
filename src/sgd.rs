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
            let grad = p.gradient.borrow();
            let mut data = p.data.borrow_mut();

            // Update rule: data = data - lr * grad
            data.scaled_add(-self.learning_rate, &*grad);
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.parameters {
            p.gradient.borrow_mut().fill(0.0);
        }
    }
}
