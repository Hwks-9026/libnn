use crate::tensor::Tensor;

pub trait Module {

    //perform forward pass
    fn forward(&self, input: &Tensor) -> Tensor;

    //return a flat list of all learnable parameters in the module and sub-modules
    fn parameters(&self) -> Vec<Tensor>;

    fn zero_gradients(&self) {
        for p in self.parameters() {
            p.gradient.borrow_mut().fill(0.0)
        }
    }
}

