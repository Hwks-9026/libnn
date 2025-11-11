use crate::module::Module;
use crate::tensor::Tensor;
use ndarray::ArrayD;
use ndarray::IxDyn;

pub struct Dense {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights_data = ArrayD::from_shape_fn(IxDyn(&[input_size, output_size]), |_| (rand::random::<f32>() - 0.5) * 0.1);
        let bias_data = ArrayD::zeros(IxDyn(&[1, output_size]));

        Self {
            weights: Tensor::leaf(weights_data),
            bias: Tensor::leaf(bias_data),
        }
    }
}

impl Module for Dense {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.matmul(&self.weights).add(&self.bias)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self { Self }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        // It has no learnable Tensors, so return an empty vec
        vec![]
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut current_tensor = input.clone();
        for layer in &self.layers {
            current_tensor = layer.forward(&current_tensor);
        }
        current_tensor
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
