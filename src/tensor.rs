use ndarray::{ArrayD, Axis, IxDyn};
use ndarray::arr0;
use ndarray::linalg::Dot;
use std::sync::{Arc, Mutex};
use std::collections::HashSet;

use std::ops::*;

type TensorData = Arc<Mutex<ArrayD<f32>>>;
type GradientData = Arc<Mutex<ArrayD<f32>>>;
type TensorId = *const Mutex<ArrayD<f32>>;

#[derive(Clone)]
pub struct Tensor {
    pub data: TensorData,
    pub gradient: GradientData,
    _children: Vec<Tensor>,
    
    //for back propagation
    _backward: Arc<Mutex<Option<Box<dyn Fn() + Send + Sync>>>>
}

// Constructors for Tensor
impl Tensor {
    pub fn leaf(data: ArrayD<f32>) -> Self {
        let grad = ArrayD::zeros(data.shape());
        Self {
            data: Arc::new(Mutex::new(data)),
            gradient: Arc::new(Mutex::new(grad)),
            _children: vec![],
            _backward: Arc::new(Mutex::new(None as Option<Box<dyn Fn() + Send + Sync>>)),
        }
    }   

    fn _internal_new(
        data: ArrayD<f32>,
        children: Vec<Tensor>,
        backward_op: Box<dyn Fn() + Send + Sync>,
    ) -> Self {
        let grad = ArrayD::zeros(data.shape());
        Self {
            data: Arc::new(Mutex::new(data)),
            gradient: Arc::new(Mutex::new(grad)),
            _children: children,
            _backward: Arc::new(Mutex::new(Some((backward_op) as Box<dyn Fn() + Send + Sync>))), 
        }
    }
}

// Math Operations
impl Tensor {
pub fn add(&self, rhs: &Tensor) -> Tensor {
    // 1. FORWARD PASS
    let self_data = self.data.lock().unwrap();
    let rhs_data = rhs.data.lock().unwrap();

    let result_data = &*self_data + &*rhs_data;

    // 2. PREPARE FOR BACKWARD PASS
    let self_grad_rc = self.gradient.clone();
    let rhs_grad_rc = rhs.gradient.clone();
    
    // We need the original shapes to detect broadcasting
    let self_shape = self_data.shape().to_vec();
    let rhs_shape = rhs_data.shape().to_vec();

    let c_grad_rc = Arc::new(Mutex::new(ArrayD::zeros(result_data.shape())));

    let c_grad_rc_clone = c_grad_rc.clone();
    let backward_op = move || {
        let c_grad = c_grad_rc_clone.lock().unwrap(); // This is [30, 4]

        // --- Grad for self (matmul result) ---
        // self_shape is [30, 4], c_grad is [30, 4]. Shapes match.
        if self_shape == c_grad.shape() {
            self_grad_rc.lock().unwrap().scaled_add(1.0, &*c_grad);
        } else {
            // Handle if 'self' was broadcasted (not our case, but good to have)
            let mut summed_grad = c_grad.clone();
            let axes_to_sum: Vec<_> = self_shape.iter().zip(c_grad.shape())
                .enumerate().filter(|(_, (a, b))| a < b).map(|(i, _)| i).collect();
            for &axis in axes_to_sum.iter().rev() {
                summed_grad = summed_grad.sum_axis(Axis(axis)).into_dyn();
            }
            let reshaped_grad = summed_grad.to_shape(IxDyn(&self_shape)).unwrap();
            self_grad_rc.lock().unwrap().scaled_add(1.0, &reshaped_grad);
        }

        // --- Grad for rhs (bias) ---
        // rhs_shape is [1, 4], c_grad is [30, 4]. Shapes DON'T match.
        if rhs_shape == c_grad.shape() {
            rhs_grad_rc.lock().unwrap().scaled_add(1.0, &*c_grad);
        } else {
            // THIS IS OUR FIX:
            // We sum the [30, 4] gradient along Axis(0)
            let summed_grad = c_grad.sum_axis(Axis(0)).into_dyn(); // Shape [4]

            // We reshape it from [4] to [1, 4] to match the bias's shape
            let reshaped_grad = summed_grad.to_shape(IxDyn(&rhs_shape)).unwrap();
            
            // Add the summed, reshaped gradient
            rhs_grad_rc.lock().unwrap().scaled_add(1.0, &reshaped_grad);
        }
    };

    // 3. CREATE NEW TENSOR
    // We need .to_owned() because the result of `+` is a view
    Tensor {
        data: Arc::new(Mutex::new(result_data.to_owned())),
        gradient: c_grad_rc,
        _children: vec![self.clone(), rhs.clone()],
        _backward: Arc::new(Mutex::new(Some(Box::new(backward_op) as Box<dyn Fn() + Send + Sync>))),
    }
}

    pub fn sub(self, rhs: &Tensor) -> Tensor {
        let self_data = self.data.lock().unwrap();
        let rhs_data = rhs.data.lock().unwrap();
        
        let result_data = &*self_data - &*rhs_data;
        
        let self_grad_rc = self.gradient.clone();
        let rhs_grad_rc = rhs.gradient.clone();
        let c_grad_rc = Arc::new(Mutex::new(ArrayD::zeros(result_data.shape())));
        
        let c_grad_rc_clone = c_grad_rc.clone(); // Clone for the closure
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.lock().unwrap();
            self_grad_rc.lock().unwrap().scaled_add(1.0, &*c_grad);
            rhs_grad_rc.lock().unwrap().scaled_add(-1.0, &*c_grad);
        };
        Tensor {
            data: Arc::new(Mutex::new(result_data)),
            gradient: c_grad_rc, // Assign the grad Arc
            _children: vec![self.clone(), rhs.clone()],
            _backward: Arc::new(Mutex::new(Some(Box::new(backward_op)))),
        }
    }

    pub fn mul(self, rhs: &Tensor) -> Tensor {
        let self_data_rc = self.data.clone();
        let rhs_data_rc = rhs.data.clone();
        
        let result_data = &*self.data.lock().unwrap() * &*rhs.data.lock().unwrap();

        let self_grad_rc = self.gradient.clone();
        let rhs_grad_rc = rhs.gradient.clone();
        
        let c_grad_rc = Arc::new(Mutex::new(ArrayD::zeros(result_data.shape())));
        let c_grad_rc_clone = c_grad_rc.clone();
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.lock().unwrap();
            
            // a.grad += c.grad * b.data
            let a_grad_update = &*c_grad * &*rhs_data_rc.lock().unwrap();
            self_grad_rc.lock().unwrap().scaled_add(1.0, &a_grad_update);

            // b.grad += c.grad * a.data
            let b_grad_update = &*c_grad * &*self_data_rc.lock().unwrap();
            rhs_grad_rc.lock().unwrap().scaled_add(1.0, &b_grad_update);
        };

        Tensor {
            data: Arc::new(Mutex::new(result_data)),
            gradient: c_grad_rc,
            _children: vec![self.clone(), rhs.clone()],
            _backward: Arc::new(Mutex::new(Some(Box::new(backward_op) as Box<dyn Fn() + Send + Sync>))),
        }
    }

    pub fn relu(&self) -> Tensor {
        let a_data = self.data.lock().unwrap();
        let result_data = a_data.mapv(|x| x.max(0.0));
        let positive_mask = a_data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        let a_grad_rc = self.gradient.clone();
        let c_grad_rc = Arc::new(Mutex::new(ArrayD::zeros(result_data.shape())));
        
        let c_grad_rc_clone = c_grad_rc.clone();
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.lock().unwrap();
            
            // a.grad += c.grad * mask
            let grad_update = &*c_grad * &positive_mask;
            a_grad_rc.lock().unwrap().scaled_add(1.0, &grad_update);
        };

        Tensor {
            data: Arc::new(Mutex::new(result_data)),
            gradient: c_grad_rc,
            _children: vec![self.clone()], // Only one child
            _backward: Arc::new(Mutex::new(Some(Box::new(backward_op) as Box<dyn Fn() + Send + Sync>))),
        }
    }

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {

        let self_data = self.data.lock().unwrap();
        let rhs_data = rhs.data.lock().unwrap();
    
        let result_data = &*self_data.dot(&*rhs_data);

        let self_data_rc = self.data.clone();
        let rhs_data_rc = rhs.data.clone();
        let self_grad_rc = self.gradient.clone();
        let rhs_grad_rc = rhs.gradient.clone();

        let c_grad_rc = Arc::new(Mutex::new(ArrayD::zeros(result_data.shape())));

        let c_grad_rc_clone = c_grad_rc.clone();
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.lock().unwrap();
            let self_data = self_data_rc.lock().unwrap();
            let rhs_data = rhs_data_rc.lock().unwrap();

            // a.grad += c.grad.dot(b.data.T)
            let a_grad_update = c_grad.dot(&transpose(&rhs_data));
            self_grad_rc.lock().unwrap().scaled_add(1.0, &a_grad_update);

            // b.grad += a.data.T.dot(c.grad)
            let b_grad_update = transpose(&self_data).dot(&*c_grad);
            rhs_grad_rc.lock().unwrap().scaled_add(1.0, &b_grad_update);
        };

        Tensor {
            data: Arc::new(Mutex::new(result_data.to_owned())),
            gradient: c_grad_rc,
            _children: vec![self.clone(), rhs.clone()],
            _backward: Arc::new(Mutex::new(Some(Box::new(backward_op) as Box<dyn Fn() + Send + Sync>))),
        }
    }

    pub fn powf(&self, n: f32) -> Tensor {
        let result_data = self.data.lock().unwrap().mapv(|x| x.powf(n));
        let self_data_rc = self.data.clone(); // Need original data for derivative
        let self_grad_rc = self.gradient.clone();
        let c_grad_rc = Arc::new(Mutex::new(ArrayD::zeros(result_data.shape())));

        let c_grad_rc_clone = c_grad_rc.clone();
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.lock().unwrap();
            let self_data = self_data_rc.lock().unwrap();

            // a.grad += c.grad * (n * a.data^(n-1))
            let deriv = self_data.mapv(|x| n * x.powf(n - 1.0));
            let grad_update = &*c_grad * &deriv;
            self_grad_rc.lock().unwrap().scaled_add(1.0, &grad_update);
        };

        Tensor {
            data: Arc::new(Mutex::new(result_data)),
            gradient: c_grad_rc,
            _children: vec![self.clone()],
            _backward: Arc::new(Mutex::new(Some(Box::new(backward_op) as Box<dyn Fn() + Send + Sync>))),
        }
    }
    pub fn mean(&self) -> Tensor {
        let self_data = self.data.lock().unwrap();
        let n = self_data.len() as f32;
        let mean_val = self_data.mean().unwrap();
        let result_data = arr0(mean_val).into_dyn();

        let self_grad_rc = self.gradient.clone();
        let self_shape = self_data.shape().to_vec(); // Need shape to broadcast
        let c_grad_rc = Arc::new(Mutex::new(ArrayD::zeros(result_data.shape())));

        let c_grad_rc_clone = c_grad_rc.clone();
        let backward_op = move || {
            let c_grad_scalar = c_grad_rc_clone.lock().unwrap()[[]];
            
            // The derivative is 1/N
            let grad_per_element = c_grad_scalar / n;
            let grad_update = ArrayD::from_elem(self_shape.clone(), grad_per_element);
            self_grad_rc.lock().unwrap().scaled_add(1.0, &grad_update);
        };

        // 3. CREATE NEW TENSOR
        Tensor {
            data: Arc::new(Mutex::new(result_data)),
            gradient: c_grad_rc,
            _children: vec![self.clone()],
            _backward: Arc::new(Mutex::new(Some(Box::new(backward_op) as Box<dyn Fn() + Send + Sync>))),
        }
    }
}

//back propagation
impl Tensor {

    /// Performs backpropagation starting from this Tensor which is assumed to be the loss
    pub fn backward(&self) {
        // 1. Construct the topological sort
        let mut topo: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<TensorId> = HashSet::new();
        build_topo(self, &mut visited, &mut topo);

        // 2. Initialize the gradient
        // - Set d(loss)/d(loss) = 1.0
        // - We use .fill() assuming the loss is a scalar (e.g., from .mean())
        self.gradient.lock().unwrap().fill(1.0);

        // 3. Back propagation
        for tensor in topo.iter().rev() {
            // If the tensor has a backward op, call it
            if let Some(backward_op) = tensor._backward.lock().unwrap().as_ref() {
                backward_op();
            }
        }
    }
}


//helper functions

fn transpose(arr_d: &ArrayD<f32>) -> ArrayD<f32> {
    arr_d.view().into_dimensionality::<ndarray::Ix2>().unwrap().t().to_owned().into_dyn()
}


fn build_topo(tensor: &Tensor, visited: &mut HashSet<TensorId>, topo: &mut Vec<Tensor>) {
    
    let id = Arc::as_ptr(&tensor.data);
    
    if !visited.contains(&id) {
        visited.insert(id);
        for child in &tensor._children {
            build_topo(child, visited, topo);
        }
        topo.push(tensor.clone());
    }
}
