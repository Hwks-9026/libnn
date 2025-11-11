use ndarray::ArrayD;
use ndarray::arr0;
use ndarray::linalg::Dot;
use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashSet;

use std::ops::*;

type TensorData = Rc<RefCell<ArrayD<f32>>>;
type GradientData = Rc<RefCell<ArrayD<f32>>>;
type TensorId = *const RefCell<ArrayD<f32>>;

#[derive(Clone)]
pub struct Tensor {
    pub data: TensorData,
    pub gradient: GradientData,
    _children: Vec<Tensor>,
    
    //for back propagation
    _backward: Rc<RefCell<Option<Box<dyn Fn()>>>>
}

// Constructors for Tensor
impl Tensor {
    pub fn leaf(data: ArrayD<f32>) -> Self {
        let grad = ArrayD::zeros(data.shape());
        Self {
            data: Rc::new(RefCell::new(data)),
            gradient: Rc::new(RefCell::new(grad)),
            _children: vec![],
            _backward: Rc::new(RefCell::new(None)),
        }
    }   

    fn _internal_new(
        data: ArrayD<f32>,
        children: Vec<Tensor>,
        backward_op: Box<dyn Fn()>,
    ) -> Self {
        let grad = ArrayD::zeros(data.shape());
        Self {
            data: Rc::new(RefCell::new(data)),
            gradient: Rc::new(RefCell::new(grad)),
            _children: children,
            _backward: Rc::new(RefCell::new(Some(backward_op))),
        }
    }
}

// Math Operations
impl Tensor {
    pub fn add(self, rhs: & Tensor) -> Tensor {
        let self_data = self.data.borrow();
        let rhs_data = rhs.data.borrow();
        
        println!("--- ADDING ---");
        dbg!(self_data.shape());
        dbg!(rhs_data.shape());
        
        let result_data = &*self_data + &*rhs_data;
        println!("--- SUCEEDED ---");
        
        let self_grad_rc = self.gradient.clone();
        let rhs_grad_rc = rhs.gradient.clone();
        let c_grad_rc = Rc::new(RefCell::new(ArrayD::zeros(result_data.shape())));
        
        let c_grad_rc_clone = c_grad_rc.clone(); // Clone for the closure
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.borrow();
            self_grad_rc.borrow_mut().scaled_add(1.0, &*c_grad);
            rhs_grad_rc.borrow_mut().scaled_add(1.0, &*c_grad);
        };
        Tensor {
            data: Rc::new(RefCell::new(result_data)),
            gradient: c_grad_rc, // Assign the grad Rc
            _children: vec![self.clone(), rhs.clone()],
            _backward: Rc::new(RefCell::new(Some(Box::new(backward_op)))),
        }
    }

    pub fn sub(self, rhs: &Tensor) -> Tensor {
        let self_data = self.data.borrow();
        let rhs_data = rhs.data.borrow();
        
        println!("--- SUBTRACTING ---");
        dbg!(self_data.shape());
        dbg!(rhs_data.shape());
        
        let result_data = &*self_data - &*rhs_data;
        println!("--- SUCEEDED ---");
        
        let self_grad_rc = self.gradient.clone();
        let rhs_grad_rc = rhs.gradient.clone();
        let c_grad_rc = Rc::new(RefCell::new(ArrayD::zeros(result_data.shape())));
        
        let c_grad_rc_clone = c_grad_rc.clone(); // Clone for the closure
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.borrow();
            self_grad_rc.borrow_mut().scaled_add(1.0, &*c_grad);
            rhs_grad_rc.borrow_mut().scaled_add(1.0, &*c_grad);
        };
        Tensor {
            data: Rc::new(RefCell::new(result_data)),
            gradient: c_grad_rc, // Assign the grad Rc
            _children: vec![self.clone(), rhs.clone()],
            _backward: Rc::new(RefCell::new(Some(Box::new(backward_op)))),
        }
    }

    pub fn mul(self, rhs: &Tensor) -> Tensor {
        let self_data_rc = self.data.clone();
        let rhs_data_rc = rhs.data.clone();
        
        let result_data = &*self.data.borrow() * &*rhs.data.borrow();

        let self_grad_rc = self.gradient.clone();
        let rhs_grad_rc = rhs.gradient.clone();
        
        let c_grad_rc = Rc::new(RefCell::new(ArrayD::zeros(result_data.shape())));
        let c_grad_rc_clone = c_grad_rc.clone();
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.borrow();
            
            // a.grad += c.grad * b.data
            let a_grad_update = &*c_grad * &*rhs_data_rc.borrow();
            self_grad_rc.borrow_mut().scaled_add(1.0, &a_grad_update);

            // b.grad += c.grad * a.data
            let b_grad_update = &*c_grad * &*self_data_rc.borrow();
            rhs_grad_rc.borrow_mut().scaled_add(1.0, &b_grad_update);
        };

        Tensor {
            data: Rc::new(RefCell::new(result_data)),
            gradient: c_grad_rc,
            _children: vec![self.clone(), rhs.clone()],
            _backward: Rc::new(RefCell::new(Some(Box::new(backward_op)))),
        }
    }

    pub fn relu(&self) -> Tensor {
        let a_data = self.data.borrow();
        let result_data = a_data.mapv(|x| x.max(0.0));
        let positive_mask = a_data.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        let a_grad_rc = self.gradient.clone();
        let c_grad_rc = Rc::new(RefCell::new(ArrayD::zeros(result_data.shape())));
        
        let c_grad_rc_clone = c_grad_rc.clone();
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.borrow();
            
            // a.grad += c.grad * mask
            let grad_update = &*c_grad * &positive_mask;
            a_grad_rc.borrow_mut().scaled_add(1.0, &grad_update);
        };

        Tensor {
            data: Rc::new(RefCell::new(result_data)),
            gradient: c_grad_rc,
            _children: vec![self.clone()], // Only one child
            _backward: Rc::new(RefCell::new(Some(Box::new(backward_op)))),
        }
    }

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {

        let self_data = self.data.borrow();
        let rhs_data = rhs.data.borrow();
    
        let result_data = &*self_data.dot(&*rhs_data);

        let self_data_rc = self.data.clone();
        let rhs_data_rc = rhs.data.clone();
        let self_grad_rc = self.gradient.clone();
        let rhs_grad_rc = rhs.gradient.clone();

        let c_grad_rc = Rc::new(RefCell::new(ArrayD::zeros(result_data.shape())));

        let c_grad_rc_clone = c_grad_rc.clone();
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.borrow();
            let self_data = self_data_rc.borrow();
            let rhs_data = rhs_data_rc.borrow();

            // a.grad += c.grad.dot(b.data.T)
            let a_grad_update = c_grad.dot(&transpose(&rhs_data));
            self_grad_rc.borrow_mut().scaled_add(1.0, &a_grad_update);

            // b.grad += a.data.T.dot(c.grad)
            let b_grad_update = transpose(&self_data).dot(&*c_grad);
            rhs_grad_rc.borrow_mut().scaled_add(1.0, &b_grad_update);
        };

        Tensor {
            data: Rc::new(RefCell::new(result_data.to_owned())),
            gradient: c_grad_rc,
            _children: vec![self.clone(), rhs.clone()],
            _backward: Rc::new(RefCell::new(Some(Box::new(backward_op)))),
        }
    }

    pub fn powf(&self, n: f32) -> Tensor {
        let result_data = self.data.borrow().mapv(|x| x.powf(n));
        let self_data_rc = self.data.clone(); // Need original data for derivative
        let self_grad_rc = self.gradient.clone();
        let c_grad_rc = Rc::new(RefCell::new(ArrayD::zeros(result_data.shape())));

        let c_grad_rc_clone = c_grad_rc.clone();
        let backward_op = move || {
            let c_grad = c_grad_rc_clone.borrow();
            let self_data = self_data_rc.borrow();

            // a.grad += c.grad * (n * a.data^(n-1))
            let deriv = self_data.mapv(|x| n * x.powf(n - 1.0));
            let grad_update = &*c_grad * &deriv;
            self_grad_rc.borrow_mut().scaled_add(1.0, &grad_update);
        };

        Tensor {
            data: Rc::new(RefCell::new(result_data)),
            gradient: c_grad_rc,
            _children: vec![self.clone()],
            _backward: Rc::new(RefCell::new(Some(Box::new(backward_op)))),
        }
    }
    pub fn mean(&self) -> Tensor {
        let self_data = self.data.borrow();
        let n = self_data.len() as f32;
        let mean_val = self_data.mean().unwrap();
        let result_data = arr0(mean_val).into_dyn();

        let self_grad_rc = self.gradient.clone();
        let self_shape = self_data.shape().to_vec(); // Need shape to broadcast
        let c_grad_rc = Rc::new(RefCell::new(ArrayD::zeros(result_data.shape())));

        let c_grad_rc_clone = c_grad_rc.clone();
        let backward_op = move || {
            let c_grad_scalar = c_grad_rc_clone.borrow()[[]];
            
            // The derivative is 1/N
            let grad_per_element = c_grad_scalar / n;
            let grad_update = ArrayD::from_elem(self_shape.clone(), grad_per_element);
            self_grad_rc.borrow_mut().scaled_add(1.0, &grad_update);
        };

        // 3. CREATE NEW TENSOR
        Tensor {
            data: Rc::new(RefCell::new(result_data)),
            gradient: c_grad_rc,
            _children: vec![self.clone()],
            _backward: Rc::new(RefCell::new(Some(Box::new(backward_op)))),
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
        self.gradient.borrow_mut().fill(1.0);

        // 3. Back propagation
        for tensor in topo.iter().rev() {
            // If the tensor has a backward op, call it
            if let Some(backward_op) = tensor._backward.borrow().as_ref() {
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
    
    let id = Rc::as_ptr(&tensor.data);
    
    if !visited.contains(&id) {
        visited.insert(id);
        for child in &tensor._children {
            build_topo(child, visited, topo);
        }
        topo.push(tensor.clone());
    }
}
