use std::collections::HashMap;

use candle_core::{backprop::GradStore, Result as CandleResult, Tensor, Var};

use crate::nn::optim::Optimizer;

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    lr: f64,
    momentum: Option<f64>,
    weight_decay: Option<f64>,
    momentum_buffers: HashMap<*const Var, Tensor>,
}

impl SGD {
    /// Create a new SGD optimizer
    pub fn new(lr: f64) -> CandleResult<Self> {
        Ok(Self {
            lr,
            momentum: None,
            weight_decay: None,
            momentum_buffers: HashMap::new(),
        })
    }

    /// Enable momentum with the given coefficient
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = Some(momentum);
        self
    }

    /// Enable weight decay (L2 regularization)
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = Some(weight_decay);
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, grads: &GradStore, vars: Vec<&mut Var>) -> CandleResult<()> {
        for var in vars {
            let ptr = var as *const Var;

            if let Some(grad) = grads.get(var) {
                let mut effective_grad = grad.clone();

                if let Some(wd) = self.weight_decay {
                    effective_grad = (effective_grad + (var.as_tensor() * wd)?)?;
                }

                if let Some(momentum) = self.momentum {
                    let buf = if let Some(buf) = self.momentum_buffers.get(&ptr) {
                        (buf * momentum) + effective_grad.clone()
                    } else {
                        Ok(effective_grad.clone())
                    }?;

                    self.momentum_buffers.insert(ptr, buf.clone());
                    effective_grad = buf;
                }

                var.set(&(var.as_tensor() - (effective_grad * self.lr)?)?)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }
}
