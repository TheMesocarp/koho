use candle_core::{backprop::GradStore, Result as CandleResult, Tensor, Var};

use crate::nn::optim::Optimizer;

/// Adam optimizer with optional weight decay (AdamW)
pub struct Adam {
    m: Vec<Tensor>, // First moment (mean)
    v: Vec<Tensor>, // Second moment (variance)
    t: f64,         // Time step
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: Option<f64>,
    decoupled: bool, // If true, use AdamW
}

impl Adam {
    /// Create a new Adam optimizer
    pub fn new(vars: Vec<&mut Var>, lr: f64) -> CandleResult<Self> {
        let mut m = Vec::new();
        let mut v = Vec::new();

        // Initialize moment buffers with zeros
        for var in &vars {
            let shape = var.shape();
            let device = var.device();
            let dtype = var.dtype();

            m.push(Tensor::zeros(shape, dtype, device)?);
            v.push(Tensor::zeros(shape, dtype, device)?);
        }

        Ok(Self {
            m,
            v,
            t: 0.0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: None,
            decoupled: false,
        })
    }

    /// Set beta coefficients
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set epsilon for numerical stability
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Enable weight decay (coupled - standard Adam with L2)
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = Some(weight_decay);
        self.decoupled = false;
        self
    }

    /// Enable decoupled weight decay (AdamW)
    pub fn decoupled_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = Some(weight_decay);
        self.decoupled = true;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, grads: &GradStore, vars: Vec<&mut Var>) -> CandleResult<()> {
        self.t += 1.0;

        let bias_correction1 = 1.0 - self.beta1.powf(self.t);
        let bias_correction2 = 1.0 - self.beta2.powf(self.t);
        if self.v.len() != vars.len() {
            return Err(candle_core::Error::Msg(
                "params do not match initialization".to_string(),
            ));
        }
        for (i, var) in vars.into_iter().enumerate() {
            if let Some(grad) = grads.get(var) {
                let mut effective_grad = grad.clone();

                if let Some(wd) = self.weight_decay {
                    if !self.decoupled {
                        effective_grad = (effective_grad + (var.as_tensor() * wd)?)?;
                    }
                }

                let m_t = &self.m[i];
                let v_t = &self.v[i];
                let m_new = ((m_t * self.beta1)? + (effective_grad.clone() * (1.0 - self.beta1))?)?;
                let v_new = ((v_t * self.beta2)? + (effective_grad.sqr()? * (1.0 - self.beta2))?)?;
                let m_hat = (&m_new / bias_correction1)?;
                let v_hat = (&v_new / bias_correction2)?;

                let update = (m_hat * self.lr)? / (v_hat.sqrt()? + self.eps)?;
                if let Some(wd) = self.weight_decay {
                    if self.decoupled {
                        // AdamW
                        var.set(&((var.as_tensor() * (1.0 - self.lr * wd))? - update)?)?;
                    } else {
                        // Standard Adam
                        var.set(&(var.as_tensor() - update)?)?;
                    }
                } else {
                    // No weight decay
                    var.set(&(var.as_tensor() - update)?)?;
                }

                self.m[i] = m_new;
                self.v[i] = v_new;
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
