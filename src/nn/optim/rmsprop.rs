use candle_core::{backprop::GradStore, Error, Result as CandleResult, Tensor, Var};

use crate::nn::optim::Optimizer;

/// RMSprop optimizer
pub struct RMSprop {
    v: Vec<Tensor>, // Running average of squared gradients
    lr: f64,
    alpha: f64, // Decay rate
    eps: f64,
    weight_decay: Option<f64>,
    momentum: Option<f64>,
    momentum_buffers: Vec<Option<Tensor>>,
    centered: bool,           // If true, use centered RMSprop
    avg: Vec<Option<Tensor>>, // For centered variant
}

impl RMSprop {
    /// Create a new RMSprop optimizer
    pub fn new(vars: Vec<&mut Var>, lr: f64) -> CandleResult<Self> {
        let mut v = Vec::new();
        let len = vars.len();
        for var in vars {
            let shape = var.shape();
            let device = var.device();
            let dtype = var.dtype();

            v.push(Tensor::zeros(shape, dtype, device)?);
        }

        Ok(Self {
            v,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: None,
            momentum: None,
            momentum_buffers: vec![None; len],
            centered: false,
            avg: vec![None; len],
        })
    }

    /// Set the decay rate (alpha)
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set epsilon for numerical stability
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Enable weight decay
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = Some(weight_decay);
        self
    }

    /// Enable momentum
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = Some(momentum);
        self
    }

    /// Enable centered RMSprop (normalized by gradient mean)
    pub fn centered(mut self, vars: Vec<Var>) -> CandleResult<Self> {
        self.centered = true;
        if self.v.len() != vars.len() {
            return Err(candle_core::Error::Msg(
                "params do not match initialization".to_string(),
            ));
        }
        for (i, var) in vars.into_iter().enumerate() {
            let shape = var.shape();
            let device = var.device();
            let dtype = var.dtype();

            self.avg[i] = Some(Tensor::zeros(shape, dtype, device)?);
        }

        Ok(self)
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, grads: &GradStore, vars: Vec<&mut Var>) -> CandleResult<()> {
        if self.v.len() != vars.len() {
            return Err(candle_core::Error::Msg(
                "params do not match initialization".to_string(),
            ));
        }
        for (i, var) in vars.into_iter().enumerate() {
            if let Some(grad) = grads.get(var) {
                let mut effective_grad = grad.clone();

                if let Some(wd) = self.weight_decay {
                    effective_grad = (effective_grad + (var.as_tensor() * wd)?)?;
                }

                let v_t = &self.v[i];
                let v_new = ((v_t * self.alpha)? + (effective_grad.sqr()? * (1.0 - self.alpha))?)?;

                let denom = if self.centered {
                    // Centered RMSprop
                    let avg_t = self.avg[i].as_ref().ok_or(Error::Msg(
                        "failed to initialize centering for RMSprop".to_string(),
                    ))?;
                    let avg_new =
                        ((avg_t * self.alpha)? + (effective_grad.clone() * (1.0 - self.alpha))?)?;
                    self.avg[i] = Some(avg_new.clone());

                    // Denominator: sqrt(E[g^2] - E[g]^2 + eps)
                    ((&v_new - avg_new.sqr()?)?.sqrt()? + self.eps)?
                } else {
                    // Standard RMSprop: denominator is sqrt(E[g^2] + eps)
                    (v_new.sqrt()? + self.eps)?
                };
                let update = (effective_grad.clone() * self.lr)? / denom;

                let final_update = if let Some(momentum) = self.momentum {
                    let buf = if let Some(buf) = &self.momentum_buffers[i] {
                        (buf * momentum)? + update?
                    } else {
                        update
                    }?;

                    self.momentum_buffers[i] = Some(buf.clone());
                    buf
                } else {
                    update?
                };

                var.set(&(var.as_tensor() - final_update)?)?;
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
