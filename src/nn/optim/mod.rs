use candle_core::{backprop::GradStore, Result as CandleResult, Var};
use candle_optimisers::{Decay, Momentum};

pub use crate::nn::optim::adam::Adam;
pub use crate::nn::optim::rmsprop::RMSprop;
pub use crate::nn::optim::sgd::SGD;

mod adam;
mod rmsprop;
mod sgd;

/// Trait for optimizers that work with mutable references to variables
pub trait Optimizer {
    /// Perform one optimization step using the gradients
    fn step(&mut self, grads: &GradStore, vars: Vec<&mut Var>) -> CandleResult<()>;

    /// Get the current learning rate
    fn learning_rate(&self) -> f64;

    /// Set the learning rate (for scheduling)
    fn set_learning_rate(&mut self, lr: f64);
}

/// Extra parameters passed as input during to the `Optimize` creation for the network.
pub enum OptimizerParams {
    SGD {
        dampening: Option<f64>,
        weight_decay: Option<Decay>,
        momentum: Option<Momentum>,
    },
    AdamW {
        beta_1: f64,
        beta_2: f64,
        eps: f64,
        weight_decay: Option<Decay>,
        amsgrad: bool,
    },
    Else,
}

/// Enum to represent which optimizer type to use
#[derive(Debug, Clone)]
pub enum OptimKind {
    SGD,
    Adam,
    AdamW, // will use Adam with decoupled weight decay
    RMSprop,
}

// Helper function to create optimizers from your existing enum
pub fn create_optimizer<'a>(
    kind: OptimKind,
    vars: Vec<&mut Var>,
    lr: f64,
    params: OptimizerParams,
) -> CandleResult<Box<dyn Optimizer + 'a>> {
    match kind {
        OptimKind::SGD => {
            if let OptimizerParams::SGD {
                momentum,
                weight_decay,
                ..
            } = params
            {
                let mut opt = SGD::new(lr)?;
                if let Some(Momentum::Nesterov(val)) = momentum {
                    opt = opt.momentum(val);
                }
                if let Some(Decay::WeightDecay(val)) = weight_decay {
                    opt = opt.weight_decay(val);
                }
                Ok(Box::new(opt))
            } else {
                Ok(Box::new(SGD::new(lr)?))
            }
        }
        OptimKind::Adam | OptimKind::AdamW => {
            if let OptimizerParams::AdamW {
                beta_1,
                beta_2,
                eps,
                weight_decay,
                ..
            } = params
            {
                let mut opt = Adam::new(vars, lr)?.betas(beta_1, beta_2).eps(eps);

                if let Some(wd) = weight_decay {
                    match wd {
                        Decay::WeightDecay(val) => {
                            opt = opt.weight_decay(val);
                        }
                        Decay::DecoupledWeightDecay(val) => {
                            opt = opt.decoupled_weight_decay(val);
                        }
                    }
                }
                Ok(Box::new(opt))
            } else {
                Ok(Box::new(Adam::new(vars, lr)?))
            }
        }
        OptimKind::RMSprop => Ok(Box::new(RMSprop::new(vars, lr)?)),
    }
}
