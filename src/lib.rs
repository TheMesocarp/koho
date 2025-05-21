//! Sheaf Neural Network implementation.
//!
//! This module provides a neural network architecture that operates on cellular sheaves,
//! allowing for topological deep learning on structured data. The SheafNN implements
//! a sequence of diffusion layers and provides training utilities.

use candle_core::{Tensor, Var};
use error::KohoError;
use math::tensors::Matrix;
use nn::{
    diffuse::DiffusionLayer,
    loss::{LossFn, LossKind},
    metrics::{EpochMetrics, TrainingMetrics},
    optimize::{OptimKind, Optimize, OptimizerParams},
};

pub mod error;
pub mod math;
pub mod nn;

/// High level trait for any modules that has parameters that can be optimized.
///
/// This trait is implemented by components containing learnable parameters
/// that can be passed to optimizers during training.
pub trait Parameterized {
    /// Returns a Vec of every Var (learnable parameter) this module owns.
    fn parameters(&self) -> Vec<Var>;
}

/// A sheaf neural network operating on k-cells.
///
/// This neural network architecture applies a sequence of diffusion operations
/// on a cellular sheaf, learning representations that respect the underlying
/// topological structure. It includes complete training functionality with
/// configurable optimizers and loss functions.
pub struct SheafNN {
    /// The sequence of diffusion layers in the network
    layers: Vec<DiffusionLayer>,
    /// The loss function used for training
    loss_fn: LossFn,
    /// The optimizer used for parameter updates
    optimizer: Optimize,
    /// The dimension of cells this network operates on
    k: usize,
}

impl SheafNN {
    /// Creates a new sheaf neural network with the specified diffusion layers and loss type.
    ///
    /// # Arguments
    /// * `k` - The dimension of cells this network will operate on
    /// * `layers` - A vector of diffusion layers that form the network
    /// * `loss` - The type of loss function to use for training
    ///
    /// # Returns
    /// A new SheafNN with default optimizer
    pub fn sequential(k: usize, layers: Vec<DiffusionLayer>, loss: LossKind) -> Self {
        let loss_fn = LossFn::new(loss);
        Self {
            layers,
            loss_fn,
            optimizer: Optimize::default(),
            k,
        }
    }

    /// Sets the optimizer for the network's training process.
    ///
    /// # Arguments
    /// * `kind` - The type of optimizer to use (e.g., SGD, Adam)
    /// * `lr` - The learning rate
    /// * `extra_params` - Additional optimizer-specific parameters
    ///
    /// # Returns
    /// Result indicating success or an error
    pub fn set_optimizer(
        &mut self,
        kind: OptimKind,
        lr: f64,
        extra_params: OptimizerParams,
    ) -> Result<(), KohoError> {
        let vars = self.parameters();
        let mut opt = Optimize::new(kind, extra_params, lr);
        opt.new_weights(vars).map_err(KohoError::Candle)?;
        self.optimizer = opt;
        Ok(())
    }

    /// Refreshes the parameters tracked by the optimizer.
    ///
    /// This should be called after manually updating model parameters
    /// to ensure the optimizer is operating on the current values.
    ///
    /// # Returns
    /// Result indicating success or an error
    pub fn refresh_optimizer(&mut self) -> Result<(), KohoError> {
        let weights = self.parameters();
        self.optimizer
            .new_weights(weights)
            .map_err(KohoError::Candle)
    }

    /// Refreshes the parameters tracked by the optimizer.
    ///
    /// This should be called after manually updating model parameters
    /// to ensure the optimizer is operating on the current values.
    ///
    /// # Returns
    /// Result indicating success or an error
    pub fn forward(&mut self, input: Matrix, down_included: bool) -> Result<Matrix, KohoError> {
        let mut output = input;
        for i in &self.layers {
            output = i.diffuse(self.k, output, down_included)?;
        }
        Ok(output)
    }

    /// Runs a backward pass to update the model parameters.
    ///
    /// This method computes the loss between the output and target,
    /// then uses the optimizer to update the model parameters accordingly.
    ///
    /// # Arguments
    /// * `output` - The output from the forward pass
    /// * `target` - The target values for supervised learning
    ///
    /// # Returns
    /// Result indicating success or an error
    pub fn backward(&mut self, output: &Matrix, target: &Matrix) -> Result<(), KohoError> {
        let output_tensor: &Tensor = output.inner();
        let target_tensor: &Tensor = target.inner();

        let loss_tensor = self
            .loss_fn
            .compute(output_tensor, target_tensor)
            .map_err(KohoError::Candle)?;

        self.optimizer
            .backward_step(&loss_tensor)
            .map_err(KohoError::Candle)?;
        Ok(())
    }

    /// Runs a complete training loop for multiple epochs.
    ///
    /// # Arguments
    /// * `data` - Vector of (input, target) matrix pairs for training
    /// * `epochs` - Number of training epochs to run
    /// * `down_included` - Whether to include diffusion from lower-dimensional cells
    ///
    /// # Returns
    /// Training metrics collected during the training process
    pub fn train(
        &mut self,
        data: &[(Matrix, Matrix)],
        epochs: usize,
        down_included: bool,
    ) -> Result<TrainingMetrics, KohoError> {
        let mut output = TrainingMetrics::new(epochs);
        for epoch in 1..=epochs {
            let mut total_loss = 0.0_f32;
            for (input, target) in data {
                let output = self.forward(input.clone(), down_included)?;
                let loss_tensor = self
                    .loss_fn
                    .compute(output.inner(), target.inner())
                    .map_err(KohoError::Candle)?;

                let loss_val = loss_tensor.to_scalar::<f32>().unwrap_or(f32::NAN);
                total_loss += loss_val;
                self.backward(&output, target)?;

                let new_weights = self.optimizer.into_inner()?;

                for (i, var) in new_weights.into_iter().enumerate() {
                    self.layers[i].update_weights(var);
                }
                self.refresh_optimizer()?;
            }

            let avg_loss = total_loss / (data.len() as f32);
            output.push(EpochMetrics::new(epoch, avg_loss));
        }
        Ok(output)
    }
}

impl Parameterized for SheafNN {
    /// Returns all learnable parameters across all layers of the network.
    ///
    /// # Returns
    /// A vector containing all the network's parameters
    fn parameters(&self) -> Vec<Var> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
