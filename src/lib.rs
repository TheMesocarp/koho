//! Sheaf Neural Network implementation.
//!
//! This module provides a neural network architecture that operates on cellular sheaves,
//! allowing for topological deep learning on structured data. The SheafNN implements
//! a sequence of diffusion layers and provides training utilities.
use candle_core::Var;
use error::KohoError;
use math::tensors::Matrix;
use nn::{
    diffuse::DiffusionLayer,
    loss::{LossFn, LossKind},
    metrics::{EpochMetrics, TrainingMetrics},
    optim::{OptimKind, OptimizerParams},
};

use crate::{math::sheaf::CellularSheaf, nn::optim::create_optimizer};

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

    /// Returns mutable references to parameters (new method)
    fn parameters_mut(&mut self) -> Vec<&mut Var>;
}

/// A sheaf neural network operating on k-cells.
///
/// This neural network architecture applies a sequence of diffusion operations
/// on a cellular sheaf, learning representations that respect the underlying
/// topological structure. It includes complete training functionality with
/// configurable optimizers and loss functions.
pub struct SheafNN {
    sheaf: CellularSheaf,
    /// The sequence of diffusion layers in the network
    layers: Vec<DiffusionLayer>,
    /// The loss function used for training
    loss_fn: LossFn,
    /// The dimension of cells this network operates on
    k: usize,
    down_included: bool,
}

impl SheafNN {
    /// Initializes a new `SheafNN` for the provided `CellularSheaf`
    ///
    /// # Arguments
    /// * `k` - The cell dimension at which the neural network sits
    /// * `down_laplacian_included` - Indicates whether to use the full Hodge Laplacian or just the up-Laplacian
    /// * `loss` - The loss type of the network
    /// * `sheaf` - The `CellularSheaf` in which to diffuse over
    ///
    /// # Returns
    /// A new `SheafNN` without diffusion layers initialized
    pub fn init(
        k: usize,
        down_laplacian_included: bool,
        loss: LossKind,
        sheaf: CellularSheaf,
    ) -> Self {
        let loss_fn = LossFn::new(loss);
        Self {
            sheaf,
            layers: Vec::new(),
            loss_fn,
            k,
            down_included: down_laplacian_included,
        }
    }

    /// Initializes the diffusion layers for a fresh sheaf neural network with the specified diffusion layers.
    ///
    /// # Arguments
    /// * `layers` - A vector of diffusion layers that form the network
    pub fn sequential(&mut self, layers: Vec<DiffusionLayer>) {
        self.layers = layers;
    }

    /// Refreshes the parameters tracked by the optimizer.
    ///
    /// This should be called after manually updating model parameters
    /// to ensure the optimizer is operating on the current values.
    ///
    /// # Returns
    /// Result indicating success or an error
    pub fn forward(&self, input: Matrix, down_included: bool) -> Result<Matrix, KohoError> {
        let mut output = input;
        for i in &self.layers {
            output = i.diffuse(&self.sheaf, self.k, output, down_included)?;
        }
        Ok(output)
    }

    /// Main training loop
    pub fn train(
        &mut self,
        data: &[(Matrix, Matrix)],
        epochs: usize,
        down_included: bool,
        optimizer_kind: OptimKind,
        lr: f64,
        optimizer_params: OptimizerParams,
    ) -> Result<TrainingMetrics, KohoError> {
        // Create the optimizer
        let mut optimizer =
            create_optimizer(optimizer_kind, self.parameters_mut(), lr, optimizer_params)
                .map_err(KohoError::Candle)?;

        let mut metrics = TrainingMetrics::new(epochs);

        for epoch in 1..=epochs {
            let mut total_loss = 0.0_f32;

            for (input, target) in data {
                // Forward pass
                let output = self.forward(input.clone(), down_included)?;

                // Compute loss
                let loss_tensor = self
                    .loss_fn
                    .compute(output.inner(), target.inner())
                    .map_err(KohoError::Candle)?;

                let loss_val = loss_tensor.to_scalar::<f32>().unwrap_or(f32::NAN);
                total_loss += loss_val;

                // Backward pass
                let grads = loss_tensor.backward().map_err(KohoError::Candle)?;

                // Optimizer step (in-place update of parameters)
                optimizer
                    .step(&grads, self.parameters_mut())
                    .map_err(KohoError::Candle)?;
            }

            let avg_loss = total_loss / (data.len() as f32);
            metrics.push(EpochMetrics::new(epoch, avg_loss));
        }
        Ok(metrics)
    }
}

impl Parameterized for SheafNN {
    /// Returns all learnable parameters across all layers of the network.
    ///
    /// # Returns
    /// A vector containing all the network's parameters
    fn parameters(&self) -> Vec<Var> {
        let mut out = self
            .layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect::<Vec<_>>();
        if self.sheaf.learned {
            out.extend(self.sheaf.parameters(self.k, self.down_included));
        }
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Var> {
        let learned = self.sheaf.learned;
        let mut out = self
            .layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect::<Vec<_>>();
        if learned {
            out.extend(self.sheaf.parameters_mut(self.k, self.down_included));
        }
        out
    }
}
