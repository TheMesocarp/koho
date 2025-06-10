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

    pub fn train_debug(
        &mut self,
        data: &[(Matrix, Matrix)],
        epochs: usize,
        down_included: bool,
        optimizer_kind: OptimKind,
        lr: f64,
        optimizer_params: OptimizerParams,
    ) -> Result<TrainingMetrics, KohoError> {
        println!("=== Training Debug Info ===");

        // Check if we have any parameters to optimize
        let params = self.parameters();
        println!("Total parameters: {}", params.len());
        for (i, param) in params.iter().enumerate() {
            let param_data = param.as_tensor().flatten_all().map_err(KohoError::Candle)?;
            let param_vec = param_data.to_vec1::<f32>().map_err(KohoError::Candle)?;
            println!(
                "Parameter {i}: shape={:?}, first_few_values={:?}",
                param.shape(),
                &param_vec[..param_vec.len().min(5)]
            );
        }

        // Create the optimizer
        let mut optimizer =
            create_optimizer(optimizer_kind, self.parameters_mut(), lr, optimizer_params)
                .map_err(KohoError::Candle)?;

        let mut metrics = TrainingMetrics::new(epochs);

        for epoch in 1..=epochs {
            let mut total_loss = 0.0_f32;

            for (batch_idx, (input, target)) in data.iter().enumerate() {
                println!("\nEpoch {epoch}, Batch {batch_idx}");

                // Print input/target info
                let input_data = input.inner().flatten_all().map_err(KohoError::Candle)?;
                let target_data = target.inner().flatten_all().map_err(KohoError::Candle)?;
                let input_vec = input_data.to_vec1::<f32>().map_err(KohoError::Candle)?;
                let target_vec = target_data.to_vec1::<f32>().map_err(KohoError::Candle)?;

                println!("Input: {input_vec:?}");
                println!("Target: {target_vec:?}");

                // Forward pass
                let output = self.forward(input.clone(), down_included)?;
                let output_data = output.inner().flatten_all().map_err(KohoError::Candle)?;
                let output_vec = output_data.to_vec1::<f32>().map_err(KohoError::Candle)?;
                println!("Output: {output_vec:?}");

                // Compute loss
                let loss_tensor = self
                    .loss_fn
                    .compute(output.inner(), target.inner())
                    .map_err(KohoError::Candle)?;

                let loss_val = loss_tensor.to_scalar::<f32>().unwrap_or(f32::NAN);
                total_loss += loss_val;
                println!("Loss: {loss_val}");

                // Check if loss tensor requires grad
                println!("Loss tensor shape: {:?}", loss_tensor.shape());
                println!("Loss tensor dtype: {:?}", loss_tensor.dtype());

                // Backward pass
                println!("Computing gradients...");
                let grads = loss_tensor.backward().map_err(KohoError::Candle)?;

                // Check gradients
                let params_mut = self.parameters_mut();
                println!("Checking gradients for {} parameters:", params_mut.len());
                for (i, param) in params_mut.iter().enumerate() {
                    if let Some(grad) = grads.get(param) {
                        let grad_data = grad.flatten_all().map_err(KohoError::Candle)?;
                        let grad_vec = grad_data.to_vec1::<f32>().map_err(KohoError::Candle)?;
                        let grad_norm = grad_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                        println!(
                            "  Param {i}: grad_norm={grad_norm}, first_few_grads={:?}",
                            &grad_vec[..grad_vec.len().min(3)]
                        );
                    } else {
                        println!("  Param {i}: NO GRADIENT FOUND");
                    }
                }

                // Optimizer step (in-place update of parameters)
                println!("Applying optimizer step...");
                let params_before: Vec<_> = self
                    .parameters_mut()
                    .iter()
                    .map(|p| {
                        p.as_tensor()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap()
                    })
                    .collect();

                optimizer
                    .step(&grads, self.parameters_mut())
                    .map_err(KohoError::Candle)?;

                let params_after: Vec<_> = self
                    .parameters_mut()
                    .iter()
                    .map(|p| {
                        p.as_tensor()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap()
                    })
                    .collect();

                // Check if parameters actually changed
                for (i, (before, after)) in
                    params_before.iter().zip(params_after.iter()).enumerate()
                {
                    let diff_norm: f32 = before
                        .iter()
                        .zip(after.iter())
                        .map(|(b, a)| (b - a).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    println!("  Param {i} change norm: {diff_norm}");
                }

                if epoch <= 3 {
                    // Only print detailed info for first few epochs
                    println!("--- End batch {batch_idx} ---");
                }
            }

            let avg_loss = total_loss / (data.len() as f32);
            metrics.push(EpochMetrics::new(epoch, avg_loss));

            if epoch <= 10 || epoch % 10 == 0 {
                println!("Epoch {epoch}: avg_loss = {avg_loss}");
            }
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

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::{
        math::{
            cell::Cell,
            sheaf::{CellularSheaf, Section},
            tensors::Matrix,
        },
        nn::{
            activate::Activations,
            diffuse::DiffusionLayer,
            loss::LossKind,
            optim::{OptimKind, OptimizerParams},
        },
    };
    use candle_core::{DType, Device};

    /// Creates a triangle (2-cell) with 3 vertices and 3 edges
    fn create_triangle_sheaf() -> Result<CellularSheaf, KohoError> {
        let mut sheaf = CellularSheaf::init(DType::F32, Device::Cpu, true);

        // Create 3 vertices (0-cells) with initial data
        let v0_data = Section::new(&[1.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v0_idx) = sheaf.attach(Cell::new(0), v0_data, None, None)?;

        let v1_data = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v1_idx) = sheaf.attach(Cell::new(0), v1_data, None, None)?;

        let v2_data = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v2_idx) = sheaf.attach(Cell::new(0), v2_data, None, None)?;

        // Create 3 edges (1-cells)
        let e0_data = Section::new(&[0.5f32], 1, Device::Cpu, DType::F32)?;
        let (_, e0_idx) = sheaf.attach(Cell::new(1), e0_data, None, Some(&[v0_idx, v1_idx]))?;

        let e1_data = Section::new(&[0.5f32], 1, Device::Cpu, DType::F32)?;
        let (_, e1_idx) = sheaf.attach(Cell::new(1), e1_data, None, Some(&[v1_idx, v2_idx]))?;

        let e2_data = Section::new(&[0.5f32], 1, Device::Cpu, DType::F32)?;
        let (_, e2_idx) = sheaf.attach(Cell::new(1), e2_data, None, Some(&[v2_idx, v0_idx]))?;

        // Create 1 triangle face (2-cell)
        let f0_data = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, _f0_idx) =
            sheaf.attach(Cell::new(2), f0_data, None, Some(&[e0_idx, e1_idx, e2_idx]))?;

        sheaf.generate_initial_restrictions(0.1)?;
        println!("uppers: {:?}", sheaf.cells.cells[0][0].upper);
        Ok(sheaf)
    }

    #[test]
    fn test_triangle_diffusion_learning() -> Result<(), KohoError> {
        let sheaf = create_triangle_sheaf()?;
        let input = sheaf.get_k_cochain(0)?;

        let target_data = vec![0.8f32, 0.6f32, 0.4f32];
        let target = Matrix::from_slice(&target_data, 1, 3, Device::Cpu, DType::F32)
            .map_err(KohoError::Candle)?;

        let training_data = vec![(input, target)];
        let mut network = SheafNN::init(0, false, LossKind::MSE, sheaf);

        let diffusion_layer = DiffusionLayer::new(0, Activations::Linear, &network.sheaf)?;
        network.sequential(vec![diffusion_layer]);

        let metrics = network.train_debug(
            &training_data,
            100,
            false,
            OptimKind::Adam,
            0.01,
            OptimizerParams::Else,
        )?;
        assert!(
            metrics.final_loss < metrics.epochs[0].loss,
            "Training should reduce loss over time"
        );

        let initial_input = network.sheaf.get_k_cochain(0)?;
        let output = network.forward(initial_input, false)?;

        // The output should be different from input (diffusion occurred)
        let input_vals = network
            .sheaf
            .get_k_cochain(0)?
            .inner()
            .to_vec2::<f32>()
            .map_err(KohoError::Candle)?;
        let output_vals = output.inner().to_vec2::<f32>().map_err(KohoError::Candle)?;

        println!("Input:  {input_vals:?}");
        println!("Output: {output_vals:?}");
        println!("Final loss: {}", metrics.final_loss);

        assert_eq!(output_vals.len(), 1, "Output should have 1 feature");
        assert_eq!(
            output_vals[0].len(),
            3,
            "Each vertex should have 3 vertices"
        );

        Ok(())
    }

    #[test]
    fn test_edge_diffusion_learning() -> Result<(), KohoError> {
        let sheaf = create_triangle_sheaf()?;

        // Test diffusion on edges (1-cells)
        let input = sheaf.get_k_cochain(1)?;
        println!("got edges");

        let target_data = vec![0.5f32, 0.3f32, 0.7f32];
        let target = Matrix::from_slice(&target_data, 1, 3, Device::Cpu, DType::F32)
            .map_err(KohoError::Candle)?;

        let training_data = vec![(input, target)];

        let mut network = SheafNN::init(1, true, LossKind::MSE, sheaf); // down_included = true
        let diffusion_layer = DiffusionLayer::new(1, Activations::Tanh, &network.sheaf)?;
        network.sequential(vec![diffusion_layer]);

        let metrics = network.train_debug(
            &training_data,
            200,
            true,
            OptimKind::Adam,
            0.2,
            OptimizerParams::Else,
        )?;
        println!("Edge diffusion final loss: {}", metrics.final_loss);

        assert!(metrics.final_loss < 1.0, "Loss should be reasonable");

        Ok(())
    }

    #[test]
    fn test_learned_vs_fixed_restrictions() -> Result<(), KohoError> {
        let sheaf_learned = create_triangle_sheaf()?;
        assert!(sheaf_learned.learned, "Sheaf should have learned=true");

        let mut sheaf_fixed = create_triangle_sheaf()?;
        sheaf_fixed.learned = false;
        let input = sheaf_learned.get_k_cochain(0)?;

        let target_data = vec![0.8f32, 0.6f32, 0.4f32];
        let target = Matrix::from_slice(&target_data, 1, 3, Device::Cpu, DType::F32)
            .map_err(KohoError::Candle)?;
        let training_data = vec![(input.clone(), target.clone())];

        // train learned network
        let mut network_learned = SheafNN::init(0, false, LossKind::MSE, sheaf_learned);
        let layer_learned = DiffusionLayer::new(0, Activations::Linear, &network_learned.sheaf)?;
        network_learned.sequential(vec![layer_learned]);

        let metrics_learned = network_learned.train(
            &training_data,
            50,
            false,
            OptimKind::Adam,
            0.01,
            OptimizerParams::Else,
        )?;

        // train fixed network
        let mut network_fixed = SheafNN::init(0, false, LossKind::MSE, sheaf_fixed);
        let layer_fixed = DiffusionLayer::new(0, Activations::Linear, &network_fixed.sheaf)?;
        network_fixed.sequential(vec![layer_fixed]);

        let metrics_fixed = network_fixed.train_debug(
            &training_data,
            50,
            false,
            OptimKind::Adam,
            0.01,
            OptimizerParams::Else,
        )?;

        println!(
            "Learned restrictions final loss: {}",
            metrics_learned.final_loss
        );
        println!(
            "Fixed restrictions final loss: {}",
            metrics_fixed.final_loss
        );

        Ok(())
    }

    #[test]
    fn test_multiple_diffusion_layers() -> Result<(), KohoError> {
        let sheaf = create_triangle_sheaf()?;
        let input = sheaf.get_k_cochain(0)?;

        let target_data = vec![0.9f32, 0.8f32, 0.7f32];
        let target = Matrix::from_slice(&target_data, 1, 3, Device::Cpu, DType::F32)
            .map_err(KohoError::Candle)?;
        let training_data = vec![(input, target)];

        let mut network = SheafNN::init(0, false, LossKind::MSE, sheaf);

        let layer1 = DiffusionLayer::new(0, Activations::Softmax, &network.sheaf)?;
        let layer2 = DiffusionLayer::new(0, Activations::Tanh, &network.sheaf)?;
        let layer3 = DiffusionLayer::new(0, Activations::Sigmoid, &network.sheaf)?;

        network.sequential(vec![layer1, layer2, layer3]);

        let metrics = network.train_debug(
            &training_data,
            175,
            false,
            OptimKind::Adam,
            0.15,
            OptimizerParams::Else,
        )?;

        println!("Multi-layer network final loss: {}", metrics.final_loss);

        let test_input = network.sheaf.get_k_cochain(0)?;
        let output = network.forward(test_input, false)?;

        assert_eq!(output.rows(), 1, "Output should have 3 vertices");

        Ok(())
    }
}
