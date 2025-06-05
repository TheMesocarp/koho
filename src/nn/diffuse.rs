//! Neural diffusion layer implementation for cellular sheaves.
//!
//! This module provides a neural network layer that performs diffusion operations on cellular sheaves,
//! allowing for learned signal processing on topological domains. The diffusion is based on the
//! Hodge Laplacian operator and can be used in graph neural networks and other topological learning models.

use std::rc::Rc;

use candle_core::Var;

use crate::{
    error::KohoError,
    math::{sheaf::CellularSheaf, tensors::Matrix},
    nn::activate::Activations,
    Parameterized,
};

/// A neural network layer that performs diffusion operations on cellular sheaves.
///
/// This layer applies the Hodge Laplacian of a cellular sheaf, followed by
/// a learned weight matrix and an activation function. It can be used to build
/// graph neural networks and other architectures that operate on topological domains.
pub struct DiffusionLayer {
    /// The underlying cellular sheaf structure
    sheaf: Rc<CellularSheaf>,
    /// Learnable weights matrix for the diffusion operation
    weights: Var,
    /// Activation function to apply after diffusion
    activation: Activations,
}

impl DiffusionLayer {
    /// Creates a new diffusion layer for the given sheaf.
    ///
    /// # Arguments
    /// * `k` - The dimension to perform diffusion on
    /// * `activation` - The activation function to apply after diffusion
    /// * `sheaf` - Reference-counted pointer to the cellular sheaf
    ///
    /// # Returns
    /// A new DiffusionLayer instance with randomly initialized weights
    pub fn new(
        k: usize,
        activation: Activations,
        sheaf: Rc<CellularSheaf>,
    ) -> Result<Self, KohoError> {
        let device = sheaf.device.clone();
        let dtype = sheaf.dtype;
        let dim = sheaf.section_spaces[k][0].0.dimension();
        let weights = Matrix::rand(dim, dim, device, dtype).map_err(KohoError::Candle)?;
        let weights = Var::from_tensor(weights.inner()).map_err(KohoError::Candle)?;
        Ok(Self {
            sheaf,
            weights,
            activation,
        })
    }

    /// Performs diffusion on the given feature matrix.
    ///
    /// This method applies the Hodge Laplacian of the sheaf to the input features,
    /// followed by multiplication with the weights matrix and application of the activation function.
    ///
    /// # Arguments
    /// * `k` - The dimension to perform diffusion on
    /// * `k_features` - The input feature matrix for cells of dimension k
    /// * `down_included` - Whether to include diffusion from lower-dimensional cells
    ///
    /// # Returns
    /// The matrix of diffused features after applying weights and activation
    pub fn diffuse(
        &self,
        k: usize,
        k_features: Matrix,
        down_included: bool,
    ) -> Result<Matrix, KohoError> {
        let diff = self.sheaf.k_hodge_laplacian(k, k_features, down_included)?;
        let weighted = self
            .weights
            .as_tensor()
            .matmul(diff.inner())
            .map_err(KohoError::Candle)?;
        self.activation
            .activate(weighted)
            .map_err(KohoError::Candle)
    }

    /// Updates the weights of the diffusion layer.
    ///
    /// # Arguments
    /// * `weights` - The new weights to set for the layer
    pub fn update_weights(&mut self, weights: Var) {
        self.weights = weights
    }
}

/// Implementation of the Parameterized trait for DiffusionLayer.
///
/// This allows the layer to be used with optimizers that expect a list of parameters.
impl Parameterized for DiffusionLayer {
    /// Returns the learnable parameters of the layer.
    ///
    /// # Returns
    /// A vector containing the weights variable
    fn parameters(&self) -> Vec<Var> {
        vec![self.weights.clone()]
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Var> {
        vec![&mut self.weights]
    }
}
