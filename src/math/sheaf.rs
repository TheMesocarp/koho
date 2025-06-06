//! Cellular sheaf mathematics implementation.
//!
//! This module provides data structures and algorithms for working with cellular sheaves,
//! which are mathematical constructs that assign vector spaces to cells in a cell complex
//! and linear maps between these spaces. Cellular sheaves can be used for data analysis,
//! signal processing on topological domains, and other applications where the topology
//! of data is important.

use candle_core::{DType, Device, Result as CandleResult, Var, WithDType};
use std::collections::HashMap;

use crate::{
    error::KohoError,
    math::{
        cell::{Cell, Skeleton},
        tensors::{Matrix, Vector},
    },
};

/// Represents a section of a sheaf, which is a vector of data associated with a cell.
pub struct Section(pub Vector);

impl Section {
    /// Creates a new section from a slice of data.
    ///
    /// # Arguments
    /// * `tensor` - The data to initialize the section with
    /// * `dimension` - The dimension of the vector space this section lives in
    /// * `device` - The device to store the tensor on (CPU/GPU)
    /// * `dtype` - The data type of the tensor elements
    pub fn new<T: WithDType>(
        tensor: &[T],
        dimension: usize,
        device: Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        Ok(Self(Vector::from_slice(tensor, dimension, device, dtype)?))
    }
}

/// A cellular sheaf defined over a cell complex.
///
/// Cellular sheaves assign data (vector spaces) to cells in a cell complex,
/// along with restriction maps that specify how data on higher-dimensional cells
/// relates to data on lower-dimensional cells.
pub struct CellularSheaf {
    /// The underlying cell complex structure
    pub cells: Skeleton,
    /// Vector spaces (stalks) assigned to each cell, organized by dimension
    pub section_spaces: Vec<Vec<Section>>,
    /// Maps between vector spaces on different cells (restriction maps)
    pub restrictions: HashMap<(usize, usize, usize, usize), Matrix>,
    /// Signs indicating the orientation relationship between cells
    pub interlinked: HashMap<(usize, usize, usize, usize), i8>,
    /// Global sections of the sheaf (data consistent across all cells)
    pub global_sections: Vec<Section>,
    /// Device for tensor operations (CPU/GPU)
    pub device: Device,
    /// Data type for tensor elements
    pub dtype: DType,
    /// Indicates of the sheaf learns restrictions, or if they are provided
    pub learned: bool,
}

impl CellularSheaf {
    /// Initializes a new, empty cellular sheaf.
    ///
    /// # Arguments
    /// * `dtype` - The data type to use for tensor elements
    /// * `device` - The device to use for tensor operations
    pub fn init(dtype: DType, device: Device, learned: bool) -> Self {
        Self {
            cells: Skeleton::init(),
            section_spaces: Vec::new(),
            restrictions: HashMap::new(),
            interlinked: HashMap::new(),
            global_sections: Vec::new(),
            device,
            dtype,
            learned,
        }
    }

    /// Attaches a cell to the base cell complex and creates a corresponding section space.
    ///
    /// # Arguments
    /// * `cell` - The cell to attach
    /// * `data` - The section data to associate with this cell
    ///
    /// # Returns
    /// A tuple of the dimension and index of the newly attached cell
    pub fn attach(
        &mut self,
        cell: Cell,
        data: Section,
        upper: Option<&[usize]>,
        lower: Option<&[usize]>,
    ) -> Result<(usize, usize), KohoError> {
        let k = cell.dimension;
        let idx = self.cells.attach(cell, upper, lower)?;
        let current_dim = self.section_spaces.len();
        match current_dim.cmp(&k) {
            std::cmp::Ordering::Less => {
                return Err(KohoError::DimensionMismatch);
            }
            std::cmp::Ordering::Equal => {
                self.section_spaces.push(vec![data]);
            }
            std::cmp::Ordering::Greater => {
                self.section_spaces[k].push(data);
            }
        }
        Ok((k, idx))
    }

    /// Sets a restriction map between two cells.
    ///
    /// Restriction maps define how data transforms when moving from one cell to another.
    ///
    /// # Arguments
    /// * `k` - The dimension of the source cell
    /// * `cell_id` - The index of the source cell
    /// * `final_k` - The dimension of the target cell
    /// * `final_cell` - The index of the target cell
    /// * `map` - The linear transformation matrix
    /// * `interlink` - Sign indicating orientation relationship (-1 or 1)
    pub fn set_restriction(
        &mut self,
        k: usize,
        cell_id: usize,
        higher_cell: usize,
        map: Matrix,
        interlink: i8,
    ) -> Result<(), KohoError> {
        if cell_id >= self.cells.cells[k].len() || higher_cell >= self.cells.cells[k + 1].len() {
            return Err(KohoError::InvalidCellIdx);
        }
        self.restrictions
            .insert((k, cell_id, k + 1, higher_cell), map);
        self.interlinked
            .insert((k, cell_id, k + 1, higher_cell), interlink);
        Ok(())
    }

    pub fn generate_initial_restrictions(&mut self, noise_limit: f64) -> Result<(), KohoError> {
        let dim = self.cells.dimension;
        for i in 0..dim - 1 {
            let sections = &self.section_spaces[i];
            let upper_sections = &self.section_spaces[i + 1];
            if sections.is_empty() {
                return Err(KohoError::NoSections { i });
            }
            if upper_sections.is_empty() {
                return Err(KohoError::NoSections { i: i + 1 });
            }
            let stalk_dim = sections[0].0.dimension();
            let stalk_dim_upper = upper_sections[0].0.dimension();

            let id = Matrix::identity(stalk_dim_upper, stalk_dim, self.device.clone(), self.dtype)
                .map_err(KohoError::Candle)?;

            for (j, _) in sections.iter().enumerate() {
                let (_, uppers) = self.cells.incidences(i, j)?;
                for k in uppers {
                    let noise =
                        Matrix::rand(stalk_dim_upper, stalk_dim, self.device.clone(), self.dtype)
                            .map_err(KohoError::Candle)?
                            .scale(noise_limit)
                            .map_err(KohoError::Candle)?;
                    let fix = id.clone().add(&noise).map_err(KohoError::Candle)?;
                    self.restrictions.insert((i, j, i + 1, *k), fix);
                }
            }
        }
        Ok(())
    }

    /// Computes the k-coboundary operator, which maps k-cochains to (k+1)-cochains.
    ///
    /// This is a key operator in sheaf cohomology that captures how data propagates
    /// from lower to higher-dimensional cells.
    ///
    /// # Arguments
    /// * `k` - The dimension of the input cochain
    /// * `k_cochain` - The input k-cochain as a vector of vectors
    /// * `stalk_dim_output` - The dimension of the output stalk
    ///
    /// # Returns
    /// The resulting (k+1)-cochain
    fn k_coboundary(
        &self,
        k: usize,
        k_cochain: Vec<Vector>,
        stalk_dim_output: usize,
    ) -> Result<Vec<Vector>, KohoError> {
        println!("here at least");
        let num_k_plus_1_cells = self.cells.cells.get(k + 1).map_or(0, |cells| cells.len());
        if num_k_plus_1_cells == 0 && k + 1 < self.cells.cells.len() {
            return Ok(Vec::new());
        } else if k + 1 >= self.cells.cells.len() {
            return Err(KohoError::DimensionMismatch);
        }
        let mut output_k_plus_1_cochain = vec![
            Vector::from_slice(
                &vec![0f32; stalk_dim_output],
                stalk_dim_output,
                self.device.clone(),
                self.dtype
            )
            .map_err(KohoError::Candle)?;
            num_k_plus_1_cells
        ];
        let tau_dim = k + 1;
        for (sigma_idx, section) in k_cochain.iter().enumerate() {
            let (upper, _) = self.cells.incidences(k, sigma_idx)?;
            for i in upper {
                if let Some(r) = self.restrictions.get(&(k, sigma_idx, tau_dim, *i)) {
                    let mut term = r.matvec(section).map_err(KohoError::Candle)?;

                    if let Some(incidence_sign) = self.interlinked.get(&(k, sigma_idx, tau_dim, *i))
                    {
                        if *incidence_sign < 0 {
                            term = term.scale(-1.0).map_err(KohoError::Candle)?;
                        }
                    }
                    output_k_plus_1_cochain[*i] = output_k_plus_1_cochain[*i]
                        .add(&term)
                        .map_err(KohoError::Candle)?;
                }
            }
        }
        Ok(output_k_plus_1_cochain)
    }

    /// Computes the adjoint of the k-th coboundary operator ((delta^k)*).
    ///
    /// This operator goes in the reverse direction of the coboundary, from (k+1)-cochains
    /// to k-cochains, and is essential for defining the Hodge Laplacian.
    ///
    /// # Arguments
    /// * `k` - The dimension to compute the adjoint for
    /// * `k_coboundary_output` - The (k+1)-cochain input
    /// * `stalk_dim_output` - The dimension of the output stalk
    ///
    /// # Returns
    /// The resulting k-cochain
    fn k_adjoint_coboundary(
        &self,
        k: usize,
        k_coboundary_output: Vec<Vector>,
        stalk_dim_output: usize,
    ) -> Result<Vec<Vector>, KohoError> {
        let num_k_cells = self.cells.cells.get(k).map_or(0, |cells| cells.len());
        if num_k_cells == 0 && k < self.cells.cells.len() {
            return Ok(Vec::new());
        } else if k >= self.cells.cells.len() {
            return Err(KohoError::DimensionMismatch);
        }

        let mut output_k_cochain = vec![
            Vector::from_slice(
                &vec![0f32; stalk_dim_output],
                stalk_dim_output,
                self.device.clone(),
                self.dtype
            )
            .map_err(KohoError::Candle)?;
            num_k_cells
        ];

        for (tau_idx, y_tau) in k_coboundary_output.iter().enumerate() {
            let tau_cell_dim = k + 1;

            let (lower_incident_cells, _) = &self.cells.incidences(tau_cell_dim, tau_idx)?;

            for sigma_idx in *lower_incident_cells {
                if let Some(r) = self
                    .restrictions
                    .get(&(k, *sigma_idx, tau_cell_dim, tau_idx))
                {
                    let mut term = r
                        .transpose_matvec(y_tau)
                        .map_err(KohoError::Candle)?;

                    if let Some(incidence_sign) =
                        self.interlinked
                            .get(&(k, *sigma_idx, tau_cell_dim, tau_idx))
                    {
                        if *incidence_sign < 0 {
                            term = term.scale(-1.0).map_err(KohoError::Candle)?;
                        }
                    }

                    output_k_cochain[*sigma_idx] = output_k_cochain[*sigma_idx]
                        .add(&term)
                        .map_err(KohoError::Candle)?;
                }
            }
        }
        Ok(output_k_cochain)
    }

    /// Retrieves the cochain (collection of vector data) for a given dimension k.
    ///
    /// # Arguments
    /// * `k` - The dimension to retrieve the cochain for
    ///
    /// # Returns
    /// A matrix containing all vectors in the k-cochain
    pub fn get_k_cochain(&self, k: usize) -> Result<Matrix, KohoError> {
        if k >= self.section_spaces.len() {
            return Err(KohoError::DimensionMismatch);
        }
        let k_sections = &self.section_spaces[k];
        let k_cochain: Vec<Vector> = k_sections.iter().map(|section| section.0.clone()).collect();
        Matrix::from_vecs(k_cochain).map_err(KohoError::Candle)
    }

    /// Computes the k-th Hodge Laplacian operator.
    ///
    /// The Hodge Laplacian combines the coboundary and its adjoint to create an operator
    /// that measures how "harmonic" data is across the sheaf. It's used for spectral
    /// analysis, smoothing, and finding patterns in data.
    ///
    /// # Arguments
    /// * `k` - The dimension to compute the Laplacian for
    /// * `k_cochain` - The input k-cochain as a matrix
    /// * `down_included` - Whether to include the "down" component (from (k-1)-cells)
    ///
    /// # Returns
    /// The Hodge Laplacian applied to the input cochain
    pub fn k_hodge_laplacian(
        &self,
        k: usize,
        k_cochain: Matrix,
        down_included: bool,
    ) -> Result<Matrix, KohoError> {
        let vecs = k_cochain.to_vectors().map_err(KohoError::Candle)?;

        let k_plus_stalk_dim = self.section_spaces[k + 1][0].0.dimension();
        let k_stalk_dim = self.section_spaces[k][0].0.dimension();
        let k_minus_stalk_dim = self.section_spaces[k - 1][0].0.dimension();

        let up_a = self.k_coboundary(k, vecs.clone(), k_plus_stalk_dim)?;
        let up_b = self.k_adjoint_coboundary(k, up_a, k_stalk_dim)?;
        if down_included {
            let down_a = self.k_adjoint_coboundary(k, vecs, k_minus_stalk_dim)?;
            let down_b = self.k_coboundary(k, down_a, k_stalk_dim)?;
            let out = Matrix::from_vecs(up_b)
                .map_err(KohoError::Candle)?
                .add(&Matrix::from_vecs(down_b).map_err(KohoError::Candle)?)
                .map_err(KohoError::Candle)?;
            return Ok(out);
        }
        Matrix::from_vecs(up_b).map_err(KohoError::Candle)
    }

    pub fn parameters(&self, k: usize, down_included: bool) -> Vec<Var> {
        self.restrictions
            .iter()
            .filter_map(|(key, x)| {
                if key.0 == k || (down_included && k - key.0 == 1) {
                    Some(x.inner().clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }

    pub fn parameters_mut(&mut self, k: usize, down_included: bool) -> Vec<&mut Var> {
        self.restrictions
            .iter_mut()
            .filter_map(|(key, x)| {
                if key.0 == k || (down_included && k - key.0 == 1) {
                    Some(x.inner_mut())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_empty_sheaf() {
        let sh = CellularSheaf::init(DType::F32, Device::Cpu, false);
        // Only the 0-skeleton should exist, with no sections or maps
        assert_eq!(
            sh.cells.cells.len(),
            1,
            "Initial skeleton has only 0-cells layer"
        );
        assert!(
            sh.section_spaces.is_empty(),
            "No section spaces before any attach"
        );
        assert!(sh.restrictions.is_empty(), "No restriction maps initially");
        assert!(sh.interlinked.is_empty(), "No orientation signs initially");
        assert!(
            sh.global_sections.is_empty(),
            "No global sections initially"
        );
    }

    #[test]
    fn test_attach_sections_and_cells() {
        let mut sh = CellularSheaf::init(DType::F32, Device::Cpu, false);
        // Attach two vertices
        let sec0 = Section::new(&[1.0f32], 1, Device::Cpu, DType::F32).unwrap();
        let (k0, idx0) = sh.attach(Cell::new(0), sec0, None, None).unwrap();
        assert_eq!((k0, idx0), (0, 0));
        let sec1 = Section::new(&[2.0f32], 1, Device::Cpu, DType::F32).unwrap();
        let (k0b, idx0b) = sh.attach(Cell::new(0), sec1, None, None).unwrap();
        assert_eq!((k0b, idx0b), (0, 1));
        assert_eq!(sh.cells.cells[0].len(), 2, "Two 0-cells attached");
        assert_eq!(sh.section_spaces[0].len(), 2, "Two sections for 0-cells");

        // Attach an edge between the two vertices
        let edge_sec = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32).unwrap();
        let (k1, idx1) = sh
            .attach(Cell::new(1), edge_sec, None, Some(&[0, 1]))
            .unwrap();
        assert_eq!((k1, idx1), (1, 0));
        assert_eq!(sh.cells.cells[1].len(), 1, "One 1-cell attached");
        assert_eq!(sh.section_spaces[1].len(), 1, "One section for 1-cell");
    }

    #[test]
    fn test_set_restriction_errors_and_success() {
        let mut sh = CellularSheaf::init(DType::F32, Device::Cpu, false);
        // No cells to restrict: should error
        let map = Matrix::zeros(3, 3, Device::Cpu, DType::F32).unwrap();
        assert!(matches!(
            sh.set_restriction(0, 0, 0, map.clone(), 1),
            Err(KohoError::InvalidCellIdx)
        ));

        // Attach one vertex and one edge
        let vsec = Section::new(&[1.0f32], 1, Device::Cpu, DType::F32).unwrap();
        sh.attach(Cell::new(0), vsec, None, None).unwrap();
        let esec = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32).unwrap();
        sh.attach(Cell::new(1), esec, None, Some(&[0])).unwrap();

        // Invalid cell_id: too large
        let bad_map = Matrix::from_vecs(vec![sh.section_spaces[0][0].0.clone()]).unwrap();
        assert!(matches!(
            sh.set_restriction(0, 1, 0, bad_map.clone(), -1),
            Err(KohoError::InvalidCellIdx)
        ));

        // Valid restriction
        let valid_map = Matrix::from_vecs(vec![sh.section_spaces[0][0].0.clone()]).unwrap();
        assert!(sh.set_restriction(0, 0, 0, valid_map, 1).is_ok());
    }

    #[test]
    fn test_get_k_cochain_and_errors() {
        let mut sh = CellularSheaf::init(DType::F32, Device::Cpu, false);
        // No sections yet -> DimensionMismatch
        assert!(matches!(
            sh.get_k_cochain(0),
            Err(KohoError::DimensionMismatch)
        ));

        // Attach a single 0-cell and retrieve its cochain
        let sec = Section::new(&[3.0f32], 1, Device::Cpu, DType::F32).unwrap();
        sh.attach(Cell::new(0), sec, None, None).unwrap();
        let mat = sh.get_k_cochain(0).unwrap();
        // Should be a 1x1 matrix containing [3.0]
        assert_eq!(&[mat.rows(), mat.cols()], &[1, 1]);
    }

    // Helper function to create a simple sheaf with two vertices and an edge
    fn setup_simple_sheaf() -> Result<CellularSheaf, KohoError> {
        let mut sheaf = CellularSheaf::init(DType::F32, Device::Cpu, false);

        // Attach two 0-cells (vertices)
        let v0_data =
            Section::new(&[1.0f32], 1, Device::Cpu, DType::F32).map_err(KohoError::Candle)?;
        let (_, v0_idx) = sheaf.attach(Cell::new(0), v0_data, None, None)?;

        let v1_data =
            Section::new(&[2.0f32], 1, Device::Cpu, DType::F32).map_err(KohoError::Candle)?;
        let (_, v1_idx) = sheaf.attach(Cell::new(0), v1_data, None, None)?;

        // Attach 1-cell (edge) connecting vertices
        let e_data =
            Section::new(&[0.0f32], 1, Device::Cpu, DType::F32).map_err(KohoError::Candle)?;
        let (_, e_idx) = sheaf.attach(Cell::new(1), e_data, None, Some(&[v0_idx, v1_idx]))?;

        // Set restriction maps (vertex -> edge maps)
        let r_v0 = Matrix::from_slice(&[1.0f32], 1, 1, Device::Cpu, DType::F32)
            .map_err(KohoError::Candle)?;
        sheaf.set_restriction(0, v0_idx, e_idx, r_v0, 1)?;

        let r_v1 = Matrix::from_slice(&[-1.0f32], 1, 1, Device::Cpu, DType::F32)
            .map_err(KohoError::Candle)?;
        sheaf.set_restriction(0, v1_idx, e_idx, r_v1, 1)?;

        Ok(sheaf)
    }

    #[test]
    fn test_coboundary_computation() -> Result<(), KohoError> {
        let sheaf = setup_simple_sheaf()?;

        // Get 0-cochain (vertex data)
        let k_cochain = sheaf
            .get_k_cochain(0)?
            .to_vectors()
            .map_err(KohoError::Candle)?;
        assert_eq!(k_cochain.len(), 2, "Should have two 0-cells");

        // Compute coboundary (should produce edge data)
        let result = sheaf.k_coboundary(0, k_cochain, 1)?;
        assert_eq!(result.len(), 1, "Should have one 1-cell");

        // Verify edge value: 1*1.0 (from v0) + (-1)*2.0 (from v1) = -1.0
        let edge_value = result[0]
            .inner()
            .squeeze(1)
            .map_err(KohoError::Candle)?
            .to_vec1::<f32>()
            .map_err(KohoError::Candle)?[0];
        assert!(
            (edge_value - (-1.0f32)).abs() < 1e-6,
            "Edge value mismatch, got: {edge_value}"
        );

        Ok(())
    }
}
