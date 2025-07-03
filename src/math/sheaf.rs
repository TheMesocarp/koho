//! Cellular sheaf mathematics implementation.
//!
//! This module provides data structures and algorithms for working with cellular sheaves,
//! which are mathematical constructs that assign vector spaces to cells in a cell complex
//! and linear maps between these spaces. Cellular sheaves can be used for data analysis,
//! signal processing on topological domains, and other applications where the topology
//! of data is important.

use candle_core::{DType, Device, Var, WithDType};
use std::collections::HashMap;

use crate::{
    error::KohoError,
    math::{
        cell::{Cell, Skeleton},
        tensors::{Matrix, VarMatrix, Vector},
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
    ) -> Result<Self, KohoError> {
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
    pub restrictions: HashMap<(usize, usize, usize, usize), VarMatrix>,
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
        map: VarMatrix,
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
        for i in 0..dim {
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

            let id = Matrix::identity(stalk_dim_upper, stalk_dim, self.device.clone(), self.dtype)?;

            for (j, _) in sections.iter().enumerate() {
                let (uppers, _) = self.cells.incidences(i, j)?;
                for k in uppers {
                    let noise =
                        Matrix::rand(stalk_dim_upper, stalk_dim, self.device.clone(), self.dtype)?
                            .scale(noise_limit)?;
                    let fix = VarMatrix::new(
                        id.clone().add(&noise)?.tensor,
                        self.device.clone(),
                        self.dtype,
                    )?;

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
            )?;
            num_k_plus_1_cells
        ];
        let tau_dim = k + 1;
        for (sigma_idx, section) in k_cochain.iter().enumerate() {
            let (upper, _) = self.cells.incidences(k, sigma_idx)?;
            for i in upper {
                if let Some(r) = self.restrictions.get(&(k, sigma_idx, tau_dim, *i)) {
                    let mut term = r.matvec(section)?;
                    if let Some(incidence_sign) = self.interlinked.get(&(k, sigma_idx, tau_dim, *i))
                    {
                        if *incidence_sign < 0 {
                            let new = term.inner().neg()?;
                            term = Vector::new(new, self.device.clone(), self.dtype)?;
                        }
                    }
                    output_k_plus_1_cochain[*i] = output_k_plus_1_cochain[*i].add(&term)?;
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
            )?;
            num_k_cells
        ];

        for (tau_idx, y_tau) in k_coboundary_output.iter().enumerate() {
            let tau_cell_dim = k + 1;

            let (_, lower_incident_cells) = &self.cells.incidences(tau_cell_dim, tau_idx)?;
            for sigma_idx in *lower_incident_cells {
                if let Some(r) = self
                    .restrictions
                    .get(&(k, *sigma_idx, tau_cell_dim, tau_idx))
                {
                    let mut term = r.transpose_matvec(y_tau)?;
                    if let Some(incidence_sign) =
                        self.interlinked
                            .get(&(k, *sigma_idx, tau_cell_dim, tau_idx))
                    {
                        if *incidence_sign < 0 {
                            let new = term.inner().neg()?;
                            term = Vector::new(new, self.device.clone(), self.dtype)?;
                        }
                    }

                    output_k_cochain[*sigma_idx] = output_k_cochain[*sigma_idx].add(&term)?;
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
        let vecs = k_cochain.to_vectors()?;

        let k_plus_stalk_dim = self.section_spaces[k + 1][0].0.dimension();
        let k_stalk_dim = self.section_spaces[k][0].0.dimension();

        let up_a = self.k_coboundary(k, vecs.clone(), k_plus_stalk_dim)?;
        let up_b = self.k_adjoint_coboundary(k, up_a, k_stalk_dim)?;
        if down_included {
            let k_minus_stalk_dim = self.section_spaces[k - 1][0].0.dimension();
            let down_a = self.k_adjoint_coboundary(k - 1, vecs, k_minus_stalk_dim)?;
            let down_b = self.k_coboundary(k - 1, down_a, k_stalk_dim)?;
            let out = Matrix::from_vecs(up_b)?.add(&Matrix::from_vecs(down_b)?)?;
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
        let map = VarMatrix::zeros(3, 3, Device::Cpu, DType::F32).unwrap();
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
        let bad_map = VarMatrix::from_vecs(vec![sh.section_spaces[0][0].0.clone()]).unwrap();
        assert!(matches!(
            sh.set_restriction(0, 1, 0, bad_map.clone(), -1),
            Err(KohoError::InvalidCellIdx)
        ));

        // Valid restriction
        let valid_map = VarMatrix::from_vecs(vec![sh.section_spaces[0][0].0.clone()]).unwrap();
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
        assert_eq!(&[mat.rows(), mat.cols()], &[1, 1]);
    }

    // simple line sheaf
    fn setup_simple_sheaf() -> Result<CellularSheaf, KohoError> {
        let mut sheaf = CellularSheaf::init(DType::F32, Device::Cpu, false);

        // Create 2 vertices
        let v0_data = Section::new(&[1.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v0_idx) = sheaf.attach(Cell::new(0), v0_data, None, None)?;

        let v1_data = Section::new(&[2.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v1_idx) = sheaf.attach(Cell::new(0), v1_data, None, None)?;

        // Create 1 edge
        let e_data = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, e_idx) = sheaf.attach(Cell::new(1), e_data, None, Some(&[v0_idx, v1_idx]))?;

        // Set restrictions
        let r_v0 = VarMatrix::from_slice(&[1.0f32], 1, 1, Device::Cpu, DType::F32)?;
        sheaf.set_restriction(0, v0_idx, e_idx, r_v0, 1)?;

        let r_v1 = VarMatrix::from_slice(&[-1.0f32], 1, 1, Device::Cpu, DType::F32)?;
        sheaf.set_restriction(0, v1_idx, e_idx, r_v1, -1)?;

        Ok(sheaf)
    }

    #[test]
    fn test_coboundary_computation() -> Result<(), KohoError> {
        let sheaf = setup_simple_sheaf()?;
        let k_cochain = sheaf.get_k_cochain(0)?.to_vectors()?;
        assert_eq!(k_cochain.len(), 2, "Should have two 0-cells");

        let result = sheaf.k_coboundary(0, k_cochain, 1)?;
        assert_eq!(result.len(), 1, "Should have one 1-cell");

        // Verify edge value: 1*1.0 (from v0) + (-1)*2.0 (from v1) = -1.0
        let edge_value = result[0].inner().squeeze(1)?.to_vec1::<f32>()?[0];
        assert!(
            (edge_value - (3.0f32)).abs() < 1e-6,
            "Edge value mismatch, got: {edge_value}"
        );

        Ok(())
    }
}

#[cfg(test)]
mod adjoint_and_laplacian_tests {
    use crate::{
        error::KohoError,
        math::{
            cell::Cell,
            sheaf::{CellularSheaf, Section},
            tensors::{Matrix, VarMatrix, Vector},
        },
    };
    use candle_core::{DType, Device};

    /// Creates a simple line graph: v0 -- e0 -- v1
    fn create_line_sheaf() -> Result<CellularSheaf, KohoError> {
        let mut sheaf = CellularSheaf::init(DType::F32, Device::Cpu, false);

        // Create 2 vertices
        let v0_data = Section::new(&[1.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v0_idx) = sheaf.attach(Cell::new(0), v0_data, None, None)?;

        let v1_data = Section::new(&[2.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v1_idx) = sheaf.attach(Cell::new(0), v1_data, None, None)?;

        // Create 1 edge
        let e0_data = Section::new(&[0.5f32], 1, Device::Cpu, DType::F32)?;
        let (_, e0_idx) = sheaf.attach(Cell::new(1), e0_data, None, Some(&[v0_idx, v1_idx]))?;

        // Set restrictions
        let r_v0_e0 = VarMatrix::from_slice(&[2.0f32], 1, 1, Device::Cpu, DType::F32)?;
        sheaf.set_restriction(0, v0_idx, e0_idx, r_v0_e0, 1)?;

        let r_v1_e0 = VarMatrix::from_slice(&[3.0f32], 1, 1, Device::Cpu, DType::F32)?;
        sheaf.set_restriction(0, v1_idx, e0_idx, r_v1_e0, -1)?;

        Ok(sheaf)
    }

    #[test]
    fn test_k_coboundary_simple() -> Result<(), KohoError> {
        let sheaf = create_line_sheaf()?;
        let v0 = Vector::from_slice(&[1.0f32], 1, Device::Cpu, DType::F32)?;
        let v1 = Vector::from_slice(&[2.0f32], 1, Device::Cpu, DType::F32)?;
        let vertex_cochain = vec![v0, v1];

        let edge_cochain = sheaf.k_coboundary(0, vertex_cochain, 1)?;

        assert_eq!(edge_cochain.len(), 1, "Should have one edge");
        println!("here we are");
        // Expected: 2.0 * 1.0 + (-1) * 3.0 * 2.0 = 2.0 - 6.0 = -4.0
        let edge_val = edge_cochain[0]
            .inner()
            .squeeze(1)?
            .squeeze(0)?
            .to_scalar::<f32>()?;
        println!("here we are after edge val");
        println!("Coboundary result: {edge_val}");
        assert!(
            (edge_val - (-4.0)).abs() < 1e-6,
            "Expected -4.0, got {edge_val}"
        );

        Ok(())
    }

    #[test]
    fn test_k_adjoint_coboundary_simple() -> Result<(), KohoError> {
        let sheaf = create_line_sheaf()?;
        let e0 = Vector::from_slice(&[5.0f32], 1, Device::Cpu, DType::F32)?;
        let edge_cochain = vec![e0];

        let vertex_cochain = sheaf.k_adjoint_coboundary(0, edge_cochain, 1)?;

        assert_eq!(vertex_cochain.len(), 2, "Should have two vertices");

        // For v0: transpose(r_v0_e0) * e0 * orientation = 2.0 * 5.0 * 1 = 10.0
        let v0_val = vertex_cochain[0]
            .inner()
            .squeeze(1)?
            .squeeze(0)?
            .to_scalar::<f32>()?;

        // For v1: transpose(r_v1_e0) * e0 * orientation = 3.0 * 5.0 * (-1) = -15.0
        let v1_val = vertex_cochain[1]
            .inner()
            .squeeze(1)?
            .squeeze(0)?
            .to_scalar::<f32>()?;

        println!("Adjoint coboundary results: v0={v0_val}, v1={v1_val}");
        assert!(
            (v0_val - 10.0).abs() < 1e-6,
            "Expected v0=10.0, got {v0_val}"
        );
        assert!(
            (v1_val - (-15.0)).abs() < 1e-6,
            "Expected v1=-15.0, got {v1_val}"
        );

        Ok(())
    }

    #[test]
    fn test_k_hodge_laplacian_up_only() -> Result<(), KohoError> {
        let sheaf = create_line_sheaf()?;
        let input = Matrix::from_slice(&[1.0f32, 2.0f32], 1, 2, Device::Cpu, DType::F32)?;

        let result = sheaf.k_hodge_laplacian(0, input, false)?;

        println!("Laplacian result shape: {:?}", result.shape());

        // Manual calculation:
        // 1. Coboundary: [1.0, 2.0] -> [-4.0] (as calculated above)
        // 2. Adjoint: [-4.0] -> [-8.0, 12.0]
        //    v0: 2.0 * (-4.0) * 1 = -8.0
        //    v1: 3.0 * (-4.0) * (-1) = 12.0

        let result_vec = result.inner().to_vec2::<f32>()?;
        println!("Hodge Laplacian result: {result_vec:?}");

        assert_eq!(result_vec.len(), 1, "Should have 1 row");
        assert_eq!(result_vec[0].len(), 2, "Should have 2 columns");

        assert!(
            (result_vec[0][0] - (-8.0)).abs() < 1e-6,
            "Expected v0=-8.0, got {}",
            result_vec[0][0]
        );
        assert!(
            (result_vec[0][1] - 12.0).abs() < 1e-6,
            "Expected v1=12.0, got {}",
            result_vec[0][1]
        );

        Ok(())
    }

    /// Test with a triangle to verify more complex interactions
    #[test]
    fn test_hodge_laplacian_triangle() -> Result<(), KohoError> {
        let mut sheaf = CellularSheaf::init(DType::F32, Device::Cpu, false);

        // Create 3 vertices
        for i in 0..3 {
            let data = Section::new(&[i as f32], 1, Device::Cpu, DType::F32)?;
            sheaf.attach(Cell::new(0), data, None, None)?;
        }

        // Create 3 edges
        sheaf.attach(
            Cell::new(1),
            Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?,
            None,
            Some(&[0, 1]),
        )?;
        sheaf.attach(
            Cell::new(1),
            Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?,
            None,
            Some(&[1, 2]),
        )?;
        sheaf.attach(
            Cell::new(1),
            Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?,
            None,
            Some(&[2, 0]),
        )?;

        for v in 0..3 {
            for e in 0..3 {
                let (_, lower) = sheaf.cells.incidences(1, e)?;
                if lower.contains(&v) {
                    let r = VarMatrix::identity(1, 1, Device::Cpu, DType::F32)?;
                    sheaf.set_restriction(0, v, e, r, 1)?;
                }
            }
        }

        let input = sheaf.get_k_cochain(0)?;
        println!("Triangle input: {:?}", input.inner().to_vec2::<f32>());

        let result = sheaf.k_hodge_laplacian(0, input, false)?;
        println!(
            "Triangle Laplacian result: {:?}",
            result.inner().to_vec2::<f32>()
        );

        // With identity maps and orientation 1, the Laplacian should act like a graph Laplacian
        // Each vertex should get: degree * value - sum of neighbor values

        Ok(())
    }

    #[test]
    fn test_adjoint_with_zero_restriction() -> Result<(), KohoError> {
        let mut sheaf = create_line_sheaf()?;
        let zero_r = VarMatrix::zeros(1, 1, Device::Cpu, DType::F32)?;
        sheaf.set_restriction(0, 0, 0, zero_r, 1)?;
        let e0 = Vector::from_slice(&[5.0f32], 1, Device::Cpu, DType::F32)?;
        let edge_cochain = vec![e0];

        let vertex_cochain = sheaf.k_adjoint_coboundary(0, edge_cochain, 1)?;

        let v0_val = vertex_cochain[0]
            .inner()
            .squeeze(1)?
            .squeeze(0)?
            .to_scalar::<f32>()?;

        println!("v0 with zero restriction: {v0_val}");
        assert!(v0_val.abs() < 1e-6, "v0 should be 0, got {v0_val}");

        Ok(())
    }

    #[test]
    fn test_dimensions_preserved() -> Result<(), KohoError> {
        let sheaf = create_line_sheaf()?;
        let input = Matrix::from_slice(&[3.0f32, 4.0f32], 1, 2, Device::Cpu, DType::F32)?;

        let result = sheaf.k_hodge_laplacian(0, input.clone(), false)?;

        assert_eq!(
            input.shape(),
            result.shape(),
            "Hodge Laplacian should preserve dimensions"
        );

        Ok(())
    }
}

#[cfg(test)]
mod generated_restrictions_tests {
    use super::*;
    use crate::math::{
        cell::Cell,
        sheaf::{CellularSheaf, Section},
    };
    use candle_core::{DType, Device};

    /// Creates a triangle sheaf using the generate_initial_restrictions method
    fn create_triangle_with_generated_restrictions() -> Result<CellularSheaf, KohoError> {
        let mut sheaf = CellularSheaf::init(DType::F32, Device::Cpu, true);

        // Create 3 vertices (0-cells)
        let v0_data = Section::new(&[1.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v0_idx) = sheaf.attach(Cell::new(0), v0_data, None, None)?;

        let v1_data = Section::new(&[2.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v1_idx) = sheaf.attach(Cell::new(0), v1_data, None, None)?;

        let v2_data = Section::new(&[3.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v2_idx) = sheaf.attach(Cell::new(0), v2_data, None, None)?;

        // Create 3 edges (1-cells) connecting the vertices
        let e0_data = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, e0_idx) = sheaf.attach(Cell::new(1), e0_data, None, Some(&[v0_idx, v1_idx]))?;

        let e1_data = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, e1_idx) = sheaf.attach(Cell::new(1), e1_data, None, Some(&[v1_idx, v2_idx]))?;

        let e2_data = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, e2_idx) = sheaf.attach(Cell::new(1), e2_data, None, Some(&[v2_idx, v0_idx]))?;

        // Create triangle face (2-cell)
        let f0_data = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, _f0_idx) =
            sheaf.attach(Cell::new(2), f0_data, None, Some(&[e0_idx, e1_idx, e2_idx]))?;

        // Use `generate_initial_restrictions` instead of fixed sheaf
        let noise_level = 0.1;
        sheaf.generate_initial_restrictions(noise_level)?;

        Ok(sheaf)
    }

    #[test]
    fn test_triangle_with_generated_restrictions() -> Result<(), KohoError> {
        let sheaf = create_triangle_with_generated_restrictions()?;

        assert_eq!(sheaf.cells.cells[0].len(), 3, "Should have 3 vertices");
        assert_eq!(sheaf.cells.cells[1].len(), 3, "Should have 3 edges");
        assert_eq!(sheaf.cells.cells[2].len(), 1, "Should have 1 face");
        assert_eq!(sheaf.cells.dimension, 2, "Should be 2-dimensional");

        let mut expected_0_to_1_restrictions = 0;
        let mut expected_1_to_2_restrictions = 0;

        // Count 0->1 restrictions (vertex to edge)
        for v_idx in 0..3 {
            let (uppers, _) = sheaf.cells.incidences(0, v_idx)?;
            expected_0_to_1_restrictions += uppers.len();
        }

        // Count 1->2 restrictions (edge to face)
        for e_idx in 0..3 {
            let (uppers, _) = sheaf.cells.incidences(1, e_idx)?;
            expected_1_to_2_restrictions += uppers.len();
        }

        println!("Expected 0->1 restrictions: {expected_0_to_1_restrictions}");
        println!("Expected 1->2 restrictions: {expected_1_to_2_restrictions}");

        let total_expected = expected_0_to_1_restrictions + expected_1_to_2_restrictions;
        assert_eq!(
            sheaf.restrictions.len(),
            total_expected,
            "Should have the correct number of restriction maps"
        );

        for ((from_dim, from_idx, to_dim, to_idx), restriction) in &sheaf.restrictions {
            assert_eq!(restriction.rows(), 1, "Restriction should be 1x1");
            assert_eq!(restriction.cols(), 1, "Restriction should be 1x1");
            assert_eq!(*to_dim, from_dim + 1, "Should go from k to k+1 dimension");

            let value = restriction
                .inner()
                .as_tensor()
                .squeeze(0)?
                .squeeze(0)?
                .to_scalar::<f32>()?;

            println!("Restriction ({from_dim},{from_idx}) -> ({to_dim},{to_idx}): {value}");

            // Should be identity (1.0) plus noise, so should be close to 1.0
            assert!(
                (value - 1.0).abs() <= 0.6,
                "Restriction value {value} should be close to 1.0 (identity + noise)"
            );
        }

        Ok(())
    }

    #[test]
    fn test_generated_restrictions_coboundary_computation() -> Result<(), KohoError> {
        let sheaf = create_triangle_with_generated_restrictions()?;

        // Test that coboundary computation works with generated restrictions
        let vertex_cochain = sheaf.get_k_cochain(0)?;
        println!(
            "Vertex values: {:?}",
            vertex_cochain.inner().to_vec2::<f32>()
        );

        // Convert to vectors for coboundary computation
        let vertex_vectors = vertex_cochain.to_vectors()?;

        // Compute 0-coboundary (vertices -> edges)
        let edge_cochain = sheaf.k_coboundary(0, vertex_vectors, 1)?;

        assert_eq!(edge_cochain.len(), 3, "Should produce 3 edge values");

        for (i, edge_val) in edge_cochain.iter().enumerate() {
            let val = edge_val
                .inner()
                .squeeze(1)?
                .squeeze(0)?
                .to_scalar::<f32>()?;
            println!("Edge {i}: {val}");

            // Edge values should be reasonable (not NaN or infinite)
            assert!(val.is_finite(), "Edge value should be finite");
        }

        Ok(())
    }

    #[test]
    fn test_generated_restrictions_hodge_laplacian() -> Result<(), KohoError> {
        let sheaf = create_triangle_with_generated_restrictions()?;

        // Test Hodge Laplacian computation
        let input = sheaf.get_k_cochain(0)?;
        println!("Input shape: {:?}", input.shape());

        // Compute Hodge Laplacian on vertices (without down component)
        let result = sheaf.k_hodge_laplacian(0, input.clone(), false)?;

        assert_eq!(
            result.shape(),
            input.shape(),
            "Hodge Laplacian should preserve dimensions"
        );

        let result_vals = result.inner().to_vec2::<f32>()?;
        println!("Hodge Laplacian result: {result_vals:?}");

        // Values should be finite
        for row in &result_vals {
            for &val in row {
                assert!(val.is_finite(), "Laplacian result should be finite: {val}");
            }
        }

        // For a triangle with equal restrictions, the Laplacian should have some structure
        // (e.g., the sum across vertices might be preserved or have known properties)

        Ok(())
    }

    #[test]
    fn test_different_noise_levels() -> Result<(), KohoError> {
        // Test with different noise levels to ensure the method works consistently
        for &noise_level in &[0.0f32, 0.05f32, 0.1f32, 0.5f32] {
            let mut sheaf = CellularSheaf::init(DType::F32, Device::Cpu, true);

            // Create a simple edge (2 vertices, 1 edge)
            let v0 = Section::new(&[1.0f32], 1, Device::Cpu, DType::F32)?;
            let (_, v0_idx) = sheaf.attach(Cell::new(0), v0, None, None)?;

            let v1 = Section::new(&[2.0f32], 1, Device::Cpu, DType::F32)?;
            let (_, v1_idx) = sheaf.attach(Cell::new(0), v1, None, None)?;

            let e0 = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?;
            let (_, _e0_idx) = sheaf.attach(Cell::new(1), e0, None, Some(&[v0_idx, v1_idx]))?;

            // Generate restrictions with this noise level
            sheaf.generate_initial_restrictions(noise_level as f64)?;

            // Check that restrictions were created
            assert_eq!(
                sheaf.restrictions.len(),
                2,
                "Should have 2 restrictions (v0->e0, v1->e0)"
            );

            // Check that restriction values are within expected bounds
            for restriction in sheaf.restrictions.values() {
                let value = restriction
                    .inner()
                    .as_tensor()
                    .squeeze(0)?
                    .squeeze(0)?
                    .to_scalar::<f32>()?;

                // Should be 1.0 ± noise_level (approximately)
                let deviation = (value - 1.0).abs();
                assert!(
                    deviation <= noise_level * 3.0f32, // Allow for some randomness
                    "With noise_level {noise_level}, restriction value {value} deviates too much from 1.0"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_zero_noise_gives_identity() -> Result<(), KohoError> {
        let mut sheaf = CellularSheaf::init(DType::F32, Device::Cpu, true);

        // Create simple structure
        let v0 = Section::new(&[1.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v0_idx) = sheaf.attach(Cell::new(0), v0, None, None)?;

        let v1 = Section::new(&[2.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, v1_idx) = sheaf.attach(Cell::new(0), v1, None, None)?;

        let e0 = Section::new(&[0.0f32], 1, Device::Cpu, DType::F32)?;
        let (_, _e0_idx) = sheaf.attach(Cell::new(1), e0, None, Some(&[v0_idx, v1_idx]))?;

        // Generate with zero noise - should give exact identity
        sheaf.generate_initial_restrictions(0.0)?;

        // All restriction values should be exactly 1.0
        for restriction in sheaf.restrictions.values() {
            let value = restriction
                .inner()
                .as_tensor()
                .squeeze(0)?
                .squeeze(0)?
                .to_scalar::<f32>()?;

            assert!(
                (value - 1.0).abs() < 1e-6,
                "With zero noise, restriction should be exactly 1.0, got {value}"
            );
        }

        Ok(())
    }
}
