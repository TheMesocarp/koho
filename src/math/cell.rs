//! Core traits and definitions for constructing a cell complex over arbitrary types.
//!
//! This module provides the fundamental building blocks for topological spaces and
//! CW complexes (cell complexes). It allows for the construction of spaces by
//! attaching cells of various dimensions along their boundaries.
//!
//! # Mathematical Background
//!
//! ## Topology
//!
//! A topology on a set X is a collection of subsets (called open sets) that satisfies:
//! - The empty set and X itself are open
//! - Arbitrary unions of open sets are open
//! - Finite intersections of open sets are open
//!
//! ## Cell Complex (CW Complex)
//!
//! A CW complex is built incrementally by:
//! - Starting with discrete points (0-cells)
//! - Attaching n-dimensional cells along their boundaries to the (n-1)-skeleton
//! - The n-skeleton consists of all cells of dimension ≤ n
//!
//! CW complexes provide a way to decompose topological spaces into simple building blocks:
//! - 0-cells: points
//! - 1-cells: line segments
//! - 2-cells: disks
//! - 3-cells: solid balls, etc.

use crate::error::KohoError;

/// A k-cell of arbitrary type and dimension
pub struct Cell {
    /// `k`, the dimension of the k-cell
    pub dimension: usize,
    /// Collection of upper incident cell IDs within the cell complex.
    ///
    /// Incident cells are those that share boundary components with this cell.
    /// For example, a 1-cell (edge) is incident to its endpoint 0-cells (vertices)
    pub upper: Vec<usize>,
    /// Collection of lower incident cell IDs within the cell complex.
    pub lower: Vec<usize>,
}

impl Cell {
    /// Generate a new k-cell
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            upper: Vec::new(),
            lower: Vec::new(),
        }
    }

    /// Find all the boundary cells and push incidence indexes to neighbors and itself
    fn attach(
        &mut self,
        skeleton: &mut Skeleton,
        upper: Option<&[usize]>,
        lower: Option<&[usize]>,
    ) {
        // Ensure skeleton has enough dimension layers
        while skeleton.cells.len() <= self.dimension {
            skeleton.cells.push(Vec::new());
        }
        
        let max = skeleton.cells[self.dimension].len();
        
        if let Some(upper) = upper {
            for i in upper {
                // Ensure the upper dimension exists
                while skeleton.cells.len() <= self.dimension + 1 {
                    skeleton.cells.push(Vec::new());
                }
                skeleton.cells[self.dimension + 1][*i].lower.push(max);
                self.upper.push(*i);
            }
        }
        
        if let Some(lower) = lower {
            for i in lower {
                skeleton.cells[self.dimension - 1][*i].upper.push(max);
                self.lower.push(*i);
            }
        }
    }
}

/// A `Skeleton` is a collection of `Cells` that have been glued together along `Cell::attach` maps.
///
/// In CW complex terminology, the n-skeleton consists of all cells of dimension ≤ n.
/// Building a CW complex involves constructing successive skeletons by attaching cells
/// of increasing dimension.
pub struct Skeleton {
    /// Dimension of the `Skeleton` (maximum cell dimension contained)
    pub dimension: usize,
    /// The collection of `Cells` forming the skeleton with [dimension][idx]
    pub cells: Vec<Vec<Cell>>,
}

impl Skeleton {
    /// Initialize a new `Skeleton`
    pub fn init() -> Self {
        Self {
            dimension: 0,
            cells: vec![Vec::new()],
        }
    }

    /// Attach a cell to the existing cell complex.
    ///
    /// This implements the core operation in building a CW complex: attaching new cells
    /// to the existing skeleton. The process involves:
    /// 1. Verifying the dimensional constraints (can only attach n-cells to (n-1)-skeleton)
    /// 2. Finding boundary points and their images under the attachment map
    /// 3. Updating incidence relationships between cells
    /// 4. Updating the skeleton's dimension if needed
    pub fn attach(
        &mut self,
        mut cell: Cell,
        upper: Option<&[usize]>,
        lower: Option<&[usize]>,
    ) -> Result<usize, KohoError> {
        let incoming_dim = cell.dimension as i64;
        if incoming_dim - self.dimension as i64 > 1 {
            return Err(KohoError::DimensionMismatch);
        }
        cell.attach(self, upper, lower);
        if cell.dimension < self.cells.len() {
            self.cells[cell.dimension].push(cell);
        }
        if incoming_dim > self.dimension as i64 {
            self.dimension = incoming_dim as usize
        }
        Ok(self.cells[incoming_dim as usize].len() - 1)
    }

    /// Returns the collection of incident cells to `cell_idx` with exactly 1 dimension difference.
    ///
    /// This separates boundary relationships (cells of dimension k-1, forming the boundary)
    /// from coboundary relationships (cells of dimension k+1, having this cell in their boundary).
    /// This distinction is important for homology and cohomology calculations.
    pub fn incidences(
        &self,
        k: usize,
        cell_idx: usize,
    ) -> Result<(&Vec<usize>, &Vec<usize>), KohoError> {
        if k >= self.cells.len() {
            return Err(KohoError::DimensionMismatch);
        }
        if cell_idx >= self.cells[k].len() {
            return Err(KohoError::InvalidCellIdx);
        }
        let upper = &self.cells[k][cell_idx].upper;
        let lower = &self.cells[k][cell_idx].lower;
        Ok((upper, lower))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attach_zero_cells() {
        let mut sk = Skeleton::init();
        // Attach a single 0-cell
        let idx = sk.attach(Cell::new(0), None, None).unwrap();
        assert_eq!(idx, 0, "Index for first 0-cell should be 1");
        assert_eq!(
            sk.cells[0].len(),
            1,
            "There should be one 0-cell in the skeleton"
        );
    }

    #[test]
    fn test_incidences_empty_and_invalid() {
        let sk = Skeleton::init();
        // No cells yet: incidences should error on invalid index
        let err = sk.incidences(0, 0).unwrap_err();
        assert!(matches!(err, KohoError::InvalidCellIdx));

        // Out-of-range dimension
        let err = sk.incidences(1, 0).unwrap_err();
        assert!(matches!(err, KohoError::DimensionMismatch));
    }

    #[test]
    fn test_dimension_mismatch_on_attach() {
        let mut sk = Skeleton::init();
        // Can't attach a 2-cell to an empty 0-skeleton
        let err = sk.attach(Cell::new(2), None, None).unwrap_err();
        assert!(matches!(err, KohoError::DimensionMismatch));
    }

    #[test]
    fn test_attach_edge_and_incidence_relations() {
        let mut sk = Skeleton::init();
        // Create two vertices (0-cells)
        sk.attach(Cell::new(0), None, None).unwrap(); // v0
        sk.attach(Cell::new(0), None, None).unwrap(); // v1
        assert_eq!(sk.cells[0].len(), 2, "There should be two 0-cells");

        // Attach an edge (1-cell) between v0 and v1
        let e_idx = sk.attach(Cell::new(1), None, Some(&[0, 1])).unwrap();
        assert_eq!(e_idx, 0, "Index for first 1-cell should be 0");
        assert_eq!(sk.cells[1].len(), 1, "There should be one 1-cell");

        // Check that the edge recorded the lower incidences
        let edge = &sk.cells[1][0];
        assert_eq!(
            edge.lower,
            vec![0, 1],
            "Edge should be incident to vertices 0 and 1"
        );

        // Check that each vertex recorded the edge in its upper incidence
        assert_eq!(
            sk.cells[0][0].upper,
            vec![0],
            "Vertex 0 should have edge 0 in its upper incidences"
        );
        assert_eq!(
            sk.cells[0][1].upper,
            vec![0],
            "Vertex 1 should have edge 0 in its upper incidences"
        );

        // Incidences API for a 0-cell
        let (upper, _lower) = sk.incidences(0, 0).unwrap();
        assert_eq!(
            upper,
            &[0],
            "Upper incidences of vertex 0 should contain the edge"
        );
    }

    #[test]
    fn test_invalid_cell_idx_in_incidences() {
        let mut sk = Skeleton::init();
        sk.attach(Cell::new(0), None, None).unwrap();
        // Querying an out-of-bounds cell index should give an error
        let err = sk.incidences(0, 1).unwrap_err();
        assert!(matches!(err, KohoError::InvalidCellIdx));
    }
}
