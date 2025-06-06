use candle_core::{DType, Device, Error, Result, Tensor, Var, WithDType};

#[derive(Debug, Clone)]
pub struct Vector {
    tensor: Tensor,
    device: Device,
    dtype: DType,
}

impl Vector {
    pub fn new(tensor: Tensor, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            tensor,
            device,
            dtype,
        })
    }

    pub fn from_slice<T: WithDType>(
        data: &[T],
        dimension: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let t = Tensor::from_slice(data, (dimension, 1), &device)?;
        Self::new(t, device, dtype)
    }

    pub fn dimension(&self) -> usize {
        self.tensor.dims()[0]
    }

    pub fn inner(&self) -> &Tensor {
        &self.tensor
    }

    pub fn dot(&self, other: &Vector) -> Result<Tensor> {
        self.tensor.transpose(0, 1)?.matmul(&other.tensor)
    }

    pub fn norm(&self) -> Result<Tensor> {
        self.tensor.sqr()?.sum_all()?.sqrt()
    }

    pub fn normalize(&self) -> Result<Self> {
        let norm = self.norm()?;
        Ok(Self {
            tensor: self.tensor.broadcast_div(&norm)?,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn add(&self, other: &Vector) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.add(&other.tensor)?,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn scale<T: WithDType>(&self, scalar: T) -> Result<Self> {
        if scalar.to_scalar().dtype() == self.dtype {
            let tensor = self.tensor.clone().to_dtype(DType::F64)?;
            return Ok(Self {
                tensor: (tensor * scalar.to_scalar().to_f64())?,
                device: self.device.clone(),
                dtype: self.dtype,
            });
        }
        Err(Error::DTypeMismatchBinaryOp {
            lhs: scalar.to_scalar().dtype(),
            rhs: self.dtype,
            op: "scalar multiply",
        })
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub tensor: Var,
    pub device: Device,
    pub dtype: DType,
}

impl Matrix {
    /// Creates a new `Matrix`.
    pub fn new(tensor: Tensor, device: Device, dtype: DType) -> Result<Self> {
        if tensor.rank() != 2 {
            return Err(Error::Msg("Matrix must be rank 2".into()));
        }
        Ok(Self {
            tensor: Var::from_tensor(&tensor)?,
            device,
            dtype,
        })
    }

    /// Creates a new `Matrix` from a slice of data.
    pub fn from_slice<T: WithDType>(
        data: &[T],
        rows: usize,
        cols: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let t = Tensor::from_slice(data, (rows, cols), &device)?;
        Self::new(t, device, dtype)
    }

    pub fn from_vecs(vecs: Vec<Vector>) -> Result<Self> {
        if vecs.is_empty() {
            return Err(Error::Msg(
                "Cannot create a matrix from an empty list of vectors.".into(),
            ));
        }

        let first_vec = &vecs[0];
        let dimension = first_vec.dimension();
        let device = first_vec.device.clone();
        let dtype = first_vec.dtype;

        let mut column_tensors = Vec::with_capacity(vecs.len());

        for (i, vec) in vecs.iter().enumerate() {
            if vec.dimension() != dimension {
                return Err(Error::Msg(format!(
                    "Vector at index {} has dimension {} but expected {}.",
                    i,
                    vec.dimension(),
                    dimension
                )));
            }
            if vec.dtype != dtype {
                return Err(Error::Msg(format!(
                    "Vector at index {} has a different dtype. Expected {:?}, got {:?}.",
                    i, dtype, vec.dtype
                )));
            }

            column_tensors.push(vec.inner().reshape((dimension, 1))?);
        }

        // Concatenate all column tensors along dimension 1 (columns)
        let matrix_tensor = Tensor::cat(&column_tensors, 1)?;

        Self::new(matrix_tensor, device, dtype)
    }

    /// Returns the shape of the matrix as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        let dims = self.tensor.dims();
        (dims[0], dims[1])
    }

    /// Returns the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.tensor.dims()[0]
    }

    /// Returns the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.tensor.dims()[1]
    }

    /// Returns a reference to the inner `Var`.
    pub fn inner(&self) -> &Var {
        &self.tensor
    }

    /// Returns a reference to the inner `Var`.
    pub fn inner_mut(&mut self) -> &mut Var {
        &mut self.tensor
    }

    /// Performs matrix multiplication: `self * other`.
    pub fn matmul(&self, other: &Matrix) -> Result<Self> {
        if self.cols() != other.rows() {
            return Err(Error::Msg(format!(
                "Matrix multiplication dimension mismatch: self_cols ({}) != other_rows ({})",
                self.cols(),
                other.rows()
            )));
        }
        let result_tensor = Var::from_tensor(&self.tensor.matmul(other.inner())?)?;
        Ok(Self {
            tensor: result_tensor,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn matvec(&self, other: &Vector) -> Result<Vector> {
        if self.cols() != other.dimension() {
            return Err(Error::Msg(format!(
                "Matrix multiplication dimension mismatch: self_cols ({}) != other_rows ({})",
                self.cols(),
                other.dimension()
            )));
        }
        let result_tensor = self.tensor.matmul(other.inner())?;
        Ok(Vector {
            tensor: result_tensor,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    /// Transposes the matrix.
    pub fn transpose(&self) -> Result<Self> {
        let result_tensor = Var::from_tensor(&self.tensor.transpose(0, 1)?)?;
        Ok(Self {
            tensor: result_tensor,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    /// Adds another matrix to this matrix element-wise.
    pub fn add(&self, other: &Matrix) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(Error::Msg(format!(
                "Matrix addition shape mismatch: self {:?} != other {:?}",
                self.shape(),
                other.shape()
            )));
        }
        let result_tensor = Var::from_tensor(&self.tensor.add(other.inner())?)?;
        Ok(Self {
            tensor: result_tensor,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    /// Scales the matrix by a scalar.
    pub fn scale<T: WithDType>(&self, scalar: T) -> Result<Self> {
        // Keep everything in the same dtype
        let scalar_tensor = Tensor::full(
            scalar.to_scalar().to_f64(),
            self.tensor.dims(),
            &self.device,
        )?
        .to_dtype(self.dtype)?;

        Ok(Self {
            tensor: Var::from_tensor(&self.tensor.mul(&scalar_tensor)?)?,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    /// Computes the Frobenius norm of the matrix.
    /// The Frobenius norm is sqrt(sum of squares of its elements).
    pub fn frobenius_norm(&self) -> Result<Tensor> {
        self.tensor.sqr()?.sum_all()?.sqrt()
    }

    pub fn to_vectors(&self) -> Result<Vec<Vector>> {
        let (_, cols) = self.shape();
        let mut cols_vectors = Vec::with_capacity(cols);

        let cols_tensors = self.tensor.chunk(cols, 1)?;

        for col_tensor in cols_tensors {
            cols_vectors.push(Vector::new(col_tensor, self.device.clone(), self.dtype)?);
        }

        Ok(cols_vectors)
    }

    /// Generates a new random matrix with elements sampled from a standard normal distribution (mean 0, std dev 1).
    pub fn rand(rows: usize, cols: usize, device: Device, dtype: DType) -> Result<Self> {
        let tensor = Tensor::randn(0.0f32, 1.0f32, (rows, cols), &device)?.to_dtype(dtype)?;
        Self::new(tensor, device, dtype)
    }

    /// Creates a new matrix filled with zeros of the specified dimensions.
    pub fn zeros(rows: usize, cols: usize, device: Device, dtype: DType) -> Result<Self> {
        let tensor = Tensor::zeros((rows, cols), dtype, &device)?;
        Self::new(tensor, device, dtype)
    }

    pub fn identity(rows: usize, cols: usize, device: Device, dtype: DType) -> Result<Self> {
        // Start with a zero matrix
        let mut tensor = Tensor::zeros((rows, cols), dtype, &device)?;

        // Set diagonal elements to 1
        let min_dim = rows.min(cols);
        for i in 0..min_dim {
            // Create a tensor with value 1.0
            let one = Tensor::ones((1, 1), dtype, &device)?;
            // Use narrow and copy to set the diagonal element
            tensor = tensor.slice_assign(&[i..i + 1, i..i + 1], &one)?;
        }

        Self::new(tensor, device, dtype)
    }

    pub fn identity_like(&self, rows: usize, cols: usize) -> Result<Self> {
        Self::identity(rows, cols, self.device.clone(), self.dtype)
    }

    pub fn transpose_matvec(&self, other: &Vector) -> Result<Vector> {
        if self.rows() != other.dimension() {
            return Err(Error::Msg(format!(
                "Transposed matrix multiplication dimension mismatch: self_rows ({}) != other_dim ({})",
                self.rows(),
                other.dimension()
            )));
        }

        let result_tensor = other
            .inner()
            .transpose(0, 1)?
            .matmul(&self.tensor)?
            .transpose(0, 1)?;

        Ok(Vector {
            tensor: result_tensor,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }
}

// Example Usage (requires a candle_core setup)
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_matrix_new_and_shape() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let t = Tensor::randn(0f32, 1f32, (2, 3), &device)?.to_dtype(dtype)?;
        let m = Matrix::new(t, device.clone(), dtype)?;

        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m.inner().dims(), &[2, 3]);
        assert_eq!(m.dtype, dtype);
        Ok(())
    }

    #[test]
    fn test_matrix_from_slice() -> Result<()> {
        let device = Device::Cpu;
        let data_f32: [f32; 6] = [1., 2., 3., 4., 5., 6.];

        let m = Matrix::from_slice(&data_f32, 2, 3, device.clone(), DType::F32)?;
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(
            m.inner().to_vec2::<f32>()?,
            vec![vec![1., 2., 3.], vec![4., 5., 6.]]
        );
        assert_eq!(m.dtype, DType::F32); // This is the struct's dtype field
        assert_eq!(m.inner().dtype(), DType::F32); // Tensor's actual dtype
        Ok(())
    }

    #[test]
    fn test_matrix_add() -> Result<()> {
        let device = Device::Cpu;
        let m1 = Matrix::from_slice(&[1f32, 2., 3., 4.], 2, 2, device.clone(), DType::F32)?;
        let m2 = Matrix::from_slice(&[5f32, 6., 7., 8.], 2, 2, device.clone(), DType::F32)?;
        let m3 = m1.add(&m2)?;
        assert_eq!(
            m3.inner().to_vec2::<f32>()?,
            vec![vec![6., 8.], vec![10., 12.]]
        );
        assert_eq!(m3.dtype, DType::F32);
        Ok(())
    }

    #[test]
    fn test_matrix_matmul() -> Result<()> {
        let device = Device::Cpu;
        let m1 = Matrix::from_slice(&[1f32, 2., 3., 4.], 2, 2, device.clone(), DType::F32)?; // 2x2
        let m2 = Matrix::from_slice(
            &[5f32, 6., 7., 8., 9., 10.],
            2,
            3,
            device.clone(),
            DType::F32,
        )?; // 2x3
        let m3 = m1.matmul(&m2)?; // Expected 2x3

        assert_eq!(m3.shape(), (2, 3));
        assert_eq!(
            m3.inner().to_vec2::<f32>()?,
            vec![vec![21., 24., 27.], vec![47., 54., 61.]]
        );
        assert_eq!(m3.dtype, DType::F32);
        Ok(())
    }

    #[test]
    fn test_matrix_transpose() -> Result<()> {
        let device = Device::Cpu;
        let m1 = Matrix::from_slice(
            &[1f32, 2., 3., 4., 5., 6.],
            2,
            3,
            device.clone(),
            DType::F32,
        )?;
        let m_t = m1.transpose()?;
        assert_eq!(m_t.shape(), (3, 2));
        assert_eq!(
            m_t.inner().to_vec2::<f32>()?,
            vec![vec![1., 4.], vec![2., 5.], vec![3., 6.]]
        );
        Ok(())
    }

    impl Matrix {
        // A more conventional scale method for numeric scalars
        pub fn scale_numeric(&self, scalar_val: f64) -> Result<Self> {
            let result_tensor = Var::from_tensor(&(self.tensor.clone().as_tensor() * scalar_val)?)?;
            Ok(Self {
                tensor: result_tensor,
                device: self.device.clone(),
                dtype: self.dtype,
            })
        }
    }

    #[test]
    fn test_matrix_scale_numeric() -> Result<()> {
        let device = Device::Cpu;
        let m1 = Matrix::from_slice(&[1f32, 2., 3., 4.], 2, 2, device.clone(), DType::F32)?;
        let m_scaled = m1.scale_numeric(2.0)?;
        assert_eq!(
            m_scaled.inner().to_vec2::<f32>()?,
            vec![vec![2., 4.], vec![6., 8.]]
        );
        Ok(())
    }

    // To test the original `scale` method, you'd need a type `T` that satisfies:
    #[test]
    fn test_frobenius_norm() -> Result<()> {
        let device = Device::Cpu;
        let m = Matrix::from_slice(&[3f32, -4., 12.], 1, 3, device.clone(), DType::F32)?; // A row vector as a 1x3 matrix
                                                                                          // Norm = sqrt(3^2 + (-4)^2 + 12^2) = sqrt(9 + 16 + 144) = sqrt(169) = 13
        let norm_tensor = m.frobenius_norm()?;
        let norm_val = norm_tensor.to_scalar::<f32>()?;
        assert!((norm_val - 13.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_matrix_vector_multiplication() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create a 2x3 matrix
        let matrix = Matrix::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            2,
            3,
            device.clone(),
            dtype,
        )?;

        // Create a 3x1 vector
        let vector = Vector::from_slice(&[7.0f32, 8.0, 9.0], 3, device.clone(), dtype)?;

        // Perform matrix-vector multiplication
        let result = matrix.matvec(&vector)?;
        let squeezed = result.inner().squeeze(1)?;
        // Check dimensions of result
        assert_eq!(result.dimension(), 2);

        // Expected result:
        // [1.0, 2.0, 3.0]   [7.0]   [1.0*7.0 + 2.0*8.0 + 3.0*9.0]   [50.0]
        // [4.0, 5.0, 6.0] × [8.0] = [4.0*7.0 + 5.0*8.0 + 6.0*9.0] = [122.0]
        //                   [9.0]

        // Convert result to a vector and verify values
        let result_vec = squeezed.to_vec1::<f32>()?;
        assert_eq!(result_vec.len(), 2);
        assert!((result_vec[0] - 50.0).abs() < 1e-6);
        assert!((result_vec[1] - 122.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_matrix_vector_dimension_mismatch() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Create a 2x3 matrix
        let matrix = Matrix::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            2,
            3,
            device.clone(),
            dtype,
        )?;

        // Create a vector with incorrect dimension (2 instead of 3)
        let vector = Vector::from_slice(&[7.0f32, 8.0], 2, device.clone(), dtype)?;

        // Attempt matrix-vector multiplication should fail
        let result = matrix.matvec(&vector);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_identity_square() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Test 3x3 identity matrix
        let i3 = Matrix::identity(3, 3, device, dtype)?;
        let values = i3.inner().to_vec2::<f32>()?;

        assert_eq!(
            values,
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ]
        );

        Ok(())
    }

    #[test]
    fn test_identity_rectangular_tall() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Test 4x2 identity-like matrix (tall)
        let i42 = Matrix::identity(4, 2, device, dtype)?;
        let values = i42.inner().to_vec2::<f32>()?;

        assert_eq!(
            values,
            vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.0, 0.0],
                vec![0.0, 0.0],
            ]
        );

        Ok(())
    }

    #[test]
    fn test_identity_rectangular_wide() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Test 2x4 identity-like matrix (wide)
        let i24 = Matrix::identity(2, 4, device, dtype)?;
        let values = i24.inner().to_vec2::<f32>()?;

        assert_eq!(
            values,
            vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0],]
        );

        Ok(())
    }

    #[test]
    fn test_identity_like_method() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F64;

        // Create a matrix to use as reference for device/dtype
        let ref_matrix = Matrix::zeros(5, 5, device, dtype)?;

        // Use identity_like to create a 3x3 identity with same settings
        let i3 = ref_matrix.identity_like(3, 3)?;

        assert_eq!(i3.dtype, dtype);
        assert_eq!(i3.shape(), (3, 3));

        // Verify it's actually an identity matrix
        let values = i3.inner().to_vec2::<f64>()?;
        assert_eq!(values[0][0], 1.0);
        assert_eq!(values[1][1], 1.0);
        assert_eq!(values[2][2], 1.0);
        assert_eq!(values[0][1], 0.0);

        Ok(())
    }

    #[test]
    fn test_identity_single_element() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;

        // Test 1x1 identity matrix
        let i1 = Matrix::identity(1, 1, device, dtype)?;
        let values = i1.inner().to_vec2::<f32>()?;

        assert_eq!(values, vec![vec![1.0]]);

        Ok(())
    }
}
