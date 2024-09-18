use crate::vector::Vector;
use std::ops::{Add, Sub, Mul, Index, IndexMut};


#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub elements: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Copy + Default,
{
    pub fn new(rows: usize, cols: usize, elements: Vec<T>) -> Self {
        Matrix {
            rows,
            cols,
            elements,
        }
    }

    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let elements = (0..rows * cols)
            .map(|idx| f(idx / cols, idx % cols))
            .collect();

        Matrix {
            rows,
            cols,
            elements,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut elements = vec![T::default(); self.elements.len()];

        for i in 0..self.rows {
            for j in 0..self.cols {
                elements[j * self.rows + i] = self[(i, j)];
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            elements,
        }
    }

}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.elements[idx.0 * self.cols + idx.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        &mut self.elements[idx.0 * self.cols + idx.1]
    }
}

impl<T> Add for &Matrix<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn add(self, other: Self) -> Matrix<T> {

        assert_eq!(self.rows, other.rows, "Matrices must have same number of rows to add.");
        assert_eq!(self.cols, other.cols, "Matrices must have same number of columns to add.");

        let elements = self.elements
            .iter()
            .zip(&other.elements)
            .map(|(&a, &b)| a + b)
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            elements,
        }
    }
}

impl<T> Sub for &Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, other: Self) -> Matrix<T> {

        assert_eq!(self.rows, other.rows, "Matrices must have the same number of rows to subtract.");
        assert_eq!(self.cols, other.cols, "Matrices must have the same number of columns to subtract.");

        let elements = self
            .elements
            .iter()
            .zip(&other.elements)
            .map(|(&a, &b)| a - b)
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            elements,
        }
    }
}

impl<T> Mul<&Vector<T>> for &Matrix<T>
where
    T: Mul<Output = T> + Add<Output = T> + Copy + Default,
{
    type Output = Vector<T>;

    fn mul(self, vector: &Vector<T>) -> Self::Output {
        assert_eq!(
            self.cols,
            vector.elements.len(),
            "Matrix columns must match vector size",
        );

        let mut result_elements = Vec::with_capacity(self.rows);

        for row in 0..self.rows {
            let mut sum = T::default();

            for col in 0..self.cols {
                sum = sum + self[(row, col)] * vector[col];
            }
            result_elements.push(sum);
        }

        Vector::new(result_elements)
    }
}


impl<T> Mul for &Matrix<T>
where
    T: Mul<Output = T> + Add<Output = T> + Copy + Default,
{
    type Output = Matrix<T>;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(self.cols, self.rows, "Matrix A columns must match Matrix B rows");

        let mut elements = Vec::with_capacity(self.rows * other.cols);

        for row in 0..self.rows {
            for col in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + self[(row, k)] * other[(k, col)];
                }
                elements.push(sum);
            }
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            elements,
        }

    }
}

impl<T> Mul<T> for &Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, scalar: T) -> Self::Output {

        let elements = self.elements
            .iter()
            .map(|&x| x * scalar)
            .collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            elements,
        }

    }
}



