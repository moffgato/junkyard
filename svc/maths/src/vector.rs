use std::ops::{
    Add,
    Sub,
    Mul,
    Index,
    IndexMut,
};

use crate::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T> {
    pub elements: Vec<T>,
}

impl<T> Vector<T> {
    pub fn new(elements: Vec<T>) -> Self {
        Vector { elements }
    }
}

impl<T> Add for &Vector<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Vector<T>;

    fn add(self, other: Self) -> Self::Output {
        let elements = self
            .elements
            .iter()
            .zip(&other.elements)
            .map(|(&a, &b)| a + b)
            .collect();
        Vector::new(elements)
    }
}

impl<T> Sub for &Vector<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Vector<T>;

    fn sub(self, other: Self) -> Vector<T> {
        assert_eq!(
            self.elements.len(),
            other.elements.len(),
            "Vectors must be the same length to subtract."
        );

        let elements = self
            .elements
            .iter()
            .zip(&other.elements)
            .map(|(&a, &b)| a - b)
            .collect();

        Vector::new(elements)
    }
}

impl<T> Mul<T> for &Vector<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Vector<T>;

    fn mul(self, scalar: T) -> Self::Output {
        let elements = self.elements
            .iter()
            .map(|&a| a * scalar)
            .collect();

        Vector::new(elements)
    }

}

impl<T> Vector<T>
where
    T: Copy + Mul<Output = T>,
{
    pub fn element_wise_mul(&self, other: &Self) -> Self {
        assert_eq!(
            self.elements.len(),
            other.elements.len(),
            "Vectors must be the same length for element-wise multiplication."
        );

        let elements = self
            .elements
            .iter()
            .zip(&other.elements)
            .map(|(&a, &b)| a * b)
            .collect();

        Vector::new(elements)
    }
}

impl<T> Vector<T>
where
    T: Mul<Output = T> + Add<Output = T> + Copy + Default,
{
    pub fn dot(&self, other: &Self) -> T {
        self.elements
            .iter()
            .zip(&other.elements)
            .map(|(&a, &b)| a * b)
            .fold(T::default(), |sum, val| sum + val)
    }
}

impl<T> Vector<T>
where
    T: Copy + Default,
{
    pub fn from_fn(size: usize, mut f: impl FnMut(usize) -> T) -> Self {
        let elements = (0..size)
            .map(|i| f(i))
            .collect();

        Vector::new(elements)
    }

    pub fn map(&self, f: impl Fn(&T) -> T) -> Self {
        let elements = self.elements
            .iter()
            .map(f)
            .collect();

        Vector { elements }
    }

    pub fn outer(&self, other: &Self) -> Matrix<T>
        where
            T: Mul<Output = T>,
    {

        let elements = self.elements
            .iter()
            .flat_map(|&a| {
                other.elements.iter().map(move |&b| a * b)
            })
            .collect();

        Matrix {
            rows: self.elements.len(),
            cols: other.elements.len(),
            elements,
        }

    }

}

impl<T> Index<usize> for Vector<T>  {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.elements[index]
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.elements[index]
    }
}


