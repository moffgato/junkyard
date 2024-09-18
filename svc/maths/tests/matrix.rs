use maths::{Vector, Matrix};


#[test]
fn test_vector_operations() {

    let [ v1, v2 ] = [
        Vector::new(vec![1.0, 2.0, 3.0]),
        Vector::new(vec![4.0, 5.0, 6.0]),
    ];
    let v3 = &v1 + &v2;

    assert_eq!(v3.elements, vec![5.0, 7.0, 9.0]);

}

#[test]
fn test_matrix_operations() {

    let [ m1, m2 ] = [
        Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
        Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]),
    ];
    let m3 = &m1 + &m2;

    assert_eq!(m3.elements, vec![6.0, 8.0, 10.0, 12.0]);

}

#[test]
fn test_matrix_vector_multiplication() {

    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let v = Vector::new(vec![7.0, 8.0, 9.0]);
    let result = &m * &v;

    assert_eq!(result.elements, vec![50.0, 122.0]);

}
