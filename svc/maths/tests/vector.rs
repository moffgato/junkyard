use maths::Vector;

#[test]
fn vector_addition() {

    let [v1, v2] = [
        Vector::new(vec![1, 2, 3]),
        Vector::new(vec![4, 5, 6]),
    ];
    let v3 = &v1 + &v2;

    assert_eq!(v3, Vector::new(vec![5, 7, 9]));

}

#[test]
fn vector_scalar_multiply() {

    let v1 = Vector::new(vec![1, 2, 3]);
    let v2 = &v1 * 2;

    assert_eq!(v2, Vector::new(vec![2, 4, 6]));

}

#[test]
fn dot_product() {

    let [v1, v2] = [
        Vector::new(vec![1, 2, 3]),
        Vector::new(vec![4, 5, 6]),
    ];
    let result = v1.dot(&v2);

    assert_eq!(result, 32);

}

