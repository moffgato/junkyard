use crate::activations::ActivationFunction;
use maths::{Matrix, Vector};
use rand::Rng;


pub struct Layer {
    pub weights: Matrix<f64>,
    pub biases: Vector<f64>,
    pub activation: ActivationFunction,
}

impl Layer {

    pub fn new(input_size: usize, output_size: usize, activation: ActivationFunction) -> Self {

        let mut rng = rand::thread_rng();
        let weights = Matrix::from_fn(output_size, input_size, |_, _| rng.gen_range(-1.0..1.0));
        let biases = Vector::from_fn(output_size, |_| rng.gen_range(-1.0..1.0));

        Layer {
            weights,
            biases,
            activation,
        }

    }

    pub fn forward(&self, input: &Vector<f64>) -> Vector<f64> {
        let z = &(&self.weights * input) + &self.biases;
        z.map(|x| self.activation.activate(*x))
    }

    pub fn backward(
        &self,
        input: &Vector<f64>,
        delta: &Vector<f64>,
    ) -> (Matrix<f64>, Vector<f64>, Vector<f64>) {

        let _activation_derivative = input.map(|x| self.activation.derivative(*x));
        let delta_weights = delta.outer(input);
        let delta_biases = delta.clone();
        let delta_input = &self.weights.transpose() * delta;

        (delta_weights, delta_biases, delta_input)
    }

}

