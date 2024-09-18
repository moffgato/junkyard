pub mod activations;
pub mod layer;
pub mod loss;


pub use crate::layer::Layer;
use crate::loss::LossFunction;
use log::debug;
use maths::Vector;



pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub loss: LossFunction,
}

impl NeuralNetwork {

    pub fn new(loss: LossFunction) -> Self {
        NeuralNetwork {
            layers: Vec::new(),
            loss,
        }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn predict(&self, input: &Vector<f64>) -> Vector<f64> {
        let mut output = input.clone();

        for layer in &self.layers {
            output = layer.forward(&output);
        }

        output
    }

    pub fn train(
        &mut self,
        inputs: &[Vector<f64>],
        targets: &[Vector<f64>],
        learning_rate: f64,
        epochs: usize,
    ) {

        for epoch in 0..epochs {
            for (input, target) in inputs.iter().zip(targets) {

                let mut activations = Vec::new();
                let mut z_values = Vec::new();
                let mut output = input.clone();

                activations.push(output.clone());

                for layer in &self.layers {

                    let weighted_input = &layer.weights * &output;
                    let z = &weighted_input + &layer.biases;

                    z_values.push(z.clone());
                    output = z.map(|x| layer.activation.activate(*x));
                    activations.push(output.clone());

                }

                let mut delta = self.loss.derivative(&output, target);

                for (i, layer) in self.layers.iter_mut().rev().enumerate() {

                    let z = &z_values[z_values.len() - i - 1];
                    let activation_derivative = z.map(|x| layer.activation.derivative(*x));

                    delta = delta.element_wise_mul(&activation_derivative);

                    let input_activation = &activations[activations.len() - i - 2];
                    let delta_weights = delta.outer(input_activation);
                    let delta_biases = delta.clone();

                    debug!("Layer {}: delta size = {}, input_activation size = {}", i, delta.elements.len(), input_activation.elements.len());
                    debug!("delta_weights dimensions: ({}, {})", delta_weights.rows, delta_weights.cols);
                    debug!("layer.weights dimensions: ({}, {})", layer.weights.rows, layer.weights.cols);


                    layer.weights = &layer.weights - &(&delta_weights * learning_rate);
                    layer.biases = &layer.biases - &(&delta_biases * learning_rate);

                    delta = &layer.weights.transpose() * &delta;

                }

                let current_loss = self.loss.loss(&self.predict(input), target);

                // print loss every 1k epochs
                if epoch % 1000 == 0 {
                    debug!("Epoch: {}, Current loss: {}", epoch, current_loss)
                }

            }
        }

    }


}



