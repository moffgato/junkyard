use {
    rand::Rng,
    ctor::ctor,
    maths::Vector,
    log::{debug, info},
    neural_net::{
        activations::ActivationFunction, loss::LossFunction, Layer, NeuralNetwork
    },
};


#[ctor]
fn setup() {
    env_logger::init();
}

#[test]
fn test_forward_propagation() {

    let mut network = NeuralNetwork::new(LossFunction::MeanSquaredError);

    network.add_layer(Layer::new(2, 3, ActivationFunction::Sigmoid));
    network.add_layer(Layer::new(3, 1, ActivationFunction::Sigmoid));

    let input = Vector::new(vec![0.5, -0.5]);
    let output = network.predict(&input);

    assert_eq!(output.elements.len(), 1);

}


#[test]
fn test_train_xor() {

    let mut network = NeuralNetwork::new(LossFunction::MeanSquaredError);

    network.add_layer(Layer::new(2, 2, ActivationFunction::Sigmoid));
    network.add_layer(Layer::new(2, 1, ActivationFunction::Sigmoid));

    let inputs = vec![
        Vector::new(vec![0.0, 0.0]),
        Vector::new(vec![0.0, 1.0]),
        Vector::new(vec![1.0, 0.0]),
        Vector::new(vec![1.0, 1.0]),
    ];

    let targets = [
        Vector::new(vec![0.0]),
        Vector::new(vec![1.0]),
        Vector::new(vec![1.0]),
        Vector::new(vec![0.0]),
    ];

    network.train(&inputs, &targets, 0.5, 10000);

    for (input, target) in inputs.iter().zip(targets.iter()) {
        let output = network.predict(input);
        let predicted = if output.elements[0] > 0.5 { 1.0 } else { 0.0 };

        assert_eq!(predicted, target.elements[0]);
    }

}

#[test]
fn test_layer_functionality() {
    let layer = Layer::new(2, 2, ActivationFunction::Sigmoid);
    let input = Vector::new(vec![0.5, 0.5]);

    // forward pass
    let output = layer.forward(&input);
    assert_eq!(output.elements.len(), 2);

    // backward pass
    let delta = Vector::new(vec![0.1, 0.2]);
    let (delta_weights, delta_biases, delta_input) = layer.backward(&input, &delta);

    assert_eq!(delta_weights.rows, 2);
    assert_eq!(delta_weights.cols, 2);
    assert_eq!(delta_biases.elements.len(), 2);
    assert_eq!(delta_input.elements.len(), 2);
}

#[test]
fn test_edge_case_zero_input() {
    let mut network = NeuralNetwork::new(LossFunction::MeanSquaredError);
    network.add_layer(Layer::new(2, 2, ActivationFunction::Sigmoid));
    network.add_layer(Layer::new(2, 1, ActivationFunction::Sigmoid));

    let input = Vector::new(vec![0.0, 0.0]);
    let output = network.predict(&input);

    assert_eq!(output.elements.len(), 1);
    assert!(output.elements[0] >= 0.0 && output.elements[0] <= 1.0);
}

#[test]
fn test_large_input_values() {
    let mut network = NeuralNetwork::new(LossFunction::MeanSquaredError);
    network.add_layer(Layer::new(2, 2, ActivationFunction::Sigmoid));
    network.add_layer(Layer::new(2, 1, ActivationFunction::Sigmoid));

    let input = Vector::new(vec![1e6, -1e6]); // mucho grande input
    let output = network.predict(&input);

    assert_eq!(output.elements.len(), 1);
    assert!(output.elements[0] >= 0.0 && output.elements[0] <= 1.0);
}

#[test]
fn test_train_validate_xor() {

    let mut network = NeuralNetwork::new(LossFunction::MeanSquaredError);

    network.add_layer(Layer::new(2, 2, ActivationFunction::Sigmoid));
    network.add_layer(Layer::new(2, 1, ActivationFunction::Sigmoid));

    let inputs = vec![
        Vector::new(vec![0.0, 0.0]),
        Vector::new(vec![0.0, 1.0]),
        Vector::new(vec![1.0, 0.0]),
        Vector::new(vec![1.0, 1.0]),
        Vector::new(vec![0.5, 0.5]),
        Vector::new(vec![0.0, 0.5]),
        Vector::new(vec![0.5, 0.0]),
    ];

    let targets = vec![
        Vector::new(vec![0.0]),
        Vector::new(vec![1.0]),
        Vector::new(vec![1.0]),
        Vector::new(vec![0.0]),
        Vector::new(vec![0.0]),
        Vector::new(vec![1.0]),
        Vector::new(vec![1.0]),
    ];

    // train
    for epoch in 0..100_000 {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            network.train(&[input.clone()], &[target.clone()], 0.01, 1);

            if epoch % 1000 == 0 {
                let loss: f64 = inputs.iter()
                    .zip(targets.iter())
                    .map(|(input, target)| network.loss.loss(&network.predict(input), target))
                    .sum();
                debug!("Epoch: {}, Loss: {}", epoch, loss);
            }
        }
    }

    // check predictions
    for (i, t) in inputs.iter().zip(targets.iter()) {
        let output = network.predict(i);
        let predicted = if output.elements[0] > 0.5 { 1.0 } else { 0.0 };

        info!("Input: {:?}, Predicted: {}, Expected: {}", i.elements, predicted, t.elements[0]);

        assert_eq!(predicted, t.elements[0]);
        assert!(output.elements[0] >= 0.0 && output.elements[0] <= 1.0);
    }
}


#[test]
fn test_random_inputs() {
    let mut network = NeuralNetwork::new(LossFunction::MeanSquaredError);
    network.add_layer(Layer::new(2, 2, ActivationFunction::Sigmoid));
    network.add_layer(Layer::new(2, 1, ActivationFunction::Sigmoid));

    let mut rng = rand::thread_rng();
    let inputs: Vec<Vector<f64>> = (0..100).map(|_| {
        Vector::new(vec![rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)])
    }).collect();

    // random targets
    let targets: Vec<Vector<f64>> = inputs.iter().map(|_| {
        Vector::new(vec![rng.gen_range(0.0..1.0)])
    }).collect();

    // treiiin
    for epoch in 0..42_000 {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            network.train(&[input.clone()], &[target.clone()], 0.01, 1);

            if epoch % 1000 == 0 {
                let loss: f64 = inputs.iter()
                    .zip(targets.iter())
                    .map(|(input, target)| network.loss.loss(&network.predict(input), target))
                    .sum();
                debug!("Epoch: {}, Loss: {}", epoch, loss);
            }
        }
    }


    for (i, t) in inputs.iter().zip(targets.iter()) {
        let output = network.predict(i);
        let predicted = if output.elements[0] > 0.5 { 1.0 } else { 0.0 };

        info!("Input: {:?}, Predicted: {}, Expected: {}", i.elements, predicted, t.elements[0]);
    }
}


#[test]
fn test_performance_on_noise() {
    let mut network = NeuralNetwork::new(LossFunction::MeanSquaredError);
    network.add_layer(Layer::new(2, 2, ActivationFunction::Sigmoid));
    network.add_layer(Layer::new(2, 1, ActivationFunction::Sigmoid));

    let inputs = vec![
        Vector::new(vec![0.0, 0.0]),
        Vector::new(vec![0.0, 1.0]),
        Vector::new(vec![1.0, 0.0]),
        Vector::new(vec![1.0, 1.0]),
    ];

    let targets = vec![
        Vector::new(vec![0.0]),
        Vector::new(vec![1.0]),
        Vector::new(vec![1.0]),
        Vector::new(vec![0.0]),
    ];

    // train
    for epoch in 0..100_000 {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            network.train(&[input.clone()], &[target.clone()], 0.01, 1);

            if epoch % 1000 == 0 {
                let loss: f64 = inputs.iter()
                    .zip(targets.iter())
                    .map(|(input, target)| network.loss.loss(&network.predict(input), target))
                    .sum();
                debug!("Epoch: {}, Loss: {}", epoch, loss);
            }
        }
    }

    // noisy inputs
    let noise_factor = 0.1;
    for (i, t) in inputs.iter().zip(targets.iter()) {
        let noisy_input = Vector::new(vec![
            i.elements[0] + noise_factor * (rand::random::<f64>() - 0.5),
            i.elements[1] + noise_factor * (rand::random::<f64>() - 0.5),
        ]);
        let output = network.predict(&noisy_input);
        let predicted = if output.elements[0] > 0.5 { 1.0 } else { 0.0 };

        info!("Noisy Input: {:?}, Predicted: {}, Expected: {}", noisy_input.elements, predicted, t.elements[0]);

        assert_eq!(predicted, t.elements[0]);

    }
}














