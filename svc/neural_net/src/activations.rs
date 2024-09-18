

#[derive(Clone, Copy)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
}

impl ActivationFunction {

    pub fn activate(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ReLU => {
                if x > 0.0 { 1.0 } else { 0.0 }
            },
            ActivationFunction::Sigmoid => {
                let sig = self.activate(x);
                sig * (1.0 - sig)
            },
            ActivationFunction::Tanh => {
                1.0 - x.tanh().powi(2)
            },
        }
    }

}

