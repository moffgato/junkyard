use maths::Vector;


pub enum LossFunction {
    MeanSquaredError,
}

impl LossFunction {

    pub fn loss(&self, output: &Vector<f64>, target: &Vector<f64>) -> f64 {
        match self {
            LossFunction::MeanSquaredError => {
                output.elements
                    .iter()
                    .zip(&target.elements)
                    .map(|(o, t)| (o - t).powi(2))
                    .sum::<f64>() / output.elements.len() as f64
            },
        }
    }

    pub fn derivative(&self, output: &Vector<f64>, target: &Vector<f64>) -> Vector<f64> {
        match self {
            LossFunction::MeanSquaredError => {
                output - target
            }
        }
    }

}

