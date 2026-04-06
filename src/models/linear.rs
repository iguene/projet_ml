pub struct LinearModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LinearModel {
    pub fn new(n_features: usize) -> LinearModel {
        LinearModel {
            weights: vec![0.0; n_features],
            bias: 0.0,
        }
    }

    pub fn predict(&self, input: &Vec<f64>) -> f64 {
        let mut sum = self.bias;
        for i in 0..self.weights.len() {
            sum += self.weights[i] * input[i];
        }
        sum
    }

    pub fn train(&mut self, data: &Vec<(Vec<f64>, f64)>, lr: f64, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (input, target) in data {
                let pred  = self.predict(input);
                let error = pred - target;
                total_error += error * error;
                for i in 0..self.weights.len() {
                    self.weights[i] -= lr * error * input[i];
                }
                self.bias -= lr * error;
            }
            if epoch % 200 == 0 {
                println!("  [Linéaire] Epoch {:4} | Erreur: {:.6}", epoch, total_error);
            }
        }
    }

    pub fn accuracy(&self, data: &Vec<(Vec<f64>, f64)>) -> f64 {
        let correct = data.iter().filter(|(input, target)| {
            let pred_class = if self.predict(input) >= 0.5 { 1.0 } else { 0.0 };
            pred_class == *target
        }).count();
        (correct as f64 / data.len() as f64) * 100.0
    }
}