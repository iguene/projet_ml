fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_deriv(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub struct MLP {
    pub w_hidden: [[f64; 4]; 4],
    pub b_hidden: [f64; 4],
    pub w_output: [f64; 4],
    pub b_output: f64,
}

impl MLP {
    pub fn new(seed: u64) -> MLP {
        let mut s = seed;
        let mut rng = || -> f64 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 1000) as f64 / 1000.0 * 2.0 - 1.0
        };
        MLP {
            w_hidden: [
                [rng(), rng(), rng(), rng()],
                [rng(), rng(), rng(), rng()],
                [rng(), rng(), rng(), rng()],
                [rng(), rng(), rng(), rng()],
            ],
            b_hidden: [rng(), rng(), rng(), rng()],
            w_output: [rng(), rng(), rng(), rng()],
            b_output: rng(),
        }
    }

    pub fn forward(&self, input: &[f64; 4]) -> (f64, [f64; 4]) {
        let mut hidden = [0.0f64; 4];
        for j in 0..4 {
            let mut sum = self.b_hidden[j];
            for i in 0..4 {
                sum += self.w_hidden[j][i] * input[i];
            }
            hidden[j] = sigmoid(sum);
        }
        let mut out_sum = self.b_output;
        for j in 0..4 {
            out_sum += self.w_output[j] * hidden[j];
        }
        (sigmoid(out_sum), hidden)
    }

    pub fn train(&mut self, data: &Vec<([f64; 4], f64)>, lr: f64, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (input, target) in data {
                let (output, hidden) = self.forward(input);
                let error = output - target;
                total_error += error * error;

                let delta_output = error * sigmoid_deriv(output);
                let mut delta_hidden = [0.0f64; 4];
                for j in 0..4 {
                    delta_hidden[j] = delta_output * self.w_output[j] * sigmoid_deriv(hidden[j]);
                }
                for j in 0..4 {
                    self.w_output[j] -= lr * delta_output * hidden[j];
                }
                self.b_output -= lr * delta_output;
                for j in 0..4 {
                    for i in 0..4 {
                        self.w_hidden[j][i] -= lr * delta_hidden[j] * input[i];
                    }
                    self.b_hidden[j] -= lr * delta_hidden[j];
                }
            }
            if epoch % 200 == 0 {
                println!("  [MLP] Epoch {:4} | Erreur: {:.6}", epoch, total_error);
            }
        }
    }

    pub fn accuracy(&self, data: &Vec<([f64; 4], f64)>) -> f64 {
        let correct = data.iter().filter(|(input, target)| {
            let (pred, _) = self.forward(input);
            let pred_class = if pred >= 0.5 { 1.0 } else { 0.0 };
            pred_class == *target
        }).count();
        (correct as f64 / data.len() as f64) * 100.0
    }
}