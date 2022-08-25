use ndarray::{Array2, Array1, Array};
use ndarray_rand::{RandomExt, rand_distr::StandardNormal};


pub struct LayerDense {
    weights: Array2<f64>,
    biases: Array1<f64>,
    pub output: Array2<f64>,
}

impl LayerDense {
    pub fn new(n_inputs: usize, n_neurons: usize) -> LayerDense {
        LayerDense {
            //neurons,
            weights: 0.1 * Array::random((n_inputs, n_neurons), StandardNormal),
            biases: Array1::from_elem(n_neurons, 0.0),
            output: Array2::zeros((n_inputs, n_neurons)),
        }
    }

    pub fn load(weights: Array2<f64>, biases: Array1<f64>) -> LayerDense {
        LayerDense {
            weights,
            biases,
            output: Array2::zeros((1, 1)),
        }
    }

    pub fn forward(&mut self, inputs: &Array2<f64>) {
        let mut result: Array2<f64> = inputs.dot(&self.weights);
        // Add biases
        result += &self.biases;
        self.output = result;
    }
}