mod layer;
mod math;

use layer::LayerDense;
use ndarray::array;

fn main() {
    let input = array![
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ];

    /* let mut hidden0 = Layer::load(
        vec![
            vec![0.2, 0.8, -0.5, 1.0],
            vec![0.5, -0.91, 0.26, -0.5],
            vec![-0.26, -0.27, 0.17, 0.87],
        ],
        vec![2.0, 3.0, 0.5]
    );

    let mut hidden1 = Layer::load(
        vec![
            vec![0.1, -0.14, 0.5],
            vec![-0.5, 0.12, -0.33],
            vec![-0.44, 0.73, -0.13],
        ],
        vec![-1.0, 2.0, -0.5]
    ); */
    let mut layer1 = LayerDense::new(4, 5);
    let mut layer2 = LayerDense::new(5, 2);

    layer1.forward(&input);
    layer2.forward(&layer1.output);
    println!("{}", layer2.output);
    //math::Matrix::print(&layer2.output);

}
