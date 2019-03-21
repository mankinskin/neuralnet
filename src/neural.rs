
use rand::rngs::{ThreadRng};
use rand::{Rng};
use std::fmt;
use crate::genetic::Gene;
fn sigmoid(x: f32) -> f32 {
    2.0 / (1.0 + std::f32::consts::E.powf(-x)) - 1.0
}

#[derive(Clone)]
struct Neuron {
    charge: f32,
    bias: f32
}

impl Neuron {
    pub fn new(bias: f32) -> Neuron {
        Neuron {
            charge: 0.0,
            bias: bias
        }
    }

    pub fn charge(&mut self, amt: f32) {
        self.charge = self.charge + amt;
    }

    fn step(threshold: f32) -> impl Fn(f32) -> f32 {
        move |x| {
            if x >= threshold { 1.0 } else { 0.0 }
        }
    }
    pub fn signal(&self) -> f32 {
        let step = Neuron::step(0.5);
        step(sigmoid(self.bias + self.charge))
    }
}

impl fmt::Debug for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.charge)
    }
}

type Layer = Vec<Neuron>;
type Layers = Vec<Layer>;

fn new_layer(width: usize) -> Layer {
    vec![Neuron::new(0.5); width]
}

pub type WeightLayer = Vec<Vec<f32>>;
pub type Weights = Vec<WeightLayer>;

fn new_weights(width: usize, depth: usize) -> Weights {
    let layer = (vec![vec![0.0; width]; width]);//.iter_mut()
            //.enumerate().map(|(i, v)| {v[i] = 1.0; v.clone()}).collect::<Vec<Vec<f32>>>();
    vec![layer; depth-1]
}

const MUTATION_CHANCE: f64 = 0.001;

#[derive(Clone)]
pub struct NeuralNet {
    // make size generic
    pub width: usize,
    pub depth: usize,
    weights: Weights,
    mutation_chance: f64
}

impl NeuralNet {
    pub fn new(width: usize, depth: usize) -> NeuralNet {
        let weights = new_weights(width, depth);
        NeuralNet {
            width,
            depth,
            weights,
            mutation_chance: MUTATION_CHANCE,
        }
    }
    pub fn from_weights(weights: Weights) -> NeuralNet {
        NeuralNet {
            width: weights[0].len(),
            depth: weights.len()+1,
            weights,
            mutation_chance: MUTATION_CHANCE,
        }
    }

        pub fn process(&self, input: Layer) -> Layer {
        assert!(input.len() == self.width);
        let mut layer = input;

        // simulate an input layer
        // charge every next layer using the previous
        // layer and the weights between the two
        for (l, weightlayer) in self.weights.iter().enumerate() {
            //  ^ layer of weights for each neuron
            let mut next = new_layer(self.width); // next layer

            // for each neuron in the input layer and its weights
            for (source, synapses) in layer.iter_mut().zip(weightlayer.iter()) {
                // for each neuron in the next layer and the corresponding weight
                for (neuron, weight) in next.iter_mut().zip(synapses.iter()) {
                    neuron.charge(weight * source.signal());
                }
            }
            layer = next;
        }
        layer
    }

    pub fn predict(&self, input: u32) -> u32 {
        let mut input_layer: Layer = new_layer(self.width);
        input_layer[input as usize].charge(1.0);
        let outlayer = self.process(input_layer);
        let answer: u32 = outlayer.iter()
            .enumerate()
            .max_by(|(_, a): &(usize, &Neuron), (_, b): &(usize, &Neuron)| a.charge.partial_cmp(&b.charge).unwrap())
            .unwrap().0 as u32;
        answer
    }

    pub fn print(&self) {
        for ni in (0..self.weights[0].len()) { // index of neuron
            for wi in (0..self.weights[0][0].len()) { // index of weight
                for li in (0..self.weights.len()-1) { // layer
                    print!("X :: {} :: ", self.weights[li][ni][wi]);
                }
                println!("X :: {} :: X*", self.weights.iter().last().unwrap()[ni][wi]);
            }
            println!("#");
        }
    }

    fn weight_count(&self) -> usize {
        self.depth * self.width.pow(2)
    }

    pub fn train(&mut self, samples: &Vec<u32>, expected: &Vec<u32>) -> f32 {

        assert!(samples.len() == expected.len(), "Not all samples have expected values!");

        let mut total_error: f32 = 0.0;

        for (sample, expect) in samples.iter().zip(expected.iter()) {
            let error = self.learn_sample(*sample, *expect);
            total_error += error;
        }
        total_error
    }
    fn learn_sample(&self, sample: u32, expected: u32) -> f32 {

        let mut input_layer: Layer = new_layer(self.width);
        input_layer[sample as usize].charge(1.0);

        let mut expected_layer: Layer = new_layer(self.width);
        expected_layer[expected as usize].charge(1.0);

        let out = self.process(input_layer);

        let deviation: Vec<f32> =
            out.iter()
            .zip(expected_layer)
            .map(|(n, e)| (e.charge - n.charge).abs())
            .collect::<Vec<f32>>();

        let total: f32 = deviation.iter().sum();
        // output error
        total
    }
}

impl Gene for NeuralNet {

    fn mutate(&mut self, rng: &mut ThreadRng) {
        for mut layer in self.weights.iter_mut() {
            for mut weights in layer.iter_mut() {
                for mut w in weights.iter_mut() {
                    if rng.gen_bool(self.mutation_chance) {
                        *w = rng.choose::<f32>(&[0.0, 1.0]).unwrap().clone();//gen_range(-1.0f32, 1.0f32);
                    }
                }
            }
        }
    }

    fn crossover(&self, other: &NeuralNet, rng: &mut ThreadRng) -> NeuralNet {

        let mut new_layers: Weights = new_weights(self.width, self.depth);

        for (mut new_layer, (layer_a, layer_b))
            in new_layers.iter_mut().zip(self.weights.iter().zip(other.weights.iter())) {

            for (mut new_weights, (weights_a, weights_b))
                in new_layer.iter_mut().zip(layer_a.iter().zip(layer_b.iter())) {

                for (mut w, (&a, &b))
                    in new_weights.iter_mut().zip(weights_a.iter().zip(weights_b.iter())) {

                    *w = if rng.gen_bool(0.5) {
                            a
                        } else {
                            b
                        };

                }
            }
        }

        let mut new_net = NeuralNet::from_weights(new_layers);
        new_net.mutation_chance = (self.mutation_chance + other.mutation_chance)/2.0;
        new_net.mutate(rng);
        new_net
    }
}
