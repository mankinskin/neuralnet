use rand::rngs::{ThreadRng};
use rand::{Rng};

pub trait Gene {
    fn mutate(&mut self, rng: &mut ThreadRng);
    fn crossover(&self, other: &Self, rng: &mut ThreadRng) -> Self;
}
