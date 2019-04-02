#![feature(range_contains)]
extern crate rand;

#[macro_use]
extern crate lazy_static;

extern crate threadpool;

extern crate num;
extern crate itertools;

use std::vec::Vec;
use std::io::*;
use std::cmp::*;
use std::cmp::Ordering::Equal;
use std::iter::Zip;
use rand::distributions::{uniform::Uniform, Normal, Distribution};
use std::{f64, f32};


mod neural;
use neural::{NeuralNet};

mod genetic;
use genetic::Gene;

fn fetch_stdin() -> String {
    let mut s = String::new();
    let _ = stdout().flush();
    stdin().read_line(&mut s).expect("Did not enter a correct string");
    if let Some('\n') = s.chars().next_back() {
        s.pop();
    }
    if let Some('\r') = s.chars().next_back() {
        s.pop();
    }
    s
}

type Input = u32;
fn parse_input(s: String) -> Input {
    match s.parse::<u32>() {
        Ok(number) => number,
        Err(msg) => panic!("Cannot parse input: '{0}'! please enter a whole number.\n Error: {1}", s, msg)
    }
}

fn gaussian_sum<T: num::Integer + Clone>(n: T) -> T {
    (n.clone()*(n + T::one()))/(T::one() + T::one())
}

const MAX_NUMBER: u32 = 20;
type Population = Vec<NeuralNet>;

fn test(population: &Population) {
    // Test
    for i in (0..MAX_NUMBER) {
        let answer = population[0].predict(i);
        if answer == i {
            println!("Correct");
        } else {
            println!("False. Got {}", answer);
        }
    }

}
fn main() {

    println!("Hello, world!");

    let mut rng = rand::thread_rng();

    println!("Training a net on all numbers (0-{})...", MAX_NUMBER);

    const SELECTION_COUNT: usize = 10;
    let POP_SIZE: usize = gaussian_sum(SELECTION_COUNT);

    let numbers: Vec<u32> = (0..MAX_NUMBER).collect();
    let mut population: Population = vec![NeuralNet::new(MAX_NUMBER as usize, 2); POP_SIZE];

    println!("Training {} networks...", POP_SIZE);

    let mut generation: u32 = 0;
    let mut scores = Vec::with_capacity(POP_SIZE);

    // Simulate a populations generations
    while {

        scores.clear();
        for net in population.iter_mut() {
            let score = net.train(&numbers, &numbers);
            scores.push(score);
        }


        //println!("Ranking...");
        let mut ranking: Vec<(&NeuralNet, f32)> =
            population.iter().zip(scores).collect();

        ranking.sort_by( |(_, e1), (_, e2)| e1.partial_cmp(&e2).unwrap_or(Equal));
        scores = ranking.iter().map(|(_, e)| e.clone()).collect::<Vec<f32>>();
        population = ranking.iter().map(|(n, _)| (*n).clone()).collect::<Vec<NeuralNet>>();

        let mut total_error: f32 = scores.iter().sum();
        let mut selection_error: f32 = (&scores[..SELECTION_COUNT]).iter().sum::<f32>();

        if generation % 100 == 0 {
            println!("\nGeneration {}", generation);
            println!("Total Error: {}", total_error);
            println!("Selection Error: {}", selection_error);
            println!("Best: {} Worst: {}", scores.iter().next().unwrap(), scores.iter().last().unwrap());
        }


        if scores[0] != 0.0 {
            //println!("Crossing selection...");
            let mut next_population: Population = Vec::with_capacity(POP_SIZE);

            let selection = (&population[0..SELECTION_COUNT]).iter();

            for (i, net) in selection.enumerate() {
                let mates = (&population[i..SELECTION_COUNT + 1]).iter();

                for other in mates {
                    let child = net.crossover(other, &mut rng);
                    //println!("New Child:");
                    //child.print();
                    next_population.push(child);
                }
            }
            population = next_population;
            generation += 1;
            true
        } else {
            false }
    } {}

    println!("Errors: {:?}", scores);
    test(&population);

    loop {
        println!("Please enter a number (0-{})...", MAX_NUMBER);
        let mut input;

        // get guess
        loop {
            // add try catch
            input = parse_input(fetch_stdin());
            match (0..MAX_NUMBER).contains(&input) {
                false => println!("Input must be in range 0-{}!", MAX_NUMBER),
                true => break,
            }
        }


        let guess: u32 = population[0].predict(input) as u32;
        if guess == input {
            println!("Correct");
        } else {
            println!("False. Got {}", guess);
        }
    }
}
