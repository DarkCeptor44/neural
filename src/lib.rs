//! # Neural
//!
//! **Neural** is a library for Genetic Algorithms in Rust.
//!
//! ## Concepts
//!
//! - [**Genetic Algorithm**](https://en.wikipedia.org/wiki/Genetic_algorithm) - A algorithm for solving optimization problems by simulating the process of natural selection, evolution, mutation and crossover (reproduction).
//! - **Gene** - A single value that represents a possible solution to a problem.
//! - **Chromosome** - A collection of genes that represent a possible solution to a problem.
//! - **Population** - A collection of chromosomes that represent potential solutions to a problem.
//! - **Selection** - A trait for selecting a subset of the population.
//! - **Crossover** - A trait for crossing over two chromosomes.
//!
//! ## Features
//!
//! - **cli** - Allows using the `with_print` method on a `PopulationBuilder` to print the population's best chromosome after each generation and enables colored output.
//!
//! ## Getting Started
//!
//! ```sh
//! cargo add neural
//!
//! # or add it with the cli feature
//! cargo add neural --features cli
//! ```
//!
//! Or add it as a dependency in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! neural = "^0.2"
//!
//! # or add it with the cli feature
//! [dependencies]
//! neural = { version = "^0.2", features = ["cli"] }
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use neural::{Gene, Population, PopulationBuilder, Result, TournamentSelection, UniformCrossover};
//! use rand::{rngs::ThreadRng, Rng};
//! use std::fmt::Display;
//!
//! #[derive(Debug, Clone, PartialEq, PartialOrd)]
//! struct F64(f64);
//!
//! impl Eq for F64 {}
//!
//! impl Display for F64 {
//!     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//!         write!(f, "{}", self.0)
//!     }
//! }
//!
//! impl From<f64> for F64 {
//!     fn from(value: f64) -> Self {
//!         Self(value)
//!     }
//! }
//!
//! impl From<F64> for f64 {
//!     fn from(val: F64) -> Self {
//!         val.0
//!     }
//! }
//!
//! impl Gene for F64 {
//!     fn generate_gene<R>(rng: &mut R) -> Self
//!     where
//!         R: Rng + ?Sized,
//!     {
//!         rng.random_range(-1.0..=1.0).into()
//!     }
//! }
//!
//! fn main() -> Result<()> {
//!     let mut pop: Population<F64, TournamentSelection, UniformCrossover, _, ThreadRng> =
//!         PopulationBuilder::new(None, |c| c.value.iter().map(|g: &F64| g.0).sum::<f64>())
//!             .with_chromo_size(50)
//!             .with_population_size(100)
//!             .with_mutation_rate(0.02)
//!             .with_elitism(true)
//!             // .with_print(true) // uncomment to print the best chromosome after each generation. requires the cli feature
//!             .build()?;
//!
//!     let num_generations = 200;
//!     match pop.evolve(num_generations) {
//!         Some(best) => {
//!             println!("Evolution complete after {num_generations} generations.");
//!             println!(
//!                 "Best fitness found: {}, chromosome: {:?}",
//!                 best.fitness, best.value
//!             );
//!         }
//!         None => println!("Evolution ended with an empty population."),
//!     }
//!
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(clippy::pedantic)]

mod crossover;
mod errors;
mod selection;

pub use crossover::{Crossover, UniformCrossover};
pub use errors::{NeuralError, Result};
use rand::{seq::IndexedRandom, Rng};
pub use selection::{RouletteWheelSelection, Selection, TournamentSelection};
use std::cmp::Ordering;

#[cfg(feature = "cli")]
use colored::Colorize;

/// Trait for generating a random Gene
pub trait Gene: Clone {
    /// Returns a random [`Gene`]
    ///
    /// ## Arguments
    ///
    /// * `rng` - The random number generator
    fn generate_gene<R>(rng: &mut R) -> Self
    where
        R: Rng + ?Sized;
}

/// Represents a Chromosome with [Genes](Gene)
#[derive(Debug, Clone, PartialEq)]
pub struct Chromosome<G> {
    pub value: Vec<G>,
    pub fitness: f64,
}

impl<G> Chromosome<G> {
    /// Creates a new [Chromosome]
    #[must_use]
    pub fn new(value: Vec<G>) -> Self {
        Self {
            value,
            fitness: 0.0,
        }
    }
}

impl<G> Eq for Chromosome<G> where G: PartialEq {}

impl<G> PartialOrd for Chromosome<G>
where
    G: PartialEq,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<G> Ord for Chromosome<G>
where
    G: PartialEq,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.fitness
            .partial_cmp(&other.fitness)
            .unwrap_or(Ordering::Equal)
    }
}

/// Represents a Population of [Chromosomes](Chromosome)
#[derive(Debug)]
pub struct Population<G, S, C, F, R>
where
    R: Rng + ?Sized,
{
    pub chromo_size: usize,
    pub pop_size: usize,
    pub mut_rate: f64,
    pub population: Vec<Chromosome<G>>,
    pub eval_fn: F,
    pub selection: S,
    pub crossover: C,
    pub elitism: bool,
    pub rng: Box<R>,

    #[cfg(feature = "cli")]
    pub print: bool,
}

impl<G, S, C, F, R> Population<G, S, C, F, R>
where
    G: Gene,
    S: Selection<G>,
    C: Crossover<G>,
    F: FnMut(&Chromosome<G>) -> f64,
    R: Rng + ?Sized,
{
    /// Returns the best [Chromosome], the chromosome with the highest fitness
    #[must_use]
    pub fn best(&self) -> Option<&Chromosome<G>> {
        let mut best_fitness = 0.0;
        let mut best_match = None;

        for (i, c) in self.population.iter().enumerate() {
            if c.fitness > best_fitness {
                best_fitness = c.fitness;
                best_match = Some(i);
            }
        }

        match best_match {
            Some(i) => Some(&self.population[i]),
            None => None,
        }
    }

    /// Evaluates the chromosomes in the population
    pub fn evaluate(&mut self) {
        self.population.iter_mut().for_each(|c| {
            c.fitness = (self.eval_fn)(c);
        });
    }

    /// Evolves the [Population] and returns the best chromosome
    ///
    /// ## Arguments
    ///
    /// * `generations` - The number of generations to evolve the [Population] for
    /// * `rng` - The random number generator
    ///
    /// ## Returns
    ///
    /// The best [Chromosome] in the [Population]
    #[allow(clippy::used_underscore_binding)]
    pub fn evolve(&mut self, generations: u32) -> Option<Chromosome<G>> {
        if self.population.is_empty() {
            return None;
        }

        let elitism_offset = usize::from(self.elitism);
        for _gen in 0..generations {
            let mut next_gen = Vec::with_capacity(self.pop_size);

            if self.elitism {
                if let Some(best) = self.best() {
                    next_gen.push(best.clone());
                }
            }

            let fill_count = self.pop_size - next_gen.len();
            for _ in elitism_offset..fill_count {
                let parents = self.selection.select(&self.population, 2, &mut self.rng);
                if parents.len() < 2 {
                    if let Some(ind) = self.population.choose(&mut self.rng) {
                        next_gen.push(ind.clone());
                    } else {
                        continue;
                    }
                    continue;
                }

                if let Some(offspring) =
                    self.crossover
                        .crossover(parents[0], parents[1], &mut self.rng)
                {
                    next_gen.push(offspring);
                }
            }

            self.population = next_gen;
            self.mutate();
            self.evaluate();

            #[cfg(feature = "cli")]
            if self.print {
                if let Some(best) = self.best() {
                    println!(
                        "Generation: {}: Best fitness = {}",
                        _gen.to_string().cyan().bold(),
                        best.fitness.to_string().cyan().bold()
                    );
                }
            }
        }

        self.best().cloned()
    }

    /// Mutates the chromosomes in the population
    ///
    /// ## Arguments
    ///
    /// * `rng` - The random number generator
    pub fn mutate(&mut self) {
        self.population.iter_mut().for_each(|c| {
            for g in &mut c.value {
                if self.rng.random_bool(self.mut_rate) {
                    *g = G::generate_gene(&mut self.rng);
                }
            }
        });
    }

    /// Returns the worst [Chromosome], the chromosome with the lowest fitness
    #[must_use]
    pub fn worst(&self) -> Option<&Chromosome<G>> {
        if self.population.is_empty() {
            return None;
        }

        match self.worst_index() {
            Some(i) => Some(&self.population[i]),
            None => None,
        }
    }

    /// Returns the index of the worst [Chromosome], the chromosome with the lowest fitness
    #[must_use]
    pub fn worst_index(&self) -> Option<usize> {
        if self.population.is_empty() {
            return None;
        }

        let mut best_fitness = self.population[0].fitness;
        let mut best_match = None;

        for (i, c) in self.population.iter().enumerate().skip(1) {
            if c.fitness < best_fitness {
                best_fitness = c.fitness;
                best_match = Some(i);
            }
        }

        best_match
    }
}

/// Builder for a [Population]
pub struct PopulationBuilder<G, S, C, F, R> {
    chromo_size: usize,
    pop_size: usize,
    mut_rate: f64,
    population: Option<Vec<Chromosome<G>>>,
    eval_fn: F,
    selection: S,
    crossover: C,
    elitism: bool,
    rng: R,

    #[cfg(feature = "cli")]
    print: bool,
}

impl<G, S, C, F, R> PopulationBuilder<G, S, C, F, R>
where
    G: Gene,
    S: Selection<G>,
    C: Crossover<G>,
    F: FnMut(&Chromosome<G>) -> f64,
    R: Rng + Default,
{
    /// Creates a new [`PopulationBuilder`]
    ///
    /// ## Arguments
    ///
    /// * `population` - The population to use
    /// * `eval_fn` - The evaluation function
    ///
    /// ## Returns
    ///
    /// A new [`PopulationBuilder`]
    #[must_use]
    pub fn new(population: Option<Vec<Chromosome<G>>>, eval_fn: F) -> Self {
        Self {
            chromo_size: 10,
            pop_size: 10,
            mut_rate: 0.015,
            population,
            eval_fn,
            selection: S::default(),
            crossover: C::default(),
            elitism: false,
            rng: R::default(),

            #[cfg(feature = "cli")]
            print: false,
        }
    }

    /// Sets the chromosome size, how many genes
    ///
    /// ## Arguments
    ///
    /// * `chromo_size` - The chromosome size
    #[must_use]
    pub fn with_chromo_size(mut self, chromo_size: usize) -> Self {
        self.chromo_size = chromo_size;
        self
    }

    /// Sets the population size
    ///
    /// ## Arguments
    ///
    /// * `pop_size` - The population size
    #[must_use]
    pub fn with_population_size(mut self, pop_size: usize) -> Self {
        self.pop_size = pop_size;
        self
    }

    /// Sets the mutation rate
    ///
    /// ## Arguments
    ///
    /// * `mutation_rate` - The mutation rate
    #[must_use]
    pub fn with_mutation_rate(mut self, mutation_rate: f64) -> Self {
        self.mut_rate = mutation_rate;
        self
    }

    /// Sets the elitism flag
    ///
    /// ## Arguments
    ///
    /// * `elitism` - Whether or not to use elitism (keep the best chromosome)
    #[must_use]
    pub fn with_elitism(mut self, elitism: bool) -> Self {
        self.elitism = elitism;
        self
    }

    /// Sets the print flag
    ///
    /// ## Arguments
    ///
    /// * `print` - The print flag
    #[must_use]
    #[cfg(feature = "cli")]
    pub fn with_print(mut self, print: bool) -> Self {
        self.print = print;
        self
    }

    /// Sets the random number generator
    ///
    /// ## Arguments
    ///
    /// * `rng` - The random number generator to use
    #[must_use]
    pub fn with_rng(mut self, rng: R) -> Self {
        self.rng = rng;
        self
    }

    /// Sets the selection method
    ///
    /// ## Arguments
    ///
    /// * `selection` - The selection method
    #[must_use]
    pub fn with_selection(mut self, selection: S) -> Self {
        self.selection = selection;
        self
    }

    /// Sets the crossover method
    ///
    /// ## Arguments
    ///
    /// * `crossover` - The crossover method
    #[must_use]
    pub fn with_crossover(mut self, crossover: C) -> Self {
        self.crossover = crossover;
        self
    }

    /// Builds the [Population]
    ///
    /// ## Errors
    ///
    /// - [`NeuralError::NoPopSize`]: If the population size is not set
    /// - [`NeuralError::NoChromoSize`]: If the chromosome size is not set
    /// - [`NeuralError::NoMutationRate`]: If the mutation rate is not set
    pub fn build(self) -> Result<Population<G, S, C, F, R>> {
        let mut n = Population {
            chromo_size: self.chromo_size,
            pop_size: self.pop_size,
            mut_rate: self.mut_rate,
            population: Vec::new(),
            eval_fn: self.eval_fn,
            selection: self.selection,
            crossover: self.crossover,
            elitism: self.elitism,
            rng: Box::new(self.rng),

            #[cfg(feature = "cli")]
            print: self.print,
        };

        if let Some(pop) = self.population {
            n.population = pop;
        } else {
            let mut pop = Vec::with_capacity(n.pop_size);
            for _ in 0..n.pop_size {
                pop.push(Chromosome {
                    value: (0..n.chromo_size)
                        .map(|_| G::generate_gene(&mut n.rng))
                        .collect(),
                    fitness: 0.0,
                });
            }
            n.population = pop;
        }

        if n.chromo_size == 0 {
            return Err(NeuralError::NoChromoSize);
        }

        if n.pop_size == 0 {
            return Err(NeuralError::NoPopSize);
        }

        if !(0.0..=1.0).contains(&n.mut_rate) {
            return Err(NeuralError::NoMutationRate);
        }

        n.evaluate();

        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::ThreadRng;

    macro_rules! generate_selection_test {
        ($name:ident, $selection:ty, $crossover:ty) => {
            #[test]
            fn $name() -> Result<()> {
                #[derive(Debug, Clone, PartialEq, PartialOrd)]
                struct F64(f64);

                impl Eq for F64 {}
                impl From<f64> for F64 {
                    fn from(value: f64) -> Self {
                        Self(value)
                    }
                }

                impl Gene for F64 {
                    fn generate_gene<R>(rng: &mut R) -> Self
                    where
                        R: Rng + ?Sized,
                    {
                        rng.random_range(-1.0..=1.0).into()
                    }
                }

                let mut pop: Population<F64, $selection, $crossover, _, ThreadRng> =
                    PopulationBuilder::new(None, |c| {
                        c.value.iter().map(|g: &F64| g.0).sum::<f64>()
                    })
                    .with_chromo_size(50)
                    .with_population_size(100)
                    .with_mutation_rate(0.02)
                    .build()?;
                let num_generations = 200;

                pop.evolve(num_generations);
                Ok(())
            }
        };
    }

    generate_selection_test!(
        test_population_roulette_wheel_uniform,
        RouletteWheelSelection,
        UniformCrossover
    );
    generate_selection_test!(
        test_population_tournament_uniform,
        TournamentSelection,
        UniformCrossover
    );

    #[test]
    #[should_panic(expected = "NoPopSize")]
    #[allow(non_local_definitions)]
    fn test_no_pop_size() {
        impl Gene for i32 {
            fn generate_gene<R>(rng: &mut R) -> Self
            where
                R: Rng + ?Sized,
            {
                rng.random()
            }
        }

        PopulationBuilder::<i32, TournamentSelection, UniformCrossover, _, ThreadRng>::new(
            None,
            |c| f64::from(c.value.iter().sum::<i32>()),
        )
        .with_population_size(0)
        .build()
        .unwrap();
    }
}
