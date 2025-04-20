# Neural

**Neural** is a library for Genetic Algorithms in Rust.

## Concepts

- [**Genetic Algorithm**](https://en.wikipedia.org/wiki/Genetic_algorithm) - A algorithm for solving optimization problems by simulating the process of natural selection, evolution, mutation and crossover (reproduction).
- **Gene** - A single value that represents a possible solution to a problem.
- **Chromosome** - A collection of genes that represent a possible solution to a problem.
- **Population** - A collection of chromosomes that represent potential solutions to a problem.
- **Selection** - A trait for selecting a subset of the population.
- **Crossover** - A trait for crossing over two chromosomes.

## Features

- **print** - Allows using the `with_print` method on a `PopulationBuilder` to print the population's best chromosome after each generation and enables colored output.

## Getting Started

```sh
cargo add neural

# or add it with the print feature
cargo add neural --features print
```

Or add it as a dependency in your `Cargo.toml`:

```toml
[dependencies]
neural = "^0.3"

# or add it with the print feature
[dependencies]
neural = { version = "^0.3", features = ["print"] }
```

## Usage

```rust
use neural::{Gene, Population, PopulationBuilder, Result, TournamentSelection, UniformCrossover};
use rand::{rngs::ThreadRng, Rng};
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct F64(f64);

impl Eq for F64 {}

impl Display for F64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<f64> for F64 {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl From<F64> for f64 {
    fn from(val: F64) -> Self {
        val.0
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

fn main() -> Result<()> {
    let mut pop: Population<F64, TournamentSelection, UniformCrossover, _, ThreadRng> =
        PopulationBuilder::new(None, |c| c.value.iter().map(|g: &F64| g.0).sum::<f64>())
            .with_chromo_size(50)
            .with_population_size(100)
            .with_mutation_rate(0.02)
            .with_elitism(true)
            // .with_print(true) // uncomment to print the best chromosome after each generation. requires the `print` feature
            .build()?;

    let num_generations = 200;
    match pop.evolve(num_generations) {
        Some(best) => {
            println!("Evolution complete after {num_generations} generations.");
            println!(
                "Best fitness found: {}, chromosome: {:?}",
                best.fitness, best.value
            );
        }
        None => println!("Evolution ended with an empty population."),
    }

    Ok(())
}
```

## License

This library is distributed under the terms of the [MIT License](LICENSE).
