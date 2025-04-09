use crate::Chromosome;
use rand::{seq::IndexedRandom, Rng};

/// Trait for selecting a subset of the population
pub trait Selection<G>: Default {
    /// Selects a subset of the population
    ///
    /// ## Arguments
    ///
    /// * `population` - The population to select from
    /// * `count` - The number of individuals to select
    /// * `rng` - The random number generator
    ///
    /// ## Returns
    ///
    /// A vector of selected individuals
    fn select<'pop, R>(
        &self,
        population: &'pop [Chromosome<G>],
        count: usize,
        rng: &mut R,
    ) -> Vec<&'pop Chromosome<G>>
    where
        R: Rng + ?Sized;
}

/// Roulette wheel selection implementation
#[derive(Debug, Default)]
pub struct RouletteWheelSelection {}

impl RouletteWheelSelection {
    /// Creates a new [`RouletteWheelSelection`]
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl<G> Selection<G> for RouletteWheelSelection {
    fn select<'pop, R>(
        &self,
        population: &'pop [Chromosome<G>],
        count: usize,
        rng: &mut R,
    ) -> Vec<&'pop Chromosome<G>>
    where
        R: Rng + ?Sized,
    {
        if population.is_empty() || count == 0 {
            return Vec::new();
        }

        let total_fitness: f64 = population.iter().map(|c| c.fitness).sum();
        let mut selected: Vec<&'pop Chromosome<G>> = Vec::with_capacity(count);

        if total_fitness <= 0.0 {
            for _ in 0..count {
                if let Some(individual) = population.choose(rng) {
                    selected.push(individual);
                }
            }
            return selected;
        }

        for _ in 0..count {
            let spin = rng.random_range(0.0..total_fitness);
            let mut current_sum = 0.0;

            for individual in population {
                current_sum += individual.fitness;

                if current_sum > spin {
                    selected.push(individual);
                    break;
                }
            }
        }

        selected
    }
}

/// Tournament selection implementation
#[derive(Debug)]
pub struct TournamentSelection {
    tournament_size: usize,
}

impl TournamentSelection {
    /// Creates a new [`TournamentSelection`]
    ///
    /// ## Arguments
    ///
    /// * `tournament_size` - The tournament size
    #[must_use]
    pub fn new(tournament_size: usize) -> Self {
        Self { tournament_size }
    }
}

impl Default for TournamentSelection {
    fn default() -> Self {
        Self::new(5)
    }
}

impl<G> Selection<G> for TournamentSelection {
    fn select<'pop, R>(
        &self,
        population: &'pop [Chromosome<G>],
        count: usize,
        rng: &mut R,
    ) -> Vec<&'pop Chromosome<G>>
    where
        R: Rng + ?Sized,
    {
        let mut selected: Vec<&'pop Chromosome<G>> = Vec::with_capacity(count);

        for _ in 0..count {
            let mut best_fitness = f64::NEG_INFINITY;
            let mut best_individual: Option<&'pop Chromosome<G>> = None;
            let current_tourney_size = self.tournament_size.min(population.len());

            for _ in 0..current_tourney_size {
                if let Some(individual) = population.choose(rng) {
                    if individual.fitness > best_fitness {
                        best_fitness = individual.fitness;
                        best_individual = Some(individual);
                    }
                }
            }

            if let Some(individual) = best_individual {
                selected.push(individual);
            } else if !population.is_empty() {
                if let Some(random_choice) = population.choose(rng) {
                    selected.push(random_choice);
                }
            }
        }

        selected
    }
}

// TODO implement rank selection
// TODO implement stochastic universal sampling selection
