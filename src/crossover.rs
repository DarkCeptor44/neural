use crate::Chromosome;
use rand::Rng;

/// Trait for crossing over two chromosomes
pub trait Crossover<G>: Default {
    /// Crosses over two chromosomes
    ///
    /// ## Arguments
    ///
    /// * `parent1` - The first parent
    /// * `parent2` - The second parent
    /// * `rng` - The random number generator
    ///
    /// ## Returns
    ///
    /// The crossed over chromosome
    fn crossover<R>(
        &self,
        parent1: &Chromosome<G>,
        parent2: &Chromosome<G>,
        rng: &mut R,
    ) -> Option<Chromosome<G>>
    where
        R: Rng + ?Sized;
}

/// Uniform crossover implementation
#[derive(Debug, Default)]
pub struct UniformCrossover {}

impl UniformCrossover {
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl<G> Crossover<G> for UniformCrossover
where
    G: Clone,
{
    fn crossover<R>(
        &self,
        parent1: &Chromosome<G>,
        parent2: &Chromosome<G>,
        rng: &mut R,
    ) -> Option<Chromosome<G>>
    where
        R: Rng + ?Sized,
    {
        if parent1.value.len() != parent2.value.len() {
            return None;
        }

        let len = parent1.value.len();
        let mut offspring = Vec::with_capacity(len);

        for i in 0..len {
            if rng.random_bool(0.5) {
                offspring.push(parent1.value[i].clone());
            } else {
                offspring.push(parent2.value[i].clone());
            }
        }

        Some(Chromosome {
            value: offspring,
            fitness: 0.0,
        })
    }
}

// TODO implement single point crossover
// TODO implement multi point crossover
