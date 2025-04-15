use thiserror::Error;

pub type Result<T> = std::result::Result<T, NeuralError>;

/// Errors for neural
#[derive(Debug, Error)]
pub enum NeuralError {
    /// The population size is 0
    #[error("population size must be greater than 0")]
    NoPopSize,

    /// The chromosome size/gene size is 0
    #[error("chromosome size must be greater than 0")]
    NoChromoSize,

    /// The mutation rate is not between 0 and 1
    #[error("mutation rate must be between 0 and 1")]
    NoMutationRate,

    /// The crossover parents don't have the same gene length
    #[error("parents must have the same gene length")]
    ParentsNotSameLength,
}
