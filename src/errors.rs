use thiserror::Error;

pub type Result<T> = std::result::Result<T, NeuralError>;

#[derive(Debug, Error)]
pub enum NeuralError {
    #[error("population size must be greater than 0")]
    NoPopSize,

    #[error("chromosome size must be greater than 0")]
    NoChromoSize,

    #[error("mutation rate must be between 0 and 1")]
    NoMutationRate,

    #[error("parents must have the same gene length")]
    ParentsNotSameLength,
}
