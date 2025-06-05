use candle_core::error::Error;
use thiserror::Error;

#[derive(Debug, Error)]
/// Error type for koho
pub enum KohoError {
    #[error("Mismatch in expected tensor dimensions")]
    DimensionMismatch,
    #[error("Cell complex hasnt been initialized properly yet")]
    CWUninitialized,
    #[error("Invalid cell index")]
    InvalidCellIdx,
    #[error("Invalid data length")]
    InvalidDataIdx,
    #[error("Null matrix was provide")]
    NullMatrix,
    #[error("Failed to find point in cell complex")]
    NoPointFound,
    #[error("Cells are not adjacent!")]
    NotAdjacent,
    #[error("Restrict is yet to be defined for this cell incidence pair")]
    NoRestrictionDefined,
    #[error("Sections for layer {i} are not yet provided")]
    NoSections { i: usize },
    #[error("Candle module error")]
    Candle(Error),
    #[error("Misc")]
    Msg(String),
}
