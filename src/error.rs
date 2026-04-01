//! Project-wide error type for the Goldilocks NTT HDL crate.

use comp_cat_rs::collapse::free_category::FreeCategoryError;

/// Unified error type for all operations in this crate.
#[derive(Debug)]
pub enum Error {
    /// Field arithmetic error (e.g., element out of range).
    Field(String),

    /// Pipeline graph construction or traversal error.
    Graph(FreeCategoryError),

    /// HDL simulation error.
    Simulation(String),

    /// Verilog generation error.
    VerilogGen(String),

    /// Verification mismatch between golden model and HDL simulation.
    VerificationMismatch {
        /// The pipeline stage where the mismatch occurred.
        stage: usize,
        /// The expected value from the golden model.
        expected: u64,
        /// The actual value from the HDL simulation.
        actual: u64,
        /// The clock cycle at which the mismatch was detected.
        cycle: u64,
    },

    /// IO error (VCD file write, etc.).
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Field(msg) => write!(f, "field error: {msg}"),
            Self::Graph(e) => write!(f, "graph error: {e}"),
            Self::Simulation(msg) => write!(f, "simulation error: {msg}"),
            Self::VerilogGen(msg) => write!(f, "verilog generation error: {msg}"),
            Self::VerificationMismatch {
                stage,
                expected,
                actual,
                cycle,
            } => write!(
                f,
                "verification mismatch at stage {stage}, cycle {cycle}: \
                 expected {expected:#018x}, got {actual:#018x}"
            ),
            Self::Io(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Graph(e) => Some(e),
            Self::Field(_)
            | Self::Simulation(_)
            | Self::VerilogGen(_)
            | Self::VerificationMismatch { .. } => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<FreeCategoryError> for Error {
    fn from(e: FreeCategoryError) -> Self {
        Self::Graph(e)
    }
}
