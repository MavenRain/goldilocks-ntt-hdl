//! Newtypes for pipeline stage parameters.

/// A stage index in the NTT pipeline (0..23).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct StageIndex(usize);

impl StageIndex {
    /// Create a new stage index.
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// The underlying index value.
    #[must_use]
    pub fn value(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for StageIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stage[{}]", self.0)
    }
}

/// The delay depth for a given SDF stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct DelayDepth(usize);

impl DelayDepth {
    /// Create a new delay depth.
    pub fn new(depth: usize) -> Self {
        Self(depth)
    }

    /// The underlying depth value.
    #[must_use]
    pub fn value(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for DelayDepth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "delay({})", self.0)
    }
}

/// Pipeline latency in clock cycles for a given module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct PipelineLatency(usize);

impl PipelineLatency {
    /// Create a new pipeline latency.
    pub fn new(cycles: usize) -> Self {
        Self(cycles)
    }

    /// The underlying cycle count.
    #[must_use]
    pub fn value(self) -> usize {
        self.0
    }
}

/// Signal type at a pipeline stage boundary.
///
/// Each vertex in the [`NttGraph`](crate::graph::ntt_graph::NttGraph)
/// maps to a `StageSignal` describing the wire type at that boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct StageSignal {
    stage: StageIndex,
    width_bits: usize,
}

impl StageSignal {
    /// Signal width for Goldilocks field elements.
    const GOLDILOCKS_WIDTH: usize = 64;

    /// Create a stage signal for a given stage boundary.
    pub fn goldilocks(stage: StageIndex) -> Self {
        Self {
            stage,
            width_bits: Self::GOLDILOCKS_WIDTH,
        }
    }

    /// The stage index this signal belongs to.
    pub fn stage(&self) -> StageIndex {
        self.stage
    }

    /// The bit width of the signal.
    #[must_use]
    pub fn width_bits(&self) -> usize {
        self.width_bits
    }
}
