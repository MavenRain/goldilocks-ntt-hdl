//! SDF stage descriptors: pure data describing how to build each pipeline stage.
//!
//! These descriptors are the morphisms in the free category interpretation.
//! They carry all parameters needed to instantiate an HDL module, but
//! contain no HDL types themselves.

use crate::field::element::GoldilocksElement;
use crate::interpret::signal::{DelayDepth, PipelineLatency, StageIndex};

/// The standard multiplier pipeline latency in clock cycles.
const MULTIPLIER_LATENCY: usize = 7;

/// Descriptor for a single SDF stage or a composed sub-pipeline.
///
/// This is the morphism type produced by
/// [`GraphMorphism::map_edge`](comp_cat_rs::collapse::free_category::GraphMorphism::map_edge)
/// and composed via the `comp_fn` in
/// [`interpret`](comp_cat_rs::collapse::free_category::interpret).
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum SdfStageDescriptor {
    /// A single SDF stage.
    Single {
        /// Which stage in the pipeline (0..23).
        stage_index: StageIndex,
        /// Delay line depth for this stage.
        delay_depth: DelayDepth,
        /// Multiplier pipeline latency.
        multiplier_latency: PipelineLatency,
        /// The primitive root of unity for twiddle generation at this stage.
        twiddle_root: GoldilocksElement,
    },
    /// The identity (pass-through) descriptor.
    Identity,
    /// A composed sequence of stage descriptors.
    Composed(Vec<SdfStageDescriptor>),
}

impl SdfStageDescriptor {
    /// Create a descriptor for a single SDF stage.
    ///
    /// Stage k has delay depth 2^(23-k) and twiddle root `w^(2^k)`
    /// where `w` is the primitive 2^24-th root of unity.
    pub fn single(stage_index: StageIndex, twiddle_root: GoldilocksElement) -> Self {
        let k = stage_index.value();
        Self::Single {
            stage_index,
            delay_depth: DelayDepth::new(1 << (23 - k)),
            multiplier_latency: PipelineLatency::new(MULTIPLIER_LATENCY),
            twiddle_root,
        }
    }

    /// The identity descriptor (pass-through, no transformation).
    pub fn identity() -> Self {
        Self::Identity
    }

    /// Compose two descriptors sequentially.
    ///
    /// This is the `comp_fn` used by
    /// [`interpret`](comp_cat_rs::collapse::free_category::interpret).
    pub fn compose(self, other: Self) -> Self {
        match (self, other) {
            (Self::Identity, b) => b,
            (a, Self::Identity) => a,
            (Self::Composed(a), Self::Composed(b)) => {
                Self::Composed(a.into_iter().chain(b).collect())
            }
            (Self::Composed(a), b) => {
                Self::Composed(a.into_iter().chain(std::iter::once(b)).collect())
            }
            (a, Self::Composed(b)) => {
                Self::Composed(std::iter::once(a).chain(b).collect())
            }
            (a, b) => Self::Composed(vec![a, b]),
        }
    }

    /// The total number of single stages in this descriptor.
    #[must_use]
    pub fn stage_count(&self) -> usize {
        match self {
            Self::Identity => 0,
            Self::Single { .. } => 1,
            Self::Composed(stages) => stages.iter().map(Self::stage_count).sum(),
        }
    }

    /// Iterate over the single-stage descriptors in order.
    pub fn singles(&self) -> Vec<&Self> {
        match self {
            Self::Identity => vec![],
            s @ Self::Single { .. } => vec![s],
            Self::Composed(stages) => stages.iter().flat_map(Self::singles).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_neutral_element() {
        let stage = SdfStageDescriptor::single(
            StageIndex::new(0),
            GoldilocksElement::ONE,
        );
        let composed_left = SdfStageDescriptor::identity().compose(stage.clone());
        let composed_right = stage.clone().compose(SdfStageDescriptor::identity());
        assert_eq!(composed_left, stage);
        assert_eq!(composed_right, stage);
    }

    #[test]
    fn compose_two_singles() {
        let a = SdfStageDescriptor::single(StageIndex::new(0), GoldilocksElement::ONE);
        let b = SdfStageDescriptor::single(StageIndex::new(1), GoldilocksElement::ONE);
        let composed = a.compose(b);
        assert_eq!(composed.stage_count(), 2);
    }

    #[test]
    fn compose_is_associative() {
        let a = SdfStageDescriptor::single(StageIndex::new(0), GoldilocksElement::ONE);
        let b = SdfStageDescriptor::single(StageIndex::new(1), GoldilocksElement::ONE);
        let c = SdfStageDescriptor::single(StageIndex::new(2), GoldilocksElement::ONE);

        let left = a.clone().compose(b.clone()).compose(c.clone());
        let right = a.compose(b.compose(c));

        // Both have the same number of stages
        assert_eq!(left.stage_count(), 3);
        assert_eq!(right.stage_count(), 3);
    }

    #[test]
    fn stage_0_has_correct_delay_depth() {
        let desc = SdfStageDescriptor::single(StageIndex::new(0), GoldilocksElement::ONE);
        match desc {
            SdfStageDescriptor::Single { delay_depth, .. } => {
                assert_eq!(delay_depth.value(), 1 << 23);
            }
            _ => panic!("expected Single variant"),
        }
    }

    #[test]
    fn stage_23_has_delay_depth_1() {
        let desc = SdfStageDescriptor::single(StageIndex::new(23), GoldilocksElement::ONE);
        match desc {
            SdfStageDescriptor::Single { delay_depth, .. } => {
                assert_eq!(delay_depth.value(), 1);
            }
            _ => panic!("expected Single variant"),
        }
    }

    #[test]
    fn singles_iterator_flattens_composed() {
        let a = SdfStageDescriptor::single(StageIndex::new(0), GoldilocksElement::ONE);
        let b = SdfStageDescriptor::single(StageIndex::new(1), GoldilocksElement::ONE);
        let c = SdfStageDescriptor::single(StageIndex::new(2), GoldilocksElement::ONE);
        let composed = a.compose(b).compose(c);
        assert_eq!(composed.singles().len(), 3);
    }
}
