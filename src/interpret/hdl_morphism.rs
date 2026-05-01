//! Graph morphism mapping the NTT graph into HDL stage descriptors.
//!
//! This is the bridge between the categorical pipeline description
//! and the concrete HDL modules.  The
//! [`interpret`](comp_cat_rs::collapse::free_category::interpret)
//! function uses this morphism to compose the full pipeline.

use comp_cat_rs::collapse::free_category::{Edge, GraphMorphism, Vertex};

use crate::field::element::GoldilocksElement;
use crate::field::roots::primitive_root_of_unity;
use crate::graph::ntt_graph::{NTT_STAGES, NttGraph};
use crate::interpret::descriptor::SdfStageDescriptor;
use crate::interpret::signal::{StageIndex, StageSignal};

/// Maps `NttGraph` vertices to signal types and edges to stage descriptors.
///
/// Holds precomputed twiddle roots for all 24 stages.
#[derive(Debug, Clone)]
pub struct HdlInterpretation {
    twiddle_roots: Vec<GoldilocksElement>,
}

impl HdlInterpretation {
    /// Create a new interpretation with precomputed twiddle roots.
    ///
    /// For stage k, the twiddle accumulator multiplies by `w^(2^k)`
    /// where `w` is the primitive 2^24-th root of unity.  This means
    /// the root for stage k is `w^(2^k)`.
    ///
    /// # Errors
    ///
    /// Returns an error if root of unity computation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use goldilocks_ntt_hdl::interpret::hdl_morphism::HdlInterpretation;
    /// use goldilocks_ntt_hdl::interpret::descriptor::SdfStageDescriptor;
    /// use goldilocks_ntt_hdl::graph::ntt_graph::{NttGraph, full_pipeline_path};
    /// use comp_cat_rs::collapse::free_category::interpret;
    ///
    /// let interp = HdlInterpretation::new().ok();
    /// let path = full_pipeline_path().ok();
    ///
    /// let count = interp.zip(path).map(|(i, p)| {
    ///     interpret::<NttGraph, _>(
    ///         &i,
    ///         &p,
    ///         |_| SdfStageDescriptor::identity(),
    ///         SdfStageDescriptor::compose,
    ///     ).stage_count()
    /// });
    /// assert_eq!(count, Some(24));
    /// ```
    pub fn new() -> Result<Self, crate::error::Error> {
        let twiddle_roots = (0..NTT_STAGES)
            .map(|k| {
                // The twiddle accumulator for stage k steps by w^(2^k).
                // This is the same as a primitive 2^(24-k)-th root of unity.
                let order_bits = u32::try_from(NTT_STAGES - k)
                    .map_err(|e| crate::error::Error::Field(e.to_string()))?;
                primitive_root_of_unity(order_bits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { twiddle_roots })
    }
}

impl GraphMorphism<NttGraph> for HdlInterpretation {
    type Object = StageSignal;
    type Morphism = SdfStageDescriptor;

    fn map_vertex(&self, v: Vertex) -> StageSignal {
        StageSignal::goldilocks(StageIndex::new(v.index()))
    }

    fn map_edge(&self, e: Edge) -> SdfStageDescriptor {
        let k = e.index();
        let twiddle_root = self
            .twiddle_roots
            .get(k)
            .copied()
            .unwrap_or(GoldilocksElement::ONE);
        SdfStageDescriptor::single(StageIndex::new(k), twiddle_root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ntt_graph::full_pipeline_path;
    use comp_cat_rs::collapse::free_category::interpret;

    #[test]
    fn interpretation_produces_24_stages() -> Result<(), crate::error::Error> {
        let interp = HdlInterpretation::new()?;
        let path = full_pipeline_path()?;
        let descriptor = interpret::<NttGraph, _>(
            &interp,
            &path,
            |_| SdfStageDescriptor::identity(),
            SdfStageDescriptor::compose,
        );
        assert_eq!(descriptor.stage_count(), NTT_STAGES);
        Ok(())
    }

    #[test]
    fn interpretation_of_single_edge() -> Result<(), crate::error::Error> {
        let interp = HdlInterpretation::new()?;
        let desc = interp.map_edge(Edge::new(0));
        assert_eq!(desc.stage_count(), 1);
        Ok(())
    }

    #[test]
    fn vertex_signals_have_64_bit_width() -> Result<(), crate::error::Error> {
        let interp = HdlInterpretation::new()?;
        let signal = interp.map_vertex(Vertex::new(0));
        assert_eq!(signal.width_bits(), 64);
        Ok(())
    }

    #[test]
    fn sub_pipeline_interpretation() -> Result<(), crate::error::Error> {
        let interp = HdlInterpretation::new()?;
        let sub_path = crate::graph::ntt_graph::sub_pipeline_path(4, 8)?;
        let descriptor = interpret::<NttGraph, _>(
            &interp,
            &sub_path,
            |_| SdfStageDescriptor::identity(),
            SdfStageDescriptor::compose,
        );
        assert_eq!(descriptor.stage_count(), 4);
        Ok(())
    }
}
