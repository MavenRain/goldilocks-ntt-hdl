//! The NTT pipeline modeled as a directed graph for the free category.
//!
//! The 24-stage SDF pipeline is represented as a linear graph with
//! 25 vertices (stage boundaries) and 24 edges (stages).  The
//! [`comp_cat_rs::collapse::free_category`] infrastructure gives us
//! path composition and the universal property for interpreting this
//! graph into concrete `hdl-cat` modules.

use comp_cat_rs::collapse::free_category::{Edge, FreeCategoryError, Graph, Path, Vertex};

/// Number of NTT stages (log2 of transform size).
pub const NTT_STAGES: usize = 24;

/// Number of vertices in the pipeline graph (one more than stages).
pub const NTT_VERTICES: usize = NTT_STAGES + 1;

/// The directed graph underlying the 2^24-point SDF NTT pipeline.
///
/// - 25 vertices (indices 0..24): stage boundaries.
///   Vertex 0 is the pipeline input, vertex 24 is the pipeline output.
/// - 24 edges (indices 0..23): SDF stages.
///   Edge k connects vertex k to vertex k+1.
///
/// This implements [`Graph`] from `comp-cat-rs`, enabling the free
/// category machinery (paths, composition, interpretation).
pub struct NttGraph;

impl Graph for NttGraph {
    fn vertex_count(&self) -> usize {
        NTT_VERTICES
    }

    fn edge_count(&self) -> usize {
        NTT_STAGES
    }

    fn source(&self, edge: Edge) -> Result<Vertex, FreeCategoryError> {
        if edge.index() < NTT_STAGES {
            Ok(Vertex::new(edge.index()))
        } else {
            Err(FreeCategoryError::EdgeOutOfBounds {
                edge,
                count: NTT_STAGES,
            })
        }
    }

    fn target(&self, edge: Edge) -> Result<Vertex, FreeCategoryError> {
        if edge.index() < NTT_STAGES {
            Ok(Vertex::new(edge.index() + 1))
        } else {
            Err(FreeCategoryError::EdgeOutOfBounds {
                edge,
                count: NTT_STAGES,
            })
        }
    }
}

/// Build the full pipeline path from vertex 0 to vertex 24.
///
/// Composes all 24 singleton edge paths into a single path
/// representing the complete NTT pipeline.
///
/// # Errors
///
/// Returns an error if path construction or composition fails
/// (should not happen for a well-formed `NttGraph`).
///
/// # Examples
///
/// ```
/// use goldilocks_ntt_hdl::graph::ntt_graph::full_pipeline_path;
///
/// let path = full_pipeline_path();
/// assert!(path.is_ok());
/// assert_eq!(path.map(|p| p.len()).unwrap_or(0), 24);
/// ```
pub fn full_pipeline_path() -> Result<Path, FreeCategoryError> {
    let graph = NttGraph;
    (0..NTT_STAGES)
        .map(|k| Path::singleton(&graph, Edge::new(k)))
        .try_fold(Path::identity(Vertex::new(0)), |acc, edge_path| {
            acc.compose(edge_path?)
        })
}

/// Build a sub-pipeline path from stage `start` to stage `end` (exclusive).
///
/// For example, `sub_pipeline_path(2, 5)` yields the path through
/// stages 2, 3, 4 (edges 2, 3, 4 connecting vertices 2 through 5).
///
/// # Errors
///
/// Returns an error if `start >= end` or `end > NTT_STAGES`.
pub fn sub_pipeline_path(start: usize, end: usize) -> Result<Path, FreeCategoryError> {
    if start >= end || end > NTT_STAGES {
        Err(FreeCategoryError::EdgeOutOfBounds {
            edge: Edge::new(end),
            count: NTT_STAGES,
        })
    } else {
        let graph = NttGraph;
        (start..end)
            .map(|k| Path::singleton(&graph, Edge::new(k)))
            .try_fold(Path::identity(Vertex::new(start)), |acc, edge_path| {
                acc.compose(edge_path?)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_has_correct_dimensions() {
        let g = NttGraph;
        assert_eq!(g.vertex_count(), 25);
        assert_eq!(g.edge_count(), 24);
    }

    #[test]
    fn edge_source_target_are_adjacent() -> Result<(), FreeCategoryError> {
        let g = NttGraph;
        (0..NTT_STAGES).try_for_each(|k| {
            let e = Edge::new(k);
            let s = g.source(e)?;
            let t = g.target(e)?;
            assert_eq!(s.index(), k);
            assert_eq!(t.index(), k + 1);
            Ok(())
        })
    }

    #[test]
    fn out_of_bounds_edge_is_error() {
        let g = NttGraph;
        assert!(g.source(Edge::new(24)).is_err());
        assert!(g.target(Edge::new(24)).is_err());
    }

    #[test]
    fn full_pipeline_path_spans_all_stages() -> Result<(), FreeCategoryError> {
        let path = full_pipeline_path()?;
        assert_eq!(path.source().index(), 0);
        assert_eq!(path.target().index(), NTT_VERTICES - 1);
        assert_eq!(path.len(), NTT_STAGES);
        Ok(())
    }

    #[test]
    fn sub_pipeline_path_spans_correct_range() -> Result<(), FreeCategoryError> {
        let path = sub_pipeline_path(5, 10)?;
        assert_eq!(path.source().index(), 5);
        assert_eq!(path.target().index(), 10);
        assert_eq!(path.len(), 5);
        Ok(())
    }

    #[test]
    fn sub_pipeline_single_stage() -> Result<(), FreeCategoryError> {
        let path = sub_pipeline_path(0, 1)?;
        assert_eq!(path.len(), 1);
        assert_eq!(path.source().index(), 0);
        assert_eq!(path.target().index(), 1);
        Ok(())
    }

    #[test]
    fn sub_pipeline_invalid_range_is_error() {
        assert!(sub_pipeline_path(5, 5).is_err());
        assert!(sub_pipeline_path(10, 5).is_err());
        assert!(sub_pipeline_path(0, 25).is_err());
    }

    #[test]
    fn composing_sub_pipelines_equals_full() -> Result<(), FreeCategoryError> {
        let first_half = sub_pipeline_path(0, 12)?;
        let second_half = sub_pipeline_path(12, 24)?;
        let composed = first_half.compose(second_half)?;
        let full = full_pipeline_path()?;
        assert_eq!(composed.source().index(), full.source().index());
        assert_eq!(composed.target().index(), full.target().index());
        assert_eq!(composed.len(), full.len());
        Ok(())
    }
}
