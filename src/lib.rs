use pyo3::prelude::*;

mod bm25l;
mod bm25okapi;
mod bm25plus;
mod index;
mod optimizations;
mod scoring;

/// A Python module implemented in Rust.
#[pymodule]
pub fn _bm25_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<bm25okapi::BM25Okapi>()?;
    m.add_class::<bm25okapi::PyPreprocessedQuery>()?;
    m.add_class::<bm25plus::BM25Plus>()?;
    m.add_class::<bm25l::BM25L>()?;
    Ok(())
}
