use crate::index::{
    build_index, calc_idf_plus, tokenize_corpus_parallel, tokenize_corpus_with_py,
    tokenized_corpus_from_vecs, IndexView,
};
use crate::scoring::{
    preprocess_query, score_all_inverted, score_documents_parallel, score_queries_batch,
    top_k_inverted, PreprocessedQuery, ScoreFormula, ScoreParams,
};
use ahash::AHashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;
use string_interner::{DefaultBackend, DefaultSymbol, StringInterner};

#[cfg(feature = "numpy")]
use numpy::PyArray1;

/// BM25Plus structure with necessary fields - optimized version
#[pyclass]
pub struct BM25Plus {
    #[pyo3(get)]
    k1: f64,
    #[pyo3(get)]
    b: f64,
    #[pyo3(get)]
    delta: f64,
    #[pyo3(get)]
    corpus_size: usize,
    #[pyo3(get)]
    avgdl: f64,
    doc_freqs: Arc<Vec<AHashMap<DefaultSymbol, u32>>>,
    idf: Arc<AHashMap<DefaultSymbol, f64>>,
    doc_len: Arc<Vec<u32>>,
    inverted: Arc<AHashMap<DefaultSymbol, Vec<(u32, u32)>>>,
    interner: Arc<StringInterner<DefaultBackend>>,
    tokenizer: Option<Py<PyAny>>,
    k1_plus1: f64,
    one_minus_b: f64,
    b_over_avgdl: f64,
}

impl BM25Plus {
    fn score_params(&self) -> ScoreParams {
        ScoreParams {
            k1: self.k1,
            k1_plus1: self.k1_plus1,
            one_minus_b: self.one_minus_b,
            b_over_avgdl: self.b_over_avgdl,
        }
    }

    fn formula(&self) -> ScoreFormula {
        ScoreFormula::Plus { delta: self.delta }
    }

    fn preprocess(&self, query: &[String]) -> PreprocessedQuery {
        preprocess_query(query, &self.interner, &self.idf)
    }

    fn index_view(&self) -> IndexView<'_> {
        IndexView {
            doc_freqs: &self.doc_freqs,
            doc_len: &self.doc_len,
            inverted: &self.inverted,
            corpus_size: self.corpus_size,
            avgdl: self.avgdl,
        }
    }
}

#[pymethods]
impl BM25Plus {
    #[new]
    #[pyo3(signature = (corpus=None, tokenizer=None, tokenized_corpus=None, k1=None, b=None, delta=None))]
    pub fn new(
        py: Python,
        corpus: Option<Vec<String>>,
        tokenizer: Option<Bound<PyAny>>,
        tokenized_corpus: Option<Vec<Vec<String>>>,
        k1: Option<f64>,
        b: Option<f64>,
        delta: Option<f64>,
    ) -> PyResult<Self> {
        let k1 = k1.unwrap_or(1.5);
        let b = b.unwrap_or(0.75);
        let delta = delta.unwrap_or(1.0);

        let tokenizer = tokenizer.map(|tk| tk.into());

        let tokenized = match (corpus, tokenized_corpus) {
            (Some(corpus_docs), None) => {
                if let Some(ref tokenizer_py) = tokenizer {
                    tokenize_corpus_with_py(py, &corpus_docs, tokenizer_py)?
                } else {
                    py.allow_threads(|| tokenize_corpus_parallel(&corpus_docs))
                }
            }
            (None, Some(tokens)) => tokenized_corpus_from_vecs(tokens),
            (Some(_), Some(_)) => {
                return Err(PyErr::new::<PyValueError, _>(
                    "Provide either corpus or tokenized_corpus, not both.",
                ));
            }
            (None, None) => {
                return Err(PyErr::new::<PyValueError, _>(
                    "Either corpus or tokenized_corpus must be provided.",
                ));
            }
        };

        if tokenized.is_empty() {
            return Err(PyErr::new::<PyValueError, _>(
                "Corpus size must be greater than zero.",
            ));
        }

        let mut interner = StringInterner::default();
        let built = py.allow_threads(|| build_index(tokenized, &mut interner));
        let idf_map = py.allow_threads(|| calc_idf_plus(built.nd, built.corpus_size));

        let k1_plus1 = k1 + 1.0;
        let one_minus_b = 1.0 - b;
        let b_over_avgdl = b / built.avgdl;

        Ok(BM25Plus {
            k1,
            b,
            delta,
            corpus_size: built.corpus_size,
            avgdl: built.avgdl,
            doc_freqs: Arc::new(built.doc_freqs),
            doc_len: Arc::new(built.doc_len),
            inverted: Arc::new(built.inverted),
            idf: Arc::new(idf_map),
            interner: Arc::new(interner),
            tokenizer,
            k1_plus1,
            one_minus_b,
            b_over_avgdl,
        })
    }

    pub fn preprocess_query(&self, query: Vec<String>) -> PyResult<crate::bm25okapi::PyPreprocessedQuery> {
        Ok(crate::bm25okapi::PyPreprocessedQuery::from_inner(self.preprocess(&query)))
    }

    pub fn get_scores(&self, py: Python, query: Vec<String>) -> PyResult<Vec<f64>> {
        if self.corpus_size == 0 {
            return Ok(vec![]);
        }

        let preprocessed = self.preprocess(&query);
        if preprocessed.is_empty() {
            return Ok(vec![0.0; self.corpus_size]);
        }

        let view = self.index_view();
        let params = self.score_params();
        let formula = self.formula();

        py.allow_threads(|| Ok(score_all_inverted(&view, &preprocessed, params, formula)))
    }

    #[cfg(feature = "numpy")]
    pub fn get_scores_numpy<'py>(
        &self,
        py: Python<'py>,
        query: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let scores = self.get_scores(py, query)?;
        Ok(PyArray1::from_vec_bound(py, scores))
    }

    #[pyo3(signature = (query, documents, n=None))]
    pub fn get_top_n(
        &self,
        py: Python,
        query: Vec<String>,
        documents: Vec<String>,
        n: Option<usize>,
    ) -> PyResult<Vec<(String, f64)>> {
        let n = n.unwrap_or(5);
        if self.corpus_size != documents.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "The documents given don't match the index corpus!",
            ));
        }

        let top = self.get_top_n_indices(py, query, Some(n))?;
        Ok(top
            .into_iter()
            .map(|(i, score)| (documents[i].clone(), score))
            .collect())
    }

    #[pyo3(signature = (query, n=None))]
    pub fn get_top_n_indices(
        &self,
        py: Python,
        query: Vec<String>,
        n: Option<usize>,
    ) -> PyResult<Vec<(usize, f64)>> {
        let n = n.unwrap_or(5);
        let preprocessed = self.preprocess(&query);
        if preprocessed.is_empty() {
            return Ok((0..self.corpus_size.min(n))
                .map(|i| (i, 0.0))
                .collect());
        }

        let view = self.index_view();
        let params = self.score_params();
        let formula = self.formula();

        py.allow_threads(|| {
            Ok(top_k_inverted(&view, &preprocessed, params, formula, n))
        })
    }

    #[pyo3(signature = (query, chunk_size=None))]
    pub fn get_scores_chunked(
        &self,
        py: Python,
        query: Vec<String>,
        chunk_size: Option<usize>,
    ) -> PyResult<Vec<f64>> {
        let _chunk_size = chunk_size.unwrap_or(1000);
        self.get_scores(py, query)
    }

    pub fn get_batch_scores(
        &self,
        py: Python,
        query: Vec<String>,
        doc_ids: Vec<usize>,
    ) -> PyResult<Vec<f64>> {
        if doc_ids.is_empty() {
            return Ok(vec![]);
        }

        if doc_ids.iter().any(|&di| di >= self.corpus_size) {
            return Err(PyErr::new::<PyValueError, _>(
                "One or more document IDs are out of range.",
            ));
        }

        let preprocessed = self.preprocess(&query);
        if preprocessed.is_empty() {
            return Ok(vec![0.0; doc_ids.len()]);
        }

        let doc_freqs = Arc::clone(&self.doc_freqs);
        let doc_len = Arc::clone(&self.doc_len);
        let params = self.score_params();
        let formula = self.formula();

        py.allow_threads(|| {
            Ok(score_documents_parallel(
                &doc_ids,
                &doc_freqs,
                &doc_len,
                &preprocessed,
                params,
                formula,
            ))
        })
    }

    pub fn get_scores_batch(
        &self,
        py: Python,
        queries: Vec<Vec<String>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        if self.corpus_size == 0 {
            return Ok(vec![]);
        }

        let preprocessed: Vec<PreprocessedQuery> =
            queries.iter().map(|q| self.preprocess(q)).collect();

        let view = self.index_view();
        let params = self.score_params();
        let formula = self.formula();

        py.allow_threads(|| {
            Ok(score_queries_batch(&view, &preprocessed, params, formula))
        })
    }
}
