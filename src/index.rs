//! Shared corpus indexing: tokenization, document frequencies, and inverted index.

use ahash::{AHashMap, AHashSet};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use string_interner::{DefaultBackend, DefaultSymbol, StringInterner};

/// Lowercase a token using an ASCII fast path when possible.
pub fn lowercase_token(token: &str) -> String {
    if token.bytes().all(|b| b.is_ascii()) {
        token.to_ascii_lowercase()
    } else {
        token.to_lowercase()
    }
}

/// Whitespace tokenize a document with lowercase tokens.
pub fn tokenize_document(doc: &str) -> SmallVec<[String; 16]> {
    doc.split_whitespace()
        .map(lowercase_token)
        .collect::<SmallVec<[String; 16]>>()
}

/// Tokenize a corpus in parallel (GIL must be released by caller).
pub fn tokenize_corpus_parallel(corpus: &[String]) -> Vec<SmallVec<[String; 16]>> {
    corpus
        .par_iter()
        .map(|doc| tokenize_document(doc))
        .collect()
}

/// Tokenize using a Python callback (requires GIL).
pub fn tokenize_corpus_with_py(
    py: Python,
    corpus: &[String],
    tokenizer_py: &Py<PyAny>,
) -> PyResult<Vec<SmallVec<[String; 16]>>> {
    let mut tokenized_corpus = Vec::with_capacity(corpus.len());
    for doc in corpus {
        let tokens: Vec<String> = tokenizer_py
            .call1(py, (doc,))
            .map_err(|e| PyValueError::new_err(format!("Tokenizer failed: {e}")))?
            .extract(py)
            .map_err(|e| PyValueError::new_err(format!("Failed to extract tokens: {e}")))?;
        tokenized_corpus.push(SmallVec::from_vec(tokens));
    }
    Ok(tokenized_corpus)
}

/// Convert pre-tokenized input from Python into SmallVec documents.
pub fn tokenized_corpus_from_vecs(corpus: Vec<Vec<String>>) -> Vec<SmallVec<[String; 16]>> {
    corpus
        .into_iter()
        .map(SmallVec::from_vec)
        .collect()
}

/// Lightweight borrowed view of a built index for scoring.
pub struct IndexView<'a> {
    pub doc_freqs: &'a [AHashMap<DefaultSymbol, u32>],
    pub doc_len: &'a [u32],
    pub inverted: &'a AHashMap<DefaultSymbol, Vec<(u32, u32)>>,
    pub corpus_size: usize,
    pub avgdl: f64,
}

impl<'a> IndexView<'a> {
    pub fn doc_len_f64(&self, doc_id: usize) -> f64 {
        self.doc_len[doc_id] as f64
    }
}

/// Built index structures shared by all BM25 variants.
pub struct BuiltIndex {
    pub doc_freqs: Vec<AHashMap<DefaultSymbol, u32>>,
    pub doc_len: Vec<u32>,
    pub inverted: AHashMap<DefaultSymbol, Vec<(u32, u32)>>,
    pub nd: AHashMap<DefaultSymbol, u32>,
    pub avgdl: f64,
    pub corpus_size: usize,
}

/// Build document frequencies, inverted index, and document-frequency counts.
pub fn build_index(
    corpus: Vec<SmallVec<[String; 16]>>,
    interner: &mut StringInterner<DefaultBackend>,
) -> BuiltIndex {
    let corpus_size = corpus.len();

    let mut all_terms = AHashSet::new();
    for doc in &corpus {
        for term in doc {
            all_terms.insert(term.as_str());
        }
    }

    let _: Vec<_> = all_terms
        .iter()
        .map(|term| interner.get_or_intern(term))
        .collect();

    let doc_data: Vec<(AHashMap<DefaultSymbol, u32>, u32, AHashSet<DefaultSymbol>)> = corpus
        .into_par_iter()
        .map(|doc| {
            let mut freq_map = AHashMap::with_capacity(doc.len().min(64));
            let mut unique_terms = AHashSet::with_capacity(doc.len().min(64));

            for term in &doc {
                if let Some(symbol) = interner.get(term) {
                    *freq_map.entry(symbol).or_insert(0) += 1;
                    unique_terms.insert(symbol);
                }
            }

            (freq_map, doc.len() as u32, unique_terms)
        })
        .collect();

    let mut doc_freqs = Vec::with_capacity(corpus_size);
    let mut doc_len = Vec::with_capacity(corpus_size);
    let mut total_len = 0u64;

    let nd: AHashMap<DefaultSymbol, u32> = doc_data
        .par_iter()
        .map(|(_, _, unique_terms)| {
            let mut local = AHashMap::new();
            for &symbol in unique_terms {
                *local.entry(symbol).or_insert(0) += 1;
            }
            local
        })
        .reduce(
            || AHashMap::new(),
            |mut acc, local| {
                for (symbol, count) in local {
                    *acc.entry(symbol).or_insert(0) += count;
                }
                acc
            },
        );

    for (freq_map, len, _) in doc_data {
        total_len += len as u64;
        doc_len.push(len);
        doc_freqs.push(freq_map);
    }

    let avgdl = total_len as f64 / corpus_size as f64;

    let mut inverted: AHashMap<DefaultSymbol, Vec<(u32, u32)>> = AHashMap::new();
    for (doc_id, freq_map) in doc_freqs.iter().enumerate() {
        for (&symbol, &freq) in freq_map {
            inverted
                .entry(symbol)
                .or_default()
                .push((doc_id as u32, freq));
        }
    }

    BuiltIndex {
        doc_freqs,
        doc_len,
        inverted,
        nd,
        avgdl,
        corpus_size,
    }
}

/// Robertson–Walker Okapi IDF with epsilon floor for negative values.
pub fn calc_idf_okapi(
    nd: AHashMap<DefaultSymbol, u32>,
    corpus_size: usize,
    epsilon: f64,
) -> AHashMap<DefaultSymbol, f64> {
    let corpus_size_f64 = corpus_size as f64;

    let idf_values: Vec<(DefaultSymbol, f64)> = nd
        .par_iter()
        .map(|(&symbol, &doc_freq)| {
            let doc_freq_f64 = doc_freq as f64;
            let idf = ((corpus_size_f64 - doc_freq_f64 + 0.5) / (doc_freq_f64 + 0.5)).ln();
            (symbol, idf)
        })
        .collect();

    let idf_sum: f64 = idf_values.par_iter().map(|(_, idf)| *idf).sum();
    let average_idf = idf_sum / idf_values.len() as f64;
    let eps = epsilon * average_idf;

    idf_values
        .into_iter()
        .map(|(symbol, idf)| {
            let adjusted_idf = if idf < 0.0 { eps } else { idf };
            (symbol, adjusted_idf)
        })
        .collect()
}

/// BM25Plus IDF.
pub fn calc_idf_plus(
    nd: AHashMap<DefaultSymbol, u32>,
    corpus_size: usize,
) -> AHashMap<DefaultSymbol, f64> {
    let corpus_size_f64 = corpus_size as f64;

    nd.into_iter()
        .map(|(symbol, freq)| {
            let freq_f64 = freq as f64;
            let idf_val = (corpus_size_f64 + 1.0).ln() - freq_f64.ln();
            (symbol, idf_val)
        })
        .collect()
}

/// BM25L IDF.
pub fn calc_idf_l(
    nd: AHashMap<DefaultSymbol, u32>,
    corpus_size: usize,
) -> AHashMap<DefaultSymbol, f64> {
    let corpus_size_f64 = corpus_size as f64;

    nd.into_iter()
        .map(|(symbol, freq)| {
            let freq_f64 = freq as f64;
            let idf_val = (corpus_size_f64 + 1.0).ln() - (freq_f64 + 0.5).ln();
            (symbol, idf_val)
        })
        .collect()
}
