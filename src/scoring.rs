//! BM25 scoring engines: inverted-index accumulation, brute-force fallback, and top-k.

use crate::index::IndexView;
use ahash::AHashMap;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::Ordering;
use string_interner::{DefaultBackend, DefaultSymbol, StringInterner};

/// Preprocessed query terms with resolved IDF values.
#[derive(Clone)]
pub struct PreprocessedQuery {
    pub terms: SmallVec<[(DefaultSymbol, f64); 8]>,
}

impl PreprocessedQuery {
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }
}

/// BM25 scoring formula variant.
#[derive(Clone, Copy)]
pub enum ScoreFormula {
    Okapi,
    Plus { delta: f64 },
    L { delta: f64 },
}

/// Shared scoring parameters precomputed at index construction.
#[derive(Clone, Copy)]
pub struct ScoreParams {
    pub k1: f64,
    pub k1_plus1: f64,
    pub one_minus_b: f64,
    pub b_over_avgdl: f64,
}

impl ScoreParams {
    pub fn norm_factor(&self, dl: f64) -> f64 {
        self.k1 * (self.one_minus_b + self.b_over_avgdl * dl)
    }
}

/// Build a preprocessed query from string terms.
pub fn preprocess_query(
    query: &[String],
    interner: &StringInterner<DefaultBackend>,
    idf: &AHashMap<DefaultSymbol, f64>,
) -> PreprocessedQuery {
    let terms: SmallVec<[(DefaultSymbol, f64); 8]> = query
        .iter()
        .filter_map(|term| interner.get(term))
        .filter_map(|symbol| idf.get(&symbol).map(|&idf_val| (symbol, idf_val)))
        .collect();

    PreprocessedQuery { terms }
}

#[inline(always)]
pub fn term_score_okapi(freq: u32, dl: f64, params: ScoreParams, idf: f64) -> f64 {
    let freq_f64 = freq as f64;
    let norm_factor = params.norm_factor(dl);
    let numerator = freq_f64 * params.k1_plus1;
    let denominator = freq_f64 + norm_factor;
    idf * (numerator / denominator)
}

#[inline(always)]
pub fn term_score_plus(freq: u32, dl: f64, params: ScoreParams, idf: f64, delta: f64) -> f64 {
    let freq_f64 = freq as f64;
    let norm_factor = params.norm_factor(dl);
    let numerator = delta + freq_f64 * params.k1_plus1;
    let denominator = norm_factor + freq_f64;
    idf * (numerator / denominator)
}

#[inline(always)]
pub fn term_score_l(freq: u32, dl: f64, params: ScoreParams, idf: f64, delta: f64) -> f64 {
    let freq_f64 = freq as f64;
    let denominator = params.one_minus_b + params.b_over_avgdl * dl;
    let ctd = if denominator > 0.0 {
        freq_f64 / denominator
    } else {
        0.0
    };
    let numerator = params.k1_plus1 * (ctd + delta);
    let denom = params.k1 + ctd + delta;
    if denom > 0.0 {
        idf * numerator / denom
    } else {
        0.0
    }
}

#[inline(always)]
fn term_score(
    formula: ScoreFormula,
    freq: u32,
    dl: f64,
    params: ScoreParams,
    idf: f64,
) -> f64 {
    match formula {
        ScoreFormula::Okapi => term_score_okapi(freq, dl, params, idf),
        ScoreFormula::Plus { delta } => term_score_plus(freq, dl, params, idf, delta),
        ScoreFormula::L { delta } => term_score_l(freq, dl, params, idf, delta),
    }
}

/// SIMD-friendly batch accumulation for a posting list (4-wide unrolled).
#[inline]
fn accumulate_postings_simd(
    scores: &mut [f64],
    doc_len: &[u32],
    postings: &[(u32, u32)],
    idf: f64,
    params: ScoreParams,
    formula: ScoreFormula,
) {
    let mut i = 0;
    let len = postings.len();

    while i + 4 <= len {
        for j in 0..4 {
            let (doc_id, freq) = postings[i + j];
            let idx = doc_id as usize;
            let dl = doc_len[idx] as f64;
            scores[idx] += term_score(formula, freq, dl, params, idf);
        }
        i += 4;
    }

    while i < len {
        let (doc_id, freq) = postings[i];
        let idx = doc_id as usize;
        let dl = doc_len[idx] as f64;
        scores[idx] += term_score(formula, freq, dl, params, idf);
        i += 1;
    }
}

/// Score all documents using the inverted index (sparse query-friendly).
pub fn score_all_inverted(
    index: &IndexView<'_>,
    query: &PreprocessedQuery,
    params: ScoreParams,
    formula: ScoreFormula,
) -> Vec<f64> {
    if query.is_empty() {
        return vec![0.0; index.corpus_size];
    }

    let mut scores = vec![0.0; index.corpus_size];

    for &(symbol, idf) in &query.terms {
        if let Some(postings) = index.inverted.get(&symbol) {
            accumulate_postings_simd(
                &mut scores,
                index.doc_len,
                postings,
                idf,
                params,
                formula,
            );
        }
    }

    scores
}

/// Score all documents with parallel brute-force scan.
pub fn score_documents_parallel(
    doc_ids: &[usize],
    doc_freqs: &[AHashMap<DefaultSymbol, u32>],
    doc_len: &[u32],
    query: &PreprocessedQuery,
    params: ScoreParams,
    formula: ScoreFormula,
) -> Vec<f64> {
    if query.is_empty() {
        return vec![0.0; doc_ids.len()];
    }

    doc_ids
        .par_iter()
        .map(|&i| {
            let doc_freq = &doc_freqs[i];
            let dl = doc_len[i] as f64;

            query.terms.iter().fold(0.0, |score, &(symbol, idf_val)| {
                if let Some(&freq) = doc_freq.get(&symbol) {
                    score + term_score(formula, freq, dl, params, idf_val)
                } else {
                    score
                }
            })
        })
        .collect()
}

/// Top-k over candidate documents from inverted index posting unions.
pub fn top_k_inverted(
    index: &IndexView<'_>,
    query: &PreprocessedQuery,
    params: ScoreParams,
    formula: ScoreFormula,
    k: usize,
) -> Vec<(usize, f64)> {
    if k == 0 {
        return Vec::new();
    }

    if query.is_empty() {
        return (0..index.corpus_size.min(k))
            .map(|doc_id| (doc_id, 0.0))
            .collect();
    }

    let mut candidate_scores: AHashMap<u32, f64> = AHashMap::new();

    for &(symbol, idf) in &query.terms {
        if let Some(postings) = index.inverted.get(&symbol) {
            for &(doc_id, freq) in postings {
                let dl = index.doc_len[doc_id as usize] as f64;
                let contribution = term_score(formula, freq, dl, params, idf);
                *candidate_scores.entry(doc_id).or_insert(0.0) += contribution;
            }
        }
    }

    let mut results: Vec<(usize, f64)> = candidate_scores
        .into_iter()
        .map(|(doc_id, score)| (doc_id as usize, score))
        .collect();

    results.sort_by(|a, b| {
        b.1
            .partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    if results.len() < k {
        let mut included: AHashMap<usize, ()> =
            results.iter().map(|(doc_id, _)| (*doc_id, ())).collect();
        for doc_id in 0..index.corpus_size {
            if results.len() >= k {
                break;
            }
            if !included.contains_key(&doc_id) {
                results.push((doc_id, 0.0));
                included.insert(doc_id, ());
            }
        }
    }

    results.truncate(k);
    results
}

/// Batch scoring for multiple queries against the same index.
pub fn score_queries_batch(
    index: &IndexView<'_>,
    queries: &[PreprocessedQuery],
    params: ScoreParams,
    formula: ScoreFormula,
) -> Vec<Vec<f64>> {
    queries
        .par_iter()
        .map(|query| score_all_inverted(index, query, params, formula))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::build_index;
    use ahash::AHashMap;
    use smallvec::smallvec;
    use string_interner::{DefaultBackend, StringInterner};

    fn sample_index() -> (
        IndexView<'static>,
        StringInterner<DefaultBackend>,
        AHashMap<DefaultSymbol, f64>,
    ) {
        let corpus = vec![
            smallvec!["hello".to_string(), "world".to_string()],
            smallvec!["hello".to_string(), "rust".to_string()],
            smallvec!["world".to_string(), "bm25".to_string()],
        ];
        let mut interner = StringInterner::default();
        let built = build_index(corpus, &mut interner);
        let nd = built.nd;
        let idf = crate::index::calc_idf_okapi(nd, built.corpus_size, 0.25);

        let doc_freqs = built.doc_freqs;
        let doc_len = built.doc_len;
        let inverted = built.inverted;
        let corpus_size = built.corpus_size;
        let avgdl = built.avgdl;

        let view = IndexView {
            doc_freqs: Box::leak(doc_freqs.into_boxed_slice()),
            doc_len: Box::leak(doc_len.into_boxed_slice()),
            inverted: Box::leak(Box::new(inverted)),
            corpus_size,
            avgdl,
        };

        (view, interner, idf)
    }

    #[test]
    fn inverted_scores_match_brute_force() {
        let (view, interner, idf) = sample_index();
        let query = preprocess_query(
            &["hello".to_string(), "world".to_string()],
            &interner,
            &idf,
        );
        let params = ScoreParams {
            k1: 1.5,
            k1_plus1: 2.5,
            one_minus_b: 0.25,
            b_over_avgdl: 0.75 / view.avgdl,
        };

        let inverted_scores = score_all_inverted(&view, &query, params, ScoreFormula::Okapi);
        let brute_scores = score_documents_parallel(
            &(0..view.corpus_size).collect::<Vec<_>>(),
            view.doc_freqs,
            view.doc_len,
            &query,
            params,
            ScoreFormula::Okapi,
        );

        for (a, b) in inverted_scores.iter().zip(brute_scores.iter()) {
            assert!((a - b).abs() < 1e-12, "scores differ: {a} vs {b}");
        }
    }

    #[test]
    fn top_k_returns_highest_scores() {
        let (view, interner, idf) = sample_index();
        let query = preprocess_query(&["hello".to_string()], &interner, &idf);
        let params = ScoreParams {
            k1: 1.5,
            k1_plus1: 2.5,
            one_minus_b: 0.25,
            b_over_avgdl: 0.75 / view.avgdl,
        };

        let all_scores = score_all_inverted(&view, &query, params, ScoreFormula::Okapi);
        let top = top_k_inverted(&view, &query, params, ScoreFormula::Okapi, 2);

        assert_eq!(top.len(), 2);
        let mut sorted: Vec<_> = all_scores.iter().enumerate().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        assert_eq!(top[0].0, sorted[0].0);
        assert_eq!(top[1].0, sorted[1].0);
    }
}
