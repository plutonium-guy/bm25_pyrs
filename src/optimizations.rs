// Additional optimization utilities for BM25 implementations

use rayon::prelude::*;
use string_interner::DefaultSymbol;

/// Partial top-k selection using nth_element-style partial sort.
pub fn select_top_k_indices(scores: &[f64], k: usize) -> Vec<usize> {
    if k >= scores.len() {
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.par_sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        return indices;
    }

    let mut indexed_scores: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();

    indexed_scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed_scores[..k].sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    indexed_scores[..k].iter().map(|(i, _)| *i).collect()
}

/// Flat parallel document scoring (avoids nested rayon parallelism).
pub fn score_documents_flat<F>(
    corpus_size: usize,
    chunk_size: usize,
    scorer: F,
) -> Vec<f64>
where
    F: Fn(usize) -> f64 + Sync,
{
    if corpus_size == 0 {
        return Vec::new();
    }

    let effective_chunk = chunk_size.max(1);
    let chunk_starts: Vec<usize> = (0..corpus_size).step_by(effective_chunk).collect();

    let chunk_scores: Vec<Vec<f64>> = chunk_starts
        .into_par_iter()
        .map(|start| {
            let end = (start + effective_chunk).min(corpus_size);
            (start..end).map(|i| scorer(i)).collect()
        })
        .collect();

    chunk_scores.into_iter().flatten().collect()
}

/// Legacy alias kept for compatibility with existing chunked API tests.
#[inline(always)]
pub fn compute_bm25_score_vectorized(
    query_terms: &[(DefaultSymbol, f64)],
    doc_freq: &ahash::AHashMap<DefaultSymbol, u32>,
    _dl: f64,
    norm_factor: f64,
    k1_plus1: f64,
) -> f64 {
    match query_terms.len() {
        0 => 0.0,
        1 => {
            let (symbol, idf_val) = query_terms[0];
            if let Some(&freq) = doc_freq.get(&symbol) {
                let freq_f64 = freq as f64;
                let numerator = freq_f64 * k1_plus1;
                let denominator = freq_f64 + norm_factor;
                idf_val * (numerator / denominator)
            } else {
                0.0
            }
        }
        2 => {
            let mut score = 0.0;
            for &(symbol, idf_val) in query_terms {
                if let Some(&freq) = doc_freq.get(&symbol) {
                    let freq_f64 = freq as f64;
                    let numerator = freq_f64 * k1_plus1;
                    let denominator = freq_f64 + norm_factor;
                    score += idf_val * (numerator / denominator);
                }
            }
            score
        }
        _ => query_terms.iter().fold(0.0, |score, &(symbol, idf_val)| {
            if let Some(&freq) = doc_freq.get(&symbol) {
                let freq_f64 = freq as f64;
                let numerator = freq_f64 * k1_plus1;
                let denominator = freq_f64 + norm_factor;
                score + idf_val * (numerator / denominator)
            } else {
                score
            }
        }),
    }
}

/// Chunked parallel scoring without nested rayon pools.
pub fn process_documents_in_chunks<F>(
    total_docs: usize,
    chunk_size: usize,
    processor: F,
) -> Vec<f64>
where
    F: Fn(usize, usize) -> Vec<f64> + Send + Sync,
{
    if total_docs == 0 {
        return Vec::new();
    }

    let effective_chunk = chunk_size.max(1);
    let chunk_starts: Vec<usize> = (0..total_docs).step_by(effective_chunk).collect();

    chunk_starts
        .into_par_iter()
        .flat_map(|start| {
            let end = (start + effective_chunk).min(total_docs);
            processor(start, end)
        })
        .collect()
}
