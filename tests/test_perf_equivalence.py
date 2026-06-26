"""
Performance and API equivalence tests for optimized BM25 paths.
"""

import concurrent.futures
import random
import string

import pytest

try:
    from bm25_rs import BM25L, BM25Okapi, BM25Plus, PyPreprocessedQuery
except ImportError:
    pytest.skip("BM25-RS not available", allow_module_level=True)


def _make_corpus(n_docs: int = 200, vocab_size: int = 500) -> list[str]:
    random.seed(42)
    vocab = [
        "".join(random.choices(string.ascii_lowercase, k=5))
        for _ in range(vocab_size)
    ]
    return [
        " ".join(random.choices(vocab, k=20))
        for _ in range(n_docs)
    ]


@pytest.fixture(scope="module")
def large_corpus():
    return _make_corpus()


@pytest.fixture(scope="module")
def bm25(large_corpus):
    return BM25Okapi(large_corpus)


class TestScoringEquivalence:
    def test_chunked_matches_scores(self, bm25):
        query = ["term", "search", "document"]
        scores = bm25.get_scores(query)
        chunked = bm25.get_scores_chunked(query, chunk_size=37)
        assert len(chunked) == len(scores)
        for a, b in zip(scores, chunked):
            assert abs(a - b) < 1e-10

    def test_batch_matches_scores(self, bm25, large_corpus):
        query = ["term", "search"]
        doc_ids = list(range(0, len(large_corpus), 7))
        batch = bm25.get_batch_scores(query, doc_ids)
        all_scores = bm25.get_scores(query)
        for i, doc_id in enumerate(doc_ids):
            assert abs(batch[i] - all_scores[doc_id]) < 1e-10

    def test_top_n_indices_agrees_with_scores(self, bm25, large_corpus):
        query = ["gtwcl", "ytfnn", "ptreu"]
        scores = bm25.get_scores(query)
        top_indices = bm25.get_top_n_indices(query, n=5)
        top_docs = bm25.get_top_n(query, large_corpus, n=5)

        assert len(top_indices) == 5
        assert len(top_docs) == 5

        for (idx, score), (doc, doc_score) in zip(top_indices, top_docs):
            assert doc == large_corpus[idx]
            assert abs(score - doc_score) < 1e-10
            assert abs(score - scores[idx]) < 1e-10

    def test_preprocessed_query_matches_scores(self, bm25):
        query = ["gtwcl", "ytfnn"]
        preprocessed = bm25.preprocess_query(query)
        assert isinstance(preprocessed, PyPreprocessedQuery)
        assert len(preprocessed) > 0

        direct = bm25.get_scores(query)
        via_preprocessed = bm25.get_scores_with_preprocessed(preprocessed)
        assert direct == via_preprocessed

    def test_batch_queries_matches_individual(self, bm25):
        queries = [["term", "search"], ["document", "query"], ["random", "words"]]
        batch = bm25.get_scores_batch(queries)
        assert len(batch) == len(queries)
        for q, scores in zip(queries, batch):
            assert scores == bm25.get_scores(q)


class TestTokenizedCorpus:
    def test_from_tokenized_corpus(self, large_corpus):
        tokenized = [doc.lower().split() for doc in large_corpus[:50]]
        bm25 = BM25Okapi(tokenized_corpus=tokenized)
        assert bm25.corpus_size == 50

        query = ["term", "search"]
        scores = bm25.get_scores(query)
        assert len(scores) == 50


class TestVariantParity:
    @pytest.mark.parametrize("cls", [BM25Okapi, BM25Plus, BM25L])
    def test_top_n_indices_available(self, cls, large_corpus):
        bm25 = cls(large_corpus[:30])
        query = ["gtwcl", "ytfnn"]
        top = bm25.get_top_n_indices(query, n=3)
        assert len(top) == 3
        assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in top)

    @pytest.mark.parametrize("cls", [BM25Okapi, BM25Plus, BM25L])
    def test_chunked_available(self, cls, large_corpus):
        bm25 = cls(large_corpus[:30])
        query = ["term"]
        scores = bm25.get_scores(query)
        chunked = bm25.get_scores_chunked(query)
        assert scores == chunked


class TestNumpyOutput:
    def test_get_scores_numpy(self, bm25):
        pytest.importorskip("numpy")
        import numpy as np

        query = ["term", "search"]
        arr = bm25.get_scores_numpy(query)
        scores = bm25.get_scores(query)

        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert arr.shape == (bm25.corpus_size,)
        assert np.allclose(arr, scores)


class TestConcurrency:
    def test_concurrent_queries(self, bm25):
        queries = [["term", "search"], ["document", "query"], ["random", "token"]] * 4

        def run_query(q):
            return sum(bm25.get_scores(q))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run_query, queries))

        expected = [sum(bm25.get_scores(q)) for q in queries]
        assert results == expected


@pytest.mark.benchmark(group="scoring")
def test_benchmark_get_scores(benchmark, bm25):
    query = ["term", "search", "document"]
    benchmark(bm25.get_scores, query)


@pytest.mark.benchmark(group="topk")
def test_benchmark_top_n_indices(benchmark, bm25):
    query = ["term", "search", "document"]
    benchmark(bm25.get_top_n_indices, query, n=10)
