"""
BM25-RS: High-Performance BM25 for Python

A blazingly fast BM25 implementation in Rust with Python bindings.
"""

from ._bm25_rs import BM25Okapi, BM25Plus, BM25L, PyPreprocessedQuery

__version__ = "1.0.5"
__author__ = "BM25-RS Team"
__email__ = "amiya8mandal@gmail.com"

__all__ = [
    "BM25Okapi",
    "BM25Plus",
    "BM25L",
    "PyPreprocessedQuery",
]

# Convenience imports for common use cases
from .utils import tokenize_text, preprocess_corpus
from .benchmarks import benchmark_bm25

__all__.extend([
    "tokenize_text",
    "preprocess_corpus",
    "benchmark_bm25",
])
