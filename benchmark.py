#!/usr/bin/env python3
"""
Simple benchmark script to test BM25 performance improvements
"""

import time
import random
import string
from typing import List

def generate_random_document(word_count: int, vocabulary: List[str]) -> str:
    """Generate a random document with specified word count."""
    return ' '.join(random.choices(vocabulary, k=word_count))

def generate_corpus(num_documents: int, avg_word_count: int = 100) -> tuple:
    """Generate a corpus of random documents."""
    vocabulary_size = 10000
    vocabulary = [''.join(random.choices(string.ascii_lowercase, k=5)) 
                  for _ in range(vocabulary_size)]
    
    corpus = [generate_random_document(avg_word_count, vocabulary) 
              for _ in range(num_documents)]
    
    return corpus, vocabulary

def benchmark_bm25():
    """Benchmark BM25 implementations."""
    print("Generating test corpus...")
    corpus, vocabulary = generate_corpus(10000, 50)
    
    # Test queries
    queries = [' '.join(random.choices(vocabulary, k=3)) for _ in range(100)]
    
    try:
        # Try to import the optimized Rust version
        from bm25_rs import BM25Okapi as BM25Rust
        
        print("Testing optimized Rust BM25...")
        start_time = time.time()
        bm25_rust = BM25Rust(corpus)
        init_time = time.time() - start_time
        print(f"Rust BM25 initialization: {init_time:.3f}s")
        
        # Test scoring performance
        start_time = time.time()
        for query in queries:
            query_tokens = query.lower().split()
            scores = bm25_rust.get_scores(query_tokens)
        scoring_time = time.time() - start_time
        print(f"Rust BM25 scoring ({len(queries)} queries): {scoring_time:.3f}s")
        print(f"Rust BM25 queries per second: {len(queries)/scoring_time:.1f}")
        
    except ImportError:
        print("Rust BM25 not available - run 'maturin develop --release' first")
    
    try:
        # Compare with Python reference implementation
        from rank_bm25 import BM25Okapi as BM25Python
        
        print("\nTesting Python reference BM25...")
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        
        start_time = time.time()
        bm25_python = BM25Python(tokenized_corpus)
        init_time = time.time() - start_time
        print(f"Python BM25 initialization: {init_time:.3f}s")
        
        # Test scoring performance
        start_time = time.time()
        for query in queries:
            query_tokens = query.lower().split()
            scores = bm25_python.get_scores(query_tokens)
        scoring_time = time.time() - start_time
        print(f"Python BM25 scoring ({len(queries)} queries): {scoring_time:.3f}s")
        print(f"Python BM25 queries per second: {len(queries)/scoring_time:.1f}")
        
    except ImportError:
        print("Python BM25 reference not available - install with 'pip install rank-bm25'")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    benchmark_bm25()