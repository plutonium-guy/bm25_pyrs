#!/usr/bin/env python3
"""
Performance comparison between different BM25 optimization approaches
"""

import time
import random
import string
from typing import List, Tuple

def generate_test_data(num_docs: int = 10000, avg_words: int = 100) -> Tuple[List[str], List[str]]:
    """Generate test corpus and queries."""
    vocabulary = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) 
                  for _ in range(10000)]
    
    corpus = []
    for _ in range(num_docs):
        doc_length = max(10, int(random.normalvariate(avg_words, avg_words * 0.3)))
        words = random.choices(vocabulary, k=doc_length)
        corpus.append(' '.join(words))
    
    queries = [' '.join(random.choices(vocabulary[:1000], k=random.randint(2, 5))) 
               for _ in range(100)]
    
    return corpus, queries

def benchmark_method(method_name: str, method_func, *args, **kwargs):
    """Benchmark a method and return timing results."""
    # Warm up
    for _ in range(3):
        try:
            method_func(*args, **kwargs)
        except:
            pass
    
    # Actual benchmark
    start_time = time.time()
    result = method_func(*args, **kwargs)
    end_time = time.time()
    
    return end_time - start_time, result

def compare_scoring_methods():
    """Compare different scoring method implementations."""
    print("=== Scoring Methods Comparison ===")
    
    corpus, queries = generate_test_data(10000, 100)
    
    try:
        from bm25_rs import BM25Okapi
        
        bm25 = BM25Okapi(corpus)
        test_query = queries[0].lower().split()
        
        print(f"Testing with {len(corpus):,} documents and query: '{' '.join(test_query[:3])}...'")
        
        # Test standard get_scores
        time_standard, scores_standard = benchmark_method(
            "Standard get_scores", 
            bm25.get_scores, 
            test_query
        )
        
        # Test chunked scoring if available
        try:
            time_chunked, scores_chunked = benchmark_method(
                "Chunked get_scores", 
                bm25.get_scores_chunked, 
                test_query
            )
            
            print(f"Standard scoring: {time_standard*1000:.2f}ms")
            print(f"Chunked scoring:  {time_chunked*1000:.2f}ms")
            print(f"Speedup: {time_standard/time_chunked:.2f}x")
            
            # Verify results are the same
            if len(scores_standard) == len(scores_chunked):
                max_diff = max(abs(a - b) for a, b in zip(scores_standard, scores_chunked))
                print(f"Max difference: {max_diff:.10f}")
            
        except AttributeError:
            print("Chunked scoring not available")
            print(f"Standard scoring: {time_standard*1000:.2f}ms")
        
    except ImportError:
        print("BM25 implementation not available")

def compare_top_n_methods():
    """Compare different top-N retrieval methods."""
    print("\n=== Top-N Methods Comparison ===")
    
    corpus, queries = generate_test_data(10000, 100)
    
    try:
        from bm25_rs import BM25Okapi
        
        bm25 = BM25Okapi(corpus)
        test_query = queries[0].lower().split()
        
        print(f"Testing top-10 retrieval with {len(corpus):,} documents")
        
        # Test standard get_top_n
        time_standard, top_n_standard = benchmark_method(
            "Standard get_top_n",
            bm25.get_top_n,
            test_query,
            corpus,
            10
        )
        
        # Test optimized get_top_n_indices if available
        try:
            time_indices, top_n_indices = benchmark_method(
                "Optimized get_top_n_indices",
                bm25.get_top_n_indices,
                test_query,
                10
            )
            
            print(f"Standard top-N:   {time_standard*1000:.2f}ms")
            print(f"Indices-only:     {time_indices*1000:.2f}ms")
            print(f"Speedup: {time_standard/time_indices:.2f}x")
            
            # Verify results match
            if len(top_n_standard) == len(top_n_indices):
                print("Results verification: ", end="")
                matches = sum(1 for i, (doc, score1) in enumerate(top_n_standard)
                            if abs(score1 - top_n_indices[i][1]) < 1e-10)
                print(f"{matches}/{len(top_n_standard)} scores match")
            
        except AttributeError:
            print("Optimized top-N not available")
            print(f"Standard top-N: {time_standard*1000:.2f}ms")
        
    except ImportError:
        print("BM25 implementation not available")

def benchmark_batch_vs_individual():
    """Compare batch operations vs individual queries."""
    print("\n=== Batch vs Individual Queries ===")
    
    corpus, queries = generate_test_data(5000, 100)
    
    try:
        from bm25_rs import BM25Okapi
        
        bm25 = BM25Okapi(corpus)
        test_queries = [q.lower().split() for q in queries[:50]]
        
        print(f"Testing {len(test_queries)} queries on {len(corpus):,} documents")
        
        # Individual queries
        start_time = time.time()
        individual_results = []
        for query in test_queries:
            scores = bm25.get_scores(query)
            individual_results.append(scores)
        individual_time = time.time() - start_time
        
        # Batch document scoring (if available)
        doc_ids = list(range(0, len(corpus), 10))  # Sample every 10th document
        
        start_time = time.time()
        batch_results = []
        for query in test_queries:
            scores = bm25.get_batch_scores(query, doc_ids)
            batch_results.append(scores)
        batch_time = time.time() - start_time
        
        print(f"Individual queries: {individual_time*1000:.2f}ms total, {individual_time/len(test_queries)*1000:.2f}ms avg")
        print(f"Batch operations:   {batch_time*1000:.2f}ms total, {batch_time/len(test_queries)*1000:.2f}ms avg")
        print(f"Batch efficiency: {individual_time/batch_time:.2f}x faster per query")
        
    except ImportError:
        print("BM25 implementation not available")

def memory_efficiency_test():
    """Test memory efficiency with different corpus sizes."""
    print("\n=== Memory Efficiency Test ===")
    
    try:
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        
        corpus_sizes = [1000, 5000, 10000, 20000]
        
        for size in corpus_sizes:
            gc.collect()
            mem_before = process.memory_info().rss / (1024 * 1024)
            
            corpus, _ = generate_test_data(size, 50)
            
            try:
                from bm25_rs import BM25Okapi
                
                start_time = time.time()
                bm25 = BM25Okapi(corpus)
                init_time = time.time() - start_time
                
                mem_after = process.memory_info().rss / (1024 * 1024)
                mem_used = mem_after - mem_before
                
                print(f"{size:5d} docs: {init_time*1000:6.1f}ms init, {mem_used:6.1f}MB, {mem_used/size*1024:.2f}KB/doc")
                
                del bm25
                del corpus
                gc.collect()
                
            except ImportError:
                print("BM25 implementation not available")
                break
                
    except ImportError:
        print("psutil not available for memory testing")

def main():
    """Run performance comparisons."""
    print("BM25 Performance Optimization Comparison")
    print("=" * 50)
    
    random.seed(42)
    
    compare_scoring_methods()
    compare_top_n_methods()
    benchmark_batch_vs_individual()
    memory_efficiency_test()
    
    print("\n" + "=" * 50)
    print("Performance comparison completed!")

if __name__ == "__main__":
    main()