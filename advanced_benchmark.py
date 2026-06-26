#!/usr/bin/env python3
"""
Advanced benchmark script to test and optimize BM25 performance
"""

import time
import random
import string
import gc
from typing import List, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_realistic_corpus(num_documents: int, avg_word_count: int = 100) -> Tuple[List[str], List[str]]:
    """Generate a more realistic corpus with varied document lengths."""
    # Create a vocabulary with realistic word frequency distribution (Zipf's law)
    vocabulary_size = 50000
    base_words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 12))) 
                  for _ in range(vocabulary_size)]
    
    # Create frequency weights following Zipf's law
    weights = [1.0 / (i + 1) for i in range(vocabulary_size)]
    
    corpus = []
    for _ in range(num_documents):
        # Vary document length realistically
        doc_length = max(10, int(random.normalvariate(avg_word_count, avg_word_count * 0.3)))
        doc_words = random.choices(base_words, weights=weights, k=doc_length)
        corpus.append(' '.join(doc_words))
    
    return corpus, base_words[:1000]  # Return smaller vocabulary for queries

def benchmark_initialization(corpus_sizes: List[int]):
    """Benchmark initialization performance across different corpus sizes."""
    print("=== Initialization Benchmark ===")
    
    for size in corpus_sizes:
        print(f"\nTesting corpus size: {size:,} documents")
        corpus, _ = generate_realistic_corpus(size, 50)
        
        # Test Rust BM25Okapi
        try:
            from bm25_rs import BM25Okapi as BM25Rust
            
            gc.collect()  # Clean memory before test
            start_time = time.time()
            bm25_rust = BM25Rust(corpus)
            init_time = time.time() - start_time
            
            print(f"  Rust BM25Okapi: {init_time:.3f}s ({size/init_time:.0f} docs/sec)")
            
            # Memory usage estimation
            import sys
            memory_mb = sys.getsizeof(bm25_rust) / (1024 * 1024)
            print(f"  Memory usage: ~{memory_mb:.1f} MB")
            
        except ImportError:
            print("  Rust BM25 not available")

def benchmark_query_performance(corpus_size: int = 10000, num_queries: int = 1000):
    """Benchmark query performance with different query patterns."""
    print(f"\n=== Query Performance Benchmark ===")
    print(f"Corpus: {corpus_size:,} documents, {num_queries} queries")
    
    corpus, vocabulary = generate_realistic_corpus(corpus_size, 100)
    
    try:
        from bm25_rs import BM25Okapi as BM25Rust
        
        # Initialize
        print("Initializing BM25...")
        start_time = time.time()
        bm25_rust = BM25Rust(corpus)
        init_time = time.time() - start_time
        print(f"Initialization: {init_time:.3f}s")
        
        # Test different query types
        query_types = {
            "Short queries (1-2 terms)": [' '.join(random.choices(vocabulary, k=random.randint(1, 2))) 
                                         for _ in range(num_queries)],
            "Medium queries (3-5 terms)": [' '.join(random.choices(vocabulary, k=random.randint(3, 5))) 
                                          for _ in range(num_queries)],
            "Long queries (6-10 terms)": [' '.join(random.choices(vocabulary, k=random.randint(6, 10))) 
                                         for _ in range(num_queries)]
        }
        
        for query_type, queries in query_types.items():
            print(f"\n{query_type}:")
            
            # Warm up
            for _ in range(10):
                query_tokens = queries[0].lower().split()
                bm25_rust.get_scores(query_tokens)
            
            # Benchmark
            start_time = time.time()
            for query in queries:
                query_tokens = query.lower().split()
                scores = bm25_rust.get_scores(query_tokens)
            
            total_time = time.time() - start_time
            qps = len(queries) / total_time
            
            print(f"  Time: {total_time:.3f}s")
            print(f"  QPS: {qps:.1f}")
            print(f"  Avg latency: {total_time/len(queries)*1000:.2f}ms")
            
    except ImportError:
        print("Rust BM25 not available")

def benchmark_concurrent_queries(corpus_size: int = 10000, num_threads: int = 4, queries_per_thread: int = 250):
    """Benchmark concurrent query performance."""
    print(f"\n=== Concurrent Query Benchmark ===")
    print(f"Corpus: {corpus_size:,} documents")
    print(f"Threads: {num_threads}, Queries per thread: {queries_per_thread}")
    
    corpus, vocabulary = generate_realistic_corpus(corpus_size, 100)
    
    try:
        from bm25_rs import BM25Okapi as BM25Rust
        
        bm25_rust = BM25Rust(corpus)
        
        # Generate queries for all threads
        all_queries = [' '.join(random.choices(vocabulary, k=random.randint(2, 5))) 
                      for _ in range(num_threads * queries_per_thread)]
        
        def worker_function(thread_queries):
            """Worker function for concurrent testing."""
            results = []
            for query in thread_queries:
                query_tokens = query.lower().split()
                scores = bm25_rust.get_scores(query_tokens)
                results.append(len(scores))
            return results
        
        # Split queries among threads
        queries_per_worker = len(all_queries) // num_threads
        query_chunks = [all_queries[i:i + queries_per_worker] 
                       for i in range(0, len(all_queries), queries_per_worker)]
        
        # Run concurrent benchmark
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, chunk) for chunk in query_chunks]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        total_queries = sum(len(chunk) for chunk in query_chunks)
        concurrent_qps = total_queries / total_time
        
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Total queries: {total_queries}")
        print(f"  Concurrent QPS: {concurrent_qps:.1f}")
        print(f"  Speedup vs single thread: {concurrent_qps / (total_queries / (total_time * num_threads)):.1f}x")
        
    except ImportError:
        print("Rust BM25 not available")

def benchmark_batch_operations(corpus_size: int = 10000):
    """Benchmark batch scoring operations."""
    print(f"\n=== Batch Operations Benchmark ===")
    
    corpus, vocabulary = generate_realistic_corpus(corpus_size, 100)
    
    try:
        from bm25_rs import BM25Okapi as BM25Rust
        
        bm25_rust = BM25Rust(corpus)
        query = ' '.join(random.choices(vocabulary, k=3))
        query_tokens = query.lower().split()
        
        # Test different batch sizes
        batch_sizes = [10, 100, 1000, min(5000, corpus_size)]
        
        for batch_size in batch_sizes:
            doc_ids = random.sample(range(corpus_size), min(batch_size, corpus_size))
            
            # Warm up
            for _ in range(5):
                bm25_rust.get_batch_scores(query_tokens, doc_ids)
            
            # Benchmark
            start_time = time.time()
            for _ in range(100):
                scores = bm25_rust.get_batch_scores(query_tokens, doc_ids)
            
            total_time = time.time() - start_time
            ops_per_sec = 100 / total_time
            
            print(f"  Batch size {batch_size:4d}: {ops_per_sec:6.1f} ops/sec, {total_time/100*1000:.2f}ms avg")
            
    except ImportError:
        print("Rust BM25 not available")

def profile_memory_usage():
    """Profile memory usage patterns."""
    print(f"\n=== Memory Usage Profile ===")
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        corpus_sizes = [1000, 5000, 10000, 25000]
        
        for size in corpus_sizes:
            # Measure before
            gc.collect()
            mem_before = process.memory_info().rss / (1024 * 1024)
            
            # Create BM25 instance
            corpus, _ = generate_realistic_corpus(size, 100)
            
            try:
                from bm25_rs import BM25Okapi as BM25Rust
                bm25_rust = BM25Rust(corpus)
                
                # Measure after
                mem_after = process.memory_info().rss / (1024 * 1024)
                mem_used = mem_after - mem_before
                mem_per_doc = mem_used / size * 1024  # KB per document
                
                print(f"  {size:5d} docs: {mem_used:6.1f} MB total, {mem_per_doc:.2f} KB/doc")
                
                # Clean up
                del bm25_rust
                del corpus
                gc.collect()
                
            except ImportError:
                print("Rust BM25 not available")
                break
                
    except ImportError:
        print("psutil not available for memory profiling")

def main():
    """Run comprehensive benchmarks."""
    print("BM25 Advanced Performance Benchmark")
    print("=" * 50)
    
    random.seed(42)  # For reproducible results
    
    # Run benchmarks
    benchmark_initialization([1000, 5000, 10000, 25000])
    benchmark_query_performance(10000, 1000)
    benchmark_concurrent_queries(10000, 4, 250)
    benchmark_batch_operations(10000)
    profile_memory_usage()
    
    print("\n" + "=" * 50)
    print("Benchmark completed!")

if __name__ == "__main__":
    main()