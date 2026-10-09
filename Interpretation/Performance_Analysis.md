# Performance Analysis – Recommendation Network

## 1. Project Overview

The project implements an item-based collaborative filtering recommendation system using cosine similarity. Sequential C and OpenMP C implementations are compared using the MovieLens dataset.

## 2. Performance Metrics

* **Execution Time:** Time taken by the program to complete its computation.
* **Speedup:** Sequential execution time divided by parallel execution time.
* **Efficiency:** Speedup divided by the number of threads, multiplied by 100.
* **Checksum:** Used to compare recommendation outputs across implementations.

## 3. Parallel Performance

OpenMP parallelizes similarity calculations and recommendation generation. Multiple threads can execute independent tasks concurrently, potentially reducing execution time. Actual performance depends on thread count, hardware, and workload.

## 4. Graph Interpretation

* **Execution Time vs Threads:** Shows how execution time changes with the number of threads.
* **Speedup vs Threads:** Shows the improvement compared with sequential execution.
* **Efficiency vs Threads:** Shows how effectively the available threads are utilized.
* **Time vs Users:** Shows how execution time changes as the dataset size increases.

The graphs should be interpreted using the measurements recorded in `Output/results.csv` and `Output/summary.csv`.

## 5. Limitations

* The dense item-similarity matrix requires substantial memory and computation.
* Performance depends on hardware and compiler configuration.
* The system currently uses cosine similarity and implements sequential and OpenMP versions.
* Recommendation quality metrics such as Precision, Recall, and NDCG are not yet evaluated.

## 6. Future Scope

* Use sparse data structures to reduce memory usage.
* Evaluate additional parallel approaches such as MPI and CUDA.
* Test more dataset sizes and thread configurations.
* Evaluate recommendation quality using Precision, Recall, and NDCG.

## 7. Conclusion

The project compares sequential and OpenMP implementations of a recommendation network. Execution time, speedup, and efficiency help evaluate the benefits and limitations of parallel execution.
