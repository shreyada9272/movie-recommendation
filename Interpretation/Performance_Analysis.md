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

## 8. Benchmarking Methodology

The sequential and OpenMP implementations are tested using the same MovieLens dataset and corresponding user and movie limits. The `run_benchmark.py` script runs the programs with selected thread counts and records execution times. Repeated runs help reduce the effect of timing fluctuations. The input configuration should remain consistent for a fair comparison.

## 9. Execution Time Analysis

Execution time measures how long each implementation takes to complete its computation. The results should be collected from the project's benchmark output files. System load, CPU scheduling, compiler settings, and workload size may affect the measurements. Therefore, repeated measurements should be used wherever possible.

## 10. Speedup Analysis

Speedup indicates the performance improvement of the parallel implementation compared with the sequential implementation.

$$
\text{Speedup}=\frac{T_{\text{seq}}}{T_{\text{parallel}}}
$$

A speedup greater than 1 indicates that the parallel implementation is faster for the tested configuration. A value near 1 indicates little improvement, while a value below 1 indicates slower parallel execution.

## 11. Parallel Efficiency Analysis

Parallel efficiency measures how effectively the OpenMP threads contribute to performance.

$$
\text{Efficiency}=\frac{\text{Speedup}}{p}\times100\%
$$

Here, \(p\) represents the number of threads used. Efficiency may decrease as the thread count increases due to synchronization overhead, scheduling costs, serial operations, and limited CPU resources.

## 12. Scalability Analysis

Scalability describes how performance changes as the workload increases. By varying the number of users and movies, the execution times of the sequential and parallel implementations can be compared across different input sizes. Larger workloads may benefit more from parallelization, but conclusions must be based on actual benchmark measurements.

## 13. Experimental Limitations

The results depend on the hardware, compiler, runtime environment, dataset, input limits, and thread count. Small workloads may produce unstable timing measurements, and increasing the number of threads does not guarantee faster execution. Performance conclusions should therefore be based on measured results rather than assumptions.
