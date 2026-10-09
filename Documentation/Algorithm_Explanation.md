# Algorithm Explanation: Item-Based Collaborative Filtering

## 1. Introduction

The Movie Recommendation System recommends movies to users based on patterns in a user-movie rating dataset. It uses item-based collaborative filtering to identify movies that have similar rating patterns and recommends movies based on a user's existing preferences.

The project uses the MovieLens 1M dataset and implements the algorithm in C. It includes a sequential implementation and an OpenMP parallel implementation to study performance.

## 2. Dataset Overview

The dataset contains movie ratings provided by users. Each rating record contains four fields:

- **User ID:** Identifies the user who rated a movie.
- **Movie ID:** Identifies the movie.
- **Rating:** The user's rating of the movie.
- **Timestamp:** The time when the rating was recorded.

The algorithm uses user IDs, movie IDs, and ratings to calculate movie similarities and generate recommendations. The timestamp is not required for the basic similarity calculation.

## 3. What Is Item-Based Collaborative Filtering?

Item-based collaborative filtering recommends items by comparing their rating patterns. In this project, the items are movies.

For example, if users who rate Movie A highly also tend to rate Movie B highly, the two movies may have similar rating patterns. If a user liked Movie A but has not watched Movie B, Movie B may be a suitable recommendation.

The method uses ratings from multiple users rather than relying on movie descriptions or genres.

## 4. Representing the Data

The ratings can be viewed as a user-item matrix:

- Each row represents a user.
- Each column represents a movie.
- Each cell contains the rating given by a user to a movie, when available.

For example:

| User   | Movie A | Movie B | Movie C |
| ------ | ------: | ------: | ------: |
| User 1 |       5 |       4 |       — |
| User 2 |       4 |       5 |       2 |
| User 3 |       1 |       — |       5 |

A dash indicates that the user has not rated that movie.

The implementation processes rating records and uses the available rating information to calculate similarities. The matrix above is only an illustrative example.

## 5. Cosine Similarity

Cosine similarity measures how similar two movies' rating patterns are across users who rated both movies.

For movies \(i\) and \(j\), the cosine similarity is:

$$\operatorname{sim}(i,j)= \frac{\sum_{u} r_{u,i}r_{u,j}} {\sqrt{\sum_{u}r_{u,i}^{2}}\sqrt{\sum_{u}r_{u,j}^{2}}}$$

Here:

- \(r_{u,i}\) is the rating given by user \(u\) to movie \(i\).
- \(r_{u,j}\) is the rating given by user \(u\) to movie \(j\).
- The sums use the rating values included by the implementation for the movie pair.

A larger cosine similarity generally indicates more similar rating patterns. The exact treatment of missing ratings depends on the implementation.

## 6. Recommendation Generation

The general recommendation process is:

1. Load the rating records from the dataset.
2. Process the available ratings for the selected users and movies.
3. Calculate similarities between movies using cosine similarity.
4. Identify movies related to the movies a user has rated.
5. Use the similarity information to generate recommendation scores.
6. Select suitable candidate movies and write the recommendations to the output file.

The precise score calculation, candidate filtering, and output format should be checked against the project's C source code before describing them as implementation details.

## 7. Sequential Implementation

The sequential program, `recommender_seq.c`, executes the recommendation computation using a single thread.

It provides a baseline for measuring execution time and checking the results of the parallel version.

## 8. OpenMP Parallel Implementation

The parallel program, `recommender_omp.c`, uses OpenMP to execute supported parts of the computation across multiple threads.

Parallel execution may reduce runtime when work can be divided among threads. However, the improvement depends on the amount of work, thread overhead, available CPU resources, and how much of the computation can be parallelized.

## 9. Correctness and Output Validation

The sequential and parallel implementations can be tested using the same dataset and configuration. Their recommendation outputs should then be compared.

A byte-for-byte comparison can establish whether the generated files match for a particular test configuration. It does not, by itself, prove correctness for every possible dataset or input size.

## 10. Performance Considerations

The computation cost can increase as the number of users, movies, and ratings grows. Calculating similarities for many movie pairs can be expensive, especially as the number of movies increases.

The project benchmarks the sequential and OpenMP implementations at different input sizes and thread counts. Measured results should be used to discuss execution time, speedup, and parallel efficiency.

## 11. Conclusion

Item-based collaborative filtering uses patterns in historical user ratings to recommend movies. Cosine similarity provides a way to compare movie rating patterns, while OpenMP allows parts of the computation to run in parallel.

Comparing sequential and parallel execution helps evaluate whether parallel processing improves performance as the dataset grows.
