# Parallel Recommendation System (MovieLens 1M + OpenMP)

Item-based collaborative filtering (cosine similarity), sequential vs OpenMP.

## 1. Get the dataset
Download https://files.grouplens.org/datasets/movielens/ml-1m.zip and unzip. You need `ml-1m/ratings.dat`
(format `UserID::MovieID::Rating::Timestamp`, 1,000,209 ratings, 6,040 users, movie ids up to 3,952).

## 2. Build  (Linux / WSL / MinGW with gcc)
    make

## 3. Run once
    ./recommender_seq ml-1m/ratings.dat 6040 3952 seq.txt
    ./recommender_omp ml-1m/ratings.dat 6040 3952 8 omp.txt
    cmp seq.txt omp.txt && echo IDENTICAL

## 4. Full experiment (tables + graphs)
    python3 run_benchmark.py --data ml-1m/ratings.dat      # -> results.csv, verifies outputs
    python3 plot_results.py                                 # -> summary.csv/.md + 5 PNG graphs
(needs `pip install pandas matplotlib tabulate`)

## 5. Larger-than-original datasets (optional, for the "scalability" section)
    python3 make_scaled_dataset.py ml-1m/ratings.dat ratings_x4.dat 4     # 24,160 users
    python3 run_benchmark.py --data ratings_x4.dat --users 6040 12080 24160 --items 3952 --reps 3
