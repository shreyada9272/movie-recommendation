# Movie Recommendation System — Sequential C and OpenMP

A movie recommendation system built using **item-based collaborative filtering** and **cosine similarity**. This project compares a sequential C implementation with a parallel C implementation using OpenMP and evaluates performance as the workload increases.

## Project Objectives

* Generate movie recommendations from user-rating data.
* Compare sequential and OpenMP parallel implementations.
* Measure execution time, speedup, and parallel efficiency.
* Study scalability as the number of users and items increases.

## Dataset

This project uses the **MovieLens 1M** ratings dataset.

The original dataset can be downloaded from:
https://files.grouplens.org/datasets/movielens/ml-1m.zip

Extract the archive and locate `ratings.dat`. The expected format is:

`UserID::MovieID::Rating::Timestamp`

Place the ratings file in the project's `Data/` directory. The repository may also contain scaled datasets for experiments.

## Repository Structure

* `Code/` — C implementations, shared header, Makefile, and Python scripts for benchmarking, plotting, and dataset generation.
* `Data/` — Movie ratings datasets used for experiments.
* `Output/` — Benchmark results, summary files, and performance graphs.
* `Interpretation/` — Written analysis of the performance results.
* `README.md` — Project overview and usage instructions.

## Requirements

* GCC compiler
* OpenMP support in GCC for the parallel implementation
* Python 3
* Python packages required by the plotting script, such as `pandas`, `matplotlib`, and `tabulate`

## Build

Run the following commands from the repository root:

```bash
cd Code
make
```

This builds the sequential and OpenMP programs using the rules defined in the Makefile.

## Run the Programs

Run the commands from the `Code/` directory. Replace the dataset path with the actual path to your ratings file.

Example:

```bash
./recommender_seq ../Data/ratings.dat 6040 3952 seq.txt
./recommender_omp ../Data/ratings.dat 6040 3952 8 omp.txt
```

The commands assume the compiled executables are named `recommender_seq` and `recommender_omp`, and that the program arguments match the current implementations.

If you want to compare output files, check that the programs use the same input and settings before comparing them.

## Run Benchmarks

From the `Code/` directory, inspect the available command-line options with:

```bash
python3 run_benchmark.py --help
```

Then run the benchmark script using the dataset path expected by the script. For example, if it accepts a `--data` argument:

```bash
python3 run_benchmark.py --data ../Data/ratings.dat
```

The benchmark script records performance measurements in a CSV file. Check the script's output path to find the generated results.

## Generate Performance Graphs

From the `Code/` directory, run:

```bash
python3 plot_results.py
```

The plotting script reads benchmark results and generates graphs and summary files. Make sure its input CSV is in the location expected by the script.

## Scalability Experiments

The project includes scripts for generating synthetic or scaled datasets. Check their help messages and source code for the exact arguments:

```bash
python3 make_scaled_dataset.py --help
python3 make_synthetic.py --help
```

Use these scripts to prepare larger workloads and compare execution time as the number of users or items increases.

## Results and Analysis

The `Output/` directory contains project results and performance graphs. The `Interpretation/Performance_Analysis.md` file discusses the observed performance.

## Team Contributions

Team members should document their actual contributions through their own Git branches and commits. Add names and contribution details here only after confirming who worked on each part.

## Notes

* Run commands from the directory specified in each section.
* Keep generated results and dataset paths consistent across experiments.
* Report performance conclusions based on measured results rather than assumptions.
