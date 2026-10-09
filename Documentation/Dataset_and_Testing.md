# Dataset and Testing Documentation

## 1. Introduction

This document describes the dataset, project organization, and testing approach used in the Movie Recommendation System project.

The project implements item-based collaborative filtering using C and compares a sequential implementation with an OpenMP parallel implementation.

## 2. Dataset Overview

The project uses the **MovieLens 1M dataset**, which contains movie ratings submitted by users.

The original MovieLens 1M dataset contains approximately:

* 1 million ratings
* 6,040 users
* 3,952 movies

Each rating record contains four fields:

| Field     | Description                                 |
| --------- | ------------------------------------------- |
| User ID   | Identifier for the user who rated the movie |
| Movie ID  | Identifier for the movie                    |
| Rating    | Rating given by the user                    |
| Timestamp | Time when the rating was recorded           |

The original ratings file uses the `::` delimiter between fields.

The project includes rating data files in its `Data` folder. The exact number of records processed during an experiment may differ from the full dataset because benchmark runs can use limits on users and movies.

## 3. Project Structure

The repository is organized into folders and files according to their purpose.

| File or folder                           | Purpose                                                                 |
| ---------------------------------------- | ----------------------------------------------------------------------- |
| `Code/`                                  | Contains the C implementations, build configuration, and Python scripts |
| `Code/recommender_seq.c`                 | Sequential recommendation implementation                                |
| `Code/recommender_omp.c`                 | OpenMP parallel recommendation implementation                           |
| `Code/common.h`                          | Shared definitions and declarations                                     |
| `Code/Makefile`                          | Build instructions for supported environments                           |
| `Code/run_benchmark.py`                  | Automates benchmark runs and compares outputs                           |
| `Code/make_scaled_dataset.py`            | Creates scaled datasets                                                 |
| `Code/make_synthetic.py`                 | Generates synthetic test data                                           |
| `Code/plot_results.py`                   | Generates performance plots from benchmark results                      |
| `Data/`                                  | Contains the rating dataset files                                       |
| `Output/`                                | Stores benchmark results, summaries, and plots                          |
| `Interpretation/Performance_Analysis.md` | Discusses performance measurements and scalability                      |
| `README.md`                              | Provides project overview and usage instructions                        |

## 4. Compilation

The sequential and parallel programs can be compiled using GCC. The exact commands depend on the compiler and operating system.

Example commands when working in a compatible GCC environment from the `Code` directory:

### Sequential version

```bash
gcc recommender_seq.c -o recommender_seq.exe -O2 -lm
