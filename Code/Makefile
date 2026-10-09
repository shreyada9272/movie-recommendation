# Makefile
# Builds the sequential and OpenMP parallel movie recommendation programs.
# Use `make` to compile the programs and `make clean` to remove build files.

CC     = gcc
# -ffp-contract=off : forbid fused multiply-add so sequential and parallel
#                     builds do identical floating point arithmetic.
CFLAGS = -O3 -march=native -ffp-contract=off -Wall -std=gnu11

all: recommender_seq recommender_omp

recommender_seq: recommender_seq.c common.h
	$(CC) $(CFLAGS) -o $@ $< -lm

recommender_omp: recommender_omp.c common.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< -lm

clean:
	rm -f recommender_seq recommender_omp
