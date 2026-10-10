/*
 * recommender_omp.c  -  OpenMP version of the same algorithm.
 *
 * Usage: ./recommender_omp ratings.dat MAX_USERS MAX_ITEMS THREADS [output_file]
 *
 * Phase 1: each item row 'a' of the similarity matrix is an independent task.
 * Phase 2: each user is an independent task (own output slot, own scratch array).
 * No two threads ever write the same memory location -> no race conditions,
 * no critical sections, no atomics.
 */
#include "common.h"
#include <omp.h>

int main(int argc, char **argv)
{
if (argc < 5) {
    printf("\n====================================\n");
    printf("   MOVIE RECOMMENDATION SYSTEM\n");
    printf("   OPENMP PARALLEL VERSION\n");
    printf("====================================\n");
    printf("This program recommends movies using\n");
    printf("parallel computation with OpenMP.\n\n");

    printf("Usage:\n");
    printf("  %s ratings.dat MAX_USERS MAX_ITEMS THREADS [output_file]\n\n", argv[0]);

    printf("Input explanations:\n");
    printf("  ratings.dat : MovieLens ratings dataset\n");
    printf("  MAX_USERS   : Maximum number of users to process\n");
    printf("  MAX_ITEMS   : Maximum number of movies to process\n");
    printf("  THREADS     : Number of parallel worker threads\n");
    printf("  output_file : Optional file to save recommendations\n\n");

    printf("MovieLens 1M contains up to 6040 users and 3952 movies.\n");
    printf("Use smaller limits to test performance at different sizes.\n");
    printf("THREADS controls how many workers run in parallel.\n\n");

    printf("Example:\n");
    printf("  %s ../Data/ratings.dat 1000 500 4 ../Output/recommendations.txt\n\n",
           argv[0]);
    return 0;
}

const char *path = argv[1];


    int max_users = atoi(argv[2]);
    int max_items = atoi(argv[3]);
    int threads   = atoi(argv[4]);
    const char *out = argc > 5 ? argv[5] : NULL;
    omp_set_num_threads(threads);

    double t0 = wtime();
    Data d = load_data(path, max_users, max_items);
    double t_load = wtime() - t0;
    const int U = d.num_users, I = d.num_items;

    float *sim = malloc((size_t)I * I * sizeof(float));
    int   *ri  = malloc((size_t)U * TOP_N * sizeof(int));
    float *rs  = malloc((size_t)U * TOP_N * sizeof(float));

    /* ---------- PHASE 1: item-item similarity matrix ---------- */
    /* Row a has (I-a-1) pairs -> uneven work -> schedule(dynamic).  */

    /*

* Allocate memory for movie similarity scores and recommendation results.
* This parallel implementation uses OpenMP to speed up computation
* across multiple threads.
  */

    double t1 = wtime();
    #pragma omp parallel for schedule(dynamic, 4)
    for (int a = 0; a < I; a++) {
        sim[(size_t)a * I + a] = 0.0f;
        for (int b = a + 1; b < I; b++) {
            float s = cosine(&d, a, b);
            sim[(size_t)a * I + b] = s;      /* each (a,b) cell is written by */
            sim[(size_t)b * I + a] = s;      /* exactly one thread            */
        }
    }
    double t2 = wtime();

    /* ---------- PHASE 2: top-N recommendations for every user ---------- */
    /* Users have different numbers of ratings -> schedule(dynamic).      */
    #pragma omp parallel
    {
        char *seen = calloc(I, 1);           /* private scratch array per thread */
        #pragma omp for schedule(dynamic, 16)
        for (int u = 0; u < U; u++)
            recommend_one_user(&d, sim, u, seen, ri + (size_t)u * TOP_N, rs + (size_t)u * TOP_N);
        free(seen);
    }
    double t3 = wtime();

    if (out) write_recs(out, &d, ri, rs);

    printf("RESULT,omp,%d,%d,%ld,%d,%.6f,%.6f,%.6f,%.6f,%016llx\n",
           U, I, d.num_ratings, threads, t_load, t2 - t1, t3 - t2, t3 - t1,
           (unsigned long long)checksum(&d, ri, rs));
    return 0;
}
