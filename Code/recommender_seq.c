/*
 * recommender_seq.c  -  SEQUENTIAL item-based collaborative filtering.
 *
 * Usage: ./recommender_seq ratings.dat MAX_USERS MAX_ITEMS [output_file]
 */
#include "common.h"


int main(int argc, char **argv)
{
    if (argc < 4) {
        printf("\n====================================\n");
        printf("   MOVIE RECOMMENDATION SYSTEM\n");
        printf("====================================\n");
        printf("This program recommends movies based on\n");
        printf("similarity between movies users have rated.\n\n");

        printf("Usage:\n");
        printf("  %s ratings.dat MAX_USERS MAX_ITEMS [output_file]\n\n", argv[0]);

        printf("Input explanations:\n");
        printf("  ratings.dat  : MovieLens ratings dataset\n");
        printf("  MAX_USERS    : Maximum number of users to process\n");
        printf("  MAX_ITEMS    : Maximum number of movies to process\n");
        printf("  output_file  : Optional file to save recommendations\n\n");

        printf("MovieLens 1M contains up to 6040 users and 3952 movies.\n");
        printf("Use smaller limits to test performance at different sizes.\n");
        printf("Example:\n");
        printf("  %s ../Data/ratings.dat 1000 500 ../Output/recommendations.txt\n\n",
               argv[0]);
        return 0;
    }

    const char *path = argv[1];
    int max_users = atoi(argv[2]);
    int max_items = atoi(argv[3]);
    const char *out = argc > 4 ? argv[4] : NULL;

    printf("\n--- Movie Recommendation Configuration ---\n");
    printf("Dataset       : %s\n", path);
    printf("Maximum users : %d\n", max_users);
    printf("Maximum movies: %d\n", max_items);
    printf("Output file   : %s\n", out ? out : "Not requested");
    printf("------------------------------------------\n\n");

    /* ---------- load (not part of the measured compute time) ---------- */
    double t0 = wtime();
    Data d = load_data(path, max_users, max_items);
    double t_load = wtime() - t0;
    const int U = d.num_users, I = d.num_items;

    /*
 * Allocate memory for the recommendation computation:
 * sim stores the similarity score for every pair of movies.
 * ri stores the recommended movie IDs for each user.
 * rs stores the predicted scores for those recommendations.
 * seen marks movies already rated by the current user.
 */
    float *sim = malloc((size_t)I * I * sizeof(float));
    int   *ri  = malloc((size_t)U * TOP_N * sizeof(int));
    float *rs  = malloc((size_t)U * TOP_N * sizeof(float));
    char  *seen = calloc(I, 1);

    /* ---------- PHASE 1: item-item similarity matrix ---------- */
    double t1 = wtime();
    for (int a = 0; a < I; a++) {
        sim[(size_t)a * I + a] = 0.0f;
        for (int b = a + 1; b < I; b++) {
            float s = cosine(&d, a, b);
            sim[(size_t)a * I + b] = s;
            sim[(size_t)b * I + a] = s;
        }
    }
    double t2 = wtime();

    /*
 * Generate the top-N recommendations for each user.
 * Users are processed one after another in this sequential version.
 */
    /* ---------- PHASE 2: top-N recommendations for every user ---------- */
    for (int u = 0; u < U; u++)
        recommend_one_user(&d, sim, u, seen, ri + (size_t)u * TOP_N, rs + (size_t)u * TOP_N);
    double t3 = wtime();

    if (out) write_recs(out, &d, ri, rs);

    /* RESULT,version,users,items,ratings,threads,load,sim,rec,total,checksum */
    printf("RESULT,seq,%d,%d,%ld,1,%.6f,%.6f,%.6f,%.6f,%016llx\n",
           U, I, d.num_ratings, t_load, t2 - t1, t3 - t2, t3 - t1,
           (unsigned long long)checksum(&d, ri, rs));
    return 0;
}
