/*
 * recommender_seq.c  -  SEQUENTIAL item-based collaborative filtering.
 *
 * Usage: ./recommender_seq ratings.dat MAX_USERS MAX_ITEMS [output_file]
 */
#include "common.h"

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s ratings.dat MAX_USERS MAX_ITEMS [output_file]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    int max_users = atoi(argv[2]);
    int max_items = atoi(argv[3]);
    const char *out = argc > 4 ? argv[4] : NULL;

    /* ---------- load (not part of the measured compute time) ---------- */
    double t0 = wtime();
    Data d = load_data(path, max_users, max_items);
    double t_load = wtime() - t0;
    const int U = d.num_users, I = d.num_items;

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
