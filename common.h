/*
 * common.h  -  Code shared by the sequential and the OpenMP version.
 *
 * Algorithm: ITEM-BASED COLLABORATIVE FILTERING with COSINE SIMILARITY
 *   Step 1: similarity between every pair of movies  (items x items)
 *   Step 2: for every user, predict a score for each movie the user has
 *           NOT rated, then keep the TOP_N best movies.
 *
 * Both versions call exactly the same kernels below, so any difference in
 * output can only come from the parallelisation itself (there is none).
 */
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#define TOP_N 10          /* recommendations per user            */
#define DAMP  1.0f        /* shrinkage: avoids "one neighbour => 5.0 stars" */

typedef struct {
    int      num_users;   /* number of users actually used                 */
    int      num_items;   /* number of items (movie ids) actually used     */
    long     num_ratings;
    uint8_t *mat;         /* dense, ITEM-major: mat[item*num_users + user], 0 = not rated */
    int     *u_ptr;       /* CSR by user: ratings of user u are u_ptr[u]..u_ptr[u+1]-1 */
    int     *u_item;      /* item index of each rating                      */
    uint8_t *u_rat;       /* rating value of each rating                    */
    float   *norm;        /* Euclidean norm of every item column            */
} Data;

/* wall-clock time in seconds */
#ifdef _WIN32
#include <windows.h>

double wtime(void) {
    static LARGE_INTEGER frequency;
    static int initialized = 0;
    LARGE_INTEGER counter;

    if (!initialized) {
        QueryPerformanceFrequency(&frequency);
        initialized = 1;
    }

    QueryPerformanceCounter(&counter);

    return (double)counter.QuadPart / frequency.QuadPart;
}

#else

#include <time.h>

double wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#endif

/* ------------------------------------------------------------------ */
/* Load MovieLens file "UserID::MovieID::Rating::Timestamp".           */
/* Only users <= max_users and movies <= max_items are kept -> this is */
/* how we create smaller datasets from the full one.                   */
/* ------------------------------------------------------------------ */
static Data load_data(const char *path, int max_users, int max_items)
{
    Data d;
    memset(&d, 0, sizeof d);
    FILE *f = fopen(path, "r");
    if (!f) { perror("cannot open ratings file"); exit(1); }

    long cap = 1 << 20, n = 0;
    int *tu = malloc(cap * sizeof(int));
    int *ti = malloc(cap * sizeof(int));
    uint8_t *tr = malloc(cap);
    int u, i, r, seen_u = 0, seen_i = 0;

    while (fscanf(f, "%d::%d::%d::%*d", &u, &i, &r) == 3) {
        if (u > max_users || i > max_items) continue;
        if (n == cap) {
            cap *= 2;
            tu = realloc(tu, cap * sizeof(int));
            ti = realloc(ti, cap * sizeof(int));
            tr = realloc(tr, cap);
        }
        tu[n] = u - 1; ti[n] = i - 1; tr[n] = (uint8_t)r; n++;
        if (u > seen_u) seen_u = u;
        if (i > seen_i) seen_i = i;
    }
    fclose(f);

    d.num_users   = seen_u;
    d.num_items   = seen_i;
    d.num_ratings = n;

    /* dense item-major matrix */
    d.mat = calloc((size_t)d.num_items * d.num_users, 1);
    for (long k = 0; k < n; k++)
        d.mat[(size_t)ti[k] * d.num_users + tu[k]] = tr[k];

    /* CSR by user (counting sort) */
    d.u_ptr  = calloc(d.num_users + 1, sizeof(int));
    d.u_item = malloc(n * sizeof(int));
    d.u_rat  = malloc(n);
    for (long k = 0; k < n; k++) d.u_ptr[tu[k] + 1]++;
    for (int x = 0; x < d.num_users; x++) d.u_ptr[x + 1] += d.u_ptr[x];
    int *fill = malloc(d.num_users * sizeof(int));
    memcpy(fill, d.u_ptr, d.num_users * sizeof(int));
    for (long k = 0; k < n; k++) {
        int pos = fill[tu[k]]++;
        d.u_item[pos] = ti[k];
        d.u_rat[pos]  = tr[k];
    }

    /* item norms */
    d.norm = malloc(d.num_items * sizeof(float));
    for (int a = 0; a < d.num_items; a++) {
        long s = 0;
        const uint8_t *row = d.mat + (size_t)a * d.num_users;
        for (int k = 0; k < d.num_users; k++) s += row[k] * row[k];
        d.norm[a] = sqrtf((float)s);
    }

    free(tu); free(ti); free(tr); free(fill);
    return d;
}

/* ------------------------------------------------------------------ */
/* KERNEL 1: cosine similarity between item a and item b.              */
/* The dot product is done in INTEGERS, so it is exact and does not    */
/* depend on the order of additions (important for identical results). */
/* ------------------------------------------------------------------ */
static inline float cosine(const Data *d, int a, int b)
{
    float den = d->norm[a] * d->norm[b];
    if (den == 0.0f) return 0.0f;
    const uint8_t *x = d->mat + (size_t)a * d->num_users;
    const uint8_t *y = d->mat + (size_t)b * d->num_users;
    int dot = 0;
    for (int k = 0; k < d->num_users; k++) dot += x[k] * y[k];
    return (float)dot / den;
}

/* ------------------------------------------------------------------ */
/* KERNEL 2: top-N recommendations for ONE user.                       */
/*   score(i) = sum_j sim(i,j)*rating(u,j) / (sum_j sim(i,j) + DAMP)   */
/*   j runs over the movies user u has rated, i over the unrated ones. */
/* `seen` is a scratch array (size num_items, all zeros on entry/exit).*/
/* Ties are broken by lower item id -> output is fully deterministic.  */
/* ------------------------------------------------------------------ */
static void recommend_one_user(const Data *d, const float *sim, int u,
                               char *seen, int *out_item, float *out_score)
{
    const int I = d->num_items;
    for (int k = d->u_ptr[u]; k < d->u_ptr[u + 1]; k++) seen[d->u_item[k]] = 1;

    int cnt = 0;
    for (int i = 0; i < I; i++) {
        if (seen[i]) continue;
        const float *row = sim + (size_t)i * I;
        float num = 0.0f, den = 0.0f;
        for (int k = d->u_ptr[u]; k < d->u_ptr[u + 1]; k++) {
            float s = row[d->u_item[k]];
            num += s * d->u_rat[k];
            den += s;
        }
        if (den <= 0.0f) continue;
        float score = num / (den + DAMP);

        /* insertion into the sorted top-N list (descending score) */
        int p = cnt < TOP_N ? cnt : TOP_N;
        while (p > 0 && score > out_score[p - 1]) p--;
        if (p >= TOP_N) continue;
        int last = cnt < TOP_N ? cnt : TOP_N - 1;
        for (int q = last; q > p; q--) {
            out_score[q] = out_score[q - 1];
            out_item[q]  = out_item[q - 1];
        }
        out_score[p] = score;
        out_item[p]  = i;
        if (cnt < TOP_N) cnt++;
    }
    for (int k = cnt; k < TOP_N; k++) { out_item[k] = -1; out_score[k] = 0.0f; }

    for (int k = d->u_ptr[u]; k < d->u_ptr[u + 1]; k++) seen[d->u_item[k]] = 0;
}

/* ------------------------------------------------------------------ */
/* Output helpers                                                      */
/* ------------------------------------------------------------------ */
static void write_recs(const char *path, const Data *d, const int *ri, const float *rs)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror("cannot write output"); exit(1); }
    for (int u = 0; u < d->num_users; u++) {
        fprintf(f, "%d", u + 1);
        for (int k = 0; k < TOP_N; k++)
            fprintf(f, " %d:%.6f", ri[(size_t)u * TOP_N + k] + 1, rs[(size_t)u * TOP_N + k]);
        fprintf(f, "\n");
    }
    fclose(f);
}

/* FNV-1a checksum over all recommended item ids and score bit patterns */
static uint64_t checksum(const Data *d, const int *ri, const float *rs)
{
    uint64_t h = 1469598103934665603ULL;
    size_t total = (size_t)d->num_users * TOP_N;
    for (size_t k = 0; k < total; k++) {
        uint32_t bits;
        memcpy(&bits, &rs[k], 4);
        h = (h ^ (uint32_t)ri[k]) * 1099511628211ULL;
        h = (h ^ bits)            * 1099511628211ULL;
    }
    return h;
}

#endif
