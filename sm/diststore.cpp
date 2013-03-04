#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <float.h>
#include <gsl/gsl_matrix.h>
#include <pthread.h>
#include "diststore.h"

extern const char *progname;

typedef struct {
    double dist __attribute__ ((aligned (8)));
    unsigned int index;
} ranking_t;

static ranking_t **rankings = 0;
static gsl_matrix *distmat = 0;
static const unsigned int bestn = 100;
static distmode_t dist_mode = UNKNOWN;
static unsigned int dist_dim = 0;
static pthread_mutex_t mutex;

static ranking_t **alloc_rankings(unsigned int dim, unsigned int n, bool top) {
    ranking_t **r = (ranking_t **) malloc(dim * sizeof(ranking_t *));
    for (unsigned int i = 0; i < dim; i++) {
        r[i] = (ranking_t *) malloc(n * sizeof(ranking_t));
        
        for (unsigned int j = 0; j < n; j++) {
            r[i][j].dist = (top) ? -DBL_MAX : DBL_MAX;
            r[i][j].index = -1;             
        }
    }
    
    return r;
}

int init_diststore(unsigned int dim, distmode_t mode) {
    free_diststore();
    
    switch (mode) {
        case MATRIX:
            distmat = gsl_matrix_calloc(dim, dim);
            if (!distmat)
                return 0;
            break;
        case TOPN:
            rankings = alloc_rankings(dim, bestn, true);
            break;
        case BOTTOMN:
            rankings = alloc_rankings(dim, bestn, false);
            break;
        default:
            fprintf(stderr, "%s: invalid distance store mode, exit.\n", progname);
            exit(EXIT_FAILURE);
    }

    dist_mode = mode;
    dist_dim = dim;
    
    if (pthread_mutex_init(&mutex, 0)) {
        fprintf(stderr, "%s: failed to initialize distance store mutex, exit.\n", progname);
        exit(EXIT_FAILURE);
    }
    
    return 1; 
}

static void insert_top(ranking_t *r, unsigned int j, double dist) {
    unsigned int pos = bestn-1;

    pthread_mutex_lock(&mutex);

    while (pos > 0 && dist >= r[pos-1].dist) {
        r[pos] = r[pos-1];
        pos--;
    }

    r[pos].dist = dist;
    r[pos].index = j;        

    pthread_mutex_unlock(&mutex);
}

static void insert_bottom(ranking_t *r, unsigned int j, double dist) {
    unsigned int pos = bestn-1;

    pthread_mutex_lock(&mutex);

    while (pos > 0 && dist <= r[pos-1].dist) {
        r[pos] = r[pos-1];
        pos--;
    }

    r[pos].dist = dist;
    r[pos].index = j;        

    pthread_mutex_unlock(&mutex);
}

/*
static void print_ranking(unsigned int image, ranking_t *r) {
    fprintf(stderr, "%s: image %d ", progname, image);
    for (unsigned int i = 0; i < bestn; i++) {
        fprintf(stderr, "(%f %d) ", r[i].dist, r[i].index);
    }
    fprintf(stderr, "\n");
}
*/

void store_dist(unsigned int i, unsigned int j, double dist) {
    switch (dist_mode) {
        case MATRIX: 
            gsl_matrix_set(distmat, i, j, dist);
            break;
        case TOPN:
            if (dist > rankings[i][bestn-1].dist) {
                insert_top(rankings[i], j, dist);
            }
            // print_ranking(i, rankings[i]);
            break;
        case BOTTOMN:
            if (dist < rankings[i][bestn-1].dist) {
                insert_bottom(rankings[i], j, dist);            
            }
            // print_ranking(i, rankings[i]);            
            break;
        default:
            fprintf(stderr, "%s: invalid distance store mode, exit.\n", progname);
            exit(EXIT_FAILURE);
    }
}

static int write_dist_matrix(const char *fname) {
    FILE *mfile = fopen(fname, "wb"); 
    if (!mfile) {
        fprintf(stderr, "%s: failed to create '%s': %s\n", progname, fname, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if (gsl_matrix_fwrite(mfile, distmat) == GSL_EFAILED) {
        fprintf(stderr, "%s: failed to write '%s': %s\n", progname, fname, strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(mfile);

    return 1;
}

static int write_dist_rankings(const char *fname) {
    char basename[1024];
    char distfname[1024];
    char indexfname[1024];    
    
    if (!strrchr(fname, '.')) {
        strcpy(basename, fname);
        sprintf(distfname, "%s_d.%s", basename, "bin");
        sprintf(indexfname, "%s_i.%s", basename, "bin");    
    }
    else {
        strcpy(basename, fname);
        char *ext = strrchr(basename, '.');
        *ext++ = '\0';

        sprintf(distfname, "%s_d.%s", basename, ext);
        sprintf(indexfname, "%s_i.%s", basename, ext);    
    }

    FILE *fdist = fopen(distfname, "wb"); 
    if (!fdist) {
        fprintf(stderr, "%s: failed to create '%s': %s\n", progname, distfname, strerror(errno));
        exit(EXIT_FAILURE);
    }
    FILE *findex = fopen(indexfname, "wb"); 
    if (!findex) {
        fprintf(stderr, "%s: failed to create '%s': %s\n", progname, indexfname, strerror(errno));
        exit(EXIT_FAILURE);
    }

    for (unsigned int i = 0; i < dist_dim; i++) {
        for (unsigned int j = 0; j < bestn; j++) {

            if (fwrite(&rankings[i][j].dist, sizeof(double), 1, fdist) != 1) {
                fprintf(stderr, "%s: failed to write '%s': %s\n", progname, distfname, strerror(errno));
                exit(EXIT_FAILURE);
            }

            if (fwrite(&rankings[i][j].index, sizeof(unsigned int), 1, findex) != 1) {
                fprintf(stderr, "%s: failed to write '%s': %s\n", progname, indexfname, strerror(errno));
                exit(EXIT_FAILURE);
            }
        }            
    }            
    fclose(fdist);
    fclose(findex);

    return 1;
}

int write_diststore(const char *fname) {
    int ret;
    
    switch (dist_mode) {
        case MATRIX:
            ret = write_dist_matrix(fname);
            break;
        case TOPN:
        case BOTTOMN:
            ret = write_dist_rankings(fname);        
            break;
        default:
            fprintf(stderr, "%s: invalid distance store mode, exit.\n", progname);
            exit(EXIT_FAILURE);
    }

    return ret;
}

void free_diststore() {
    if (distmat) {
        gsl_matrix_free(distmat);
        distmat = 0;
    }
        
    if (rankings) {
        for (unsigned int i = 0; i < dist_dim; i++) {
            free(rankings[i]);
        }
        free(rankings);
        rankings = 0;
    }        
    
    pthread_mutex_destroy(&mutex);
    
    dist_mode = UNKNOWN;
    dist_dim = 0;
}    

