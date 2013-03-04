#ifndef COMMON_H
#define COMMON_H

#include <gsl/gsl_matrix.h>

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define GSL_SF_PSI_1 -0.577215664901532


typedef enum {DISTMATRIX, DISTBESTN} compmode_t;

// count number of lines in a file
int count_listfile(const char *list_fname);

// check if computation mode is valid (i.e. 'matrix' or 'bestn')
compmode_t check_compmode(const char *arg);

// log-determinant of matrix 
double precalc_determinant(gsl_matrix *A);

// calculate inverse of A and store in B
gsl_matrix *precalc_inverse(gsl_matrix *A);

typedef struct {
    unsigned int thread_num;
    unsigned int num_threads;
    unsigned int num_images;
    unsigned int start_image;
    unsigned int process_images;
    void *metric_params;    
} thread_params_t;

int start_threads(unsigned int num_threads, unsigned int num_images, unsigned int start_image, unsigned int process_images, void *(*thread_function)(void*), void *metric_params);

int start_benchmark(unsigned int num_threads, unsigned int num_images, compmode_t mode, void *(*thread_function)(void*), void *metric_params, const unsigned int queries = 1000);

// returns 1 if matrix contains NaN
int contains_nan(gsl_matrix *A);
// returns 1 if matrix contains Inf
int contains_inf(gsl_matrix *A);

int open_models_filelist(const char *fname, unsigned int *num, FILE **f);

int open_models(const char *fname, unsigned int *num, unsigned int model_size, FILE **f);

// start timer
void tic();

// return elapsed time in milliseconds
unsigned int toc();

#endif /* COMMON_H */
