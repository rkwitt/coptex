#define _FILE_OFFSET_BITS 64

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <sys/time.h>
#include <time.h>
#include "pthread.h"
#include "common.h"

extern char *progname;
extern unsigned int verbose;

int count_listfile(const char *list_fname) {
    FILE *file = fopen(list_fname, "rt");
    if (!file)
        return -1;

    char line[1024];
    int fcount = 0;
    while (fgets(line, sizeof(line), file) != NULL)
        fcount++;
    fclose(file);
    
    return fcount;
}

compmode_t check_compmode(const char *arg) {
    compmode_t mode;
    
    if (!strcmp(arg, "matrix"))
        mode = DISTMATRIX;
    else if (!strcmp(arg, "bestn"))
        mode = DISTBESTN;
    else {
        fprintf(stderr, "%s: invalid computation mode '%s', exit.\n", progname, arg);
        exit(EXIT_FAILURE);
    }
    
    return mode;
}

// log-determinant of matrix 
double precalc_determinant(gsl_matrix *A) {
	int s = 0; 

	gsl_permutation *p = gsl_permutation_alloc(A->size1);
	gsl_matrix *LU = gsl_matrix_calloc(A->size1, A->size2);

	gsl_matrix_memcpy(LU,A);
	gsl_linalg_LU_decomp(LU,p,&s);
	double det = gsl_linalg_LU_lndet(LU);

	gsl_matrix_free(LU);
	gsl_permutation_free(p);

	return det;
}

// calculate inverse of A and store in B
gsl_matrix *precalc_inverse(gsl_matrix *A) {
	int s = 0;
	gsl_permutation *p = gsl_permutation_alloc(A->size1);
	gsl_matrix *T = gsl_matrix_calloc(A->size1,A->size2);

	gsl_matrix_memcpy(T,A);
	gsl_linalg_LU_decomp(T,p,&s); // LU decomposition
	gsl_matrix *B = gsl_matrix_calloc(T->size1,T->size2); // allocate space for inverse
	gsl_linalg_LU_invert(T, p, B);	// invert using GSL

	gsl_permutation_free(p);
	gsl_matrix_free(T);

	return B; // return inverse
}

int start_threads(unsigned int num_threads, unsigned int num_images, unsigned int start_image, unsigned int process_images, void *(*thread_function)(void*), void *metric_params) {
    if (!thread_function || !metric_params)
        return 0;

    if (start_image >= num_images) {
        fprintf(stderr, "%s: start image index beyond number of images: %d >= %d\n", progname, start_image, num_images);
        return 0;
    }

    if (process_images > num_images) {
        process_images = num_images;
    }

    if (process_images == 0 || (start_image + process_images) > num_images) {
        process_images = num_images - start_image;
    }

    fprintf(stderr, "%s: processing %d images\n", progname, process_images);

    if (num_threads > 1 && num_images >= num_threads) {
        pthread_t *threads = (pthread_t *) malloc(num_threads * sizeof(pthread_t));
        
        thread_params_t *thread_params = (thread_params_t *) malloc(num_threads * sizeof(thread_params_t));
        for (unsigned int i = 0; i < num_threads; i++) {
            int ret;
            
            unsigned int start_image = i * (num_images / num_threads);
            unsigned int process_images = num_images / num_threads;

            if (i == num_threads - 1) {
                process_images += num_images % num_threads;
            }

            thread_params[i].thread_num = i;
            thread_params[i].num_threads = num_threads;
            thread_params[i].num_images = num_images;            
            thread_params[i].metric_params = metric_params;
            thread_params[i].start_image = start_image;
            thread_params[i].process_images = process_images;
                                                
            if ((ret = pthread_create(&threads[i], 0, thread_function, &thread_params[i]))) {
                fprintf(stderr, "%s: failed to create thread, ret = %d, exit.\n", progname, ret);
                return 0;
            }
        }        
        
        for (unsigned int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], 0);
        }        
        
        free(threads);
        free(thread_params);
    }
    else {
        static thread_params_t thread_params;

        thread_params.thread_num = 1;
        thread_params.num_threads = 1;
        thread_params.num_images = num_images;
        thread_params.metric_params = metric_params;
        thread_params.start_image = start_image;
        thread_params.process_images = process_images;
    
        thread_function(&thread_params);
    }

    return 1;
}

int contains_inf(gsl_matrix *A) {
    for (unsigned int i = 0; i < A->size1; i++) {
        for (unsigned int j = 0; j < A->size2; j++) {    
            if (isinf(gsl_matrix_get(A, i, j))) return 1;
        }
    }        
    return 0;
}


int contains_nan(gsl_matrix *A) {
    for (unsigned int i = 0; i < A->size1; i++) {
        for (unsigned int j = 0; j < A->size2; j++) {    
            if (isnan(gsl_matrix_get(A, i, j))) return 1;
        }
    }        
    return 0;
}
int open_models_filelist(const char *fname, unsigned int *num, FILE **f) {
    if (!fname || !num || !f) {
        fprintf(stderr, "%s: parameter error, exit.\n", progname);
        return 0;
    }

    *num = count_listfile(fname);  // number of images  

    *f = fopen(fname, "rt");
    if (!*f || num < 0) {
        fprintf(stderr, "%s: failed to open '%s': %s\n", progname, fname, strerror(errno));
        return 0;
    }
    
    if (*num == 0) {
        fprintf(stderr, "%s: no entries in file list to process, exit.\n", progname);
        return 0;
    }

    return 1;
}

int open_models(const char *fname, unsigned int *num, unsigned int model_size, FILE **f) {
    if (!fname || !num || !f || !model_size) {
        fprintf(stderr, "%s: parameter error, exit.\n", progname);
        return 0;
    }
    
    *f = fopen(fname, "rb");
    if (!*f) {
        fprintf(stderr, "%s: failed to open '%s': %s\n", progname, fname, strerror(errno));
        return 0;
    }
    
    fseeko(*f, 0, SEEK_END);
    off_t file_offset = ftello(*f);
    *num = file_offset / model_size;
    fseeko(*f, 0, SEEK_SET);

    if (*num == 0) {
        fprintf(stderr, "%s: no entries in model file to process, exit.\n", progname);
        return 0;
    }

    return 1;
}

int start_benchmark(unsigned int num_threads, unsigned int num_images, compmode_t mode, void *(*thread_function)(void*), void *metric_params, const unsigned int queries) {
    static thread_params_t thread_params;

    verbose = false;

    if (mode != DISTMATRIX) { 
        fprintf(stderr, "%s: don't use 'bestn' mode with benchmarking, exit.\n", progname);
        return 0;
    }

    if (num_threads != 1 || num_images < 1024) {
        fprintf(stderr, "%s: must have at least 1024 models and only one thread for benchmarking, exit.\n", progname);
        return 0;
    }

    fprintf(stderr, "%s: running in benchmark mode (1 thread, 1024 models)...\n", progname);
    fprintf(stderr, "%s: no distance output will be written...\n", progname);

    srand(1234);

    tic();
    for (unsigned int query = 0; query < queries; query++) {
        thread_params.thread_num = 1;
        thread_params.num_threads = 1;
        thread_params.num_images = 1024;
        thread_params.metric_params = metric_params;
        thread_params.start_image = rand() % 1024;
        thread_params.process_images = 1;

        fprintf(stderr, "%s: query id: %d/%d (image %d)\n", progname, query, queries, thread_params.start_image);
        thread_function(&thread_params);
    }
    unsigned ms = toc();
    
    fprintf(stdout, "%s: %d queries in %d ms, done.\n", progname, queries, ms);
    
    return 1;    
}

static struct timeval ticstart;

void tic() {
    if (gettimeofday(&ticstart, 0) < 0) {
        fprintf(stderr, "%s: failed to query timer, exit.\n", progname);
        exit(EXIT_FAILURE);
    }
}

unsigned int toc() {
    struct timeval tocstop;
    
    if (gettimeofday(&tocstop, 0) < 0) {
        fprintf(stderr, "%s: failed to query timer, exit.\n", progname);
        exit(EXIT_FAILURE);
    }

    unsigned long long ms;
    
    ms = (tocstop.tv_sec - ticstart.tv_sec) * 1000;
    if (tocstop.tv_usec < ticstart.tv_usec) ms -= round((ticstart.tv_usec - tocstop.tv_usec) / 1000.0);
    else ms += round((tocstop.tv_usec - ticstart.tv_usec) / 1000.0);
    
    return ms;
}

