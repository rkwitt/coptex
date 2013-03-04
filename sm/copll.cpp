/** 
  * compute the log-likelihood of a collection of query feature vectors under
  * copula-based feature representations;
  *
  * currently supported copulas are: gaussian, student-t
  * currently supported margins are: ggd, weibull, gamma
  *
  * author(s): Kwitt Roland, Meerwald Peter, 2010
  */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <sys/types.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_cdf.h>
#include <math.h>
#include "diststore.h"
#include "common.h"
#include "pthread.h"
#include "getopt.h"
#include <set>

#include "fmath.hpp"

#define EIGEN2_SUPPORT
#include "Eigen/Array"
#include "Eigen/Core"

USING_PART_OF_NAMESPACE_EIGEN

#define ITMAX 200
#define EPS 3.0e-7
#define FPMIN 1.0e-30
#define COPULA_TYPE_GAUSSIAN 0
#define COPULA_TYPE_STUDENTT 1

// globals
const char *progname = "copll";
unsigned int verbose = 0;

extern int isnan(double);
static void ctbll(gsl_matrix **data, gsl_matrix ** rhoList, gsl_matrix **marginList, MatrixXf **invRhoList, double *cholDiagSumList, double *nulist, char *base_path, char **modelname_list, unsigned int D, unsigned int N, unsigned int NS, unsigned int P, unsigned int S,unsigned int C, unsigned int num, unsigned int start_image, unsigned int process_images);
static double copulapdf_gaussian(gsl_matrix *unif, MatrixXf *irho, double cholDiagSum, unsigned int D, unsigned int NS);
static double copulapdf_studentt(gsl_matrix *unif, MatrixXf *irho, double cholDiagSum, double nu, unsigned int D, unsigned int NS);
static double *getDiagonal(gsl_matrix *A,unsigned int dim);
static gsl_matrix * getLT(gsl_matrix *A, unsigned int dim);
static gsl_matrix *compCholInv(gsl_matrix *Rho, unsigned int D);
static double compCholDiag(gsl_matrix *Rho, unsigned int D);
static void *ctbll_thread(void *thread_params);
static double norminv(double p);
static double norminvf(double p);
static double tinv(double p, double nu);
static void gser(double *gamser, double a, double x, double *gln);
static void gcf(double *gammcf, double a, double x, double *gln);
static double plica(double a, double x);
static double ggd_cdf(double x, double a, double b);
static double gammq(double a, double x);
static double gammp(double a, double x);

typedef struct {
    gsl_matrix **data;
	gsl_matrix **rhoList;
	gsl_matrix **marginList;
	MatrixXf **invRhoList;
	double *cholDiagSumList;
	double *nulist;
    unsigned int D; // dimensionality
	unsigned int N; // sample length
	unsigned int NS; // subsampled length	
	unsigned int P; // distribution specifier
	unsigned int S; // stepsize (take every n-th coefficient)
	unsigned int C; // copula
	char *base_path;
	char **modelname_list;
} metric_params_t;

// default values
const char *DIST_OUT_FNAME = "dist.bin";
const char *LIST_FNAME = "filelist.txt";
const char *MODEL_FNAME = "";
const char *BASE_PATH = "";
const unsigned int DEFAULT_DIMENSIONALITY = 6;
const unsigned int DEFAULT_DATALEN = 256;
const unsigned int DEFAULT_START_IMAGE = 0;
const unsigned int DEFAULT_PROCESS_IMAGES = 0;
const unsigned int DEFAULT_STEPSIZE = 1;
const unsigned int DEFAULT_NUM_THREADS = 1;
const unsigned int DEFAULT_MARGIN_DISTRIBUTION = 0; // 0..Weibull, 1..Gamma, 2...GGD
const unsigned int DEFAULT_COPULA = 0; // 0..Normal, 1..Student-t

static std::set<unsigned int> subsampl_set;

static void usage() {
    fprintf(stderr,
        "usage: %s [-D n] [-N n] [-S n] [-P n] [-C n] [-B dir] [-d file] [-l file] [-s n] [-c n] [-t n] [-m mode]\n"
        "\n"
        "\t-B dir\t\tpath to model files\n\t\t\t(default: /tmp)\n"
        "\t-D n\t\tnumber of dimensions\n\t\t\t(default: %d)\n"
        "\t-S n\t\tstepsize for Likelihood computation (take every n-th coefficient) \n\t\t\t(default: %d)\n"
    		"\t-N n\t\tlength of coefficient data\n\t\t\t(default: %d)\n"
    		"\t-P n\t\tspecify marginal distribution (0..Weibull,1..Gamma,2..GGD)\n\t\t\t(default: %d)\n"
        "\t-C n\t\tspecify Copula to use (0..Normal,1..Student-t)\n\t\t\t(default: %d)\n"
		    "\t-d file\t\tfile name of output distance matrix\n\t\t\t(default: %s)\n"
        "\t-l file\t\tfile name with list of files to process\n\t\t\t(default: %s)\n"
        "\t-s n\t\timage to start from, should be multiple of 16 (default: 0)\n"
        "\t-c n\t\tnumber of images to process (default: 0 .. all)\n"
        "\t-t n\t\tnumber of threads to use (default: 1)\n"                        
        "\t-m mode\t\tcomputation mode 'matrix' or 'bestn' (default: 'matrix')\n"
        "\t-v\t\tverbose\n",
        progname, DEFAULT_DIMENSIONALITY, DEFAULT_STEPSIZE, DEFAULT_DATALEN, DEFAULT_MARGIN_DISTRIBUTION, DEFAULT_COPULA, DIST_OUT_FNAME, LIST_FNAME);
}

static void read_model_data(const char *base_path, char **modelname_list, gsl_matrix *coefdata, unsigned int model, unsigned int D, unsigned int N, unsigned int NS, unsigned int S) {
	char modelname[1024]; // name of model
	
    /* read data files (DTCWT coefficients) */
    sprintf(modelname, "%s%s.data", base_path, modelname_list[model]);
    FILE *mfile = fopen(modelname, "rb");
	if (!mfile) {
       	fprintf(stderr, "%s: failed to open '%s': %s\n", progname, modelname, strerror(errno));
       	exit(EXIT_FAILURE);
    }
	
    if (coefdata == 0) {
    	fprintf(stderr, "%s: no memory for '%s'\n", progname, modelname);
    	exit(EXIT_FAILURE);
    }

    
    if (S > 1) {
        // subsample coefs

        if (subsampl_set.empty()) {
            for (unsigned int i = 0; i < NS; ) {
                unsigned int r = rand() % N;
                if (subsampl_set.find(r) == subsampl_set.end()) {
                    subsampl_set.insert(r);
                    i++;
                }
            }
        }
        
        gsl_matrix *coefdata_full = gsl_matrix_alloc(D, N);
        if (gsl_matrix_fread(mfile, coefdata_full) == GSL_EFAILED) {
            	fprintf(stderr, "%s: failed to read '%s': %s\n", progname, modelname, strerror(errno));
            	exit(EXIT_FAILURE);
        }

        for (unsigned int d = 0; d < D; d++) {

            unsigned int ns = 0;
            for (std::set<unsigned int>::iterator i = subsampl_set.begin(); i != subsampl_set.end(); i++, ns++) {
                
                gsl_matrix_set(coefdata, d, ns, gsl_matrix_get(coefdata_full, d, *i));
            }
        }
        gsl_matrix_free(coefdata_full);
    }
    else {
        if (gsl_matrix_fread(mfile, coefdata) == GSL_EFAILED) {
            	fprintf(stderr, "%s: failed to read '%s': %s\n", progname, modelname, strerror(errno));
            	exit(EXIT_FAILURE);
        }
    }    
    
    fclose(mfile);
}

void gsl_error_handler(const char *reason, const char *file, int line, int gsl_errno) {
    static unsigned int error_count = 0;
    fprintf(stderr, "%s: GSL error '%s', code %d, error count %d\n", progname, reason, gsl_errno, error_count);
    error_count++;
}

int main(int argc, char *argv[]) {
    int c;
    unsigned int num=0;
	unsigned int cnt=0;
	gsl_matrix *Rho = NULL;					// cov. matrix
	gsl_matrix *marginParams = NULL;		// marin parameters
	gsl_matrix **coeflist = NULL;			// list of coefficient data
	gsl_matrix **RhoList = NULL;			// list of covariance matrices
	gsl_matrix **marginParamList = NULL; 	// list of marginal distribution parameters (either Weibull, Gamma or GGD)
	MatrixXf **invRhoList = NULL; 			// list of precomputed inverses of covariance matrices
	double *cholDiagSumList = NULL;
    char **modelname_list = NULL;
    double *nulist = NULL;

    unsigned int D = DEFAULT_DIMENSIONALITY;
    unsigned int N = DEFAULT_DATALEN;
    unsigned int NS = DEFAULT_DATALEN;      
	unsigned int S = DEFAULT_STEPSIZE;
	unsigned int P = DEFAULT_MARGIN_DISTRIBUTION;
	unsigned int C = DEFAULT_COPULA;
	unsigned int start_image = DEFAULT_START_IMAGE;
    unsigned int process_images = DEFAULT_PROCESS_IMAGES;
    unsigned int num_threads = DEFAULT_NUM_THREADS;
        
    char dist_out_fname[1024]; strcpy(dist_out_fname, DIST_OUT_FNAME);
    char list_fname[1024]; strcpy(list_fname, LIST_FNAME);
    char model_fname[1024]; strcpy(model_fname, MODEL_FNAME);  
    char base_path[1024]; strcpy(base_path, BASE_PATH);  
    compmode_t mode = DISTMATRIX;
    bool benchmark = false;
    bool largescale = false;

    // uhuh beware!
    gsl_set_error_handler (gsl_error_handler);

    while ((c = getopt(argc, argv, "D:N:P:S:U:d:l:L:s:c:m:t:vhTB:XC:")) != EOF) {
        switch (c) {
            case 'D':
                D = atoi(optarg);
                break;
			case 'N':
                N = atoi(optarg);
                break;
            case 'C':
                C = atoi(optarg);
                break;
			case 'S':
				S = atoi(optarg);
				break;
			case 'P':
				P = atoi(optarg);
				break;
		    case 'd':
                strcpy(dist_out_fname, optarg);
                break;
            case 'l':
                strcpy(list_fname, optarg);
                break;
            case 'L':
                strcpy(model_fname, optarg);
                break;
            case 's':
                start_image = atoi(optarg);
                break;
            case 'c':
                process_images = atoi(optarg);
                break;
            case 'm':
                mode = check_compmode(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            case 'T':
                benchmark = true;
                break;                
            case 'B':
                strcpy(base_path, optarg);
                if (strlen(base_path) > 0 && base_path[strlen(base_path)-1] != '/') strcat(base_path, "/");
                break;
            case 'X':
                // don't read model data in advance, load on demand
                largescale = true;
                break;
            case 'h':
            case '?': 
                usage();
                exit(0);                                           
            case 'v':
                verbose = 1;
                break;
        }
    }
    argc -= optind;
    argv += optind;

    if (N % S != 0) {
        fprintf(stderr, "%s: N=%d must be a multiple of S=%d, exit.\n", progname, N, S);
        exit(EXIT_FAILURE);
    }
    NS = N / S;

	/* the filelist holds all the data file names and hence we can determine how much memory we need */
    if (strlen(model_fname) == 0) 
	{
        FILE *listfile = 0;
        if (!open_models_filelist(list_fname, &num, &listfile)) {
            exit(EXIT_FAILURE);
        }

	    // allocate enough space for all images
		coeflist = (gsl_matrix **)malloc(num * sizeof(gsl_matrix *));
		RhoList = (gsl_matrix **)malloc(num * sizeof(gsl_matrix *));
		marginParamList = (gsl_matrix **)malloc(num *sizeof(gsl_matrix *));
		invRhoList =  (MatrixXf **)malloc(num *sizeof(MatrixXf *));
		cholDiagSumList = (double *)malloc(num * sizeof(double));
		modelname_list = (char **) malloc(num * sizeof(char *));
		nulist = (double *)malloc(num*sizeof(double));
		
        cnt = 0;
        if (verbose) fprintf(stderr, "%s: trying to read %d models ...\n", progname, num);
        char line[1024]; // whole line
	    while (fgets(line, sizeof(line), listfile) != NULL) {
        	char modelname[1024]; // name of model

            if (strlen(line) <= 1) continue;
	        line[strlen(line)-1] = '\0';

            modelname_list[cnt] = strdup(line);
		    /* read data files (DTCWT coefficients), for largescale experiments reading is deferred until needed */
            if (largescale) {
                coeflist[cnt] = NULL;

          		if (verbose)
                    fprintf(stderr, "%s: defer reading coefficient data (%d x %d), model %s\n", progname, N, D, line);
            }
            else {
                coeflist[cnt] = gsl_matrix_alloc(D, NS);
                read_model_data(base_path, modelname_list, coeflist[cnt], cnt, D, N, NS, S);

          		if (verbose)
                    fprintf(stderr, "%s: reading coefficient data (%d x %d), model %s\n", progname, N, D, line);
            }

			/* read covariance matrices */
		    sprintf(modelname, "%s%s.Rho", base_path, line);
		    FILE *mfile = fopen(modelname, "rb");
			if (!mfile) {
               	fprintf(stderr, "%s: failed to open '%s': %s\n", progname, modelname, strerror(errno));
               	exit(EXIT_FAILURE);
		    }
			
		    Rho = gsl_matrix_alloc(D,D);
		    if (gsl_matrix_fread(mfile, Rho) == GSL_EFAILED) {
				fprintf(stderr, "%s: failed to read '%s': %s\n", progname, modelname, strerror(errno));
				exit(EXIT_FAILURE);
		    }
		    RhoList[cnt] = Rho;
		    gsl_matrix *irho = compCholInv(Rho,D);
			invRhoList[cnt] = new MatrixXf((int)D, (int)D);
		    for (unsigned int c = 0; c < D; c++)
    		    for (unsigned int d = 0; d < D; d++)		    
                    (*invRhoList[cnt])(c,d) = gsl_matrix_get(irho, c, d);
			cholDiagSumList[cnt] = compCholDiag(Rho,D);
		    fclose(mfile);
				
			/* read nulist in case we have a student t copula */
			if (C == COPULA_TYPE_STUDENTT) {
				sprintf(modelname, "%s%s.nu", base_path,line);
				mfile = fopen(modelname, "rb");
				if (!mfile) {
		           	fprintf(stderr, "%s: failed to open '%s': %s\n", progname, modelname, strerror(errno));
		           	exit(EXIT_FAILURE);
				}	
				if(fread(&nulist[cnt],sizeof(double),1,mfile)!=1)	{
					fprintf(stderr, "%s: failed to read '%s': %s\n", progname, modelname, strerror(errno));
					exit(EXIT_FAILURE);
				}
				fclose(mfile);
			}

			/* read distribution parameters of marginal distributions - both Gamma and Weibull have 2 params */
		    sprintf(modelname, "%s%s.margins", base_path, line);
		    mfile = fopen(modelname, "rb");
			if (!mfile) {
               	fprintf(stderr, "%s: failed to open '%s': %s\n", progname, modelname, strerror(errno));
               	exit(EXIT_FAILURE);
		    }
			
		    marginParams = gsl_matrix_alloc(2,D);
		    if (gsl_matrix_fread(mfile, marginParams) == GSL_EFAILED) {
				fprintf(stderr, "%s: failed to read '%s': %s\n", progname, modelname, strerror(errno));
				exit(EXIT_FAILURE);
		    }
		    marginParamList[cnt] = marginParams;
		    fclose(mfile);
			cnt++;
		}
	    fclose(listfile);
    }
    if (!init_diststore(num, (mode == DISTMATRIX) ? MATRIX : TOPN)) {
        fprintf(stderr, "%s: failed to initialize distance store, exit.\n", progname);
        exit(EXIT_FAILURE);
    }

    metric_params_t metric_params;
    metric_params.data = coeflist; 		// list holding all the coefficient data
	metric_params.rhoList = RhoList;	// list of correlation matrices
	metric_params.marginList = marginParamList; 		// list of marginal distribution parameters
	metric_params.cholDiagSumList = cholDiagSumList;	// list of diagonal element sums
	metric_params.nulist = nulist;
	metric_params.invRhoList = invRhoList;
	metric_params.D = D; 		// dimensionality 
	metric_params.N = N; 		// sample length
	metric_params.NS = NS; 		// subsampled length	
	metric_params.P = P; 		// distribution specifier
	metric_params.S = S; 		// stepsize
	metric_params.C = C; 		// copula
    metric_params.base_path = base_path;
    metric_params.modelname_list = modelname_list;    
	
	if (!benchmark) {
        if (!start_threads(num_threads, num, start_image, process_images, ctbll_thread, &metric_params)) {
            fprintf(stderr, "%s: failed to start thread(s), exit.\n", progname);
            exit(EXIT_FAILURE);
        }

        if (!write_diststore(dist_out_fname)) {
            fprintf(stderr, "%s: failed to write distance store, exit.\n", progname);
            exit(EXIT_FAILURE);
        }
    }
    else {
        if (!start_benchmark(num_threads, num, mode, ctbll_thread, &metric_params, 10)) {
            fprintf(stderr, "%s: failed to run benchmark, exit.\n", progname);
            exit(EXIT_FAILURE);
        }
    }
    free_diststore();
	// free memory
    for (unsigned int i = 0; i < cnt; i++) {
		if (coeflist[i])
    		gsl_matrix_free(coeflist[i]);
		gsl_matrix_free(RhoList[i]);
		gsl_matrix_free(marginParamList[i]);
		delete invRhoList[i];
   		free(modelname_list[i]);
    }
	
	if (C == COPULA_TYPE_STUDENTT) {
		free(nulist);
	}
	free(cholDiagSumList);
	free(coeflist);
	free(RhoList);
	free(invRhoList);
	free(marginParamList);
	free(modelname_list);
	
	return EXIT_SUCCESS;
}

static double *getDiagonal(gsl_matrix *A,unsigned int dim) {	
	double *diag = (double *)malloc(dim * sizeof(double));
	for (unsigned int j = 0; j < dim; j++) {
		diag[j] = A->data[j*A->tda+j];
	}
	return diag;
}

static gsl_matrix * getLT(gsl_matrix *A, unsigned int dim) {
	gsl_matrix *LT = gsl_matrix_calloc(dim, dim);
	for (unsigned int i = 0; i < dim; i++){
		for (unsigned int j = 0; j <= i; j++) {
			LT->data[i * LT->tda + j] = A->data[i * A->tda + j];
		}
	}
	return LT;
}

static gsl_matrix *compCholInv(gsl_matrix *Rho, unsigned int D) {
	gsl_matrix *A = gsl_matrix_alloc(D,D);
	gsl_matrix_memcpy(A,Rho);
	gsl_linalg_cholesky_decomp(A);
	
	gsl_permutation *p = gsl_permutation_alloc(D);
	gsl_matrix *invA = gsl_matrix_calloc(D,D);
	gsl_matrix *LT = getLT(A,D);

	int signum = 0;
	gsl_linalg_LU_decomp(LT, p, &signum);
	gsl_linalg_LU_invert(LT, p, invA);
	
	gsl_permutation_free(p);
	gsl_matrix_free(A);
	gsl_matrix_free(LT);
	return invA;
}

static double compCholDiag(gsl_matrix *Rho, unsigned int D) {
	gsl_matrix *A = gsl_matrix_alloc(D, D);
	gsl_matrix_memcpy(A, Rho);

	gsl_linalg_cholesky_decomp(A);

	double *diag = getDiagonal(A, D);
	double logdiagsum = 0.0;
	for (unsigned int d = 0; d < D; d++) {
		logdiagsum += gsl_sf_log(diag[d]);
	}
	gsl_matrix_free(A);
	free(diag);
	return logdiagsum;
}

static inline double sqr(double x) {
	return x * x;
}

static double copulapdf_gaussian(gsl_matrix *unif, MatrixXf *irho, double cholDiagSum, unsigned int D, unsigned int NS) {
    MatrixXf nd((int)D, (int)NS);

	for (unsigned int c=0; c < D; c++) {
		double *up = &unif->data[c * unif->tda];
		for (unsigned int n=0; n < NS; n++) {
			nd(c, n) = norminvf(*up);	
			up++;
		}
	}
    double colval = ((*irho*nd).cwise().square() - nd.cwise().square()).sum();
	colval *= -0.5;
	colval -= NS*cholDiagSum;
	return colval/NS;
}

static double copulapdf_studentt(gsl_matrix *unif, MatrixXf *irho, double cholDiagSum, double nu, unsigned int D, unsigned int NS) {
    MatrixXf td((int)D, (int)NS);
	for (unsigned int c=0; c < D; c++) {
		double *up = &unif->data[c * unif->tda];
		for (unsigned int n=0; n < NS; n++) {
			td(c, n) = tinv(*up, nu);	
			up++;
		}
	}
    MatrixXf z = *irho*td; // D * NS
	const double f = lgamma((nu + D) / 2.0) + (D - 1) * lgamma(nu / 2.0) - D * lgamma((nu + 1.0) / 2.0) - cholDiagSum;
	MatrixXf nuVec = MatrixXf::Constant(1,NS,nu);
	MatrixXf nuMat = MatrixXf::Constant(D,NS,nu);
	MatrixXf mulMat = MatrixXf::Constant(D,NS,-(nu+1)/2.0);
	MatrixXf mulVec = MatrixXf::Constant(1,NS,-(nu+D)/2.0);
	MatrixXf numer = ((z.cwise().square().colwise().sum().cwise()/nuVec).cwise() + 1).cwise().log().cwise() * mulVec;
	MatrixXf denum = (((td.cwise().square().cwise() / nuMat).cwise() + 1).cwise().log().cwise() * mulMat).colwise().sum();
	double val = (numer-denum).sum();
	val += NS*f;
    return (1.0/NS)*val;
}

static double iter_weibull(gsl_matrix *coefQuery, gsl_matrix *marginCandidate, gsl_matrix *transformedQuery, unsigned int D, unsigned int NS) {
	double pdfval = 0.0;

	// iterate over dimensions
	for (unsigned int c=0; c < D; c++) {
		gsl_vector_view coefrow = gsl_matrix_row(coefQuery, c);
		gsl_vector_view paramcol = gsl_matrix_column(marginCandidate, c);

		double *rowdata = coefrow.vector.data;
		const float a = paramcol.vector.data[0*paramcol.vector.stride];
		const float b = paramcol.vector.data[1*paramcol.vector.stride];
		const float la = fmath::log(a);
		const float bm1 = b-1.0f;
		const float lbmla = fmath::log(b)-la;

        VectorXf cd(NS);
		for (unsigned int n = 0; n < NS; n ++) {
		    cd(n) = *rowdata;
			rowdata += coefrow.vector.stride;
		}
		
		VectorXf lvalmla = cd.cwise().log().cwise() - la;
	    VectorXf pvalab = (b * lvalmla).cwise().exp();
	    pdfval += NS*lbmla + ((bm1 * lvalmla) - pvalab).sum();
	    VectorXf tq = (-1.0 * pvalab).cwise().exp();
	    for (unsigned int n = 0; n < NS; n++) 	
		{
	        gsl_matrix_set(transformedQuery, c, n, 1.0 - tq(n));
		}
	}

	return pdfval;
}

static double iter_ggd(gsl_matrix *coefQuery, gsl_matrix *marginCandidate, gsl_matrix *transformedQuery, unsigned int D, unsigned int NS) {
	double pdfval = 0.0;
	double log2 = log(2.0);
	
	// iterate over dimensions
	for (unsigned int c=0; c < D; c++) {
		gsl_vector_view coefrow = gsl_matrix_row(coefQuery, c);
		gsl_vector_view paramcol = gsl_matrix_column(marginCandidate, c);

		double *rowdata = coefrow.vector.data;
		const double a = paramcol.vector.data[0*paramcol.vector.stride];
		const double b = paramcol.vector.data[1*paramcol.vector.stride];
		const double la = fmath::log(a);
		const double lb = fmath::log(b);
		const double pab = pow(a,b);
        const double g1b = lgamma(1.0/b);
		const double cterm = lb-log2 - la -g1b;
	
		//fprintf(stderr, "a=%0.5f, b=%0.5f, cdf=%0.5f, cterm=%0.5f\n", a,b,ggd_cdf(4.4,a,b),cterm);
		//getchar();
	
		VectorXf val(NS);
		for (unsigned int n = 0; n < NS; n ++) {
		    val(n) = *rowdata;
			rowdata += coefrow.vector.stride;
			transformedQuery->data[c * transformedQuery->tda + n] = ggd_cdf(val(n),a,b);	
		}
		VectorXf tmp = val.cwise().abs().cwise().pow(b)/pab;
		pdfval += NS*cterm - tmp.sum();
	}
	return pdfval;
}

static double iter_gamma(gsl_matrix *coefQuery, gsl_matrix *marginCandidate, gsl_matrix *transformedQuery, unsigned int D, unsigned int NS) {
	double pdfval = 0.0;

	// iterate over dimensions
	for (unsigned int c=0; c < D; c++) {
		gsl_vector_view coefrow = gsl_matrix_row(coefQuery, c);
		gsl_vector_view paramcol = gsl_matrix_column(marginCandidate, c);

		double *rowdata = coefrow.vector.data;
		const double a = paramcol.vector.data[0*paramcol.vector.stride];
		const double b = paramcol.vector.data[1*paramcol.vector.stride];
		const double ga = lgamma(a);
		const double lb = fmath::log(b);
		const double am1 = a-1.0f;		
		const double cterm = -ga-a*lb;

		VectorXf val(NS);
		for (unsigned int n = 0; n < NS; n ++) {
		    val(n) = *rowdata;
			//transformedQuery->data[c * transformedQuery->tda + n] = gsl_cdf_gamma_P(val(n),a,b);
			transformedQuery->data[c * transformedQuery->tda + n] = gammp(a,val(n)/b);
			rowdata += coefrow.vector.stride;
		}
		VectorXf cd_tmp =  -1.0f * (val/b) + (val.cwise().log()) * am1;
		pdfval += NS*cterm + cd_tmp.sum();
	}
	return pdfval;
}

// Likelihood Computation of each sample under each model (classic ML selection as proposed by Vasconcelos in Probabilistic CBIR
static void ctbll(gsl_matrix **data, gsl_matrix **rhoList, gsl_matrix **marginList, MatrixXf **invRhoList, double *cholDiagSumList, double *nulist, char *base_path, char **modelname_list, unsigned int D, unsigned int N, unsigned int NS, unsigned int P, unsigned int S, unsigned int C, unsigned int num, unsigned int start_image, unsigned int process_images) {
	gsl_matrix *transformedQuery = gsl_matrix_alloc(D,NS);
	gsl_matrix *coefdata = gsl_matrix_alloc(D, NS);
	
	for (unsigned i = start_image; i < start_image+process_images; i++) { // iterate over all images (queries)
		if (verbose)  {
            fprintf(stderr, "%s: processing query %d, %.2f %% done\n", progname, i, (i-start_image)*100.0/process_images);
		}
		// transform coefficients of the query image
		gsl_matrix *coefQuery = data[i];
		if (coefQuery == NULL)  {
            fprintf(stderr, "%s: reading coefficient data (%d x %d), model %s\n", progname, N, D, modelname_list[i]);
            read_model_data(base_path, modelname_list, coefdata, i, D, N, NS, S);
    		coefQuery = coefdata;
		}
		
		for (unsigned int j = 0; j < num; j++) { // iterate over all images (candidates)
			// Rho and margin parameters for the j-th database model
			gsl_matrix *marginCandidate = marginList[j];
			MatrixXf *invRhoCandidate = invRhoList[j];
			double cholDiagSumCandidate = cholDiagSumList[j];
			double pdfval = 0.0;

			switch (P) {
				case 0:
					pdfval = iter_weibull(coefQuery, marginCandidate, transformedQuery, D, NS);
					break;
				case 1:
					pdfval = iter_gamma(coefQuery, marginCandidate, transformedQuery, D, NS);
					break;
				case 2:
					pdfval = iter_ggd(coefQuery, marginCandidate, transformedQuery, D, NS);
					break;
				default:
					fprintf(stderr, "%s: invalid margin distribution '%d', exit.\n", progname, P);
					exit(EXIT_FAILURE);
			}
			
			double copula_ll;
			switch (C) {
			    case 0:
			        copula_ll = copulapdf_gaussian(transformedQuery, invRhoCandidate, cholDiagSumCandidate, D, NS);
			        break;
                case 1:
			        copula_ll = copulapdf_studentt(transformedQuery, invRhoCandidate, cholDiagSumCandidate, nulist[j], D, NS);
		            break;
                default:
					fprintf(stderr, "%s: invalid copula '%d', exit.\n", progname, C);
					exit(EXIT_FAILURE);
            }                
			        
			double totalLogLikelihood = copula_ll + (1.0/NS)*pdfval;
			
			if (isnan(totalLogLikelihood)) {
				store_dist(i, j, log(0));
			} else {
				// fprintf(stderr, "%0.5f\n", 0.5 + 1.0/M_PI * atan( (1+ 0.44)*tan((-1.1-0.44)/2)/(0.44-1)));
				// fprintf(stderr, "%0.5f\n", atan(-2.2));
				// fprintf(stderr, "%0.5f\n",gsl_sf_gamma_inc_Q(3.3,1.2));
				// fprintf(stderr, "%0.5f\n",gammq(3.3,1.2));
				// fprintf(stderr, "%0.5f\n",ggd_cdf(1.5,3,0.55));
				// fprintf(stderr, "%0.5f\n", totalLikelihood);
				// fprintf(stderr, "%0.5f\n", 1/exp(gammaln(2.0))* gammp(2.0,1.2/3.0));
				// fprintf(stderr, "%0.5f\n", gsl_cdf_gamma_P (1.2,2,3));	
				// fprintf(stderr, "%0.5f\n", log(gsl_ran_weibull_pdf(1.2,2,3)));
				// fprintf(stderr, "%0.5f\n", -gammaln(2.0)-2.0*log(3.0)-(1.2/3.0)+(2.0-1)*log(1.2));
				// fprintf(stderr, "%0.5f\n", gsl_sf_log(3)-log(2)+(3.0-1.0)*(log(1.2)-log(2.0))-pow((1.2/2.0),3.0));
				store_dist(i, j, totalLogLikelihood);
			}
		}		
	}

	gsl_matrix_free(transformedQuery);
	gsl_matrix_free(coefdata);
}

static void *ctbll_thread(void *thread_params) {
    if (!thread_params) 
        return 0;
        
    thread_params_t *p = (thread_params_t *) thread_params;
    metric_params_t *m = (metric_params_t *) p->metric_params;
    if (!m)
        return 0;
       
    if (verbose)
        fprintf(stderr, "%s: starting thread %d, processing images %d to %d (%d images)\n", progname, p->thread_num, p->start_image, p->start_image+p->process_images-1, p->process_images);
	
	ctbll(m->data, m->rhoList, m->marginList, m->invRhoList, m->cholDiagSumList, m->nulist, m->base_path, m->modelname_list, m->D, m->N, m->NS, m->P, m->S, m->C, p->num_images, p->start_image, p->process_images);
	return 0;
}

static double ggd_cdf(double x, double a, double b){
	if (x<=0) {
		return (plica(1.0/b,pow(-x/a,b))/(2.0*tgamma(1.0/b)));
	} else {
		return (1-plica(1.0/b, pow(x/a,b))/(2.0*tgamma(1.0/b)));
	}
}

static double plica(double a, double x) {
	return tgamma(a)*gammq(a,x);
}

/* Implementation of the inverse CDF function of the standardized Gaussian proposed
 by Peter Acklam; for more information visit his website http://home.online.no/~pjacklam/notes/invnorm/ */ 
static double norminv(double p) {
	double a1 = -39.69683028665376;
	double a2 = 220.9460984245205;
	double a3 = -275.9285104469687;
	double a4 = 138.3577518672690;
	double a5 =-30.66479806614716;
	double a6 = 2.506628277459239;
	
	double b1 = -54.47609879822406;
	double b2 = 161.5858368580409;
	double b3 = -155.6989798598866;
	double b4 = 66.80131188771972;
	double b5 = -13.28068155288572;
	
	double c1 = -0.007784894002430293;
	double c2 = -0.3223964580411365;
	double c3 = -2.400758277161838;
	double c4 = -2.549732539343734;
	double c5 = 4.374664141464968;
	double c6 = 2.938163982698783;
	
	double d1 = 0.007784695709041462;
	double d2 = 0.3224671290700398;
	double d3 = 2.445134137142996;
	double d4 = 3.754408661907416;
		
	//Define break-points.
	double p_low =  0.02425;
	double p_high = 1 - p_low;
	double q = 0.0;
	double x = 0.0;
	double r = 0.0;
	
	//Rational approximation for lower region.
	if (0 < p && p < p_low) {
		q = sqrt(-2*log(p));
		x = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
	}
	//Rational approximation for central region.
	if (p_low <= p && p <= p_high) {
		q = p - 0.5;
		r = q*q;
		x = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
	}
	//Rational approximation for upper region.
	if (p_high < p && p < 1) {
		q = sqrt(-2*log(1-p));
		x = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
	}
	return x;
}

static double norminvf(double p) {
	double a1 = -39.69683028665376;
	double a2 = 220.9460984245205;
	double a3 = -275.9285104469687;
	double a4 = 138.3577518672690;
	double a5 =-30.66479806614716;
	double a6 = 2.506628277459239;
	
	double b1 = -54.47609879822406;
	double b2 = 161.5858368580409;
	double b3 = -155.6989798598866;
	double b4 = 66.80131188771972;
	double b5 = -13.28068155288572;
	
	double c1 = -0.007784894002430293;
	double c2 = -0.3223964580411365;
	double c3 = -2.400758277161838;
	double c4 = -2.549732539343734;
	double c5 = 4.374664141464968;
	double c6 = 2.938163982698783;
	
	double d1 = 0.007784695709041462;
	double d2 = 0.3224671290700398;
	double d3 = 2.445134137142996;
	double d4 = 3.754408661907416;
		
	//Define break-points.
	double p_low =  0.02425;
	double p_high = 1.0 - p_low;
	double q = 0.0;
	double x = 0.0;
	double r = 0.0;
	
	//Rational approximation for lower region.
	if (0.0 < p && p < p_low) {
		q = sqrt(-2.0*fmath::log(p));
		x = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
	}
	//Rational approximation for central region.
	else if (p_low <= p && p <= p_high) {
		q = p - 0.5;
		r = q*q;
		x = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
	}
	//Rational approximation for upper region.
	else if (p_high < p && p < 1.0) {
		q = sqrt(-2.0*fmath::log(1.0-p));
		x = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
	}
	return x;
}

static double gammp(double a, double x) {
	double gamser,gammcf,gln;
	if (x < 0.0 || a <= 0.0) 
	{
		fprintf(stderr,"Invalid arguments in routine gammp");
		exit(EXIT_FAILURE);
	}
	if (x < (a+1.0)) 	
	{
		gser(&gamser,a,x,&gln);
		return gamser;
	} 
	else 
	{ 
		gcf(&gammcf,a,x,&gln);
		return 1.0-gammcf;
	}
}

static double gammq(double a, double x) {
	double gamser,gammcf,gln;
	if (x < 0.0 || a <= 0.0) {
		fprintf(stderr, "invalid argument to gammq\n");
		exit(-1);
	}
	if (x < (a+1.0)) { 
		gser(&gamser,a,x,&gln); 
		return 1.0-gamser;
	} else { 
		gcf(&gammcf,a,x,&gln); 
		return gammcf;
	}
}

static void gser(double *gamser, double a, double x, double *gln) {
	int n;
	double sum,del,ap;
	*gln=lgamma(a);
	if (x <= 0.0) {
		if (x < 0.0) 
			fprintf(stderr,"x less than 0 in routine gser");
		*gamser=0.0;
		return;
	} else {
		ap=a;
		del=sum=1.0/a;
		for (n=1;n<=ITMAX;n++) {
			++ap;
			del *= x/ap;
			sum += del;
			if (fabs(del) < fabs(sum)*EPS) {
				*gamser=sum*exp(-x+a*log(x)-(*gln));
				return;
			}
		}
		fprintf(stderr, "a too large, ITMAX too small in routine gser");
		exit(-1);
		return;
	}
}

static void gcf(double *gammcf, double a, double x, double *gln) {
	int i;
	double an,b,c,d,del,h;
	*gln=lgamma(a);
	b=x+1.0-a;
	c=1.0/FPMIN;
	d=1.0/b;
	h=d;
	for (i=1;i<=ITMAX;i++) {
		an = -i*(i-a);
		b += 2.0;
		d=an*d+b;
		if (fabs(d) < FPMIN) d=FPMIN;
		c=b+an/c;
		if (fabs(c) < FPMIN) c=FPMIN;
		d=1.0/d;
		del=d*c;
		h *= del;
		if (fabs(del-1.0) < EPS) break;
	}
	if (i > ITMAX) {
		fprintf(stderr,"a too large, ITMAX too small in gcf");
		exit(-1);
	}
	*gammcf=exp(-x+a*log(x)-(*gln))*h;
}

static double tinv(double p, double nu) {
    return gsl_cdf_tdist_Pinv(p,nu);
}

