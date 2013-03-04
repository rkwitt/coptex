coptex
======

Modeling of wavelet coefficient statistics using copulas (with application to image retrieval)

Requirements
------------

- GNU Scientific Library (GSL, version 1.14 and higher), download from http://www.gnu.org/software/gsl/
- Eigen3, download from http://eigen.tuxfamily.org

Compilation (C\+\+ part of the code)
------------------------------------

Lets assume that you installed GSL under '/Software/gsl', the Eigen include files are 
located at '/usr/include/eigen3' and you checked out the code at '/tmp/coptex'. 

**Note:** The model estimation part of the code is written in MATLAB, the similarity
measurement part is in C/C\+\+ (under the directory 'sm').

Unfortunately, the code is not CMake'ified yet, so you have to change the 'Makefile'. You should
only need to modify the line
``` bash
COPT = -g -O3 -Wall -fomit-frame-pointer -ffast-math -msse3 -mfpmath=sse -I. -I/Users/rkwitt/Software/gsl/include -L/Users/rkwitt/Software/gsl/lib -I/opt/local/include/eigen3
```
by changing the elements
``` bash
-I/Users/rkwitt/Software/gsl/include
-I/Software/gsl/include
-I/opt/local/include/eigen3
```
to
``` bash
-I/Software/gsl/include
-L/Software/gsl/lib
-I/usr/include/eigen3
```
Then run 'make'.

Testing
-------

To test the code, start MATLAB, and 'cd' into the base directory of coptex, i.e.,
```matlab
cd '/tmp/coptex'
```
Next, edit the 'example.m' file and change

```matlab
basedir = '/Users/rkwitt/Remote/coptex-read-only';
dumpdir = '/tmp/x';
```
to
```matlab
basedir = '/tmp/coptex';
dumpdir = '/tmp/testdata';
```
where '/tmp/testdata' will be created when running
```matlab
example
```
This will run a DT-CWT (Dual-Tree Complex Wavelet Transform) decomposition 
(using Nick Kingsbury's MATLAB code) on all test images and model the wavelet 
coefficient distributions on the third decomposition level by a *Gaussian 
copula with Weibull margins*. All estimated parameters will be stored under 
'/tmp/testdata' and a file list will be created that holds the basenames of 
all estimated models.

Finally, you can compute a similarity matrix for all testimages, using 
the estimated models, by running the 'copll' binary, i.e.,

```bash
cd /tmp/testdata
/tmp/coptex/sm/copll -B /tmp/testdata -D 18 -N 256 -l /tmp/testdata/filelist.txt -d /tmp/D.bin
```
'-D 18' means that we have 18 subbands on one level of a DT-CWT and '-N 256' says that there
are 256 coefficients (16x16) on that level. For further details, see
```bash
/tmp/coptex/sm/copll --help
```

References
----------
Please cite
```bibtex
@article{Kwitt11e,
  author = {R. Kwitt and P. Meerwald and A. Uhl},
  title = {Efficient Texture Image Retrieval Using Copulas in a Bayesian Framework},
  journal = {IEEE Transactions on Image Processing},
  year = {2011},
  volume = {20},
  number = {7},
  pages = {2063-2077}
}
```
if you use this code for your research.









