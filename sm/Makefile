VPATH = cephes

COPT = -g -O3 -Wall -fomit-frame-pointer -ffast-math -msse3 -mfpmath=sse -I. -I/Users/rkwitt/Software/gsl/include -L/Users/rkwitt/Software/gsl/lib -I/opt/local/include/eigen3
CXX = g++ $(COPT)
CC = gcc $(COPT)

LIBS = -L. -lstex -lgsl -lgslcblas -lpthread -lm

all: copll

clean:
	rm -f *.o copll libstex.a
	rm -rf copll.dSYM

libstex.a: common.o diststore.o
	$(AR) rcv $@ common.o diststore.o

copll: copll.cpp libstex.a
	$(CXX) -mfpmath=sse -msse3 -o $@ copll.cpp $(LIBS)
