#ifndef DISTSTORE_H
#define DISTSTORE_H

typedef enum {MATRIX, TOPN, BOTTOMN, UNKNOWN} distmode_t;

int init_diststore(unsigned int dim, distmode_t mode);
void store_dist(unsigned int i, unsigned int j, double dist);
int write_diststore(const char *fname);
void free_diststore();

#endif /* DISTSTORE_H */
