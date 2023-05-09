#ifndef KMEANS_H_
#define KMEANS_H_

#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <omp.h>


/* k-means with using OpenMP here */
double getEvDist(const double *x1, const double *x2, const int m);
void elementaryAutoscaling(double *x, const int n, const int m, const int id);
void autoscalingOpenMP(double *x, const int n, const int m);
int getCluster(const double *x, const double *c, const int m, const int k);
void detCoresOpenMP(const double *x, double *c, const int *sn, const int k, const int m);
void detStartSplittingSimple(const double *x, const double *c, int *y, int *nums, const int m, const int k, const int id);
void detStartSplittingOpenMP(const double *x, const double *c, int *y, int *nums, const int n, const int m, const int k);
void simpleCalcCores(const double *x, double *c, const int *res, const int *nums, const int m, const int id);
void calcCoresOpenMP(const double *x, double *c, const int *y, const int *nums, const int n, const int m);
int simpleCheckSplitting(const double *x, const double *c, int *res, int *nums, const int m, const int k, const int id);
char checkSplittingOpenMP(const double *x, const double *c, int *res, int *nums, const int n, const int m, const int k);
char constr(const int *y, const int val, const int s);
void startCoreNums(int *y, const int k, const int n);
void kmeansOpenMP(const double *X, int *y, const int n, const int m, const int k);

/*k-means without using OpenMP*/
void autoscaling(double *x, const int n, const int m);
void detCores(const double *x, double *c, const int *sn, const int k, const int m);
void detStartSplitting(const double *x, const double *c, int *y, int *nums, const int n, const int m, const int k);
void calcCores(const double *x, double *c, const int *res, const int *nums, const int n, const int m);
char checkSplitting(const double *x, const double *c, int *res, int *nums, const int n, const int m, const int k);
void kmeans(const double *X, int *y, const int n, const int m, const int k);

#endif
