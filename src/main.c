#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "help.h"
#include "kmeans.h"


int main (int argc, char *argv[]) {
	if (argc < 7) {
		puts("Not enough parameters...");
		exit(1);
	}
	const int n = atoi(argv[1]), m = atoi(argv[2]), k = atoi(argv[3]);
	if ((n < 0) || (m < 0) || (k < 0) || (k > n)) {
		puts("Value of parameters is incorrect...");
		exit(1);
	}
	double *x = (double*)malloc(n * m * sizeof(double));
	fscanfData(argv[4], x, n * m);
	int *y1 = (int*)malloc(n * sizeof(int));
	int *y2 = (int*)malloc(n * sizeof(int));
	double t1, t2;
	t1 = omp_get_wtime();
	kmeansOpenMP(x, y1, n, m, k);
	t1 = omp_get_wtime() - t1;
	t2 = omp_get_wtime();
	kmeans(x, y2, n, m, k);
	t2 = omp_get_wtime() - t2;
	printf("Speedup: %.5lf\n", t2 / t1);
	fprintfTime(argv[5], t1);
	fprintfTime(argv[6], t2);
	if (argc > 7) {
		fprintfResults(argv[7], y1, n, m, k);
		double a1, a2;
		if (argc > 8) {
			int *ideal = (int*)malloc(n * sizeof(int));
			fscanfSplitting(argv[8], ideal, n);
			a1 = getAccuracy(ideal, y1, n);
			a2 = getAccuracy(ideal, y2, n);
			free(ideal);
			printf("Accuracy of clustering with using OpenMP = %.5lf;\nAccuracy of clustering without using OpenMP = %.5lf;\n", a1, a2);
		}
	}
	if (check(&y1[0], &y2[0], n)) {
		puts("OK...");
	}
	free(y1);
	free(y2);
	free(x);
	return 0;
}


