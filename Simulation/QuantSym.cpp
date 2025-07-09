#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define MC_DEFAULT_STEPS 100000
#define MC_EQ_STEPS 50000

void parse_args(int argc, char* argv[], char** mode, int* N, double* Tmin, double* Tmax,
				int* Tsteps, unsigned long* MC_steps, char** input_file, char** output_file) {
	*mode = NULL; *N = 0; *Tmin = 0; *Tmax = 0; *Tsteps = 0;
	*MC_steps = MC_DEFAULT_STEPS;
	*input_file = NULL;
	*output_file = NULL;

	for (int i = 1; i < argc; i++) {
		if (strncmp(argv[i], "mode=", 5) == 0) *mode = argv[i] + 5;
		else if (strncmp(argv[i], "N=", 2) == 0) *N = atoi(argv[i] + 2);
		else if (strncmp(argv[i], "T_min=", 6) == 0) *Tmin = atof(argv[i] + 6);
		else if (strncmp(argv[i], "T_max=", 6) == 0) *Tmax = atof(argv[i] + 6);
		else if (strncmp(argv[i], "T_steps=", 8) == 0) *Tsteps = atoi(argv[i] + 8);
		else if (strncmp(argv[i], "MC_steps=", 9) == 0) *MC_steps = strtoul(argv[i] + 9, NULL, 10);
		else if (strncmp(argv[i], "input=", 6) == 0) *input_file = argv[i] + 6;
		else if (strncmp(argv[i], "output=", 7) == 0) *output_file = argv[i] + 7;
		else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); exit(1); }
	}

	if (!*mode || *N <= 0 || *Tsteps <= 0) {
		fprintf(stderr, "Usage: ./QuantSym mode=<exact|mc> N=<int> T_min=<double> T_max=<double> T_steps=<int> "
				"[MC_steps=<int>] [input=<file>] [output=<file>]\n");
		exit(1);
	}
}

void load_from_file(const char* fname, int* N, double*** J, double** h) {
	FILE* f = fopen(fname, "r");
	if (!f) { perror("fopen"); exit(1); }
	int fileN;
	fscanf(f, "%d", &fileN);
	if (fileN != *N) { fprintf(stderr, "File N (%d) != arg N (%d)\n", fileN, *N); exit(1); }
	*J = (double**)malloc((*N) * sizeof(double*));
	for (int i = 0; i < *N; i++) {
		(*J)[i] = (double*)malloc((*N) * sizeof(double));
	}
	for (int i = 0; i < *N; i++) {
		for (int j = 0; j < *N; j++) {
			fscanf(f, "%lf", &(*J)[i][j]);
		}
	}
	*h = (double*)malloc((*N) * sizeof(double));
	for (int i = 0; i < *N; i++) {
		fscanf(f, "%lf", &(*h)[i]);
	}
	fclose(f);
}

int main(int argc, char* argv[]) {
	char* mode;
	int N, Tsteps;
	double T_min, T_max;
	unsigned long MC_steps;
	char* input_file;
	char* output_file;

	parse_args(argc, argv, &mode, &N, &T_min, &T_max, &Tsteps, &MC_steps, &input_file, &output_file);

	FILE* output_fp = NULL;
	if (output_file) {
		output_fp = fopen(output_file, "w");
		if (!output_fp) {
			perror("Failed to open output file");
			return 1;
		}
		fprintf(output_fp, "T E_avg Heat_Capacity\n");
	}

	double** J = (double**)malloc(N * sizeof(double*));
	double* h = (double*)malloc(N * sizeof(double));

	srand(485876587); //special fixed for reproducibility of results

	if (input_file) {
		load_from_file(input_file, &N, &J, &h);
	}
	else {
		for (int i = 0; i < N; i++) {
			h[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
			J[i] = (double*)malloc(N * sizeof(double));
			for (int j = 0; j < N; j++) {
				if (i < j) J[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
				else J[i][j] = J[j][i];
			}
		}
	}

	double maxC = -INFINITY;
	double Tc = 0.0;
	for (int step = 0; step <= Tsteps; step++)
	{
		double T = T_min + (T_max - T_min) * step / Tsteps;
		double beta = 1.0 / T;
		double E_avg = 0.0;
		double E2_avg = 0.0;

		if (strcmp(mode, "exact") == 0) {
			unsigned long long max_s = 1ULL << N;
			double Z = 0.0;
			for (unsigned long long s = 0; s < max_s; s++) {
				int* sz = (int*)malloc(N * sizeof(int));
				for (int i = 0; i < N; i++) sz[i] = (s & (1ULL << i)) ? 1 : -1;
				double E = 0.0;

				for (int i = 0; i < N; i++) {
					for (int j = i + 1; j < N; j++) {
						E += J[i][j] * sz[i] * sz[j];
					}
				}
				free(sz);
				double w = exp(-beta * E);
				Z += w;
				E_avg += E * w;
				E2_avg += E * E * w;
			}
			E_avg /= Z;
			E2_avg /= Z;
		}
		else {
			int* sz = (int*)malloc(N * sizeof(int));
			for (int i = 0; i < N; i++) sz[i] = (rand() % 2) ? 1 : -1;
			// Equilibration
			for (unsigned long long it = 0; it < MC_EQ_STEPS; it++) {
				int i = rand() % N;
				double dE = 0.0;
				for (int j = 0; j < N; j++) if (i != j) dE += 2 * J[i][j] * sz[i] * sz[j];
				if (dE < 0 || exp(-beta * dE) >((double)rand() / RAND_MAX)) sz[i] *= -1;
			}
			// Sampling
			for (unsigned long long it = 0; it < MC_steps; it++) {
				int i = rand() % N;
				double dE = 0.0;
				for (int j = 0; j < N; j++) if (i != j) dE += 2 * J[i][j] * sz[i] * sz[j];
				if (dE < 0 || exp(-beta * dE) >((double)rand() / RAND_MAX)) sz[i] *= -1;
				double E = 0.0;
				for (int a = 0; a < N; a++) {
					for (int b = a + 1; b < N; b++) {
						E += J[a][b] * sz[a] * sz[b];
					}
				}
				E_avg += E;
				E2_avg += E * E;
			}
			free(sz);
			E_avg /= MC_steps;
			E2_avg /= MC_steps;
		}

		double C = (E2_avg - E_avg * E_avg) * beta * beta;
		if (C > maxC) {
			maxC = C;
			Tc = T;
		}
		printf("%f %f %f\n", T, E_avg, C);
		if (output_fp) {
			fprintf(output_fp, "%f %f %f\n", T, E_avg, C);
		}
	}

	printf("# Critical temperature Tc = %f, with C_max = %f\n", Tc, maxC);
	if (output_fp) {
		fprintf(output_fp, "# Critical temperature Tc = %f, with C_max = %f\n", Tc, maxC);
		fclose(output_fp);
	}

	// Cleanup
	for (int i = 0; i < N; i++) free(J[i]);
	free(J);
	free(h);
	_getch();
	return 0;
}
