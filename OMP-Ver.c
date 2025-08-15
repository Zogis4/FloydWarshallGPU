// floyd_warshall_openmp.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define INF 1000000
#define MAX_NODES 50
#define MIN_NODES 4

void generateRandomGraph(int **dist, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                dist[i][j] = 0;
            } else {
                int has_edge = rand() % 2;
                dist[i][j] = has_edge ? (rand() % 20 + 1) : INF;
            }
        }
    }
}

void floydWarshall(int **dist, int n) {
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

void printGraph(int **dist, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (dist[i][j] == INF)
                printf("INF ");
            else
                printf("%3d ", dist[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int n = rand() % (MAX_NODES - MIN_NODES + 1) + MIN_NODES;

    int **dist = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        dist[i] = (int *)malloc(n * sizeof(int));
    }

    generateRandomGraph(dist, n);

    printf("Original graph (%d nodes):\n", n);
    printGraph(dist, n);

    double start = omp_get_wtime();
    floydWarshall(dist, n);
    double end = omp_get_wtime();

    printf("\nShortest distances:\n");
    printGraph(dist, n);
    printf("\nExecution time: %.6f seconds\n", end - start);

    for (int i = 0; i < n; i++) free(dist[i]);
    free(dist);

    return 0;
}
