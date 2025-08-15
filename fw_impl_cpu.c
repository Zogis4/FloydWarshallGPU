// fw_impl_cpu.c
#include "fw_common.h"

void inicializarMatriz(int **matriz, int n) {
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            matriz[i][j] = (i == j) ? 0 : INF;
}

void imprimirMatriz(int **matriz, int n, const char *rotulo) {
    int i, j;
    printf("\n%s (%dx%d):\n", rotulo, n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (matriz[i][j] == INF) printf("  INF");
            else                     printf("%5d", matriz[i][j]);
        }
        printf("\n");
    }
}

void obterCoordenadas(int *coords, int numPares) {
    int k;
    printf("Digite %d pares de coordenadas (i j):\n", numPares);
    for (k = 0; k < numPares; k++) {
        printf("Par %d: ", k+1);
        scanf("%d %d", &coords[2*k], &coords[2*k+1]);
    }
}

void imprimirCoordenadasSelecionadas(int **matriz, int n, const int *coords, int numPares, const char *rotulo) {
    int k;
    printf("\n%s:\n", rotulo);
    for (k = 0; k < numPares; k++) {
        int i = coords[2*k], j = coords[2*k+1];
        if (i >= n || j >= n) {
            printf("  (%2d,%2d): Invalido (indice max %d)\n", i, j, n-1);
            continue;
        }
        printf("  (%2d,%2d): ", i, j);
        if (matriz[i][j] == INF) printf("INF\n");
        else                     printf("%d\n", matriz[i][j]);
    }
}

void gerarGrafoAleatorio(int **matriz, int n, int densidade) {
    int i, j;
    inicializarMatriz(matriz, n);
    srand((unsigned)time(NULL));
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            if (i != j && (rand() % 100) < densidade)
                matriz[i][j] = 1 + rand() % 100;
}

void lerGrafoFixo(int **matriz, int n, const char *arquivo) {
    FILE *file = fopen(arquivo, "r");
    int i, j, peso;
    if (!file) { perror("arquivo"); exit(EXIT_FAILURE); }
    inicializarMatriz(matriz, n);
    while (fscanf(file, "%d %d %d", &i, &j, &peso) == 3) {
        if (i < n && j < n) matriz[i][j] = peso;
    }
    fclose(file);
}

void definirGrafoFixo(int **matriz, int n) {
    int i, j;
    int g[5][5] = {
        {0, 4, INF, 5, INF},
        {INF, 0, 1, INF, 6},
        {2, INF, 0, 3, INF},
        {INF, INF, 1, 0, 2},
        {1, INF, INF, 4, 0}
    };
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            matriz[i][j] = g[i][j];
}

void floydWarshallSequencial(int **dist, int n) {
    int k, i, j;
    for (k = 0; k < n; k++)
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                if (dist[i][k] != INF && dist[k][j] != INF &&
                    dist[i][j] > dist[i][k] + dist[k][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
}

/* Versao OpenMP portavel para MSVC: sem collapse(2) e com variaveis declaradas fora do for */
void floydWarshallOpenMP(int **d, int n) {
    int k;
    for (k = 0; k < n; ++k) {
        int idx; /* declarar fora do for para o MSVC nao reclamar */
        #pragma omp parallel for schedule(static)
        for (idx = 0; idx < n * n; ++idx) {
            int i = idx / n;
            int j = idx % n;

            int dik = d[i][k];
            int kdj = d[k][j];
            if (dik != INF && kdj != INF) {
                long long via = (long long)dik + (long long)kdj;
                if (via < d[i][j]) d[i][j] = (int)via;
            }
        }
    }
}
