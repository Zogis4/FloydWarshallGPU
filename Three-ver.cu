#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include "fw_bench.h"
#define INF INT_MAX // Valor que representa infinito

// Protótipos das funções
void inicializarMatriz(int **matriz, int n);
void imprimirMatriz(int **matriz, int n, const char *rotulo);
void obterCoordenadas(int *coords, int numPares);
void imprimirCoordenadasSelecionadas(int **matriz, int n, const int *coords, int numPares, const char *rotulo);
void floydWarshallSequencial(int **dist, int n);
void floydWarshallOpenMP(int **dist, int n);
void gerarGrafoAleatorio(int **matriz, int n, int densidade);
void lerGrafoFixo(int **matriz, int n, const char *arquivo);
void definirGrafoFixo(int **matriz, int n);

// Funções CUDA
#ifdef __CUDACC__
void floydWarshallCUDA(int **dist, int n);
__global__ void floydWarshallKernel(int *dist, int n, int k);
#endif

int main() {
    int n, escolha, escolhaGrafo;
    int **dist;
    clock_t inicio, fim;
    double tempo_cpu_usado;

    printf("Implementação do Algoritmo de Floyd-Warshall\n");
    printf("Digite o número de vértices: ");
    scanf("%d", &n);

    // Alocar memória para a matriz de distâncias
    dist = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        dist[i] = (int *)malloc(n * sizeof(int));
    }

    printf("\nEscolha o método de entrada do grafo:\n");
    printf("1. Gerar grafo aleatório\n");
    printf("2. Ler grafo fixo de arquivo\n");
    printf("3. Usar grafo fixo embutido (apenas para 5 vértices)\n");
    printf("Digite sua escolha: ");
    scanf("%d", &escolhaGrafo);

    if (escolhaGrafo == 1) {
        int densidade;
        printf("Digite a densidade do grafo (porcentagem de arestas, 0-100): ");
        scanf("%d", &densidade);
        gerarGrafoAleatorio(dist, n, densidade);
    } else if (escolhaGrafo == 2) {
        char nomeArquivo[100];
        printf("Digite o nome do arquivo: ");
        scanf("%s", nomeArquivo);
        lerGrafoFixo(dist, n, nomeArquivo);
    } else if (escolhaGrafo == 3) {
        if (n != 5) {
            printf("Erro: Grafo fixo embutido só funciona para 5 vértices\n");
            return 1;
        }
        definirGrafoFixo(dist, n);
    } else {
        printf("Escolha inválida\n");
        return 1;
    }

    // Imprimir estado inicial
    printf("\n=== GRAFO INICIAL ===\n");
    //imprimirMatriz(dist, n, "Matriz Completa");

    // Obter e imprimir coordenadas selecionadas
    int numCoordenadas;
    printf("\nDigite o número de pares de coordenadas para acompanhar: ");
    scanf("%d", &numCoordenadas);
    int *coordenadas = (int *)malloc(2 * numCoordenadas * sizeof(int));
    obterCoordenadas(coordenadas, numCoordenadas);
    imprimirCoordenadasSelecionadas(dist, n, coordenadas, numCoordenadas, "Coordenadas Iniciais");

    // Escolher implementação
    printf("\nEscolha a implementação:\n");
    printf("1. Sequencial\n");
    printf("2. Paralela com OpenMP\n");
#ifdef __CUDACC__
    printf("3. Paralela com CUDA\n");
#endif
    printf("Digite sua escolha: ");
    scanf("%d", &escolha);

    inicio = clock();
    switch (escolha) {
        case 1:
            floydWarshallSequencial(dist, n);
            break;
        case 2:
            floydWarshallOpenMP(dist, n);
            break;
#ifdef __CUDACC__
        case 3:
            floydWarshallCUDA(dist, n);
            break;
#endif
        default:
            printf("Escolha inválida\n");
            return 1;
    }
    fim = clock();
    tempo_cpu_usado = ((double)(fim - inicio)) / CLOCKS_PER_SEC;

    // Imprimir resultados
    printf("\n=== RESULTADOS FINAIS ===\n");
    printf("Tempo de execução: %.4f segundos\n", tempo_cpu_usado);
    //imprimirMatriz(dist, n, "Matriz de Resultados");
    imprimirCoordenadasSelecionadas(dist, n, coordenadas, numCoordenadas, "Coordenadas Finais");

    // Liberar memória
    free(coordenadas);
    for (int i = 0; i < n; i++) {
        free(dist[i]);
    }
    free(dist);

    return 0;
}

void inicializarMatriz(int **matriz, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matriz[i][j] = (i == j) ? 0 : INF;
        }
    }
}

void imprimirMatriz(int **matriz, int n, const char *rotulo) {
    printf("\n%s (%dx%d):\n", rotulo, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (matriz[i][j] == INF) {
                printf("  INF");
            } else {
                printf("%5d", matriz[i][j]);
            }
        }
        printf("\n");
    }
}

void obterCoordenadas(int *coords, int numPares) {
    printf("Digite %d pares de coordenadas (i j):\n", numPares);
    for (int k = 0; k < numPares; k++) {
        printf("Par %d: ", k+1);
        scanf("%d %d", &coords[2*k], &coords[2*k+1]);
    }
}

void imprimirCoordenadasSelecionadas(int **matriz, int n, const int *coords, int numPares, const char *rotulo) {
    printf("\n%s:\n", rotulo);
    for (int k = 0; k < numPares; k++) {
        int i = coords[2*k];
        int j = coords[2*k+1];
        if (i >= n || j >= n) {
            printf("  (%2d,%2d): Inválido (índice máximo é %d)\n", i, j, n-1);
            continue;
        }
        printf("  (%2d,%2d): ", i, j);
        if (matriz[i][j] == INF) {
            printf("INF\n");
        } else {
            printf("%d\n", matriz[i][j]);
        }
    }
}

void gerarGrafoAleatorio(int **matriz, int n, int densidade) {
    inicializarMatriz(matriz, n);
    srand(time(NULL));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && rand() % 100 < densidade) {
                matriz[i][j] = 1 + rand() % 100;
            }
        }
    }
}

void lerGrafoFixo(int **matriz, int n, const char *arquivo) {
    FILE *file = fopen(arquivo, "r");
    if (file == NULL) {
        perror("Erro ao abrir o arquivo");
        exit(EXIT_FAILURE);
    }

    inicializarMatriz(matriz, n);

    int i, j, peso;
    while (fscanf(file, "%d %d %d", &i, &j, &peso) == 3) {
        if (i >= n || j >= n) {
            printf("Aviso: Aresta (%d,%d) excede tamanho da matriz %d\n", i, j, n);
            continue;
        }
        matriz[i][j] = peso;
    }

    fclose(file);
}

void definirGrafoFixo(int **matriz, int n) {
    int grafoFixo[5][5] = {
        {0, 4, INF, 5, INF},
        {INF, 0, 1, INF, 6},
        {2, INF, 0, 3, INF},
        {INF, INF, 1, 0, 2},
        {1, INF, INF, 4, 0}
    };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matriz[i][j] = grafoFixo[i][j];
        }
    }
}

void floydWarshallSequencial(int **dist, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF && 
                    dist[i][j] > dist[i][k] + dist[k][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

void floydWarshallOpenMP(int **dist, int n) {
    for (int k = 0; k < n; k++) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int dik = dist[i][k];
                int kdj = dist[k][j];
                if (dik != INF && kdj != INF) {
                    long long via = (long long)dik + (long long)kdj;
                    if (via < dist[i][j]) dist[i][j] = (int)via;
                }
            }
        }
    }
}


#ifdef __CUDACC__
void floydWarshallCUDA(int **dist, int n) {
    int *d_dist;
    size_t tamanho = n * n * sizeof(int);

    // Achatar a matriz para enviar ao CUDA
    int *matrizLinear = (int *)malloc(tamanho);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrizLinear[i * n + j] = dist[i][j];
        }
    }

    // Alocar memória no dispositivo
    cudaMalloc(&d_dist, tamanho);
    cudaMemcpy(d_dist, matrizLinear, tamanho, cudaMemcpyHostToDevice);

    // Configurar kernel
    dim3 bloco(16, 16);
    dim3 grade((n + bloco.x - 1) / bloco.x, 
               (n + bloco.y - 1) / bloco.y);

    // Executar Floyd-Warshall na GPU
    for (int k = 0; k < n; k++) {
        floydWarshallKernel<<<grade, bloco>>>(d_dist, n, k);
        cudaDeviceSynchronize();
    }

    // Copiar resultado de volta para a CPU
    cudaMemcpy(matrizLinear, d_dist, tamanho, cudaMemcpyDeviceToHost);

    // Desfazer o achatamento da matriz
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dist[i][j] = matrizLinear[i * n + j];
        }
    }

    // Liberar memória
    free(matrizLinear);
    cudaFree(d_dist);
}

__global__ void floydWarshallKernel(int *dist, int n, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n && j < n) {
        int ik = i * n + k;
        int kj = k * n + j;
        int ij = i * n + j;

        if (dist[ik] != INF && dist[kj] != INF && dist[ij] > dist[ik] + dist[kj]) {
            dist[ij] = dist[ik] + dist[kj];
        }
    }
}
#endif
