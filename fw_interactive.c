// fw_interactive.c
#include "fw_common.h"

int main(void) {
    int n, escolha, escolhaGrafo;
    int **dist;
    clock_t inicio, fim;
    double tempo;

    printf("Implementacao do Algoritmo de Floyd-Warshall\n");
    printf("Digite o numero de vertices: ");
    if (scanf("%d", &n) != 1) return 1;

    dist = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) dist[i] = (int*)malloc(n * sizeof(int));

    printf("\nEscolha o metodo de entrada do grafo:\n");
    printf("1. Gerar grafo aleatorio\n");
    printf("2. Ler grafo fixo de arquivo\n");
    printf("3. Usar grafo fixo embutido (n=5)\n");
    printf("> ");
    if (scanf("%d", &escolhaGrafo) != 1) return 1;

    if (escolhaGrafo == 1) {
        int densidade;
        printf("Densidade (0-100): ");
        scanf("%d", &densidade);
        gerarGrafoAleatorio(dist, n, densidade);
    } else if (escolhaGrafo == 2) {
        char nomeArquivo[256];
        printf("Arquivo: ");
        scanf("%255s", nomeArquivo);
        lerGrafoFixo(dist, n, nomeArquivo);
    } else if (escolhaGrafo == 3) {
        if (n != 5) { printf("Erro: fixo so com n=5\n"); return 1; }
        definirGrafoFixo(dist, n);
    } else {
        printf("Opcao invalida\n"); return 1;
    }

    int num;
    printf("\nQuantos pares (i j) acompanhar: ");
    scanf("%d", &num);
    int *coords = (int*)malloc(sizeof(int)*2*num);
    obterCoordenadas(coords, num);
    imprimirCoordenadasSelecionadas(dist, n, coords, num, "Coordenadas iniciais");

    printf("\nEscolha a implementacao:\n");
    printf("1. Sequencial\n");
    printf("2. OpenMP\n");
#ifdef __CUDACC__
    printf("3. CUDA\n");
#endif
    printf("> ");
    scanf("%d", &escolha);

    inicio = clock();
    switch (escolha) {
        case 1: floydWarshallSequencial(dist, n); break;
        case 2: floydWarshallOpenMP(dist, n);     break;
#ifdef __CUDACC__
        case 3: floydWarshallCUDA(dist, n);       break;
#endif
        default: printf("Opcao invalida\n"); return 1;
    }
    fim = clock();
    tempo = (double)(fim - inicio) / CLOCKS_PER_SEC;

    printf("\n=== RESULTADO ===\n");
    printf("Tempo (CPU clock): %.4f s\n", tempo);
    imprimirCoordenadasSelecionadas(dist, n, coords, num, "Coordenadas finais");

    free(coords);
    for (int i = 0; i < n; i++) free(dist[i]);
    free(dist);
    return 0;
}
