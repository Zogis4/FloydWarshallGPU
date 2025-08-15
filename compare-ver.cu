// fw_bloques_check.cu
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#define INF INT_MAX
#define MAX_MISMATCH_PRINT 50
#define VERIFICAR 1

#ifndef TAM_BLOCO
#define TAM_BLOCO 32
#endif

// --- Prototipos ---
void inicializarMatriz(int **matriz, int n);
void imprimirMatriz(int **matriz, int n, const char *rotulo);
void obterCoordenadas(int *coords, int numPares);
void imprimirCoordenadasSelecionadas(int **matriz, int n, const int *coords, int numPares, const char *rotulo);
void floydWarshallSequencial(int **dist, int n);
void floydWarshallOpenMP(int **dist, int n);
void gerarGrafoAleatorio(int **matriz, int n, int densidade);
void lerGrafoFixo(int **matriz, int n, const char *arquivo);
void definirGrafoFixo(int **matriz, int n);
int compararMatrizes(int **A, int **B, int n);

// ---------------- CUDA: Floyd–Warshall com Tiling ----------------
#ifdef __CUDACC__
#ifndef CHK
#define CHK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA erro %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} }while(0)
#endif

__global__ void fase1_blocoDiagonal(int *d, int n, int kB) {
    __shared__ int tile[TAM_BLOCO][TAM_BLOCO];
    int tx = threadIdx.x, ty = threadIdx.y;
    int iG = kB*TAM_BLOCO + ty, jG = kB*TAM_BLOCO + tx;
    tile[ty][tx] = (iG<n && jG<n) ? d[iG*n + jG] : INF;
    __syncthreads();
    for (int kk=0; kk<TAM_BLOCO; ++kk){
        __syncthreads();
        int via1 = tile[ty][kk];
        int via2 = tile[kk][tx];
        if (via1!=INF && via2!=INF){
            long long soma = (long long)via1 + (long long)via2;
            if (soma < tile[ty][tx]) tile[ty][tx]=(int)soma;
        }
    }
    __syncthreads();
    if (iG<n && jG<n) d[iG*n + jG] = tile[ty][tx];
}

__global__ void fase2_linhaK(int *d, int n, int kB) {
    int jB = blockIdx.x;
    if (jB==kB) return;
    __shared__ int diag[TAM_BLOCO][TAM_BLOCO];
    __shared__ int alvo[TAM_BLOCO][TAM_BLOCO];
    int tx = threadIdx.x, ty = threadIdx.y;
    int iG = kB*TAM_BLOCO + ty, jG = jB*TAM_BLOCO + tx;
    int di = kB*TAM_BLOCO + ty, dj = kB*TAM_BLOCO + tx;
    diag[ty][tx] = (di<n && dj<n) ? d[di*n + dj] : INF;
    alvo[ty][tx] = (iG<n && jG<n) ? d[iG*n + jG] : INF;
    __syncthreads();
    for (int kk=0; kk<TAM_BLOCO; ++kk){
        __syncthreads();
        int via1 = diag[ty][kk];
        int via2 = alvo[kk][tx];
        if (via1!=INF && via2!=INF){
            long long soma = (long long)via1 + (long long)via2;
            if (soma < alvo[ty][tx]) alvo[ty][tx]=(int)soma;
        }
    }
    __syncthreads();
    if (iG<n && jG<n) d[iG*n + jG] = alvo[ty][tx];
}

__global__ void fase2_colunaK(int *d, int n, int kB) {
    int iB = blockIdx.y;
    if (iB==kB) return;
    __shared__ int diag[TAM_BLOCO][TAM_BLOCO];
    __shared__ int alvo[TAM_BLOCO][TAM_BLOCO];
    int tx = threadIdx.x, ty = threadIdx.y;
    int iG = iB*TAM_BLOCO + ty, jG = kB*TAM_BLOCO + tx;
    int di = kB*TAM_BLOCO + ty, dj = kB*TAM_BLOCO + tx;
    diag[ty][tx] = (di<n && dj<n) ? d[di*n + dj] : INF;
    alvo[ty][tx] = (iG<n && jG<n) ? d[iG*n + jG] : INF;
    __syncthreads();
    for (int kk=0; kk<TAM_BLOCO; ++kk){
        __syncthreads();
        int via1 = alvo[ty][kk];
        int via2 = diag[kk][tx];
        if (via1!=INF && via2!=INF){
            long long soma = (long long)via1 + (long long)via2;
            if (soma < alvo[ty][tx]) alvo[ty][tx]=(int)soma;
        }
    }
    __syncthreads();
    if (iG<n && jG<n) d[iG*n + jG] = alvo[ty][tx];
}

__global__ void fase3_restante(int *d, int n, int kB) {
    int iB = blockIdx.y, jB = blockIdx.x;
    if (iB==kB || jB==kB) return;
    __shared__ int col[TAM_BLOCO][TAM_BLOCO];
    __shared__ int row[TAM_BLOCO][TAM_BLOCO];
    __shared__ int c  [TAM_BLOCO][TAM_BLOCO];
    int tx = threadIdx.x, ty = threadIdx.y;
    int iG = iB*TAM_BLOCO + ty, jG = jB*TAM_BLOCO + tx;
    c[ty][tx] = (iG<n && jG<n) ? d[iG*n + jG] : INF;
    int jK = kB*TAM_BLOCO + tx;
    int iK = kB*TAM_BLOCO + ty;
    col[ty][tx] = (iG<n && jK<n) ? d[iG*n + jK] : INF;
    row[ty][tx] = (iK<n && jG<n) ? d[iK*n + jG] : INF;
    __syncthreads();
    for (int kk=0; kk<TAM_BLOCO; ++kk){
        __syncthreads();
        int via1 = col[ty][kk];
        int via2 = row[kk][tx];
        if (via1!=INF && via2!=INF){
            long long soma = (long long)via1 + (long long)via2;
            if (soma < c[ty][tx]) c[ty][tx]=(int)soma;
        }
    }
    __syncthreads();
    if (iG<n && jG<n) d[iG*n + jG] = c[ty][tx];
}

void floydWarshallCUDA(int **dist, int n) {
    size_t bytes = (size_t)n * (size_t)n * sizeof(int);
    int *h = (int*)malloc(bytes);
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) h[i*n+j] = dist[i][j];

    cudaDeviceProp p{}; CHK(cudaGetDeviceProperties(&p, 0));
    if (bytes > p.totalGlobalMem * 0.8) {
        fprintf(stderr, "[AVISO] Matriz (%.2f MB) > 80%% da VRAM (%.2f MB). Considere out-of-core.\n",
                bytes/1048576.0, p.totalGlobalMem/1048576.0);
    }

    int *d=nullptr; CHK(cudaMalloc(&d, bytes));
    CHK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));

    dim3 bloco(TAM_BLOCO, TAM_BLOCO);
    int nB = (n + TAM_BLOCO - 1)/TAM_BLOCO;

    for (int kB=0; kB<nB; ++kB) {
        fase1_blocoDiagonal<<<1, bloco>>>(d, n, kB);                  CHK(cudaDeviceSynchronize());
        fase2_linhaK      <<<dim3(nB,1), bloco>>>(d, n, kB);           CHK(cudaDeviceSynchronize());
        fase2_colunaK     <<<dim3(1,nB), bloco>>>(d, n, kB);           CHK(cudaDeviceSynchronize());
        fase3_restante    <<<dim3(nB,nB), bloco>>>(d, n, kB);          CHK(cudaDeviceSynchronize());
    }

    CHK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) dist[i][j] = h[i*n+j];
    cudaFree(d); free(h);
}
#endif

// ----------------- Programa "host" -----------------
int main(){
    int n, escolha, escolhaGrafo;
    int **dist;
    clock_t inicio, fim;
    double tempo;

    printf("Floyd–Warshall (Seq / OpenMP / CUDA-tiling) + Verificacao\n");
    printf("Digite o numero de vertices: ");
    scanf("%d",&n);

    dist=(int**)malloc(n*sizeof(int*));
    for(int i=0;i<n;i++) dist[i]=(int*)malloc(n*sizeof(int));

    printf("\nMetodo de entrada do grafo:\n1) Aleatorio\n2) Arquivo\n3) Fixo (5 nos)\n> ");
    scanf("%d",&escolhaGrafo);
    if (escolhaGrafo==1){
        int dens; printf("Densidade (0-100): "); scanf("%d",&dens);
        gerarGrafoAleatorio(dist,n,dens);
    } else if (escolhaGrafo==2){
        char arq[128]; printf("Arquivo: "); scanf("%127s", arq);
        lerGrafoFixo(dist,n,arq);
    } else if (escolhaGrafo==3){
        if (n!=5){ puts("Erro: fixo so com n=5."); return 1; }
        definirGrafoFixo(dist,n);
    } else { puts("Invalida."); return 1; }

    //imprimirMatriz(dist,n,"Matriz inicial");
    int num; printf("\nQuantos pares (i j) acompanhar: "); scanf("%d",&num);
    int* coords=(int*)malloc(sizeof(int)*2*num);
    obterCoordenadas(coords,num);
    imprimirCoordenadasSelecionadas(dist,n,coords,num,"Coordenadas iniciais");

    printf("\nImplementacao:\n1) Sequencial\n2) OpenMP\n");
#ifdef __CUDACC__
    printf("3) CUDA (tiling)\n");
#endif
    printf("> ");
    scanf("%d",&escolha);

#if VERIFICAR
    int **copia = (int**)malloc(n*sizeof(int*));
    for(int i=0;i<n;i++){ copia[i]=(int*)malloc(n*sizeof(int));
        for(int j=0;j<n;j++) copia[i][j]=dist[i][j]; }
#endif

    inicio=clock();
    switch (escolha){
        case 1: floydWarshallSequencial(dist,n); break;
        case 2: floydWarshallOpenMP(dist,n);     break;
#ifdef __CUDACC__
        case 3: floydWarshallCUDA(dist,n);       break;
#endif
        default: puts("Invalida."); return 1;
    }
    fim=clock();
    tempo = (double)(fim-inicio)/CLOCKS_PER_SEC;

    puts("\n=== RESULTADO ===");
    printf("Tempo (clock CPU): %.4f s\n", tempo);
    //imprimirMatriz(dist,n,"Matriz final");
    imprimirCoordenadasSelecionadas(dist,n,coords,num,"Coordenadas finais");

#if VERIFICAR
    floydWarshallSequencial(copia, n);
    int mism = compararMatrizes(dist, copia, n);
    if (mism==0) puts("\nOK GPU/Paralelo confere com CPU sequencial.");
    else         printf("\n%d diferencas.\n", mism);
    for(int i=0;i<n;i++) free(copia[i]); free(copia);
#endif

    free(coords);
    for(int i=0;i<n;i++) free(dist[i]); free(dist);
    return 0;
}

// ----------------- Utilitarios -----------------
void inicializarMatriz(int **m, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i][j] = (i == j) ? 0 : INF;
}

void imprimirMatriz(int **m, int n, const char *r) {
    printf("\n%s (%dx%d):\n", r, n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            (m[i][j] == INF) ? printf("  INF") : printf("%5d", m[i][j]);
        puts("");
    }
}

void obterCoordenadas(int *c, int num) {
    for (int k = 0; k < num; k++) {
        printf("Par %d: ", k + 1);
        scanf("%d %d", &c[2 * k], &c[2 * k + 1]);
    }
}

void imprimirCoordenadasSelecionadas(int **m, int n, const int *c, int num, const char *r) {
    char buf[32];
    printf("\n%s:\n", r);
    for (int k = 0; k < num; k++) {
        int i = c[2 * k], j = c[2 * k + 1];
        if (i >= n || j >= n) {
            printf("  (%2d,%2d): invalido (max %d)\n", i, j, n - 1);
            continue;
        }
        if (m[i][j] == INF)
            printf("  (%2d,%2d): INF\n", i, j);
        else {
            snprintf(buf, sizeof(buf), "%d", m[i][j]);
            printf("  (%2d,%2d): %s\n", i, j, buf);
        }
    }
}

void gerarGrafoAleatorio(int **m, int n, int d) {
    inicializarMatriz(m, n);
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j && (rand() % 100 < d))
                m[i][j] = 1 + rand() % 100;
}

void lerGrafoFixo(int **m, int n, const char *arq) {
    FILE *f = fopen(arq, "r");
    if (!f) {
        perror("arquivo");
        exit(1);
    }
    inicializarMatriz(m, n);
    int i, j, w;
    while (fscanf(f, "%d %d %d", &i, &j, &w) == 3) {
        if (i < n && j < n) m[i][j] = w;
    }
    fclose(f);
}

void definirGrafoFixo(int **m, int n) {
    int g[5][5] = {
        {0, 4, INF, 5, INF},
        {INF, 0, 1, INF, 6},
        {2, INF, 0, 3, INF},
        {INF, INF, 1, 0, 2},
        {1, INF, INF, 4, 0}
    };
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i][j] = g[i][j];
}

void floydWarshallSequencial(int **d, int n) {
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (d[i][k] != INF && d[k][j] != INF && d[i][j] > d[i][k] + d[k][j])
                    d[i][j] = d[i][k] + d[k][j];
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

int compararMatrizes(int **A, int **B, int n) {
    char ba[32], bb[32];
    int mism = 0, shown = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int a = A[i][j], b = B[i][j];
            if (a != b) {
                if (shown < MAX_MISMATCH_PRINT) {
                    if (a == INF) snprintf(ba, sizeof(ba), "INF");
                    else snprintf(ba, sizeof(ba), "%d", a);
                    if (b == INF) snprintf(bb, sizeof(bb), "INF");
                    else snprintf(bb, sizeof(bb), "%d", b);
                    printf("diff (%d,%d): A=%s B=%s\n", i, j, ba, bb);
                    shown++;
                }
                mism++;
            }
        }
    }
    if (mism > shown) printf("... e mais %d diferencas\n", mism - shown);
    return mism;
}
