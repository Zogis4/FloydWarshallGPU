// fw_common.h
#ifndef FW_COMMON_H
#define FW_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#ifdef _OPENMP
#  include <omp.h>
#endif

#define INF INT_MAX

#ifdef __cplusplus
extern "C" {
#endif

/* protótipos CPU */
void inicializarMatriz(int **matriz, int n);
void imprimirMatriz(int **matriz, int n, const char *rotulo);
void obterCoordenadas(int *coords, int numPares);
void imprimirCoordenadasSelecionadas(int **matriz, int n, const int *coords, int numPares, const char *rotulo);
void gerarGrafoAleatorio(int **matriz, int n, int densidade);
void lerGrafoFixo(int **matriz, int n, const char *arquivo);
void definirGrafoFixo(int **matriz, int n);
void floydWarshallSequencial(int **dist, int n);
void floydWarshallOpenMP(int **dist, int n);

/* protótipo CUDA */
void floydWarshallCUDA(int **dist, int n);

#ifdef __cplusplus
}
#endif

#endif /* FW_COMMON_H */
