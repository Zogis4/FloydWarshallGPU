/********************* fw_bench.h (header-only) *****************************/
#ifndef FW_BENCH_H
#define FW_BENCH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

#ifdef __CUDACC__
  #include <cuda_runtime.h>
#endif

/* Suas funcoes devem existir no TU que linka com este header */
#ifdef __cplusplus
extern "C" {
#endif
void gerarGrafoAleatorio(int **matriz, int n, int densidade);
void floydWarshallSequencial(int **dist, int n);
void floydWarshallOpenMP(int **dist, int n);
#ifdef __CUDACC__
void floydWarshallCUDA(int **dist, int n);
#endif
int  compararMatrizes(int **A, int **B, int n);
#ifdef __cplusplus
}
#endif

/* Temporizador host */
static inline double fw_now_secs_host(void){
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

/* Medir tempo CUDA (em ms) chamando uma funcao de execucao */
#ifdef __CUDACC__
typedef void (*fw_runner_t)(int **, int);
static float fw_cuda_time_ms_run(fw_runner_t kernel_run, int **dist, int n){
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel_run(dist, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms=0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}
#else
typedef void (*fw_runner_t)(int **, int);
#endif

/* Matriz util */
static int **fw_alloc_mat(int n){
    int **m = (int**)malloc(n*sizeof(int*));
    int *pool = (int*)malloc((size_t)n*(size_t)n*sizeof(int));
    for(int i=0;i<n;i++) m[i]=pool + (size_t)i*(size_t)n;
    return m;
}
static void fw_free_mat(int **m){
    if(!m) return;
    free(m[0]); free(m);
}
static void fw_copy_mat(int **dst, int **src, int n){
    memcpy(dst[0], src[0], (size_t)n*(size_t)n*sizeof(int));
}
static size_t fw_bytes_mat(int n){ return (size_t)n*(size_t)n*sizeof(int); }

/* VRAM info */
#ifdef __CUDACC__
static void fw_cuda_mem_info(size_t *free_b, size_t *total_b){
    size_t f=0,t=0; cudaMemGetInfo(&f,&t);
    if(free_b) *free_b=f; if(total_b) *total_b=t;
}
#endif

/* Wrappers uniformes */
static void fw_run_seq (int **d, int n){ floydWarshallSequencial(d,n); }
static void fw_run_omp (int **d, int n){ floydWarshallOpenMP(d,n); }
#ifdef __CUDACC__
static void fw_run_cuda(int **d, int n){ floydWarshallCUDA(d,n); }
#endif

/* Metricas */
typedef struct {
    const char* method;   /* "seq", "omp", "cuda" */
    int n;
    int density;
    int reps;
    double time_sec_mean;
    double time_sec_std;
    double speedup_vs_seq;
    double iters_per_sec; /* n^3 / time */
    double host_bytes;
    double device_bytes;
    size_t cuda_free_before;
    size_t cuda_free_after;
} fw_metrics_t;

static void fw_mean_std(const double *v, int m, double *mean, double *std){
    double s=0, q=0;
    for(int i=0;i<m;i++) s+=v[i];
    double mu = s/m;
    for(int i=0;i<m;i++){ double d=v[i]-mu; q+=d*d; }
    *mean = mu;
    *std  = (m>1)? sqrt(q/(m-1)) : 0.0;
}

static fw_metrics_t fw_bench_one(const char* method_name, fw_runner_t runner,
                                 int n, int density, int reps, int verify_with_seq){
    fw_metrics_t out;
    out.method = method_name;
    out.n = n;
    out.density = density;
    out.reps = reps;
    out.speedup_vs_seq = 0.0;
    out.iters_per_sec = 0.0;
    out.host_bytes = (double)fw_bytes_mat(n);
#ifdef __CUDACC__
    out.device_bytes = (strcmp(method_name,"cuda")==0) ? (double)fw_bytes_mat(n) : 0.0;
#else
    out.device_bytes = 0.0;
#endif
    out.cuda_free_before=0; out.cuda_free_after=0;

    int **base = fw_alloc_mat(n);
    gerarGrafoAleatorio(base, n, density);

    int **gold = NULL;
    if (verify_with_seq){
        gold = fw_alloc_mat(n);
        fw_copy_mat(gold, base, n);
        floydWarshallSequencial(gold, n);
    }

    /* warmup */
    {
        int **tmp = fw_alloc_mat(n);
        fw_copy_mat(tmp, base, n);
#ifdef __CUDACC__
        if (strcmp(method_name,"cuda")==0){
            (void)fw_cuda_time_ms_run(runner, tmp, n);
        } else
#endif
        {
            double t0=fw_now_secs_host(); runner(tmp,n); double t1=fw_now_secs_host(); (void)t0; (void)t1;
        }
        fw_free_mat(tmp);
    }

    double *times = (double*)malloc((size_t)reps*sizeof(double));
    for(int r=0;r<reps;r++){
        int **work = fw_alloc_mat(n);
        fw_copy_mat(work, base, n);

#ifdef __CUDACC__
        if (strcmp(method_name,"cuda")==0){
            size_t fb=0,tb=0; fw_cuda_mem_info(&fb,&tb); out.cuda_free_before = fb;
            float ms = fw_cuda_time_ms_run(runner, work, n);
            times[r] = ms/1000.0;
            fw_cuda_mem_info(&fb,&tb); out.cuda_free_after = fb;
        } else
#endif
        {
            double t0 = fw_now_secs_host();
            runner(work, n);
            double t1 = fw_now_secs_host();
            times[r] = t1 - t0;
        }

        if (verify_with_seq){
            int **chk = fw_alloc_mat(n);
            fw_copy_mat(chk, work, n);
            int mism = compararMatrizes(chk, gold, n);
            if (mism != 0){
                fprintf(stderr, "[WARN] %s n=%d dens=%d -> %d diferencas vs seq\n",
                        method_name, n, density, mism);
            }
            fw_free_mat(chk);
        }

        fw_free_mat(work);
    }

    fw_mean_std(times, reps, &out.time_sec_mean, &out.time_sec_std);
    free(times);

    double n3 = (double)n*(double)n*(double)n;
    if (out.time_sec_mean > 0) out.iters_per_sec = n3 / out.time_sec_mean;

    fw_free_mat(base);
    if (gold) fw_free_mat(gold);
    return out;
}

/* CSV */
static void fw_csv_header(FILE* f){
    fprintf(f, "method,n,density,reps,time_mean_s,time_std_s,speedup_vs_seq,"
               "iters_per_sec,host_bytes,device_bytes,cuda_free_before,cuda_free_after\n");
}
static void fw_csv_row(FILE* f, const fw_metrics_t* m){
    fprintf(f, "%s,%d,%d,%d,%.6f,%.6f,%.3f,%.3f,%.0f,%.0f,%zu,%zu\n",
        m->method, m->n, m->density, m->reps,
        m->time_sec_mean, m->time_sec_std, m->speedup_vs_seq,
        m->iters_per_sec, m->host_bytes, m->device_bytes,
        m->cuda_free_before, m->cuda_free_after);
}

/* Runner completo */
static int fw_run_full_benchmark(const int *Ns, int countN,
                                 const int *densities, int countD,
                                 int reps, int verify){
    FILE* f = fopen("fw_bench.csv","w");
    if(!f){ perror("fw_bench.csv"); return 1; }
    fw_csv_header(f);

    for(int di=0; di<countD; ++di){
        int dens = densities[di];

        for(int ni=0; ni<countN; ++ni){
            int n = Ns[ni];

            fw_metrics_t m_seq  = fw_bench_one("seq",  fw_run_seq,  n, dens, reps, verify);
            fw_csv_row(f, &m_seq);

            fw_metrics_t m_omp  = fw_bench_one("omp",  fw_run_omp,  n, dens, reps, verify);
            m_omp.speedup_vs_seq = (m_seq.time_sec_mean>0)? (m_seq.time_sec_mean / m_omp.time_sec_mean) : 0.0;
            fw_csv_row(f, &m_omp);

#ifdef __CUDACC__
            fw_metrics_t m_cuda = fw_bench_one("cuda", fw_run_cuda, n, dens, reps, verify);
            m_cuda.speedup_vs_seq = (m_seq.time_sec_mean>0)? (m_seq.time_sec_mean / m_cuda.time_sec_mean) : 0.0;
            fw_csv_row(f, &m_cuda);
#endif

            fflush(f);
            fprintf(stdout, "OK n=%d dens=%d | seq=%.3fs, omp=%.3fs"
#ifdef __CUDACC__
                    ", cuda=%.3fs"
#endif
                    "\n",
                    n, dens, m_seq.time_sec_mean, m_omp.time_sec_mean
#ifdef __CUDACC__
                    , m_cuda.time_sec_mean
#endif
            );
        }
    }

    fclose(f);
    printf("\nResultados salvos em fw_bench.csv\n");
    return 0;
}

/* helper de argv */
static int fw_has_flag(int argc, char**argv, const char*flag){
    for(int i=1;i<argc;i++) if(strcmp(argv[i],flag)==0) return 1;
    return 0;
}

#endif /* FW_BENCH_H */
/******************* end of fw_bench.h ***************************************/
