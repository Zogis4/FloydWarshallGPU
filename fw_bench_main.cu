// fw_bench_main.cu  (compatível com C++14: sem structured bindings)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <utility>
#include <cmath>

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <cuda_runtime.h>


#include "fw_common.h"


// ---------- CLI helpers ----------
static bool has_flag(int argc, char**argv, const char*flag){
    for (int i=1;i<argc;i++) if (std::strcmp(argv[i],flag)==0) return true;
    return false;
}
static const char* get_opt(int argc, char**argv, const char*key){
    for (int i=1;i+1<argc;i++) if (std::strcmp(argv[i],key)==0) return argv[i+1];
    return nullptr;
}
static std::vector<int> parse_list_int(const char* s){
    std::vector<int> out;
    if(!s || !*s) return out;
    std::string str(s);
    size_t p=0;
    while(p<str.size()){
        size_t q=str.find_first_of(",;",p);
        if(q==std::string::npos) q=str.size();
        std::string tok=str.substr(p,q-p);
        if(!tok.empty()) out.push_back(std::atoi(tok.c_str()));
        p=q+1;
    }
    return out;
}

// ---------- matriz int** ----------
static int** alloc_mat(int n){
    int **m=(int**)std::malloc(n*sizeof(int*));
    for(int i=0;i<n;i++) m[i]=(int*)std::malloc(n*sizeof(int));
    return m;
}
static void free_mat(int **m,int n){
    for(int i=0;i<n;i++) std::free(m[i]);
    std::free(m);
}
static void copy_mat(int **dst,int **src,int n){
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) dst[i][j]=src[i][j];
}

// ---------- tempos ----------
static double now_cpu(){
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

static float time_cuda_kernel_loop_ms(int **mat, int n){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    floydWarshallCUDA(mat, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms=0.f;
    cudaEventElapsedTime(&ms,start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

// ---------- benchmark ----------
struct Result {
    int n;
    int density;
    int reps;
    double t_seq_mean, t_seq_std;
    double t_omp_mean, t_omp_std;
    double t_cuda_mean, t_cuda_std;
    double speedup_omp, speedup_cuda;
};

static std::pair<double,double> mean_std(const std::vector<double>& v){
    if(v.empty()) return std::make_pair(0.0,0.0);
    double m=0.0;
    for(size_t i=0;i<v.size();++i) m+=v[i];
    m/= (double)v.size();
    double s=0.0;
    if (v.size()>1){
        for(size_t i=0;i<v.size();++i){
            double d=v[i]-m; s+=d*d;
        }
        s = std::sqrt(s/(double)(v.size()-1));
    }
    return std::make_pair(m,s);
}

int main(int argc, char** argv){
    std::vector<int> Ns   = {256, 384, 512, 768, 1024};
    std::vector<int> Dens = {25, 50, 75};
    int reps = 5;
    bool verify = true;
    int omp_threads = 0;

    if (const char* s = get_opt(argc, argv, "--ns"))   { auto v=parse_list_int(s); if(!v.empty()) Ns=v; }
    if (const char* s = get_opt(argc, argv, "--dens")) { auto v=parse_list_int(s); if(!v.empty()) Dens=v; }
    if (const char* s = get_opt(argc, argv, "--reps")) { int r=std::atoi(s); if(r>0) reps=r; }
    if (has_flag(argc, argv, "--no-verify")) verify=false;
    if (const char* s = get_opt(argc, argv, "--omp-threads")) { int t=std::atoi(s); if(t>0) omp_threads=t; }

#ifdef _OPENMP
    if (omp_threads>0){
        omp_set_num_threads(omp_threads);
        std::printf("[INFO] OpenMP threads = %d\n", omp_threads);
    } else {
        std::printf("[INFO] OpenMP threads via OMP_NUM_THREADS/default.\n");
    }
#else
    if (omp_threads>0) std::printf("[WARN] Binario sem OpenMP; --omp-threads ignorado.\n");
#endif

    int devCount=0; cudaGetDeviceCount(&devCount);
    if (devCount<=0){
        std::printf("[WARN] Nenhuma GPU CUDA detectada. Parte CUDA sera pulada.\n");
    } else {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop,0);
        std::printf("[INFO] GPU 0: %s | VRAM: %.1f GB | CC %d.%d\n",
            prop.name, prop.totalGlobalMem/1073741824.0, prop.major, prop.minor);
    }

    FILE* csv=fopen("fw_bench.csv","w");
    if(!csv){ std::perror("fw_bench.csv"); return 1; }
    std::fprintf(csv,"method,n,density,rep,time_s\n");

    std::vector<Result> summary;

    for(size_t di=0; di<Dens.size(); ++di){
        int d = Dens[di];
        for(size_t ni=0; ni<Ns.size(); ++ni){
            int n = Ns[ni];
            std::printf("\n=== n=%d, dens=%d%%, reps=%d ===\n", n, d, reps);

            int **base=alloc_mat(n);
            gerarGrafoAleatorio(base, n, d);

            std::vector<double> seq_times, omp_times, cuda_times;

            for(int r=0;r<reps;r++){
                // SEQ
                int **A=alloc_mat(n);
                copy_mat(A, base, n);
                double t0=now_cpu();
                floydWarshallSequencial(A,n);
                double t1=now_cpu();
                double dt=t1-t0;
                seq_times.push_back(dt);
                std::fprintf(csv,"seq,%d,%d,%d,%.9f\n", n, d, r+1, dt);

                // OpenMP
                int **B=alloc_mat(n);
                copy_mat(B, base, n);
                double t2=now_cpu();
                floydWarshallOpenMP(B,n);
                double t3=now_cpu();
                double dt2=t3-t2;
                omp_times.push_back(dt2);
                std::fprintf(csv,"openmp,%d,%d,%d,%.9f\n", n, d, r+1, dt2);

                if (verify){
                    bool ok=true;
                    for(int i=0;i<n && ok;i++)
                        for(int j=0;j<n;j++)
                            if (A[i][j]!=B[i][j]) { ok=false; break; }
                    if(!ok) std::printf("[WARN] Diferenca (OpenMP vs Seq) n=%d dens=%d rep=%d\n", n,d,r+1);
                }

                // CUDA
                if (devCount>0){
                    int **C=alloc_mat(n);
                    copy_mat(C, base, n);
                    float ms=time_cuda_kernel_loop_ms(C, n);
                    double dt3=ms/1000.0;
                    cuda_times.push_back(dt3);
                    std::fprintf(csv,"cuda,%d,%d,%d,%.9f\n", n, d, r+1, dt3);

                    if (verify){
                        bool ok=true;
                        for(int i=0;i<n && ok;i++)
                            for(int j=0;j<n;j++)
                                if (A[i][j]!=C[i][j]) { ok=false; break; }
                        if(!ok) std::printf("[WARN] Diferenca (CUDA vs Seq) n=%d dens=%d rep=%d\n", n,d,r+1);
                    }
                    free_mat(C,n);
                }

                free_mat(B,n);
                free_mat(A,n);
            }

            std::pair<double,double> pseq = mean_std(seq_times);
            std::pair<double,double> pomp = mean_std(omp_times);
            std::pair<double,double> pcud = mean_std(cuda_times);

            double m_seq  = pseq.first, sd_seq  = pseq.second;
            double m_omp  = pomp.first, sd_omp  = pomp.second;
            double m_cuda = pcud.first, sd_cuda = pcud.second;

            Result R;
            R.n=n; R.density=d; R.reps=reps;
            R.t_seq_mean=m_seq; R.t_seq_std=sd_seq;
            R.t_omp_mean=m_omp; R.t_omp_std=sd_omp;
            R.t_cuda_mean=m_cuda; R.t_cuda_std=sd_cuda;
            R.speedup_omp  = (m_omp>0.0)? (m_seq/m_omp) : 0.0;
            R.speedup_cuda = (m_cuda>0.0)? (m_seq/m_cuda): 0.0;
            summary.push_back(R);

            std::printf("SEQ  : mean=%.6fs std=%.6f\n", m_seq, sd_seq);
            std::printf("OpenMP: mean=%.6fs std=%.6f  speedup=%.2fx\n", m_omp, sd_omp, R.speedup_omp);
            if (devCount>0)
                std::printf("CUDA : mean=%.6fs std=%.6f  speedup=%.2fx\n", m_cuda, sd_cuda, R.speedup_cuda);

            free_mat(base,n);
        }
    }

    fclose(csv);

    std::printf("\n===== RESUMO =====\n");
    std::printf("n,density,reps,seq_mean,omp_mean,cuda_mean,speedup_omp,speedup_cuda\n");
    for(size_t i=0;i<summary.size();++i){
        const Result& r=summary[i];
        std::printf("%d,%d,%d,%.6f,%.6f,%.6f,%.2f,%.2f\n",
            r.n, r.density, r.reps,
            r.t_seq_mean, r.t_omp_mean, r.t_cuda_mean,
            r.speedup_omp, r.speedup_cuda);
    }

    std::printf("\nOK. CSV salvo em fw_bench.csv\n");
    std::printf("Ex.: fwbench.exe --ns 256,512,768,1024 --dens 25,50 --reps 5 --omp-threads 16\n");
    return 0;
}
