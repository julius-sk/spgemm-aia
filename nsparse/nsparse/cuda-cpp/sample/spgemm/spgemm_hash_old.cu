#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <cusparse_v2.h>

#include <nsparse.hpp>
#include <CSR.hpp>
#include <SpGEMM.hpp>
#include <HashSpGEMM_volta_old.hpp>

typedef int IT;
#ifdef FLOAT
typedef float VT;
#else
typedef double VT;
#endif

template <class idType, class valType>
void spgemm_hash(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{

    idType i;
  
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;
  
    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Memcpy A and B from Host to Device */
    a.memcpyHtD();
    b.memcpyHtD();
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, flop_count);

    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    for (i = 0; i < SpGEMM_TRI_NUM; i++) {
        if (i > 0) {
            c.release_csr();
        }
        cudaEventRecord(event[0], 0);
        SpGEMM_Hash(a, b, c);
        cudaEventRecord(event[1], 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&msec, event[0], event[1]);
    
        if (i > 0) {
            ave_msec += msec;
        }
    }
    ave_msec /= SpGEMM_TRI_NUM - 1;
    printf("SpGEMM Intermediate product: %ld[GFLOPS]\n", flop_count/2);
    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    printf("SpGEMM using CSR format (Hash): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    /* Numeric Only */
    // ave_msec = 0;
    // for (i = 0; i < SpGEMM_TRI_NUM; i++) {
    //     cudaEventRecord(event[0], 0);
    //     SpGEMM_Hash_Numeric(a, b, c);
    //     cudaEventRecord(event[1], 0);
    //     cudaDeviceSynchronize();
    //     cudaEventElapsedTime(&msec, event[0], event[1]);
    
    //     if (i > 0) {
    //         ave_msec += msec;
    //     }
    // }
    // ave_msec /= SpGEMM_TRI_NUM - 1;

    // flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
    // printf("SpGEMM using CSR format (Hash, only numeric phase): %f[GFLOPS], %f[ms]\n", flops, ave_msec);

    c.memcpyDtH();
    c.release_csr();

// #ifdef sfDEBUG
//     CSR<IT, VT> cusparse_c;
//     SpGEMM_cuSPARSE(a, b, cusparse_c);
//     if (c == cusparse_c) {
//         cout << "HashSpGEMM is correctly executed" << endl;
//     }
//     cout << "Nnz of A: " << a.nnz << endl; 
//     cout << "Number of intermediate products: " << flop_count / 2 << endl; 
//     cout << "Nnz of C: " << c.nnz << endl; 
//     cusparse_c.release_cpu_csr();
// #endif

    a.release_csr();
    b.release_csr();

    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
}

template <class idType>
void set_flop_per_row(const idType* arpt, const idType* acol, const idType* brpt, 
                     long long int* flop_per_row, idType nrow, long long int& max_nnz) {
    max_nnz = 0;
    #pragma omp parallel 
    {
        long long int local_max_nnz = 0;
        #pragma omp for
        for (idType i = 0; i < nrow; i++) {
            long long int flop_per_row_local = 0;
            for (idType j = arpt[i]; j < arpt[i + 1]; j++) {
                flop_per_row_local += brpt[acol[j] + 1] - brpt[acol[j]];
            }
            flop_per_row[i] = flop_per_row_local;
            local_max_nnz = std::max(local_max_nnz, flop_per_row_local);
        }
        #pragma omp critical
        {
            max_nnz = std::max(max_nnz, local_max_nnz);
        }
    }
}

template <class idType, class valType>
void get_spgemm_flop(const CSR<idType, valType>& a, const CSR<idType, valType>& b, 
                     long long int& flop, long long int& max_nnz) {
    // Allocate host memory for flop counts
    long long int* flop_per_row = new long long int[a.nrow];
    
    // Calculate flops per row and get max nnz
    set_flop_per_row(a.rpt, a.colids, b.rpt, flop_per_row, a.nrow, max_nnz);
    
    // Reduce to get total flops
    flop = 0;
    #pragma omp parallel for reduction(+:flop)
    for (idType i = 0; i < a.nrow; i++) {
        flop += flop_per_row[i];
    }
    
    // Multiply by 2 for multiply-add operations
    flop *= 2;
    
    // Print stats
    std::cout << "Maximum NNZ in any row: " << max_nnz << std::endl;
    std::cout << "intermediate product number: " << (flop)/2 << std::endl;
    
    // Clean up
    delete[] flop_per_row;
}


/*Main Function*/
int main(int argc, char *argv[])
{
    CSR<IT, VT> a, b, c;

    /* Set CSR reding from MM file or generating random matrix */
    cout << "Initialize Matrix A" << endl;
    cout << "Read matrix data from " << argv[1] << endl;
    a.init_data_from_mtx(argv[1]);

    cout << "Initialize Matrix B" << endl;
    cout << "Read matrix data from " << argv[1] << endl;
    b.init_data_from_mtx(argv[1]);
    long long int flop;
    long long int max_nnz;
    get_spgemm_flop(a,b,flop,max_nnz);
    /* Execution of SpGEMM on GPU */
    spgemm_hash(a, b, c);
    
    a.release_cpu_csr();
    b.release_cpu_csr();
    c.release_cpu_csr();
  
    return 0;

}