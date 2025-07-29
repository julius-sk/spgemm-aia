#include <iostream>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <nsparse.hpp>
#include <nsparse_asm.hpp>
#include <CSR.hpp>
#include <BIN.hpp>

#ifndef HASHSPGEMM_H
#define HASHSPGEMM_H

#define BIN_NUM 7
#define PWARP 4
#define TS_S_P 32 //Table Size for Symbolic in PWARP per row
#define TS_N_P 16 //Table Size for Numeric in PWARP per row
#define TS_S_T 512 //Table Size for Symbolic in Thread block per row
#define TS_N_T 256 //Table Size for Numeric in Thread block per row

#define SHARED_S_P 4096 // Total table sizes required by one thread block in PWARP Symbolic
#define SHARED_N_P 2048 // Total table sizes required by one thread block in PWARP Numeric
#define HASH_SCAL 107

#define ORIGINAL_HASH


template <class idType>
__global__ void init_id_table(idType *d_id_table, idType nnz)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }
    d_id_table[i] = -1;
}

template <class idType, class valType>
__global__ void init_value_table(valType *d_values, idType nnz)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) {
        return;
    }
    d_values[i] = 0;
}

template <class idType>
__global__ void hash_symbolic_pwarp(const idType *__restrict__ aia_d_row_1,
                                    const idType *__restrict__ aia_d_nnz_1,
                                    const idType* __restrict__ d_bcolids,
                                    const idType *d_permutation,
                                    idType *d_row_nz,
                                    idType bin_offset, idType M)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    idType rid = i / PWARP;
    idType tid = i % PWARP;
    idType local_rid = rid % (blockDim.x / PWARP);
  
    idType j, k;
    idType soffset;
    idType bcol, key, hash, adr, nz, old;
    __shared__ idType id_table[SHARED_S_P];
  
    soffset = local_rid * TS_S_P;
  
    for (j = tid; j < TS_S_P; j += PWARP) {
        id_table[soffset + j] = -1;
    }
    if (rid >= M) {
        return;
    }

    rid = rid + bin_offset;
    nz = 0;
    for (j = aia_d_row_1[2*rid] + tid; j < aia_d_row_1[2*rid+1]; j += PWARP) {
        for (k = aia_d_nnz_1[2*j]; k < aia_d_nnz_1[2*j+1]; k++) {
            bcol = d_bcolids[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (TS_S_P - 1);
            adr = soffset + hash;
            while (1) {
                if (id_table[adr] == key) {
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (TS_S_P - 1);
                    adr = soffset + hash;
                }
            }
        }
    }
  
    for (j = PWARP / 2; j >= 1; j /= 2) {
        nz += __shfl_xor_sync(0xffffffff, nz, j);
    }

    if (tid == 0) {
        d_row_nz[d_permutation[rid]] = nz;
    }
}

template <class idType, int SH_ROW>
__global__ void hash_symbolic_tb(const idType *__restrict__ aia_d_row_1,
                                 const idType *__restrict__ aia_d_nnz_1,
                                 const idType* __restrict__ d_bcolids,
                                 const idType *d_permutation,
                                 idType *d_row_nz,
                                 idType bin_offset, idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp_size_num - 1);
    idType wid = threadIdx.x / warp_size_num;
    idType wnum = blockDim.x / warp_size_num;
    idType j, k;
    idType bcol, key, hash, old;
    idType nz, adr;

    __shared__ idType id_table[SH_ROW];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        id_table[j] = -1;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();

    nz = 0;
    rid = rid + bin_offset;
    for (j = aia_d_row_1[2*rid] + wid; j < aia_d_row_1[2*rid+1]; j += wnum) {
        for (k = aia_d_nnz_1[2*j] + tid; k < aia_d_nnz_1[2*j+1]; k += warp_size_num) {
            bcol = d_bcolids[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (1) {
                if (id_table[adr] == key) {
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                    adr = hash;
                }
            }
        }
    }

    for (j = warp_size_num / 2; j >= 1; j /= 2) {
        nz += __shfl_xor_sync(0xffffffff, nz, j);
    }
  
    __syncthreads();
    if (threadIdx.x == 0) {
        id_table[0] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        atomicAdd(id_table, nz);
    }
    __syncthreads();
  
    if (threadIdx.x == 0) {
        d_row_nz[d_permutation[rid]] = id_table[0];
    }
}

template <class idType, int SH_ROW>
__global__ void hash_symbolic_tb_large(const idType *__restrict__ aia_d_row_1, 
                                       const idType *__restrict__ aia_d_nnz_1, 
                                       const idType* __restrict__ d_bcolids,
                                       const idType *d_permutation,
                                       idType *d_row_nz,
                                       idType *d_fail_count, idType *d_fail_perm,
                                       idType bin_offset, idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp_size_num - 1);
    idType wid = threadIdx.x / warp_size_num;
    idType wnum = blockDim.x / warp_size_num;
    idType j, k;
    idType bcol, key, hash, old;
    idType adr;

    __shared__ idType id_table[SH_ROW];
    __shared__ idType snz[1];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        id_table[j] = -1;
    }
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
  
    if (rid >= M) {
        return;
    }
  
    __syncthreads();
  
    rid = rid + bin_offset;
    idType count = 0;
    idType border = SH_ROW >> 1;
    for (j = aia_d_row_1[2*rid] + wid; j < aia_d_row_1[2*rid+1]; j += wnum) {
        for (k = aia_d_nnz_1[2*j] + tid; k < aia_d_nnz_1[2*j+1]; k += warp_size_num) {
            bcol = d_bcolids[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            adr = hash;
            while (count < border && snz[0] < border) {
                if (id_table[adr] == key) {
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(snz, 1);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                    adr = hash;
                    count++;
                }
            }
            if (count >= border || snz[0] >= border) {
                break;
            }
        }
        if (count >= border || snz[0] >= border) {
            break;
        }
    }
  
    __syncthreads();
    if (count >= border || snz[0] >= border) {
        if (threadIdx.x == 0) {
            idType d = atomicAdd(d_fail_count, 1);
            d_fail_perm[d] = d_permutation[rid];
        }
    }
    else {
        if (threadIdx.x == 0) {
            d_row_nz[d_permutation[rid]] = snz[0];
        }
    }
}

template <class idType>
__global__ void hash_symbolic_gl(const idType * d_arpt,
                                 const idType *__restrict__ aia_d_nnz_1, 
                                 const idType* __restrict__ d_bcol,
                                 const idType *d_permutation, 
                                 idType *d_row_nz, idType *d_id_table,
                                 idType max_row_nz, idType bin_offset, idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp_size_num - 1);
    idType wid = threadIdx.x / warp_size_num;
    idType wnum = blockDim.x / warp_size_num;
    idType j, k;
    idType bcol, key, hash, old;
    idType nz, adr;
    idType offset = rid * max_row_nz;

    __shared__ idType snz[1];
    if (threadIdx.x == 0) {
        snz[0] = 0;
    }
    __syncthreads();
  
    if (rid >= M) {
        return;
    }
  
    nz = 0;
    rid = d_permutation[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum) {
        for (k = aia_d_nnz_1[2*j] + tid; k < aia_d_nnz_1[2*j+1]; k += warp_size_num) {
            bcol = d_bcol[k];
            key = bcol;
            hash = (bcol * HASH_SCAL) % max_row_nz;
            adr = offset + hash;
            while (1) {
                if (d_id_table[adr] == key) {
                    break;
                }
                else if (d_id_table[adr] == -1) {
                    old = atomicCAS(d_id_table + adr, -1, key);
                    if (old == -1) {
                        nz++;
                        break;
                    }
                }
                else {
                    hash = (hash + 1) % max_row_nz;
                    adr = offset + hash;
                }
            }
        }
    }
  
    for (j = warp_size_num / 2; j >= 1; j /= 2) {
        nz += __shfl_xor_sync(0xffffffff, nz, j);
    }
  
    if (tid == 0) {
        atomicAdd(snz, nz);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        d_row_nz[rid] = snz[0];
    }
}

template <class idType>
__global__ void hash_symbolic_gl2(const idType *__restrict__ aia_d_row_1,
                                 const idType *__restrict__ aia_d_nnz_1,
                                  const idType* __restrict__ d_bcol,
                                  const idType *d_permutation, 
                                  idType *d_row_nz, idType *d_id_table,
                                  idType max_row_nz, idType bin_offset,
                                  idType total_row_num, idType conc_row_num)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp_size_num - 1);
    idType wid = threadIdx.x / warp_size_num;
    idType wnum = blockDim.x / warp_size_num;
    idType j, k;
    idType bcol, key, hash, old;
    idType nz, adr;
    idType offset;
    idType target;

    offset = rid * max_row_nz;
    __shared__ idType snz[1];
  
    for (; rid < total_row_num; rid += conc_row_num) {
        for (j = threadIdx.x; j < max_row_nz; j += blockDim.x) {
            d_id_table[offset + j] = -1;
        }
        if (threadIdx.x == 0) {
            snz[0] = 0;
        }
        __syncthreads();

        nz = 0;
        target = rid + bin_offset;
    for (j = aia_d_row_1[2*rid] + wid; j < aia_d_row_1[2*rid+1]; j += wnum) {
        for (k = aia_d_nnz_1[2*j] + tid; k < aia_d_nnz_1[2*j+1]; k += warp_size_num) {
                bcol = d_bcol[k];
                key = bcol;
                hash = (bcol * HASH_SCAL) % max_row_nz;
                adr = offset + hash;
                while (1) {
                    if (d_id_table[adr] == key) {
                        break;
                    }
                    else if (d_id_table[adr] == -1) {
                        old = atomicCAS(d_id_table + adr, -1, key);
                        if (old == -1) {
                            nz++;
                            break;
                        }
                    }
                    else {
                        hash = (hash + 1) % max_row_nz;
                        adr = offset + hash;
                    }
                }
            }
        }
  
        for (j = warp_size_num / 2; j >= 1; j /= 2) {
            nz += __shfl_xor_sync(0xffffffff, nz, j);
        }
  
        if (tid == 0) {
            atomicAdd(snz, nz);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            d_row_nz[d_permutation[target]] = snz[0];
        }
        __syncthreads();
    }
}

template <class idType, class valType>
void hash_symbolic(idType *ad_rpt, idType *bd_colids, CSR<idType, valType> &c, 
                   idType *aia_d_row_1, 
                   idType *aia_d_nnz_1,
                   BIN<idType, BIN_NUM> &bin, idType nrow)
{
    idType i;
    idType GS, BS;
    for (i = BIN_NUM - 1; i >= 0; i--) {
        if (bin.bin_size[i] > 0) {
            switch (i) {
            case 0 :
                BS = 512;
                GS = div_round_up(bin.bin_size[i] * PWARP, BS);
                hash_symbolic_pwarp<<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, bd_colids,bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                cout << "case 0 finish" << endl;
                break;                
            case 1 :
                BS = 64;
                GS = bin.bin_size[i];
            	hash_symbolic_tb<idType, 512><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1,bd_colids,bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	cout << "case 1 finish" << endl;
                break;
            case 2 :
                BS = 128;
                GS = bin.bin_size[i];
            	hash_symbolic_tb<idType, 1024><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1,bd_colids,bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	cout << "case 2 finish" << endl;
                break;
            case 3 :
                BS = 256;
                GS = bin.bin_size[i];
            	hash_symbolic_tb<idType, 2048><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, bd_colids,bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	cout << "case 3 finish" << endl;
                break;
            case 4 :
                BS = 512;
                GS = bin.bin_size[i];
            	hash_symbolic_tb<idType, 4096><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1,bd_colids, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	cout << "case 4 finish" << endl;
                break;
            case 5 :
                BS = 1024;
                GS = bin.bin_size[i];
            	hash_symbolic_tb<idType, 8192><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, bd_colids, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
            	cout << "case 5 finish" << endl;
                break;
            case 6 :
                {
#ifdef ORIGINAL_HASH
            	    idType fail_count;
            	    idType *d_fail_count, *d_fail_perm;
            	    fail_count = 0;
            	    checkCudaErrors(cudaMalloc((void **)&d_fail_count, sizeof(idType)));
            	    checkCudaErrors(cudaMalloc((void **)&d_fail_perm, sizeof(idType) * bin.bin_size[i]));
            	    cudaMemcpy(d_fail_count, &fail_count, sizeof(idType), cudaMemcpyHostToDevice);
            	    BS = 1024;
            	    GS = bin.bin_size[i];
                    cout << "case 6 start" << endl;
            	    hash_symbolic_tb_large<idType, 8192><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, bd_colids, bin.d_permutation,bin.d_count, d_fail_count, d_fail_perm, bin.bin_offset[i], bin.bin_size[i]);
            	    cudaMemcpy(&fail_count, d_fail_count, sizeof(idType), cudaMemcpyDeviceToHost);
            	    if (fail_count > 0) {
                        cout << "Matrix is large" << endl;
              	        idType max_row_nz = bin.max_flop;
            	        size_t table_size = (size_t)max_row_nz * fail_count;
            	        idType *d_id_table;
            	        checkCudaErrors(cudaMalloc((void **)&(d_id_table), sizeof(idType) * table_size));
            	        BS = 1024;
            	        GS = div_round_up(table_size, BS);
            	        init_id_table<idType><<<GS, BS, 0, bin.stream[i]>>>(d_id_table, table_size);
            	        GS = bin.bin_size[i];
	                    hash_symbolic_gl<idType><<<GS, BS, 0, bin.stream[i]>>>(ad_rpt, aia_d_nnz_1, bd_colids, d_fail_perm, bin.d_count, d_id_table, max_row_nz, 0, fail_count);
                        cudaFree(d_id_table);
  	                }
                    cudaFree(d_fail_count);
                    cudaFree(d_fail_perm);
                    cout << "case 6 finish" << endl;
#else
                    idType max_row_nz = bin.max_flop;
                    idType conc_row_num = min(56 * 2, bin.bin_size[i]);
                    idType table_size = max_row_nz * conc_row_num;
                    while (table_size * sizeof(idType) > 1024 * 1024 * 1024) {
                        conc_row_num /= 2;
                        table_size = max_row_nz * conc_row_num;
                    }
                    idType *d_id_table;
                    checkCudaErrors(cudaMalloc((void **)&d_id_table, sizeof(idType) * table_size));
                    BS = 1024;
                    // GS = div_round_up(table_size, BS);
                    // init_id_table<idType><<<GS, BS, 0, bin.stream[i]>>>(d_id_table, table_size);
                    GS = conc_row_num;
                    hash_symbolic_gl2<idType><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, bd_colids, bin.d_permutation, bin.d_count, d_id_table, max_row_nz, bin.bin_offset[i], bin.bin_size[i], conc_row_num);
                    cudaFree(d_id_table);
#endif
                }
                break;
            default:
                ;
            }
        }
    }
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, bin.d_count, bin.d_count + (nrow + 1), c.d_rpt, 0);
    cudaMemcpy(&(c.nnz), c.d_rpt + c.nrow, sizeof(idType), cudaMemcpyDeviceToHost);
}

template <class idType, class valType, bool sort>
__global__ void hash_numeric_pwarp(const idType* aia_d_row_1,
                                 const idType* aia_d_nnz_1,
                                 const valType *d_aval, const idType* __restrict__ d_bcol, 
                                 const valType* __restrict__ d_bval, idType *d_crpt, 
                                 idType *d_ccol, valType *d_cval, const idType *d_permutation,
                                 idType *d_nz, idType bin_offset, idType bin_size)
{
    idType i = blockIdx.x * blockDim.x + threadIdx.x;
    idType rid = i / PWARP;
    idType tid = i % PWARP;
    idType local_rid = rid % (blockDim.x / PWARP);
    idType j;

    __shared__ idType id_table[SHARED_N_P];
    __shared__ valType value_table[SHARED_N_P];
  
    idType soffset = local_rid * (TS_N_P);
  
    for (j = tid; j < TS_N_P; j += PWARP) {
        id_table[soffset + j] = -1;
        value_table[soffset + j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }
    rid = rid + bin_offset;
  
    if (tid == 0) {
        d_nz[d_permutation[rid]] = 0;
    }

    idType k;
    idType bcol, hash, key, adr;
    idType offset = d_crpt[d_permutation[rid]];
    idType old, index;
    valType aval, bval;

    for (j = aia_d_row_1[2*rid] + tid; j < aia_d_row_1[2*rid + 1]; j += PWARP) {       
        aval = ld_gbl_val(d_aval + j);
        for (k = aia_d_nnz_1[2*j]; k < aia_d_nnz_1[2*j + 1]; k++) {
            bcol = d_bcol[k];
            bval = d_bval[k];
            
            key = bcol;
            hash = (bcol * HASH_SCAL) & ((TS_N_P) - 1);
            adr = soffset + hash;
            while (1) {
                if (id_table[adr] == key) {
                    atomicAdd(value_table + adr, aval * bval);
                    break;
                }
                else if (id_table[adr] == -1) {
                    old = atomicCAS(id_table + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(value_table + adr, aval * bval);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & ((TS_N_P) - 1);
                    adr = soffset + hash;
                }
            }
        }
    }

    __syncthreads();
    
    for (j = tid; j < (TS_N_P); j += PWARP) {
        if (id_table[soffset + j] != -1) {
            index = atomicAdd(d_nz + d_permutation[rid], 1);
            id_table[soffset + index] = id_table[soffset + j];
            value_table[soffset + index] = value_table[soffset + j];
        }
    }
    
    __syncthreads();
    
    idType nz = d_nz[d_permutation[rid]];
    if (sort) {
        // Sorting for shared data
        idType count, target;
        for (j = tid; j < nz; j += PWARP) {
            target = id_table[soffset + j];
            count = 0;
            for (k = 0; k < nz; k++) {
                count += (unsigned int)(id_table[soffset + k] - target) >> 31;
            }
            d_ccol[offset + count] = id_table[soffset + j];
            d_cval[offset + count] = value_table[soffset + j];
        }
    }
    else {
        // No sorting
        for (j = tid; j < nz; j += PWARP) {
            d_ccol[offset + j] = id_table[soffset + j];
            d_cval[offset + j] = value_table[soffset + j];
        }
    }
}

template <class idType, class valType, int SH_ROW, bool sort>
__global__ void hash_numeric_tb(const idType * aia_d_row_1,
                                 const idType * aia_d_nnz_1,                                 
                                 const valType *d_avalues, 
                                 const idType* __restrict__ d_bcolids, const valType* __restrict__ d_bvalues,
                                  idType *d_crpt, idType *d_ccolids, valType *d_cvalues, 
                                  const idType *d_permutation, idType *d_nz, idType bin_offset, idType bin_size)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp_size_num - 1);
    idType wid = threadIdx.x / warp_size_num;
    idType wnum = blockDim.x / warp_size_num;
    idType j;
    __shared__ idType id_table[SH_ROW];
    __shared__ valType value_table[SH_ROW];
  
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x) {
        id_table[j] = -1;
        value_table[j] = 0;
    }
  
    if (rid >= bin_size) {
        return;
    }

    rid = rid + bin_offset;

    if (threadIdx.x == 0) {
        d_nz[d_permutation[rid]] = 0;
    }
    __syncthreads();

    idType k;
    idType bcolids, hash, key;
    idType offset = d_crpt[d_permutation[rid]];
    idType old, index;
    valType avalues, bvalues;

    for (j = aia_d_row_1[2*rid] + wid; j < aia_d_row_1[2*rid + 1]; j += wnum) {       
        avalues = ld_gbl_val(d_avalues + j);
        for (k = aia_d_nnz_1[2*j]+ tid; k < aia_d_nnz_1[2*j + 1]; k += warp_size_num) {
            bcolids = d_bcolids[k];
            bvalues = d_bvalues[k];
	
            key = bcolids;
            hash = (bcolids * HASH_SCAL) & (SH_ROW - 1);
            while (1) {
                if (id_table[hash] == key) {
                    atomicAdd(value_table + hash, avalues * bvalues);
                    break;
                }
                else if (id_table[hash] == -1) {
                    old = atomicCAS(id_table + hash, -1, key);
                    if (old == -1) {
                        atomicAdd(value_table + hash, avalues * bvalues);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) & (SH_ROW - 1);
                }
            }
        }
    }
  
    __syncthreads();
    if (threadIdx.x < warp_size_num) {
        for (j = tid; j < SH_ROW; j += warp_size_num) {
            if (id_table[j] != -1) {
                index = atomicAdd(d_nz + d_permutation[rid], 1);
                id_table[index] = id_table[j];
                value_table[index] = value_table[j];
            }
        }
    }
    __syncthreads();
    idType nz = d_nz[d_permutation[rid]];
    if (sort) {
        /* Sorting for shared data */
        idType count, target;
        for (j = threadIdx.x; j < nz; j += blockDim.x) {
            target = id_table[j];
            count = 0;
            for (k = 0; k < nz; k++) {
                count += (unsigned int)(id_table[k] - target) >> 31;
            }
            d_ccolids[offset + count] = id_table[j];
            d_cvalues[offset + count] = value_table[j];
        }
    }
    else {
        /* No Sorting */
        for (j = threadIdx.x; j < nz; j += blockDim.x) {
            d_ccolids[offset + j] = id_table[j];
            d_cvalues[offset + j] = value_table[j];
        }
    }
}

#ifdef ORIGINAL_HASH
template <class idType, class valType, bool sort>
__global__ void hash_numeric_gl(const idType *__restrict__ aia_d_row_1,
                                 const idType *__restrict__ aia_d_nnz_1, 
                                  const valType *d_avalues, const idType* __restrict__ d_bcolids, 
                                  const valType* __restrict__ d_bvalues, idType *d_crpt, 
                                  idType *d_ccolids, valType *d_cvalues, const idType *d_permutation, 
                                  idType *d_nz, idType *d_id_table, valType *d_value_table, idType max_row_nz, 
                                  idType bin_offset, idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp_size_num - 1);
    idType wid = threadIdx.x / warp_size_num;
    idType wnum = blockDim.x / warp_size_num;
    idType j;
  
    if (rid >= M) {
        return;
    }

    idType doffset = rid * max_row_nz;

    rid = rid + bin_offset;
  
    if (threadIdx.x == 0) {
        d_nz[d_permutation[rid]] = 0;
    }
    __syncthreads();

    idType k;
    idType bcolids, hash, key, adr;
    idType offset = d_crpt[d_permutation[rid]];
    idType old, index;
    valType avalues, bvalues;

    for (j = aia_d_row_1[2*rid] + wid; j < aia_d_row_1[2*rid + 1]; j += wnum) {       
        avalues = ld_gbl_val(d_avalues + j);
        for (k = aia_d_nnz_1[2*j]+ tid; k < aia_d_nnz_1[2*j + 1]; k += warp_size_num) {
            bcolids = d_bcolids[k];
            bvalues = d_bvalues[k];
      
            key = bcolids;
            hash = (bcolids * HASH_SCAL) % max_row_nz;
            adr = doffset + hash;
            while (1) {
                if (d_id_table[adr] == key) {
                    atomicAdd(d_value_table + adr, avalues * bvalues);
                    break;
                }
                else if (d_id_table[adr] == -1) {
                    old = atomicCAS(d_id_table + adr, -1, key);
                    if (old == -1) {
                        atomicAdd(d_value_table + adr, avalues * bvalues);
                        break;
                    }
                }
                else {
                    hash = (hash + 1) % max_row_nz;
                    adr = doffset + hash;
                }
            }
        }
    }
  
    __syncthreads();
    if (threadIdx.x < warp_size_num) {
        for (j = tid; j < max_row_nz; j += warp_size_num) {
            if (d_id_table[doffset + j] != -1) {
                index = atomicAdd(d_nz + d_permutation[rid], 1);
                d_id_table[doffset + index] = d_id_table[doffset + j];
                d_value_table[doffset + index] = d_value_table[doffset + j];
            }
        }
    }
    __syncthreads();
    idType nz = d_nz[d_permutation[rid]];
    if (sort) {
        /* Sorting for shared data */
        idType count, target;
        for (j = threadIdx.x; j < nz; j += blockDim.x) {
            target = d_id_table[doffset + j];
            count = 0;
            for (k = 0; k < nz; k++) {
                count += (unsigned int)(d_id_table[doffset + k] - target) >> 31;
            }
            d_ccolids[offset + count] = d_id_table[doffset + j];
            d_cvalues[offset + count] = d_value_table[doffset + j];
        }
    }
    else {
        /* No sorting */
        for (j = threadIdx.x; j < nz; j += blockDim.x) {
            d_ccolids[offset + j] = d_id_table[doffset + j];
            d_cvalues[offset + j] = d_value_table[doffset + j];
        }
    }
}
#else
template <class idType, class valType, bool sort>
__global__ void hash_numeric_gl(const idType *d_arpt, const idType *d_acolids, const valType *d_avalues, const idType* __restrict__ d_brpt, const idType* __restrict__ d_bcolids, const valType* __restrict__ d_bvalues, idType *d_crpt, idType *d_ccolids, valType *d_cvalues, const idType *d_permutation, idType *d_nz, idType *d_id_table, valType *d_value_table, idType max_row_nz, idType bin_offset, idType M)
{
    idType rid = blockIdx.x;
    idType tid = threadIdx.x & (warp_size_num - 1);
    idType wid = threadIdx.x / warp_size_num;
    idType wnum = blockDim.x / warp_size_num;
    idType j;
    idType conc_row_num = gridDim.x;
    idType target;
  
    if (rid >= M) {
        return;
    }

    idType doffset = rid * max_row_nz;
    
    for (; rid < M; rid += conc_row_num) {
        target = d_permutation[rid + bin_offset];
  
        for (j = threadIdx.x; j < max_row_nz; j += blockDim.x) {
            d_id_table[doffset + j] = -1;
            d_value_table[doffset + j] = 0;
        }
        if (threadIdx.x == 0) {
            d_nz[target] = 0;
        }
        __syncthreads();

        idType acolids;
        idType k;
        idType bcolids, hash, key, adr;
        idType offset = d_crpt[target];
        idType old, index;
        valType avalues, bvalues;

        for (j = d_arpt[target] + wid; j < d_arpt[target + 1]; j += wnum) {
            acolids = ld_gbl_col(d_acolids + j);
            avalues = ld_gbl_val(d_avalues + j);
            for (k = d_brpt[acolids] + tid; k < d_brpt[acolids + 1]; k += warp_size_num) {
                bcolids = d_bcolids[k];
                bvalues = d_bvalues[k];
      
                key = bcolids;
                hash = (bcolids * HASH_SCAL) % max_row_nz;
                adr = doffset + hash;
                while (1) {
                    if (d_id_table[adr] == key) {
                        atomicAdd(d_value_table + adr, avalues * bvalues);
                        break;
                    }
                    else if (d_id_table[adr] == -1) {
                        old = atomicCAS(d_id_table + adr, -1, key);
                        if (old == -1) {
                            atomicAdd(d_value_table + adr, avalues * bvalues);
                            break;
                        }
                    }
                    else {
                        hash = (hash + 1) % max_row_nz;
                        adr = doffset + hash;
                    }
                }
            }
        }
  
        __syncthreads();
        if (threadIdx.x < warp_size_num) {
            for (j = tid; j < max_row_nz; j += warp_size_num) {
                if (d_id_table[doffset + j] != -1) {
                    index = atomicAdd(d_nz + target, 1);
                    d_id_table[doffset + index] = d_id_table[doffset + j];
                    d_value_table[doffset + index] = d_value_table[doffset + j];
                }
            }
        }
        __syncthreads();
        idType nz = d_nz[target];
        if (sort) {
            /* Sorting for shared data */
            idType count, target;
            for (j = threadIdx.x; j < nz; j += blockDim.x) {
                target = d_id_table[doffset + j];
                count = 0;
                for (k = 0; k < nz; k++) {
                    count += (unsigned int)(d_id_table[doffset + k] - target) >> 31;
                }
                d_ccolids[offset + count] = d_id_table[doffset + j];
                d_cvalues[offset + count] = d_value_table[doffset + j];
            }
        }
        else {
            /* No sorting */
            for (j = threadIdx.x; j < nz; j += blockDim.x) {
                d_ccolids[offset + j] = d_id_table[doffset + j];
                d_cvalues[offset + j] = d_value_table[doffset + j];
            }
        }
    }
}
#endif

template <typename idType, typename valType>
void print_csr_matrix(const CSR<idType, valType>& matrix, const char* name, int max_rows = 10, int max_nnz = 20) {
    std::vector<idType> h_rpt(matrix.nrow + 1);
    std::vector<idType> h_colids(std::min(matrix.nnz, static_cast<idType>(max_nnz)));
    std::vector<valType> h_values(std::min(matrix.nnz, static_cast<idType>(max_nnz)));

    // Copy data from device to host
    cudaMemcpy(h_rpt.data(), matrix.d_rpt, (matrix.nrow + 1) * sizeof(idType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colids.data(), matrix.d_colids, h_colids.size() * sizeof(idType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_values.data(), matrix.d_values, h_values.size() * sizeof(valType), cudaMemcpyDeviceToHost);

    std::cout << "Matrix " << name << " (CSR format):" << std::endl;
    std::cout << "Dimensions: " << matrix.nrow << " x " << matrix.ncolumn << std::endl;
    std::cout << "Non-zero elements: " << matrix.nnz << std::endl;

    // Print row pointers
    std::cout << "Row pointers: ";
    for (int i = 0; i <= std::min(matrix.nrow, max_rows); ++i) {
        std::cout << h_rpt[i] << " ";
    }
    if (matrix.nrow > max_rows) std::cout << "...";
    std::cout << std::endl;

    // Print column indices and values
    std::cout << "Column indices: ";
    for (int i = 0; i < h_colids.size(); ++i) {
        std::cout << h_colids[i] << " ";
    }
    if (matrix.nnz > max_nnz) std::cout << "...";
    std::cout << std::endl;

    std::cout << "Values: ";
    for (int i = 0; i < h_values.size(); ++i) {
        std::cout << h_values[i] << " ";
    }
    if (matrix.nnz > max_nnz) std::cout << "...";
    std::cout << std::endl;
}

template <class idType, class valType, bool sort>
void hash_numeric(valType *ad_values, idType *bd_colids,valType *bd_values,CSR<idType, valType> &c, 
                   idType *aia_d_row_1, 
                   idType *aia_d_nnz_1,
                   BIN<idType, BIN_NUM> &bin)
{
    idType i;
    idType GS, BS;
    for (i = BIN_NUM - 1; i >= 0; i--) {
        if (bin.bin_size[i] > 0) {
            switch (i) {
            case 0:
                BS = 512;
                GS = div_round_up(bin.bin_size[i] * PWARP, BS);
                hash_numeric_pwarp<idType, valType, sort><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, ad_values, bd_colids, bd_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                cout << "case 0 finish" << endl;
                break;
            case 1:
                BS = 64;
                GS = bin.bin_size[i];
                hash_numeric_tb<idType, valType, 256, sort><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, ad_values, bd_colids, bd_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                cout << "case 1 finish" << endl;
                break;
            case 2:
                BS = 128;
                GS = bin.bin_size[i];
                hash_numeric_tb<idType, valType, 512, sort><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, ad_values, bd_colids, bd_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                cout << "case 2 finish" << endl;
                break;
            case 3:
                BS = 256;
                GS = bin.bin_size[i];
                hash_numeric_tb<idType, valType, 1024, sort><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, ad_values, bd_colids, bd_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                cout << "case 3 finish" << endl;
                break;
            case 4:
                BS = 512;
                GS = bin.bin_size[i];
                hash_numeric_tb<idType, valType, 2048, sort><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, ad_values, bd_colids, bd_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                cout << "case 4 finish" << endl;
                break;
            case 5:
                BS = 1024;
                GS = bin.bin_size[i];
                hash_numeric_tb<idType, valType, 4096, sort><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, ad_values, bd_colids, bd_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, bin.bin_offset[i], bin.bin_size[i]);
                cout << "case 5 finish" << endl;
                break;
            case 6 :
                {
#ifdef ORIGINAL_HASH
                    idType max_row_nz = bin.max_nz * 2;
                    idType table_size = max_row_nz * bin.bin_size[i];
#else
                    idType max_row_nz = 8192;
                    while (max_row_nz <= bin.max_nz) {
                        max_row_nz *= 2;
                    }
                    idType conc_row_num = min(56 * 2, bin.bin_size[i]);
                    idType table_size = max_row_nz * conc_row_num;
#endif
                    idType *d_id_table;
                    valType *d_value_table;
                    checkCudaErrors(cudaMalloc((void **)&(d_id_table), sizeof(idType) * table_size));
                    checkCudaErrors(cudaMalloc((void **)&(d_value_table), sizeof(valType) * table_size));
                    BS = 1024; 
#ifdef ORIGINAL_HASH
                    GS = div_round_up(table_size, BS);
                    init_id_table<<<GS, BS, 0, bin.stream[i]>>>(d_id_table, table_size);
                    init_value_table<<<GS, BS, 0, bin.stream[i]>>>(d_value_table, table_size);
                    GS = bin.bin_size[i];
#else
                    GS = conc_row_num;
#endif
                    hash_numeric_gl<idType, valType, sort><<<GS, BS, 0, bin.stream[i]>>>(aia_d_row_1, aia_d_nnz_1, ad_values, bd_colids, bd_values, c.d_rpt, c.d_colids, c.d_values, bin.d_permutation, bin.d_count, d_id_table, d_value_table, max_row_nz, bin.bin_offset[i], bin.bin_size[i]);
                    cudaFree(d_id_table);
                    cudaFree(d_value_table);
                }
                break;
            }
        }
    }
    cudaDeviceSynchronize();
}

// template <class idType, class idType,class idType, class valType>
// void read_1_new(idType *res1,idType *res2, idType *buffer, idType *ind1, valType n_now) {
//     //#pragma omp parallel for
//     for (size_t j = 0 ; j < n_now; ++j) {
//         res1[j]=buffer[ind1[j]];
//         res2[j]=buffer[ind1[j]+1];
//     }
// //std::cout << "result size: " << sizeof(res_1)/sizeof(res[0]) << std::endl;    
// }


template <typename idType, typename valType>
__global__ void read_1_new_kernel(idType *res1, idType *res2, const idType *buffer1, const idType *ind1, const idType *buffer2, const idType *ind2, valType n_now) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_now) {
        idType index = ind1[idx];
        res1[2*idx] = buffer1[index];
        res1[2*idx+1] = buffer1[index + 1];
        for (idType j = res1[2*idx]; j < res1[2*idx+1]; j ++) {
           idType index2 = ind2[j];
           res2[2*j] = buffer2[index2];
           res2[2*j+1] = buffer2[index2 + 1]; 
        }
    }
}

template <typename idType, typename valType>
void read_1_new(idType *res1, idType *res2, idType *buffer1, idType *ind1, idType *buffer2, idType *ind2, valType n_now) {
    const int BLOCK_SIZE = 1024;
    int num_blocks = (n_now + BLOCK_SIZE - 1) / BLOCK_SIZE;

    read_1_new_kernel<<<num_blocks, BLOCK_SIZE>>>(res1, res2, buffer1, ind1, buffer2, ind2, n_now);
    cudaDeviceSynchronize(); // Wait for the kernel to finish
}

template <typename idType>
void print_partial_data(const char* name, idType* d_array, int nrow, int num_elements = 10) {
    std::vector<idType> h_array(num_elements);
    
    // Copy a portion of the data from device to host
    cudaMemcpy(h_array.data(), d_array, num_elements * sizeof(idType), cudaMemcpyDeviceToHost);
    
    // Print the data
    std::cout << "First " << num_elements << " elements of " << name << ":" << std::endl;
    for (int i = 0; i < num_elements; ++i) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // If there are more elements, print the last few
    if (nrow > num_elements) {
        cudaMemcpy(h_array.data(), d_array + (nrow - num_elements), num_elements * sizeof(idType), cudaMemcpyDeviceToHost);
        std::cout << "Last " << num_elements << " elements of " << name << ":" << std::endl;
        for (int i = 0; i < num_elements; ++i) {
            std::cout << h_array[i] << " ";
        }
        std::cout << std::endl;
    }
}

// template <typename idType>
// void print_bin_sizes(const BIN<idType, BIN_NUM>& bin) {

// }


template <bool sort, class idType, class valType>
float SpGEMM_Hash(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    cudaEvent_t event[2];
    float msec;
    for (int i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }

    BIN<idType, BIN_NUM> bin(a.nrow);
        float total_AIA_time=0;
    c.nrow = a.nrow;
    c.ncolumn = b.ncolumn;
    c.device_malloc = true;
    cudaMalloc((void **)&(c.d_rpt), sizeof(idType) * (c.nrow + 1));
    
    bin.set_max_bin(a.d_rpt, a.d_colids, b.d_rpt, a.nrow, TS_S_P, TS_S_T); 
    cudaEventRecord(event[0], 0);
    idType *aia_d_row_1;
    idType *aia_d_nnz_1;
    cudaMalloc((void**)&aia_d_row_1, 2*a.nrow * sizeof(idType));
    cudaMalloc((void**)&aia_d_nnz_1, 2*a.nnz * sizeof(idType));
    //read_1_new(aia_d_row,a.d_rpt, bin.d_permutation, nrow);

    read_1_new(aia_d_row_1, aia_d_nnz_1, a.d_rpt, bin.d_permutation, b.d_rpt, a.d_colids,a.nrow);
    cudaEventRecord(event[1], 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&msec, event[0], event[1]);
         total_AIA_time=total_AIA_time+msec;
    cout << "AIA runtime: " << msec << endl;
// print_partial_data("aia_d_row_1", aia_d_row_1, a.nrow);
// print_partial_data("aia_d_nnz_1", aia_d_nnz_1, a.nrow);
//    //print_bin_sizes(bin);

//     std::cout << "Bin sizes:" << std::endl;
    
    //Copy bin sizes from device to host
    // idType h_bin_sizes[BIN_NUM];
    // cudaMemcpy(h_bin_sizes, bin.d_bin_size, BIN_NUM * sizeof(idType), cudaMemcpyDeviceToHost);
    // std::cout << "Debug: Bin sizes after setting:" << std::endl;
    // for (int i = 0; i < BIN_NUM; ++i) {
    //     std::cout << "  Bin " << i << ": " << h_bin_sizes[i] << " elements" << std::endl;
    // }    
   
    // //Optionally, print bin offsets
    // idType h_bin_offsets[BIN_NUM];
    // cudaMemcpy(h_bin_offsets, bin.d_bin_offset, BIN_NUM * sizeof(idType), cudaMemcpyDeviceToHost);
    
    // std::cout << "Bin offsets:" << std::endl;
    // for (int i = 0; i < BIN_NUM; ++i) {
    //     std::cout << "  Bin " << i << " offset: " << h_bin_offsets[i] << std::endl;
    // }

    
    cudaEventRecord(event[0], 0);
    hash_symbolic(a.d_rpt, b.d_colids, c, aia_d_row_1, aia_d_nnz_1, bin, a.nrow);
    cudaEventRecord(event[1], 0);   
    cudaDeviceSynchronize();
     cout << "hash_symbolic done " << endl;
    cudaEventElapsedTime(&msec, event[0], event[1]);
    // cout << "HashSymbolic: " << msec << endl;
    
    cudaMalloc((void **)&(c.d_colids), sizeof(idType) * (c.nnz));
    cudaMalloc((void **)&(c.d_values), sizeof(valType) * (c.nnz));
    
    bin.set_min_bin(a.nrow, TS_N_P, TS_N_T);
    cudaEventRecord(event[0], 0);
    read_1_new(aia_d_row_1, aia_d_nnz_1, a.d_rpt, bin.d_permutation, b.d_rpt, a.d_colids,a.nrow);
    cudaEventRecord(event[1], 0);
   cudaDeviceSynchronize();
    cudaEventElapsedTime(&msec, event[0], event[1]);
    total_AIA_time=total_AIA_time+msec;
    cout << "AIA2 runtime: " << msec << endl;

    cudaEventRecord(event[0], 0);
    hash_numeric<idType, valType, sort>(a.d_values, b.d_colids,b.d_values, c, aia_d_row_1, aia_d_nnz_1, bin);
    cudaEventRecord(event[1], 0);
    cudaDeviceSynchronize();
    cout << "hash_numeric done " << endl;
    cudaEventElapsedTime(&msec, event[0], event[1]);
    return total_AIA_time;
    //print_csr_matrix(c, "C (Result)");
    // cout << "HashNumeric: " << msec << endl;
}

template <class idType, class valType>
float SpGEMM_Hash(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    float total_AIA_time=0;
    total_AIA_time=SpGEMM_Hash<true, idType, valType>(a, b, c);
    return total_AIA_time;
}

template <bool sort, class idType, class valType>
void SpGEMM_Hash_Numeric(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    BIN<idType, BIN_NUM> bin(a.nrow);

    bin.set_min_bin(c.d_rpt, a.nrow, TS_N_P, TS_N_T);
    hash_numeric<idType, valType, sort>(a, b, c, bin);
}

template <class idType, class valType>
void SpGEMM_Hash_Numeric(CSR<idType, valType> a, CSR<idType, valType> b, CSR<idType, valType> &c)
{
    SpGEMM_Hash_Numeric<true, idType, valType>(a, b, c);
}



#endif

