#include <iostream>
#include <string>
#include <cuda.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
using namespace std;

#ifndef CSR_H
#define CSR_H
template <class idType, class valType>
class CSR
{
public:
    CSR():nrow(0), ncolumn(0), nnz(0), device_malloc(false)
    {
    }
    ~CSR()
    {
    }
    void release_cpu_csr()
    {
        delete[] rpt;
        delete[] colids;
        delete[] values;
    }
    void release_csr()
    {
        if (device_malloc) {
            cudaFree(d_rpt);
            cudaFree(d_colids);
            cudaFree(d_values);
        }
        device_malloc = false;
    }
    bool operator==(CSR mat)
    {
        bool f = false;
        if (nrow != mat.nrow) {
            cout << "Number of row is not correct: " << nrow << ", " << mat.nrow << endl;
            return f;
        }
        if (ncolumn != mat.ncolumn) {
            cout << "Number of column is not correct" << ncolumn << ", " << mat.ncolumn << endl;
            return f;
        }
        if (nnz != mat.nnz) {
            cout << "Number of nz is not correct" << nnz << ", " << mat.nnz << endl;
            return f;
        }
        if (rpt == NULL || mat.rpt == NULL || colids == NULL || mat.colids == NULL || values == NULL || mat.values == NULL) {
            cout << "NULL Pointer" << endl;
            return f;
        }
        for (idType i = 0; i < nrow + 1; ++i) {
            if (rpt[i] != mat.rpt[i]) {
                cout << "rpt[" << i << "] is not correct" << endl;
                return f;
            }
        }
        for (idType i = 0; i < nnz; ++i) {
            if (colids[i] != mat.colids[i]) {
                cout << "colids[" << i << "] is not correct" << endl;
                return f;
            }
        }
        idType total_fail = 10;
        valType delta, base, scale;
        for (idType i = 0; i < nnz; ++i) {
            delta = values[i] - mat.values[i];
            base = values[i];
            if (delta < 0) {
                delta *= -1;
            }
            if (base < 0) {
                base *= -1;
            }
            scale = 1000;
            if (sizeof(valType) == sizeof(double)) {
                scale *= 1000;
            }
            if (delta * scale * 100 > base) {
                cout << i << ": " << values[i] << ", " << mat.values[i] << endl;
                total_fail--;
            }
            if (total_fail == 0) {
                cout << "values[" << i << "] is not correct" << endl;
                return f;
            }
        }
        f = true;
        return f;
    }

    void init_data_from_mtx(string file_path);
    void memcpyHtD()
    {
        if (!device_malloc) {
            cout << "Allocating memory space for matrix data on device memory" << endl;
            cudaMalloc((void **)&d_rpt, sizeof(idType) * (nrow + 1));
            cudaMalloc((void **)&d_colids, sizeof(idType) * nnz);
            cudaMalloc((void **)&d_values, sizeof(valType) * nnz);
        }
        cout << "Copying matrix data to GPU device" << endl;
        cudaMemcpy(d_rpt, rpt, sizeof(idType) * (nrow + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colids, colids, sizeof(idType) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, values, sizeof(valType) * nnz, cudaMemcpyHostToDevice);
        device_malloc = true;
    }
    void memcpyDtH()
    {
        rpt = new idType[nrow + 1];
        colids = new idType[nnz];
        values = new valType[nnz];        
        cout << "Matrix data is copied to Host" << endl;
        cudaMemcpy(rpt, d_rpt, sizeof(idType) * (nrow + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(colids, d_colids, sizeof(idType) * nnz, cudaMemcpyDeviceToHost);
        cudaMemcpy(values, d_values, sizeof(valType) * nnz, cudaMemcpyDeviceToHost);
    }
    void set_all_nnz_to_one()
    {
        for (idType i = 0; i < nnz; ++i) {
            values[i] = 1;
        }

        if (device_malloc) {
            cudaMemcpy(d_values, values, sizeof(valType) * nnz, cudaMemcpyHostToDevice);
        }
    }
    void spmv_cpu(valType *x, valType *y);
    // void normalize_columns(); // for mca
    // void hadamard_power(valType power); // for mca
    void normalize_columns();
    //void normalize_columns_2();
    void hadamard_power(valType power);
    void prune_and_scale(valType threshold, int top_k);
    bool is_converged(const CSR& other, valType epsilon); // for mca
    // void prune_and_scale(valType threshold, int top_k); // for mca
    //void correct_csr(); // for mca

    int extract_clusters();

    idType *rpt;
    idType *colids;
    valType *values;
    idType *d_rpt;
    idType *d_colids;
    valType *d_values;
    idType nrow;
    idType ncolumn;
    idType nnz;
    bool host_malloc;
    bool device_malloc;
};

template <class idType, class valType>
void CSR<idType, valType>::init_data_from_mtx(string file_path)
{
    idType i, num;
    bool isUnsy;
    char *line, *ch;
    FILE *fp;
    idType *col_coo, *row_coo, *nnz_num, *each_row_index;
    valType *val_coo;
    idType LINE_LENGTH_MAX = 256;

    device_malloc = false;
    
    isUnsy = false;
    line = new char[LINE_LENGTH_MAX];
  
    /* Open File */
    fp = fopen(file_path.c_str(), "r");
    if (fp == NULL) {
        cout << "Cannot find file" << endl;
        exit(1);
    }

    fgets(line, LINE_LENGTH_MAX, fp);
    if (strstr(line, "general")) {
        isUnsy = true;
    }
    do {
        fgets(line, LINE_LENGTH_MAX, fp);
    } while(line[0] == '%');
  
    /* Get size info */
    sscanf(line, "%d %d %d", &nrow, &ncolumn, &nnz);
    
    /* Store in COO format */
    num = 0;
    col_coo = new idType[nnz];
    row_coo = new idType[nnz];
    val_coo = new valType[nnz];

    while (fgets(line, LINE_LENGTH_MAX, fp)) {
        ch = line;
        /* Read first word (row id)*/
        row_coo[num] = (idType)(atoi(ch) - 1);
        ch = strchr(ch, ' ');
        ch++;
        /* Read second word (column id)*/
        col_coo[num] = (idType)(atoi(ch) - 1);
        ch = strchr(ch, ' ');

        if (ch != NULL) {
            ch++;
            /* Read third word (value data)*/
            val_coo[num] = (valType)atof(ch);
            ch = strchr(ch, ' ');
        }
        else {
            val_coo[num] = 1.0;
        }
        num++;
    }
    fclose(fp);
    delete[] line;

    /* Count the number of non-zero in each row */
    nnz_num = new idType[nrow];
    for (i = 0; i < nrow; i++) {
        nnz_num[i] = 0;
    }
    for (i = 0; i < num; i++) {
        nnz_num[row_coo[i]]++;
        if(col_coo[i] != row_coo[i] && isUnsy == false) {
            nnz_num[col_coo[i]]++;
            nnz++;
        }
    }

    /* Allocation of rpt, col, val */
    rpt = new idType[nrow + 1];
    colids = new idType[nnz];
    values = new valType[nnz];

    rpt[0] = 0;
    for (i = 0; i < nrow; i++) {
        rpt[i + 1] = rpt[i] + nnz_num[i];
    }

    each_row_index = new idType[nrow];
    for (i = 0; i < nrow; i++) {
        each_row_index[i] = 0;
    }
  
    for (i = 0; i < num; i++) {
        colids[rpt[row_coo[i]] + each_row_index[row_coo[i]]] = col_coo[i];
        values[rpt[row_coo[i]] + each_row_index[row_coo[i]]++] = val_coo[i];
    
        if (col_coo[i] != row_coo[i] && isUnsy == false) {
            colids[rpt[col_coo[i]] + each_row_index[col_coo[i]]] = row_coo[i];
            values[rpt[col_coo[i]] + each_row_index[col_coo[i]]++] = val_coo[i];
        }
    }

    cout << "Row: " << nrow << ", Column: " << ncolumn << ", Nnz: " << nnz << endl;

    delete[] nnz_num;
    delete[] row_coo;
    delete[] col_coo;
    delete[] val_coo;
    delete[] each_row_index;

}

// Kernel for column sum calculation
// template <class idType, class valType>
// __global__ void column_sum_kernel(idType nrow, idType *rpt, idType *colids, valType *values, valType *col_sums) {
//     idType i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < nrow) {
//         for (idType j = rpt[i]; j < rpt[i + 1]; ++j) {
//             atomicAdd(&col_sums[colids[j]], values[j]);
//         }
//     }
// }

// Kernel for normalization
// template <class idType, class valType>
// __global__ void normalize_kernel(idType nrow, idType *rpt, idType *colids, valType *values, valType *col_sums) {
//     idType i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < nrow) {
//         for (idType j = rpt[i]; j < rpt[i + 1]; ++j) {
//             values[j] /= col_sums[colids[j]];
//         }
//     }
// }

// template <class idType, class valType>
// void CSR<idType, valType>::normalize_columns(){
//     thrust::device_vector<valType> d_col_sums(ncolumn, 0.0);
//     valType *col_sums_ptr = thrust::raw_pointer_cast(d_col_sums.data());

//     int block_size = 256;
//     int num_blocks = (nrow + block_size - 1) / block_size;

//     column_sum_kernel<<<num_blocks, block_size>>>(nrow, d_rpt, d_colids, d_values, col_sums_ptr);
//     normalize_kernel<<<num_blocks, block_size>>>(nrow, d_rpt, d_colids, d_values, col_sums_ptr);
// }

// Kernel for Hadamard power
// template <class idType, class valType>
// __global__ void hadamard_power_kernel(idType nnz, valType *values, valType power) {
//     idType i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < nnz) {
//         values[i] = pow(values[i], power);
//     }
// }

// template <class idType, class valType>
// void CSR<idType, valType>::hadamard_power(valType power) {
//     int block_size = 256;
//     int num_blocks = (nnz + block_size - 1) / block_size;
//     hadamard_power_kernel<<<num_blocks, block_size>>>(nnz, d_values, power);
// }

// Step 1: Kernel to compute new row sizes
// template <class idType, class valType>
// __global__ void compute_new_row_sizes(idType nrow, idType *rpt, valType *values, 
//                                       valType threshold, int top_k, idType *new_sizes) {
//     idType i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < nrow) {
//         idType row_start = rpt[i];
//         idType row_end = rpt[i+1];
//         idType new_size = 0;
//         for (idType j = row_start; j < row_end && new_size < top_k; ++j) {
//             if (values[j] > threshold) {
//                 ++new_size;
//             }
//         }
//         new_sizes[i] = new_size;
//     }
// }

// Step 2: Kernel to prune and copy data
// template <class idType, class valType>
// __global__ void prune_and_copy(idType nrow, idType *rpt, idType *colids, valType *values, 
//                                valType threshold, int top_k, 
//                                idType *new_rpt, idType *new_colids, valType *new_values) {
//     idType i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < nrow) {
//         idType row_start = rpt[i];
//         idType row_end = rpt[i+1];
//         idType new_row_start = new_rpt[i];
        
//         // Sort the row in-place
//         thrust::sort_by_key(thrust::device, 
//                             values + row_start, values + row_end, 
//                             colids + row_start, 
//                             thrust::greater<valType>());

//         // Copy elements above threshold, up to top_k
//         idType new_idx = new_row_start;
//         for (idType j = row_start; j < row_end && (new_idx - new_row_start) < top_k; ++j) {
//             if (values[j] > threshold) {
//                 new_values[new_idx] = values[j];
//                 new_colids[new_idx] = colids[j];
//                 ++new_idx;
//             }
//         }
//     }
// }

// Main function to prune and scale
// template <class idType, class valType>
// void CSR<idType, valType>::prune_and_scale(valType threshold, int top_k) {
//     thrust::device_vector<idType> d_new_sizes(nrow);
//     idType *new_sizes_ptr = thrust::raw_pointer_cast(d_new_sizes.data());

//     int block_size = 256;
//     int num_blocks = (nrow + block_size - 1) / block_size;

//     // Step 1: Compute new row sizes
//     compute_new_row_sizes<<<num_blocks, block_size>>>(nrow, d_rpt, d_values, threshold, top_k, new_sizes_ptr);

//     // Step 2: Compute new_rpt using prefix sum
//     thrust::device_vector<idType> d_new_rpt(nrow + 1);
//     thrust::exclusive_scan(d_new_sizes.begin(), d_new_sizes.end(), d_new_rpt.begin() + 1);
//     d_new_rpt[0] = 0;  // Ensure the first element is 0

//     // Get the new total number of non-zero elements
//     idType new_nnz = d_new_rpt.back() + d_new_sizes.back();

//     // Allocate memory for new CSR
//     idType *d_new_colids;
//     valType *d_new_values;
//     cudaMalloc(&d_new_colids, new_nnz * sizeof(idType));
//     cudaMalloc(&d_new_values, new_nnz * sizeof(valType));

//     // Step 3: Prune and copy data
//     idType *new_rpt_ptr = thrust::raw_pointer_cast(d_new_rpt.data());
//     prune_and_copy<<<num_blocks, block_size>>>(nrow, d_rpt, d_colids, d_values, threshold, top_k, 
//                                                new_rpt_ptr, d_new_colids, d_new_values);

//     // Update CSR structure
//     cudaFree(d_rpt);
//     cudaFree(d_colids);
//     cudaFree(d_values);
//     nnz = new_nnz;
//     d_rpt = new_rpt_ptr;
//     d_colids = d_new_colids;
//     d_values = d_new_values;
// }

template <class idType, class valType>
void CSR<idType, valType>::normalize_columns()
{
    vector<valType> col_sums(ncolumn, 0.0);
    for (idType i = 0; i < nrow; ++i) {
        for (idType j = rpt[i]; j < rpt[i + 1]; ++j) {
            col_sums[colids[j]] += values[j] ;
        }
    }
    for (idType i = 0; i < nrow; ++i) {
        for (idType j = rpt[i]; j < rpt[i + 1]; ++j) {
            values[j] /=col_sums[colids[j]];
        }
    }
}

template <class idType, class valType>
void CSR<idType, valType>::hadamard_power(valType power)
{
    for (idType i = 0; i < nnz; ++i) {
        values[i] = pow(values[i], power);
    }
}

template <class idType, class valType>
bool CSR<idType, valType>::is_converged(const CSR& other, valType epsilon)
{
    if (nnz != other.nnz) return false;
    for (idType i = 0; i < nnz; ++i) {
        if (abs(values[i] - other.values[i]) > epsilon) return false;
    }
    return true;
}

template <class idType, class valType>
void CSR<idType, valType>::prune_and_scale(valType threshold, int top_k)
{
    // Step 1: Create column-wise representation
    vector<vector<pair<valType, idType>>> col_values(ncolumn);
    for (idType i = 0; i < nrow; ++i) {
        for (idType j = rpt[i]; j < rpt[i + 1]; ++j) {
            if (values[j] > threshold) {
                col_values[colids[j]].push_back({values[j], i});
            }
        }
    }

    // Step 2: Sort and prune each column
    vector<vector<pair<valType, idType>>> pruned_cols(ncolumn);
    idType max_new_nnz = 0;
    for (idType col = 0; col < ncolumn; ++col) {
        sort(col_values[col].begin(), col_values[col].end(), greater<pair<valType, idType>>());
        idType col_nnz = min(static_cast<size_t>(top_k), col_values[col].size());
        pruned_cols[col] = vector<pair<valType, idType>>(col_values[col].begin(), col_values[col].begin() + col_nnz);
        max_new_nnz += col_nnz;
    }

    // Step 3: Create new CSR structure
    idType* new_rpt = new idType[nrow + 1]();
    idType* new_colids = new idType[max_new_nnz];
    valType* new_values = new valType[max_new_nnz];

    // Count entries per row
    for (idType col = 0; col < ncolumn; ++col) {
        for (const auto& entry : pruned_cols[col]) {
            ++new_rpt[entry.second + 1];
        }
    }

    // Cumulative sum for new_rpt
    for (idType i = 1; i <= nrow; ++i) {
        new_rpt[i] += new_rpt[i - 1];
    }

    // Fill new_colids and new_values
    vector<idType> row_counters(nrow, 0);
    for (idType col = 0; col < ncolumn; ++col) {
        for (const auto& entry : pruned_cols[col]) {
            idType row = entry.second;
            idType pos = new_rpt[row] + row_counters[row];
            new_colids[pos] = col;
            new_values[pos] = entry.first;
            ++row_counters[row];
        }
    }

    // Update CSR structure
    delete[] rpt;
    delete[] colids;
    delete[] values;
    nnz = new_rpt[nrow];
    rpt = new_rpt;
    colids = new_colids;
    values = new_values;
}

// template <class idType, class valType>
// void CSR<idType, valType>::correct_csr()
// {
//     idType new_row_count = 0;
//     vector<idType> new_rpt;
//     new_rpt.push_back(0);

//     for (idType i = 0; i < nrow; ++i) {
//         if (rpt[i] != rpt[i + 1]) {  // Non-empty row
//             for (idType j = rpt[i]; j < rpt[i + 1]; ++j) {
//                 colids[new_rpt.back()] = colids[j];
//                 values[new_rpt.back()] = values[j];
//                 new_rpt.back()++;
//             }
//             new_row_count++;
//             if (new_row_count < nrow) new_rpt.push_back(new_rpt.back());
//         }
//     }

//     delete[] rpt;
//     rpt = new idType[new_row_count + 1];
//     std::copy(new_rpt.begin(), new_rpt.end(), rpt);

//     nrow = new_row_count;
//     nnz = new_rpt.back();
// }

// template <class idType, class valType>
// vector<vector<idType>> CSR<idType, valType>::extract_clusters()
// {
//     correct_csr();
//     vector<vector<idType>> clusters;
//     vector<bool> is_attractor(nrow, false);
    
//     // First pass: Identify attractors
//     for (idType i = 0; i < nrow; ++i) {
//         if (values[rpt[i]] > 0) {  // Check if diagonal element is positive
//             is_attractor[i] = true;
//         }
//     }
    
//     // Second pass: Form clusters
//     for (idType i = 0; i < nrow; ++i) {
//         if (is_attractor[i]) {
//             vector<idType> cluster;
//             for (idType j = rpt[i]; j < rpt[i + 1]; ++j) {
//                 if (values[j] > 0) {
//                     cluster.push_back(colids[j]);
//                 }
//             }
//             clusters.push_back(cluster);
//         }
//     }
    
//     return clusters;
// }

template <class idType, class valType>
int CSR<idType, valType>::extract_clusters()
{
    //vector<vector<idType>> groups;
    //if (nrow == 0) return groups;

    //vector<idType> current_group;
    //current_group.push_back(0);  // Start with the first element
    int group_number=0;
    for (idType i = 1; i < nrow+1; ++i) {
        if (rpt[i] != rpt[i-1]) {
        //     // This row has the same pointer as the previous one, add it to the current group
        //     current_group.push_back(i);
        // } else {
        //     // This row has a different pointer, start a new group
        //     if (!current_group.empty()) {
        //         groups.push_back(current_group);
        //         current_group.clear();
        //     }
        //     current_group.push_back(i);
        // }
            group_number++;
        }

    // Add the last group if it's not empty    
    }
    return group_number;
}

template <class idType, class valType>
void CSR<idType, valType>::spmv_cpu(valType *x, valType *y)
{
    idType i, j;
    valType ans;
  
    for (i = 0; i < nrow; ++i) {
        ans = 0;
        for (j = 0; j < (rpt[i + 1] - rpt[i]); j++) {
            ans += values[rpt[i] + j] * x[colids[rpt[i] + j]];
        }
        y[i] = ans;
    }
}

#endif
