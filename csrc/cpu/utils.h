#pragma once

#include <torch/extension.h>

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

inline int divup(int x, int y) {
    return (x + y - 1) / y;
}

void divide_work(int* work_range, int64_t* indptr, int len_indptr, int total_num_work, int num_threads) {
    int cur_row = 0;
    int chunk_size = 0;
    int begin_row;
    work_range[0] = 0;
    work_range[num_threads] = len_indptr-1;
    for (int i = 0; i < num_threads-1; i++) {
        begin_row= work_range[i];
        chunk_size = divup(total_num_work, num_threads - i);
        while (indptr[cur_row] - indptr[begin_row] < chunk_size && cur_row < (len_indptr-1))
            ++cur_row;
        work_range[i+1] = cur_row;
        total_num_work -= (indptr[cur_row] - indptr[begin_row]);
    }
}

void init_num_threads(int& max_num_threads, int& num_threads_on_row, int& num_threads_on_col) {
    switch(max_num_threads) {
        case 1: {
            num_threads_on_row = 1;
            num_threads_on_col = 1;
        }break;
        case 2: {
            num_threads_on_row = 2;
            num_threads_on_col = 1;
        }break;
        case 4: {
            num_threads_on_row = 4;
            num_threads_on_col = 1;
        }break;
        case 8: {
            num_threads_on_row = 8;
            num_threads_on_col = 1;
        }break;
        case 12: {
            num_threads_on_row = 12;
            num_threads_on_col = 1;
        }break;
        case 24: {
            num_threads_on_row = 12;
            num_threads_on_col = 2;
        }break;
        case 36: {
            num_threads_on_row = 12;
            num_threads_on_col = 3;
        }break;
        case 48: {
            num_threads_on_row = 48;
            num_threads_on_col = 1;
        }break;
        default: {
            num_threads_on_row = max_num_threads;
            num_threads_on_col = 1;
        }break;
    }
}
