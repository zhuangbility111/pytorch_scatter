#include "segment_csr_cpu.h"

#include "index_info.h"
#include "reducer.h"
#include "utils.h"

#include <iostream>
#include <limits>
#include <chrono>
#include <utility>
// #include <omp.h>
#define _OPENMP
#include <ATen/ParallelOpenMP.h>

using namespace std::chrono;

#ifdef __ARM_FEATURE_SVE
    #include <arm_sve.h>
    #define VEC_LEN 16
#endif /* __ARM_FEATURE_SVE */

void segment_csr_cpu_kernel_for_min(float* src_data, int64_t* indptr_data, float* out_data,
									int output_rows, int output_cols,
									int row_begin_on_indptr, int row_end_on_indptr,
									int col_begin, int col_end) {
    // outer loop along output row
    for (int idx_out = row_begin_on_indptr; idx_out < row_end_on_indptr; idx_out++) {
        int row_begin_on_src = static_cast<int>(indptr_data[idx_out]);
        int row_end_on_src = static_cast<int>(indptr_data[idx_out + 1]);

        // loop along output col
        for (int idx_col = col_begin; idx_col < col_end; idx_col += VEC_LEN) {
            svbool_t pg = svwhilelt_b32(idx_col, col_end);
            // svfloat32_t va = svld1(pg, &(out_data[idx_out * output_col + idx_col]));
            svfloat32_t va = svdup_f32(std::numeric_limits<float>::max());
            svfloat32_t vb;

            // inner loop along source row
            for (int idx_src_row = row_begin_on_src; idx_src_row < row_end_on_src; idx_src_row++) {
                vb = svld1(pg, &(src_data[idx_src_row * output_cols + idx_col]));
				__builtin_prefetch(&(src_data[(idx_src_row + 7) * output_cols + idx_col]), 0, 2);
                va = svmin_f32_x(pg, va, vb);
            }
            svst1(pg, &(out_data[idx_out * output_cols + idx_col]), va);
        }
    }
}

void segment_csr_cpu_kernel_for_max(float* src_data, int64_t* indptr_data, float* out_data,
									int output_rows, int output_cols,
									int row_begin_on_indptr, int row_end_on_indptr,
									int col_begin, int col_end) {
    // outer loop along output row
    for (int idx_out = row_begin_on_indptr; idx_out < row_end_on_indptr; idx_out++) {
        int row_begin_on_src = static_cast<int>(indptr_data[idx_out]);
        int row_end_on_src = static_cast<int>(indptr_data[idx_out + 1]);

        // loop along output col
        for (int idx_col = col_begin; idx_col < col_end; idx_col += VEC_LEN) {
            svbool_t pg = svwhilelt_b32(idx_col, col_end);
            // svfloat32_t va = svld1(pg, &(out_data[idx_out * output_col + idx_col]));
            svfloat32_t va = svdup_f32(std::numeric_limits<float>::lowest());
            svfloat32_t vb;

            // inner loop along source row
            for (int idx_src_row = row_begin_on_src; idx_src_row < row_end_on_src; idx_src_row++) {
                vb = svld1(pg, &(src_data[idx_src_row * output_cols + idx_col]));
				__builtin_prefetch(&(src_data[(idx_src_row + 7) * output_cols + idx_col]), 0, 2);
                va = svmax_f32_x(pg, va, vb);
            }
            svst1(pg, &(out_data[idx_out * output_cols + idx_col]), va);
        }
    }
}

void segment_csr_cpu_kernel_for_sum(float* src_data, int64_t* indptr_data, float* out_data,
									int output_rows, int output_cols,
									int row_begin_on_indptr, int row_end_on_indptr,
									int col_begin, int col_end) {
    // outer loop along output row
    for (int idx_out = row_begin_on_indptr; idx_out < row_end_on_indptr; idx_out++) {
        int row_begin_on_src = static_cast<int>(indptr_data[idx_out]);
        int row_end_on_src = static_cast<int>(indptr_data[idx_out + 1]);

        // loop along output col
        for (int idx_col = col_begin; idx_col < col_end; idx_col += VEC_LEN) {
            svbool_t pg = svwhilelt_b32(idx_col, col_end);
            // svfloat32_t va = svld1(pg, &(out_data[idx_out * output_col + idx_col]));
            svfloat32_t va = svdup_f32(0.0);
            svfloat32_t vb;

            // inner loop along source row
            for (int idx_src_row = row_begin_on_src; idx_src_row < row_end_on_src; idx_src_row++) {
                vb = svld1(pg, &(src_data[idx_src_row * output_cols + idx_col]));
				__builtin_prefetch(&(src_data[(idx_src_row + 7) * output_cols + idx_col]), 0, 2);
                va = svadd_f32_x(pg, va, vb);
            }
            svst1(pg, &(out_data[idx_out * output_cols + idx_col]), va);
        }
    }
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_csr_cpu_new_version(torch::Tensor src, torch::Tensor indptr,
							torch::optional<torch::Tensor> optional_out,
							std::string reduce) {
    auto start_time_all = system_clock::now();
    CHECK_CPU(src);
    CHECK_CPU(indptr);
    if (optional_out.has_value())
        CHECK_CPU(optional_out.value());

    CHECK_INPUT(src.dim() >= indptr.dim());

    auto sizes = indptr.sizes().vec();
    for (auto i = 0; i < indptr.dim() - 1; i++)
        sizes[i] = src.size(i);
    indptr = indptr.expand(sizes);

    auto dim = indptr.dim() - 1;

    src = src.contiguous();

    torch::Tensor out;
    if (optional_out.has_value()) {
        out = optional_out.value().contiguous();
        for (auto i = 0; i < out.dim(); i++)
            if (i != dim)
                CHECK_INPUT(src.size(i) == out.size(i));
        CHECK_INPUT(src.numel() == 0 || out.size(dim) == indptr.size(dim) - 1);
    } else {
        sizes = src.sizes().vec();
        sizes[dim] = std::max<int64_t>(indptr.size(dim) - 1, 0);
        // zeros ?
        out = torch::empty(sizes, src.options());
    }

    torch::optional<torch::Tensor> arg_out = torch::nullopt;
    int64_t *arg_out_data = nullptr;
    if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
        arg_out = torch::full(out.sizes(), src.size(dim), indptr.options());
        arg_out_data = arg_out.value().data_ptr<int64_t>();
    }

    if (src.numel() == 0) {
        if (!optional_out.has_value())
            out.fill_(0);
        return std::make_tuple(out, arg_out);
    }

    if (dim == 0 &&
        src.scalar_type() == at::ScalarType::Float &&
        out.scalar_type() == at::ScalarType::Float &&
        indptr.scalar_type() == at::ScalarType::Long &&
		(reduce2REDUCE.at(reduce) == SUM || reduce2REDUCE.at(reduce) == MAX || reduce2REDUCE.at(reduce) == MIN)) {
        // obtain data pointer
        float* src_data = src.data_ptr<float>();
        float* out_data = out.data_ptr<float>();
        int64_t* indptr_data = indptr.data_ptr<int64_t>();

		int output_rows = static_cast<int>(out.size(dim) * (indptr.numel() / indptr.size(-1)));
    	int output_cols = static_cast<int>(out.numel() / output_rows);
    	int src_rows    = static_cast<int>(src.size(dim));

        int max_num_threads = omp_get_max_threads();
        int num_threads_on_row, num_threads_on_col;

        // set the number of threads allocated on rows and cols
        init_num_threads(max_num_threads, num_threads_on_row, num_threads_on_col);
        int row_chunk_size = divup(output_rows, num_threads_on_row);
        int col_chunk_size = divup(output_cols, num_threads_on_col);

        auto start_time_1 = system_clock::now();
        // divide work on row based on num_threads_on_row
        int work_range_on_row[num_threads_on_row + 1];
        divide_work(work_range_on_row, indptr_data, indptr.numel(), src_rows, num_threads_on_row);
        duration<double, std::milli> diff = (system_clock::now() - start_time_1);
        std::cout << "elapsed time of dividing work: " << diff.count() << std::endl;

        auto start_time = system_clock::now();
		        /*
        #pragma omp parallel
        {
        */
        at::parallel_for(0, at::get_num_threads(), 0, [&](int64_t start, int64_t end) {
            int tid = at::get_thread_num();
            // int tid = omp_get_thread_num();
            // int num_threads = omp_get_num_threads();

            // obtain work range of each thread
            int row_begin_on_indptr = work_range_on_row[tid/num_threads_on_col];
            int row_end_on_indptr = work_range_on_row[tid/num_threads_on_col + 1];
            int col_begin = tid % num_threads_on_col * col_chunk_size;
            int col_end = std::min(output_cols, col_begin + col_chunk_size);

			if (reduce2REDUCE.at(reduce) == MIN)
				segment_csr_cpu_kernel_for_min(src_data, indptr_data, out_data, output_rows, output_cols,
											   row_begin_on_indptr, row_end_on_indptr, col_begin, col_end);
			else if (reduce2REDUCE.at(reduce) == MAX)
				segment_csr_cpu_kernel_for_max(src_data, indptr_data, out_data, output_rows, output_cols,
											   row_begin_on_indptr, row_end_on_indptr, col_begin, col_end);
			else if (reduce2REDUCE.at(reduce) == SUM)
				segment_csr_cpu_kernel_for_sum(src_data, indptr_data, out_data, output_rows, output_cols,
											   row_begin_on_indptr, row_end_on_indptr, col_begin, col_end);

        });
        duration<double, std::milli> diff1 = (system_clock::now() - start_time);
        std::cout << "elapsed time of kernel = " << diff1.count() << "ms" << std::endl;
    } else {
        // std::cout << "datatype of src or datatype of out is not float, or datatype of indptr is not long." << std::endl;
		auto N = out.size(dim) * (indptr.numel() / indptr.size(-1));
  		auto K = out.numel() / N;
  		auto E = src.size(dim);

  		auto indptr_info = getTensorInfo<int64_t>(indptr);
  		auto stride = indptr_info.strides[indptr_info.dims - 1];
  		std::vector<int64_t> args(K);
  		AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, src.scalar_type(), "_", [&] {
  		  auto src_data = src.data_ptr<scalar_t>();
  		  auto out_data = out.data_ptr<scalar_t>();

  		  std::vector<scalar_t> vals(K);
  		  int64_t row_start, row_end;
  		  AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
  		    for (auto n = 0; n < N; n++) {
  		      auto offset = IndexPtrToOffset<int64_t>::get(n, indptr_info);
  		      row_start = indptr_info.data[offset];
  		      row_end = indptr_info.data[offset + stride];

  		      offset = (n / (indptr.size(-1) - 1)) * E * K;
  		      for (auto k = 0; k < K; k++)
  		        vals[k] = Reducer<scalar_t, REDUCE>::init();

  		      for (auto e = row_start; e < row_end; e++)
  		        for (auto k = 0; k < K; k++)
  		          Reducer<scalar_t, REDUCE>::update(
  		              &vals[k], src_data[offset + e * K + k], &args[k], e);

  		      for (auto k = 0; k < K; k++)
  		        Reducer<scalar_t, REDUCE>::write(out_data + n * K + k, vals[k],
  		                                         arg_out_data + n * K + k, args[k],
  		                                         row_end - row_start);
  		    }
  		  });
  		});
    }
    duration<double, std::milli> diff = (system_clock::now() - start_time_all);
    std::cout << "total elapsed time of my method: " << diff.count() << std::endl;

    return std::make_tuple(out, arg_out);
}


std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_csr_cpu(torch::Tensor src, torch::Tensor indptr,
                torch::optional<torch::Tensor> optional_out,
                std::string reduce) {
  CHECK_CPU(src);
  CHECK_CPU(indptr);
  if (optional_out.has_value())
    CHECK_CPU(optional_out.value());

  CHECK_INPUT(src.dim() >= indptr.dim());

  auto sizes = indptr.sizes().vec();
  for (auto i = 0; i < indptr.dim() - 1; i++)
    sizes[i] = src.size(i);
  indptr = indptr.expand(sizes);

  auto dim = indptr.dim() - 1;

  src = src.contiguous();

  torch::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (auto i = 0; i < out.dim(); i++)
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
    CHECK_INPUT(src.numel() == 0 || out.size(dim) == indptr.size(dim) - 1);
  } else {
    sizes = src.sizes().vec();
    sizes[dim] = std::max<int64_t>(indptr.size(dim) - 1, 0);
    out = torch::empty(sizes, src.options());
  }

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full(out.sizes(), src.size(dim), indptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  if (src.numel() == 0) {
    if (!optional_out.has_value())
      out.fill_(0);
    return std::make_tuple(out, arg_out);
  }

  auto N = out.size(dim) * (indptr.numel() / indptr.size(-1));
  auto K = out.numel() / N;
  auto E = src.size(dim);

  // printf("target_dim = %d, output.size(dim) = %d, out.numel() / output.size(dim) = %d, src.size(dim) = %d\n", dim, N, K, E);

  auto indptr_info = getTensorInfo<int64_t>(indptr);
  auto stride = indptr_info.strides[indptr_info.dims - 1];
  std::vector<int64_t> args(K);
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, src.scalar_type(), "_", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    std::vector<scalar_t> vals(K);
    int64_t row_start, row_end;
    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      for (auto n = 0; n < N; n++) {
        auto offset = IndexPtrToOffset<int64_t>::get(n, indptr_info);
        row_start = indptr_info.data[offset];
        row_end = indptr_info.data[offset + stride];

        offset = (n / (indptr.size(-1) - 1)) * E * K;
        for (auto k = 0; k < K; k++)
          vals[k] = Reducer<scalar_t, REDUCE>::init();

        for (auto e = row_start; e < row_end; e++)
          for (auto k = 0; k < K; k++)
            Reducer<scalar_t, REDUCE>::update(
                &vals[k], src_data[offset + e * K + k], &args[k], e);

        for (auto k = 0; k < K; k++)
          Reducer<scalar_t, REDUCE>::write(out_data + n * K + k, vals[k],
                                           arg_out_data + n * K + k, args[k],
                                           row_end - row_start);
      }
    });
  });

  return std::make_tuple(out, arg_out);
}

torch::Tensor gather_csr_cpu(torch::Tensor src, torch::Tensor indptr,
                             torch::optional<torch::Tensor> optional_out) {
  CHECK_CPU(src);
  CHECK_CPU(indptr);
  if (optional_out.has_value())
    CHECK_CPU(optional_out.value());

  CHECK_INPUT(src.dim() >= indptr.dim());

  auto sizes = indptr.sizes().vec();
  for (auto i = 0; i < indptr.dim() - 1; i++)
    sizes[i] = src.size(i);
  indptr = indptr.expand(sizes);

  auto dim = indptr.dim() - 1;
  CHECK_INPUT(src.size(dim) == 0 || src.size(dim) == indptr.size(dim) - 1);

  src = src.contiguous();

  torch::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (auto i = 0; i < out.dim(); i++)
      if (i != dim)
        CHECK_INPUT(src.size(i) == out.size(i));
  } else {
    auto sizes = src.sizes().vec();
    if (src.numel() > 0)
      sizes[dim] = *indptr.flatten()[-1].data_ptr<int64_t>();
    else
      sizes[dim] = 0;
    out = torch::empty(sizes, src.options());
  }

  if (src.numel() == 0) {
    if (!optional_out.has_value())
      out.fill_(0);
    return out;
  }

  auto N = src.size(dim) * (indptr.numel() / indptr.size(-1));
  auto K = src.numel() / N;
  auto E = out.size(dim);

  auto indptr_info = getTensorInfo<int64_t>(indptr);
  auto stride = indptr_info.strides[indptr_info.dims - 1];
  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, src.scalar_type(), "_", [&] {
    auto src_data = src.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    std::vector<scalar_t> vals(K);
    int64_t row_start, row_end;
    for (auto n = 0; n < N; n++) {
      auto offset = IndexPtrToOffset<int64_t>::get(n, indptr_info);
      row_start = indptr_info.data[offset];
      row_end = indptr_info.data[offset + stride];

      for (auto k = 0; k < K; k++)
        vals[k] = src_data[n * K + k];

      offset = (n / (indptr.size(-1) - 1)) * E * K;
      for (auto e = row_start; e < row_end; e++)
        for (auto k = 0; k < K; k++)
          out_data[offset + e * K + k] = vals[k];
    }
  });

  return out;
}
