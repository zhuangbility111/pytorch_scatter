#include <Python.h>
#include <torch/script.h>
#include <chrono>
using namespace std::chrono;

#include "cpu/segment_csr_cpu.h"
#include "utils.h"

#ifdef WITH_CUDA
#include "cuda/segment_csr_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__segment_csr_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__segment_csr_cpu(void) { return NULL; }
#endif
#endif

void check_error(float* src, float* dest, int64_t row, int64_t col, float err) {
	float total_diff = 0.0;
    for (int64_t i = 0; i < row; i++) {
        for (int64_t j = 0; j < col; j++) {
            float diff = std::fabs(src[i*col + j] - dest[i*col + j]);
            if (diff > err) {
                total_diff += diff;
                std::cout << "On row = " << i << ", col = " << j
                          << ", src = " << src[i*col + j]
                          << ", dest = " << dest[i*col + j]
                          << ", diff = " << diff << std::endl;
            }
        }
    }
    std::cout << "total_diff = " << total_diff << std::endl;
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_csr_fw(torch::Tensor src, torch::Tensor indptr,
               torch::optional<torch::Tensor> optional_out,
               std::string reduce) {
  if (src.device().is_cuda()) {
#ifdef WITH_CUDA
    return segment_csr_cuda(src, indptr, optional_out, reduce);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
	// auto start_time_all = system_clock::now();
	std::tuple<torch::Tensor, torch::optional<torch::Tensor>> tmp1 = 
		segment_csr_cpu(src, indptr, optional_out, reduce);
	// duration<double, std::milli> diff = (system_clock::now() - start_time_all);
    // std::cout << "total elapsed time of original segment_csr: " << diff.count() << std::endl;

	/*
	auto start_time_new = system_clock::now();
	std::tuple<torch::Tensor, torch::optional<torch::Tensor>> tmp1 = 
		segment_csr_cpu_new_version(src, indptr, optional_out, reduce);
	duration<double, std::milli> diff_new = (system_clock::now() - start_time_new);
    std::cout << "total elapsed time of my segment_csr: " << diff_new.count() << std::endl;
	*/

/*
	float* src_data = std::get<0>(tmp).data_ptr<float>();
	float* dest_data = std::get<0>(tmp1).data_ptr<float>();
	auto row = indptr.numel() - 1;
	auto col = src.numel() / src.size(0);
	check_error(src_data, dest_data, row, col, 0.00001);
*/

	return tmp1;
  }
}

torch::Tensor gather_csr_fw(torch::Tensor src, torch::Tensor indptr,
                            torch::optional<torch::Tensor> optional_out) {
  if (src.device().is_cuda()) {
#ifdef WITH_CUDA
    return gather_csr_cuda(src, indptr, optional_out);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return gather_csr_cpu(src, indptr, optional_out);
  }
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SegmentSumCSR : public torch::autograd::Function<SegmentSumCSR> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable indptr,
                               torch::optional<Variable> optional_out) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto out = std::get<0>(segment_csr_fw(src, indptr, optional_out, "sum"));
    ctx->save_for_backward({indptr});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto indptr = saved[0];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    auto grad_in = torch::empty(src_shape, grad_out.options());
    gather_csr_fw(grad_out, indptr, grad_in);
    return {grad_in, Variable(), Variable()};
  }
};

class SegmentMeanCSR : public torch::autograd::Function<SegmentMeanCSR> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable indptr,
                               torch::optional<Variable> optional_out) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto out = std::get<0>(segment_csr_fw(src, indptr, optional_out, "mean"));
    ctx->save_for_backward({indptr});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto indptr = saved[0];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    auto grad_in = torch::empty(src_shape, grad_out.options());
    if (grad_in.numel() > 0) {
      gather_csr_fw(grad_out, indptr, grad_in);
      auto indptr1 = indptr.narrow(-1, 0, indptr.size(-1) - 1);
      auto indptr2 = indptr.narrow(-1, 1, indptr.size(-1) - 1);
      auto count = (indptr2 - indptr1).to(grad_in.options());
      count = gather_csr_fw(count, indptr, torch::nullopt);
      for (auto i = 0; i < grad_out.dim() - indptr.dim(); i++)
        count = count.unsqueeze(-1);
      grad_in.true_divide_(count);
    }
    return {grad_in, Variable(), Variable()};
  }
};

class SegmentMinCSR : public torch::autograd::Function<SegmentMinCSR> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable indptr,
                               torch::optional<Variable> optional_out) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = segment_csr_fw(src, indptr, optional_out, "min");
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result).value();
    ctx->save_for_backward({indptr, arg_out});
    ctx->mark_non_differentiable({arg_out});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out, arg_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto indptr = saved[0];
    auto arg_out = saved[1];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[indptr.dim() - 1] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(indptr.dim() - 1, arg_out, grad_out);
    grad_in =
        grad_in.narrow(indptr.dim() - 1, 0, src_shape[indptr.dim() - 1] - 1);
    return {grad_in, Variable(), Variable()};
  }
};

class SegmentMaxCSR : public torch::autograd::Function<SegmentMaxCSR> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable indptr,
                               torch::optional<Variable> optional_out) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto result = segment_csr_fw(src, indptr, optional_out, "max");
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result).value();
    ctx->save_for_backward({indptr, arg_out});
    ctx->mark_non_differentiable({arg_out});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out, arg_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto indptr = saved[0];
    auto arg_out = saved[1];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());
    src_shape[indptr.dim() - 1] += 1;
    auto grad_in = torch::zeros(src_shape, grad_out.options());
    grad_in.scatter_(indptr.dim() - 1, arg_out, grad_out);
    grad_in =
        grad_in.narrow(indptr.dim() - 1, 0, src_shape[indptr.dim() - 1] - 1);
    return {grad_in, Variable(), Variable()};
  }
};

class GatherCSR : public torch::autograd::Function<GatherCSR> {
public:
  static variable_list forward(AutogradContext *ctx, Variable src,
                               Variable indptr,
                               torch::optional<Variable> optional_out) {
    ctx->saved_data["src_shape"] = src.sizes();
    auto out = gather_csr_fw(src, indptr, optional_out);
    ctx->save_for_backward({indptr});
    if (optional_out.has_value())
      ctx->mark_dirty({optional_out.value()});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto indptr = saved[0];
    auto src_shape = list2vec(ctx->saved_data["src_shape"].toIntList());

    auto grad_in = torch::empty(src_shape, grad_out.options());
    segment_csr_fw(grad_out, indptr, grad_in, "sum");
    return {grad_in, Variable(), Variable()};
  }
};

torch::Tensor segment_sum_csr(torch::Tensor src, torch::Tensor indptr,
                              torch::optional<torch::Tensor> optional_out) {
  return SegmentSumCSR::apply(src, indptr, optional_out)[0];
}

torch::Tensor segment_mean_csr(torch::Tensor src, torch::Tensor indptr,
                               torch::optional<torch::Tensor> optional_out) {
  return SegmentMeanCSR::apply(src, indptr, optional_out)[0];
}

std::tuple<torch::Tensor, torch::Tensor>
segment_min_csr(torch::Tensor src, torch::Tensor indptr,
                torch::optional<torch::Tensor> optional_out) {
  auto result = SegmentMinCSR::apply(src, indptr, optional_out);
  return std::make_tuple(result[0], result[1]);
}

std::tuple<torch::Tensor, torch::Tensor>
segment_max_csr(torch::Tensor src, torch::Tensor indptr,
                torch::optional<torch::Tensor> optional_out) {
  auto result = SegmentMaxCSR::apply(src, indptr, optional_out);
  return std::make_tuple(result[0], result[1]);
}

torch::Tensor gather_csr(torch::Tensor src, torch::Tensor indptr,
                         torch::optional<torch::Tensor> optional_out) {
  return GatherCSR::apply(src, indptr, optional_out)[0];
}

static auto registry =
    torch::RegisterOperators()
        .op("torch_scatter::segment_sum_csr", &segment_sum_csr)
        .op("torch_scatter::segment_mean_csr", &segment_mean_csr)
        .op("torch_scatter::segment_min_csr", &segment_min_csr)
        .op("torch_scatter::segment_max_csr", &segment_max_csr)
        .op("torch_scatter::gather_csr", &gather_csr);
