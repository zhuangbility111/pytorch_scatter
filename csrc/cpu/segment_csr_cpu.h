#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_csr_cpu_new_version(torch::Tensor src, torch::Tensor indptr,
							torch::optional<torch::Tensor> optional_out,
							std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
segment_csr_cpu(torch::Tensor src, torch::Tensor indptr,
                torch::optional<torch::Tensor> optional_out,
                std::string reduce);

torch::Tensor gather_csr_cpu(torch::Tensor src, torch::Tensor indptr,
                             torch::optional<torch::Tensor> optional_out);
