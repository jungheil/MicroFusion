/**
 * File: ONNXInference.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_ONNXINFERENCE_H
#define MICRO_FUSION_ONNXINFERENCE_H

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace mc {

class ONNXInference {
 public:
  ONNXInference(const std::string &name, const std::string &model_path,
                int intra_threads);

  [[nodiscard]] std::tuple<std::vector<std::vector<int64_t>>,
                           std::vector<std::vector<int64_t>>>
  GetIODims(uint8_t batchsize) const;

  [[nodiscard]] std::vector<float> Preprocess(const std::vector<cv::Mat> &imgs,
                                              size_t batchsize,
                                              uint8_t input_node_idx) const;

  [[nodiscard]] std::vector<std::vector<float>> Postprocess(
      const std::vector<float> &output_tensor_values, size_t batchsize,
      uint8_t output_node_idx) const;

  std::vector<std::vector<float>> Process(
      std::vector<std::vector<float>> &input_tensors, size_t batchsize) const;

  [[nodiscard]] size_t get_input_count() const { return input_count_; }
  [[nodiscard]] size_t get_output_count() const { return output_count_; }
  [[nodiscard]] std::vector<std::string> get_input_node_names() const {
    return input_node_names_;
  }
  [[nodiscard]] std::vector<std::string> get_output_node_names() const {
    return output_node_names_;
  }

 protected:
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator_;
  size_t input_count_;
  size_t output_count_;
  std::vector<std::string> input_node_names_;
  std::vector<std::string> output_node_names_;
  std::vector<std::vector<int64_t>> input_node_dims_;
  std::vector<std::vector<int64_t>> output_node_dims_;
};

}  // namespace mc

#endif  // MICRO_FUSION_ONNXINFERENCE_H
