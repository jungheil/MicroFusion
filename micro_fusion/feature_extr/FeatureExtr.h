/**
 * File: FeatureExtr.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_FEATUREEXTR_H
#define MICRO_FUSION_FEATUREEXTR_H
#include <Eigen/Eigen>
#include <cassert>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>
#include <vector>

#include "Feature.h"
#include "ONNXInference.h"

namespace mc {

class OnnxFeatureExtr : public FeatureExtr {
 public:
  explicit OnnxFeatureExtr(std::string name, std::string model_path,
                           int intra_threads)
      : onnx_inference_(std::move(name), std::move(model_path), intra_threads) {
    assert(onnx_inference_.get_input_count() == 1);
    for (; output_idx_ < onnx_inference_.get_output_count(); ++output_idx_) {
      if (onnx_inference_.get_output_node_names()[output_idx_] == "output") {
        break;
      }
      assert(output_idx_ < onnx_inference_.get_output_count());
    };
  }
  ~OnnxFeatureExtr() override = default;
  [[nodiscard]] std::vector<std::shared_ptr<Eigen::VectorXf>> GetTargetsFeature(
      std::vector<cv::Mat> const &imgs) const override;

 private:
  ONNXInference onnx_inference_;
  size_t output_idx_ = 0;
};
}  // namespace mc

#endif  // MICRO_FUSION_FEATUREEXTR_H
