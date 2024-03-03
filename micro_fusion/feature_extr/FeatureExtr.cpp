/**
 * File: FeatureExtr.cpp
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#include "FeatureExtr.h"

#include <Eigen/Eigen>
#include <memory>

namespace mc {
std::vector<std::shared_ptr<Eigen::VectorXf>>
OnnxFeatureExtr::GetTargetsFeature(std::vector<cv::Mat> const &imgs) const {
  std::vector<std::shared_ptr<Eigen::VectorXf>> targets_feature;
  auto image_tensor = onnx_inference_.Preprocess(imgs, 0);
  auto input = std::vector<std::vector<float>>{image_tensor};
  auto output = onnx_inference_.Process(input);

  auto features =
      onnx_inference_.Postprocess(output.at(output_idx_), output_idx_);
  for (auto &f : features) {
    auto feature = Eigen::Map<Eigen::VectorXf>(f.data(), f.size());
    feature = feature.normalized();
    targets_feature.emplace_back(std::make_shared<Eigen::VectorXf>(feature));
  }
  return targets_feature;
}
} // namespace mc