/**
 * File: Feature.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_FEATURE_H
#define MICRO_FUSION_FEATURE_H

#include <memory>

#include "Eigen/Eigen"
#include "opencv2/opencv.hpp"

namespace mc {
class FeatureExtr {
 public:
  using Ptr = std::shared_ptr<FeatureExtr>;
  using UniPtr = std::unique_ptr<FeatureExtr>;
  FeatureExtr() = default;
  virtual ~FeatureExtr() = default;
  [[nodiscard]] virtual std::vector<std::shared_ptr<Eigen::VectorXf>>
  GetTargetsFeature(std::vector<cv::Mat> const &imgs) const = 0;
};

}  // namespace mc
#endif  // MICRO_FUSION_FEATURE_H
