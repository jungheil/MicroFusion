/**
 * File: PositionFusion.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_POSITIONFUSION_H
#define MICRO_FUSION_POSITIONFUSION_H

#include <Eigen/Eigen>
#include <memory>

#include "BaseObject.h"
#include "Common.h"

namespace mc {
// 计算位置概率密度
class PositionProbabilityByDist {
 public:
  static double Get(const Eigen::Vector3d &x, const Eigen::Vector3d &mu,
                    const Eigen::Matrix3d &covariance) {
    if (covariance != covariance_) {
      covariance_ = covariance;
      num_cache_ =
          1. / (pow(2 * M_PI, 1.5) * pow(covariance.determinant(), 0.5));
      inv_cache_ = covariance.inverse();
    }
    return num_cache_ *
           exp(-0.5 * (x - mu).transpose() * inv_cache_ * (x - mu));
  }

 private:
  static Eigen::Matrix3d covariance_;
  static double num_cache_;
  static Eigen::Matrix3d inv_cache_;
};
class PositionFusion {
 public:
  using Ptr = std::shared_ptr<PositionFusion>;
  using UniPtr = std::unique_ptr<PositionFusion>;

  PositionFusion() = default;
  virtual ~PositionFusion() = default;
  virtual Eigen::Vector3d Get(std::vector<DecTarget::Ptr>) = 0;
};
// 目标融合
class PositionFusionBayes : public PositionFusion {
 public:
  PositionFusionBayes() = default;
  ~PositionFusionBayes() override = default;
  static double GetConfidence(Eigen::Vector3d const &x_i,
                              Eigen::Vector3d const &x_j,
                              Eigen::Matrix3d const &covariance_i);

  static std::vector<int64_t> GetConfidenceTarget(
      std::vector<DecTarget::Ptr> targets);

  Eigen::Vector3d Get(std::vector<DecTarget::Ptr> targets) override;
};

}  // namespace mc

#endif  // POSITION_FILTER_POSITIONFUSION_H
