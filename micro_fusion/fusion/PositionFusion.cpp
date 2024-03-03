/**
 * File: PositionFusion.cpp
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#include "PositionFusion.h"

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>

namespace mc {
Eigen::Matrix3d PositionProbabilityByDist::covariance_ =
    Eigen::Matrix3d::Zero();
double PositionProbabilityByDist::num_cache_ = 0;
Eigen::Matrix3d PositionProbabilityByDist::inv_cache_ = Eigen::Matrix3d::Zero();

double PositionFusionBayes::GetConfidence(Eigen::Vector3d const &x_i,
                                          Eigen::Vector3d const &x_j,
                                          Eigen::Matrix3d const &covariance_i) {
  auto probability = [&](double x1, double x2, double x3) {
    return PositionProbabilityByDist::Get(Eigen::Vector3d(x1, x2, x3), x_i,
                                          covariance_i);
  };
  auto x_min = x_i.cwiseMin(x_j);
  auto x_max = x_i.cwiseMax(x_j);
  auto confidence =
      8 * boost::math::quadrature::trapezoidal(
              [&](double x1) {
                return boost::math::quadrature::trapezoidal(
                    [&](double x2) {
                      return boost::math::quadrature::trapezoidal(
                          [&](double x3) { return probability(x1, x2, x3); },
                          x_min(2), x_max(2), 1e-2);
                    },
                    x_min(1), x_max(1), 1e-2);
              },
              x_min(0), x_max(0), 1e-2);

  return confidence;
}

std::vector<int64_t>
PositionFusionBayes::GetConfidenceTarget(std::vector<DecTarget::Ptr> targets) {
  Eigen::MatrixXd confidence_matrix(targets.size(), targets.size());
  confidence_matrix.setZero();
  for (int i = 0; i < targets.size(); ++i) {
    for (int j = 0; j < targets.size(); ++j) {
      if (i == j) {
        confidence_matrix(i, j) = 0;
      } else {
        confidence_matrix(i, j) =
            GetConfidence(targets[i]->get_estimate_position(),
                          targets[j]->get_estimate_position(),
                          targets[i]->get_estimate_covariance());
      }
    }
  }
  //  cout << "confidence matrix: " << endl;
  //  cout << confidence_matrix << endl << endl;
  // TODO 需要根据上述置信度矩阵，选择合适的传感器，将其idx返回。

  auto ret = std::vector<int64_t>();
  ret.reserve(targets.size());
  for (int i = 0; i < targets.size(); ++i) {
    ret.push_back(i);
  }
  return ret;
}

Eigen::Vector3d PositionFusionBayes::Get(std::vector<DecTarget::Ptr> targets) {
  auto confidence_targets_idx = GetConfidenceTarget(targets);
  std::vector<DecTarget::Ptr> confidence_targets{};
  confidence_targets.reserve(confidence_targets_idx.size());
  for (auto idx : confidence_targets_idx) {
    confidence_targets.push_back(targets[idx]);
  }
  Eigen::Vector3d nsum;
  Eigen::Matrix3d dsum;
  nsum.setZero();
  dsum.setZero();
  for (auto &target : confidence_targets) {
    nsum += target->get_estimate_covariance().inverse() *
            target->get_estimate_position();
    dsum += target->get_estimate_covariance().inverse();
  }
  return dsum.inverse() * nsum;
}

} // namespace mc