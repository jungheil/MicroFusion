/**
 * File: Toolkit.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_TOOLKIT_H
#define MICRO_FUSION_TOOLKIT_H

#include <Eigen/Eigen>
#include <tuple>

namespace mc {
std::tuple<Eigen::Vector3d, Eigen::Vector3d, double> GetRandomTarget(
    Eigen::Vector3d target_position) {
  Eigen::Vector3d target_orientation =
      Eigen::Vector3d::Random() * (rand() % 100);
  Eigen::Vector3d measurement_position = target_position - target_orientation;
  double target_distance = target_orientation.norm();
  return std::make_tuple(measurement_position, target_orientation.normalized(),
                         target_distance);
}

}  // namespace mc
#endif  // MICRO_FUSION_TOOLKIT_H
