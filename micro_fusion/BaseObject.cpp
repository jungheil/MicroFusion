/**
 * File: BaseObject.cpp
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#include "BaseObject.h"

namespace mc {

uint64_t DecTargetSeries::target_id_gen_ = 1;
uint64_t FusTarget::target_id_gen_ = 1;

Eigen::Vector3d DecTarget::CalcTargetPosition() const {
  return measurement_position_ +
         target_orientation_.normalized() * target_distance_;
}

Eigen::Matrix3d DecTarget::CalcTargetCovariance(float_t vertical_coff,
                                                float_t horizontal_coff) const {
  Eigen::Matrix3d target_covariance;
  target_covariance.setZero();
  target_covariance(0, 0) = vertical_coff * target_distance_;
  target_covariance(1, 1) = horizontal_coff * target_distance_;
  target_covariance(2, 2) = horizontal_coff * target_distance_;

  auto R = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(1, 0, 0),
                                              target_orientation_.normalized())
               .toRotationMatrix();

  target_covariance = R * target_covariance * R.transpose();

  return target_covariance;
}

void DecTargetSeries::AddTarget(DecTarget::Ptr target_ptr) {
  assert(target_ptr->get_uav_info()->get_uav_id() == uav_info_->get_uav_id() &&
         "UAV ID not match!");
  target_list_.put(target_ptr->get_update_time(), target_ptr);
  img_feature_ptr_ = target_ptr->get_img_feature_ptr();
  img_feature_type_ = target_ptr->get_img_feature_type();
  update_time_ = target_ptr->get_update_time();
}

void FusTarget::AddTarget(DecTarget::Ptr target_ptr) {
  if (target_series_list_.exist(target_ptr->get_uav_info()->get_uav_id())) {
    target_series_list_.get(target_ptr->get_uav_info()->get_uav_id())
        ->AddTarget(target_ptr);
  } else {
    auto target_series_ptr = std::make_shared<DecTargetSeries>(
        target_ptr->get_uav_info(), target_history_size_);
    target_series_ptr->AddTarget(target_ptr);
    target_series_list_.put(target_ptr->get_uav_info()->get_uav_id(),
                            target_series_ptr);
  }
  update_time_ = target_ptr->get_update_time();
}
DecTargetSeries::Ptr FusTarget::GetTargetSeries(uint64_t uav_id) {
  if (target_series_list_.exist(uav_id)) {
    return target_series_list_.get(uav_id);
  } else {
    return nullptr;
  }
}

DecTargetSeries::Ptr FusTarget::GetTargetSeries() {
  if (target_series_list_.size()) {
    return target_series_list_.get_head();
  } else {
    return nullptr;
  }
}

} // namespace mc
