/**
 * File: BaseObject.cpp
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#include "BaseObject.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace mc {

DecTarget::DecTarget(UAVInfo::Ptr uav_ptr, cv::Mat img, std::string img_type,
                     Eigen::Vector3d measurement_position,
                     Eigen::Vector3d target_orientation,
                     float_t target_distance, double vertical_coff,
                     double horizontal_coff)
    : uuid_(boost::uuids::to_string(boost::uuids::random_generator()())),
      uav_info_(uav_ptr),
      img_(img),
      target_type_(img_type),
      measurement_position_(measurement_position),
      target_orientation_(target_orientation),
      target_distance_(target_distance) {
  estimate_position_ = CalcTargetPosition();
  estimate_covariance_ = CalcTargetCovariance(vertical_coff, horizontal_coff);
}

DecTarget::DecTarget(UAVInfo::Ptr uav_ptr, cv::Mat img, std::string img_type,
                     Eigen::Vector3d target_position)
    : uuid_(boost::uuids::to_string(boost::uuids::random_generator()())),
      uav_info_(uav_ptr),
      img_(img),
      target_type_(img_type) {
  estimate_position_ = target_position;
  estimate_covariance_ = Eigen::Matrix3d::Identity();
}

DecTargetSeries::DecTargetSeries(std::shared_ptr<UAVInfo> uav_info,
                                 size_t target_list_size)
    : uuid_(boost::uuids::to_string(boost::uuids::random_generator()())),
      uav_info_(uav_info),
      target_list_(target_list_size, 0) {}

FusTarget::FusTarget(size_t max_target_num, time_t max_cache_time,
                     size_t target_history_size)
    : uuid_(boost::uuids::to_string(boost::uuids::random_generator()())),
      target_series_list_(max_target_num, max_cache_time) {
  target_history_size_ = target_history_size;
};

// uint64_t DecTargetSeries::target_id_gen_ = 1;
// uint64_t FusTarget::target_id_gen_ = 1;

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
  target_list_.put(target_ptr->get_time_stamp(), target_ptr);
  img_feature_ptr_ = target_ptr->get_img_feature_ptr();
  img_type_ = target_ptr->get_target_type();
  update_time_ = target_ptr->get_time_stamp();
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
  update_time_ = target_ptr->get_time_stamp();
  target_type_ = target_ptr->get_target_type();
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

}  // namespace mc
