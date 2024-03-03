/**
 * File: BaseObject.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_BASEOBJECT_H
#define MICRO_FUSION_BASEOBJECT_H

#include <Eigen/Eigen>
#include <cstdint>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "Common.h"

namespace mc {

class UAVInfo {
public:
  using Ptr = std::shared_ptr<UAVInfo>;
  explicit UAVInfo(uint64_t uav_id) : uav_id_(uav_id) {}

  [[nodiscard]] uint64_t get_uav_id() const { return uav_id_; }

protected:
  uint64_t uav_id_;
};

class DecTarget {
public:
  using Ptr = std::shared_ptr<DecTarget>;

  DecTarget(UAVInfo::Ptr uav_ptr, cv::Mat img,
            Eigen::Vector3d measurement_position,
            Eigen::Vector3d target_orientation, float_t target_distance,
            double vertical_coff = 0.05, double horizontal_coff = 0.01)
      : uav_info_(uav_ptr), img_(img),
        measurement_position_(measurement_position),
        target_orientation_(target_orientation),
        target_distance_(target_distance) {
    estimate_position_ = CalcTargetPosition();
    estimate_covariance_ = CalcTargetCovariance(vertical_coff, horizontal_coff);
  }

public:
  [[nodiscard]] Eigen::Vector3d CalcTargetPosition() const;

  [[nodiscard]] Eigen::Matrix3d
  CalcTargetCovariance(float_t vertical_coff, float_t horizontal_coff) const;

  [[nodiscard]] const Eigen::Vector3d &get_estimate_position() const {
    return estimate_position_;
  }

  [[nodiscard]] const Eigen::Matrix3d get_estimate_covariance() const {
    return estimate_covariance_;
  }

  [[nodiscard]] time_t get_update_time() const { return update_time_; }

  [[nodiscard]] const UAVInfo::Ptr get_uav_info() const { return uav_info_; }

  [[nodiscard]] const cv::Mat &get_img() const { return img_; }

  void set_img_feature_ptr(std::shared_ptr<Eigen::VectorXf> feature) {
    img_feature_ptr_ = feature;
  }
  void set_feature_type(std::string feature_type) {
    img_feature_type_ = feature_type;
  }
  [[nodiscard]] std::shared_ptr<Eigen::VectorXf> get_img_feature_ptr() const {
    return img_feature_ptr_;
  }
  [[nodiscard]] const std::string &get_img_feature_type() const {
    return img_feature_type_;
  }

protected:
  UAVInfo::Ptr uav_info_;
  // uint64_t target_id_ = 0;

  cv::Mat img_;
  std::shared_ptr<Eigen::VectorXf> img_feature_ptr_;
  std::string img_feature_type_ = "";

  // 原始测量信息
  Eigen::Vector3d measurement_position_;
  Eigen::Vector3d target_orientation_;
  float_t target_distance_;

  // 估计信息
  Eigen::Vector3d estimate_position_;
  Eigen::Matrix3d estimate_covariance_;
  Eigen::Vector3d estimate_velocity_;

  time_t update_time_ = time(nullptr);
};

class DecTargetSeries {
public:
  using Ptr = std::shared_ptr<DecTargetSeries>;

  explicit DecTargetSeries(std::shared_ptr<UAVInfo> uav_info,
                           size_t target_list_size)
      : uav_target_id_(target_id_gen_++), uav_info_(uav_info),
        target_list_(target_list_size, 0) {}

  void AddTarget(DecTarget::Ptr target_ptr);

  [[nodiscard]] DecTarget::Ptr GetLatestTarget() {
    return target_list_.get_data().back().second;
  };

  [[nodiscard]] std::shared_ptr<Eigen::VectorXf> get_img_feature_ptr() const {
    return img_feature_ptr_;
  }
  [[nodiscard]] const std::string &get_img_feature_type() const {
    return img_feature_type_;
  }

protected:
  LRUCache<time_t, DecTarget::Ptr> target_list_;
  std::shared_ptr<UAVInfo> uav_info_;
  uint64_t uav_target_id_;
  time_t update_time_ = time(nullptr);

  std::shared_ptr<Eigen::VectorXf> img_feature_ptr_;
  std::string img_feature_type_;

private:
  static uint64_t target_id_gen_;
};

class FusTarget {
public:
  using Ptr = std::shared_ptr<FusTarget>;

  FusTarget(size_t max_target_num, time_t max_cache_time,
            size_t target_history_size)
      : target_id_(target_id_gen_++),
        target_series_list_(max_target_num, max_cache_time) {
    target_history_size_ = target_history_size;
  };

  void AddTarget(DecTarget::Ptr target_ptr);

  [[nodiscard]] std::vector<DecTargetSeries::Ptr> GetAllTargetSeries() {
    std::vector<DecTargetSeries::Ptr> series_ptr_list_;
    auto data = target_series_list_.get_data();
    series_ptr_list_.reserve(data.size());
    for (auto &d : data) {
      series_ptr_list_.push_back(d.second);
    }
    return series_ptr_list_;
  };

  [[nodiscard]] DecTargetSeries::Ptr GetLatestSeries() {
    auto data = target_series_list_.get_data();
    return data.front().second;
  }

  [[nodiscard]] DecTargetSeries::Ptr GetTargetSeries(uint64_t uav_id);
  [[nodiscard]] DecTargetSeries::Ptr GetTargetSeries();

  [[nodiscard]] uint64_t get_target_id() const { return target_id_; }

  [[nodiscard]] time_t get_update_time() const { return update_time_; }

  void set_estimate_position(Eigen::Vector3d position) {
    estimate_position_ = position;
  };

  [[nodiscard]] const Eigen::Vector3d &get_estimate_position() const {
    return estimate_position_;
  }

protected:
  LRUCache<uint64_t, DecTargetSeries::Ptr> target_series_list_;
  Eigen::Vector3d estimate_position_;
  std::vector<std::shared_ptr<Eigen::VectorXf>> img_feature_ptr_list_;
  uint64_t target_id_;
  time_t update_time_ = time(nullptr);
  // bool is_tracked_ = false;

  size_t target_history_size_;

private:
  static uint64_t target_id_gen_;
};

} // namespace mc

#endif // MICRO_FUSION_BASEOBJECT_H
