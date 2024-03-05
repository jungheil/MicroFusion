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
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
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

  DecTarget(UAVInfo::Ptr uav_ptr, cv::Mat img, std::string img_type,
            Eigen::Vector3d measurement_position,
            Eigen::Vector3d target_orientation, float_t target_distance,
            double vertical_coff = 0.05, double horizontal_coff = 0.01)
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

  DecTarget(UAVInfo::Ptr uav_ptr, cv::Mat img, std::string img_type,
            Eigen::Vector3d target_position)
      : uuid_(boost::uuids::to_string(boost::uuids::random_generator()())),
        uav_info_(uav_ptr),
        img_(img),
        target_type_(img_type) {
    estimate_position_ = target_position;
    estimate_covariance_ = Eigen::Matrix3d::Identity();
  }

 public:
  [[nodiscard]] Eigen::Vector3d CalcTargetPosition() const;

  [[nodiscard]] Eigen::Matrix3d CalcTargetCovariance(
      float_t vertical_coff, float_t horizontal_coff) const;

  [[nodiscard]] const Eigen::Vector3d &get_estimate_position() const {
    return estimate_position_;
  }

  [[nodiscard]] const Eigen::Matrix3d get_estimate_covariance() const {
    return estimate_covariance_;
  }

  [[nodiscard]] time_t get_time_stamp() const { return time_stamp_; }

  void set_time_stamp(time_t time_stamp) { time_stamp_ = time_stamp; }

  [[nodiscard]] const UAVInfo::Ptr get_uav_info() const { return uav_info_; }

  [[nodiscard]] const cv::Mat &get_img() const { return img_; }

  void set_img_feature_ptr(std::shared_ptr<Eigen::VectorXf> feature) {
    img_feature_ptr_ = feature;
  }
  [[nodiscard]] std::shared_ptr<Eigen::VectorXf> get_img_feature_ptr() const {
    return img_feature_ptr_;
  }
  [[nodiscard]] const std::string &get_target_type() const {
    return target_type_;
  }

 protected:
  UAVInfo::Ptr uav_info_;
  std::string uuid_;

  cv::Mat img_;
  std::shared_ptr<Eigen::VectorXf> img_feature_ptr_;
  std::string target_type_;

  // 原始测量信息
  Eigen::Vector3d measurement_position_;
  Eigen::Vector3d target_orientation_;
  float_t target_distance_;

  // 估计信息
  Eigen::Vector3d estimate_position_;
  Eigen::Matrix3d estimate_covariance_;
  Eigen::Vector3d estimate_velocity_;

  time_t time_stamp_ = time(nullptr);
};

class DecTargetSeries {
 public:
  using Ptr = std::shared_ptr<DecTargetSeries>;

  explicit DecTargetSeries(std::shared_ptr<UAVInfo> uav_info,
                           size_t target_list_size)
      : uuid_(boost::uuids::to_string(boost::uuids::random_generator()())),
        uav_info_(uav_info),
        target_list_(target_list_size, 0) {}

  void AddTarget(DecTarget::Ptr target_ptr);

  [[nodiscard]] DecTarget::Ptr GetLatestTarget() {
    return target_list_.get_data().back().second;
  };

  [[nodiscard]] std::shared_ptr<Eigen::VectorXf> get_img_feature_ptr() const {
    return img_feature_ptr_;
  }
  [[nodiscard]] const std::string &get_img_type() const { return img_type_; }

 protected:
  LRUCache<time_t, DecTarget::Ptr> target_list_;
  std::shared_ptr<UAVInfo> uav_info_;
  std::string uuid_;
  std::string target_type_;
  time_t update_time_ = time(nullptr);

  std::shared_ptr<Eigen::VectorXf> img_feature_ptr_;
  std::string img_type_;
};

class FusTarget {
 public:
  using Ptr = std::shared_ptr<FusTarget>;

  FusTarget(size_t max_target_num, time_t max_cache_time,
            size_t target_history_size)
      : uuid_(boost::uuids::to_string(boost::uuids::random_generator()())),
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

  [[nodiscard]] const std::string &get_uuid() const { return uuid_; }

  [[nodiscard]] time_t get_update_time() const { return update_time_; }

  void set_estimate_position(Eigen::Vector3d position) {
    estimate_position_ = position;
  };

  [[nodiscard]] const Eigen::Vector3d &get_estimate_position() const {
    return estimate_position_;
  }

  [[nodiscard]] const std::string &get_target_type() const {
    return target_type_;
  }

 protected:
  LRUCache<uint64_t, DecTargetSeries::Ptr> target_series_list_;
  Eigen::Vector3d estimate_position_;
  std::vector<std::shared_ptr<Eigen::VectorXf>> img_feature_ptr_list_;
  std::string uuid_;
  std::string target_type_;
  time_t update_time_ = time(nullptr);
  // bool is_tracked_ = false;

  size_t target_history_size_;

  // private:
  //   static uint64_t target_id_gen_;
};

}  // namespace mc

#endif  // MICRO_FUSION_BASEOBJECT_H
