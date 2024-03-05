/**
 * File: FusionParts.cpp
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#include "FusionParts.h"

#include <cstddef>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "BaseObject.h"
#include "feature_extr/FeatureExtr.h"

namespace mc {

BaseFeatureExtrPart::BaseFeatureExtrPart(
    std::unordered_map<std::string, std::string> weights_path,
    int intra_threads) {
  assert(weights_path.find("common") != weights_path.end());
  extr_map_["common"] = std::make_unique<OnnxFeatureExtr>(
      "common", weights_path.find("common")->second, intra_threads);
};

// TODO 根据设定类型选择不同的特征提取器
void BaseFeatureExtrPart::Get(std::vector<DecTarget::Ptr> &targets) const {
  std::vector<cv::Mat> imgs;
  imgs.reserve(targets.size());
  for (auto &t : targets) {
    imgs.push_back(t->get_img());
  }
  auto feature_list = extr_map_.find("common")->second->GetTargetsFeature(imgs);
  for (size_t i = 0; i < feature_list.size(); ++i) {
    targets[i]->set_img_feature_ptr(feature_list[i]);
  }
}
void BasePositionPart::Get(std::vector<FusTarget::Ptr> &targets) const {
  time_t now_time = time(nullptr);
  for (auto &t : targets) {
    if (now_time - t->get_update_time() <= active_time_) {
      auto target_series = t->GetAllTargetSeries();
      std::vector<DecTarget::Ptr> active_targets;
      for (auto &s : target_series) {
        active_targets.push_back(s->GetLatestTarget());
      }
      auto position = position_fusion_ptr_->Get(active_targets);
      t->set_estimate_position(position);
    }
  }
}

std::vector<std::vector<double>> BaseTargetMatchPart::ClacCosts(
    std::vector<DecTarget::Ptr> const &targets,
    std::vector<FusTarget::Ptr> const &his_targets) const {
  std::vector<std::vector<double>> cost_matrix(
      targets.size(),
      std::vector<double>(his_targets.size(),
                          std::numeric_limits<double>::infinity()));
  for (size_t i = 0; i < targets.size(); ++i) {
    for (size_t j = 0; j < his_targets.size(); ++j) {
      DecTargetSeries::Ptr target_series = his_targets[j]->GetTargetSeries(
          targets[i]->get_uav_info()->get_uav_id());
      if (target_series == nullptr) {
        target_series = his_targets[j]->GetLatestSeries();
      }
      assert(target_series != nullptr && "Target series not found!");
      if (targets[i]->get_target_type() == target_series->get_img_type()) {
        auto l2_distance =
            (*(target_series->GetLatestTarget()->get_img_feature_ptr()) -
             *(targets[i]->get_img_feature_ptr()))
                .norm();
        cost_matrix[i][j] = l2_distance;
      }
    }
  }
  return cost_matrix;
}

std::vector<std::string> BaseTargetMatchPart::Get(
    std::vector<DecTarget::Ptr> &targets,
    std::vector<FusTarget::Ptr> &his_targets_dict_) const {
  std::vector<std::string> ret(targets.size());
  auto groups = subgroup_ptr_->Get(targets, his_targets_dict_);
  for (auto &g : groups) {
    auto costs = ClacCosts(g.first, g.second);
    auto assignment = assignment_ptr_->Get(costs);
    for (auto const &a : assignment) {
      auto source_feature = g.first[a.first]->get_img_feature_ptr();
      auto target_series = g.second[a.second]->GetTargetSeries(
          g.first[a.first]->get_uav_info()->get_uav_id());
      if (target_series == nullptr) {
        target_series = g.second[a.second]->GetLatestSeries();
      }
      auto target_feature = target_series->get_img_feature_ptr();
      if ((*source_feature - *target_feature).norm() < img_feature_threshold_) {
        ret[a.first] = g.second[a.second]->get_uuid();
      }
    }
  }
  return ret;
}

}  // namespace mc