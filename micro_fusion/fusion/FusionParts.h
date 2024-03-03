/**
 * File: FusionParts.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_FUSIONPARSTS_H
#define MICRO_FUSION_FUSIONPARSTS_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Fusion.h"
#include "PositionFusion.h"
#include "assignment/Assignment.h"
#include "feature_extr/FeatureExtr.h"

namespace mc {

class BaseFeatureExtrPart : public FusionFeatureExtrPart {
public:
  explicit BaseFeatureExtrPart(
      std::unordered_map<std::string, std::string> weights_path,
      int intra_threads = 1) {
    assert(weights_path.find("common") != weights_path.end());
    extr_map_["common"] = std::make_unique<OnnxFeatureExtr>(
        "common", weights_path.find("common")->second, intra_threads);
  };
  void Get(std::vector<DecTarget::Ptr> &targets) const override;

private:
  std::unordered_map<std::string, OnnxFeatureExtr::UniPtr> extr_map_{};
};

class BasePositionPart : public FusionPositionPart {
public:
  explicit BasePositionPart(time_t active_time)
      : FusionPositionPart(active_time) {
    position_fusion_ptr_ = std::make_unique<PositionFusionBayes>();
  }
  void Get(std::vector<FusTarget::Ptr> &targets) const override;

private:
  PositionFusion::UniPtr position_fusion_ptr_;
};

class BaseTargetMatchPart : public FusionTargetMatchPart {
public:
  explicit BaseTargetMatchPart(float img_feature_threshold,
                               double search_radius)
      : img_feature_threshold_(img_feature_threshold) {
    assignment_ptr_ = std::make_unique<MIPAssignment>();
    subgroup_ptr_ = std::make_unique<NeighborsSubGroup>(search_radius);
  }
  std::vector<uint64_t>
  Get(std::vector<DecTarget::Ptr> &targets,
      std::vector<FusTarget::Ptr> &his_targets) const override;

private:
  [[nodiscard]] std::vector<std::vector<double>>
  ClacCosts(std::vector<DecTarget::Ptr> const &targets,
            std::vector<FusTarget::Ptr> const &his_targets) const;

private:
  Assignment::UniPtr assignment_ptr_;
  SubGroup::UniPtr subgroup_ptr_;

  float img_feature_threshold_;
};

} // namespace mc

#endif // MICRO_FUSION_FUSIONPARSTS_H
