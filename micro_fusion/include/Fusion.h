/**
 * File: Fusion.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_FUSION_H
#define MICRO_FUSION_FUSION_H

#include <ctime>
#include <memory>
#include <utility>
#include <vector>

#include "BaseObject.h"
#include "TargetCache.h"

namespace mc {
class FusionFeatureExtrPart {
public:
  using Ptr = std::shared_ptr<FusionFeatureExtrPart>;
  virtual void Get(std::vector<DecTarget::Ptr> &targets) const = 0;
};
class FusionPositionPart {
public:
  explicit FusionPositionPart(time_t active_time) : active_time_(active_time){};
  using Ptr = std::shared_ptr<FusionPositionPart>;
  virtual void Get(std::vector<FusTarget::Ptr> &targets) const = 0;

protected:
  time_t active_time_;
};
class FusionTargetMatchPart {
public:
  using Ptr = std::shared_ptr<FusionTargetMatchPart>;
  virtual std::vector<std::string>
  Get(std::vector<DecTarget::Ptr> &targets,
      std::vector<FusTarget::Ptr> &his_targets) const = 0;
};
class Fusion {
public:
  explicit Fusion(TargetCache::Ptr target_cache_ptr)
      : target_cache_ptr_(std::move(target_cache_ptr)){};

  virtual std::vector<FusTarget::Ptr>
  AddDecTarget(std::vector<DecTarget::Ptr> dec_targets) = 0;

protected:
  TargetCache::Ptr target_cache_ptr_;
};

class BaseFusion : public Fusion {
public:
  explicit BaseFusion(TargetCache::Ptr target_cache_ptr, size_t series_size,
                      size_t fustarget_size, time_t fustarget_expire_time,
                      time_t position_active_time, float img_feature_threshold,
                      double search_radius,
                      std::unordered_map<std::string, std::string> weights_path,
                      int intra_threads)
      : Fusion(target_cache_ptr), series_size_(series_size),
        fustarget_size_(fustarget_size),
        fustarget_expire_time_(fustarget_expire_time),
        position_active_time_(position_active_time),
        img_feature_threshold_(img_feature_threshold),
        search_radius_(search_radius), weights_path_(std::move(weights_path)),
        intra_threads_(intra_threads) {
    InitParts();
  };
  std::vector<FusTarget::Ptr>
  AddDecTarget(std::vector<DecTarget::Ptr> dec_targets) override;

protected:
  void InitParts();
  FusionFeatureExtrPart::Ptr feature_extr_;
  FusionPositionPart::Ptr position_fusion_;
  FusionTargetMatchPart::Ptr target_match_;

  size_t series_size_;
  size_t fustarget_size_;
  time_t fustarget_expire_time_;
  time_t position_active_time_;
  float img_feature_threshold_;
  double search_radius_;
  std::unordered_map<std::string, std::string> weights_path_;
  int intra_threads_ = 1;
};

} // namespace mc
#endif // MICRO_FUSION_FUSION_H
