/**
 * File: Fusion.cpp
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#include "Fusion.h"

#include <memory>
#include <set>
#include <vector>

#include "BaseObject.h"
#include "FusionParts.h"

namespace mc {
void BaseFusion::InitParts() {
  feature_extr_ =
      std::make_shared<BaseFeatureExtrPart>(weights_path_, intra_threads_);
  position_fusion_ = std::make_shared<BasePositionPart>(position_active_time_);
  target_match_ = std::make_shared<BaseTargetMatchPart>(img_feature_threshold_,
                                                        search_radius_);
}

std::vector<FusTarget::Ptr> BaseFusion::AddDecTarget(
    std::vector<DecTarget::Ptr> dec_targets) {
  for (const auto &t : dec_targets) {
    assert(t->get_uav_info() != nullptr);
  }

  feature_extr_->Get(dec_targets);
  auto his_targets = target_cache_ptr_->GetAllTrackTargets();
  // split dec_targets by uav
  std::unordered_map<uint64_t, std::vector<DecTarget::Ptr>> dec_targets_map;
  for (auto &t : dec_targets) {
    if (dec_targets_map.find(t->get_uav_info()->get_uav_id()) ==
        dec_targets_map.end()) {
      dec_targets_map[t->get_uav_info()->get_uav_id()] = {};
    }
    dec_targets_map[t->get_uav_info()->get_uav_id()].push_back(t);
  }
  std::set<FusTarget::Ptr> active_targets_set;

  for (auto &it : dec_targets_map) {
    auto target_id = target_match_->Get(it.second, his_targets);
    for (size_t i = 0; i < it.second.size(); ++i) {
      if (target_id[i].empty()) {
        auto fus_target_ptr = std::make_shared<FusTarget>(
            fustarget_size_, fustarget_expire_time_, series_size_);
        fus_target_ptr->AddTarget(it.second[i]);
        target_cache_ptr_->AddTarget(fus_target_ptr);
        active_targets_set.insert(fus_target_ptr);
      } else {
        target_cache_ptr_->UpdateTarget(target_id[i], it.second[i]);
        active_targets_set.insert(target_cache_ptr_->GetTarget(target_id[i]));
      }
    }
  }

  auto active_targets = std::vector<FusTarget::Ptr>(active_targets_set.begin(),
                                                    active_targets_set.end());
  position_fusion_->Get(active_targets);
  return active_targets;
}

}  // namespace mc