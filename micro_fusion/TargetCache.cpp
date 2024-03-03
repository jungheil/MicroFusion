/**
 * File: TargetCache.cpp
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#include "TargetCache.h"

namespace mc {
void LRUTargetCache::UpdateTarget(uint64_t target_id,
                                  DecTarget::Ptr target_ptr) {
  assert(cache_.exist(target_id));
  auto fus_target_ptr = cache_.get(target_id, true);
  fus_target_ptr->AddTarget(target_ptr);
}
void LRUTargetCache::AddTarget(FusTarget::Ptr target) {
  cache_.put(target->get_target_id(), target);
}

FusTarget::Ptr LRUTargetCache::GetTarget(uint64_t target_id) {
  return cache_.get(target_id, false);
}

std::vector<FusTarget::Ptr> LRUTargetCache::GetAllTrackTargets() {
  std::vector<FusTarget::Ptr> targets;

  for (auto &target : cache_.get_data()) {
    targets.push_back(target.second);
  }
  return targets;
}
} // namespace mc
