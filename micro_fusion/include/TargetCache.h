/**
 * File: TargetCache.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_TARGETCACHE_H
#define MICRO_FUSION_TARGETCACHE_H

#include <ctime>
#include <memory>
#include <vector>

#include "BaseObject.h"
#include "Common.h"

namespace mc {
class TargetCache {
public:
  using Ptr = std::shared_ptr<TargetCache>;
  TargetCache() = default;
  virtual ~TargetCache() = default;
  virtual void UpdateTarget(uint64_t target_id, DecTarget::Ptr target_ptr) = 0;
  virtual void AddTarget(FusTarget::Ptr target) = 0;
  virtual FusTarget::Ptr GetTarget(uint64_t target_id) = 0;
  virtual std::vector<FusTarget::Ptr> GetAllTrackTargets() = 0;
};

class LRUTargetCache : public TargetCache {
public:
  explicit LRUTargetCache(size_t cache_size, time_t expire_time)
      : cache_(cache_size, expire_time) {}
  void UpdateTarget(uint64_t target_id, DecTarget::Ptr target_ptr) override;
  void AddTarget(FusTarget::Ptr target) override;
  FusTarget::Ptr GetTarget(uint64_t target_id) override;
  std::vector<FusTarget::Ptr> GetAllTrackTargets() override;

protected:
  LRUCache<size_t, FusTarget::Ptr> cache_;
};

} // namespace mc

#endif // MICRO_FUSION_TARGETCACHE_H
