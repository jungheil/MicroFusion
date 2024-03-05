/**
 * File: Persistent.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_PERSISTENT_H
#define MICRO_FUSION_PERSISTENT_H

#include <unordered_map>

#include "BaseObject.h"
#include "Feature.h"

namespace mc {

class Persistent {
 public:
  Persistent(std::string db_path) : db_path_(std::move(db_path)) {}
  virtual std::string LookUpTarget(FusTarget::Ptr target) = 0;
  virtual void StorageTarget(FusTarget::Ptr target) = 0;
  virtual std::vector<DecTarget::Ptr> GetTarget(const std::string &uuid) = 0;
  virtual void DelTarget(const std::string &uuid) = 0;

 protected:
  std::string db_path_;
};

class BasePersistent : public Persistent {
 public:
  explicit BasePersistent(
      std::string db_path, float search_radius,
      std::unordered_map<std::string, std::string> weights_path,
      int intra_threads, float match_threshold);

  std::string LookUpTarget(FusTarget::Ptr target) override;

  void StorageTarget(FusTarget::Ptr target) override;

  std::vector<DecTarget::Ptr> GetTarget(const std::string &uuid) override;

  void DelTarget(const std::string &uuid) override;

 private:
  void UpdateTarget(FusTarget::Ptr target, const std::string &uuid);

 private:
  float search_radius_;
  float match_threshold;
  std::unordered_map<std::string, FeatureExtr::UniPtr> extr_map_{};
};

}  // namespace mc

#endif  // MICRO_FUSION_PERSISTENT_H