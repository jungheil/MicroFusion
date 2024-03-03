/**
 * File: Assignment.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_ASSIGNMENT_H
#define MICRO_FUSION_ASSIGNMENT_H

#ifdef MACRO_WIN32
#define USE_GLOP
#endif

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "BaseObject.h"
#include "Common.h"
#include "nanoflann.hpp"
#include "ortools/linear_solver/linear_solver.h"

namespace mc {
class Assignment {
public:
  using Ptr = std::shared_ptr<Assignment>;
  using UniPtr = std::unique_ptr<Assignment>;
  Assignment() = default;
  virtual ~Assignment() = default;
  virtual std::vector<std::pair<size_t, size_t>>
  Get(std::vector<std::vector<double>> costs) = 0;
};

class MIPAssignment : public Assignment {
public:
  MIPAssignment();
  ~MIPAssignment() override = default;
  std::vector<std::pair<size_t, size_t>>
  Get(std::vector<std::vector<double>> costs) override;

private:
  std::unique_ptr<operations_research::MPSolver> solver_ = nullptr;
};

using TargetAssignmentGroup =
    std::pair<std::vector<DecTarget::Ptr>, std::vector<FusTarget::Ptr>>;

class SubGroup {
public:
  using Ptr = std::shared_ptr<SubGroup>;
  using UniPtr = std::unique_ptr<SubGroup>;
  SubGroup() = default;
  virtual ~SubGroup() = default;
  virtual std::vector<TargetAssignmentGroup>
  Get(std::vector<DecTarget::Ptr> &targets,
      std::vector<FusTarget::Ptr> &his_targets) = 0;
};

class NoSubGroup : public SubGroup {
public:
  NoSubGroup() = default;
  ~NoSubGroup() override = default;
  std::vector<TargetAssignmentGroup>
  Get(std::vector<DecTarget::Ptr> &targets,
      std::vector<FusTarget::Ptr> &his_targets) override {
    return std::vector<TargetAssignmentGroup>{
        std::make_pair(targets, his_targets)};
  }
};

struct KDDecTargetCloud {
  std::vector<DecTarget::Ptr> pts;

  [[nodiscard]] inline size_t kdtree_get_point_count() const {
    return pts.size();
  }

  [[nodiscard]] inline double kdtree_get_pt(const size_t idx,
                                            const size_t dim) const {
    if (dim == 0)
      return pts[idx]->get_estimate_position()[0];
    else
      return pts[idx]->get_estimate_position()[1];
  }

  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};
class NeighborsSubGroup : public SubGroup {
public:
  explicit NeighborsSubGroup(double search_radius)
      : search_radius_(search_radius){};
  ~NeighborsSubGroup() override = default;
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, KDDecTargetCloud>, KDDecTargetCloud,
      2>;
  std::vector<TargetAssignmentGroup>
  Get(std::vector<DecTarget::Ptr> &targets,
      std::vector<FusTarget::Ptr> &his_targets) override;

private:
  std::unordered_map<size_t, std::vector<size_t>>
  GetNeighborsMap(std::vector<DecTarget::Ptr> &targets,
                  std::vector<FusTarget::Ptr> &his_targets);
  std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>
  MergeTarget(std::unordered_map<size_t, std::vector<size_t>> &map) const;
  std::pair<std::vector<size_t>, std::vector<size_t>>
  MergeTarget_(std::unordered_map<size_t, std::vector<size_t>> &map,
               std::unordered_map<size_t, std::vector<size_t>> &inversed_map,
               std::unordered_map<size_t, bool> &visited_target,
               std::unordered_map<size_t, bool> &visited_source,
               size_t target_idx) const;

private:
  double search_radius_;
  KDDecTargetCloud cloud_{};
  KDTree index_ = KDTree(2, cloud_);
};
} // namespace mc

#endif // MICRO_FUSION_ASSIGNMENT_H
