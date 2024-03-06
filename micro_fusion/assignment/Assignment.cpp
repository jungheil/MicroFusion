/**
 * File: Assignment.cpp
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#include "Assignment.h"

#include <utility>

#include "KM.h"

// using namespace operations_research;

namespace mc {

std::vector<std::pair<size_t, size_t>> KMAssignment::Get(
    std::vector<std::vector<double>> costs) {
  return KM::Get(costs);
}

std::unordered_map<size_t, std::vector<size_t>>
NeighborsSubGroup::GetNeighborsMap(std::vector<DecTarget::Ptr> &targets,
                                   std::vector<FusTarget::Ptr> &his_targets) {
  std::unordered_map<size_t, std::vector<size_t>> ret{};

  cloud_.pts = targets;
  index_.buildIndex();

  double query[2] = {0, 0};
  for (size_t i = 0; i < his_targets.size(); ++i) {
    auto position = his_targets[i]->get_estimate_position();
    query[0] = position[0];
    query[1] = position[1];
    std::vector<nanoflann::ResultItem<uint32_t, double>> matches;

    const size_t matches_num =
        index_.radiusSearch(&query[0], search_radius_, matches);
    for (auto &m : matches) {
      ret[i].push_back(m.first);
    }
  }
  return ret;
}

std::vector<TargetAssignmentGroup> NeighborsSubGroup::Get(
    std::vector<DecTarget::Ptr> &targets,
    std::vector<FusTarget::Ptr> &his_targets) {
  auto map = GetNeighborsMap(targets, his_targets);
  auto merged = MergeTarget(map);
  std::vector<TargetAssignmentGroup> ret;
  for (const auto &entry : merged) {
    std::vector<DecTarget::Ptr> source;
    std::vector<FusTarget::Ptr> target;
    for (const auto &idx : entry.first) {
      source.push_back(targets[idx]);
    }
    for (const auto &idx : entry.second) {
      target.push_back(his_targets[idx]);
    }
    ret.emplace_back(source, target);
  }
  return ret;
}
std::pair<std::vector<size_t>, std::vector<size_t>>
NeighborsSubGroup::MergeTarget_(
    std::unordered_map<size_t, std::vector<size_t>> &map,
    std::unordered_map<size_t, std::vector<size_t>> &inversed_map,
    std::unordered_map<size_t, bool> &visited_target,
    std::unordered_map<size_t, bool> &visited_source, size_t target_idx) const {
  if (visited_target[target_idx]) {
    return {};
  }
  visited_target[target_idx] = true;
  std::vector<size_t> target{target_idx};
  std::vector<size_t> source;
  for (const auto &idx : inversed_map[target_idx]) {
    if (visited_source[idx]) {
      continue;
    }
    visited_source[idx] = true;
    source.push_back(idx);
    for (const auto &idx_target : map[idx]) {
      auto ret = MergeTarget_(map, inversed_map, visited_target, visited_source,
                              idx_target);
      source.insert(source.end(), ret.first.begin(), ret.first.end());
      target.insert(target.end(), ret.second.begin(), ret.second.end());
    }
  }
  return std::make_pair(source, target);
}

std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>
NeighborsSubGroup::MergeTarget(
    std::unordered_map<size_t, std::vector<size_t>> &map) const {
  std::unordered_map<size_t, std::vector<size_t>> inversed_map;
  for (auto &it : map) {
    for (auto it2 = it.second.begin(); it2 != it.second.end(); it2++) {
      inversed_map[*it2].push_back(it.first);
    }
  }

  std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> ret;
  std::unordered_map<size_t, bool> visited_target;
  std::unordered_map<size_t, bool> visited_source;
  for (const auto &entry : inversed_map) {
    auto source_target = MergeTarget_(map, inversed_map, visited_target,
                                      visited_source, entry.first);
    if (!source_target.first.empty() && !source_target.second.empty()) {
      ret.push_back(source_target);
    }
  }
  return ret;
}

}  // namespace mc
