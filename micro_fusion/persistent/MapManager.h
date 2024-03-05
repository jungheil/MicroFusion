
/**
 * File: MapManager.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_MAPMANAGER_H
#define MICRO_FUSION_MAPMANAGER_H

#include <atomic>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Common.h"
#include "PTarget.pb.h"
#include "nanoflann.hpp"
// #include "BaseObject.h"

namespace mc {

struct PersistentPoint {
  std::string uuid;
  double x;
  double y;
  double z;
};

class MapManager {
 private:
  MapManager(std::string db_path) : db_path_(db_path) {
    if (!std::filesystem::exists(db_path_ / "index.bin") ||
        !std::filesystem::exists(db_path_ / "pointcloud.bin")) {
      if (!std::filesystem::exists(db_path_)) {
        std::filesystem::create_directory(db_path_);
      };
      index_ptr_ = std::make_unique<KDTree>(
          2, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    } else {
      assert(std::filesystem::exists(db_path_ / "index.bin") &&
             std::filesystem::exists(db_path_ / "pointcloud.bin"));

      index_ptr_ = std::make_unique<KDTree>(
          2, cloud_,
          nanoflann::KDTreeSingleIndexAdaptorParams(
              10,
              nanoflann::KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex));
      LoadPointcloud();

      LoadIndex();

      assert(cloud_.pts.size() == index_ptr_->size_);
    }
  };
  ~MapManager() = default;
  MapManager(const MapManager &) = delete;
  MapManager &operator=(const MapManager &) = delete;

 private:
  class Deletor {
   public:
    ~Deletor() {
      for (auto &it : MapManager::instance_map_) {
        delete it.second.load();
      }
    }
  };
  static Deletor deletor;

 private:
  static std::unordered_map<std::string, std::atomic<MapManager *>>
      instance_map_;
  static std::mutex mut;

 public:
  static MapManager *Instance(std::string db_path) {
    if (instance_map_.find(db_path) == instance_map_.end()) {
      std::lock_guard<std::mutex> lock(mut);
      if (instance_map_.find(db_path) == instance_map_.end()) {
        instance_map_[db_path] = new MapManager(db_path);
      }
    }
    return instance_map_[db_path];
  }

 public:
  void SavePointcloud() {
    PIndexPointArray index_point_array;
    for (const auto &p : cloud_.pts) {
      PIndexPoint *index_point = index_point_array.add_points();
      index_point->set_uuid(p.uuid);
      index_point->set_x(p.x);
      index_point->set_y(p.y);
      index_point->set_z(p.z);
    }
    std::ofstream f(db_path_ / "pointcloud.bin", std::ofstream::binary);
    index_point_array.SerializeToOstream(&f);
    f.close();
  }
  void LoadPointcloud() {
    PIndexPointArray index_point_array;
    std::ifstream f(db_path_ / "pointcloud.bin", std::ifstream::binary);
    index_point_array.ParseFromIstream(&f);
    f.close();
    cloud_.pts.clear();
    for (const auto &p : index_point_array.points()) {
      PersistentPoint point;
      point.uuid = p.uuid();
      point.x = p.x();
      point.y = p.y();
      point.z = p.z();
      cloud_.pts.push_back(point);
    }
  }
  void SaveIndex() {
    std::ofstream f(db_path_ / "index.bin", std::ofstream::binary);
    if (f.bad()) throw std::runtime_error("Error writing index file!");

    index_ptr_->saveIndex(f);
    f.close();
  }
  void LoadIndex() {
    std::ifstream f(db_path_ / "index.bin", std::ofstream::binary);
    if (f.fail()) throw std::runtime_error("Error reading index file!");
    index_ptr_->loadIndex(f);

    f.close();
  }

  void AddPoint(PTarget &p_target) {
    PersistentPoint point;
    std::string uuid =
        boost::uuids::to_string(boost::uuids::random_generator()());
    point.uuid = uuid;
    point.x = p_target.position().x();
    point.y = p_target.position().y();
    point.z = p_target.position().z();
    cloud_.pts.push_back(point);
    index_ptr_->buildIndex();
    SaveIndex();
    SavePointcloud();

    PTargetArray target_array;
    target_array.set_uuid(uuid);
    target_array.set_type(p_target.type());
    target_array.mutable_latest_position()->set_x(p_target.position().x());
    target_array.mutable_latest_position()->set_y(p_target.position().y());
    target_array.mutable_latest_position()->set_z(p_target.position().z());
    target_array.set_update_time(p_target.time_stamp());
    target_array.add_targets()->CopyFrom(p_target);

    std::ofstream f(db_path_ / (uuid + ".bin"), std::ofstream::binary);
    target_array.SerializeToOstream(&f);
    f.close();
  }

  void UpdatePoint(PTarget &p_target, const std::string &uuid) {
    auto target_array = LoadPoint(uuid);

    target_array.mutable_latest_position()->set_x(p_target.position().x());
    target_array.mutable_latest_position()->set_y(p_target.position().y());
    target_array.mutable_latest_position()->set_z(p_target.position().z());
    target_array.set_update_time(p_target.time_stamp());
    target_array.add_targets()->CopyFrom(p_target);

    std::ofstream f(db_path_ / (uuid + ".bin"), std::ofstream::binary);
    target_array.SerializeToOstream(&f);
    f.close();

    auto it = std::find_if(
        cloud_.pts.begin(), cloud_.pts.end(),
        [uuid](const PersistentPoint &p) { return p.uuid == uuid; });
    if (it != cloud_.pts.end()) {
      it->x = p_target.position().x();
      it->y = p_target.position().y();
      it->z = p_target.position().z();
      index_ptr_->buildIndex();
      SaveIndex();
      SavePointcloud();
    }
  }

  void DeletePoint(const std::string &uuid) {
    std::filesystem::remove(db_path_ / (uuid + ".bin"));
    auto it = std::find_if(
        cloud_.pts.begin(), cloud_.pts.end(),
        [uuid](const PersistentPoint &p) { return p.uuid == uuid; });
    if (it != cloud_.pts.end()) {
      cloud_.pts.erase(it);
      index_ptr_->buildIndex();
      SaveIndex();
      SavePointcloud();
    }
  }

  PTargetArray LoadPoint(const std::string &uuid) {
    PTargetArray target_array;
    assert(std::filesystem::exists(db_path_ / (uuid + ".bin")) &&
           "Target file not found!");
    std::ifstream f(db_path_ / (uuid + ".bin"), std::ifstream::binary);
    target_array.ParseFromIstream(&f);
    f.close();
    return target_array;
  }

  std::vector<PTargetArray> GetNeighbour(double x, double y, double radius) {
    double query[2] = {x, y};
    std::vector<nanoflann::ResultItem<uint32_t, double>> matches;
    const size_t matches_num =
        index_ptr_->radiusSearch(&query[0], radius, matches);
    std::vector<PTargetArray> ret;
    ret.reserve(matches_num);
    for (auto &m : matches) {
      auto target = LoadPoint(cloud_.pts[m.first].uuid);
      ret.push_back(target);
    }

    return ret;
  }

 private:
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, KDDPointCloud<PersistentPoint>>,
      KDDPointCloud<PersistentPoint>, 2>;
  std::filesystem::path db_path_;
  KDDPointCloud<PersistentPoint> cloud_{};
  std::unique_ptr<KDTree> index_ptr_;
};

std::unordered_map<std::string, std::atomic<MapManager *>>
    MapManager::instance_map_;
std::mutex MapManager::mut;

}  // namespace mc

#endif  // MICRO_FUSION_MAPMANAGER_H