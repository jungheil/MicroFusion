/**
 * File: Common.h
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#ifndef MICRO_FUSION_COMMON_H
#define MICRO_FUSION_COMMON_H

#include <cassert>
#include <cstddef>
#include <ctime>
#include <list>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#define M_PI 3.14159265358979323846 /* pi */

namespace mc {

class Singleton {
 public:
  static Singleton &GetInstance() {
    static std::once_flag s_flag;
    std::call_once(s_flag, [&]() { instance_.reset(new Singleton); });

    return *instance_;
  }

  ~Singleton() = default;

 private:
  Singleton() = default;

  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

 private:
  static std::unique_ptr<Singleton> instance_;
};

template <class KEY_T, class VAL_T>
class LRUCache {
 private:
  std::list<std::pair<KEY_T, VAL_T>> item_list;
  std::unordered_map<KEY_T, std::tuple<decltype(item_list.begin()), time_t>>
      item_map;
  size_t cache_size_;
  // expire time 60s
  time_t expire_time_;

 private:
  void clean(void) {
    while (item_map.size() > cache_size_) {
      auto last_it = item_list.end();
      last_it--;

      item_map.erase(last_it->first);
      item_list.pop_back();
    }
    if (expire_time_ > 0) {
      while (!item_map.empty()) {
        auto last_it = item_list.end();
        last_it--;
        if (time(nullptr) - std::get<1>(item_map[last_it->first]) >
            expire_time_) {
          item_map.erase(last_it->first);
          item_list.pop_back();
        } else {
          break;
        }
      }
    }
  }

 public:
  LRUCache(int cache_size_, time_t expire_time)
      : cache_size_(cache_size_), expire_time_(expire_time) {}

  void put(const KEY_T &key, const VAL_T &val) {
    auto it = item_map.find(key);
    if (it != item_map.end()) {
      item_list.erase(std::get<0>(it->second));
      item_map.erase(it);
    }
    item_list.push_front(std::make_pair(key, val));
    item_map.insert(
        std::make_pair(key, std::make_tuple(item_list.begin(), time(nullptr))));
    clean();
  }
  bool exist(const KEY_T &key) { return (item_map.count(key) > 0); }

  VAL_T get_head() {
    assert(!item_list.empty());
    return item_list.front().second;
  }
  VAL_T get(const KEY_T &key, bool is_update = true) {
    assert(exist(key));
    auto it = std::get<0>(item_map.find(key)->second);
    if (is_update) {
      std::get<1>(item_map.find(key)->second) = time(nullptr);
      item_list.splice(item_list.begin(), item_list, it);
    }
    return it->second;
  }
  const std::list<std::pair<KEY_T, VAL_T>> &get_data() { return item_list; }
  size_t size() { return item_map.size(); }
};

template <typename T>
struct KDDPointCloud {
  std::vector<T> pts;

  [[nodiscard]] inline size_t kdtree_get_point_count() const {
    return pts.size();
  }

  [[nodiscard]] inline double kdtree_get_pt(const size_t idx,
                                            const size_t dim) const {
    if (dim == 0)
      return pts[idx].x;
    else if (dim == 1)
      return pts[idx].y;
    else
      return pts[idx].z;
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

}  // namespace mc
#endif  // MICRO_FUSION_COMMON_H
