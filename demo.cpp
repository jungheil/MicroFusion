/**
 * File: main.cpp
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#include <memory>
#include <random>

#include "MicroFusion.h"

// #include "micro_fusion/feature_extr/FeatureExtr.h"
// #include "micro_fusion/fusion/PositionFusion.h"
// #include "micro_fusion/include/Toolkit.h"

using std::cout;
using std::endl;

int main() {
#ifdef MACRO_WIN32
  std::string image_demo_path = "../../resource/img01.jpg";
  std::string reid_model_path = "../../resource/reid.onnx";
#else
  std::string image_demo_path = "../resource/img01.jpg";
  std::string reid_model_path = "../resource/reid.onnx";
#endif

  std::vector<mc::DecTarget::Ptr> targets;
  // 随机生成目标
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, 0.1);
  mc::UAVInfo::Ptr uav_info = std::make_shared<mc::UAVInfo>(1);
  cv::Mat img = cv::imread(image_demo_path);

  // 新建目标融合数据缓存
  auto cache = std::make_shared<mc::LRUTargetCache>(10, 30);
  // 新建目标融合器
  auto fusion = mc::BaseFusion(cache, 10, 10, 30, 3, 0.5, 5,
                               {{"common", reid_model_path}}, 8);

  // 构建目标
  for (int i = 0; i < 4; ++i) {
    auto [measurement_position, target_orientation, target_distance] =
        mc::GetRandomTarget(Eigen::Vector3d(4, 8, 1));
    auto target_ptr = std::make_shared<mc::DecTarget>(
        uav_info, img, "common", measurement_position, target_orientation,
        target_distance + dist(gen));
    targets.emplace_back(target_ptr);
  }
  auto [measurement_position, target_orientation, target_distance] =
      mc::GetRandomTarget(Eigen::Vector3d(8, 4, 8));
  auto err_target_ptr = std::make_shared<mc::DecTarget>(
      uav_info, img, "common", measurement_position, target_orientation,
      target_distance + dist(gen));
  targets.emplace_back(err_target_ptr);

  // 运行时在此循环，一般为while(true)，此处为了演示只运行五次
  for (auto t : targets) {
    // 输入某时刻检测到的目标
    fusion.AddDecTarget(std::vector<mc::DecTarget::Ptr>{t});
  }

  // 通过缓存获取所有目标
  auto t = cache->GetAllTrackTargets();
  for (auto &i : t) {
    cout << i->get_estimate_position().transpose() << endl;
  }

  // 历史目标融合检测器
  mc::BasePersistent persistent("./db", 10, {{"common", reid_model_path}}, 8,
                                0.2, 0);

  // 添加感兴趣目标，此处为了演示添加索引为0的目标
  // persistent.StorageTarget(cache->GetAllTrackTargets()[0]);

  // 查找某个目标的历史记录
  auto uuid = persistent.LookUpTarget(cache->GetAllTrackTargets()[0]);
  auto his_targes = persistent.GetTarget(uuid);
  // 打印该历史目标的所有历史检测位置
  for (auto &t : his_targes) {
    cout << t->get_estimate_position().transpose() << endl;
  }

  return 0;
}