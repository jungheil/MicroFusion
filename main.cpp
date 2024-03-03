/**
 * File: main.cpp
 * License: MIT License
 * Copyright: (c) 2024 Rongxi Li <lirx67@mail2.sysu.edu.cn>
 * Created: 2024-01-31
 * Brief:
 */

#include <memory>
#include <random>

#include "BaseObject.h"
#include "Fusion.h"
#include "TargetCache.h"
#include "Toolkit.h"

// #include "micro_fusion/feature_extr/FeatureExtr.h"
// #include "micro_fusion/fusion/PositionFusion.h"
// #include "micro_fusion/include/Toolkit.h"

using std::cout;
using std::endl;

int main() {
  std::vector<mc::DecTarget::Ptr> targets;
  // 随机生成目标
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(0.0, 0.1);
  mc::UAVInfo::Ptr uav_info = std::make_shared<mc::UAVInfo>(1);
  cv::Mat img = cv::imread("../resource/img01.jpg");
  auto cache = std::make_shared<mc::LRUTargetCache>(10, 30);
  auto fusion = mc::BaseFusion(cache, 10, 10, 30, 3, 0.5, 5,
                               {{"common", "../resource/reid.onnx"}}, 8);
  for (int i = 0; i < 4; ++i) {
    auto [measurement_position, target_orientation, target_distance] =
        mc::GetRandomTarget(Eigen::Vector3d(4, 8, 1));
    auto target_ptr = std::make_shared<mc::DecTarget>(
        uav_info, img, measurement_position, target_orientation,
        target_distance + dist(gen));
    targets.emplace_back(target_ptr);
  }
  auto [measurement_position, target_orientation, target_distance] =
      mc::GetRandomTarget(Eigen::Vector3d(8, 4, 8));
  auto err_target_ptr = std::make_shared<mc::DecTarget>(
      uav_info, img, measurement_position, target_orientation,
      target_distance + dist(gen));
  targets.emplace_back(err_target_ptr);
  for (auto t : targets) {
    fusion.AddDecTarget(std::vector<mc::DecTarget::Ptr>{t});
  }
  //  fusion.AddDecTarget(targets);

  auto t = cache->GetAllTrackTargets();
  for (auto &i : t) {
    cout << i->get_estimate_position().transpose() << endl;
  }

  // mc::PositionFusionBayes position_fusion_bayes;
  // position_fusion_bayes.GetConfidenceTarget(targets);

  // cout << "ground truth: 4. 8. 1." << endl;
  // for (int i = 0; i < 5; ++i) {
  //   cout << "measurement " << i << ": "
  //        << targets[i]->get_estimate_position().transpose() << endl;
  // }
  // targets.pop_back();
  // // TODO
  // //
  // 通过GetConfidenceTarget，过滤掉离群值，输入给下面position_fusion_bayes.Get。
  // cout << "fusion position: " <<
  // position_fusion_bayes.Get(targets).transpose()
  //      << endl;
  // mc::OnnxFeatureExtr fe("aaa",
  // "/home/cat/XXX/zy/zy_fusion/resource/reid.onnx",
  //                        8);
  // std::vector<cv::Mat> imgs = {img};
  // auto targets_feature = fe.GetTargetsFeature(imgs);
  // cout << "feature: " << (*targets_feature[0]).transpose() << endl;

  return 0;
}