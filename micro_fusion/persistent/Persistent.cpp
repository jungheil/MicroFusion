#include "Persistent.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "BaseObject.h"
#include "MapManager.h"
#include "feature_extr/FeatureExtr.h"

namespace mc {

PTarget PTargetGenerator(FusTarget::Ptr target) {
  PTarget p_target;
  p_target.set_uuid(target->get_uuid());
  p_target.set_time_stamp(target->get_update_time());
  p_target.set_type(target->get_target_type());
  p_target.mutable_position()->set_x(target->get_estimate_position()[0]);
  p_target.mutable_position()->set_y(target->get_estimate_position()[1]);
  p_target.mutable_position()->set_z(target->get_estimate_position()[2]);

  auto latest_dec = target->GetTargetSeries()->GetLatestTarget();
  p_target.mutable_image()->set_cols(latest_dec->get_img().cols);
  p_target.mutable_image()->set_rows(latest_dec->get_img().rows);

  std::vector<uint8_t> encode_img;
  std::vector<int> params;
  params.push_back(cv::IMWRITE_JPEG_QUALITY);
  params.push_back(95);
  cv::imencode(".jpg", latest_dec->get_img(), encode_img, params);
  p_target.mutable_image()->set_data(encode_img.data(), encode_img.size());
  p_target.mutable_image()->set_size(encode_img.size());
  return p_target;
}

BasePersistent::BasePersistent(
    std::string db_path, float search_radius,
    std::unordered_map<std::string, std::string> weights_path,
    int intra_threads, float match_threshold)
    : Persistent(std::move(db_path)),
      search_radius_(search_radius),
      match_threshold(match_threshold) {
  assert(weights_path.find("common") != weights_path.end());
  extr_map_["common"] = std::make_unique<OnnxFeatureExtr>(
      "common", weights_path.find("common")->second, intra_threads);
}

std::string BasePersistent::LookUpTarget(FusTarget::Ptr target) {
  auto map_manager = MapManager::Instance(db_path_);
  auto neighbour = map_manager->GetNeighbour(target->get_estimate_position()[0],
                                             target->get_estimate_position()[1],
                                             search_radius_);
  std::vector<DecTarget::Ptr> targets;
  if (neighbour.empty()) {
    return "";
  }
  std::vector<cv::Mat> imgs;
  for (const auto &n : neighbour) {
    auto target = n.targets(n.targets_size() - 1);
    std::vector<uint8_t> decode_img(target.image().data().begin(),
                                    target.image().data().end());
    cv::Mat img = cv::imdecode(decode_img, cv::IMREAD_COLOR);
    imgs.push_back(img);
  }
  imgs.push_back(target->GetTargetSeries()->GetLatestTarget()->get_img());
  auto feature_list = extr_map_.find("common")->second->GetTargetsFeature(imgs);
  auto target_feature = feature_list.back();
  feature_list.pop_back();

  std::vector<float> dists;
  for (const auto &f : feature_list) {
    dists.push_back((*f - *target_feature).norm());
  }
  // get the nearest neighbour idx
  auto min_idx = std::min_element(dists.begin(), dists.end()) - dists.begin();
  if (dists[min_idx] > match_threshold) {
    return "";
  } else {
    UpdateTarget(target, neighbour[min_idx].uuid());
    return neighbour[min_idx].uuid();
  }
}

void BasePersistent::UpdateTarget(FusTarget::Ptr target,
                                  const std::string &uuid) {
  auto map_manager = MapManager::Instance(db_path_);
  auto new_target = PTargetGenerator(target);
  map_manager->UpdatePoint(new_target, uuid);
}

void BasePersistent::StorageTarget(FusTarget::Ptr target) {
  auto map_manager = MapManager::Instance(db_path_);

  PTarget p_target = PTargetGenerator(target);
  map_manager->AddPoint(p_target);
}
std::vector<DecTarget::Ptr> BasePersistent::GetTarget(const std::string &uuid) {
  if (uuid.empty()) {
    return {};
  }
  auto map_manager = MapManager::Instance(db_path_);
  auto target_array = map_manager->LoadPoint(uuid);
  std::vector<DecTarget::Ptr> ret;

  for (const auto &t : target_array.targets()) {
    std::vector<uint8_t> decode_img(t.image().data().begin(),
                                    t.image().data().end());
    cv::Mat img = cv::imdecode(decode_img, cv::IMREAD_COLOR);
    auto dec_target = std::make_shared<DecTarget>(
        nullptr, img, t.type(),
        Eigen::Vector3d(t.position().x(), t.position().y(), t.position().z()));
    dec_target->set_time_stamp(t.time_stamp());
    ret.push_back(dec_target);
  }
  return ret;
}
void BasePersistent::DelTarget(const std::string &uuid) {
  auto map_manager = MapManager::Instance(db_path_);
  map_manager->DeletePoint(uuid);
}

}  // namespace mc