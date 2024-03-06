#include <cstddef>
#include <limits>
#include <vector>

class KM {
 public:
  static std::vector<std::pair<size_t, size_t>> Get(
      std::vector<std::vector<double>> cost_matrix) {
    if (cost_matrix.size() <= cost_matrix[0].size())
      return KM_(cost_matrix);
    else {
      std::vector<std::vector<double>> new_cost_matrix(
          cost_matrix[0].size(), std::vector<double>(cost_matrix.size(), 0));
      for (size_t i = 0; i < cost_matrix.size(); ++i)
        for (size_t j = 0; j < cost_matrix[0].size(); ++j)
          new_cost_matrix[j][i] = cost_matrix[i][j];
      auto ret = KM_(new_cost_matrix);
      std::vector<std::pair<size_t, size_t>> ret_;
      for (auto i : ret) {
        ret_.emplace_back(i.second, i.first);
      }
      return ret_;
    }
  }

 private:
  static std::vector<std::pair<size_t, size_t>> KM_(
      std::vector<std::vector<double>> cost_matrix) {
    nx_ = cost_matrix.size();
    ny_ = cost_matrix[0].size();
    cost_matrix_ = cost_matrix;

    lx_ = std::vector<float>(nx_, 0);
    ly_ = std::vector<float>(ny_, 0);
    match_ = std::vector<size_t>(ny_, -1);
    slack_ = std::vector<float>(ny_, 0);
    visx_ = std::vector<bool>(nx_, false);
    visy_ = std::vector<bool>(ny_, false);

    for (size_t x = 0; x < nx_; ++x) {
      slack_ = std::vector<float>(ny_, std::numeric_limits<float>::infinity());
      while (true) {
        visx_ = std::vector<bool>(nx_, false);
        visy_ = std::vector<bool>(ny_, false);
        if (FindPath(x))
          break;
        else {
          float delta = std::numeric_limits<float>::infinity();
          for (size_t i = 0; i < ny_; ++i)
            if (!visy_[i] && delta > slack_[i]) delta = slack_[i];
          for (size_t i = 0; i < nx_; ++i)
            if (visx_[i]) lx_[i] -= delta;
          for (size_t j = 0; j < ny_; ++j) {
            if (visy_[j])
              ly_[j] += delta;
            else
              slack_[j] -= delta;
          }
        }
      }
    }
    std::vector<std::pair<size_t, size_t>> ret;
    for (size_t i = 0; i < ny_; ++i) {
      if (match_[i] != -1) {
        ret.emplace_back(match_[i], i);
      }
    }
    return ret;
  }

  static bool FindPath(size_t x) {
    float temp_delta;
    visx_[x] = true;
    for (int y = 0; y < ny_; ++y) {
      if (visy_[y]) continue;
      temp_delta = lx_[x] + ly_[y] - cost_matrix_[x][y];
      if (temp_delta == 0) {
        visy_[y] = true;
        if (match_[y] == -1 || FindPath(match_[y])) {
          match_[y] = x;
          return true;
        }
      } else if (slack_[y] > temp_delta)
        slack_[y] = temp_delta;
    }
    return false;
  }

 private:
  static size_t nx_, ny_;
  static std::vector<std::vector<double>> cost_matrix_;
  static std::vector<size_t> match_;
  static std::vector<float> lx_, ly_, slack_;
  static std::vector<bool> visx_, visy_;
};

size_t KM::nx_ = 0;
size_t KM::ny_ = 0;
std::vector<std::vector<double>> KM::cost_matrix_;
std::vector<size_t> KM::match_;
std::vector<float> KM::lx_, KM::ly_, KM::slack_;
std::vector<bool> KM::visx_, KM::visy_;