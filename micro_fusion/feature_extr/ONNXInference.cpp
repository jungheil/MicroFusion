
#include "ONNXInference.h"

#include <numeric>

namespace mc {

#ifdef MACRO_WIN32

std::wstring s2ws(const std::string &s) {
  size_t convertedChars = 0;
  std::string curLocale = setlocale(LC_ALL, NULL); // curLocale="C"
  setlocale(LC_ALL, "chs");
  const char *source = s.c_str();
  size_t charNum = sizeof(char) * s.size() + 1;

  wchar_t *dest = new wchar_t[charNum];
  mbstowcs_s(&convertedChars, dest, charNum, source, _TRUNCATE);
  std::wstring result = dest;
  delete[] dest;
  setlocale(LC_ALL, curLocale.c_str());
  return result;
}

#endif

template <typename T> T VectorProduct(const std::vector<T> &v) {
  return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

ONNXInference::ONNXInference(const std::string &name,
                             const std::string &model_path, int intra_threads) {
  env_ = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                    name.c_str());
  allocator_ = std::make_unique<Ort::AllocatorWithDefaultOptions>();
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(intra_threads);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef MACRO_WIN32
  session_ = std::make_unique<Ort::Session>(*env_, s2ws(model_path).c_str(),
                                            session_options);
#else
  session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(),
                                            session_options);
#endif

  input_count_ = session_->GetInputCount();
  output_count_ = session_->GetOutputCount();
  for (size_t i = 0; i < input_count_; ++i) {
#ifdef MACRO_WIN32
    std::string name = session_->GetInputName(i, *allocator_);
#else
    std::string name = session_->GetInputNameAllocated(i, *allocator_).get();
#endif

    input_node_names_.push_back(name);
  }

  for (size_t i = 0; i < output_count_; ++i) {
#ifdef MACRO_WIN32
    std::string name = session_->GetOutputName(i, *allocator_);
#else
    std::string name = session_->GetOutputNameAllocated(i, *allocator_).get();
#endif
    output_node_names_.push_back(name);
  }

  for (size_t i = 0; i < input_count_; ++i) {
    input_node_dims_.push_back(
        session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < output_count_; ++i) {
    output_node_dims_.push_back(
        session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
};

std::vector<float> ONNXInference::Preprocess(const std::vector<cv::Mat> &imgs,
                                             uint8_t input_node_idx) const {
  auto input_dims = std::get<0>(GetIODims(imgs.size())).at(input_node_idx);
  assert(input_dims.at(0) >= imgs.size());
  auto input_size = VectorProduct(input_dims);

  std::vector<float> input_tensor_values(input_size * imgs.size());
  for (int i = 0; i < imgs.size(); ++i) {
    cv::Mat temp_img;
    cv::resize(imgs[i], temp_img,
               cv::Size(input_node_dims_.at(input_node_idx).at(3),
                        input_node_dims_.at(input_node_idx).at(2)),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(temp_img, temp_img, cv::ColorConversionCodes::COLOR_BGR2RGB);
    temp_img.convertTo(temp_img, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(temp_img, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, temp_img);
    cv::dnn::blobFromImage(temp_img, temp_img);

    std::copy(temp_img.begin<float>(), temp_img.end<float>(),
              input_tensor_values.begin() + i * input_size / input_dims.at(0));
  }
  return input_tensor_values;
}

std::vector<std::vector<float>>
ONNXInference::Postprocess(const std::vector<float> &output_tensor_values,
                           uint8_t output_node_idx) const {
  auto output_dims = std::get<1>(GetIODims(1)).at(output_node_idx);
  auto output_size = VectorProduct(output_dims);
  std::vector<std::vector<float>> post_values;
  for (int i = 0; i < output_dims.at(0); ++i) {
    std::vector<float> temp_output_tensor_values(
        output_tensor_values.begin() + i * output_size / output_dims.at(0),
        output_tensor_values.begin() +
            (i + 1) * output_size / output_dims.at(0));
    post_values.push_back(temp_output_tensor_values);
  }
  return post_values;
}

std::vector<std::vector<float>> ONNXInference::Process(
    std::vector<std::vector<float>> &input_tensor_values) const {
  std::vector<Ort::Value> input_tensors;
  std::vector<Ort::Value> output_tensors;
  input_tensors.reserve(input_count_);
  output_tensors.reserve(output_count_);

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto [input_dims, output_dims] = GetIODims(input_tensor_values.size());

  std::vector<std::vector<float>> output_tensor_values;
  for (size_t i = 0; i < output_count_; ++i) {
    output_tensor_values.emplace_back(VectorProduct(output_dims.at(i)));
  }

  for (size_t i = 0; i < input_count_; ++i) {
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, input_tensor_values.at(i).data(),
        VectorProduct(input_dims.at(i)), input_dims.at(i).data(),
        input_dims.at(i).size()));
  }
  for (size_t i = 0; i < output_count_; ++i) {
    output_tensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, output_tensor_values.at(i).data(),
        VectorProduct(output_dims.at(i)), output_dims.at(i).data(),
        output_dims.at(i).size()));
  }
  std::vector<const char *> input_node_names;
  std::vector<const char *> output_node_names;
  for (size_t i = 0; i < input_count_; ++i) {
    input_node_names.push_back(input_node_names_.at(i).c_str());
  }
  for (size_t i = 0; i < output_count_; ++i) {
    output_node_names.push_back(output_node_names_.at(i).c_str());
  }
  session_->Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                input_tensors.data(), input_count_, output_node_names.data(),
                output_tensors.data(), output_count_);

  return output_tensor_values;
}

std::tuple<std::vector<std::vector<int64_t>>, std::vector<std::vector<int64_t>>>
ONNXInference::GetIODims(uint8_t batchsize) const {
  std::vector<std::vector<int64_t>> input_dims;
  std::vector<std::vector<int64_t>> output_dims;
  for (size_t i = 0; i < input_count_; ++i) {
    std::vector<int64_t> input_dim = input_node_dims_.at(i);
    if (input_dim.at(0) == -1) {
      input_dim.at(0) = batchsize;
    }
    input_dims.push_back(input_dim);
  }
  for (size_t i = 0; i < output_count_; ++i) {
    std::vector<int64_t> output_dim = output_node_dims_.at(i);
    if (output_dim.at(0) == -1) {
      output_dim.at(0) = batchsize;
    }
    output_dims.push_back(output_dim);
  }
  return std::make_tuple(input_dims, output_dims);
};
} // namespace mc