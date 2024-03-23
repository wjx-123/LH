#ifndef ONNXMODEL_H
#define ONNXMODEL_H
#include <windows.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <stringapiset.h>

class OnnxModelAI {
public:
    OnnxModelAI(const std::string& model_path);
    std::pair<int, float> predict(cv::Mat& imgSrc);

private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<int64_t> input_dims;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    size_t num_input_nodes;
    size_t num_output_nodes;

    std::wstring stringToWstring(const std::string& str);
    void preprocess(cv::Mat& imgSrc, std::vector<float>& inputTensorValues);
    void center_resize(cv::Mat& src);
};

#endif // ONNXMODEL_H