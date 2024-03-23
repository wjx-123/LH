#include <windows.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <stringapiset.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <string>

class uNetOnnx {
public:
    uNetOnnx();
    ~uNetOnnx();
    
    //读取模型-dnn
    cv::dnn::Net loadUnetModel(std::string& modelFile);

    //输入图片得到分割后的vector<RotatedRect> 0是中间的小黑块，其余的是边上的引脚
    std::vector<cv::Rect> getDnnClassesRect(cv::Mat img, cv::dnn::Net uNetModel);

    
private:
    //用opencv的dnn模块调onnx模型
    cv::Mat dnnGetClasses(cv::Mat img, cv::dnn::Net uNetModel);
    //根据语义分割得到的图片进行外接矩形的获取--旋转矩形
    std::vector<cv::RotatedRect> getMultipodPackRoatedRect(cv::Mat uNetImg);
    //根据语义分割得到的图片进行外接矩形的获取--普通矩形
    std::vector<cv::Rect> getMultipodPackRect(cv::Mat uNetImg);

    //onnx版本--加载模型
    Ort::Session loadModel(const std::wstring& model_path);
    //onnx版本--预测
    cv::Mat performInference(Ort::Session& session, const cv::Mat& input_image);
    //计算输入张量的总元素数量
    size_t vector_product(const std::vector<int64_t>& vec);
    //将分割结果映射到颜色空间
    cv::Mat mapToColors(const cv::Mat& segmentation_result);

    std::vector<Ort::Value> runModel(Ort::Session& session, const Ort::Value& input_tensor);
};