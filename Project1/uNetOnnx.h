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
    
    //��ȡģ��-dnn
    cv::dnn::Net loadUnetModel(std::string& modelFile);

    //����ͼƬ�õ��ָ���vector<RotatedRect> 0���м��С�ڿ飬������Ǳ��ϵ�����
    std::vector<cv::Rect> getDnnClassesRect(cv::Mat img, cv::dnn::Net uNetModel);

    
private:
    //��opencv��dnnģ���onnxģ��
    cv::Mat dnnGetClasses(cv::Mat img, cv::dnn::Net uNetModel);
    //��������ָ�õ���ͼƬ������Ӿ��εĻ�ȡ--��ת����
    std::vector<cv::RotatedRect> getMultipodPackRoatedRect(cv::Mat uNetImg);
    //��������ָ�õ���ͼƬ������Ӿ��εĻ�ȡ--��ͨ����
    std::vector<cv::Rect> getMultipodPackRect(cv::Mat uNetImg);

    //onnx�汾--����ģ��
    Ort::Session loadModel(const std::wstring& model_path);
    //onnx�汾--Ԥ��
    cv::Mat performInference(Ort::Session& session, const cv::Mat& input_image);
    //����������������Ԫ������
    size_t vector_product(const std::vector<int64_t>& vec);
    //���ָ���ӳ�䵽��ɫ�ռ�
    cv::Mat mapToColors(const cv::Mat& segmentation_result);

    std::vector<Ort::Value> runModel(Ort::Session& session, const Ort::Value& input_tensor);
};