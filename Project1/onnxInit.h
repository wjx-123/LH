#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

//using namespace std;yolo一件搜索
//using namespace cv;
//using namespace Ort;

struct Net_config
{
	float confThreshold; // Confidence threshold置信度阈值
	float nmsThreshold;  // Non-maximum suppression threshold 非极大抑制
	float objThreshold;  //Object Confidence threshold 对象置信度
	std::string modelpath;
};

typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

class YOLOInit
{
public:
	YOLOInit(Net_config config);
	~YOLOInit();

public:
	const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

	const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
						   {436, 615, 739, 380, 925, 792} };
public:
	//frame传入的图，uNetModel unet模型, useUnet是否使用unet获取子框，不使用的话则有传统筛选颜色方法获取
	std::vector<std::pair<cv::Rect2f, cv::Rect2f>> detectInit(cv::Mat& frame);
	int endsWithInit(std::string s, std::string sub);
	std::vector<std::vector<cv::Rect2f>> changeBoxInfoToRectInit(std::vector<BoxInfo> temp);//转格式
private:
	float* anchors;
	int num_stride;
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	std::vector<std::string> class_names;
	int num_class;
	int seg_num_class;

	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	const bool keep_ratio = true;
	std::vector<float> input_image_;
	void normalize_Init(cv::Mat img);
	void nmsInit(std::vector<BoxInfo>& input_boxes);
	cv::Mat resize_imageInit(cv::Mat srcimg, int* newh, int* neww, int* top, int* left);

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1");//创建环境
	//Env env1;
	Ort::Session* ort_session = nullptr;
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	std::vector<char*> input_names;
	std::vector<char*> output_names;
	std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs

	std::vector<cv::Mat>images;//图像容器 
	std::vector<BoxInfo> generate_boxes;// 框
};