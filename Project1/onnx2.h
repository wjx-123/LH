#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include <tuple>
#include "eliminateYoloBackground.h"


//using namespace std;yoloһ������
//using namespace cv;
using namespace Ort;

struct Net_config
{
	float confThreshold; // Confidence threshold���Ŷ���ֵ
	float nmsThreshold;  // Non-maximum suppression threshold �Ǽ�������
	float objThreshold;  //Object Confidence threshold �������Ŷ�
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

class YOLO
{
public:
	YOLO(Net_config config);
	~YOLO();

public:
	const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

	const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
						   {436, 615, 739, 380, 925, 792} };
public:

	/*
	* std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>>
	* <�ڿ飬vector<����>>
	*/
	std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> getCPCoordinate(cv::Mat& img);

	int endsWith(std::string s, std::string sub);
	std::vector<std::vector<cv::Rect2f>> changeBoxInfoToRect(std::vector<BoxInfo> temp);//ת��ʽ
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
	
	void normalize_(cv::Mat img);
	void nms(std::vector<BoxInfo>& input_boxes);
	cv::Mat resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1");//��������
	//Env env1;
	Ort::Session* ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	std::vector<char*> input_names;
	std::vector<char*> output_names;
	std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs

	std::vector<cv::Mat>images;//ͼ������ 
	std::vector<BoxInfo> generate_boxes;// ��

	eliminateYoloBackground eliminateYoloBackground;


	/*
	* �ҵ���������
	*/
	std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> getBlackPosition(std::vector<BoxInfo> generate_boxes, cv::Mat& frame);

	/*
	* �ָ�ͼƬ
	*/
	std::vector<std::tuple<cv::Mat, int, int>> splitImage(const cv::Mat& img, int max_width = 2448, int max_height = 2048, float overlap = 0.1);

	std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> detect(cv::Mat& frame);
};