#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <utility>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//fov路径规划后的拼图
// 定义一个结构体来存储旋转角度和平移量
struct Transform {
	double rotationAngle;  // 旋转角度（度）
	std::pair<int, int> translation;  // 平移量（x偏移, y偏移）
};

class fov_scan {
public:
	cv::Mat fov_puzzle(std::vector<cv::Rect2f> fov, std::vector<cv::Mat> fov_img, cv::Mat initImage);
	
	//把fov的x坐标按照板的中线对称
	std::vector<cv::Point2f> fovCoordinateSymmetry(std::vector<cv::Point2f> fov, cv::Point2f qishidian, cv::Point2f zhongzhidian);

	std::vector<cv::Rect2f> fovCoordinateSymmetryPix(std::vector<cv::Rect2f> fov, cv::Mat initImage);

	//输入mask图找到圆心
	cv::Point2f getCircleCenter(cv::Mat);

	//根据偏移和原图得到结果图
	cv::Mat getShiftImage(cv::Mat &iniImg, int shiftX, int shiftY);

	//特征点匹配求偏移
	cv::Point2f getSHift(cv::Mat mMask, cv::Mat dMask);

	// 旋转函数 旋转方向逆时针为正
	cv::Mat RotateImg(cv::Mat image, double angle);

	//根据两个偏移量返回偏移角度和平移距离
	Transform calculateTransform(const std::pair<int, int>& offset1, const std::pair<int, int>& offset2);
};
	