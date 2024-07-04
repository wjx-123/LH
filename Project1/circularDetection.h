#pragma once
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class circular {
public:
	circular();
	~circular();

	/*
	* cv::Mat circle :原始环形图像
	* cv::Point center : 原始图像中圆环的圆心
	* int Radius: 内半径
	* int RingStride: 外半径
	*/
	cv::Mat circleToRectangle(const cv::Mat& src, const cv::Point& center, int innerRadius, int outerRadius);

	/*
	* 输入图片找圆
	*/
	std::vector<cv::Vec3f> detectAndDrawCircles(const cv::Mat src);

	/*
	* 展开的矩形图坐标点转为环形图片的坐标点
	* rectPoint 矩形坐标
	* center 环形中心点的坐标
	* innerRadius 环形的内半径
	* outerRadius 外半径
	*/
	cv::Point rectToCircle(const cv::Point& rectPoint, const cv::Point& center, int innerRadius, int outerRadius);

	/*
	* 找出缺陷位置
	* 输入 图片
	* 输出缺陷
	*/
	std::vector<std::vector<cv::Point>> detectDefects(const cv::Mat& img);



private:
	/*
	* rect转为四个point
	*/
	std::vector<std::vector<cv::Point>> convertRectsToPoints(const std::vector<cv::Rect>& defectRects);

	/*找中位数*/
	double findMedian(std::vector<int> nums);
};