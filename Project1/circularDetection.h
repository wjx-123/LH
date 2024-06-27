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
};