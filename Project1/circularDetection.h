#pragma once
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class circular {
public:
	circular();
	~circular();

	/*
	* cv::Mat circle :ԭʼ����ͼ��
	* cv::Point center : ԭʼͼ����Բ����Բ��
	* int Radius: �ڰ뾶
	* int RingStride: ��뾶
	*/
	cv::Mat circleToRectangle(const cv::Mat& src, const cv::Point& center, int innerRadius, int outerRadius);

	/*
	* ����ͼƬ��Բ
	*/
	std::vector<cv::Vec3f> detectAndDrawCircles(const cv::Mat src);
};