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

	/*
	* չ���ľ���ͼ�����תΪ����ͼƬ�������
	* rectPoint ��������
	* center �������ĵ������
	* innerRadius ���ε��ڰ뾶
	* outerRadius ��뾶
	*/
	cv::Point rectToCircle(const cv::Point& rectPoint, const cv::Point& center, int innerRadius, int outerRadius);

	/*
	* �ҳ�ȱ��λ��
	* ���� ͼƬ
	* ���ȱ��
	*/
	std::vector<std::vector<cv::Point>> detectDefects(const cv::Mat& img);



private:
	/*
	* rectתΪ�ĸ�point
	*/
	std::vector<std::vector<cv::Point>> convertRectsToPoints(const std::vector<cv::Rect>& defectRects);

	/*����λ��*/
	double findMedian(std::vector<int> nums);
};