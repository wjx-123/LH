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
//fov·���滮���ƴͼ
// ����һ���ṹ�����洢��ת�ǶȺ�ƽ����
struct Transform {
	double rotationAngle;  // ��ת�Ƕȣ��ȣ�
	std::pair<int, int> translation;  // ƽ������xƫ��, yƫ�ƣ�
};

class fov_scan {
public:
	cv::Mat fov_puzzle(std::vector<cv::Rect2f> fov, std::vector<cv::Mat> fov_img, cv::Mat initImage);
	
	//��fov��x���갴�հ�����߶Գ�
	std::vector<cv::Point2f> fovCoordinateSymmetry(std::vector<cv::Point2f> fov, cv::Point2f qishidian, cv::Point2f zhongzhidian);

	std::vector<cv::Rect2f> fovCoordinateSymmetryPix(std::vector<cv::Rect2f> fov, cv::Mat initImage);

	//����maskͼ�ҵ�Բ��
	cv::Point2f getCircleCenter(cv::Mat);

	//����ƫ�ƺ�ԭͼ�õ����ͼ
	cv::Mat getShiftImage(cv::Mat &iniImg, int shiftX, int shiftY);

	//������ƥ����ƫ��
	cv::Point2f getSHift(cv::Mat mMask, cv::Mat dMask);

	// ��ת���� ��ת������ʱ��Ϊ��
	cv::Mat RotateImg(cv::Mat image, double angle);

	//��������ƫ��������ƫ�ƽǶȺ�ƽ�ƾ���
	Transform calculateTransform(const std::pair<int, int>& offset1, const std::pair<int, int>& offset2);
};
	