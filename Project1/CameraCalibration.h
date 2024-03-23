#pragma once
#include <iostream>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

class caneraCalibration
{
public:
	caneraCalibration();
	~caneraCalibration();
public:
	/*
	* std::vector<cv::Point2f> physics:�������������������
	* std::vector<cv::Point2f> pixel:�������������������
	* double& cameraAngle:�������ƫת�Ƕ�
	* cv::Matx22d& modelMartrix:���ؽ�������
	* ����һ���ض�Ӧ��ʵ������(0.01333)
	*/
	double calibration(std::vector<cv::Point2f> physics, std::vector<cv::Point2f> pixel, double& cameraAngle, cv::Matx22d& modelMartrix);
private:
	//��ȡ���Ͻǡ����½Ǻ����»����ϵ����� ����˳�� ���ϣ����£�����
	//�������Ŀ����Ϊ�˿������⴫�����������˳�򣬵�������ģ���ԭ�㲻��������ϣ����������ʱ����
	std::vector<cv::Point2f> getMaxAndMin(std::vector<cv::Point2f> temp);

	//�����½�Ϊԭ�㣬����Ƕ�  ����ֵ�����Ƕ�
	std::vector<double> calculateAngle(std::vector<cv::Point2f> temp);

	//�����������
	double distance(double x1, double y1, double x2, double y2);

	//ͨ���Ƕȵõ�ת������
	//sign�Ǳ�־λ true��ʾ��� false��ʾ�۽�
	cv::Matx22d getConversionMatrix(double angle, bool sign);

	//����pix(һ���ض�Ӧ������������)������������������� �����������������
	double getPix(std::vector<cv::Point2f> physics, std::vector<cv::Point2f> pixel);
};
