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
	* std::vector<cv::Point2f> physics:传入三个点的物理坐标
	* std::vector<cv::Point2f> pixel:传入三个点的像素坐标
	* double& cameraAngle:返回相机偏转角度
	* cv::Matx22d& modelMartrix:返回矫正矩阵
	* 返回一像素对应的实际坐标(0.01333)
	*/
	double calibration(std::vector<cv::Point2f> physics, std::vector<cv::Point2f> pixel, double& cameraAngle, cv::Matx22d& modelMartrix);
private:
	//获取左上角、右下角和坐下或右上的坐标 返回顺序 左上，右下，左下
	//这个函数目的是为了可以随意传入输入坐标的顺序，但是由于模组的原点不是在最边上，所以这个暂时不用
	std::vector<cv::Point2f> getMaxAndMin(std::vector<cv::Point2f> temp);

	//以坐下角为原点，计算角度  返回值两个角度
	std::vector<double> calculateAngle(std::vector<cv::Point2f> temp);

	//计算两点距离
	double distance(double x1, double y1, double x2, double y2);

	//通过角度得到转化矩阵
	//sign是标志位 true表示锐角 false表示钝角
	cv::Matx22d getConversionMatrix(double angle, bool sign);

	//计算pix(一像素对应多少物理坐标)传入三个点的物理坐标 和三个点的像素坐标
	double getPix(std::vector<cv::Point2f> physics, std::vector<cv::Point2f> pixel);
};
