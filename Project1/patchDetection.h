#pragma once
#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

class pathDetection
{
private:
	/*贴片的中间框检测小黑块的有无*/
	bool ifBlackBlockExist(const cv::Mat& img, const cv::Mat& templ, double threshold = 0.2);

	/*旋转模板图*/
	cv::Mat rotateImg(const cv::Mat& src, double angle);
public:

	/*通过设置黑色像素的阈值找外边框 暂时没用*/
	cv::Rect2f getBlackBlock(cv::Mat image);

	/* 
	* 判断各个角度的模板是否存在（小黑块有无）
	* img:待检测的图
	* temp1:模板图
	* angel:允许模板图旋转的角度，默认检测模板图的方向和旋转九十度
	* threshold:设置的检测阈值,默认0.2
	*/
	bool variousAngleExist(const cv::Mat& img, const cv::Mat& templ, double angle,double threshold = 0.2);

	/*
	* 检测贴片中间小黑块的偏移角度
	* 输入 贴片的外框
	*/
	double detectOffsetAngle(cv::Mat img);

	/*
	* 判断是否错件
	* img:检测图
	* temp:模板图
	* threshold:阈值 默认0.8
	* 返回:true or false
	*/
	double ifDetectionWrong(cv::Mat& img, cv::Mat& temp, double threshold = 0.8);

	
};

