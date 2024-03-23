#pragma once
#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <omp.h>  // “˝»ÎOpenMP
//#include <opencv2/cudaarithm.hpp>
class colorMethod
{
public:
	colorMethod();
	~colorMethod();
	void test(cv::Mat& img);
	void test1(cv::Mat& img);

	void test2();
private:

};

