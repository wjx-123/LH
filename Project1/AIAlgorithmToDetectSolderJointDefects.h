#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
//这个文件是焊点缺陷检测的智能方法，但不规范，重写后见solderDefectAI
/****************************************
@brief     :  分割onnxruntime
@input	   :  图像
@output    :  掩膜
*****************************************/
void center_resize(cv::Mat& src);
void SegmentAIONNX(cv::Mat& imgSrc, int width, int height);