#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
//����ļ��Ǻ���ȱ�ݼ������ܷ����������淶����д���solderDefectAI
/****************************************
@brief     :  �ָ�onnxruntime
@input	   :  ͼ��
@output    :  ��Ĥ
*****************************************/
void center_resize(cv::Mat& src);
void SegmentAIONNX(cv::Mat& imgSrc, int width, int height);