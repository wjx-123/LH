#pragma once
#pragma once
#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc_c.h>

class splicBoard {
struct nms_data_struct//nms专用
{
     float val;
     cv::Point loc;
     cv::Rect roi;

     void getRect(cv::Size temp_size)
     {
         roi = cv::Rect(loc.x, loc.y, temp_size.width, temp_size.height);
     }
    };
public:
    //模板匹配方法
	cv::Mat matchingSplic(cv::Mat image, std::vector<cv::Rect2f> rect);
    //筛选直线
    std::vector<cv::Rect2f> linesMatchSplic(cv::Mat image, std::vector<cv::Rect2f> rect);

private:
    //平方差匹配法  速度太慢 
	void ColorMatch_TM_SQDIFF(const cv::Mat& img, const cv::Mat& templ);
    //nms 排序
    static bool comp(nms_data_struct data1, nms_data_struct data2); 
    //nms 看两个框是否相交，相交则直接舍弃，否则保留
    int nms_detect(nms_data_struct data, std::vector<nms_data_struct> data_list);
    //nms
    int nms_temp_min(cv::Mat& input, cv::Mat result, cv::Size temp_size, bool save_output = true, float thr = 0.03);
};
