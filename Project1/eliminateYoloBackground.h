#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <future>
#include <numeric>

class eliminateYoloBackground {
public:
	eliminateYoloBackground();
	~eliminateYoloBackground();

	//调用函数 输入扣出的小图和类型(两脚T/三脚R/多脚D) 返回vector<pair>
	std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> getBoundAndPin(cv::Mat& image,std::string types);

	//BFS并查找小黑块边界矩形
	std::pair<cv::Point, cv::Point> findBoundingRectangle(const cv::Mat& img, int colorRange);
	std::pair<cv::Point, cv::Point> findBoundingRectangle_heibai(const cv::Mat& img, float whiteRatioThreshold);
	
	//从找到的小黑块的边界开始向外搜索
	std::vector<cv::Rect2f> findPinsAroundBlackBox(cv::Mat& img, cv::Rect2f& blackBox, cv::Mat& hsvImg);

	//特把三引脚的分出
	std::vector<cv::Rect2f> findPinsAroundBlackBox_ofThree(cv::Mat& img, cv::Rect2f& blackBox, cv::Mat& hsvImg);

	cv::Mat test(cv::Mat &img, std::string types);//轮廓信息测试

	cv::Mat useHsvTest(cv::Mat& image);//使用hsv从原图中把引脚筛出来

private:
	//全部搜索，输入每一个小区域 后面两个是两个基准线 0代表下面的边，1代表左边的，2代表上边，3代表右边
	std::vector<cv::Rect> getPinRect(cv::Mat& imgInit, cv::Rect rectInit, char baseline, int closestEdge);

	//特把三引脚分出
	std::vector<cv::Rect> getPinRect_ofThree(cv::Mat& imgInit, cv::Rect rectInit, char baseline, int closestEdge);

	//计算白色像素的数量
	int countWhitePixels(const cv::Mat& line);

	//以左边界为基准，合并矩形
	void mergeRectanglesByLeftEdge(std::vector<cv::Rect>& rectangles, int threshold);

	//以上边界为基准，合并矩形
	void mergeRectanglesByTopEdge(std::vector<cv::Rect>& rectangles, int threshold);

	//检查矩形中是否有白色像素点
	bool containsWhitePixel(const cv::Mat& image, const cv::Rect& rect);

	//根据距离中心矩形的距离筛选
	void filterRectangles(std::vector<cv::Rect>& rectangles, const cv::Rect& rectInit, int closestEdge, double thresholdRatio);

	//调整黑块位置(两脚类专用)
	void adjustRect(cv::Rect2f& rect, const cv::Size& imageSize);

	//调整引脚框位置(两脚类专用)
	void moveToIntersect(cv::Rect& rectToMove, const cv::Rect& referenceRect);
};