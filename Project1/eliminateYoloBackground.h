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
	std::pair<cv::Rect2f, std::vector<cv::Rect2f>> getBoundAndPin(cv::Mat& image,std::string types);

	//BFS并查找小黑块边界矩形
	std::pair<cv::Point, cv::Point> findBoundingRectangle(const cv::Mat& img, int colorRange);
	std::pair<cv::Point, cv::Point> findBoundingRectangle_heibai(const cv::Mat& img, float whiteRatioThreshold);
	
	//从找到的小黑块的边界开始向外搜索
	std::vector<cv::Rect2f> findPinsAroundBlackBox(cv::Mat& img, cv::Rect2f& blackBox, cv::Mat& hsvImg);

	//特把三引脚的分出
	std::vector<cv::Rect2f> findPinsAroundBlackBox_ofThree(cv::Mat& img, cv::Rect2f& blackBox, cv::Mat& hsvImg);

	//轮廓信息测试
	cv::Mat test(cv::Mat &img, std::string types);

	//使用hsv从原图中把引脚筛出来
	cv::Mat useHsvTest(cv::Mat& image);

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
	void adjustRect(cv::Rect& rect, const cv::Size& imageSize);

	//调整引脚框位置(两脚类专用)
	void moveToIntersect(cv::Rect& rectToMove, const cv::Rect& referenceRect);

	//反转图片后转化坐标(两脚类专用)
	void transformRectCoordinates(cv::Rect& rect);

	//判断函数--矩形是否更接近正方形(三脚类专用)
	static bool compareRectsCloseToSquare(const cv::Rect2f& a, const cv::Rect2f& b);

	//判断函数--矩形是否近似相等(多脚类专用)
	static bool rectsAreSimilar(const cv::Rect2f& a, const cv::Rect2f& b);

	//查找出现频率最高的“模板”矩形(多脚类专用)
	cv::Rect2f findTemplateRect(const std::vector<cv::Rect2f>& rects);

	//过滤掉面积大于模板面积两倍的矩形(多脚类专用)
	void filterRects(std::vector<cv::Rect2f>& rects, const cv::Rect2f& templateRect);

	//处理多引脚矩形
	void processRects(std::vector<cv::Rect2f>& rects);

	//计算对称矩形(多引脚专用)
	cv::Rect2f calculateSymmetricRect(const cv::Rect2f& sourceRect, const cv::Rect2f& black_rect);

	//判断对称位置的矩形是否存在(多引脚专用)
	bool isOverlappingMoreThanHalf(const cv::Rect2f& rect1, const cv::Rect2f& rect2);

	//根据对称添加矩形(多引脚专用)
	void addSymmetricRectsIfNeeded(std::vector<cv::Rect2f>& rects, const cv::Rect2f& black_rect);
private:

	static constexpr float SIMILARITY_THRESHOLD = 5.0f;
};