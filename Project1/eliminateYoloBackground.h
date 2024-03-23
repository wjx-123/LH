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

	//���ú��� ����۳���Сͼ������(����T/����R/���D) ����vector<pair>
	std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> getBoundAndPin(cv::Mat& image,std::string types);

	//BFS������С�ڿ�߽����
	std::pair<cv::Point, cv::Point> findBoundingRectangle(const cv::Mat& img, int colorRange);
	std::pair<cv::Point, cv::Point> findBoundingRectangle_heibai(const cv::Mat& img, float whiteRatioThreshold);
	
	//���ҵ���С�ڿ�ı߽翪ʼ��������
	std::vector<cv::Rect2f> findPinsAroundBlackBox(cv::Mat& img, cv::Rect2f& blackBox, cv::Mat& hsvImg);

	//�ذ������ŵķֳ�
	std::vector<cv::Rect2f> findPinsAroundBlackBox_ofThree(cv::Mat& img, cv::Rect2f& blackBox, cv::Mat& hsvImg);

	cv::Mat test(cv::Mat &img, std::string types);//������Ϣ����

	cv::Mat useHsvTest(cv::Mat& image);//ʹ��hsv��ԭͼ�а�����ɸ����

private:
	//ȫ������������ÿһ��С���� ����������������׼�� 0��������ıߣ�1������ߵģ�2�����ϱߣ�3�����ұ�
	std::vector<cv::Rect> getPinRect(cv::Mat& imgInit, cv::Rect rectInit, char baseline, int closestEdge);

	//�ذ������ŷֳ�
	std::vector<cv::Rect> getPinRect_ofThree(cv::Mat& imgInit, cv::Rect rectInit, char baseline, int closestEdge);

	//�����ɫ���ص�����
	int countWhitePixels(const cv::Mat& line);

	//����߽�Ϊ��׼���ϲ�����
	void mergeRectanglesByLeftEdge(std::vector<cv::Rect>& rectangles, int threshold);

	//���ϱ߽�Ϊ��׼���ϲ�����
	void mergeRectanglesByTopEdge(std::vector<cv::Rect>& rectangles, int threshold);

	//���������Ƿ��а�ɫ���ص�
	bool containsWhitePixel(const cv::Mat& image, const cv::Rect& rect);

	//���ݾ������ľ��εľ���ɸѡ
	void filterRectangles(std::vector<cv::Rect>& rectangles, const cv::Rect& rectInit, int closestEdge, double thresholdRatio);

	//�����ڿ�λ��(������ר��)
	void adjustRect(cv::Rect2f& rect, const cv::Size& imageSize);

	//�������ſ�λ��(������ר��)
	void moveToIntersect(cv::Rect& rectToMove, const cv::Rect& referenceRect);
};