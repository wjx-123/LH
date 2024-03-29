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
	std::pair<cv::Rect2f, std::vector<cv::Rect2f>> getBoundAndPin(cv::Mat& image,std::string types);

	//BFS������С�ڿ�߽����
	std::pair<cv::Point, cv::Point> findBoundingRectangle(const cv::Mat& img, int colorRange);
	std::pair<cv::Point, cv::Point> findBoundingRectangle_heibai(const cv::Mat& img, float whiteRatioThreshold);
	
	//���ҵ���С�ڿ�ı߽翪ʼ��������
	std::vector<cv::Rect2f> findPinsAroundBlackBox(cv::Mat& img, cv::Rect2f& blackBox, cv::Mat& hsvImg);

	//�ذ������ŵķֳ�
	std::vector<cv::Rect2f> findPinsAroundBlackBox_ofThree(cv::Mat& img, cv::Rect2f& blackBox, cv::Mat& hsvImg);

	//������Ϣ����
	cv::Mat test(cv::Mat &img, std::string types);

	//ʹ��hsv��ԭͼ�а�����ɸ����
	cv::Mat useHsvTest(cv::Mat& image);

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
	void adjustRect(cv::Rect& rect, const cv::Size& imageSize);

	//�������ſ�λ��(������ר��)
	void moveToIntersect(cv::Rect& rectToMove, const cv::Rect& referenceRect);

	//��תͼƬ��ת������(������ר��)
	void transformRectCoordinates(cv::Rect& rect);

	//�жϺ���--�����Ƿ���ӽ�������(������ר��)
	static bool compareRectsCloseToSquare(const cv::Rect2f& a, const cv::Rect2f& b);

	//�жϺ���--�����Ƿ�������(�����ר��)
	static bool rectsAreSimilar(const cv::Rect2f& a, const cv::Rect2f& b);

	//���ҳ���Ƶ����ߵġ�ģ�塱����(�����ר��)
	cv::Rect2f findTemplateRect(const std::vector<cv::Rect2f>& rects);

	//���˵��������ģ����������ľ���(�����ר��)
	void filterRects(std::vector<cv::Rect2f>& rects, const cv::Rect2f& templateRect);

	//��������ž���
	void processRects(std::vector<cv::Rect2f>& rects);

	//����Գƾ���(������ר��)
	cv::Rect2f calculateSymmetricRect(const cv::Rect2f& sourceRect, const cv::Rect2f& black_rect);

	//�ж϶Գ�λ�õľ����Ƿ����(������ר��)
	bool isOverlappingMoreThanHalf(const cv::Rect2f& rect1, const cv::Rect2f& rect2);

	//���ݶԳ���Ӿ���(������ר��)
	void addSymmetricRectsIfNeeded(std::vector<cv::Rect2f>& rects, const cv::Rect2f& black_rect);
private:

	static constexpr float SIMILARITY_THRESHOLD = 5.0f;
};