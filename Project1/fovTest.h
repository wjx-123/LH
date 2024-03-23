#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>


class FovRoute {
public:
    FovRoute(int k, int epsilon, int lambda_value, int alpha_k);

    std::vector<cv::Rect> coverPointsWithRectangles(const std::vector<cv::Rect2f>& smallRects, int width, int height, cv::Size imageSize);

private:
    int k, epsilon, lambda_value, alpha_k;

    bool isPointInRect(const cv::Point& point, const cv::Rect& rect);

    cv::Rect findBestNewRect(const cv::Rect2f& smallRect, int width, int height, const std::vector<cv::Rect>& existingRects, const cv::Size& imageSize, const std::vector<cv::Rect2f>& allSmallRects);

    int getOverlapArea(const cv::Rect& rect1, const cv::Rect& rect2);
};