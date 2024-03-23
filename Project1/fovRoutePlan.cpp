#include "fovRoutePlan.h"

fovRoute::fovRoute()
{
}

fovRoute::~fovRoute()
{
}

std::vector<cv::Rect> fovRoute::coverPointsWithRectangles(const std::vector<cv::Point>& points, int width, int height)
{
    std::vector<cv::Point> sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
        });

    std::vector<cv::Rect> result;
    for (const auto& point : sortedPoints) {
        bool placed = false;
        // 检查是否可以将点放入已有的矩形框中
        for (auto& rect : result) {
            if (point.x >= rect.x && point.x < rect.x + rect.width &&
                point.y >= rect.y && point.y < rect.y + rect.height ) {
                placed = true;
                break;
            }
        }
        // 如果点没有被放入任何矩形框，则在该点周围创建一个新的矩形框
        if (!placed) {
      /*      cv::Point topLeft(std::max(0, point.x - width / 2),
                std::max(0, point.y - height / 2));
            cv::Point bottomRight(topLeft.x + width, topLeft.y + height);

            result.push_back(cv::Rect(topLeft, bottomRight));*/
            cv::Rect newRect = findBestNewRect(point, width, height, result);
            result.push_back(newRect);
        }
    }

    return result;
}

cv::Rect fovRoute::findBestNewRect(const cv::Point& point, int width, int height, const std::vector<cv::Rect>& existingRects)
{
    // 可能的新矩形框位置列表
    std::vector<cv::Rect> possibleRects;

    // 检查四个方向的位置
    possibleRects.push_back(cv::Rect(point.x - width, point.y - height / 2, width, height));
    possibleRects.push_back(cv::Rect(point.x, point.y - height / 2, width, height));
    possibleRects.push_back(cv::Rect(point.x - width / 2, point.y - height, width, height));
    possibleRects.push_back(cv::Rect(point.x - width / 2, point.y, width, height));

    // 评估每个位置的重叠面积
    int minOverlapArea = INT_MAX;
    cv::Rect bestRect;
    for (const auto& rect : possibleRects) {
        int overlapArea = 0;
        for (const auto& existingRect : existingRects) {
            overlapArea += (rect & existingRect).area();
        }
        if (overlapArea < minOverlapArea) {
            minOverlapArea = overlapArea;
            bestRect = rect;
        }
    }

    return bestRect;
}
