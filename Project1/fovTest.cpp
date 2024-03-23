#include"fovTest.h"

FovRoute::FovRoute(int k, int epsilon, int lambda_value, int alpha_k)
    : k(k), epsilon(epsilon), lambda_value(lambda_value), alpha_k(alpha_k) {
    std::cout << "start" << std::endl;
}

std::vector<cv::Rect> FovRoute::coverPointsWithRectangles(const std::vector<cv::Rect2f>& smallRects, int width, int height, cv::Size imageSize) {
    /*std::vector<cv::Point> sortedPoints = points;
    std::sort(sortedPoints.begin(), sortedPoints.end(), [](const cv::Point& a, const cv::Point& b) {
    return a.x < b.x;
    });

    std::vector<cv::Rect> result;
    for (const auto& point : sortedPoints) {
        bool placed = false;
        for (const auto& rect : result) {
            if (isPointInRect(point, rect)) {
                placed = true;
                break;
            }
        }

        if (!placed) {
            cv::Rect newRect = findBestNewRect(point, width, height, result);
            result.push_back(newRect);
        }
    }

    return result;*/

    std::vector<cv::Rect> result;
    for (const auto& smallRect : smallRects) {
        bool placed = false;
        for (const auto& bigRect : result) {
            if (bigRect.contains(smallRect.tl()) && bigRect.contains(smallRect.br())) {
                placed = true;
                break;
            }
        }

        if (!placed) {
            cv::Rect newRect = findBestNewRect(smallRect, width, height, result,imageSize, smallRects);
            result.push_back(newRect);
        }
    }

    return result;
}

bool FovRoute::isPointInRect(const cv::Point& point, const cv::Rect& rect) {
    return rect.contains(point);
}

cv::Rect FovRoute::findBestNewRect(const cv::Rect2f& smallRect, int width, int height, const std::vector<cv::Rect>& existingRects, const cv::Size& imageSize, const std::vector<cv::Rect2f>& allSmallRects) {
    /*std::vector<cv::Rect> possibleRects = {
    cv::Rect(point.x - width, point.y - height / 2, width, height),
    cv::Rect(point.x, point.y - height / 2, width, height),
    cv::Rect(point.x - width / 2, point.y - height, width, height),
    cv::Rect(point.x - width / 2, point.y, width, height)
};

    int minOverlapArea = INT_MAX;
    cv::Rect bestRect;
    for (const auto& newRect : possibleRects) {
        int overlapArea = 0;
        for (const auto& existingRect : existingRects) {
            overlapArea += getOverlapArea(newRect, existingRect);
        }
        if (overlapArea < minOverlapArea) {
            minOverlapArea = overlapArea;
            bestRect = newRect;
        }
    }

    return bestRect;*/

    std::vector<cv::Rect> possibleRects = {
        cv::Rect(smallRect.x - width, smallRect.y - height / 2, width, height),
        cv::Rect(smallRect.x, smallRect.y - height / 2, width, height),
        cv::Rect(smallRect.x - width / 2, smallRect.y - height, width, height),
        cv::Rect(smallRect.x - width / 2, smallRect.y, width, height)
    };

    int minOverlapArea = INT_MAX;
    int maxContainedSmallRects = 0;
    cv::Rect bestRect;
    for (const auto& newRect : possibleRects) {
        int overlapArea = 0;
        int containedSmallRects = 0;
        for (const auto& existingRect : existingRects) {
            overlapArea += getOverlapArea(newRect, existingRect);
        }
        for (const auto& rect : allSmallRects) {
            if (newRect.contains(rect.tl()) && newRect.contains(rect.br())) {
                containedSmallRects++;
            }
        }

        // 选择重叠面积最小且包含最多未覆盖小框的新矩形
        if ((containedSmallRects > maxContainedSmallRects) ||
            (containedSmallRects == maxContainedSmallRects && overlapArea < minOverlapArea)) {
            minOverlapArea = overlapArea;
            maxContainedSmallRects = containedSmallRects;
            bestRect = newRect;
        }
    }

    return bestRect;
}

int FovRoute::getOverlapArea(const cv::Rect& rect1, const cv::Rect& rect2) {
    cv::Rect intersection = rect1 & rect2;
    return intersection.area();
}


//#include <iostream>
//#include"onnx.h"
//int main() {
//    Net_config yolo_nets = { 0.4, 0.4, 0.4,"best6.onnx" };//best_rpc.onnx   10.12_rpc
//    YOLO yolo_model(yolo_nets);
//
//
//    vector<cv::Rect2f> smt_frame;
//
//    cv::Mat img1 = cv::imread("C:\\Users\\LENOVO\\Pictures\\1\\1\\1.jpg");
//    smt_frame = yolo_model.detect(img1);
//
//    std::vector<cv::Point> points;
//
//    for (auto dian : smt_frame)
//    {
//        points.push_back(cv::Point(dian.x, dian.y));
//        cv::rectangle(img1, dian, cv::Scalar(2, 255, 255), 10);
//    }
//
//    FovRoute fov_route(1, 1, 1, 1);
//    //std::vector<cv::Point> points = { {100, 150}, {200, 250}, {300, 350} };
//    int width = 2448;
//    int height = 2048;
//
//    std::vector<cv::Rect> rectangles = fov_route.coverPointsWithRectangles(points, width, height);
//
//    for (const auto& rect : rectangles) {
//        if (rect.x < 0 || rect.y < 0)
//        {
//            rectangles.pop_back();
//            continue;
//        }
//        cv::rectangle(img1, rect, cv::Scalar(2, 2, 255), 20);
//    }
//    std::cout << rectangles.size() << std::endl;
//   
//    cv::imwrite("C:\\Users\\LENOVO\\Pictures\\1122Test.jpg", img1);
//    return 0;
//
//}
//
