#include "circularDetection.h"

circular::circular()
{
}

circular::~circular()
{
}


cv::Mat circular::circleToRectangle(const cv::Mat& src, const cv::Point& center, int innerRadius, int outerRadius)
{
    int ringWidth = outerRadius - innerRadius;
    int ringCircumference = 2 * CV_PI * outerRadius;

    cv::Mat rectImg(cv::Size(ringCircumference, ringWidth), src.type());

    for (int y = 0; y < ringWidth; y++) {
        for (int x = 0; x < ringCircumference; x++) {
            double theta = 2 * CV_PI * x / ringCircumference;
            int r = innerRadius + y;

            int srcX = center.x + r * cos(theta);
            int srcY = center.y + r * sin(theta);

            if (srcX >= 0 && srcX < src.cols && srcY >= 0 && srcY < src.rows) {
                rectImg.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(srcY, srcX);
            }
        }
    }

    return rectImg;
}

std::vector<cv::Vec3f> circular::detectAndDrawCircles(const cv::Mat src)
{
    cv::Mat gray;
    // 将图像转换为灰度图像
    cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 进行高斯模糊，减少噪声
    GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);

    std::vector<cv::Vec3f> circles;
    // 使用HoughCircles函数检测圆
    HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows / 8, 200, 100, 0, 0);

    // 画出检测到的圆并输出坐标
    //for (size_t i = 0; i < circles.size(); i++) {
    //    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    //    int radius = cvRound(circles[i][2]);
    //    // 画圆心
    //    circle(src, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
    //    // 画圆周
    //    circle(src, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
    //    std::cout << "Circle " << i + 1 << ": Center = (" << center.x << ", " << center.y << "), Radius = " << radius << std::endl;
    //}

    // 显示结果
    /*cv::namedWindow("Detected Circles", cv::WINDOW_AUTOSIZE);
    cv::imshow("Detected Circles", src);
    cv::waitKey(0)*/;
	return circles;
}

cv::Point circular::rectToCircle(const cv::Point& rectPoint, const cv::Point& center, int innerRadius, int outerRadius)
{
    // 计算外周长
    int ringCircumference = 2 * CV_PI * outerRadius;

    // 计算角度 theta
    double theta = 2 * CV_PI * rectPoint.x / ringCircumference;

    // 计算半径 r
    int r = innerRadius + rectPoint.y;

    // 将极坐标转换为笛卡尔坐标
    int srcX = center.x + r * cos(theta);
    int srcY = center.y + r * sin(theta);

    return cv::Point(srcX, srcY);
}

std::vector<std::vector<cv::Point>> circular::detectDefects(const cv::Mat& img)
{
    /*这一段是普通图片*/
    //// 转换为灰度图
    //cv::Mat gray;
    //cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    //// 应用高斯模糊去噪
    //cv::Mat blurred;
    //cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    //// 使用自适应阈值操作提取黑色区域
    //cv::Mat binary;
    //cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);

    //// 应用形态学操作去除噪声
    //cv::Mat morph;
    //cv::morphologyEx(binary, morph, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4)));

    //// 检测轮廓
    //std::vector<std::vector<cv::Point>> contours;
    //cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //// 筛选面积大于等于50的轮廓
    //std::vector<std::vector<cv::Point>> filteredContours;
    //for (const auto& contour : contours) {
    //    double area = cv::contourArea(contour);
    //    if (area >= 50) {
    //        filteredContours.push_back(contour);
    //    }
    //}

    //return filteredContours;


    //这一段用滑动窗口来做
    std::vector<cv::Rect> defectRects;

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 使用阈值操作提取白色区域
    cv::Mat binary;
    cv::threshold(gray, binary, 200, 255, cv::THRESH_BINARY); // 白色区域阈值

    // 检测白色区域轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<int> heightOfWhite; // 记录每个白色区域的height

    // 遍历每个白色区域
    for (const auto& contour : contours) {
        cv::Rect boundingBox = cv::boundingRect(contour);
        if (boundingBox.height > 20 && boundingBox.height < 500) {
            heightOfWhite.push_back(boundingBox.height);
        }
    }

    double h_median = findMedian(heightOfWhite);

    // 遍历每个白色区域并检测缺陷
    for (const auto& contour : contours) {
        cv::Rect boundingBox = cv::boundingRect(contour);
        if (boundingBox.height > 20 && boundingBox.height < 500) {
            if (boundingBox.area() > 300 && boundingBox.height < h_median - 10) { // 如果白色区域高度小于中位数一定范围
                cv::Rect defectRect(boundingBox.x, boundingBox.y + boundingBox.height, boundingBox.width, h_median - boundingBox.height);
                defectRects.push_back(defectRect);
            }
            else {
                cv::Mat roi = img(boundingBox);

                // 转换为灰度图
                cv::Mat roiGray;
                cv::cvtColor(roi, roiGray, cv::COLOR_BGR2GRAY);

                // 使用自适应阈值操作提取黑色区域
                cv::Mat roiBinary;
                cv::adaptiveThreshold(roiGray, roiBinary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);

                // 应用形态学操作去除噪声
                cv::Mat morph;
                cv::morphologyEx(roiBinary, morph, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

                // 检测黑色区域轮廓
                std::vector<std::vector<cv::Point>> blackContours;
                cv::findContours(morph, blackContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                // 筛选面积大于等于50且长宽比不超过一定比例的轮廓，并调整坐标
                for (const auto& blackContour : blackContours) {
                    double area = cv::contourArea(blackContour);
                    if (area >= 50) {
                        cv::Rect blackBoundingBox = cv::boundingRect(blackContour);
                        double aspectRatio = static_cast<double>(blackBoundingBox.width) / blackBoundingBox.height;

                        // 筛选长宽比小于一定比例的轮廓
                        if (aspectRatio > 0.2) {
                            blackBoundingBox.x += boundingBox.x;
                            blackBoundingBox.y += boundingBox.y;
                            defectRects.push_back(blackBoundingBox);
                        }
                    }
                }
            }
        }
    }

    // 转成点
    return convertRectsToPoints(defectRects);
}

std::vector<std::vector<cv::Point>> circular::convertRectsToPoints(const std::vector<cv::Rect>& defectRects)
{
    std::vector<std::vector<cv::Point>> rectPoints;

    for (const auto& rect : defectRects) {
        std::vector<cv::Point> points;
        points.emplace_back(rect.x, rect.y); // 左上角
        points.emplace_back(rect.x + rect.width, rect.y); // 右上角
        points.emplace_back(rect.x + rect.width, rect.y + rect.height); // 右下角
        points.emplace_back(rect.x, rect.y + rect.height); // 左下角
        rectPoints.push_back(points);
    }

    return rectPoints;
}

double circular::findMedian(std::vector<int> nums)
{
    // 排序
    std::sort(nums.begin(), nums.end());

    // 计算中位数
    size_t size = nums.size();
    if (size % 2 == 0) {
        // 偶数个元素，中位数是中间两个元素的平均值
        return (nums[size / 2 - 1] + nums[size / 2]) / 2.0;
    }
    else {
        // 奇数个元素，中位数是中间的那个元素
        return nums[size / 2];
    }
}
