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
