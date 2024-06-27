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
    // ��ͼ��ת��Ϊ�Ҷ�ͼ��
    cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // ���и�˹ģ������������
    GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);

    std::vector<cv::Vec3f> circles;
    // ʹ��HoughCircles�������Բ
    HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows / 8, 200, 100, 0, 0);

    // ������⵽��Բ���������
    //for (size_t i = 0; i < circles.size(); i++) {
    //    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    //    int radius = cvRound(circles[i][2]);
    //    // ��Բ��
    //    circle(src, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
    //    // ��Բ��
    //    circle(src, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
    //    std::cout << "Circle " << i + 1 << ": Center = (" << center.x << ", " << center.y << "), Radius = " << radius << std::endl;
    //}

    // ��ʾ���
    /*cv::namedWindow("Detected Circles", cv::WINDOW_AUTOSIZE);
    cv::imshow("Detected Circles", src);
    cv::waitKey(0)*/;
	return circles;
}
