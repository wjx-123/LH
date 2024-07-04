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

cv::Point circular::rectToCircle(const cv::Point& rectPoint, const cv::Point& center, int innerRadius, int outerRadius)
{
    // �������ܳ�
    int ringCircumference = 2 * CV_PI * outerRadius;

    // ����Ƕ� theta
    double theta = 2 * CV_PI * rectPoint.x / ringCircumference;

    // ����뾶 r
    int r = innerRadius + rectPoint.y;

    // ��������ת��Ϊ�ѿ�������
    int srcX = center.x + r * cos(theta);
    int srcY = center.y + r * sin(theta);

    return cv::Point(srcX, srcY);
}

std::vector<std::vector<cv::Point>> circular::detectDefects(const cv::Mat& img)
{
    /*��һ������ͨͼƬ*/
    //// ת��Ϊ�Ҷ�ͼ
    //cv::Mat gray;
    //cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    //// Ӧ�ø�˹ģ��ȥ��
    //cv::Mat blurred;
    //cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    //// ʹ������Ӧ��ֵ������ȡ��ɫ����
    //cv::Mat binary;
    //cv::adaptiveThreshold(blurred, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);

    //// Ӧ����̬ѧ����ȥ������
    //cv::Mat morph;
    //cv::morphologyEx(binary, morph, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4)));

    //// �������
    //std::vector<std::vector<cv::Point>> contours;
    //cv::findContours(morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //// ɸѡ������ڵ���50������
    //std::vector<std::vector<cv::Point>> filteredContours;
    //for (const auto& contour : contours) {
    //    double area = cv::contourArea(contour);
    //    if (area >= 50) {
    //        filteredContours.push_back(contour);
    //    }
    //}

    //return filteredContours;


    //��һ���û�����������
    std::vector<cv::Rect> defectRects;

    // ת��Ϊ�Ҷ�ͼ
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // ʹ����ֵ������ȡ��ɫ����
    cv::Mat binary;
    cv::threshold(gray, binary, 200, 255, cv::THRESH_BINARY); // ��ɫ������ֵ

    // ����ɫ��������
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<int> heightOfWhite; // ��¼ÿ����ɫ�����height

    // ����ÿ����ɫ����
    for (const auto& contour : contours) {
        cv::Rect boundingBox = cv::boundingRect(contour);
        if (boundingBox.height > 20 && boundingBox.height < 500) {
            heightOfWhite.push_back(boundingBox.height);
        }
    }

    double h_median = findMedian(heightOfWhite);

    // ����ÿ����ɫ���򲢼��ȱ��
    for (const auto& contour : contours) {
        cv::Rect boundingBox = cv::boundingRect(contour);
        if (boundingBox.height > 20 && boundingBox.height < 500) {
            if (boundingBox.area() > 300 && boundingBox.height < h_median - 10) { // �����ɫ����߶�С����λ��һ����Χ
                cv::Rect defectRect(boundingBox.x, boundingBox.y + boundingBox.height, boundingBox.width, h_median - boundingBox.height);
                defectRects.push_back(defectRect);
            }
            else {
                cv::Mat roi = img(boundingBox);

                // ת��Ϊ�Ҷ�ͼ
                cv::Mat roiGray;
                cv::cvtColor(roi, roiGray, cv::COLOR_BGR2GRAY);

                // ʹ������Ӧ��ֵ������ȡ��ɫ����
                cv::Mat roiBinary;
                cv::adaptiveThreshold(roiGray, roiBinary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);

                // Ӧ����̬ѧ����ȥ������
                cv::Mat morph;
                cv::morphologyEx(roiBinary, morph, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

                // ����ɫ��������
                std::vector<std::vector<cv::Point>> blackContours;
                cv::findContours(morph, blackContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                // ɸѡ������ڵ���50�ҳ���Ȳ�����һ������������������������
                for (const auto& blackContour : blackContours) {
                    double area = cv::contourArea(blackContour);
                    if (area >= 50) {
                        cv::Rect blackBoundingBox = cv::boundingRect(blackContour);
                        double aspectRatio = static_cast<double>(blackBoundingBox.width) / blackBoundingBox.height;

                        // ɸѡ�����С��һ������������
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

    // ת�ɵ�
    return convertRectsToPoints(defectRects);
}

std::vector<std::vector<cv::Point>> circular::convertRectsToPoints(const std::vector<cv::Rect>& defectRects)
{
    std::vector<std::vector<cv::Point>> rectPoints;

    for (const auto& rect : defectRects) {
        std::vector<cv::Point> points;
        points.emplace_back(rect.x, rect.y); // ���Ͻ�
        points.emplace_back(rect.x + rect.width, rect.y); // ���Ͻ�
        points.emplace_back(rect.x + rect.width, rect.y + rect.height); // ���½�
        points.emplace_back(rect.x, rect.y + rect.height); // ���½�
        rectPoints.push_back(points);
    }

    return rectPoints;
}

double circular::findMedian(std::vector<int> nums)
{
    // ����
    std::sort(nums.begin(), nums.end());

    // ������λ��
    size_t size = nums.size();
    if (size % 2 == 0) {
        // ż����Ԫ�أ���λ�����м�����Ԫ�ص�ƽ��ֵ
        return (nums[size / 2 - 1] + nums[size / 2]) / 2.0;
    }
    else {
        // ������Ԫ�أ���λ�����м���Ǹ�Ԫ��
        return nums[size / 2];
    }
}
