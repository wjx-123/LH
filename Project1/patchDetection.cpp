#include "patchDetection.h"

cv::Mat pathDetection::rotateImg(const cv::Mat& src, double angle)
{
    // ��ȡͼ��ߴ����ת����
    cv::Point2f center(src.cols / 2.0, src.rows / 2.0);
    // ������ת����
    cv::Mat rot = getRotationMatrix2D(center, angle, 1.0);
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
    // ������ת�����Կ���ƽ��
    rot.at<double>(0, 2) += bbox.width / 2.0 - src.cols / 2.0;
    rot.at<double>(1, 2) += bbox.height / 2.0 - src.rows / 2.0;
    // ������ת
    cv::Mat dst;
    warpAffine(src, dst, rot, bbox.size());
    return dst;
}

bool pathDetection::variousAngleExist(const cv::Mat& img, const cv::Mat& templ, double angle, double threshold)
{
    bool isMatchFound_zero = ifBlackBlockExist(img, templ);
    bool isMatchFound_ninety = ifBlackBlockExist(img, rotateImg(templ, 90));
    return isMatchFound_zero || isMatchFound_ninety;
}

double pathDetection::detectOffsetAngle(cv::Mat img)
{
    // �����µĳߴ�
    double scale = 80.0 / std::min(img.cols, img.rows);
    int newWidth = static_cast<int>(img.cols * scale);
    int newHeight = static_cast<int>(img.rows * scale);

    // ����ͼƬ
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(newWidth, newHeight));

    double angle = 0;
    // ת��Ϊ�Ҷ�ͼ
    cv::Mat gray;
    cvtColor(resizedImg, gray, cv::COLOR_BGR2GRAY);

    // ʶ���ɫ����
    cv::Mat blackAreas;
    cv::threshold(gray, blackAreas, 50, 255, cv::THRESH_BINARY_INV); // ������ֵ��ʶ���ɫ����

    // ��Ե���
    cv::Mat edges;
    Canny(blackAreas, edges, 50, 150);

    // Ӧ�û���任��ֱ��
    std::vector<cv::Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 40, 50, 10);

    // ��ȡͼƬ�����ĵ�
    cv::Point center(resizedImg.cols / 2, resizedImg.rows / 2);

    double targetLength = 55.0;
    double minDifference = 1000;
    cv::Vec4i closestLine;

    // ����Ƕ�
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4i l = lines[i];
        double angle1 = std::atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;

        //ɸѡ����
       /* double length = std::sqrt(pow(l[0] - l[2], 2) + pow(l[1] - l[3], 2));
        double difference = std::abs(length - targetLength);
        if (difference < minDifference) {
            minDifference = difference;
            closestLine = l;
        }*/

        // �����߶ε��е�  ɸѡ�����ľ���
        cv::Point midPoint((l[0] + l[2]) / 2, (l[1] + l[3]) / 2);

        // �����е㵽���ĵ�ľ���
        double distance = cv::norm(midPoint - center);

        // ������ӽ����ĵ��߶�
        if (distance < minDifference) {
            minDifference = distance;
            closestLine = l;
        }

        //std::cout << "Line " << i << ": Angle = " << angle1 << std::endl;
        //cv::line(resizedImg, cv::Point(lines[i][0],lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0), 2);
    }
    angle = std::atan2(closestLine[3] - closestLine[1], closestLine[2] - closestLine[0]) * 180.0 / CV_PI;
    //cv::line(resizedImg, cv::Point(closestLine[0], closestLine[1]), cv::Point(closestLine[2], closestLine[3]), cv::Scalar(0, 255, 0), 2);
    return std::abs(angle);
}

double pathDetection::ifDetectionWrong(cv::Mat& img, cv::Mat& temp, double threshold)
{
    cv::Mat hsv_img1, hsv_img2;
    cvtColor(img, hsv_img1, cv::COLOR_BGR2HSV);
    cvtColor(temp, hsv_img2, cv::COLOR_BGR2HSV);

    // ����͹�һ��ֱ��ͼ
    int h_bins = 50, s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };

    cv::MatND hist_img1, hist_img2;
    calcHist(&hsv_img1, 1, channels, cv::Mat(), hist_img1, 2, histSize, ranges, true, false);
    normalize(hist_img1, hist_img1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    calcHist(&hsv_img2, 1, channels, cv::Mat(), hist_img2, 2, histSize, ranges, true, false);
    normalize(hist_img2, hist_img2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    // �Ƚ�ֱ��ͼ
    double res = compareHist(hist_img1, hist_img2, cv::HISTCMP_CORREL);
    return res >= threshold ? true : false;
}


cv::Rect2f pathDetection::getBlackBlock(cv::Mat image)
{
    std::vector<cv::Point2f> res;
    // ��Ĥ
    cv::Mat mask;
    // �洢��ɫ�㼯
    std::vector<cv::Point> whitePoints;

    //��ͼƬתΪhsv�ĸ�ʽ
    cv::cvtColor(image,image, cv::COLOR_BGR2HSV);

    // ɸѡ����ɫ��Χ
    cv::Scalar lowerRange(0, 0, 0);    // ��ͷ�Χ��hsv��
    cv::Scalar upperRange(180, 255, 255);    // ��߷�Χ��hsv��

    // ����inRange������������ɫ��Χɸѡ���ص�
    cv::inRange(image, lowerRange, upperRange, mask);

    // ����ͼ���ÿ������
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            // ��ȡ����ֵ
            uchar pixel = mask.at<uchar>(i, j);

            // �ж����ص�ֵ�Ƿ�Ϊ��ɫ��255��
            if (pixel == 255) {
                // ����ɫ���ص����������ɫ���ص㼯
                whitePoints.push_back(cv::Point(j, i));
            }
        }
    }
    cv::Rect2f boundingRect;
    if (whitePoints.empty())
    {
        boundingRect = { 0.25f * image.cols, 0.25f * image.rows, 0.5f * image.cols, 0.5f * image.rows };
        return boundingRect;
    }
    // ��ת������Ϻ��� minAreaRect ������С����������
    cv::RotatedRect rotatedBoundingRect = cv::minAreaRect(whitePoints);

    // ת��Ϊ�����Σ�cv::Rect2f��
    boundingRect = rotatedBoundingRect.boundingRect();
    //// ��ȡ������ε��ĸ�����
    //cv::Point2f rectPoints[4];
    //boundingRect.points(rectPoints);
    auto distanceCenter = std::sqrt(std::pow(0.5 * image.cols - (boundingRect.x + boundingRect.width * 0.5) ,2) +
        std::pow(0.5 * image.rows - (boundingRect.y + boundingRect.height * 0.5), 2));
    if (boundingRect.width > 0.6 * image.cols || boundingRect.height > 0.6 * image.rows || distanceCenter > 40 || boundingRect.width < 30 || boundingRect.height < 30)
    {
        return { 0.25f * image.cols, 0.25f * image.rows, 0.5f * image.cols, 0.5f * image.rows };
    }
	return boundingRect;
}

bool pathDetection::ifBlackBlockExist(const cv::Mat& img,const cv::Mat& templ,double threshold)
{
    // �������ڴ洢����ľ���
    cv::Mat result;
    matchTemplate(img, templ, result, cv::TM_CCOEFF_NORMED);

    // �ҵ����ƥ��λ��
    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    //cv::Point matchLoc;
    //matchLoc = maxLoc;
    //rectangle(img, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);
    //cv::imshow("Matched Result", img);
    //cv::waitKey(0);

    // �ж��Ƿ��ҵ�ƥ����
    return maxVal >= threshold;
}
