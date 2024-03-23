#include "patchDetection.h"

cv::Mat pathDetection::rotateImg(const cv::Mat& src, double angle)
{
    // 获取图像尺寸和旋转中心
    cv::Point2f center(src.cols / 2.0, src.rows / 2.0);
    // 计算旋转矩阵
    cv::Mat rot = getRotationMatrix2D(center, angle, 1.0);
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
    // 调整旋转矩阵以考虑平移
    rot.at<double>(0, 2) += bbox.width / 2.0 - src.cols / 2.0;
    rot.at<double>(1, 2) += bbox.height / 2.0 - src.rows / 2.0;
    // 进行旋转
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
    // 计算新的尺寸
    double scale = 80.0 / std::min(img.cols, img.rows);
    int newWidth = static_cast<int>(img.cols * scale);
    int newHeight = static_cast<int>(img.rows * scale);

    // 缩放图片
    cv::Mat resizedImg;
    cv::resize(img, resizedImg, cv::Size(newWidth, newHeight));

    double angle = 0;
    // 转换为灰度图
    cv::Mat gray;
    cvtColor(resizedImg, gray, cv::COLOR_BGR2GRAY);

    // 识别黑色区域
    cv::Mat blackAreas;
    cv::threshold(gray, blackAreas, 50, 255, cv::THRESH_BINARY_INV); // 调整阈值来识别黑色区域

    // 边缘检测
    cv::Mat edges;
    Canny(blackAreas, edges, 50, 150);

    // 应用霍夫变换找直线
    std::vector<cv::Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 40, 50, 10);

    // 获取图片的中心点
    cv::Point center(resizedImg.cols / 2, resizedImg.rows / 2);

    double targetLength = 55.0;
    double minDifference = 1000;
    cv::Vec4i closestLine;

    // 计算角度
    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4i l = lines[i];
        double angle1 = std::atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;

        //筛选长度
       /* double length = std::sqrt(pow(l[0] - l[2], 2) + pow(l[1] - l[3], 2));
        double difference = std::abs(length - targetLength);
        if (difference < minDifference) {
            minDifference = difference;
            closestLine = l;
        }*/

        // 计算线段的中点  筛选离中心距离
        cv::Point midPoint((l[0] + l[2]) / 2, (l[1] + l[3]) / 2);

        // 计算中点到中心点的距离
        double distance = cv::norm(midPoint - center);

        // 更新最接近中心的线段
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

    // 计算和归一化直方图
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

    // 比较直方图
    double res = compareHist(hist_img1, hist_img2, cv::HISTCMP_CORREL);
    return res >= threshold ? true : false;
}


cv::Rect2f pathDetection::getBlackBlock(cv::Mat image)
{
    std::vector<cv::Point2f> res;
    // 掩膜
    cv::Mat mask;
    // 存储白色点集
    std::vector<cv::Point> whitePoints;

    //把图片转为hsv的格式
    cv::cvtColor(image,image, cv::COLOR_BGR2HSV);

    // 筛选的颜色范围
    cv::Scalar lowerRange(0, 0, 0);    // 最低范围（hsv）
    cv::Scalar upperRange(180, 255, 255);    // 最高范围（hsv）

    // 调用inRange函数，基于颜色范围筛选像素点
    cv::inRange(image, lowerRange, upperRange, mask);

    // 遍历图像的每个像素
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            // 获取像素值
            uchar pixel = mask.at<uchar>(i, j);

            // 判断像素的值是否为白色（255）
            if (pixel == 255) {
                // 将白色像素点的坐标加入白色像素点集
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
    // 旋转矩形拟合函数 minAreaRect 进行最小外包矩形拟合
    cv::RotatedRect rotatedBoundingRect = cv::minAreaRect(whitePoints);

    // 转换为正矩形（cv::Rect2f）
    boundingRect = rotatedBoundingRect.boundingRect();
    //// 获取外包矩形的四个顶点
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
    // 创建用于存储结果的矩阵
    cv::Mat result;
    matchTemplate(img, templ, result, cv::TM_CCOEFF_NORMED);

    // 找到最佳匹配位置
    double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    //cv::Point matchLoc;
    //matchLoc = maxLoc;
    //rectangle(img, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);
    //cv::imshow("Matched Result", img);
    //cv::waitKey(0);

    // 判断是否找到匹配项
    return maxVal >= threshold;
}
