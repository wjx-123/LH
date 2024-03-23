#include "colorMethod.h"

colorMethod::colorMethod()
{
}

colorMethod::~colorMethod()
{
}

void colorMethod::test(cv::Mat& img)
{
    // 1. 读取图片
    /*cv::Mat img = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\6.jpg");
    if (img.empty()) {
        std::cout << "Could not read the image." << std::endl;
    }*/

    // 2. 将图片从BGR转换到HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // 3. 定义HSV颜色范围（例如，筛选绿色区域）
    cv::Scalar lower_green(0, 100, 0);
    cv::Scalar upper_green(200, 255, 255);

    // 4. 创建掩码
    cv::Mat mask;
    cv::inRange(hsv, lower_green, upper_green, mask);

    // 5. 应用掩码
    cv::Mat result;
    cv::bitwise_and(img, img, result, mask);

    // 计算筛选出的像素占比
    double totalPixels = img.rows * img.cols;

    // 在计算掩码的非零像素之前，启用OpenMP并行区域
    int count = 0;
#pragma omp parallel for reduction(+:count)  // 使用reduction来累加每个线程的计数
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            if (mask.at<uchar>(i, j) > 0) {
                count++;
            }
        }
    }
    double ratio = static_cast<double>(count) / totalPixels;

    // 显示结果
    std::cout << "Filtered Pixels Ratio with OpenMP: " << ratio << std::endl;


   // double filteredPixels = cv::countNonZero(mask);
    //double ratio = filteredPixels / totalPixels;

    // 显示结果
    //std::cout << "Filtered Pixels Ratio: " << ratio << std::endl;
    //cv::imshow("Original Image", img);
    //cv::imshow("Mask", mask);
    //cv::imshow("Result", result);
    //cv::waitKey(0);


    // 上传图像到GPU
    //cv::cuda::GpuMat d_img, d_hsv, d_mask;
    //d_img.upload(img);

    //// 在GPU上转换颜色空间
    //cv::cuda::cvtColor(d_img, d_hsv, cv::COLOR_BGR2HSV);

    //// 在GPU上应用inRange
    //cv::Scalar lower_green(50, 100, 100);
    //cv::Scalar upper_green(70, 255, 255);
    //cv::cuda::inRange(d_hsv, lower_green, upper_green, d_mask);

    //// 将掩码下载到CPU内存以进行像素计数（当前OpenCV的CUDA模块不支持在GPU上直接计数）
    //cv::Mat mask;
    //d_mask.download(mask);

    //// 计算非零像素的数量
    //int count = cv::countNonZero(mask);
    //double ratio = static_cast<double>(count) / (img.rows * img.cols);

    //std::cout << "Filtered Pixels Ratio: " << ratio << std::endl;
}

void colorMethod::test1(cv::Mat& img)
{
    // 2. 将图片从BGR转换到HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // 3. 定义HSV颜色范围（例如，筛选绿色区域）
    cv::Scalar lower_green(0, 100, 0);
    cv::Scalar upper_green(200, 255, 255);

    // 4. 创建掩码
    cv::Mat mask;
    cv::inRange(hsv, lower_green, upper_green, mask);

    // 5. 应用掩码
    cv::Mat result;
    cv::bitwise_and(img, img, result, mask);

    // 计算筛选出的像素占比
    double totalPixels = img.rows * img.cols;
    double filteredPixels = cv::countNonZero(mask);
    double ratio = filteredPixels / totalPixels;

    // 显示结果
    std::cout << "Filtered Pixels Ratio: " << ratio << std::endl;
    //cv::imshow("Original Image", img);
    //cv::imshow("Mask", mask);
    //cv::imshow("Result", result);
    //cv::waitKey(0);
}

void colorMethod::test2()
{
    // 读取图片
    cv::Mat img = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\2499.jpg");

    // 转换到HSV颜色空间
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // 定义红色的HSV颜色范围
    cv::Scalar lower_red1(0, 120, 70);
    cv::Scalar upper_red1(10, 255, 255);
    cv::Scalar lower_red2(170, 120, 70);
    cv::Scalar upper_red2(180, 255, 255);

    // 创建掩码
    cv::Mat mask1, mask2;
    cv::inRange(hsv, lower_red1, upper_red1, mask1);
    cv::inRange(hsv, lower_red2, upper_red2, mask2);
    cv::Mat mask = mask1 | mask2;

    // 查找掩码中的轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 过滤并绘制轮廓
    for (size_t i = 0; i < contours.size(); i++) {
        // 可能需要进一步的过滤轮廓
        cv::Rect boundingBox = cv::boundingRect(contours[i]);
        cv::rectangle(img, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 255, 0), 2);

        // 打印坐标
        std::cout << "Red box coordinates: " << boundingBox << std::endl;
    }

    // 显示结果
    cv::imshow("Result", img);
    cv::waitKey(0);
}
