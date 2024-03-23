#include "colorMethod.h"

colorMethod::colorMethod()
{
}

colorMethod::~colorMethod()
{
}

void colorMethod::test(cv::Mat& img)
{
    // 1. ��ȡͼƬ
    /*cv::Mat img = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\6.jpg");
    if (img.empty()) {
        std::cout << "Could not read the image." << std::endl;
    }*/

    // 2. ��ͼƬ��BGRת����HSV��ɫ�ռ�
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // 3. ����HSV��ɫ��Χ�����磬ɸѡ��ɫ����
    cv::Scalar lower_green(0, 100, 0);
    cv::Scalar upper_green(200, 255, 255);

    // 4. ��������
    cv::Mat mask;
    cv::inRange(hsv, lower_green, upper_green, mask);

    // 5. Ӧ������
    cv::Mat result;
    cv::bitwise_and(img, img, result, mask);

    // ����ɸѡ��������ռ��
    double totalPixels = img.rows * img.cols;

    // �ڼ�������ķ�������֮ǰ������OpenMP��������
    int count = 0;
#pragma omp parallel for reduction(+:count)  // ʹ��reduction���ۼ�ÿ���̵߳ļ���
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            if (mask.at<uchar>(i, j) > 0) {
                count++;
            }
        }
    }
    double ratio = static_cast<double>(count) / totalPixels;

    // ��ʾ���
    std::cout << "Filtered Pixels Ratio with OpenMP: " << ratio << std::endl;


   // double filteredPixels = cv::countNonZero(mask);
    //double ratio = filteredPixels / totalPixels;

    // ��ʾ���
    //std::cout << "Filtered Pixels Ratio: " << ratio << std::endl;
    //cv::imshow("Original Image", img);
    //cv::imshow("Mask", mask);
    //cv::imshow("Result", result);
    //cv::waitKey(0);


    // �ϴ�ͼ��GPU
    //cv::cuda::GpuMat d_img, d_hsv, d_mask;
    //d_img.upload(img);

    //// ��GPU��ת����ɫ�ռ�
    //cv::cuda::cvtColor(d_img, d_hsv, cv::COLOR_BGR2HSV);

    //// ��GPU��Ӧ��inRange
    //cv::Scalar lower_green(50, 100, 100);
    //cv::Scalar upper_green(70, 255, 255);
    //cv::cuda::inRange(d_hsv, lower_green, upper_green, d_mask);

    //// ���������ص�CPU�ڴ��Խ������ؼ�������ǰOpenCV��CUDAģ�鲻֧����GPU��ֱ�Ӽ�����
    //cv::Mat mask;
    //d_mask.download(mask);

    //// ����������ص�����
    //int count = cv::countNonZero(mask);
    //double ratio = static_cast<double>(count) / (img.rows * img.cols);

    //std::cout << "Filtered Pixels Ratio: " << ratio << std::endl;
}

void colorMethod::test1(cv::Mat& img)
{
    // 2. ��ͼƬ��BGRת����HSV��ɫ�ռ�
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // 3. ����HSV��ɫ��Χ�����磬ɸѡ��ɫ����
    cv::Scalar lower_green(0, 100, 0);
    cv::Scalar upper_green(200, 255, 255);

    // 4. ��������
    cv::Mat mask;
    cv::inRange(hsv, lower_green, upper_green, mask);

    // 5. Ӧ������
    cv::Mat result;
    cv::bitwise_and(img, img, result, mask);

    // ����ɸѡ��������ռ��
    double totalPixels = img.rows * img.cols;
    double filteredPixels = cv::countNonZero(mask);
    double ratio = filteredPixels / totalPixels;

    // ��ʾ���
    std::cout << "Filtered Pixels Ratio: " << ratio << std::endl;
    //cv::imshow("Original Image", img);
    //cv::imshow("Mask", mask);
    //cv::imshow("Result", result);
    //cv::waitKey(0);
}

void colorMethod::test2()
{
    // ��ȡͼƬ
    cv::Mat img = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\2499.jpg");

    // ת����HSV��ɫ�ռ�
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // �����ɫ��HSV��ɫ��Χ
    cv::Scalar lower_red1(0, 120, 70);
    cv::Scalar upper_red1(10, 255, 255);
    cv::Scalar lower_red2(170, 120, 70);
    cv::Scalar upper_red2(180, 255, 255);

    // ��������
    cv::Mat mask1, mask2;
    cv::inRange(hsv, lower_red1, upper_red1, mask1);
    cv::inRange(hsv, lower_red2, upper_red2, mask2);
    cv::Mat mask = mask1 | mask2;

    // ���������е�����
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // ���˲���������
    for (size_t i = 0; i < contours.size(); i++) {
        // ������Ҫ��һ���Ĺ�������
        cv::Rect boundingBox = cv::boundingRect(contours[i]);
        cv::rectangle(img, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 255, 0), 2);

        // ��ӡ����
        std::cout << "Red box coordinates: " << boundingBox << std::endl;
    }

    // ��ʾ���
    cv::imshow("Result", img);
    cv::waitKey(0);
}
