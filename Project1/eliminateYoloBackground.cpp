#include "eliminateYoloBackground.h"

eliminateYoloBackground::eliminateYoloBackground()
{
}

eliminateYoloBackground::~eliminateYoloBackground()
{
}

std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> eliminateYoloBackground::getBoundAndPin(cv::Mat& image, std::string types)
{
    std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> result;
    cv::Mat heibai = test(image,types);
    if (types == "T")//����
    {
        std::vector<cv::Rect2f> tPin;
        cv::Rect2f rectHeibai = { static_cast<float>(image.cols * 0.3), static_cast<float>(image.rows * 0.35), static_cast<float>(image.cols * 0.35), static_cast<float>(image.rows * 0.25) };
        cv::rectangle(heibai, rectHeibai, cv::Scalar(0, 0, 0), cv::FILLED);
        auto [topLeft, bottomRight] = findBoundingRectangle_heibai(heibai, 0.1);
        cv::Rect2f black_rect = cv::Rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
        //cv::rectangle(image, black_rect, cv::Scalar(0, 255, 100), 2);
        adjustRect(black_rect, image.size());
        //cv::rectangle(image, black_rect, cv::Scalar(255, 0, 0), 2);
        cv::Mat topImg = heibai({ 0,0,static_cast<int>(image.cols),static_cast<int>(black_rect.y) });
        cv::Mat bottomImg = heibai({ 0,static_cast<int>(black_rect.y + black_rect.height),static_cast<int>(image.cols),static_cast<int>(image.rows - (black_rect.y + black_rect.height)) });
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(topImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        // ����������������С��Ӿ���
        cv::Rect boundingBox;
        for (size_t i = 0; i < contours.size(); i++) {
            boundingBox |= cv::boundingRect(contours[i]);
        }

        //cv::rectangle(topImg, boundingBox, cv::Scalar(255), 2);
        //cv::rectangle(image, boundingBox, cv::Scalar(0, 255, 0), 4);

        contours.clear();

        cv::findContours(bottomImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Rect boundingBox1;
        for (size_t i = 0; i < contours.size(); i++) {
            boundingBox1 |= cv::boundingRect(contours[i]);
        }
        //cv::rectangle(bottomImg, boundingBox1, cv::Scalar(255), 2);
        //cv::rectangle(image, { boundingBox1.x, static_cast<int>(boundingBox1.y + (black_rect.y + black_rect.height)) ,boundingBox1 .width,boundingBox1.height}, cv::Scalar(0, 255, 0), 4);
        moveToIntersect(boundingBox, black_rect);
        tPin.push_back(boundingBox);
        cv::Rect boundingBox1Adjusted(static_cast<float>(boundingBox1.x),
            static_cast<float>(boundingBox1.y + black_rect.y + black_rect.height),
            static_cast<float>(boundingBox1.width),
            static_cast<float>(boundingBox1.height));
        moveToIntersect(boundingBox1Adjusted, black_rect);
        tPin.push_back(boundingBox1Adjusted);
        //tPin.push_back({ static_cast<float>(boundingBox1.x), static_cast<float>(boundingBox1.y + (black_rect.y + black_rect.height)) ,static_cast<float>(boundingBox1.width),static_cast<float>(boundingBox1.height) });
        std::pair pair  = { black_rect ,tPin };
        result.push_back(pair);
        return result;
    }
    else if (types == "R")//����
    {
        auto [topLeft, bottomRight] = findBoundingRectangle_heibai(heibai, 0.2);
        cv::Mat hsvImage = useHsvTest(image);//��hsv������ͼ��һ�½���
        cv::Rect2f black_rect = cv::Rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
        auto pinVector = findPinsAroundBlackBox_ofThree(heibai, black_rect, hsvImage);
        std::pair pair = { black_rect, pinVector };
        result.push_back(pair);
    }
    else
    {
        auto [topLeft, bottomRight] = findBoundingRectangle_heibai(heibai, 0.2);
        cv::Mat hsvImage = useHsvTest(image);//��hsv������ͼ��һ�½���
        cv::Rect2f black_rect = cv::Rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
        auto pinVector = findPinsAroundBlackBox(heibai, black_rect, hsvImage);
        std::pair pair = { black_rect, pinVector };
        result.push_back(pair);
    }
    
    
    return result;
}

std::pair<cv::Point, cv::Point> eliminateYoloBackground::findBoundingRectangle(const cv::Mat& img, int colorRange)
{
    // �õ����ĵ�
    cv::Point center(img.cols / 2, img.rows / 2);
    cv::Vec3b centerColor = img.at<cv::Vec3b>(center);

    // ����߽��������
    cv::Point topLeft(img.cols, img.rows);
    cv::Point bottomRight(0, 0);

    // ��¼����
    std::vector<std::vector<bool>> visited(img.rows, std::vector<bool>(img.cols, false));

    // Lambda �������(�����м���������)
    auto isColorSimilar = [centerColor, colorRange](const cv::Vec3b& color) {
        for (int i = 0; i < 3; ++i) {
            if (std::abs(color[i] - centerColor[i]) > colorRange) return false;
        }
        return true;
        };

    // bfs
    std::queue<cv::Point> q;
    q.push(center);
    visited[center.y][center.x] = true;

    while (!q.empty()) {
        cv::Point p = q.front();
        q.pop();

        // ���±߽��
        topLeft.x = std::min(topLeft.x, p.x);
        topLeft.y = std::min(topLeft.y, p.y);
        bottomRight.x = std::max(bottomRight.x, p.x);
        bottomRight.y = std::max(bottomRight.y, p.y);

        // ����ĸ���������
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue; // �������ĵ㱾��

                int newX = p.x + dx;
                int newY = p.y + dy;

                // ���߽��Լ������Ƿ�������δ������
                if (newX >= 0 && newX < img.cols && newY >= 0 && newY < img.rows &&
                    !visited[newY][newX] && isColorSimilar(img.at<cv::Vec3b>(newY, newX))) {
                    q.push(cv::Point(newX, newY));
                    visited[newY][newX] = true;
                }
            }
        }
    }
    return { topLeft, bottomRight };
}

//std::pair<cv::Point, cv::Point> eliminateYoloBackground::findBoundingRectangle_heibai(const cv::Mat& img)
//{
//    // �õ����ĵ�
//    cv::Point center(img.cols / 2, img.rows / 2);
//    uchar centerColor = img.at<uchar>(center);
//
//    // ����߽��������
//    cv::Point topLeft(img.cols, img.rows);
//    cv::Point bottomRight(0, 0);
//
//    // ��¼����
//    std::vector<std::vector<bool>> visited(img.rows, std::vector<bool>(img.cols, false));
//
//    // bfs
//    std::queue<cv::Point> q;
//    q.push(center);
//    visited[center.y][center.x] = true;
//
//    while (!q.empty()) {
//        cv::Point p = q.front();
//        q.pop();
//
//        // ���±߽��
//        topLeft.x = std::min(topLeft.x, p.x);
//        topLeft.y = std::min(topLeft.y, p.y);
//        bottomRight.x = std::max(bottomRight.x, p.x);
//        bottomRight.y = std::max(bottomRight.y, p.y);
//
//        // ����ĸ���������
//        for (int dy = -1; dy <= 1; ++dy) {
//            for (int dx = -1; dx <= 1; ++dx) {
//                if (dx == 0 && dy == 0) continue; // �������ĵ㱾��
//
//                int newX = p.x + dx;
//                int newY = p.y + dy;
//
//                // ���߽��Լ������Ƿ��ѱ�����
//                if (newX >= 0 && newX < img.cols && newY >= 0 && newY < img.rows &&
//                    !visited[newY][newX] && img.at<uchar>(newY, newX) == centerColor) {
//                    q.push(cv::Point(newX, newY));
//                    visited[newY][newX] = true;
//                }
//            }
//        }
//    }
//    return { topLeft, bottomRight };
//}


std::pair<cv::Point, cv::Point> eliminateYoloBackground::findBoundingRectangle_heibai(const cv::Mat& img, float whiteRatioThreshold)
{
    int rows = img.rows;
    int cols = img.cols;
    cv::Point center(cols / 2, rows / 2);

    int top = center.y, bottom = center.y, left = center.x, right = center.x;
    bool stopTop = false, stopBottom = false, stopLeft = false, stopRight = false;

    // ��ɢ�����ĵ�������
    while (true) {
        bool changed = false; // �����жϴ˴�ѭ���Ƿ��б߽���ɢ

        // ������ɢ����
        if (!stopTop) {
            if (top > 0) {
                int whiteCount = countWhitePixels(img.row(top - 1).colRange(left, right + 1));
                float whiteRatio = static_cast<float>(whiteCount) / (right - left + 1);
                if (whiteRatio >= whiteRatioThreshold || top - 1 == 0) {
                    stopTop = true;
                }
                else {
                    top--;
                    changed = true;
                }
            }
            else {
                stopTop = true;
            }
        }

        // ������ɢ����
        if (!stopBottom) {
            if (bottom < rows - 1) {
                int whiteCount = countWhitePixels(img.row(bottom + 1).colRange(left, right + 1));
                float whiteRatio = static_cast<float>(whiteCount) / (right - left + 1);
                if (whiteRatio >= whiteRatioThreshold || bottom + 1 == rows - 1) {
                    stopBottom = true;
                }
                else {
                    bottom++;
                    changed = true;
                }
            }
            else {
                stopBottom = true;
            }
        }

        // ������ɢ����
        if (!stopLeft) {
            if (left > 0) {
                int whiteCount = countWhitePixels(img.col(left - 1).rowRange(top, bottom + 1));
                float whiteRatio = static_cast<float>(whiteCount) / (bottom - top + 1);
                if (whiteRatio >= whiteRatioThreshold || left - 1 == 0) {
                    stopLeft = true;
                }
                else {
                    left--;
                    changed = true;
                }
            }
            else {
                stopLeft = true;
            }
        }

        // ������ɢ����
        if (!stopRight) {
            if (right < cols - 1) {
                int whiteCount = countWhitePixels(img.col(right + 1).rowRange(top, bottom + 1));
                float whiteRatio = static_cast<float>(whiteCount) / (bottom - top + 1);
                if (whiteRatio >= whiteRatioThreshold or right + 1 == cols - 1) {
                    stopRight = true;
                }
                else {
                    right++;
                    changed = true;
                }
            }
            else {
                stopRight = true;
            }
        }

        // ���������Ա�ֹͣ��ɢ�����˳�ѭ��
        /*if ((stopTop && stopBottom) || (stopLeft && stopRight)) {
            break;
        }*/

        // ���û�б߽���ɢ����ʣ���������������һ��ֹͣ�����˳�ѭ��
        if (!changed) {
            break;
        }
    }

    cv::Point topLeft(left, top);
    cv::Point bottomRight(right, bottom);
    return { topLeft, bottomRight };
}

std::vector<cv::Rect2f> eliminateYoloBackground::findPinsAroundBlackBox(cv::Mat& img, cv::Rect2f& blackBox, cv::Mat& hsvImg)
{
    //�����м�ľ��ο��Ϊ�ĸ����֣��첽����
    std::vector<cv::Rect2f> res, finalRes;
    float width = blackBox.width;
    float height = blackBox.height;

    // ȷ�����ߺͶ̱�
    float longEdge = std::max(width, height);
    float shortEdge = std::min(width, height);

    // ��鳤�̱ߵĲ���Ƿ񳬹����ߵ�20%
    bool longEdgeIsWidth = width > height;
    bool differenceExceedsThreshold = (longEdge - shortEdge) > longEdge * 0.2;

    // �󶨳�Ա����
    auto boundFunc = std::bind(&eliminateYoloBackground::getPinRect, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);

    // ���ݳ��̱ߵĲ��ѡ�������ı�
    std::future<std::vector<cv::Rect>> futureTop, futureBottom, futureLeft, futureRight;
    if (differenceExceedsThreshold && longEdgeIsWidth) {
        // ��೬��20%�������ǿ��ȣ��������·���
        std::cout << "��೬��20%�������ǿ��ȣ��������·���" << std::endl;
        futureTop = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, 0, blackBox.width, blackBox.y), 'L', 0);
        futureBottom = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, blackBox.y + blackBox.height, blackBox.width, img.rows - (blackBox.y + blackBox.height)), 'L', 2);
    }
    else if (differenceExceedsThreshold && !longEdgeIsWidth) {
        // ��೬��20%�������Ǹ߶ȣ��������ҷ���
        std::cout << "��೬��20%�������Ǹ߶ȣ��������ҷ���" << std::endl;
        futureLeft = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(0, blackBox.y, blackBox.x, blackBox.height), 'T', 3);
        futureRight = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x + blackBox.width, blackBox.y, img.cols - (blackBox.x + blackBox.width), blackBox.height), 'T', 1);
    }
    else {
        // ��಻����20%���������з���
        std::cout << "��಻����20%���������з���" << std::endl;
        futureTop = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, 0, blackBox.width, blackBox.y), 'L', 0);
        futureBottom = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, blackBox.y + blackBox.height, blackBox.width, img.rows - (blackBox.y + blackBox.height)), 'L', 2);
        futureLeft = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(0, blackBox.y, blackBox.x, blackBox.height), 'T', 3);
        futureRight = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x + blackBox.width, blackBox.y, img.cols - (blackBox.x + blackBox.width), blackBox.height), 'T', 1);
    }

    // �ռ����
    if (futureTop.valid()) {
        auto pinsTop = futureTop.get();
        res.insert(res.end(), pinsTop.begin(), pinsTop.end());
    }
    if (futureBottom.valid()) {
        auto pinsBottom = futureBottom.get();
        res.insert(res.end(), pinsBottom.begin(), pinsBottom.end());
    }
    if (futureLeft.valid()) {
        auto pinsLeft = futureLeft.get();
        res.insert(res.end(), pinsLeft.begin(), pinsLeft.end());
    }
    if (futureRight.valid()) {
        auto pinsRight = futureRight.get();
        res.insert(res.end(), pinsRight.begin(), pinsRight.end());
    }

    //��hsvͼȡ����ɸѡ����
    for (auto temp : res)
    {
        cv::rectangle(img, temp, cv::Scalar(255), 2);
        if (containsWhitePixel(hsvImg, temp))
        {
            finalRes.push_back(temp);
        }
    }
    return finalRes;
}

std::vector<cv::Rect2f> eliminateYoloBackground::findPinsAroundBlackBox_ofThree(cv::Mat& img, cv::Rect2f& blackBox, cv::Mat& hsvImg)
{
    //�����м�ľ��ο��Ϊ�ĸ����֣��첽����
    std::vector<cv::Rect2f> res, finalRes;
    float width = blackBox.width;
    float height = blackBox.height;

    // ȷ�����ߺͶ̱�
    float longEdge = std::max(width, height);
    float shortEdge = std::min(width, height);

    // ��鳤�̱ߵĲ���Ƿ񳬹����ߵ�20%
    bool longEdgeIsWidth = width > height;
    bool differenceExceedsThreshold = (longEdge - shortEdge) > longEdge * 0.2;

    // �󶨳�Ա����
    auto boundFunc = std::bind(&eliminateYoloBackground::getPinRect_ofThree, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);

    // ���ݳ��̱ߵĲ��ѡ�������ı�
    std::future<std::vector<cv::Rect>> futureTop, futureBottom, futureLeft, futureRight;
    if (differenceExceedsThreshold && longEdgeIsWidth) {
        // ��೬��20%�������ǿ��ȣ��������·���
        std::cout << "��೬��20%�������ǿ��ȣ��������·���" << std::endl;
        futureTop = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, 0, blackBox.width, blackBox.y), 'L', 0);
        futureBottom = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, blackBox.y + blackBox.height, blackBox.width, img.rows - (blackBox.y + blackBox.height)), 'L', 2);
    }
    else if (differenceExceedsThreshold && !longEdgeIsWidth) {
        // ��೬��20%�������Ǹ߶ȣ��������ҷ���
        std::cout << "��೬��20%�������Ǹ߶ȣ��������ҷ���" << std::endl;
        futureLeft = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(0, blackBox.y, blackBox.x, blackBox.height), 'T', 3);
        futureRight = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x + blackBox.width, blackBox.y, img.cols - (blackBox.x + blackBox.width), blackBox.height), 'T', 1);
    }
    else {
        // ��಻����20%���������з���
        std::cout << "��಻����20%���������з���" << std::endl;
        futureTop = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, 0, blackBox.width, blackBox.y), 'L', 0);
        futureBottom = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, blackBox.y + blackBox.height, blackBox.width, img.rows - (blackBox.y + blackBox.height)), 'L', 2);
        futureLeft = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(0, blackBox.y, blackBox.x, blackBox.height), 'T', 3);
        futureRight = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x + blackBox.width, blackBox.y, img.cols - (blackBox.x + blackBox.width), blackBox.height), 'T', 1);
    }

    // �ռ����
    if (futureTop.valid()) {
        auto pinsTop = futureTop.get();
        res.insert(res.end(), pinsTop.begin(), pinsTop.end());
    }
    if (futureBottom.valid()) {
        auto pinsBottom = futureBottom.get();
        res.insert(res.end(), pinsBottom.begin(), pinsBottom.end());
    }
    if (futureLeft.valid()) {
        auto pinsLeft = futureLeft.get();
        res.insert(res.end(), pinsLeft.begin(), pinsLeft.end());
    }
    if (futureRight.valid()) {
        auto pinsRight = futureRight.get();
        res.insert(res.end(), pinsRight.begin(), pinsRight.end());
    }
    float blackBoxArea = blackBox.width * blackBox.height;
    //��hsvͼȡ����ɸѡ����
    for (auto temp : res)
    {
        float tempArea = temp.width * temp.height;

        // ������temp�Ƿ������ɫ���أ������������blackBox�����֮����1000����֮��
        if (containsWhitePixel(hsvImg, temp) &&
            tempArea <= (blackBoxArea * 0.4) && tempArea >= 1000) {
            finalRes.push_back(temp);
        }
    }
    return finalRes;
}

cv::Mat eliminateYoloBackground::test(cv::Mat& img, std::string types)
{
    if (types != "T")
    {
        // ת��Ϊ�Ҷ�ͼ
        cv::Mat gray, eroded;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        int temp = cv::threshold(gray, gray, 0, 255, cv::THRESH_TRIANGLE);

        // Ӧ����ֵ�������Ա��������
        cv::Mat thresh;
        cv::threshold(gray, thresh, temp, 255, cv::THRESH_BINARY);
        //cv::Rect rectMask =  { static_cast<int>(img.cols * 0.25), static_cast<int>(img.rows * 0.4), static_cast<int>(img.cols * 0.5), static_cast<int>(img.rows * 0.3)};
        //cv::rectangle(thresh, rectMask, cv::Scalar(0), cv::FILLED);

        int kernelSize = 5;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

        // Ӧ�ñ�����
        cv::Mat closed;
        cv::morphologyEx(thresh, closed, cv::MORPH_CLOSE, kernel);

        // Ӧ�ÿ�����
        cv::Mat opened;
        cv::morphologyEx(closed, opened, cv::MORPH_OPEN, kernel);

        //��ֵ�˲���������ȥ��
        cv::medianBlur(opened, opened, kernelSize);

        // 2. ��ͨ�������ȥ��С����
        cv::Mat labels, stats, centroids;
        //cv::Mat filteredImage = cv::Mat::zeros(opened.size(), CV_8UC1);
        int nLabels = cv::connectedComponentsWithStats(opened, labels, stats, centroids);

        // ���������ֵ������С���
        int areaThreshold = 600; // ������Ҫ��������ֵ

        // �������м�⵽�����
        for (int label = 1; label < nLabels; label++) {
            int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area < areaThreshold) {
                // ��filteredImage�ϻ������
                opened.setTo(0, labels == label);
            }
        }
        return thresh;
    }
    else
    {
        cv::Mat hsvImage;
        cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsvChannels(3);
        cv::split(hsvImage, hsvChannels);
        cv::Mat vimg = hsvChannels[2];
        cv::threshold(vimg, vimg, 0, 255, cv::THRESH_TRIANGLE);
        return vimg;
    }
}

cv::Mat eliminateYoloBackground::useHsvTest(cv::Mat& image)
{
    // ���Ƚ�ͼ���BGRת����HSV��ɫ�ռ�
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels(3);
    cv::split(hsvImage, hsvChannels);
    cv::Mat vimg = hsvChannels[2],himg = hsvChannels[0],simg = hsvChannels[1];
    int temp = cv::threshold(vimg, vimg, 0, 255, cv::THRESH_TRIANGLE);
    cv::threshold(himg, himg, 0, 255, cv::THRESH_TRIANGLE);
    cv::threshold(simg, simg, 0, 255, cv::THRESH_TRIANGLE);

    cv::Mat simg1 = hsvChannels[1];
    cv::threshold(simg, simg, 0, 255, cv::THRESH_TRIANGLE);
    //cv::threshold(hsvChannels[2], hsvChannels[2], temp, 255, cv::THRESH_BINARY);
    // �����ɫ�ͻ�ɫ��HSV��ɫ��Χ
    //cv::Scalar lowerRed1(0, 120, 70);
    //cv::Scalar upperRed1(10, 255, 255);
    //cv::Scalar lowerRed2(170, 120, 70);
    //cv::Scalar upperRed2(180, 255, 255);
    //cv::Scalar lowerYellow(25, 50, 50);
    //cv::Scalar upperYellow(60, 255, 255);
    //// ��ɫ��HSV��Χ
    //cv::Scalar lowerBlack(0, 0, 0);
    //cv::Scalar upperBlack(180, 255, 50); // ���������趨�ϵͣ����Ͷȿ��ԽϿ���
    //// ��ɫ��HSV��Χ
    //cv::Scalar lowerGray(0, 0, 50);
    //cv::Scalar upperGray(180, 50, 220); // ���Ͷ����޽ϵͣ����ȸ��Ǵӽϰ��������ķ�Χ
    //
    //// ɸѡ��ɫ��Χ����������
    //cv::Mat maskRed1, maskRed2, maskYellow, maskBlack, maskGray;
    //cv::inRange(hsvImage, lowerRed1, upperRed1, maskRed1);
    //cv::inRange(hsvImage, lowerRed2, upperRed2, maskRed2);
    //cv::inRange(hsvImage, lowerYellow, upperYellow, maskYellow);
    //cv::inRange(hsvImage, lowerBlack, upperBlack, maskBlack);
    //cv::inRange(hsvImage, lowerGray, upperGray, maskGray);

    //// �ϲ���ɫ������������룬Ȼ�����ɫ����ϲ�
    //cv::Mat maskRed = maskRed1 | maskRed2;
    //cv::Mat maskCombined = maskRed | maskYellow;
    //// ��һ������ɫ�ͻ�ɫ������뵽�ϲ����������
    ////maskCombined = maskCombined | maskBlack | maskGray;
    //int morph_size = 2;  // �ṹԪ�صĳߴ磬������Ҫ���е���
    //cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
    //    cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
    //    cv::Point(morph_size, morph_size));
    //cv::morphologyEx(maskCombined, maskCombined, cv::MORPH_OPEN, element);
    return himg;
}

std::vector<cv::Rect> eliminateYoloBackground::getPinRect(cv::Mat& imgInit, cv::Rect rectInit, char baseline, int closestEdge)
{
    cv::Mat img = imgInit(rectInit);   
    // ��������
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> rectangles;

    // ����������������ȡ����Ӿ���
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        
        // ��������
        if (std::max(rect.width,rect.height) < std::min(img.rows,img.cols) * 0.2) 
        {
            continue;
        }
        rect.x += rectInit.x;
        rect.y += rectInit.y;
        
        rectangles.push_back(rect);
    }

    // ���ݻ�׼�ߺϲ�����
    if (baseline == 'L') {
        mergeRectanglesByLeftEdge(rectangles, 5);
    }
    else if (baseline == 'T') {
        mergeRectanglesByTopEdge(rectangles, 5);
    }

    filterRectangles(rectangles, rectInit, closestEdge, 1.2);
    return rectangles;
}

std::vector<cv::Rect> eliminateYoloBackground::getPinRect_ofThree(cv::Mat& imgInit, cv::Rect rectInit, char baseline, int closestEdge)
{
    cv::Mat img = imgInit(rectInit);
    // ��������
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> rectangles;

    // ����������������ȡ����Ӿ���
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);

        // ��������
        if (std::max(rect.width, rect.height) < std::min(img.rows, img.cols) * 0.2)
        {
            continue;
        }
        rect.x += rectInit.x;
        rect.y += rectInit.y;

        rectangles.push_back(rect);
    }

    // ���ݻ�׼�ߺϲ�����
    if (baseline == 'L') {
        mergeRectanglesByLeftEdge(rectangles, 5);
    }
    else if (baseline == 'T') {
        mergeRectanglesByTopEdge(rectangles, 5);
    }

    return rectangles;
}

int eliminateYoloBackground::countWhitePixels(const cv::Mat& line)
{
    return cv::countNonZero(line == 255);
}

void eliminateYoloBackground::mergeRectanglesByLeftEdge(std::vector<cv::Rect>& rectangles, int threshold)
{
    // �ϲ��߼���������߽�ľ���
    std::vector<cv::Rect> merged;
    while (!rectangles.empty()) {
        // Always merge with the first rectangle
        cv::Rect& base = rectangles.front();
        auto it = rectangles.begin();
        for (auto jt = rectangles.begin() + 1; jt != rectangles.end();) {
            if (std::abs(jt->x - base.x) < threshold) {
                base = base | *jt; // �ϲ�����
                jt = rectangles.erase(jt);
            }
            else {
                ++jt;
            }
        }
        merged.push_back(base);
        rectangles.erase(it); 
    }
    rectangles.swap(merged); 
}

void eliminateYoloBackground::mergeRectanglesByTopEdge(std::vector<cv::Rect>& rectangles, int threshold)
{
    // �ϲ��߼��������ϱ߽�ľ���
    std::vector<cv::Rect> merged;
    while (!rectangles.empty()) {
        // Always merge with the first rectangle
        cv::Rect& base = rectangles.front();
        auto it = rectangles.begin();
        for (auto jt = rectangles.begin() + 1; jt != rectangles.end();) {
            if (std::abs(jt->y - base.y) < threshold) {
                base = base | *jt; // �ϲ�����
                jt = rectangles.erase(jt);
            }
            else {
                ++jt;
            }
        }
        merged.push_back(base);
        rectangles.erase(it); // Remove the merged rectangle
    }
    rectangles.swap(merged);
}

bool eliminateYoloBackground::containsWhitePixel(const cv::Mat& image, const cv::Rect& rect)
{
    // ��ȡROI
    cv::Mat roi = image(rect);
    // ���ROI���Ƿ��з�������
    return cv::countNonZero(roi) > 1;
}

void eliminateYoloBackground::filterRectangles(std::vector<cv::Rect>& rectangles, const cv::Rect& rectInit, int closestEdge, double thresholdRatio)
{
    std::vector<double> distances;

    // ����ÿ���������ĵ㵽ָ���ߵľ���
    for (const auto& rect : rectangles) {
        cv::Point rectCenter(rect.x + rect.width / 2, rect.y + rect.height / 2);
        double distance = 0.0;

        switch (closestEdge) {
        case 0: // �±�
            distance = std::abs(rectCenter.y - rectInit.br().y);
            break;
        case 1: // ���
            distance = std::abs(rectCenter.x - rectInit.x);
            break;
        case 2: // �ϱ�
            distance = std::abs(rectCenter.y - rectInit.y);
            break;
        case 3: // �ұ�
            distance = std::abs(rectCenter.x - rectInit.br().x);
            break;
        }
        distances.push_back(distance);
    }

    // ��������ƽ��ֵ
    double averageDistance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
    double distanceThreshold = averageDistance * thresholdRatio; // ʹ��ƽ��ֵ��һ��������Ϊ��ֵ

    // �޳����볬����ֵ�Ҳ���߽��ཻ�ľ���
    auto newEnd = std::remove_if(rectangles.begin(), rectangles.end(),
        [&](const cv::Rect& rect) -> bool {
            cv::Point rectCenter(rect.x + rect.width / 2, rect.y + rect.height / 2);
            double distance = 0.0;
            bool intersects = false;

            // ������벢�ж��Ƿ��ཻ
            switch (closestEdge) {
            case 0: // �±�
                distance = std::abs(rectCenter.y - rectInit.br().y);
                intersects = (rect.br().y > rectInit.br().y - 2);
                break;
            case 1: // ���
                distance = std::abs(rectCenter.x - rectInit.x);
                intersects = (rect.x < rectInit.x + 2);
                break;
            case 2: // �ϱ�
                distance = std::abs(rectCenter.y - rectInit.y);
                intersects = (rect.y < rectInit.y + 2);
                break;
            case 3: // �ұ�
                distance = std::abs(rectCenter.x - rectInit.br().x);
                intersects = (rect.br().x > rectInit.br().x - 2);
                break;
            }

            // ������εľ��볬����ֵ�Ҳ���߽��ཻ����Ӧ�޳�
            return distance > distanceThreshold || !intersects;
        });

    rectangles.erase(newEnd, rectangles.end());
}

void eliminateYoloBackground::adjustRect(cv::Rect2f& rect, const cv::Size& imageSize)
{
    //// ����"����"����ֵ
    const float threshold_short = 60.0;
    const float threshold_long = 10.0;
    //// ͼƬ���ĵ�����
    //const float imageCenterX = imageSize.width / 2.0;
    //const float imageCenterY = imageSize.height / 2.0;

    //// �������ĵ�����
    //const float rectCenterX = rect.x + rect.width / 2.0;
    //const float rectCenterY = rect.y + rect.height / 2.0;

    //// �����ο���ͼƬ��Ե�ľ����Ƿ�С����ֵ
    //bool isCloseToLeftEdge = rect.x < threshold;
    //bool isCloseToRightEdge = imageSize.width - (rect.x + rect.width) < threshold;

    //// �����߻��ұ�̫��������е���
    //if (isCloseToLeftEdge || isCloseToRightEdge) {
    //    // ͼƬ�̱ߵĳ���
    //    float shortSideLength = std::min(imageSize.width, imageSize.height);

    //    // ������ľ��ο�����ͼƬ�̱ߵ�����֮һ
    //    float newWidth = shortSideLength * 0.625;

    //    // ������߶�̫������������Ӧ��ͼƬ���Ķ���
    //    if (isCloseToLeftEdge && isCloseToRightEdge) {
    //        rect.x = imageCenterX - newWidth / 2.0;
    //    }
    //    else if (isCloseToLeftEdge) {
    //        // ���ֻ�����̫���������������ƶ�
    //        rect.x = imageCenterX - (imageSize.width - rect.x - rect.width) - newWidth;
    //    }
    //    else {
    //        // ���ֻ���ұ�̫���������������ƶ�
    //        rect.x = imageCenterX - rect.x - newWidth;
    //    }

    //    // ���¾��ο���
    //    rect.width = newWidth;
    //}


    // ͼƬ���ĵ�����
    float imageCenterX = imageSize.width / 2.0f;
    float imageCenterY = imageSize.height / 2.0f;

    // �����ε�ÿ�����Ƿ���ڽӽ�ͼƬ��Ե
    bool isCloseToLeftEdge = rect.x < threshold_long;
    bool isCloseToRightEdge = imageSize.width - (rect.x + rect.width) < threshold_long;
    bool isCloseToTopEdge = rect.y < threshold_short;
    bool isCloseToBottomEdge = imageSize.height - (rect.y + rect.height) < threshold_short;

    // �����߻��ұ߹��ڽӽ���Ե
    if (isCloseToLeftEdge || isCloseToRightEdge) {
        // ֻ��һ�߿���ʱ�������ñߵ���Ա߶ԳƵ�λ��
        if (!(isCloseToLeftEdge && isCloseToRightEdge)) {
            rect.x = imageCenterX - rect.width / 2.0f;
        }
        else {
            // ���߶����ڽӽ�ʱ����������ΪͼƬ���ȵ�����֮һ
            rect.width = imageSize.width / 3.0f;
            rect.x = imageCenterX - rect.width / 2.0f;
        }
    }

    // ������߻�ױ߹��ڽӽ���Ե
    if (isCloseToTopEdge || isCloseToBottomEdge) {
        // ֻ��һ�߿���ʱ�������ñߵ���Ա߶ԳƵ�λ��
        if (!(isCloseToTopEdge && isCloseToBottomEdge)) {
            rect.y = imageCenterY - rect.height / 2.0f;
        }
        else {
            // ���߶����ڽӽ�ʱ�������߶�ΪͼƬ�߶ȵ�����֮һ
            rect.height = imageSize.height / 3.0f;
            rect.y = imageCenterY - rect.height / 2.0f;
        }
    }
}

void eliminateYoloBackground::moveToIntersect(cv::Rect& rectToMove, const cv::Rect& referenceRect)
{
    // �������Ƿ��ཻ
    auto isIntersecting = [](const cv::Rect& rect1, const cv::Rect& rect2) -> bool {
        return !(rect1.x > rect2.x + rect2.width ||
            rect1.x + rect1.width < rect2.x ||
            rect1.y > rect2.y + rect2.height ||
            rect1.y + rect1.height < rect2.y);
        };

    // ���rectToMove����referenceRect�ཻ
    if (!isIntersecting(rectToMove, referenceRect)) {
        // ���rectToMove�Ƿ���referenceRect���Ϸ�
        if (rectToMove.y + rectToMove.height <= referenceRect.y) {
            // ����ǣ���rectToMove�ĵױ�������referenceRect�Ķ��߶���
            rectToMove.height = referenceRect.y - rectToMove.y;
        }
        // ���򣬼��rectToMove�Ƿ���referenceRect���·�
        else if (rectToMove.y >= referenceRect.y + referenceRect.height) {
            // �����ƶ�������rectToMove�Ķ���������referenceRect�ĵױ߶���
            int moveUpBy = rectToMove.y - (referenceRect.y + referenceRect.height);
            // ����rectToMove��λ�úʹ�С��ȷ���ױ߲���
            rectToMove.y -= moveUpBy;
            rectToMove.height += moveUpBy;
        }
    }
}
