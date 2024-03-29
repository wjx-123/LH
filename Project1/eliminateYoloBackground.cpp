#include "eliminateYoloBackground.h"

eliminateYoloBackground::eliminateYoloBackground()
{
}

eliminateYoloBackground::~eliminateYoloBackground()
{
}

std::pair<cv::Rect2f, std::vector<cv::Rect2f>> eliminateYoloBackground::getBoundAndPin(cv::Mat& image, std::string types)
{
    cv::Mat heibai = test(image,types);
    if (types == "T")//两脚
    {
        bool ifTransform = false;
        //判断是否是竖直的
        if (heibai.rows < heibai.cols) 
        {
            ifTransform = true;
            cv::transpose(image, image); // 转置
            cv::flip(image, image, 0);   // 沿y轴翻转
            cv::transpose(heibai, heibai); // 转置
            cv::flip(heibai, heibai, 0);   // 沿y轴翻转
        }
        std::vector<cv::Rect2f> tPin;
        cv::Rect2f rectHeibai = { static_cast<float>(image.cols * 0.3), static_cast<float>(image.rows * 0.35), static_cast<float>(image.cols * 0.35), static_cast<float>(image.rows * 0.25) };
        cv::rectangle(heibai, rectHeibai, cv::Scalar(0, 0, 0), cv::FILLED);
        auto [topLeft, bottomRight] = findBoundingRectangle_heibai(heibai, 0.1);
        cv::Rect black_rect = cv::Rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
        //cv::rectangle(image, black_rect, cv::Scalar(0, 255, 100), 2);
        adjustRect(black_rect, image.size());
        //cv::rectangle(image, black_rect, cv::Scalar(255, 0, 0), 2);
        cv::Mat topImg = heibai({ 0,0,static_cast<int>(image.cols),static_cast<int>(black_rect.y) });
        cv::Mat bottomImg = heibai({ 0,static_cast<int>(black_rect.y + black_rect.height),static_cast<int>(image.cols),static_cast<int>(image.rows - (black_rect.y + black_rect.height)) });
        /*扩散*/
        cv::bitwise_not(topImg, topImg);
        auto [topLeft_top, bottomRight_top] = findBoundingRectangle_heibai(topImg, 0.5);
        cv::Rect topImg_rect = cv::Rect(topLeft_top.x, topLeft_top.y, bottomRight_top.x - topLeft_top.x, bottomRight_top.y - topLeft_top.y);
        moveToIntersect(topImg_rect, black_rect);
        cv::bitwise_not(bottomImg, bottomImg);
        auto [topLeft_bottom, bottomRight_bottom] = findBoundingRectangle_heibai(bottomImg, 0.5);
        cv::Rect2f bottomImg_rect = cv::Rect(topLeft_bottom.x, topLeft_bottom.y, bottomRight_bottom.x - topLeft_bottom.x, bottomRight_bottom.y - topLeft_bottom.y);
        cv::Rect boundingBox1Adjusted(static_cast<float>(bottomImg_rect.x),
                static_cast<float>(bottomImg_rect.y + black_rect.y + black_rect.height),
                static_cast<float>(bottomImg_rect.width),
                static_cast<float>(bottomImg_rect.height));
        moveToIntersect(boundingBox1Adjusted, black_rect);

        if (ifTransform = true)
        {
            cv::flip(image, image, 0);   // 沿 y 轴翻转
            cv::transpose(image, image);
            transformRectCoordinates(topImg_rect);
            transformRectCoordinates(boundingBox1Adjusted);
            transformRectCoordinates(black_rect);
        }

        /*cv::rectangle(image, topImg_rect, cv::Scalar(255, 0, 0), 2);
        cv::rectangle(image, boundingBox1Adjusted, cv::Scalar(255, 0, 0), 2);*/
        tPin.push_back(topImg_rect);
        tPin.push_back(boundingBox1Adjusted);

        // 异步地反转上部图像并寻找边界矩形
        //auto futureTop = std::async(std::launch::async, [&]() {
        //    cv::bitwise_not(topImg, topImg); // 反转颜色
        //    auto [topLeft_top, bottomRight_top] = findBoundingRectangle_heibai(topImg, 0.5);
        //    cv::Rect topImg_rect(topLeft_top.x, topLeft_top.y, bottomRight_top.x - topLeft_top.x, bottomRight_top.y - topLeft_top.y);
        //    moveToIntersect(topImg_rect, black_rect);
        //    return topImg_rect;
        //    });

        //// 异步地反转下部图像并寻找边界矩形
        //auto futureBottom = std::async(std::launch::async, [&]() {
        //    cv::bitwise_not(bottomImg, bottomImg); // 反转颜色
        //    auto [topLeft_bottom, bottomRight_bottom] = findBoundingRectangle_heibai(bottomImg, 0.5);
        //    cv::Rect bottomImg_rect(topLeft_bottom.x, topLeft_bottom.y + black_rect.y + black_rect.height, bottomRight_bottom.x - topLeft_bottom.x, bottomRight_bottom.y - topLeft_bottom.y);
        //    moveToIntersect(bottomImg_rect, black_rect);
        //    return bottomImg_rect;
        //    });

        //// 获取异步操作的结果
        //cv::Rect topImg_rect = futureTop.get();
        //cv::Rect bottomImg_rect = futureBottom.get();
        //tPin.push_back(topImg_rect);
        //tPin.push_back(bottomImg_rect);

        std::pair pair  = { black_rect ,tPin };
        return pair;
    }
    else if (types == "R")//三脚
    {
        //把三脚的三分之一正方形也中间填充一下
        cv::rectangle(heibai, cv::Point(heibai.cols / 3, heibai.rows / 3), cv::Point(heibai.cols * 2 / 3, heibai.rows * 2 / 3), cv::Scalar(0, 0, 0), cv::FILLED);
        auto [topLeft, bottomRight] = findBoundingRectangle_heibai(heibai, 0.2);
        cv::Mat hsvImage = useHsvTest(image);//用hsv处理的图做一下交集
        cv::Rect2f black_rect = cv::Rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
        auto pinVector = findPinsAroundBlackBox_ofThree(hsvImage, black_rect, hsvImage);//heibai
        //截取最接近正方形的前三个矩形
        if (pinVector.size() > 3)
        {
            std::sort(pinVector.begin(), pinVector.end(), compareRectsCloseToSquare);
            // 保留最接近正方形的前三个矩形框
            pinVector.resize(3);
        }
        std::pair pair = { black_rect, pinVector };
        return pair;
    }
    else
    {
        auto [topLeft, bottomRight] = findBoundingRectangle_heibai(heibai, 0.2);
        cv::Mat hsvImage = useHsvTest(image);//用hsv处理的图做一下交集
        cv::Rect2f black_rect = cv::Rect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
        auto pinVector = findPinsAroundBlackBox(heibai, black_rect, hsvImage);
        processRects(pinVector);
        addSymmetricRectsIfNeeded(pinVector, black_rect);
        std::pair pair = { black_rect, pinVector };
        return pair;
    }
}

std::pair<cv::Point, cv::Point> eliminateYoloBackground::findBoundingRectangle(const cv::Mat& img, int colorRange)
{
    // 得到中心点
    cv::Point center(img.cols / 2, img.rows / 2);
    cv::Vec3b centerColor = img.at<cv::Vec3b>(center);

    // 定义边界的两个点
    cv::Point topLeft(img.cols, img.rows);
    cv::Point bottomRight(0, 0);

    // 记录像素
    std::vector<std::vector<bool>> visited(img.rows, std::vector<bool>(img.cols, false));

    // Lambda 检查像素(和正中间的像素相比)
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

        // 更新边界点
        topLeft.x = std::min(topLeft.x, p.x);
        topLeft.y = std::min(topLeft.y, p.y);
        bottomRight.x = std::max(bottomRight.x, p.x);
        bottomRight.y = std::max(bottomRight.y, p.y);

        // 检查四个相邻像素
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue; // 跳过中心点本身

                int newX = p.x + dx;
                int newY = p.y + dy;

                // 检查边界以及像素是否相似且未被访问
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
//    // 得到中心点
//    cv::Point center(img.cols / 2, img.rows / 2);
//    uchar centerColor = img.at<uchar>(center);
//
//    // 定义边界的两个点
//    cv::Point topLeft(img.cols, img.rows);
//    cv::Point bottomRight(0, 0);
//
//    // 记录像素
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
//        // 更新边界点
//        topLeft.x = std::min(topLeft.x, p.x);
//        topLeft.y = std::min(topLeft.y, p.y);
//        bottomRight.x = std::max(bottomRight.x, p.x);
//        bottomRight.y = std::max(bottomRight.y, p.y);
//
//        // 检查四个相邻像素
//        for (int dy = -1; dy <= 1; ++dy) {
//            for (int dx = -1; dx <= 1; ++dx) {
//                if (dx == 0 && dy == 0) continue; // 跳过中心点本身
//
//                int newX = p.x + dx;
//                int newY = p.y + dy;
//
//                // 检查边界以及像素是否已被访问
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

    // 扩散搜索的迭代过程
    while (true) {
        bool changed = false; // 用于判断此次循环是否有边界扩散

        // 向上扩散搜索
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

        // 向下扩散搜索
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

        // 向左扩散搜索
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

        // 向右扩散搜索
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

        // 如果两个相对边停止扩散，则退出循环
        /*if ((stopTop && stopBottom) || (stopLeft && stopRight)) {
            break;
        }*/

        // 如果没有边界扩散，且剩余的两边中至少有一边停止，则退出循环
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
    //根据中间的矩形框分为四个部分，异步搜索
    std::vector<cv::Rect2f> res, finalRes;
    float width = blackBox.width;
    float height = blackBox.height;

    // 确定长边和短边
    float longEdge = std::max(width, height);
    float shortEdge = std::min(width, height);

    // 检查长短边的差距是否超过长边的20%
    bool longEdgeIsWidth = width > height;
    bool differenceExceedsThreshold = (longEdge - shortEdge) > longEdge * 0.2;

    // 绑定成员函数
    auto boundFunc = std::bind(&eliminateYoloBackground::getPinRect, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);

    // 根据长短边的差距选择搜索的边
    std::future<std::vector<cv::Rect>> futureTop, futureBottom, futureLeft, futureRight;
    if (differenceExceedsThreshold && longEdgeIsWidth) {
        // 差距超过20%，长边是宽度，搜索上下方向
        std::cout << "差距超过20%，长边是宽度，搜索上下方向" << std::endl;
        futureTop = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, 0, blackBox.width, blackBox.y), 'L', 0);
        futureBottom = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, blackBox.y + blackBox.height, blackBox.width, img.rows - (blackBox.y + blackBox.height)), 'L', 2);
    }
    else if (differenceExceedsThreshold && !longEdgeIsWidth) {
        // 差距超过20%，长边是高度，搜索左右方向
        std::cout << "差距超过20%，长边是高度，搜索左右方向" << std::endl;
        futureLeft = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(0, blackBox.y, blackBox.x, blackBox.height), 'T', 3);
        futureRight = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x + blackBox.width, blackBox.y, img.cols - (blackBox.x + blackBox.width), blackBox.height), 'T', 1);
    }
    else {
        // 差距不超过20%，搜索所有方向
        std::cout << "差距不超过20%，搜索所有方向" << std::endl;
        futureTop = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, 0, blackBox.width, blackBox.y), 'L', 0);
        futureBottom = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, blackBox.y + blackBox.height, blackBox.width, img.rows - (blackBox.y + blackBox.height)), 'L', 2);
        futureLeft = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(0, blackBox.y, blackBox.x, blackBox.height), 'T', 3);
        futureRight = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x + blackBox.width, blackBox.y, img.cols - (blackBox.x + blackBox.width), blackBox.height), 'T', 1);
    }

    // 收集结果
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

    //用hsv图取交集筛选矩形
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
    //根据中间的矩形框分为四个部分，异步搜索
    std::vector<cv::Rect2f> res, finalRes;
    float width = blackBox.width;
    float height = blackBox.height;

    // 确定长边和短边
    float longEdge = std::max(width, height);
    float shortEdge = std::min(width, height);

    // 检查长短边的差距是否超过长边的20%
    bool longEdgeIsWidth = width > height;
    bool differenceExceedsThreshold = (longEdge - shortEdge) > longEdge * 0.2;

    // 绑定成员函数
    auto boundFunc = std::bind(&eliminateYoloBackground::getPinRect_ofThree, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);

    // 根据长短边的差距选择搜索的边
    std::future<std::vector<cv::Rect>> futureTop, futureBottom, futureLeft, futureRight;
    if (differenceExceedsThreshold && longEdgeIsWidth) {
        // 差距超过20%，长边是宽度，搜索上下方向
        std::cout << "差距超过20%，长边是宽度，搜索上下方向" << std::endl;
        futureTop = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, 0, blackBox.width, blackBox.y), 'L', 0);
        futureBottom = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, blackBox.y + blackBox.height, blackBox.width, img.rows - (blackBox.y + blackBox.height)), 'L', 2);
    }
    else if (differenceExceedsThreshold && !longEdgeIsWidth) {
        // 差距超过20%，长边是高度，搜索左右方向
        std::cout << "差距超过20%，长边是高度，搜索左右方向" << std::endl;
        futureLeft = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(0, blackBox.y, blackBox.x, blackBox.height), 'T', 3);
        futureRight = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x + blackBox.width, blackBox.y, img.cols - (blackBox.x + blackBox.width), blackBox.height), 'T', 1);
    }
    else {
        // 差距不超过20%，搜索所有方向
        std::cout << "差距不超过20%，搜索所有方向" << std::endl;
        futureTop = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, 0, blackBox.width, blackBox.y), 'L', 0);
        futureBottom = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x, blackBox.y + blackBox.height, blackBox.width, img.rows - (blackBox.y + blackBox.height)), 'L', 2);
        futureLeft = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(0, blackBox.y, blackBox.x, blackBox.height), 'T', 3);
        futureRight = std::async(std::launch::async, boundFunc, std::ref(img), cv::Rect(blackBox.x + blackBox.width, blackBox.y, img.cols - (blackBox.x + blackBox.width), blackBox.height), 'T', 1);
    }

    // 收集结果
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
    //用hsv图取交集筛选矩形
    for (auto temp : res)
    {
        float tempArea = temp.width * temp.height;

        // 检查矩形temp是否包含白色像素，并且面积介于blackBox的五分之二与1000像素之间
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
        // 转换为灰度图
        cv::Mat gray, eroded;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        int temp = cv::threshold(gray, gray, 0, 255, cv::THRESH_TRIANGLE);

        // 应用阈值操作，以便轮廓检测
        cv::Mat thresh;
        cv::threshold(gray, thresh, temp, 255, cv::THRESH_BINARY);
        //cv::Rect rectMask =  { static_cast<int>(img.cols * 0.25), static_cast<int>(img.rows * 0.4), static_cast<int>(img.cols * 0.5), static_cast<int>(img.rows * 0.3)};
        //cv::rectangle(thresh, rectMask, cv::Scalar(0), cv::FILLED);

        int kernelSize = 5;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

        // 应用闭运算
        cv::Mat closed;
        cv::morphologyEx(thresh, closed, cv::MORPH_CLOSE, kernel);

        // 应用开运算
        cv::Mat opened;
        cv::morphologyEx(closed, opened, cv::MORPH_OPEN, kernel);

        //中值滤波进行噪声去除
        cv::medianBlur(opened, opened, kernelSize);

        // 2. 连通组件分析去除小对象
        cv::Mat labels, stats, centroids;
        //cv::Mat filteredImage = cv::Mat::zeros(opened.size(), CV_8UC1);
        int nLabels = cv::connectedComponentsWithStats(opened, labels, stats, centroids);

        // 设置面积阈值来过滤小组件
        int areaThreshold = 600; // 根据需要调整此阈值

        // 遍历所有检测到的组件
        for (int label = 1; label < nLabels; label++) {
            int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area < areaThreshold) {
                // 在filteredImage上绘制组件
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
    // 首先将图像从BGR转换到HSV颜色空间
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
    // 定义红色和黄色的HSV颜色范围
    //cv::Scalar lowerRed1(0, 120, 70);
    //cv::Scalar upperRed1(10, 255, 255);
    //cv::Scalar lowerRed2(170, 120, 70);
    //cv::Scalar upperRed2(180, 255, 255);
    //cv::Scalar lowerYellow(25, 50, 50);
    //cv::Scalar upperYellow(60, 255, 255);
    //// 黑色的HSV范围
    //cv::Scalar lowerBlack(0, 0, 0);
    //cv::Scalar upperBlack(180, 255, 50); // 明度上限设定较低，饱和度可以较宽泛
    //// 灰色的HSV范围
    //cv::Scalar lowerGray(0, 0, 50);
    //cv::Scalar upperGray(180, 50, 220); // 饱和度上限较低，明度覆盖从较暗到较亮的范围
    //
    //// 筛选颜色范围并创建掩码
    //cv::Mat maskRed1, maskRed2, maskYellow, maskBlack, maskGray;
    //cv::inRange(hsvImage, lowerRed1, upperRed1, maskRed1);
    //cv::inRange(hsvImage, lowerRed2, upperRed2, maskRed2);
    //cv::inRange(hsvImage, lowerYellow, upperYellow, maskYellow);
    //cv::inRange(hsvImage, lowerBlack, upperBlack, maskBlack);
    //cv::inRange(hsvImage, lowerGray, upperGray, maskGray);

    //// 合并红色区域的两个掩码，然后与黄色掩码合并
    //cv::Mat maskRed = maskRed1 | maskRed2;
    //cv::Mat maskCombined = maskRed | maskYellow;
    //// 进一步将黑色和灰色掩码加入到合并后的掩码中
    ////maskCombined = maskCombined | maskBlack | maskGray;
    //int morph_size = 2;  // 结构元素的尺寸，根据需要进行调整
    //cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
    //    cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
    //    cv::Point(morph_size, morph_size));
    //cv::morphologyEx(maskCombined, maskCombined, cv::MORPH_OPEN, element);
    return himg;
}

std::vector<cv::Rect> eliminateYoloBackground::getPinRect(cv::Mat& imgInit, cv::Rect rectInit, char baseline, int closestEdge)
{
    cv::Mat img = imgInit(rectInit);   
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> rectangles;

    // 遍历所有轮廓并获取其外接矩形
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        
        // 条件过滤
        if (std::max(rect.width,rect.height) < std::min(img.rows,img.cols) * 0.2) 
        {
            continue;
        }
        rect.x += rectInit.x;
        rect.y += rectInit.y;
        
        rectangles.push_back(rect);
    }

    // 根据基准线合并矩形
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
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> rectangles;

    // 遍历所有轮廓并获取其外接矩形
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);

        // 条件过滤
        if (std::max(rect.width, rect.height) < std::min(img.rows, img.cols) * 0.2)
        {
            continue;
        }
        rect.x += rectInit.x;
        rect.y += rectInit.y;

        rectangles.push_back(rect);
    }

    // 根据基准线合并矩形
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
    // 合并逻辑，基于左边界的距离
    std::vector<cv::Rect> merged;
    while (!rectangles.empty()) {
        // Always merge with the first rectangle
        cv::Rect& base = rectangles.front();
        auto it = rectangles.begin();
        for (auto jt = rectangles.begin() + 1; jt != rectangles.end();) {
            if (std::abs(jt->x - base.x) < threshold) {
                base = base | *jt; // 合并矩形
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
    // 合并逻辑，基于上边界的距离
    std::vector<cv::Rect> merged;
    while (!rectangles.empty()) {
        // Always merge with the first rectangle
        cv::Rect& base = rectangles.front();
        auto it = rectangles.begin();
        for (auto jt = rectangles.begin() + 1; jt != rectangles.end();) {
            if (std::abs(jt->y - base.y) < threshold) {
                base = base | *jt; // 合并矩形
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
    // 提取ROI
    cv::Mat roi = image(rect);
    // 检查ROI中是否有非零像素
    return cv::countNonZero(roi) > 1;
}

void eliminateYoloBackground::filterRectangles(std::vector<cv::Rect>& rectangles, const cv::Rect& rectInit, int closestEdge, double thresholdRatio)
{
    std::vector<double> distances;

    // 计算每个矩形中心点到指定边的距离
    for (const auto& rect : rectangles) {
        cv::Point rectCenter(rect.x + rect.width / 2, rect.y + rect.height / 2);
        double distance = 0.0;

        switch (closestEdge) {
        case 0: // 下边
            distance = std::abs(rectCenter.y - rectInit.br().y);
            break;
        case 1: // 左边
            distance = std::abs(rectCenter.x - rectInit.x);
            break;
        case 2: // 上边
            distance = std::abs(rectCenter.y - rectInit.y);
            break;
        case 3: // 右边
            distance = std::abs(rectCenter.x - rectInit.br().x);
            break;
        }
        distances.push_back(distance);
    }

    // 计算距离的平均值
    double averageDistance = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();
    double distanceThreshold = averageDistance * thresholdRatio; // 使用平均值的一定比例作为阈值

    // 剔除距离超过阈值且不与边界相交的矩形
    auto newEnd = std::remove_if(rectangles.begin(), rectangles.end(),
        [&](const cv::Rect& rect) -> bool {
            cv::Point rectCenter(rect.x + rect.width / 2, rect.y + rect.height / 2);
            double distance = 0.0;
            bool intersects = false;

            // 计算距离并判断是否相交
            switch (closestEdge) {
            case 0: // 下边
                distance = std::abs(rectCenter.y - rectInit.br().y);
                intersects = (rect.br().y > rectInit.br().y - 2);
                break;
            case 1: // 左边
                distance = std::abs(rectCenter.x - rectInit.x);
                intersects = (rect.x < rectInit.x + 2);
                break;
            case 2: // 上边
                distance = std::abs(rectCenter.y - rectInit.y);
                intersects = (rect.y < rectInit.y + 2);
                break;
            case 3: // 右边
                distance = std::abs(rectCenter.x - rectInit.br().x);
                intersects = (rect.br().x > rectInit.br().x - 2);
                break;
            }

            // 如果矩形的距离超过阈值且不与边界相交，则应剔除
            return distance > distanceThreshold || !intersects;
        });

    rectangles.erase(newEnd, rectangles.end());
}

void eliminateYoloBackground::adjustRect(cv::Rect& rect, const cv::Size& imageSize)
{
    //// 定义"过近"的阈值
    const float threshold_short = 60.0;
    const float threshold_long = 10.0;
    //// 图片中心点坐标
    //const float imageCenterX = imageSize.width / 2.0;
    //const float imageCenterY = imageSize.height / 2.0;

    //// 矩形中心点坐标
    //const float rectCenterX = rect.x + rect.width / 2.0;
    //const float rectCenterY = rect.y + rect.height / 2.0;

    //// 检查矩形框与图片边缘的距离是否小于阈值
    //bool isCloseToLeftEdge = rect.x < threshold;
    //bool isCloseToRightEdge = imageSize.width - (rect.x + rect.width) < threshold;

    //// 如果左边或右边太近，则进行调整
    //if (isCloseToLeftEdge || isCloseToRightEdge) {
    //    // 图片短边的长度
    //    float shortSideLength = std::min(imageSize.width, imageSize.height);

    //    // 调整后的矩形宽度是图片短边的三分之一
    //    float newWidth = shortSideLength * 0.625;

    //    // 如果两边都太近，矩形中心应与图片中心对齐
    //    if (isCloseToLeftEdge && isCloseToRightEdge) {
    //        rect.x = imageCenterX - newWidth / 2.0;
    //    }
    //    else if (isCloseToLeftEdge) {
    //        // 如果只有左边太近，将矩形向右移动
    //        rect.x = imageCenterX - (imageSize.width - rect.x - rect.width) - newWidth;
    //    }
    //    else {
    //        // 如果只有右边太近，将矩形向左移动
    //        rect.x = imageCenterX - rect.x - newWidth;
    //    }

    //    // 更新矩形宽度
    //    rect.width = newWidth;
    //}


    // 图片中心点坐标
    float imageCenterX = imageSize.width / 2.0f;
    float imageCenterY = imageSize.height / 2.0f;

    // 检查矩形的每条边是否过于接近图片边缘
    bool isCloseToLeftEdge = rect.x < threshold_long;
    bool isCloseToRightEdge = imageSize.width - (rect.x + rect.width) < threshold_long;
    bool isCloseToTopEdge = rect.y < threshold_short;
    bool isCloseToBottomEdge = imageSize.height - (rect.y + rect.height) < threshold_short;

    // 如果左边或右边过于接近边缘
    if (isCloseToLeftEdge || isCloseToRightEdge) {
        // 只有一边靠近时，调整该边到与对边对称的位置
        if (!(isCloseToLeftEdge && isCloseToRightEdge)) {
            rect.x = imageCenterX - rect.width / 2.0f;
        }
        else {
            // 两边都过于接近时，调整宽度为图片宽度的三分之一
            rect.width = imageSize.width * 0.75f;/// 3.0f
            rect.x = imageCenterX - rect.width / 2.0f;
        }
    }

    // 如果顶边或底边过于接近边缘
    if (isCloseToTopEdge || isCloseToBottomEdge) {
        // 只有一边靠近时，调整该边到与对边对称的位置
        if (!(isCloseToTopEdge && isCloseToBottomEdge)) {
            rect.y = imageCenterY - rect.height / 2.0f;
        }
        else {
            // 两边都过于接近时，调整高度为图片高度的三分之一
            rect.height = imageSize.height / 3.0f;
            rect.y = imageCenterY - rect.height / 2.0f;
        }
    }
}

void eliminateYoloBackground::moveToIntersect(cv::Rect& rectToMove, const cv::Rect& referenceRect)
{
    // 检查矩形是否相交
    auto isIntersecting = [](const cv::Rect& rect1, const cv::Rect& rect2) -> bool {
        return !(rect1.x > rect2.x + rect2.width ||
            rect1.x + rect1.width < rect2.x ||
            rect1.y > rect2.y + rect2.height ||
            rect1.y + rect1.height < rect2.y);
        };

    // 如果rectToMove不与referenceRect相交
    if (!isIntersecting(rectToMove, referenceRect)) {
        // 检查rectToMove是否在referenceRect的上方
        if (rectToMove.y + rectToMove.height <= referenceRect.y) {
            // 如果是，则将rectToMove的底边下移与referenceRect的顶边对齐
            rectToMove.height = referenceRect.y - rectToMove.y;
        }
        // 否则，检查rectToMove是否在referenceRect的下方
        else if (rectToMove.y >= referenceRect.y + referenceRect.height) {
            // 计算移动量，将rectToMove的顶边上移与referenceRect的底边对齐
            int moveUpBy = rectToMove.y - (referenceRect.y + referenceRect.height);
            // 更新rectToMove的位置和大小，确保底边不动
            rectToMove.y -= moveUpBy;
            rectToMove.height += moveUpBy;
        }
    }
}

void eliminateYoloBackground::transformRectCoordinates(cv::Rect& rect)
{
    // 将竖直方向的 x 坐标赋值给新的 y 坐标
    float x = rect.y;
    // 将竖直方向的 y 坐标转换为新的 x 坐标
    float y = rect.x;
    rect.x = x;
    rect.y = y;

    // 交换宽度和高度
    float temp = rect.width;
    rect.width = rect.height;
    rect.height = temp;
}

bool eliminateYoloBackground::compareRectsCloseToSquare(const cv::Rect2f& a, const cv::Rect2f& b)
{
    float diffA = std::abs(a.width - a.height);
    float diffB = std::abs(b.width - b.height);
    return diffA < diffB; // 按照接近正方形的程度排序
}

bool eliminateYoloBackground::rectsAreSimilar(const cv::Rect2f& a, const cv::Rect2f& b)
{
    return std::abs(a.x - b.x) < SIMILARITY_THRESHOLD &&
        std::abs(a.y - b.y) < SIMILARITY_THRESHOLD &&
        std::abs(a.width - b.width) < SIMILARITY_THRESHOLD &&
        std::abs(a.height - b.height) < SIMILARITY_THRESHOLD;
}

cv::Rect2f eliminateYoloBackground::findTemplateRect(const std::vector<cv::Rect2f>& rects)
{
    std::vector<std::pair<cv::Rect2f, int>> frequency; // 矩形及其出现频率
    for (const auto& rect : rects) {
        bool found = false;
        for (auto& item : frequency) {
            if (rectsAreSimilar(rect, item.first)) {
                item.second++;
                found = true;
                break;
            }
        }
        if (!found) {
            frequency.push_back({ rect, 1 });
        }
    }

    auto maxElem = std::max_element(frequency.begin(), frequency.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

    return maxElem->first;
}

void eliminateYoloBackground::filterRects(std::vector<cv::Rect2f>& rects, const cv::Rect2f& templateRect)
{
    float templateArea = templateRect.area();
    rects.erase(std::remove_if(rects.begin(), rects.end(),
        [templateArea](const cv::Rect2f& rect) {
            return rect.area() > 2 * templateArea || rect.area() < templateArea / 2;
        }),
        rects.end());
}

void eliminateYoloBackground::processRects(std::vector<cv::Rect2f>& rects)
{
    if (rects.size() > 1) 
    {
        cv::Rect2f templateRect = findTemplateRect(rects);
        filterRects(rects, templateRect);
    }
    
}

cv::Rect2f eliminateYoloBackground::calculateSymmetricRect(const cv::Rect2f& sourceRect, const cv::Rect2f& black_rect)
{
    cv::Point2f center_of_black = cv::Point2f(black_rect.x + black_rect.width / 2, black_rect.y + black_rect.height / 2);
    cv::Point2f source_center = cv::Point2f(sourceRect.x + sourceRect.width / 2, sourceRect.y + sourceRect.height / 2);
    cv::Point2f sym_center;

    // 计算对称中心点
    sym_center.x = center_of_black.x + (center_of_black.x - source_center.x);
    sym_center.y = center_of_black.y + (center_of_black.y - source_center.y);

    // 根据对称中心点创建对称矩形
    cv::Rect2f symmetricRect(sym_center.x - sourceRect.width / 2, sym_center.y - sourceRect.height / 2, sourceRect.width, sourceRect.height);

    return symmetricRect;
}

bool eliminateYoloBackground::isOverlappingMoreThanHalf(const cv::Rect2f& rect1, const cv::Rect2f& rect2)
{
    cv::Rect2f intersection = rect1 & rect2;
    float intersectionArea = intersection.area();
    return intersectionArea >= rect2.area() * 0.5 || intersectionArea >= rect1.area() * 0.5;
}

void eliminateYoloBackground::addSymmetricRectsIfNeeded(std::vector<cv::Rect2f>& rects, const cv::Rect2f& black_rect)
{
    float symmetryAxisY = black_rect.y + (black_rect.height / 2.0f);
    std::vector<cv::Rect2f> newRects;

    for (const auto& rect : rects) {
        float centerY = rect.y + (rect.height / 2.0f);
        float symmetricalCenterY = (symmetryAxisY - centerY) + symmetryAxisY;
        cv::Rect2f symmetricalRect(rect.x, symmetricalCenterY - (rect.height / 2.0f), rect.width, rect.height);

        // 检查是否已经存在对称的矩形
        bool exists = std::any_of(rects.begin(), rects.end(), [&symmetricalRect](const cv::Rect2f& r) {
            float rCenterY = r.y + (r.height / 2.0f);
            float symRectCenterY = symmetricalRect.y + (symmetricalRect.height / 2.0f);
            // 检查 y 方向上中心点是否接近
            if (std::abs(rCenterY - symRectCenterY) <= r.height / 2.0f) {
                // 检查重叠面积是否超过 50%
                cv::Rect2f intersection = r & symmetricalRect;
                return (intersection.area() >= r.area() * 0.5 || intersection.area() >= symmetricalRect.area() * 0.5);
            }
            return false;
            });

        if (!exists) {
            newRects.push_back(symmetricalRect);
        }
    }

    rects.insert(rects.end(), newRects.begin(), newRects.end());
}

