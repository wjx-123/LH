#include "halconMatch.h"
HMatch::HMatch()
{
}
HMatch::~HMatch()
{
}
int HMatch::getNumberOfPanel(cv::Rect2f firstRect, cv::Size2f imageSize)
{
    // 计算每个维度上最多可以放置的矩形数量
    int numRectsWidth = static_cast<int>(imageSize.width / firstRect.width);
    int numRectsHeight = static_cast<int>(imageSize.height / firstRect.height);

    // 计算总的矩形数量
    int totalRects = numRectsWidth * numRectsHeight;
    return totalRects;
}
void HMatch::getPanelFrames(int numberOfPanel, cv::Mat Image, cv::Rect2f model, std::vector<cv::Rect2f>& result)
{
    /*基于ncc的匹配*/
    float xModel = model.x;
    float yModel = model.y;
    std::cout<< getNumberOfPanel(Image.size(), model) << std::endl;
    double temp = (double)Image.cols / SCALE_FACTOR;//转化系数
    resizeImage(Image);//压缩图片
    singleImage(Image);//转单通道
    ZoomChange(model,temp);//缩放model
    HalconCpp::HObject Himage = MatToHImage(Image);//大图mat转halcon
    std::vector<cv::Point2f> centerPoints = HalconMatch_ncc(numberOfPanel, Himage, model);
    result = fromPointToRect(centerPoints,model,temp);
    alignRectangles(result,model);

    /*基于shape的匹配*/
    //HalconCpp::HObject HImage = MatToHImage(Image);
    ////HalconCpp::HObject MImage = MatToHImage(modelImg);
    //std::vector<cv::Point2f> centerPoints = HalconMatch_shape(numberOfPanel,HImage,model);
    //result = fromPointToRect(centerPoints, model,1);
}

void HMatch::getMaskShift(int numberOfPanel, cv::Mat dMask, cv::Mat mMask, cv::Rect2f& result)
{
    singleImage(dMask);//转单通道
    singleImage(mMask);
    HalconCpp::HObject HDMask = MatToHImage(dMask);//大图mat转halcon
    HalconCpp::HObject HMMask = MatToHImage(mMask);//大图mat转halcon

    //std::vector<cv::Point2f> centerPoints = HalconMatch_MaskShift(numberOfPanel, HDMask, HMMask);//基于相关性
    std::vector<cv::Point2f> centerPoints = HalconMatch_Ncc_MaskShift(numberOfPanel, HDMask, HMMask);//基于ncc
    cv::Rect2f model = cv::Rect2f(0,0,mMask.cols, mMask.rows);
    auto temp = fromPointToRect(centerPoints, model, 1);
    result = temp[0];
}

std::vector<std::pair<float, float>> HMatch::matchRectangles(const cv::Mat& srcImage, const cv::Rect2f& templateRect, const std::vector<cv::Rect2f>& rectangles, float scale)
{
    std::vector<std::pair<float, float>> offsets;

    // 对模板图像进行缩放
    cv::Mat templateImage = srcImage(templateRect);
    cv::resize(templateImage, templateImage, cv::Size(), scale, scale, cv::INTER_AREA);

    // 特征检测器和描述符
    auto detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> templateKeyPoints;
    cv::Mat templateDescriptors;
    detector->detectAndCompute(templateImage, cv::noArray(), templateKeyPoints, templateDescriptors);

    // 创建匹配器
    cv::BFMatcher matcher(cv::NORM_L2);

    // 遍历矩形
    for (const auto& rect : rectangles) {
        cv::Mat rectImage = srcImage(rect);
        cv::resize(rectImage, rectImage, cv::Size(), scale, scale, cv::INTER_AREA);  // 对每个矩形图像进行缩放

        std::vector<cv::KeyPoint> rectKeyPoints;
        cv::Mat rectDescriptors;
        detector->detectAndCompute(rectImage, cv::noArray(), rectKeyPoints, rectDescriptors);

        // 特征匹配
        std::vector<cv::DMatch> matches;
        matcher.match(templateDescriptors, rectDescriptors, matches);

        // 筛选好的匹配
        double max_dist = 0; double min_dist = 100;
        for (int i = 0; i < templateDescriptors.rows; i++) {
            double dist = matches[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }
        std::vector<cv::DMatch> good_matches;
        for (int i = 0; i < templateDescriptors.rows; i++) {
            if (matches[i].distance <= std::max(2 * min_dist, 0.02)) {
                good_matches.push_back(matches[i]);
            }
        }

        // 估计变换矩阵
        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;
        for (size_t i = 0; i < good_matches.size(); i++) {
            obj.push_back(templateKeyPoints[good_matches[i].queryIdx].pt);
            scene.push_back(rectKeyPoints[good_matches[i].trainIdx].pt);
        }
        cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);

        // 计算偏移
        if (!H.empty()) {
            double dx = H.at<double>(0, 2) / scale; // 修正缩放因子
            double dy = H.at<double>(1, 2) / scale; // 修正缩放因子
            offsets.push_back(std::make_pair(dx, dy));
        }
        else {
            offsets.push_back(std::make_pair(0.0f, 0.0f)); // No match found or poor matching
        }
    }

    return offsets;
}

void HMatch::singleImage(cv::Mat& Image)
{
    if (Image.channels() != 1)
    {
        // 转换为灰度图像（单通道）
        cv::cvtColor(Image, Image, cv::COLOR_BGR2GRAY);
    }

}
void HMatch::resizeImage(cv::Mat& Image)
{
    int width = Image.cols;
    int height = Image.rows;

    if (width <= SCALE_FACTOR && height <= SCALE_FACTOR) {
        return;
    }

    float ratio = static_cast<float>(SCALE_FACTOR) / std::max(width, height);

    int new_width = static_cast<int>(width * ratio);
    int new_height = static_cast<int>(height * ratio);

    cv::resize(Image, Image, cv::Size(new_width, new_height), cv::INTER_LANCZOS4);
}
// Procedures 
// Chapter: Develop
// Short Description: Open a new graphics window that preserves the aspect ratio of the given image. 
void HMatch::dev_open_window_fit_image(HalconCpp::HObject ho_Image, HalconCpp::HTuple hv_Row, HalconCpp::HTuple hv_Column,
    HalconCpp::HTuple hv_WidthLimit, HalconCpp::HTuple hv_HeightLimit, HalconCpp::HTuple* hv_WindowHandle)
{

    // Local iconic variables

    // Local control variables
    HalconCpp::HTuple  hv_MinWidth, hv_MaxWidth, hv_MinHeight;
    HalconCpp::HTuple  hv_MaxHeight, hv_ResizeFactor, hv_ImageWidth, hv_ImageHeight;
    HalconCpp::HTuple  hv_TempWidth, hv_TempHeight, hv_WindowWidth, hv_WindowHeight;

    //This procedure opens a new graphics window and adjusts the size
    //such that it fits into the limits specified by WidthLimit
    //and HeightLimit, but also maintains the correct image aspect ratio.
    //
    //If it is impossible to match the minimum and maximum extent requirements
    //at the same time (f.e. if the image is very long but narrow),
    //the maximum value gets a higher priority,
    //
    //Parse input tuple WidthLimit
    if (0 != (HalconCpp::HTuple(int((hv_WidthLimit.TupleLength()) == 0)).TupleOr(int(hv_WidthLimit < 0))))
    {
        hv_MinWidth = 500;
        hv_MaxWidth = 800;
    }
    else if (0 != (int((hv_WidthLimit.TupleLength()) == 1)))
    {
        hv_MinWidth = 0;
        hv_MaxWidth = hv_WidthLimit;
    }
    else
    {
        hv_MinWidth = ((const HalconCpp::HTuple&)hv_WidthLimit)[0];
        hv_MaxWidth = ((const HalconCpp::HTuple&)hv_WidthLimit)[1];
    }
    //Parse input tuple HeightLimit
    if (0 != (HalconCpp::HTuple(int((hv_HeightLimit.TupleLength()) == 0)).TupleOr(int(hv_HeightLimit < 0))))
    {
        hv_MinHeight = 400;
        hv_MaxHeight = 600;
    }
    else if (0 != (int((hv_HeightLimit.TupleLength()) == 1)))
    {
        hv_MinHeight = 0;
        hv_MaxHeight = hv_HeightLimit;
    }
    else
    {
        hv_MinHeight = ((const HalconCpp::HTuple&)hv_HeightLimit)[0];
        hv_MaxHeight = ((const HalconCpp::HTuple&)hv_HeightLimit)[1];
    }
    //
    //Test, if window size has to be changed.
    hv_ResizeFactor = 1;
    GetImageSize(ho_Image, &hv_ImageWidth, &hv_ImageHeight);
    //First, expand window to the minimum extents (if necessary).
    if (0 != (HalconCpp::HTuple(int(hv_MinWidth > hv_ImageWidth)).TupleOr(int(hv_MinHeight > hv_ImageHeight))))
    {
        hv_ResizeFactor = (((hv_MinWidth.TupleReal()) / hv_ImageWidth).TupleConcat((hv_MinHeight.TupleReal()) / hv_ImageHeight)).TupleMax();
    }
    hv_TempWidth = hv_ImageWidth * hv_ResizeFactor;
    hv_TempHeight = hv_ImageHeight * hv_ResizeFactor;
    //Then, shrink window to maximum extents (if necessary).
    if (0 != (HalconCpp::HTuple(int(hv_MaxWidth < hv_TempWidth)).TupleOr(int(hv_MaxHeight < hv_TempHeight))))
    {
        hv_ResizeFactor = hv_ResizeFactor * ((((hv_MaxWidth.TupleReal()) / hv_TempWidth).TupleConcat((hv_MaxHeight.TupleReal()) / hv_TempHeight)).TupleMin());
    }
    hv_WindowWidth = hv_ImageWidth * hv_ResizeFactor;
    hv_WindowHeight = hv_ImageHeight * hv_ResizeFactor;
    //Resize window
    HalconCpp::SetWindowAttr("background_color", "black");
    OpenWindow(hv_Row, hv_Column, hv_WindowWidth, hv_WindowHeight, 0, "visible", "", &(*hv_WindowHandle));
    HalconCpp::HDevWindowStack::Push((*hv_WindowHandle));
    if (HalconCpp::HDevWindowStack::IsOpen())
        SetPart(HalconCpp::HDevWindowStack::GetActive(), 0, 0, hv_ImageHeight - 1, hv_ImageWidth - 1);
    return;
}

cv::Mat HMatch::HImageToMat(HalconCpp::HObject& H_img)
{
    cv::Mat cv_img;
    HalconCpp::HTuple channels, w, h;

    HalconCpp::ConvertImageType(H_img, &H_img, "byte");
    HalconCpp::CountChannels(H_img, &channels);

    if (channels.I() == 1)
    {
        HalconCpp::HTuple pointer;
        GetImagePointer1(H_img, &pointer, nullptr, &w, &h);
        int width = w.I(), height = h.I();
        int size = width * height;
        cv_img = cv::Mat::zeros(height, width, CV_8UC1);
        memcpy(cv_img.data, (void*)(pointer.L()), size);
    }

    else if (channels.I() == 3)
    {
        HalconCpp::HTuple pointerR, pointerG, pointerB;
        HalconCpp::GetImagePointer3(H_img, &pointerR, &pointerG, &pointerB, nullptr, &w, &h);
        int width = w.I(), height = h.I();
        int size = width * height;
        cv_img = cv::Mat::zeros(height, width, CV_8UC3);
        uchar* R = (uchar*)(pointerR.L());
        uchar* G = (uchar*)(pointerG.L());
        uchar* B = (uchar*)(pointerB.L());
        for (int i = 0; i < height; ++i)
        {
            uchar* p = cv_img.ptr<uchar>(i);
            for (int j = 0; j < width; ++j)
            {
                p[3 * j] = B[i * width + j];
                p[3 * j + 1] = G[i * width + j];
                p[3 * j + 2] = R[i * width + j];
            }
        }
    }
    return cv_img;
}

HalconCpp::HObject HMatch::MatToHImage(cv::Mat& cv_img)
{
    HalconCpp::HObject H_img;

    if (cv_img.channels() == 1)
    {
        int height = cv_img.rows, width = cv_img.cols;
        int size = height * width;
        uchar* temp = new uchar[size];

        memcpy(temp, cv_img.data, size);
        HalconCpp::GenImage1(&H_img, "byte", width, height, (Hlong)(temp));

        delete[] temp;
    }
    else if (cv_img.channels() == 3)
    {
        int height = cv_img.rows, width = cv_img.cols;
        int size = height * width;
        uchar* B = new uchar[size];
        uchar* G = new uchar[size];
        uchar* R = new uchar[size];

        for (int i = 0; i < height; i++)
        {
            uchar* p = cv_img.ptr<uchar>(i);
            for (int j = 0; j < width; j++)
            {
                B[i * width + j] = p[3 * j];
                G[i * width + j] = p[3 * j + 1];
                R[i * width + j] = p[3 * j + 2];
            }
        }
        HalconCpp::GenImage3(&H_img, "byte", width, height, (Hlong)(R), (Hlong)(G), (Hlong)(B));

        delete[] R;
        delete[] G;
        delete[] B;
    }
    return H_img;
}

void HMatch::ZoomChange(cv::Rect2f &rect, double temp)
{
    rect.x = rect.x / temp;
    rect.y = rect.y / temp;
    rect.width = rect.width / temp;
    rect.height = rect.height / temp;
}

std::vector<cv::Point2f> HMatch::HalconMatch_ncc(int numberOfPanel, HalconCpp::HObject Himage, cv::Rect2f modelRect)
{
    HalconCpp::HObject  ho_ModelRegion, ho_TemplateImage;

    // Local control variables
    HalconCpp::HTuple  hv_ModelID, hv_ModelRegionArea, hv_RefRow;
    HalconCpp::HTuple  hv_RefColumn, hv_Row, hv_Column;
    HalconCpp::HTuple  hv_Angle, hv_Score;
    //
    //Matching 01: ************************************************
    //Matching 01: BEGIN of generated code for model initialization
    //Matching 01: ************************************************
    //Matching 01: Build the ROI from basic regions
    GenRectangle1(&ho_ModelRegion, modelRect.y, modelRect.x, modelRect.y + modelRect.height, modelRect.x + modelRect.width);
    //
    //Matching 01: Reduce the model template
    ReduceDomain(Himage, ho_ModelRegion, &ho_TemplateImage);
    //
    //Matching 01: Create the correlation model
    CreateNccModel(ho_TemplateImage, "auto", HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(360).TupleRad(),
        "auto", "use_polarity", &hv_ModelID);

    //Matching 01: Get the reference position
    AreaCenter(ho_ModelRegion, &hv_ModelRegionArea, &hv_RefRow, &hv_RefColumn);
    //Matching 01: Find the modelImage（in）：单通道图像，它的区域可被创建为模板
    /***
    **      ModelID（in）：模板句柄         
    **      AngleStart（in）：模板的最小旋转        
    **      AngleExtent（in）：旋转角度范围          
    **      MinScore（in）：被找到的模板最小分数
    **      NumMatches（in）：被找到的模板个数          
    **      MaxOverlap（in）：被找到的模板实例最大重叠部分           
    **      SubPixel（in）：亚像素级别标志,true, false     
    **      NumLevels（in）：金字塔层级数          
    **      Row（out）：被找到的模板实例行坐标
    **      Column（out）：被找到的模板实例列坐标       
    **      Angle（out）：被找到的模板实例的旋转角度        
    **      Score（out）：被找到的模板实例的分数
    ***/
        FindNccModel(Himage, hv_ModelID, HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(360).TupleRad(),
            0.5, numberOfPanel, 0.5, "true", 0, &hv_Row, &hv_Column, &hv_Angle, &hv_Score);
        std::vector<cv::Point2f> result;
        for (int i = 0; i < hv_Row.Length(); i++)
        {
            //std::cout << i << ":" << hv_Row[i].D() << " " << hv_Column[i].D() << " " << hv_Score[i].D() << std::endl;
            result.push_back(cv::Point(hv_Column[i].D(), hv_Row[i].D()));
        }
    return result;
}

std::vector<cv::Point2f> HMatch::HalconMatch_shape(int numberOfPanel, HalconCpp::HObject Himage, cv::Rect2f modelRect)
{
    // 本地图像变量
    HalconCpp::HObject ho_ModelRegion, ho_TemplateImage, ho_ModelContours, ho_TransContours;
    HalconCpp::HTuple hv_ModelID, hv_Angle, hv_Row, hv_Column, hv_Score, hv_HomMat2D;
    HalconCpp::SetSystem("border_shape_models", "false");
    // 使用 OpenCV 矩形参数在 Halcon 中生成模型区域
    HalconCpp::GenRectangle1(&ho_ModelRegion, modelRect.y, modelRect.x,
        modelRect.y + modelRect.height, modelRect.x + modelRect.width);

    // 提取模板图像
    HalconCpp::ReduceDomain(Himage, ho_ModelRegion, &ho_TemplateImage);

    try
    {
        // 创建形状模型
        CreateShapeModel(ho_TemplateImage, 6, HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(360).TupleRad(),
            HalconCpp::HTuple(0.028).TupleRad(), (HalconCpp::HTuple("point_reduction_high").Append("no_pregeneration")),
            "use_polarity", ((HalconCpp::HTuple(78).Append(136)).Append(51)), 4, &hv_ModelID);
    }
    catch (HalconCpp::HException& e) {
        std::cerr << "Error in CreateShapeModel: " << e.ErrorMessage() << std::endl;
    }
   

    // 获取模型轮廓
    HalconCpp::GetShapeModelContours(&ho_ModelContours, hv_ModelID, 1);

    // 准备返回的结果，使用cv::Point2f
    std::vector<cv::Point2f> foundPositions;

    try
    {
        /*HalconCpp::FindShapeModel(Himage, hv_ModelID, 0, HalconCpp::HTuple(360).TupleRad(), 0.5, numberOfPanel, 0.5,
            "least_squares", HalconCpp::HTuple(), HalconCpp::HTuple(),
            &hv_Row, &hv_Column, &hv_Angle, &hv_Score);*/
        HalconCpp::FindShapeModel(Himage, hv_ModelID, 0, HalconCpp::HTuple(360).TupleRad(), 0.1, numberOfPanel, 0.5,
            "least_squares", 0 ,0 ,
            &hv_Row, &hv_Column, &hv_Angle, &hv_Score);
    }
    catch (HalconCpp::HException& e)
    {
        std::cerr << "Error in FindShapeModel: " << e.ErrorMessage() << std::endl;
    }
    

    //if (hv_Score.Length() > 0) {
    //        // 转换模型轮廓到检测到的位置
    //    HalconCpp::HomMat2dIdentity(&hv_HomMat2D);
    //    HalconCpp::HomMat2dRotate(hv_HomMat2D, hv_Angle[0], 0, 0, &hv_HomMat2D);
    //    HalconCpp::HomMat2dTranslate(hv_HomMat2D, hv_Row[0], hv_Column[0], &hv_HomMat2D);
    //    HalconCpp::AffineTransContourXld(ho_ModelContours, &ho_TransContours, hv_HomMat2D);

    //    // 保存位置，转换为cv::Point2f
    //    foundPositions.push_back(cv::Point2f(hv_Column[0].D(), hv_Row[0].D()));
    //}
    for (int i = 0; i < hv_Row.Length(); i++)
    {
        HalconCpp::HomMat2dIdentity(&hv_HomMat2D);
        HalconCpp::HomMat2dRotate(hv_HomMat2D, hv_Angle[i], 0, 0, &hv_HomMat2D);
        HalconCpp::HomMat2dTranslate(hv_HomMat2D, hv_Row[i], hv_Column[0], &hv_HomMat2D);
        HalconCpp::AffineTransContourXld(ho_ModelContours, &ho_TransContours, hv_HomMat2D);

        // 保存位置，转换为cv::Point2f
        foundPositions.push_back(cv::Point2f(hv_Column[i].D(), hv_Row[i].D()));
    }


    return foundPositions;
}

std::vector<cv::Rect2f> HMatch::fromPointToRect(std::vector<cv::Point2f> points, cv::Rect2f modelRect, double temp)
{
    double width = modelRect.width;
    double height = modelRect.height;
    std::vector<cv::Rect2f> result;
    for (int i = 0; i < points.size(); i++)
    {
        cv::Rect rect((points[i].x - width * 0.5)  * temp, (points[i].y - height * 0.5) * temp, width * temp, height * temp);
        result.push_back(rect);
    }
    return result;
}

int HMatch::getNumberOfPanel(cv::Size imageSize, cv::Rect2f model)
{
    // 定义矩形框的容器
    std::vector<cv::Rect2f> rectangles;

    // 计算水平和垂直方向上能够容纳多少个矩形框
    int horizontalCount = imageSize.width / model.width;
    int verticalCount = imageSize.height / model.height;

    // 计算总共可以容纳的矩形框数
    int totalRectangles = horizontalCount * verticalCount;

    // 绘制矩形框并检查是否与已绘制的矩形框相交
    for (int i = 0; i < totalRectangles; ++i) {
        int x = i % horizontalCount * model.width;
        int y = i / horizontalCount * model.height;
        cv::Rect2f rect(x, y, model.width, model.height);

        // 检查是否与已经绘制的矩形框相交
        bool overlaps = false;
        for (const cv::Rect2f& existingRect : rectangles) {
            if (doRectanglesIntersect(rect, existingRect)) {
                overlaps = true;
                break;
            }
        }

        if (!overlaps) {
            rectangles.push_back(rect);
        }
    }

    return rectangles.size();
}

bool HMatch::doRectanglesIntersect(const cv::Rect2f& rect1, const cv::Rect2f& rect2)
{
    return (rect1 & rect2).area() > 0;
}

std::vector<cv::Point2f> HMatch::HalconMatch_MaskShift(int numberOfPanel, HalconCpp::HObject DMaskImg, HalconCpp::HObject MMaskImg)
{
    // 初始化本地图标变量
    HalconCpp::HObject ho_Image, ho_ModelRegion, ho_TemplateImage;
    HalconCpp::HObject ho_ModelContours, ho_TransContours;

    // 初始化本地控制变量
    HalconCpp::HTuple hv_ModelID, hv_RefRow, hv_RefColumn, hv_HomMat2D;
    HalconCpp::HTuple hv_Row, hv_Column, hv_Angle, hv_Score;

    // 设置系统参数
    HalconCpp::SetSystem("border_shape_models", "false");

    // 使用MMaskImg作为模板图像
    ho_TemplateImage = MMaskImg;
    HalconCpp::HTuple hv_Parameters;
    hv_Parameters.Append(3);
    hv_Parameters.Append(4);
    hv_Parameters.Append(4);
    // 从整个模板图像创建形状模型
    HalconCpp::CreateShapeModel(ho_TemplateImage, 6, HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(0).TupleRad(),
        HalconCpp::HTuple(0).TupleRad(), "point_reduction_high", "use_polarity",
        hv_Parameters, 3, &hv_ModelID);

    // 获取模型轮廓
    HalconCpp::GetShapeModelContours(&ho_ModelContours, hv_ModelID, 1);

    // 从模板图像获取参考位置
    HalconCpp::HTuple hv_Width, hv_Height;

    // 获取模板图像的尺寸
    HalconCpp::GetImageSize(ho_TemplateImage, &hv_Width, &hv_Height);

    // 由于我们假设模板图像的中心为参考点
    hv_RefRow = hv_Height / 2;
    hv_RefColumn = hv_Width / 2;

    HalconCpp::VectorAngleToRigid(0, 0, 0, hv_RefRow, hv_RefColumn, 0, &hv_HomMat2D);
    HalconCpp::AffineTransContourXld(ho_ModelContours, &ho_TransContours, hv_HomMat2D);

    // 匹配过程
    HalconCpp::HTuple hv_RowCheck, hv_ColumnCheck, hv_AngleCheck, hv_ScoreCheck;
    /*
    * DMaskImg:要在其中查找模型的图像
    * hv_ModelID:先前使用 CreateShapeModel 或相关函数创建的形状模型的标识符
    * 0 (AngleStart):搜索模型的起始旋转角度，以弧度为单位。在这个例子中，它被设置为 0，表示没有旋转
    * HalconCpp::HTuple(0).TupleRad() (AngleExtent):搜索模型的旋转角度范围，以弧度为单位。这里设置为 0，意味着不考虑旋转。
    * 0.1 (MinScale):MinScale 是最小缩放比例
    * 0.2 (MaxScale):MaxScale 是最大缩放比例
    * 0.1 (MinScore):接受匹配的最低分数。这是一个介于 0 和 1 之间的值，用于确定匹配的质量。在这个例子中，它被设置为 0.1，表示接受至少 10% 相似度的匹配。
    * least_squares" (SubPixel):这个参数指定了子像素级别的匹配算法。"least_squares" 表示使用最小二乘法，提高匹配位置的精度。
    * 0 (NumMatches):要找到的匹配数量的上限。0 表示没有限制。
    * 0.9 (Greediness):搜索算法的贪婪程度。在这个例子中，它被设置为 0.9，较高的值意味着算法在找到一个良好匹配后不太可能继续寻找其他匹配。
    * hv_RowCheck, hv_ColumnCheck:这两个数组包含了每个找到匹配的中心点的行和列坐标。
    * hv_AngleCheck:包含了每个找到匹配的旋转角度。
    * hv_ScoreCheck:包含了每个找到匹配的分数
    */
    HalconCpp::FindShapeModel(DMaskImg, hv_ModelID, 0, HalconCpp::HTuple(0).TupleRad(), 0.1, 0.2,
        0.1, "least_squares", 0, 0.9, &hv_RowCheck, &hv_ColumnCheck,
        &hv_AngleCheck, &hv_ScoreCheck);
    for (int i = 0; i < hv_RowCheck.Length(); i++)
    {
        std::cout << hv_RowCheck[i].D() << "///" << hv_ColumnCheck[i].D() << std::endl;
    }

    std::vector<cv::Point2f> results;

    // 检查是否找到匹配，并处理结果
    if (hv_ScoreCheck.Length() > 0)
    {
        // 通常，我们会考虑所有找到的匹配项，但这里我们只处理第一个匹配项
        cv::Point2f result;
        result.x = hv_ColumnCheck[0].D();
        result.y = hv_RowCheck[0].D();

        results.push_back(result);
    }

    // 返回匹配点或根据需求返回其他数据
    return results;
}

std::vector<cv::Point2f> HMatch::HalconMatch_Ncc_MaskShift(int numberOfPanel, HalconCpp::HObject DMaskImg, HalconCpp::HObject MMaskImg)
{
    HalconCpp::HObject  ho_ModelRegion, ho_TemplateImage;

    // Local control variables
    HalconCpp::HTuple  hv_ModelID, hv_ModelRegionArea, hv_RefRow;
    HalconCpp::HTuple  hv_RefColumn, hv_Row, hv_Column;
    HalconCpp::HTuple  hv_Angle, hv_Score;

    // 使用MMaskImg作为模板图像
    ho_TemplateImage = MMaskImg;
    // 从整个模板图像创建形状模型模型的图像 (ho_TemplateImage)、金字塔等级 (6)、起始角度 (0 弧度)、结束角度 (360 弧度)、角度步长 (0.1 弧度) 以及 use_polarity 标志
    HalconCpp::CreateNccModel(ho_TemplateImage, "auto", HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(0).TupleRad(),
        "auto", "use_polarity", &hv_ModelID);

    //AreaCenter(ho_ModelRegion, &hv_ModelRegionArea, &hv_RefRow, &hv_RefColumn);

    /***
    **      ModelID（in）：模板句柄
    **      AngleStart（in）：模板的最小旋转
    **      AngleExtent（in）：旋转角度范围
    **      MinScore（in）：被找到的模板最小分数
    **      NumMatches（in）：被找到的模板个数
    **      MaxOverlap（in）：被找到的模板实例最大重叠部分
    **      SubPixel（in）：亚像素级别标志,true, false
    **      NumLevels（in）：金字塔层级数
    **      Row（out）：被找到的模板实例行坐标
    **      Column（out）：被找到的模板实例列坐标
    **      Angle（out）：被找到的模板实例的旋转角度
    **      Score（out）：被找到的模板实例的分数
    ***/
    FindNccModel(DMaskImg, hv_ModelID, HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(0).TupleRad(),
        0.5, numberOfPanel, 0.5, "true", 0, &hv_Row, &hv_Column, &hv_Angle, &hv_Score);

    for (int i = 0; i < hv_Row.Length(); i++)
    {
        std::cout << hv_Row[i].D() << "///" << hv_Column[i].D() << std::endl;
    }

    std::vector<cv::Point2f> results;

    // 检查是否找到匹配，并处理结果
    if (hv_Score.Length() > 0)
    {
        // 通常，我们会考虑所有找到的匹配项，但这里我们只处理第一个匹配项
        cv::Point2f result;
        result.x = hv_Column[0].D();
        result.y = hv_Row[0].D();

        results.push_back(result);
    }

    // 返回匹配点或根据需求返回其他数据
    return results;
}

void HMatch::alignRectangles(std::vector<cv::Rect2f>& rects, cv::Rect2f modelRect)
{
    std::vector<float> recordedX;  // 用于存储之前的x坐标
    std::vector<float> recordedY;  // 用于存储之前的y坐标
    recordedX.push_back(modelRect.x);
    recordedY.push_back(modelRect.y);

    for (auto& rect : rects) {
        bool xAdjusted = false;
        bool yAdjusted = false;

        // 检查x坐标是否接近之前记录的任何x坐标
        for (float a : recordedX) {
            if (std::abs(rect.x - a) <= 100) {
                rect.x = a;  // 调整x坐标
                xAdjusted = true;
                break;  // 找到接近的x坐标后跳出循环
            }
        }

        // 检查y坐标是否接近之前记录的任何y坐标
        for (float a : recordedY) {
            if (std::abs(rect.y - a) <= 100) {
                rect.y = a;  // 调整y坐标
                yAdjusted = true;
                break;  // 找到接近的y坐标后跳出循环
            }
        }

        // 如果没有调整x和y坐标，记录这个框的坐标
        if (!xAdjusted) {
            recordedX.push_back(rect.x);
        }
        if (!yAdjusted) {
            recordedY.push_back(rect.y);
        }
    }
}
