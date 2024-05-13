#include "halconMatch.h"
HMatch::HMatch()
{
}
HMatch::~HMatch()
{
}
int HMatch::getNumberOfPanel(cv::Rect2f firstRect, cv::Size2f imageSize)
{
    // ����ÿ��ά���������Է��õľ�������
    int numRectsWidth = static_cast<int>(imageSize.width / firstRect.width);
    int numRectsHeight = static_cast<int>(imageSize.height / firstRect.height);

    // �����ܵľ�������
    int totalRects = numRectsWidth * numRectsHeight;
    return totalRects;
}
void HMatch::getPanelFrames(int numberOfPanel, cv::Mat Image, cv::Rect2f model, std::vector<cv::Rect2f>& result)
{
    /*����ncc��ƥ��*/
    float xModel = model.x;
    float yModel = model.y;
    std::cout<< getNumberOfPanel(Image.size(), model) << std::endl;
    double temp = (double)Image.cols / SCALE_FACTOR;//ת��ϵ��
    resizeImage(Image);//ѹ��ͼƬ
    singleImage(Image);//ת��ͨ��
    ZoomChange(model,temp);//����model
    HalconCpp::HObject Himage = MatToHImage(Image);//��ͼmatתhalcon
    std::vector<cv::Point2f> centerPoints = HalconMatch_ncc(numberOfPanel, Himage, model);
    result = fromPointToRect(centerPoints,model,temp);
    alignRectangles(result,model);

    /*����shape��ƥ��*/
    //HalconCpp::HObject HImage = MatToHImage(Image);
    ////HalconCpp::HObject MImage = MatToHImage(modelImg);
    //std::vector<cv::Point2f> centerPoints = HalconMatch_shape(numberOfPanel,HImage,model);
    //result = fromPointToRect(centerPoints, model,1);
}

void HMatch::getMaskShift(int numberOfPanel, cv::Mat dMask, cv::Mat mMask, cv::Rect2f& result)
{
    singleImage(dMask);//ת��ͨ��
    singleImage(mMask);
    HalconCpp::HObject HDMask = MatToHImage(dMask);//��ͼmatתhalcon
    HalconCpp::HObject HMMask = MatToHImage(mMask);//��ͼmatתhalcon

    //std::vector<cv::Point2f> centerPoints = HalconMatch_MaskShift(numberOfPanel, HDMask, HMMask);//���������
    std::vector<cv::Point2f> centerPoints = HalconMatch_Ncc_MaskShift(numberOfPanel, HDMask, HMMask);//����ncc
    cv::Rect2f model = cv::Rect2f(0,0,mMask.cols, mMask.rows);
    auto temp = fromPointToRect(centerPoints, model, 1);
    result = temp[0];
}

std::vector<std::pair<float, float>> HMatch::matchRectangles(const cv::Mat& srcImage, const cv::Rect2f& templateRect, const std::vector<cv::Rect2f>& rectangles, float scale)
{
    std::vector<std::pair<float, float>> offsets;

    // ��ģ��ͼ���������
    cv::Mat templateImage = srcImage(templateRect);
    cv::resize(templateImage, templateImage, cv::Size(), scale, scale, cv::INTER_AREA);

    // �����������������
    auto detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> templateKeyPoints;
    cv::Mat templateDescriptors;
    detector->detectAndCompute(templateImage, cv::noArray(), templateKeyPoints, templateDescriptors);

    // ����ƥ����
    cv::BFMatcher matcher(cv::NORM_L2);

    // ��������
    for (const auto& rect : rectangles) {
        cv::Mat rectImage = srcImage(rect);
        cv::resize(rectImage, rectImage, cv::Size(), scale, scale, cv::INTER_AREA);  // ��ÿ������ͼ���������

        std::vector<cv::KeyPoint> rectKeyPoints;
        cv::Mat rectDescriptors;
        detector->detectAndCompute(rectImage, cv::noArray(), rectKeyPoints, rectDescriptors);

        // ����ƥ��
        std::vector<cv::DMatch> matches;
        matcher.match(templateDescriptors, rectDescriptors, matches);

        // ɸѡ�õ�ƥ��
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

        // ���Ʊ任����
        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;
        for (size_t i = 0; i < good_matches.size(); i++) {
            obj.push_back(templateKeyPoints[good_matches[i].queryIdx].pt);
            scene.push_back(rectKeyPoints[good_matches[i].trainIdx].pt);
        }
        cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);

        // ����ƫ��
        if (!H.empty()) {
            double dx = H.at<double>(0, 2) / scale; // ������������
            double dy = H.at<double>(1, 2) / scale; // ������������
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
        // ת��Ϊ�Ҷ�ͼ�񣨵�ͨ����
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
    //Matching 01: Find the modelImage��in������ͨ��ͼ����������ɱ�����Ϊģ��
    /***
    **      ModelID��in����ģ����         
    **      AngleStart��in����ģ�����С��ת        
    **      AngleExtent��in������ת�Ƕȷ�Χ          
    **      MinScore��in�������ҵ���ģ����С����
    **      NumMatches��in�������ҵ���ģ�����          
    **      MaxOverlap��in�������ҵ���ģ��ʵ������ص�����           
    **      SubPixel��in���������ؼ����־,true, false     
    **      NumLevels��in�����������㼶��          
    **      Row��out�������ҵ���ģ��ʵ��������
    **      Column��out�������ҵ���ģ��ʵ��������       
    **      Angle��out�������ҵ���ģ��ʵ������ת�Ƕ�        
    **      Score��out�������ҵ���ģ��ʵ���ķ���
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
    // ����ͼ�����
    HalconCpp::HObject ho_ModelRegion, ho_TemplateImage, ho_ModelContours, ho_TransContours;
    HalconCpp::HTuple hv_ModelID, hv_Angle, hv_Row, hv_Column, hv_Score, hv_HomMat2D;
    HalconCpp::SetSystem("border_shape_models", "false");
    // ʹ�� OpenCV ���β����� Halcon ������ģ������
    HalconCpp::GenRectangle1(&ho_ModelRegion, modelRect.y, modelRect.x,
        modelRect.y + modelRect.height, modelRect.x + modelRect.width);

    // ��ȡģ��ͼ��
    HalconCpp::ReduceDomain(Himage, ho_ModelRegion, &ho_TemplateImage);

    try
    {
        // ������״ģ��
        CreateShapeModel(ho_TemplateImage, 6, HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(360).TupleRad(),
            HalconCpp::HTuple(0.028).TupleRad(), (HalconCpp::HTuple("point_reduction_high").Append("no_pregeneration")),
            "use_polarity", ((HalconCpp::HTuple(78).Append(136)).Append(51)), 4, &hv_ModelID);
    }
    catch (HalconCpp::HException& e) {
        std::cerr << "Error in CreateShapeModel: " << e.ErrorMessage() << std::endl;
    }
   

    // ��ȡģ������
    HalconCpp::GetShapeModelContours(&ho_ModelContours, hv_ModelID, 1);

    // ׼�����صĽ����ʹ��cv::Point2f
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
    //        // ת��ģ����������⵽��λ��
    //    HalconCpp::HomMat2dIdentity(&hv_HomMat2D);
    //    HalconCpp::HomMat2dRotate(hv_HomMat2D, hv_Angle[0], 0, 0, &hv_HomMat2D);
    //    HalconCpp::HomMat2dTranslate(hv_HomMat2D, hv_Row[0], hv_Column[0], &hv_HomMat2D);
    //    HalconCpp::AffineTransContourXld(ho_ModelContours, &ho_TransContours, hv_HomMat2D);

    //    // ����λ�ã�ת��Ϊcv::Point2f
    //    foundPositions.push_back(cv::Point2f(hv_Column[0].D(), hv_Row[0].D()));
    //}
    for (int i = 0; i < hv_Row.Length(); i++)
    {
        HalconCpp::HomMat2dIdentity(&hv_HomMat2D);
        HalconCpp::HomMat2dRotate(hv_HomMat2D, hv_Angle[i], 0, 0, &hv_HomMat2D);
        HalconCpp::HomMat2dTranslate(hv_HomMat2D, hv_Row[i], hv_Column[0], &hv_HomMat2D);
        HalconCpp::AffineTransContourXld(ho_ModelContours, &ho_TransContours, hv_HomMat2D);

        // ����λ�ã�ת��Ϊcv::Point2f
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
    // ������ο������
    std::vector<cv::Rect2f> rectangles;

    // ����ˮƽ�ʹ�ֱ�������ܹ����ɶ��ٸ����ο�
    int horizontalCount = imageSize.width / model.width;
    int verticalCount = imageSize.height / model.height;

    // �����ܹ��������ɵľ��ο���
    int totalRectangles = horizontalCount * verticalCount;

    // ���ƾ��ο򲢼���Ƿ����ѻ��Ƶľ��ο��ཻ
    for (int i = 0; i < totalRectangles; ++i) {
        int x = i % horizontalCount * model.width;
        int y = i / horizontalCount * model.height;
        cv::Rect2f rect(x, y, model.width, model.height);

        // ����Ƿ����Ѿ����Ƶľ��ο��ཻ
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
    // ��ʼ������ͼ�����
    HalconCpp::HObject ho_Image, ho_ModelRegion, ho_TemplateImage;
    HalconCpp::HObject ho_ModelContours, ho_TransContours;

    // ��ʼ�����ؿ��Ʊ���
    HalconCpp::HTuple hv_ModelID, hv_RefRow, hv_RefColumn, hv_HomMat2D;
    HalconCpp::HTuple hv_Row, hv_Column, hv_Angle, hv_Score;

    // ����ϵͳ����
    HalconCpp::SetSystem("border_shape_models", "false");

    // ʹ��MMaskImg��Ϊģ��ͼ��
    ho_TemplateImage = MMaskImg;
    HalconCpp::HTuple hv_Parameters;
    hv_Parameters.Append(3);
    hv_Parameters.Append(4);
    hv_Parameters.Append(4);
    // ������ģ��ͼ�񴴽���״ģ��
    HalconCpp::CreateShapeModel(ho_TemplateImage, 6, HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(0).TupleRad(),
        HalconCpp::HTuple(0).TupleRad(), "point_reduction_high", "use_polarity",
        hv_Parameters, 3, &hv_ModelID);

    // ��ȡģ������
    HalconCpp::GetShapeModelContours(&ho_ModelContours, hv_ModelID, 1);

    // ��ģ��ͼ���ȡ�ο�λ��
    HalconCpp::HTuple hv_Width, hv_Height;

    // ��ȡģ��ͼ��ĳߴ�
    HalconCpp::GetImageSize(ho_TemplateImage, &hv_Width, &hv_Height);

    // �������Ǽ���ģ��ͼ�������Ϊ�ο���
    hv_RefRow = hv_Height / 2;
    hv_RefColumn = hv_Width / 2;

    HalconCpp::VectorAngleToRigid(0, 0, 0, hv_RefRow, hv_RefColumn, 0, &hv_HomMat2D);
    HalconCpp::AffineTransContourXld(ho_ModelContours, &ho_TransContours, hv_HomMat2D);

    // ƥ�����
    HalconCpp::HTuple hv_RowCheck, hv_ColumnCheck, hv_AngleCheck, hv_ScoreCheck;
    /*
    * DMaskImg:Ҫ�����в���ģ�͵�ͼ��
    * hv_ModelID:��ǰʹ�� CreateShapeModel ����غ�����������״ģ�͵ı�ʶ��
    * 0 (AngleStart):����ģ�͵���ʼ��ת�Ƕȣ��Ի���Ϊ��λ������������У���������Ϊ 0����ʾû����ת
    * HalconCpp::HTuple(0).TupleRad() (AngleExtent):����ģ�͵���ת�Ƕȷ�Χ���Ի���Ϊ��λ����������Ϊ 0����ζ�Ų�������ת��
    * 0.1 (MinScale):MinScale ����С���ű���
    * 0.2 (MaxScale):MaxScale ��������ű���
    * 0.1 (MinScore):����ƥ�����ͷ���������һ������ 0 �� 1 ֮���ֵ������ȷ��ƥ�������������������У���������Ϊ 0.1����ʾ�������� 10% ���ƶȵ�ƥ�䡣
    * least_squares" (SubPixel):�������ָ���������ؼ����ƥ���㷨��"least_squares" ��ʾʹ����С���˷������ƥ��λ�õľ��ȡ�
    * 0 (NumMatches):Ҫ�ҵ���ƥ�����������ޡ�0 ��ʾû�����ơ�
    * 0.9 (Greediness):�����㷨��̰���̶ȡ�����������У���������Ϊ 0.9���ϸߵ�ֵ��ζ���㷨���ҵ�һ������ƥ���̫���ܼ���Ѱ������ƥ�䡣
    * hv_RowCheck, hv_ColumnCheck:���������������ÿ���ҵ�ƥ������ĵ���к������ꡣ
    * hv_AngleCheck:������ÿ���ҵ�ƥ�����ת�Ƕȡ�
    * hv_ScoreCheck:������ÿ���ҵ�ƥ��ķ���
    */
    HalconCpp::FindShapeModel(DMaskImg, hv_ModelID, 0, HalconCpp::HTuple(0).TupleRad(), 0.1, 0.2,
        0.1, "least_squares", 0, 0.9, &hv_RowCheck, &hv_ColumnCheck,
        &hv_AngleCheck, &hv_ScoreCheck);
    for (int i = 0; i < hv_RowCheck.Length(); i++)
    {
        std::cout << hv_RowCheck[i].D() << "///" << hv_ColumnCheck[i].D() << std::endl;
    }

    std::vector<cv::Point2f> results;

    // ����Ƿ��ҵ�ƥ�䣬��������
    if (hv_ScoreCheck.Length() > 0)
    {
        // ͨ�������ǻῼ�������ҵ���ƥ�������������ֻ�����һ��ƥ����
        cv::Point2f result;
        result.x = hv_ColumnCheck[0].D();
        result.y = hv_RowCheck[0].D();

        results.push_back(result);
    }

    // ����ƥ����������󷵻���������
    return results;
}

std::vector<cv::Point2f> HMatch::HalconMatch_Ncc_MaskShift(int numberOfPanel, HalconCpp::HObject DMaskImg, HalconCpp::HObject MMaskImg)
{
    HalconCpp::HObject  ho_ModelRegion, ho_TemplateImage;

    // Local control variables
    HalconCpp::HTuple  hv_ModelID, hv_ModelRegionArea, hv_RefRow;
    HalconCpp::HTuple  hv_RefColumn, hv_Row, hv_Column;
    HalconCpp::HTuple  hv_Angle, hv_Score;

    // ʹ��MMaskImg��Ϊģ��ͼ��
    ho_TemplateImage = MMaskImg;
    // ������ģ��ͼ�񴴽���״ģ��ģ�͵�ͼ�� (ho_TemplateImage)���������ȼ� (6)����ʼ�Ƕ� (0 ����)�������Ƕ� (360 ����)���ǶȲ��� (0.1 ����) �Լ� use_polarity ��־
    HalconCpp::CreateNccModel(ho_TemplateImage, "auto", HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(0).TupleRad(),
        "auto", "use_polarity", &hv_ModelID);

    //AreaCenter(ho_ModelRegion, &hv_ModelRegionArea, &hv_RefRow, &hv_RefColumn);

    /***
    **      ModelID��in����ģ����
    **      AngleStart��in����ģ�����С��ת
    **      AngleExtent��in������ת�Ƕȷ�Χ
    **      MinScore��in�������ҵ���ģ����С����
    **      NumMatches��in�������ҵ���ģ�����
    **      MaxOverlap��in�������ҵ���ģ��ʵ������ص�����
    **      SubPixel��in���������ؼ����־,true, false
    **      NumLevels��in�����������㼶��
    **      Row��out�������ҵ���ģ��ʵ��������
    **      Column��out�������ҵ���ģ��ʵ��������
    **      Angle��out�������ҵ���ģ��ʵ������ת�Ƕ�
    **      Score��out�������ҵ���ģ��ʵ���ķ���
    ***/
    FindNccModel(DMaskImg, hv_ModelID, HalconCpp::HTuple(0).TupleRad(), HalconCpp::HTuple(0).TupleRad(),
        0.5, numberOfPanel, 0.5, "true", 0, &hv_Row, &hv_Column, &hv_Angle, &hv_Score);

    for (int i = 0; i < hv_Row.Length(); i++)
    {
        std::cout << hv_Row[i].D() << "///" << hv_Column[i].D() << std::endl;
    }

    std::vector<cv::Point2f> results;

    // ����Ƿ��ҵ�ƥ�䣬��������
    if (hv_Score.Length() > 0)
    {
        // ͨ�������ǻῼ�������ҵ���ƥ�������������ֻ�����һ��ƥ����
        cv::Point2f result;
        result.x = hv_Column[0].D();
        result.y = hv_Row[0].D();

        results.push_back(result);
    }

    // ����ƥ����������󷵻���������
    return results;
}

void HMatch::alignRectangles(std::vector<cv::Rect2f>& rects, cv::Rect2f modelRect)
{
    std::vector<float> recordedX;  // ���ڴ洢֮ǰ��x����
    std::vector<float> recordedY;  // ���ڴ洢֮ǰ��y����
    recordedX.push_back(modelRect.x);
    recordedY.push_back(modelRect.y);

    for (auto& rect : rects) {
        bool xAdjusted = false;
        bool yAdjusted = false;

        // ���x�����Ƿ�ӽ�֮ǰ��¼���κ�x����
        for (float a : recordedX) {
            if (std::abs(rect.x - a) <= 100) {
                rect.x = a;  // ����x����
                xAdjusted = true;
                break;  // �ҵ��ӽ���x���������ѭ��
            }
        }

        // ���y�����Ƿ�ӽ�֮ǰ��¼���κ�y����
        for (float a : recordedY) {
            if (std::abs(rect.y - a) <= 100) {
                rect.y = a;  // ����y����
                yAdjusted = true;
                break;  // �ҵ��ӽ���y���������ѭ��
            }
        }

        // ���û�е���x��y���꣬��¼����������
        if (!xAdjusted) {
            recordedX.push_back(rect.x);
        }
        if (!yAdjusted) {
            recordedY.push_back(rect.y);
        }
    }
}
