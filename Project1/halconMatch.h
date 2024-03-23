#include <halconcpp/HalconCpp.h>
#include "Halcon.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "HalconCpp.h"
#include <iostream>

#define SCALE_FACTOR 500//缩放到500

class HMatch {
public:
    HMatch();
    ~HMatch();
public:
    /*
    * 拼板框算几拼
    * firstRect:手画的第一个拼板框坐标
    * imageSize:大图长宽
    */
    int getNumberOfPanel(cv::Rect2f firstRect, cv::Size2f imageSize);

    /*
    * 得到拼板框的坐标
    * numberOfPanel:拼版的数量 手动输入
    * Image:传入的大图
    * model:传入第一个拼板框的坐标
    * result:返回所有拼板框的坐标
    */
    void getPanelFrames(int numberOfPanel, cv::Mat Image,cv::Rect2f model,std::vector<cv::Rect2f> &result);

    /*
    * 检测过程中mark的匹配算偏移
    * numberOfPanel:返回的数量 手动输入 这个函数只返回一个
    * dMask:传入的运行图(fov大图)
    * mMask:传入的模板图(mask的框图)
    * result:返回运行图中模板图的坐标
    */
    void getMaskShift(int numberOfPanel, cv::Mat dMask, cv::Mat mMask, cv::Rect2f &result );
private:
    //转为单通道
    void singleImage(cv::Mat& Image);
    //压缩图片
    void resizeImage(cv::Mat &Image);
    //用halcon的窗口显示图片
    void dev_open_window_fit_image(HalconCpp::HObject ho_Image, HalconCpp::HTuple hv_Row, HalconCpp::HTuple hv_Column,
            HalconCpp::HTuple hv_WidthLimit, HalconCpp::HTuple hv_HeightLimit, HalconCpp::HTuple* hv_WindowHandle);

    //halcon类型图转mat
    cv::Mat HImageToMat(HalconCpp::HObject& H_img);
    //mat转halcon
    HalconCpp::HObject MatToHImage(cv::Mat & cv_img);
    //根据转化系数转化rect
    void ZoomChange(cv::Rect2f &rect,double temp);
    /*
    * 基于halcon的相关性模板匹配
    * numberOfPanel:拼板框数量
    * Himage:压缩后的大图
    * model:模板位置
    */
    std::vector<cv::Point2f> HalconMatch(int numberOfPanel, HalconCpp::HObject Himage, cv::Rect2f modelRect);
    /*
    * 根据中心点返回rect
    * points : 中心点集合
    * modelRect : 模板rect 用来确定长宽
    * temp : 缩放系数
    */
    std::vector<cv::Rect2f> fromPointToRect(std::vector<cv::Point2f> points, cv::Rect2f modelRect,double temp);

    /*
    * 根据模板框和大图尺寸算出拼板框数量
    * imageSize : 大图尺寸
    * model : 模板框
    * 返回 : 拼版框数量
    */
    int getNumberOfPanel(cv::Size imageSize, cv::Rect2f model);

    //getNumberOfPanel用 判断是否相交
    bool doRectanglesIntersect(const cv::Rect2f& rect1, const cv::Rect2f& rect2);

    /*
    * 基于halcon的相关性模板匹配
    * numberOfPanel:返回的数量 1
    * DMaskImg:运行图
    * MMaskImg:模板图
    */
    std::vector<cv::Point2f> HalconMatch_MaskShift(int numberOfPanel, HalconCpp::HObject DMaskImg, HalconCpp::HObject MMaskImg);

    /*
    * 基于halcon的ncc方法，相关性的方法不稳定 模板匹配
    * numberOfPanel:返回的数量 1
    * DMaskImg:运行图
    * MMaskImg:模板图
    */
    std::vector<cv::Point2f> HalconMatch_Ncc_MaskShift(int numberOfPanel, HalconCpp::HObject DMaskImg, HalconCpp::HObject MMaskImg);
};

