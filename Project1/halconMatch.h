#include <halconcpp/HalconCpp.h>
#include "Halcon.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "HalconCpp.h"
#include <iostream>

#define SCALE_FACTOR 500//���ŵ�500

class HMatch {
public:
    HMatch();
    ~HMatch();
public:
    /*
    * ƴ����㼸ƴ
    * firstRect:�ֻ��ĵ�һ��ƴ�������
    * imageSize:��ͼ����
    */
    int getNumberOfPanel(cv::Rect2f firstRect, cv::Size2f imageSize);

    /*
    * �õ�ƴ��������
    * numberOfPanel:ƴ������� �ֶ�����
    * Image:����Ĵ�ͼ
    * model:�����һ��ƴ��������
    * result:��������ƴ��������
    */
    void getPanelFrames(int numberOfPanel, cv::Mat Image,cv::Rect2f model,std::vector<cv::Rect2f> &result);

    /*
    * ��������mark��ƥ����ƫ��
    * numberOfPanel:���ص����� �ֶ����� �������ֻ����һ��
    * dMask:���������ͼ(fov��ͼ)
    * mMask:�����ģ��ͼ(mask�Ŀ�ͼ)
    * result:��������ͼ��ģ��ͼ������
    */
    void getMaskShift(int numberOfPanel, cv::Mat dMask, cv::Mat mMask, cv::Rect2f &result );
private:
    //תΪ��ͨ��
    void singleImage(cv::Mat& Image);
    //ѹ��ͼƬ
    void resizeImage(cv::Mat &Image);
    //��halcon�Ĵ�����ʾͼƬ
    void dev_open_window_fit_image(HalconCpp::HObject ho_Image, HalconCpp::HTuple hv_Row, HalconCpp::HTuple hv_Column,
            HalconCpp::HTuple hv_WidthLimit, HalconCpp::HTuple hv_HeightLimit, HalconCpp::HTuple* hv_WindowHandle);

    //halcon����ͼתmat
    cv::Mat HImageToMat(HalconCpp::HObject& H_img);
    //matתhalcon
    HalconCpp::HObject MatToHImage(cv::Mat & cv_img);
    //����ת��ϵ��ת��rect
    void ZoomChange(cv::Rect2f &rect,double temp);
    /*
    * ����halcon�������ģ��ƥ��
    * numberOfPanel:ƴ�������
    * Himage:ѹ����Ĵ�ͼ
    * model:ģ��λ��
    */
    std::vector<cv::Point2f> HalconMatch(int numberOfPanel, HalconCpp::HObject Himage, cv::Rect2f modelRect);
    /*
    * �������ĵ㷵��rect
    * points : ���ĵ㼯��
    * modelRect : ģ��rect ����ȷ������
    * temp : ����ϵ��
    */
    std::vector<cv::Rect2f> fromPointToRect(std::vector<cv::Point2f> points, cv::Rect2f modelRect,double temp);

    /*
    * ����ģ���ʹ�ͼ�ߴ����ƴ�������
    * imageSize : ��ͼ�ߴ�
    * model : ģ���
    * ���� : ƴ�������
    */
    int getNumberOfPanel(cv::Size imageSize, cv::Rect2f model);

    //getNumberOfPanel�� �ж��Ƿ��ཻ
    bool doRectanglesIntersect(const cv::Rect2f& rect1, const cv::Rect2f& rect2);

    /*
    * ����halcon�������ģ��ƥ��
    * numberOfPanel:���ص����� 1
    * DMaskImg:����ͼ
    * MMaskImg:ģ��ͼ
    */
    std::vector<cv::Point2f> HalconMatch_MaskShift(int numberOfPanel, HalconCpp::HObject DMaskImg, HalconCpp::HObject MMaskImg);

    /*
    * ����halcon��ncc����������Եķ������ȶ� ģ��ƥ��
    * numberOfPanel:���ص����� 1
    * DMaskImg:����ͼ
    * MMaskImg:ģ��ͼ
    */
    std::vector<cv::Point2f> HalconMatch_Ncc_MaskShift(int numberOfPanel, HalconCpp::HObject DMaskImg, HalconCpp::HObject MMaskImg);
};

