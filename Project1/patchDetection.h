#pragma once
#include <iostream>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

class pathDetection
{
private:
	/*��Ƭ���м����С�ڿ������*/
	bool ifBlackBlockExist(const cv::Mat& img, const cv::Mat& templ, double threshold = 0.2);

	/*��תģ��ͼ*/
	cv::Mat rotateImg(const cv::Mat& src, double angle);
public:

	/*ͨ�����ú�ɫ���ص���ֵ����߿� ��ʱû��*/
	cv::Rect2f getBlackBlock(cv::Mat image);

	/* 
	* �жϸ����Ƕȵ�ģ���Ƿ���ڣ�С�ڿ����ޣ�
	* img:������ͼ
	* temp1:ģ��ͼ
	* angel:����ģ��ͼ��ת�ĽǶȣ�Ĭ�ϼ��ģ��ͼ�ķ������ת��ʮ��
	* threshold:���õļ����ֵ,Ĭ��0.2
	*/
	bool variousAngleExist(const cv::Mat& img, const cv::Mat& templ, double angle,double threshold = 0.2);

	/*
	* �����Ƭ�м�С�ڿ��ƫ�ƽǶ�
	* ���� ��Ƭ�����
	*/
	double detectOffsetAngle(cv::Mat img);

	/*
	* �ж��Ƿ���
	* img:���ͼ
	* temp:ģ��ͼ
	* threshold:��ֵ Ĭ��0.8
	* ����:true or false
	*/
	double ifDetectionWrong(cv::Mat& img, cv::Mat& temp, double threshold = 0.8);

	
};

