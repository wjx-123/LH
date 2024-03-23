#include "splicingBoard.h"

cv::Mat splicBoard::matchingSplic(cv::Mat image, std::vector<cv::Rect2f> rect)
{
	cv::Mat temp = cv::imread("C:\\Users\\LENOVO\\Pictures\\temp2.bmp");

	//nms
	cv::Mat result;
	cv::matchTemplate(image, temp, result, cv::TM_SQDIFF);
	nms_temp_min(image, result, temp.size(), true, 0);

	//ColorMatch_TM_SQDIFF(image,temp);

	//这一段是单目标匹配
	//cv::Mat result;
	//cv::matchTemplate(image, temp, result, cv::TM_CCOEFF_NORMED);

	//double maxVal, minVal;
	//cv::Point minLoc, maxLoc;
	////寻找匹配结果中的最大值和最小值以及坐标位置
	//minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	//// 绘制最佳匹配区域
	//cv::rectangle(image, cv::Rect(maxLoc.x, maxLoc.y, temp.cols, temp.rows), cv::Scalar(0, 0, 255), 2);

	//cv::imshow("原图", image);
	//cv::imshow("模板图", temp);
	//cv::imshow("结果图", result);
	//cv::waitKey(0);
	//cv::destroyAllWindows();

	return cv::Mat();
}

std::vector<cv::Rect2f> splicBoard::linesMatchSplic(cv::Mat image1, std::vector<cv::Rect2f> rect)
{
	// 读取图片
	cv::Mat image = cv::imread("C:\\Users\\LENOVO\\Pictures\\pinban.bmp", cv::IMREAD_GRAYSCALE);

	// 对图片进行预处理，例如边缘检测
	cv::Mat edges;  
	cv::Canny(image, edges, 100, 200);
	cv::namedWindow("222",0);
	cv::imshow("222", edges);
	cv::waitKey(0);
	// 进行直线检测
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(edges, lines, 1, CV_PI / 180, 100);

	// 绘制检测到的直线
	cv::Mat result(image.size(), CV_8UC3);
	cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0];
		float theta = lines[i][1];
		double a = cos(theta);
		double b = sin(theta);
		double x0 = a * rho;
		double y0 = b * rho;
		cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
		cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
		if (std::abs(pt1.x + pt2.x) < 10)
		{
			std::cout << pt1 << "__" << pt2 << std::endl;
			cv::line(result, pt1, pt2, cv::Scalar(0, 0, 255), 2);
		}
		
	}

	// 显示结果
	cv::namedWindow("111",0);
	cv::imshow("111", result);
	cv::waitKey(0);
	return std::vector<cv::Rect2f>();
}

//平方差匹配法  速度太慢 
void splicBoard::ColorMatch_TM_SQDIFF(const cv::Mat& img, const cv::Mat& templ) {
	//判断图像是否存在
	if (img.empty() || templ.empty()) {
		std::cout << "Error: Input image or template is empty." << std::endl;
		return;
	}
	//判断图像是否是彩色图像
	if (img.channels() != 3 || templ.channels() != 3) {
		std::cout << "Error: Both input image and template must be color images." << std::endl;
		return;
	}

	int result_cols = img.cols - templ.cols + 1;//移动的列数
	int result_rows = img.rows - templ.rows + 1;//移动的行数

	cv::Mat result(result_rows, result_cols, CV_32FC1);//存储平方差结果     CV_32FC1 32bites   F 单精度浮点数  C1 通道数1
	result.setTo(cv::Scalar::all(0));//初始化为0

	for (int y = 0; y < result_rows; ++y) {
		for (int x = 0; x < result_cols; ++x) {
			float sqdiff = 0.0f;

			for (int j = 0; j < templ.rows; ++j) {
				for (int i = 0; i < templ.cols; ++i) {
					cv::Vec3b img_pixel = img.at<cv::Vec3b>(y + j, x + i);//Vec3b 长度为3的unchar类型向量
					cv::Vec3b templ_pixel = templ.at <cv::Vec3b>(j, i);

					for (int k = 0; k < 3; ++k) {
						float diff = static_cast<float>(img_pixel[k]) - static_cast<float>(templ_pixel[k]);//计算平方差
						sqdiff += diff * diff;//累加
					}
				}
			}

			result.at<float>(y, x) = sqdiff;//记录结果
		}
	}
	double minVal;//最小值
	cv::Point minLoc;//最小值所在位置
	minMaxLoc(result, &minVal, nullptr, &minLoc, nullptr);//在数组中寻找全局最大最小值

	cv::Mat img_display;
	img.copyTo(img_display);
	rectangle(img_display, minLoc, cv::Point(minLoc.x + templ.cols, minLoc.y + templ.rows), cv::Scalar(0, 255, 0), 2);
	imshow("SQDIFF", img_display);
	cv::waitKey(0);
}

bool splicBoard::comp(nms_data_struct data1, nms_data_struct data2)
{
	return data1.val < data2.val;
}

int splicBoard::nms_detect(nms_data_struct data, std::vector<nms_data_struct> data_list)
{
	for (int i = 0; i < data_list.size(); i++)
	{
		cv::Rect rect = data.roi & data_list[i].roi;
		if (rect.width || rect.height)
			return false;
	}
	return true;
}

int splicBoard::nms_temp_min(cv::Mat& input, cv::Mat result, cv::Size temp_size, bool save_output, float thr)
{
	// get minVal vector and minLoc vector
	std::vector<nms_data_struct> nms_data;
	for (int i = 0; i < result.cols; i++)
	{
		for (int j = 0; j < result.rows; j++)
		{
			if (result.at<float>(j, i) <= thr)
			{
				nms_data_struct data;
				data.val = result.at<float>(j, i);
				data.loc = cv::Point(i, j);
				data.getRect(temp_size);
				nms_data.push_back(data);
			}
		}
	}

	//Sort from smallest to largest
	std::sort(nms_data.begin(), nms_data.end(), comp);

	//nms
	std::vector<nms_data_struct> nms_output_list;
	for (int i = 0; i < nms_data.size(); i++)
	{
		if (i == 0)
		{
			nms_data_struct data;
			data.val = nms_data[i].val;
			data.loc = nms_data[i].loc;
			data.roi = nms_data[i].roi;
			nms_output_list.push_back(data);
			continue;
		}

		int ret = nms_detect(nms_data[i], nms_output_list);
		if (ret)
		{
			nms_data_struct data;
			data.val = nms_data[i].val;
			data.loc = nms_data[i].loc;
			data.getRect(temp_size);
			nms_output_list.push_back(data);
		}
	}

	//save output in current path
	if (save_output)
	{
		for (int i = 0; i < nms_output_list.size(); i++)
		{
			cv::rectangle(input, nms_output_list[i].roi, cv::Scalar(0, 255, 0));
			cv::Size text_size = cv::getTextSize(std::to_string(nms_output_list[i].val), cv::FONT_HERSHEY_COMPLEX, 1.0f, 1, (int*)int());
			cv::rectangle(input, cv::Rect(nms_output_list[i].loc.x, nms_output_list[i].loc.y - text_size.height, text_size.width, \
				text_size.height), cv::Scalar(0, 255, 0), -1);
			cv::putText(input, std::to_string(nms_output_list[i].val), nms_output_list[i].loc, cv::FONT_HERSHEY_COMPLEX, 1.0f, \
				cv::Scalar(0, 0, 0));

		}
		//cv::imwrite("./nms_output_TM_SQDIFF2.jpg", input);
		cv::namedWindow("res",0);
		cv::imshow("res", input);
	    cv::waitKey(0);
		cv::destroyAllWindows();
	}

	return 0;
}