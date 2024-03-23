#include "CameraCalibration.h"

caneraCalibration::caneraCalibration()
{
}

caneraCalibration::~caneraCalibration()
{
}

double caneraCalibration::calibration(std::vector<cv::Point2f> physics, std::vector<cv::Point2f> pixel, double& cameraAngle, cv::Matx22d& modelMartrix) {//物理坐标只有算pix的时候用一下
	//std::vector<double> modelCameraAngle = calculateAngle(getMaxAndMin(pixel));
	std::vector<double> modelCameraAngle = calculateAngle(pixel);
	std::cout << modelCameraAngle[0] << "---" << modelCameraAngle[1] << std::endl;
	if (abs(modelCameraAngle[0] - modelCameraAngle[1]) < 0.001)//则视为模组无误差，只需要标定相机
	{
		cameraAngle = modelCameraAngle[0];
		modelMartrix = { 1,0,0,1 };
	}
	else//模组和相机都需要标定一下
	{
		int follow = abs(modelCameraAngle[0]) < abs(modelCameraAngle[1]) ? 0 : 1;//绝对值小的当作矩阵角度
		bool sign = modelCameraAngle[follow] > 0 ? false : true;
		modelMartrix = getConversionMatrix(modelCameraAngle[follow] , sign);
		cameraAngle = modelCameraAngle[1] - modelCameraAngle[0];//小于0 顺时针转
	}
	double invert = getPix(physics,pixel);
	return invert;

}

std::vector<cv::Point2f> caneraCalibration::getMaxAndMin(std::vector<cv::Point2f> temp)
{
	cv::Point2f leftUp, rightDown, other;//other都变成左下角
	
	leftUp.x = std::min({ temp[0].x, temp[1].x, temp[2].x });
	leftUp.y = std::min({ temp[0].y, temp[1].y, temp[2].y });
	rightDown.x = std::max({ temp[0].x, temp[1].x, temp[2].x });
	rightDown.y = std::max({ temp[0].y, temp[1].y, temp[2].y });
	other.x = leftUp.x;
	other.y = rightDown.y;
	std::vector<cv::Point2f> gather;
	gather.push_back(leftUp);
	gather.push_back(rightDown);
	gather.push_back(other);
	return gather;
}

std::vector<double> caneraCalibration::calculateAngle(std::vector<cv::Point2f> temp)
{
	double angle1 = atan((temp[2].x - temp[0].x) / (temp[2].y - temp[0].y));
	double angle2 = atan((temp[1].y - temp[2].y) / (temp[1].x - temp[2].x));
	std::cout << "angle1:" << angle1 << "angle2:" << angle2 << std::endl;
	std::vector<double> modelCameraAngle;
	modelCameraAngle.push_back(angle1);
	modelCameraAngle.push_back(angle2);
	return modelCameraAngle;
}

double caneraCalibration::distance(double x1, double y1, double x2, double y2)
{
	double dx = x2 - x1;
	double dy = y2 - y1;
	return sqrt(dx * dx + dy * dy);
}

cv::Matx22d caneraCalibration::getConversionMatrix(double angle, bool sign)
{
	if (sign == false)
	{
		return cv::Matx22d(1 / cos(angle),0,tan(angle),1);
	}
	else 
	{
		return cv::Matx22d(1 / cos(angle),0,-tan(angle),1);
	}
	
}

double caneraCalibration::getPix(std::vector<cv::Point2f> physics, std::vector<cv::Point2f> pixel)
{
	/*physics = getMaxAndMin(physics);
	pixel = getMaxAndMin(pixel);*/
	double temp1 = distance(physics[0].x, physics[0].y, physics[2].x, physics[2].y) / distance(pixel[0].x, pixel[0].y, pixel[2].x, pixel[2].y);
	double temp2 = distance(physics[1].x, physics[1].y, physics[2].x, physics[2].y) / distance(pixel[1].x, pixel[1].y, pixel[2].x, pixel[2].y); ;
	return (temp1 + temp2) / 2;
}

