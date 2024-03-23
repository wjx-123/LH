/*fov路径规划*/
//我自己写
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>


class fovRoute
{
public:
	fovRoute();
	~fovRoute();
	std::vector<cv::Rect> coverPointsWithRectangles(const std::vector<cv::Point>& points, int width = 2448, int height = 2048);
private:
	//检查每个可能的新矩形框位置，然后选择与现有矩形框重叠最少的位置。这可以通过计算每个可能位置的矩形框与现有矩形框的重叠面积来实现
	cv::Rect findBestNewRect(const cv::Point& point, int width, int height, const std::vector<cv::Rect>& existingRects);
};