/*fov·���滮*/
//���Լ�д
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
	//���ÿ�����ܵ��¾��ο�λ�ã�Ȼ��ѡ�������о��ο��ص����ٵ�λ�á������ͨ������ÿ������λ�õľ��ο������о��ο���ص������ʵ��
	cv::Rect findBestNewRect(const cv::Point& point, int width, int height, const std::vector<cv::Rect>& existingRects);
};