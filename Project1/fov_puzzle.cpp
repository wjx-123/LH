#include"fov_puzzle.h"
////��Ϊ�����и����������������չһ�����룬����ͼƬ��Ӧ�����⣬�÷�����ʱ����
//cv::Mat fov_scan::fov_puzzle(std::vector<cv::Rect2f> fov, std::vector<cv::Mat> fov_img)
//{
//	cv::Size smallSize;
//	smallSize.height = 0;
//	smallSize.width = 0;
//	//����һ�³��Ϳ�����ֵ
//	for (int i = 0; i < fov.size(); i++)
//	{
//		if (fov[i].x + fov[i].width > smallSize.width)
//		{
//			smallSize.width = fov[i].x + fov[i].width;
//		}
//		if (fov[i].y + fov[i].height > smallSize.height)
//		{
//			smallSize.height = fov[i].y + fov[i].height;
//		}
//	}
//	smallSize.height += 2048;
//	smallSize.width += 2448;
//	cv::Mat background = cv::Mat(smallSize, CV_8UC3, cv::Scalar(0, 0, 0));// ������ͨ����ɫͼ��
//	for (int i = 0; i < fov.size(); i++)
//	{
//		if (fov_img[i].channels() != 3) {
//			std::cout << "The image is not three channel:" << fov_img[i].channels() << std::endl;
//
//			/*cv::Mat three_channel = cv::Mat::zeros(fov_img[i].rows, fov_img[i].cols, CV_8UC3);
//			std::vector<cv::Mat> channels;
//			for (int j = 0; j < 3; j++)
//			{
//				channels.push_back(fov_img[i]);
//			}
//			merge(channels, three_channel);*/
//		}
//		cv::Rect2f rectRoi = cv::Rect2f( fov[i].x + 0.5 * fov[i].width , fov[i].y + 0.5 * fov[i].height,fov[i].width,fov[i].height);
//		// ��Сͼ���Ƶ���ͼ��ָ��λ��
//		cv::Mat imageROI = fov_img[i];//�õ�Сͼ
//		cv::flip(imageROI, imageROI, 1);
//		imageROI.copyTo(background(rectRoi));
//	}
//	cv::flip(background, background, 1);
//	cv::flip(background, background, -1);
//
//	cv::Rect2f resultRoi = cv::Rect2f(1224,1024,smallSize.width - 1224,smallSize.height - 1024);
//	cv::Mat resultBackground = background(resultRoi).clone();
//	return resultBackground;
//}



//�������Ѹ����������Ϊ0
cv::Mat fov_scan::fov_puzzle(std::vector<cv::Rect2f> fov, std::vector<cv::Mat> fov_img, cv::Mat initImage)
{
	//std::vector<cv::Rect2f> new_fov = fovCoordinateSymmetryPix(fov, initImage);
	fov = fovCoordinateSymmetryPix(fov,initImage);
	cv::Size smallSize;
	smallSize.height = 0;
	smallSize.width = 0;
	//����һ�³��Ϳ�����ֵ
	for (int i = 0; i < fov.size(); i++)
	{
		if (fov[i].x + fov[i].width > smallSize.width)
		{
			smallSize.width = fov[i].x + fov[i].width;
		}
		if (fov[i].y + fov[i].height > smallSize.height)
		{
			smallSize.height = fov[i].y + fov[i].height;
		}
	}
	cv::Mat background = cv::Mat(smallSize, CV_8UC3, cv::Scalar(0, 0, 0));// ������ͨ����ɫͼ��
	for (int i = 0; i < fov.size(); i++)
	{
		if (fov[i].x < 0 || fov[i].y < 0)
		{
			/*fov_img[i] = fov_img[i](cv::Rect(fov[i].x > 0 ? fov[i].x : 0, 
											 fov[i].y > 0 ? fov[i].y : 0,
											 fov[i].x > 0 ? fov[i].width : fov[i].width - std::abs(fov[i].x),
											 fov[i].y > 0 ? fov[i].height : fov[i].height - std::abs(fov[i].y))).clone();*/
			fov_img[i] = fov_img[i](cv::Rect(fov[i].x > 0 ? 0 : std::abs(fov[i].x),
										  fov[i].y > 0 ? 0 : std::abs(fov[i].y),
										  fov[i].x > 0 ? fov[i].width : fov[i].width - std::abs(fov[i].x),
										  fov[i].y > 0 ? fov[i].height : fov[i].height - std::abs(fov[i].y))).clone();
		}
		cv::Rect2f rectRoi = cv::Rect2f(fov[i].x > 0 ? fov[i].x : 0,
										fov[i].y > 0 ? fov[i].y : 0,
										fov[i].x > 0 ? fov[i].width : fov[i].width - std::abs(fov[i].x),
										fov[i].y > 0 ? fov[i].height : fov[i].height - std::abs(fov[i].y));
		// ��Сͼ���Ƶ���ͼ��ָ��λ��
		cv::Mat imageROI = fov_img[i];//�õ�Сͼ
		cv::flip(imageROI, imageROI, 1);
		imageROI.copyTo(background(rectRoi));
	}
	cv::flip(background, background, 1);
	cv::flip(background, background, -1);

	/*cv::Rect2f resultRoi = cv::Rect2f(1224, 1024, smallSize.width - 1224, smallSize.height - 1024);
	cv::Mat resultBackground = background(resultRoi).clone();*/
	return background;
}


//std::vector<cv::Rect2f> fov_scan::fovCoordinateSymmetry(std::vector<cv::Rect2f> fov, cv::Mat entireImage)
//{
//	//float tempY = entireImage.rows * 0.5;//ˮƽ��
//	for (int i = 0; i < fov.size(); i++)
//	{
//		float symY = entireImage.rows - fov[i].y;
//		fov[i].y = symY;
//	}
//	return fov;
//}

std::vector<cv::Point2f> fov_scan::fovCoordinateSymmetry(std::vector<cv::Point2f> fov, cv::Point2f qishidian, cv::Point2f zhongzhidian)
{
	//float tempY = entireImage.rows * 0.5;//ˮƽ��
	//float tempY = (qishidian.y - zhongzhidian.y) * 0.5;
	for (int i = 0; i < fov.size(); i++)
	{
		float symY = (qishidian.y + zhongzhidian.y) - fov[i].y;
		fov[i].y = symY;
	}
	return fov;
}

std::vector<cv::Rect2f> fov_scan::fovCoordinateSymmetryPix(std::vector<cv::Rect2f> fov, cv::Mat initImage)
{
	for (int i = 0; i < fov.size(); i++)
	{
		float symY = initImage.rows - fov[i].y;
		fov[i].y = symY;
	}
	return fov;
}

cv::Point2f fov_scan::getCircleCenter(cv::Mat )
{
	// �����񻯺ˣ�����ֵ������Ҫ����1����������ͨ������������Χ�Ǹ���
	cv::Mat sharpeningKernel = (cv::Mat_<double>(3, 3) <<
		-1, -1, -1,
		-1, 9, -1,
		-1, -1, -1);
	// ����ͼ��
	cv::Mat image = cv::imread("C:\\Users\\LENOVO\\Pictures\\smt\\mask4.jpg", cv::IMREAD_COLOR);

	// ת��Ϊ�Ҷ�ͼ��
	cv::Mat gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

	// Ӧ���񻯺�
	cv::Mat sharpenedImage;
	cv::filter2D(gray, sharpenedImage, image.depth(), sharpeningKernel);

	// ����Բ�任���Բ
	std::vector<cv::Vec3f> circles;
	HoughCircles(sharpenedImage, circles, cv::HOUGH_GRADIENT, 1, gray.rows / 2, 200, 50, 0, 0);

	// ��ʼ�����뾶�Ͷ�Ӧ��Բ
	int maxRadius = 0;
	cv::Vec3i maxCircle;

	// �ҵ��뾶����Բ
	for (size_t i = 0; i < circles.size(); i++) {
		cv::Vec3i c = circles[i];
		int radius = c[2];
		if (radius > maxRadius) {
			maxRadius = radius;
			maxCircle = c;
		}
	}

	cv::Point center = cv::Point(maxCircle[0], maxCircle[1]);
	// ����Բ��
	circle(image, center, 1, cv::Scalar(0, 100, 100), 3, cv::LINE_AA);
	// ����Բ����
	circle(image, center, maxRadius, cv::Scalar(255, 0, 255), 3, cv::LINE_AA);

	return center;
}

cv::Mat fov_scan::getShiftImage(cv::Mat &iniImg, int shiftX, int shiftY)
{
	cv::Mat shiftedImg = cv::Mat::zeros(iniImg.size(), iniImg.type());

	// ������ɫ�߽�
	shiftedImg.setTo(cv::Scalar(0, 0, 255));

	// ����ͼ��ƫ�Ƶ�Ŀ������
	cv::Rect srcRect = cv::Rect(
		std::max(-shiftX, 0),
		std::max(-shiftY, 0),
		iniImg.cols - abs(shiftX),
		iniImg.rows - abs(shiftY)
	);

	cv::Rect dstRect = cv::Rect(
		std::max(shiftX, 0),
		std::max(shiftY, 0),
		srcRect.width,
		srcRect.height
	);

	// ��ԭʼͼ���һ���ָ��Ƶ���λ��
	iniImg(srcRect).copyTo(shiftedImg(dstRect));

	return shiftedImg;
}

cv::Point2f fov_scan::getSHift(cv::Mat mMask, cv::Mat dMask)
{
	// ��ʼ��ORB�����
	cv::Ptr<cv::ORB> detector = cv::ORB::create();
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;

	// ���ؼ���
	detector->detectAndCompute(mMask, cv::Mat(), keypoints1, descriptors1);
	detector->detectAndCompute(dMask, cv::Mat(), keypoints2, descriptors2);

	// ʹ��BFMatcher����ƥ��
	cv::BFMatcher matcher(cv::NORM_HAMMING);
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	// Ѱ�����ƥ��
	double min_dist = DBL_MAX;
	cv::DMatch best_match;

	for (int i = 0; i < matches.size(); i++) {
		if (matches[i].distance < min_dist) {
			min_dist = matches[i].distance;
			best_match = matches[i];
		}
	}

	// �������ƥ����ƫ����
	cv::Point2f pt1 = keypoints1[best_match.queryIdx].pt;
	cv::Point2f pt2 = keypoints2[best_match.trainIdx].pt;
	cv::Point2f offset = pt2 - pt1;
	std::cout << "Best Match Offset: " << offset << std::endl;
	return offset;
	// �������ƥ��
	//cv::Mat img_matches;
	//std::vector<cv::DMatch> best_matches;
	//best_matches.push_back(best_match);
	//drawMatches(img1, keypoints1, img2, keypoints2, best_matches, img_matches);
	//cv::imshow("Best Match", img_matches);
	//cv::waitKey(0);
}

cv::Mat fov_scan::RotateImg(cv::Mat image, double angle)
{
	cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
	cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
	cv::Rect bbox = cv::RotatedRect(center, image.size(), angle).boundingRect();

	// ������ת�����Կ���ƽ��
	rotationMatrix.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rotationMatrix.at<double>(1, 2) += bbox.height / 2.0 - center.y;

	cv::Mat rotatedImage;
	// ֱ�Ӷ�ԭͼ�������ת
	cv::warpAffine(image, rotatedImage, rotationMatrix, bbox.size());

	// �ü�ͼ���Ա���ԭʼ�ߴ�
	int cropX = (rotatedImage.cols - image.cols) / 2;
	int cropY = (rotatedImage.rows - image.rows) / 2;
	cv::Rect cropRect(cropX, cropY, image.cols, image.rows);
	cv::Mat croppedImage = rotatedImage(cropRect);

	return croppedImage;
}

Transform fov_scan::calculateTransform(const std::pair<int, int>& offset1, const std::pair<int, int>& offset2)
{
	Transform transform;

	// ��ȡƫ����
	int dx1 = offset1.first, dy1 = offset1.second;
	int dx2 = offset2.first, dy2 = offset2.second;

	// �������Ͳ��
	double dot = dx1 * dx2 + dy1 * dy2;
	double cross = dx1 * dy2 - dy1 * dx2;

	// ��������������ģ
	double mag1 = sqrt(dx1 * dx1 + dy1 * dy1);
	double mag2 = sqrt(dx2 * dx2 + dy2 * dy2);

	// ��ֹ������
	if (mag1 == 0 || mag2 == 0) {
		transform.rotationAngle = 0; // �򷵻�һ�������־
		transform.translation = { 0, 0 };
		return transform;
	}

	// ����нǵ�����ֵ
	double cos_angle = dot / (mag1 * mag2);

	// ����Ƕȣ����ȣ�
	double angle_rad = acos(std::max(-1.0, std::min(1.0, cos_angle)));

	// ����ķ��ž�����ת����
	if (cross < 0) {
		angle_rad = -angle_rad;
	}

	// ת��Ϊ��
	transform.rotationAngle = angle_rad * (180.0 / M_PI);

	// ����ƽ����
	transform.translation = { (dx1 + dx2) / 2, (dy1 + dy2) / 2 };

	return transform;
}


//�����ϵ�
//std::vector<cv::Point2f> Scan::fovCoordinateSymmetry(std::vector<cv::Point2f> fov, cv::Point2f qishidian, cv::Point2f zhongzhidian)
//{
//	//float tempY = entireImage.rows * 0.5;//ˮƽ��
//	//float tempY = (qishidian.y - zhongzhidian.y) * 0.5;
//	for (int i = 0; i < fov.size(); i++)
//	{
//		float symY = (qishidian.y + zhongzhidian.y) - fov[i].y;
//		fov[i].y = symY;
//	}
//	return fov;
//}
//
//std::vector<cv::Rect2f> Scan::fovCoordinateSymmetryPix(std::vector<cv::Rect2f> fov, cv::Mat initImage)
//{
//	for (int i = 0; i < fov.size(); i++)
//	{
//		float symY = initImage.rows - fov[i].y;
//		fov[i].y = symY;
//	}
//	return fov;
//}