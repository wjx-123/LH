#pragma once
#include"onnx2.h"

YOLO::YOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;

	std::string classesFile = "class1.names";
	std::string model_path = config.modelpath;
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);//图形优化
	ort_session = new Session(env, widestr.c_str(), sessionOptions);//加载模型
	size_t numInputNodes = ort_session->GetInputCount();//输入节点
	size_t numOutputNodes = ort_session->GetOutputCount();//输出节点
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));

		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];
	this->num_proposal = output_node_dims[0][1];

	std::ifstream ifs(classesFile.c_str());
	std::string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	/*class_names.push_back("r");
	class_names.push_back("p");
	class_names.push_back("c");*/
	this->num_class = class_names.size();

	if (endsWith(config.modelpath, "best_rtp.onnx"))
	{
		anchors = (float*)anchors_1280;
		this->num_stride = 4;
	}
	else
	{
		anchors = (float*)anchors_640;
		this->num_stride = 3;
	}
}

YOLO::~YOLO()
{

}

cv::Mat YOLO::resize_image(cv::Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	cv::Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, cv::BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}

void YOLO::normalize_(cv::Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;

			}
		}
	}
}

int YOLO::endsWith(std::string s, std::string sub)
{
	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

std::vector<std::vector<cv::Rect2f>> YOLO::changeBoxInfoToRect(std::vector<BoxInfo> temp)
{
	std::vector<std::vector<cv::Rect2f>> smt_frame(2);
	std::vector<cv::Rect2f> rVector, cVector;
	for (int i = 0; i < temp.size(); i++)
	{
		if (temp[i].label == 0 || temp[i].label == 1)//r || p
		{
			cv::Rect2f rect(temp[i].x1, temp[i].y1, temp[i].x2 - temp[i].x1, temp[i].y2 - temp[i].y1);
			rVector.push_back(rect);
		}
		else if (temp[i].label == 2)//c
		{
			cv::Rect2f rect(temp[i].x1, temp[i].y1, temp[i].x2 - temp[i].x1, temp[i].y2 - temp[i].y1);
			cVector.push_back(rect);
		}
		for (auto r : rVector)
		{
			smt_frame[0].push_back(r);
		}
		for (auto c : cVector)
		{
			smt_frame[1].push_back(c);
		}
	}
	return smt_frame;
}

//std::vector<std::vector<cv::Rect2f>> YOLO::getCPCoordinate(cv::Mat img)
//{
	//std::vector<std::vector<cv::Rect2f>> res(2);
	//std::vector<cv::Rect2f> temp = splitImage(img.size()); //得到分割的小图的rect
	//for (auto rect : temp)
	//{
	//	std::vector<std::vector<cv::Rect2f>> smt_frame;
	//	cv::Mat smallImg = img(rect);
	//	smt_frame = detect(smallImg);
	//	//cv::rectangle(img, rect, cv::Scalar(2, 255, 255), 10);
	//	if (smt_frame.size() != 0)
	//	{
	//		for (int i = 0; i < smt_frame.size(); i++)
	//		{
	//			for (int j = 0; j < smt_frame[i].size(); j++)
	//			{
	//				smt_frame[i][j].x += rect.x;
	//				smt_frame[i][j].y += rect.y;
	//			}
	//		}
	//		for (int x = 0; x < smt_frame[0].size(); x++)
	//		{
	//			res[0].push_back(smt_frame[0][x]);
	//		}
	//		for (int y = 0; y < smt_frame[1].size(); y++)
	//		{
	//			res[1].push_back(smt_frame[1][y]);
	//		}
	//	}
	//}
	//return res;
//}

void YOLO::nms(std::vector<BoxInfo>& input_boxes)
{
	std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	std::vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	std::vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (std::max)(float(0), xx2 - xx1 + 1);
			float h = (std::max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());

}

std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> YOLO::detect(cv::Mat& frame)
{
	cv::Mat img = frame.clone();
	int img_height = img.rows;
	int img_width = img.cols;
	if (img_height < 6500 && img_width < 6500)
	{
		images.push_back(img);
	}
	else
	{
		images.push_back(img(cv::Range(0, 0.6 * img_height), cv::Range(0, 0.6 * img_width)));
		images.push_back(img(cv::Range(0, 0.6 * img_height), cv::Range(0.4 * img_width, img_width)));
		images.push_back(img(cv::Range(0.4 * img_height, img_height), cv::Range(0, 0.6 * img_width)));
		images.push_back(img(cv::Range(0.4 * img_height, img_height), cv::Range(0.4 * img_width, img_width)));
	}

	for (int m = 0; m < images.size(); m++)
	{
		int newh = 0, neww = 0, padh = 0, padw = 0;
		cv::Mat dstimg = this->resize_image(images[m], &newh, &neww, &padh, &padw);
		this->normalize_(dstimg);
		std::array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };//n c h w 1 3 1280 1280

		auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);//CPU
		Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

		// 开始推理
		std::vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
		/////generate proposals

		float ratioh = (float)images[m].rows / newh, ratiow = (float)images[m].cols / neww;
		int n = 0, q = 0, i = 0, j = 0, row_ind = 0, k = 0;
		const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
		for (n = 0; n < this->num_stride; n++)   ///特征图尺度
		{
			const float stride = pow(2, n + 3);
			int num_grid_x = (int)ceil((this->inpWidth / stride));
			int num_grid_y = (int)ceil((this->inpHeight / stride));
			for (q = 0; q < 3; q++)    ///anchor
			{
				const float anchor_w = this->anchors[n * 6 + q * 2];
				const float anchor_h = this->anchors[n * 6 + q * 2 + 1];
				for (i = 0; i < num_grid_y; i++)
				{
					for (j = 0; j < num_grid_x; j++)
					{
						float box_score = pdata[4];
						if (box_score > this->objThreshold)
						{
							int max_ind = 0;
							float max_class_socre = 0;
							for (k = 0; k < num_class; k++)
							{
								if (pdata[k + 5] > max_class_socre)
								{
									max_class_socre = pdata[k + 5];
									max_ind = k;
								}
							}
							max_class_socre *= box_score;
							if (max_class_socre > this->confThreshold)
							{
								float X = pdata[0];
								float Y = pdata[1];
								float W = pdata[2];
								float H = pdata[3];

								float xmin = (X - padw - 0.5 * W) * ratiow;//映射比例换算之后
								float ymin = (Y - padh - 0.5 * H) * ratioh;
								float xmax = (X - padw + 0.5 * W) * ratiow;
								float ymax = (Y - padh + 0.5 * H) * ratioh;

								if (m == 0)
								{
									xmin = xmin;
									ymin = ymin;
									xmax = xmax;
									ymax = ymax;

								}
								else if (m == 1)
								{
									xmin = xmin + 0.4 * img_width;
									ymin = ymin;
									xmax = xmax + 0.4 * img_width;
									ymax = ymax;

								}
								else if (m == 2)
								{
									xmin = xmin;
									ymin = ymin + 0.4 * img_height;
									xmax = xmax;
									ymax = ymax + 0.4 * img_height;
								}
								else if (m == 3)
								{
									xmin = xmin + 0.4 * img_width;
									ymin = ymin + 0.4 * img_height;
									xmax = xmax + 0.4 * img_width;
									ymax = ymax + 0.4 * img_height;
								}
								generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, max_ind });
							}
						}
						row_ind++;
						pdata += nout;
					}
				}
			}
		}
	}
	images.clear();//图像容器清理

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		rectangle(img, cv::Point(xmin, ymin), cv::Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), cv::Scalar(0, 255, 0), 3);
		std::string label = std::format("%.2f", generate_boxes[i].score);
		label = this->class_names[generate_boxes[i].label] + ":" + label;
		putText(img, label, cv::Point(xmin, ymin - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
	}
	cv::imwrite("C:\\Users\\LENOVO\\Pictures\\1\\1\\7_ini.jpg", img);
	auto smt_frame = getBlackPosition(generate_boxes, frame);
	generate_boxes.clear();
	return smt_frame;
}

std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> YOLO::getBlackPosition(std::vector<BoxInfo> generate_boxes, cv::Mat& frame)
{
	std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> result;
	for (int i = 0; i < generate_boxes.size(); i++)
	{
		try {
			cv::Rect2f smtRect = cv::Rect2f(
				generate_boxes[i].x1 < generate_boxes[i].x2 ? generate_boxes[i].x1 : generate_boxes[i].x2,
				generate_boxes[i].y1 < generate_boxes[i].y2 ? generate_boxes[i].y1 : generate_boxes[i].y2,
				generate_boxes[i].x1 < generate_boxes[i].x2 ? generate_boxes[i].x2 - generate_boxes[i].x1 : generate_boxes[i].x1 - generate_boxes[i].x2,
				generate_boxes[i].y1 < generate_boxes[i].y2 ? generate_boxes[i].y2 - generate_boxes[i].y1 : generate_boxes[i].y1 - generate_boxes[i].y2
			);

			// 检查smtRect是否超出图像范围，如果是则跳过当前循环
			if (smtRect.x < 0 || smtRect.y < 0 || smtRect.x + smtRect.width > frame.cols || smtRect.y + smtRect.height > frame.rows)
			{
				continue; // 跳过当前循环迭代
			}

			// 从大图中把这个小图扣出来
			cv::Mat smallSmtImg = frame(smtRect).clone();
			std::string typeOfLable;
			if (generate_boxes[i].label == 0)
			{
				typeOfLable = "R";
			}
			else if (generate_boxes[i].label == 1)
			{
				typeOfLable = "T";
			}
			else
			{
				typeOfLable = "P";
			}

			auto temp = eliminateYoloBackground.getBoundAndPin(smallSmtImg, typeOfLable);
			temp.first.x += smtRect.x;
			temp.first.y += smtRect.y;
			for (auto& rect : temp.second)
			{
				rect.x += smtRect.x;
				rect.y += smtRect.y;
			}
			result.push_back(temp);
		}
		catch (const std::exception& e) {
			// 处理异常，可以记录日志或者直接忽略
			/*std::cerr << "Exception caught: " << e.what() << std::endl;*/
			std::cout << "搜索引脚错误" << std::endl;
			continue; // 跳过当前循环迭代，继续下一个
		}
	}
	return result;
}



std::vector<std::tuple<cv::Mat, int, int>> YOLO::splitImage(const cv::Mat& img, int max_width, int max_height, float overlap)
{
	std::vector<std::tuple<cv::Mat, int, int>> result;
	int img_height = img.rows;
	int img_width = img.cols;

	if (img_height <= max_height && img_width <= max_width) {
		result.push_back(std::make_tuple(img, 0, 0));
		return result;
	}

	int num_blocks_w = std::ceil(img_width / (float)(max_width * (1 - overlap)));
	int num_blocks_h = std::ceil(img_height / (float)(max_height * (1 - overlap)));

	for (int h = 0; h < num_blocks_h; ++h) {
		for (int w = 0; w < num_blocks_w; ++w) {
			int x_start = w * max_width - std::floor(w * max_width * overlap);
			int y_start = h * max_height - std::floor(h * max_height * overlap);
			x_start = std::max(0, x_start);
			y_start = std::max(0, y_start);

			// 对于最后一个块，直接使用图像的宽度和高度作为结束坐标
			int x_end = (w == num_blocks_w - 1) ? img_width : std::min(img_width, x_start + max_width);
			int y_end = (h == num_blocks_h - 1) ? img_height : std::min(img_height, y_start + max_height);

			cv::Rect region(x_start, y_start, x_end - x_start, y_end - y_start);
			cv::Mat img_block = img(region);
			result.push_back(std::make_tuple(img_block, x_start, y_start));
		}
	}

	return result;
}

std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> YOLO::getCPCoordinate(cv::Mat& img)
{
	clock_t start1, end1;
	std::vector<std::pair<cv::Rect2f, std::vector<cv::Rect2f>>> res;
	auto splittedImages = splitImage(img);
	for (int i = 0; i < splittedImages.size(); i++)
	{
		start1 = clock();
		auto simImg_vectorOfRect = detect(std::get<0>(splittedImages[i]));
		end1 = clock();
		std::cout << "一张检测:" << (static_cast<double>(end1) - start1) / CLOCKS_PER_SEC << std::endl;
		int x_image = std::get<1>(splittedImages[i]);
		int y_image = std::get<2>(splittedImages[i]);
		for (int j = 0; j < simImg_vectorOfRect.size(); j++)
		{
			simImg_vectorOfRect[j].first.x += x_image;
			simImg_vectorOfRect[j].first.y += y_image;
			for (int x = 0; x < simImg_vectorOfRect[j].second.size(); x++)
			{
				simImg_vectorOfRect[j].second[x].x += x_image;
				simImg_vectorOfRect[j].second[x].y += y_image;
			}
		}
		res.insert(res.end(), std::make_move_iterator(simImg_vectorOfRect.begin()), std::make_move_iterator(simImg_vectorOfRect.end()));
	}
	return res;
}
